// SPDX-License-Identifier: MIT
#pragma once

// Sparse brick-grid gas solver: data layout and CPU reference kernels.
//
// Everything here is laid out the way a GPU compute backend consumes it: flat integer tables for
// topology and one flat float array per channel. The CPU kernels in this file are the reference
// that GPU kernels are checked against; each one is a pure gather over stored voxels and maps to
// one compute shader.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace manta_improve::sparse {

struct Extent {
  int x = 0;
  int y = 0;
  int z = 0;
};

enum CellType : std::uint32_t {
  /** Not simulated: free air with pressure zero. Smoke may flow into it. */
  CellOpen = 0,
  /** Simulated cell with a pressure unknown. */
  CellFluid = 1,
  /** Closed domain wall. The outermost voxel layer of the domain, as in Mantaflow. */
  CellSolid = 2,
};

/**
 * Which bricks exist and how they connect.
 *
 * The domain is divided into cubic bricks of `brick_size` voxels per side. Only active bricks
 * have storage. A stored voxel is addressed as `slot * voxels_per_brick + local`, with `local`
 * X-fastest inside the brick.
 */
struct Topology {
  Extent extent;
  int brick_size = 8;
  Extent brick_extent;

  /** Per brick of the domain (X-fastest): storage slot, or -1 if the brick is inactive. */
  std::vector<std::int32_t> brick_table;
  /** Per slot: brick coordinates x, y, z and one unused entry. */
  std::vector<std::int32_t> brick_coords;
  /** Per slot: slots of the -X, +X, -Y, +Y, -Z, +Z neighbor bricks, or -1. */
  std::vector<std::int32_t> neighbors;
  /** Per stored voxel: a #CellType. */
  std::vector<std::uint32_t> cell_type;

  std::size_t brick_count() const { return brick_coords.size() / 4; }
  std::size_t voxels_per_brick() const
  {
    return static_cast<std::size_t>(brick_size) * brick_size * brick_size;
  }
  std::size_t voxel_count() const { return brick_count() * voxels_per_brick(); }

  /** Index of the stored voxel at domain coordinates, or -1 if it has no storage. */
  std::int64_t index_of(int x, int y, int z) const;
  /** Cell type at any coordinates, stored or not. */
  CellType type_at(int x, int y, int z) const;
};

/**
 * Build a topology from per-brick activity flags (X-fastest over the brick grid), after growing
 * the active set by `dilation` bricks in every direction. With `classify_cells` false,
 * `cell_type` is left empty for a caller that classifies cells itself, such as the GPU driver.
 */
Topology build_topology(Extent extent,
                        int brick_size,
                        const std::vector<std::uint8_t> &brick_active,
                        int dilation,
                        bool classify_cells = true);

/** Staggered (MAC) gas state on a sparse topology. Velocity is in voxels per second. */
struct State {
  Topology topology;
  std::vector<float> density;
  std::vector<float> temperature;
  /** Velocity through the lower X, Y, and Z face of each voxel. */
  std::vector<float> u, v, w;
  std::vector<float> pressure;
};

struct ActivityThresholds {
  float scalar = 1.0e-5f;
  /**
   * Velocity alone does not keep a brick active by default. The pressure solve induces a small
   * velocity everywhere, so any finite default would activate the whole domain. Air motion in
   * bricks without smoke or heat is discarded, as in other sparse smoke solvers.
   */
  float velocity = std::numeric_limits<float>::infinity();
};

/** Dense X-fastest arrays of `extent` voxels; velocity arrays may be empty for a fluid at rest. */
struct DenseFields {
  Extent extent;
  std::vector<float> density;
  std::vector<float> temperature;
  std::vector<float> u, v, w;
};

State from_dense(const DenseFields &dense,
                 int brick_size,
                 const ActivityThresholds &thresholds,
                 int dilation);
DenseFields to_dense(const State &state);

/**
 * Recompute which bricks are active from the current values and move the state onto the new
 * topology. Values in bricks that become inactive are discarded.
 */
void retopologize(State &state, const ActivityThresholds &thresholds, int dilation);

enum class PressureSolver {
  /** Jacobi-preconditioned conjugate gradients, run to `pressure_accuracy`. */
  ConjugateGradient,
  /** A fixed number of Jacobi sweeps. No reductions, so it runs unchanged on a GPU. */
  Jacobi,
  /**
   * A fixed number of multigrid V-cycles (`pressure_iterations`) with red-black Gauss-Seidel
   * smoothing. No reductions either. Requires a power-of-two brick size.
   */
  Multigrid,
};

enum class Advection {
  /** First-order semi-Lagrangian. Stable and very diffusive. */
  SemiLagrangian,
  /**
   * MacCormack: a forward and a backward semi-Lagrangian trace, with the error estimate added
   * back and the result clamped to the values the forward trace interpolated. The scheme the
   * Mantaflow node uses, though not operation for operation.
   */
  MacCormack,
};

struct Settings {
  /** Time step of step(). advance() chooses its own steps and ignores this. */
  float dt = 1.0f / 30.0f;
  Advection advection = Advection::MacCormack;
  /** advance(): largest distance, in voxels, anything may travel in one step. */
  float cfl = 2.0f;
  /** advance(): fail instead of taking more steps than this for one frame. */
  int max_substeps = 64;
  /** Acceleration in voxels per second squared. */
  float gravity[3] = {0.0f, 0.0f, -9.81f};
  float density_buoyancy = 0.0f;
  float heat_buoyancy = 1.0f;

  PressureSolver pressure_solver = PressureSolver::ConjugateGradient;
  float pressure_accuracy = 1.0e-4f;
  int pressure_iterations = 400;

  /** Multigrid: red-black sweeps before and after each coarse correction, and on the last level. */
  int multigrid_pre_sweeps = 2;
  int multigrid_post_sweeps = 2;
  /**
   * Multigrid: sweeps before and after the correction on the coarse levels. Coarse sweeps cost
   * little work but two dispatches each on a GPU, where the dispatch count bounds the solve.
   */
  int multigrid_coarse_pre_sweeps = 2;
  int multigrid_coarse_post_sweeps = 2;
  int multigrid_coarsest_sweeps = 8;
  /** Multigrid: recursive cycles per coarse solve. 1 is a V-cycle, 2 a W-cycle. */
  int multigrid_coarse_cycles = 2;
  /**
   * Multigrid: how many of the finest coarse levels are solved with `multigrid_coarse_cycles`.
   * Every further level doubles the visits to all levels below it. 1 costs about 40% fewer GPU
   * dispatches than 2 and converges almost as fast per cycle (0.34 against 0.31).
   */
  int multigrid_w_depth = 1;
  /** Multigrid: factor applied to coarse corrections; aggregation underestimates them. */
  float multigrid_overcorrection = 1.4f;

  ActivityThresholds thresholds;
  /** Bricks of padding around active content. Smoke must not cross it within one step. */
  int dilation = 1;
};

struct StepInfo {
  int pressure_iterations = 0;
  /** Largest remaining divergence of any fluid cell, in 1/s. */
  float max_divergence = 0.0f;
  std::size_t active_bricks = 0;
  /** The time step taken. */
  float dt = 0.0f;
};

struct FrameInfo {
  int substeps = 0;
  /** Largest velocity component seen at the start of any substep, in voxels per second. */
  float max_speed = 0.0f;
  StepInfo last_step;
};

/* Reference kernels. Each reads whole channels and writes one output channel. */

/** Trilinear sample of a channel whose values sit at `index + offset`; missing voxels read 0. */
float sample(const Topology &topology,
             const std::vector<float> &channel,
             float x,
             float y,
             float z,
             float offset_x,
             float offset_y,
             float offset_z);

/** First-order semi-Lagrangian advection of a channel staggered at `offset`. */
void advect(const State &state,
            const std::vector<float> &source,
            float offset_x,
            float offset_y,
            float offset_z,
            float dt,
            std::vector<float> &result);

/**
 * MacCormack correction of a forward-advected channel. `forward` must be advect() of `source`
 * with the same arguments.
 */
void advect_maccormack(const State &state,
                       const std::vector<float> &source,
                       const std::vector<float> &forward,
                       float offset_x,
                       float offset_y,
                       float offset_z,
                       float dt,
                       std::vector<float> &result);

/** Largest absolute velocity component of any stored face. */
float max_speed(const State &state);

/** Time step for the next substep of a frame: the rest of the frame, limited by the CFL number. */
float cfl_time_step(float remaining, float cfl, float speed);

/** Mantaflow-style buoyancy on faces between two fluid cells. */
void add_buoyancy(State &state, const std::vector<float> &factor, float coefficient, float dt,
                  const float gravity[3]);

/** Zero every face that touches a solid cell. */
void enforce_walls(State &state);

/** Divergence of fluid cells in 1/s; zero elsewhere. */
void divergence(const State &state, std::vector<float> &result);

/** One Jacobi sweep of the pressure Poisson system `A p = -divergence`. */
void jacobi_sweep(const Topology &topology,
                  const std::vector<float> &negative_divergence,
                  const std::vector<float> &pressure,
                  std::vector<float> &result);

/**
 * One coarse level of the pressure multigrid. Coarse cells aggregate 2x2x2 cells of the level
 * below; the operator is the Galerkin product for piecewise-constant transfer, which stays a
 * 7-point stencil: a diagonal and one weight per lower face.
 *
 * Levels with fewer than one cell per brick side left are stored densely over the (coarsened)
 * brick grid; the others are stored per brick slot like the voxels.
 */
struct MultigridLevel {
  bool dense = false;
  /** Brick-stored levels: cells per brick side. */
  int side = 0;
  /** Dense levels: grid size. */
  Extent dims;
  std::vector<float> diagonal, weight_x, weight_y, weight_z;

  std::size_t cell_count() const { return diagonal.size(); }
};

struct Multigrid {
  /** levels[0] is one coarsening above the voxels. */
  std::vector<MultigridLevel> levels;
};

/** Index of a coarse cell at level coordinates, or -1 if it has no storage. */
std::int64_t multigrid_index(const Topology &topology, const MultigridLevel &level, int x, int y,
                             int z);

/** The levels of the hierarchy for a topology, with empty operator arrays. */
std::vector<MultigridLevel> multigrid_layout(const Topology &topology);
/** Number of cells a level stores on a topology. */
std::size_t multigrid_cell_count(const Topology &topology, const MultigridLevel &level);

Multigrid build_multigrid(const Topology &topology);

/** Run `cycles` V-cycles on `A pressure = rhs`, starting from the given pressure. */
void multigrid_solve(const Topology &topology,
                     const Multigrid &multigrid,
                     const std::vector<float> &rhs,
                     const Settings &settings,
                     int cycles,
                     std::vector<float> &pressure);

/** Largest absolute residual of `A pressure = rhs` over fluid cells. */
float poisson_residual(const Topology &topology,
                       const std::vector<float> &rhs,
                       const std::vector<float> &pressure);

/** Subtract the pressure gradient from every face that borders a fluid cell. */
void subtract_gradient(State &state);

/** Solve for pressure with the configured solver and make the velocity divergence-free. */
StepInfo project(State &state, const Settings &settings);

/** Advance one time step: retopologize, advect, add buoyancy, enforce walls, project. */
StepInfo step(State &state, const Settings &settings);

/**
 * Advance by `frame_time` in CFL-limited substeps. Throws std::runtime_error, leaving the state
 * partly advanced, if that takes more than `max_substeps`.
 */
FrameInfo advance(State &state, const Settings &settings, float frame_time);

}  // namespace manta_improve::sparse
