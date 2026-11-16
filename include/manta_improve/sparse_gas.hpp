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
  /**
   * Obstacles: one bit per voxel of the domain (X-fastest, 32 voxels per word), set where the
   * voxel is solid and at rest. Empty when there are none. Stored obstacle voxels are solid
   * cells; an obstacle voxel without storage is open air like any other unstored voxel, which
   * is safe because the active region always extends past the smoke.
   */
  std::vector<std::uint32_t> obstacle;

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
  /** Whether the obstacle bit of a voxel inside the domain is set. */
  bool is_obstacle(int x, int y, int z) const;
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
                        bool classify_cells = true,
                        std::vector<std::uint32_t> obstacle = {});

/** Pack a dense X-fastest array, nonzero where a voxel is solid, into Topology::obstacle. */
std::vector<std::uint32_t> pack_obstacle(const Extent &extent,
                                         const std::vector<std::uint8_t> &solid);

/**
 * Values on some bricks of the domain, kept apart from a State's storage: what sources emit and
 * how obstacles move. The bricks are those of the State's brick grid.
 */
struct BrickField {
  int brick_size = 8;
  Extent brick_extent;
  /** Per brick of the domain (X-fastest): slot, or -1 where every channel is zero. */
  std::vector<std::int32_t> table;
  /** Per channel: `brick_size`^3 values per slot, X-fastest inside each brick. */
  std::vector<std::vector<float>> channels;

  std::size_t slot_count() const;
  bool empty() const { return slot_count() == 0; }
  /** Value of a channel at domain coordinates; zero outside the stored bricks. */
  float at(std::size_t channel, int x, int y, int z) const;
};

/**
 * Keep the bricks of dense X-fastest arrays (each covering `extent`) where any of them is
 * nonzero. Empty arrays are not allowed; the result has one channel per array, in order.
 */
BrickField brick_field_from_dense(const Extent &extent,
                                  int brick_size,
                                  const std::vector<const std::vector<float> *> &channels);

enum class SourceMode {
  /** Raise density and temperature to the source's value where it is higher (Blender's default). */
  Raise,
  /** Add the source's value per second. */
  Add,
};

/**
 * What sources emit at the start of every step they are passed to. Channels: density,
 * temperature and, optionally, a cell-centered velocity u, v, w in voxels per second that is
 * set on the lower faces of every voxel emitting density or temperature, as Mantaflow sets an
 * emitter's initial velocity. Bricks holding a source stay active.
 */
struct Sources {
  SourceMode mode = SourceMode::Raise;
  BrickField field;
};

/** Sources from dense fields: density and temperature, and the velocity if `u` is not empty. */
Sources sources_from_dense(const struct DenseFields &dense, int brick_size, SourceMode mode);

/** Staggered (MAC) gas state on a sparse topology. Velocity is in voxels per second. */
struct State {
  Topology topology;
  std::vector<float> density;
  std::vector<float> temperature;
  /** Velocity through the lower X, Y, and Z face of each voxel. */
  std::vector<float> u, v, w;
  std::vector<float> pressure;
  /**
   * Velocity of the obstacles in voxels per second, cell-centered: three channels, read at
   * obstacle voxels only. Empty when the obstacles are at rest. A face between a solid and any
   * other cell takes the velocity of the obstacle side (the mean of both if both are solid);
   * the domain's outer wall is at rest.
   */
  BrickField obstacle_velocity;
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
  /** The last pressure, the first guess of the next solve; may be empty. */
  std::vector<float> pressure;
  /** Nonzero where a voxel is a solid obstacle; may be empty. */
  std::vector<std::uint8_t> obstacle;
  /** Cell-centered velocity of the obstacles in voxels per second; empty when at rest. */
  std::vector<float> obstacle_u, obstacle_v, obstacle_w;
};

State from_dense(const DenseFields &dense,
                 int brick_size,
                 const ActivityThresholds &thresholds,
                 int dilation);
DenseFields to_dense(const State &state);

/**
 * Recompute which bricks are active from the current values and move the state onto the new
 * topology. Values in bricks that become inactive are discarded. Bricks holding a source are
 * active.
 */
void retopologize(State &state,
                  const ActivityThresholds &thresholds,
                  int dilation,
                  const Sources *sources = nullptr);

/**
 * Replace the obstacles, for example with a collider's position in the next frame. The bricks
 * stay as they are and their cells are classified again. Smoke and heat inside voxels that
 * become solid are dropped at the start of the next step. `velocity` is as
 * State::obstacle_velocity.
 */
void set_obstacles(State &state, std::vector<std::uint32_t> mask, BrickField velocity = {});

/** Obstacle mask and velocity from the obstacle fields of `dense` (which may be empty). */
void set_obstacles_from_dense(State &state, const struct DenseFields &dense);

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

enum class TurbulenceMask {
  /** Full strength everywhere in the simulated bricks. */
  None,
  /** Scaled by density up to `mask_level`, full strength above it. */
  Density,
  /** Scaled by temperature up to `mask_level`, full strength above it. */
  Temperature,
};

/**
 * A swirling noise force: the curl of a smooth gradient-noise vector field, so it stirs the
 * smoke without pushing it apart or together. Like a turbulence or disturbance force in
 * EmberGen or Houdini, it changes how the smoke moves; it is not upres detail.
 */
struct Turbulence {
  /** Acceleration in voxels per second squared; 0 turns the force off. */
  float strength = 0.0f;
  /** Size of the swirls in voxels. */
  float scale = 16.0f;
  /** How many times per second the pattern changes completely; 0 freezes it. */
  float speed = 1.0f;
  std::uint32_t seed = 0;
  /**
   * Masked by density by default. Unmasked, the force stirs the air in every simulated brick,
   * which spreads the smoke, which activates more bricks: at 64x64x96 a plume then fills the
   * domain within 50 frames instead of rising.
   */
  TurbulenceMask mask = TurbulenceMask::Density;
  float mask_level = 0.1f;
};

struct Settings {
  /** Time step of step(). advance() chooses its own steps and ignores this. */
  float dt = 1.0f / 30.0f;
  /**
   * Simulation time in seconds at the start of step(), which the turbulence pattern follows.
   * advance() advances it through its substeps.
   */
  float time = 0.0f;
  Turbulence turbulence;
  Advection advection = Advection::MacCormack;
  /**
   * Trace departure points with the velocity half way along the path (second order) instead of
   * the velocity at the arrival point. Costs three more velocity samples per trace.
   */
  bool midpoint_trace = true;
  /**
   * Advection-reflection (Zehnder, Narain and Thomaszewski 2018): a step is two half steps. The
   * first advects, applies forces and projects; the velocity is then reflected about its
   * projection (twice the projected velocity minus the unprojected one) and advected through the
   * projected field for the second half, followed by forces and a second projection. Costs about
   * two plain steps and keeps the rotation that a single projection per step removes. Turn it
   * off, with a CFL number of 4, for a faster and smoother result.
   */
  bool reflection = true;
  /**
   * MacCormack: keep advection from changing the total density and temperature. The correction
   * is not conservative; next to obstacles its clamp adds smoke step after step (Mantaflow's own
   * kernels gain 60% in 60 frames on the plate scene of the tests). After each scalar channel is
   * corrected, the corrections of the sign that changed the total are scaled down uniformly
   * until the total matches the channel before advection, or until they are gone. Every value
   * stays between the first-order result and the clamped correction, so no new extremes appear.
   */
  bool conserve_mass = true;
  /** advance(): largest distance, in voxels, anything may travel in one step. */
  float cfl = 6.0f;
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
            std::vector<float> &result,
            bool midpoint = false);

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
                       std::vector<float> &result,
                       bool midpoint = false);

/**
 * Scale down the corrections that `advect_maccormack` made (`result` minus `forward`) of one
 * sign, uniformly, so that `result` totals as much as `source`. Corrections of the other sign
 * are kept. If the corrections of that sign are too small, they are removed entirely and the
 * remaining difference is left. Solid voxels hold zero in all three channels.
 */
void conserve_correction(const std::vector<float> &source,
                         const std::vector<float> &forward,
                         std::vector<float> &result);

/** Largest absolute velocity component of any stored face. */
float max_speed(const State &state);

/** Time step for the next substep of a frame: the rest of the frame, limited by the CFL number. */
float cfl_time_step(float remaining, float cfl, float speed);

/**
 * The turbulence force on faces between two fluid cells, over `dt` seconds, with the pattern
 * at `time`. Does nothing if its strength is zero.
 */
void add_turbulence(State &state, const Turbulence &turbulence, float time, float dt);

/** The two time slices of the turbulence pattern at `time` and the blend between them. */
void turbulence_time(const Turbulence &turbulence, float time, int &slice, float &blend);

/** Mantaflow-style buoyancy on faces between two fluid cells. */
void add_buoyancy(State &state, const std::vector<float> &factor, float coefficient, float dt,
                  const float gravity[3]);

/**
 * Set every face that touches a solid cell to the obstacle's velocity (zero for obstacles at
 * rest and for the domain's outer wall). See State::obstacle_velocity.
 */
void enforce_walls(State &state);

/**
 * Start of a step: zero density and temperature in solid cells, then apply `sources` (may be
 * null) over `dt` seconds.
 */
void inject(State &state, const Sources *sources, float dt);

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

/**
 * Subtract the pressure gradient from every face that borders a fluid cell. Faces touching a
 * solid are left as enforce_walls set them.
 */
void subtract_gradient(State &state);

/** Solve for pressure with the configured solver and make the velocity divergence-free. */
StepInfo project(State &state, const Settings &settings);

/**
 * Advance one time step: retopologize, inject sources, advect, add buoyancy, enforce walls,
 * project. `sources` may be null.
 */
StepInfo step(State &state, const Settings &settings, const Sources *sources = nullptr);

/**
 * Advance by `frame_time` in CFL-limited substeps. Throws std::runtime_error, leaving the state
 * partly advanced, if that takes more than `max_substeps`.
 */
FrameInfo advance(State &state,
                  const Settings &settings,
                  float frame_time,
                  const Sources *sources = nullptr);

}  // namespace manta_improve::sparse
