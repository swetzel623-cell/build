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
};

struct Settings {
  float dt = 1.0f / 30.0f;
  /** Acceleration in voxels per second squared. */
  float gravity[3] = {0.0f, 0.0f, -9.81f};
  float density_buoyancy = 0.0f;
  float heat_buoyancy = 1.0f;

  PressureSolver pressure_solver = PressureSolver::ConjugateGradient;
  float pressure_accuracy = 1.0e-4f;
  int pressure_iterations = 400;

  ActivityThresholds thresholds;
  /** Bricks of padding around active content. Smoke must not cross it within one step. */
  int dilation = 1;
};

struct StepInfo {
  int pressure_iterations = 0;
  /** Largest remaining divergence of any fluid cell, in 1/s. */
  float max_divergence = 0.0f;
  std::size_t active_bricks = 0;
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

/** Subtract the pressure gradient from every face that borders a fluid cell. */
void subtract_gradient(State &state);

/** Solve for pressure with the configured solver and make the velocity divergence-free. */
StepInfo project(State &state, const Settings &settings);

/** Advance one time step: retopologize, advect, add buoyancy, enforce walls, project. */
StepInfo step(State &state, const Settings &settings);

}  // namespace manta_improve::sparse
