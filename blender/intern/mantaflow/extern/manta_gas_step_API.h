/* SPDX-License-Identifier: GPL-2.0-or-later */
#pragma once

#include <cstddef>
#include <string>

namespace blender {

struct MantaGasExtent3 {
  int x = 0;
  int y = 0;
  int z = 0;
};

struct MantaGasVec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

enum class MantaGasVelocityLayout {
  /** One vector per cell center. Converting to and from the solver's layout averages neighbors. */
  CellCentered,
  /**
   * Mantaflow's MAC layout: component `c` of cell `(i, j, k)` lives on the cell's lower face along
   * axis `c`. This is the solver's own state, so it survives a step boundary without loss.
   */
  Staggered,
};

/**
 * Borrowed dense grid buffers in Mantaflow's X-fastest layout.
 *
 * Velocity is measured in grid cells per second. `velocity_layout` describes the input buffer. A
 * cell-centered input is resampled to faces once, which is intended for authored initial or
 * emitted velocities. On success the buffer always holds the staggered solver field, reported in
 * #MantaGasStepDiagnostics::velocity_layout; feeding that back as #Staggered continues the
 * simulation exactly where it stopped. Pressure is both an input and an output so the
 * serializable state can retain the previous pressure field as an initial guess.
 */
struct MantaGasState {
  MantaGasExtent3 resolution;
  float *density = nullptr;
  float *temperature = nullptr;
  MantaGasVec3 *velocity = nullptr;
  float *pressure = nullptr;
  MantaGasVelocityLayout velocity_layout = MantaGasVelocityLayout::CellCentered;
};

enum class MantaGasPreconditioner {
  /** Modified incomplete Cholesky. Iteration count grows with resolution. */
  MIC,
  /**
   * Mantaflow's geometric multigrid, constructed and destroyed inside the call. This does not use
   * Mantaflow's `solvePressure` entry point, whose multigrid modes share a process-global cache.
   */
  Multigrid,
};

struct MantaGasStepSettings {
  /* Frame time and internal substeps are measured in seconds. */
  float delta_time = 1.0f / 30.0f;
  float cfl = 2.0f;
  int max_substeps = 32;

  /* Grid-space acceleration in cells per second squared, used only for buoyancy. */
  MantaGasVec3 gravity = {0.0f, 0.0f, -9.81f};
  float density_buoyancy = 0.0f;
  float heat_buoyancy = 1.0f;

  float pressure_accuracy = 1.0e-3f;
  /** Scales the resolution-dependent iteration limit of the MIC preconditioner only. */
  float pressure_iteration_factor = 1.5f;
  MantaGasPreconditioner preconditioner = MantaGasPreconditioner::Multigrid;
};

struct MantaGasStepDiagnostics {
  int substeps = 0;
  float simulated_time = 0.0f;
  /* Largest MAC velocity norm sampled by the CFL scheduler, in cells per second. */
  float max_velocity = 0.0f;
  /** Largest conjugate-gradient iteration count of any substep's pressure solve. */
  int pressure_iterations = 0;
  /** Largest final residual (maximum norm) of any substep's pressure solve. */
  float pressure_residual = 0.0f;
  /** False if any substep stopped at its iteration limit before reaching the accuracy. */
  bool pressure_converged = true;
  /** Layout of the caller's velocity buffer on return. Unchanged on failure or zero time. */
  MantaGasVelocityLayout velocity_layout = MantaGasVelocityLayout::CellCentered;
};

/**
 * Advance one closed-domain smoke state using compiled Mantaflow kernels.
 *
 * The complete frame, including all internal substeps, remains in Mantaflow storage. No modifier,
 * Python script, disk cache, global solver singleton, or retained solver pointer is used. On
 * failure, the caller's buffers are not modified and error_message describes the failure.
 */
bool manta_gas_simulation_step(const MantaGasState &state,
                               const MantaGasStepSettings &settings,
                               MantaGasStepDiagnostics &diagnostics,
                               std::string &error_message);

}  // namespace blender
