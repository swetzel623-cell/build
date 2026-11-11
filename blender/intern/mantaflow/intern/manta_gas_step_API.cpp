/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "manta_gas_step_API.h"

#include "manta_gas_step_driver.h"

#include "manta.h"
#include "fluidsolver.h"
#include "conjugategrad.h"
#include "grid.h"
#include "multigrid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace Manta {

/* Direct compiled entry points from Mantaflow's preprocessed plugins. */
void advectSemiLagrange(const FlagGrid *flags,
                        const MACGrid *vel,
                        GridBase *grid,
                        int order,
                        Real strength,
                        int orderSpace,
                        bool openBounds,
                        int boundaryWidth,
                        int clampMode,
                        int orderTrace);

void addBuoyancy(const FlagGrid &flags,
                 const Grid<Real> &density,
                 MACGrid &vel,
                 Vec3 gravity,
                 Real coefficient,
                 bool scale);

void setWallBcs(const FlagGrid &flags,
                MACGrid &vel,
                const MACGrid *obvel,
                const MACGrid *fractions,
                const Grid<Real> *phiObs,
                int boundaryWidth);

/* The pieces of Mantaflow's `solvePressure`, minus `solvePressureSystem`: that function keeps
 * its multigrid in a process-global map for both multigrid modes. */
void computePressureRhs(Grid<Real> &rhs,
                        const MACGrid &vel,
                        const Grid<Real> &pressure,
                        const FlagGrid &flags,
                        Real cgAccuracy,
                        const Grid<Real> *phi,
                        const Grid<Real> *perCellCorr,
                        const MACGrid *fractions,
                        const MACGrid *obvel,
                        Real gfClamp,
                        Real cgMaxIterFac,
                        bool precondition,
                        int preconditioner,
                        bool enforceCompatibility,
                        bool useL2Norm,
                        bool zeroPressureFixing,
                        const Grid<Real> *curv,
                        Real surfTens);

void fixPressure(int fixPidx,
                 Real value,
                 Grid<Real> &rhs,
                 Grid<Real> &A0,
                 Grid<Real> &Ai,
                 Grid<Real> &Aj,
                 Grid<Real> &Ak);

void correctVelocity(MACGrid &vel,
                     Grid<Real> &pressure,
                     const FlagGrid &flags,
                     Real cgAccuracy,
                     const Grid<Real> *phi,
                     const Grid<Real> *perCellCorr,
                     const MACGrid *fractions,
                     Real gfClamp,
                     Real cgMaxIterFac,
                     bool precondition,
                     int preconditioner,
                     bool enforceCompatibility,
                     bool useL2Norm,
                     bool zeroPressureFixing,
                     const Grid<Real> *curv,
                     Real surfTens);

}  // namespace Manta

namespace blender {
namespace {

constexpr int pressure_preconditioner_mic = 1;  // Manta::PcMIC from plugin/pressure.cpp.
constexpr int multigrid_max_iterations = 100;   // As in Mantaflow's solvePressureSystem.

struct PressureSolveResult {
  int iterations = 0;
  float residual = 0.0f;
  bool converged = false;
};

/** First fluid cell in Mantaflow's preferred order for pinning the pressure of a closed domain. */
Manta::IndexInt pressure_pin_index(const Manta::FlagGrid &flags)
{
  const Manta::Vec3i top_center(flags.getSizeX() / 2, flags.getSizeY() - 1, flags.getSizeZ() / 2);
  for (int offset = 0; offset < 3; ++offset) {
    const Manta::Vec3i position = top_center - Manta::Vec3i(0, offset, 0);
    if (flags.isFluid(position)) {
      return flags.index(position);
    }
  }
  for (int k = 1; k < flags.getSizeZ() - 1; ++k) {
    for (int j = 1; j < flags.getSizeY() - 1; ++j) {
      for (int i = 1; i < flags.getSizeX() - 1; ++i) {
        if (flags.isFluid(i, j, k)) {
          return flags.index(i, j, k);
        }
      }
    }
  }
  return -1;
}

/**
 * Closed-domain pressure projection with every solver object owned by this call.
 *
 * This follows Mantaflow's `solvePressure` step for step (right-hand side, Laplace matrix,
 * pressure pinning, preconditioned CG, velocity correction) so the result matches the legacy
 * solver. The only difference is ownership of the preconditioner.
 */
PressureSolveResult solve_pressure(Manta::FluidSolver &solver,
                                   Manta::MACGrid &velocity,
                                   Manta::Grid<Real> &pressure,
                                   Manta::FlagGrid &flags,
                                   const MantaGasStepSettings &settings)
{
  const Real accuracy = static_cast<Real>(settings.pressure_accuracy);
  const Real iteration_factor = static_cast<Real>(settings.pressure_iteration_factor);
  const Real ghost_fluid_clamp = static_cast<Real>(1.0e-4);

  Manta::Grid<Real> rhs(&solver);
  Manta::computePressureRhs(rhs,
                            velocity,
                            pressure,
                            flags,
                            accuracy,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            ghost_fluid_clamp,
                            iteration_factor,
                            true,
                            pressure_preconditioner_mic,
                            false,
                            false,
                            true,
                            nullptr,
                            static_cast<Real>(0.0));

  Manta::Grid<Real> residual(&solver);
  Manta::Grid<Real> search(&solver);
  Manta::Grid<Real> a0(&solver);
  Manta::Grid<Real> ai(&solver);
  Manta::Grid<Real> aj(&solver);
  Manta::Grid<Real> ak(&solver);
  Manta::Grid<Real> tmp(&solver);
  Manta::MakeLaplaceMatrix(flags, a0, ai, aj, ak, nullptr);

  /* A closed domain has no empty cell, so the system is singular until one pressure is fixed. */
  const Manta::IndexInt pin_index = pressure_pin_index(flags);
  if (pin_index >= 0) {
    Manta::fixPressure(static_cast<int>(pin_index), static_cast<Real>(0.0), rhs, a0, ai, aj, ak);
  }

  Manta::GridCg<Manta::ApplyMatrix> cg(
      pressure, rhs, residual, search, flags, tmp, {&a0, &ai, &aj, &ak});
  cg.setAccuracy(accuracy);
  cg.setUseL2Norm(false);

  /* Declared before use so they outlive every CG iteration; only one kind is allocated. */
  std::unique_ptr<Manta::GridMg> multigrid;
  std::vector<std::unique_ptr<Manta::Grid<Real>>> mic_grids;
  int max_iterations = 0;
  if (settings.preconditioner == MantaGasPreconditioner::Multigrid) {
    multigrid = std::make_unique<Manta::GridMg>(pressure.getSize());
    cg.setMGPreconditioner(Manta::GridCgInterface::PC_MGP, multigrid.get());
    max_iterations = multigrid_max_iterations;
  }
  else {
    for (int i = 0; i < 4; ++i) {
      mic_grids.push_back(std::make_unique<Manta::Grid<Real>>(&solver));
    }
    cg.setICPreconditioner(Manta::GridCgInterface::PC_mICP,
                           mic_grids[0].get(),
                           mic_grids[1].get(),
                           mic_grids[2].get(),
                           mic_grids[3].get());
    max_iterations = static_cast<int>(iteration_factor * static_cast<Real>(flags.getSize().max()));
  }

  PressureSolveResult result;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (!cg.iterate()) {
      result.converged = true;
      break;
    }
  }
  result.iterations = static_cast<int>(cg.getIterations());
  result.residual = static_cast<float>(cg.getResNorm());

  Manta::correctVelocity(velocity,
                         pressure,
                         flags,
                         accuracy,
                         nullptr,
                         nullptr,
                         nullptr,
                         ghost_fluid_clamp,
                         iteration_factor,
                         true,
                         pressure_preconditioner_mic,
                         false,
                         false,
                         true,
                         nullptr,
                         static_cast<Real>(0.0));
  return result;
}

std::size_t checked_cell_count(const MantaGasExtent3 &resolution)
{
  if (resolution.x < 4 || resolution.y < 4 || resolution.z < 4) {
    throw std::runtime_error("Gas Simulation Step requires a 3D resolution of at least 4^3.");
  }

  const std::size_t x = static_cast<std::size_t>(resolution.x);
  const std::size_t y = static_cast<std::size_t>(resolution.y);
  const std::size_t z = static_cast<std::size_t>(resolution.z);
  if (x > std::numeric_limits<std::size_t>::max() / y ||
      x * y > std::numeric_limits<std::size_t>::max() / z)
  {
    throw std::runtime_error("Gas Simulation Step resolution overflows the grid size.");
  }
  return x * y * z;
}

void validate_buffers(const MantaGasState &state, const std::size_t cell_count)
{
  if (!state.density || !state.temperature || !state.velocity || !state.pressure) {
    throw std::runtime_error("Gas Simulation Step received a null grid buffer.");
  }

  for (std::size_t i = 0; i < cell_count; ++i) {
    if (!std::isfinite(state.density[i]) || !std::isfinite(state.temperature[i]) ||
        !std::isfinite(state.pressure[i]) || !std::isfinite(state.velocity[i].x) ||
        !std::isfinite(state.velocity[i].y) || !std::isfinite(state.velocity[i].z))
    {
      throw std::runtime_error("Gas Simulation Step input contains a non-finite value.");
    }
  }
}

void validate_settings(const MantaGasStepSettings &settings)
{
  if (!std::isfinite(settings.delta_time) || settings.delta_time < 0.0f ||
      !std::isfinite(settings.cfl) || settings.cfl <= 0.0f || settings.max_substeps < 1)
  {
    throw std::runtime_error("Gas Simulation Step has invalid time-step settings.");
  }
  if (!std::isfinite(settings.gravity.x) || !std::isfinite(settings.gravity.y) ||
      !std::isfinite(settings.gravity.z) || !std::isfinite(settings.density_buoyancy) ||
      !std::isfinite(settings.heat_buoyancy) || !std::isfinite(settings.pressure_accuracy) ||
      settings.pressure_accuracy <= 0.0f || !std::isfinite(settings.pressure_iteration_factor) ||
      settings.pressure_iteration_factor <= 0.0f)
  {
    throw std::runtime_error("Gas Simulation Step has invalid force or pressure settings.");
  }
}

float max_mac_velocity(const Manta::MACGrid &velocity)
{
  double max_squared = 0.0;
  const Manta::Vec3i extent = velocity.getSize();
  const Manta::IndexInt size = static_cast<Manta::IndexInt>(extent.x) *
                               static_cast<Manta::IndexInt>(extent.y) *
                               static_cast<Manta::IndexInt>(extent.z);
  for (Manta::IndexInt i = 0; i < size; ++i) {
    const Manta::Vec3 &value = velocity[i];
    const double x = static_cast<double>(value.x);
    const double y = static_cast<double>(value.y);
    const double z = static_cast<double>(value.z);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      return std::numeric_limits<float>::infinity();
    }
    max_squared = std::max(max_squared, x * x + y * y + z * z);
  }
  return static_cast<float>(std::sqrt(max_squared));
}

bool grids_are_finite(const Manta::Grid<Real> &density,
                      const Manta::Grid<Real> &temperature,
                      const Manta::Grid<Real> &pressure,
                      const Manta::MACGrid &velocity)
{
  const Manta::Vec3i extent = density.getSize();
  const Manta::IndexInt size = static_cast<Manta::IndexInt>(extent.x) *
                               static_cast<Manta::IndexInt>(extent.y) *
                               static_cast<Manta::IndexInt>(extent.z);
  for (Manta::IndexInt i = 0; i < size; ++i) {
    if (!std::isfinite(static_cast<double>(density[i])) ||
        !std::isfinite(static_cast<double>(temperature[i])) ||
        !std::isfinite(static_cast<double>(pressure[i])))
    {
      return false;
    }
  }
  return std::isfinite(max_mac_velocity(velocity));
}

void advect(Manta::FlagGrid &flags,
            Manta::MACGrid &velocity,
            Manta::GridBase &grid)
{
  Manta::advectSemiLagrange(
      &flags, &velocity, &grid, 2, 1.0, 1, false, -1, 2, 1);
}

}  // namespace

bool manta_gas_simulation_step(const MantaGasState &state,
                               const MantaGasStepSettings &settings,
                               MantaGasStepDiagnostics &diagnostics,
                               std::string &error_message)
{
  diagnostics = {};
  diagnostics.velocity_layout = state.velocity_layout;
  error_message.clear();

  try {
    const std::size_t cell_count = checked_cell_count(state.resolution);
    validate_buffers(state, cell_count);
    validate_settings(settings);

    manta_gas::GasStepDriver driver(settings.delta_time, settings.cfl, settings.max_substeps);
    if (!driver.valid()) {
      throw std::runtime_error("Gas Simulation Step could not initialize its frame scheduler.");
    }
    if (driver.finished()) {
      return true;
    }

    /* All objects are local to this call. In particular, no state is shared between zones. */
    Manta::FluidSolver solver(
        Manta::Vec3i(state.resolution.x, state.resolution.y, state.resolution.z));
    Manta::FlagGrid flags(&solver);
    Manta::Grid<Real> density(&solver);
    Manta::Grid<Real> temperature(&solver);
    Manta::Grid<Real> pressure(&solver);
    Manta::MACGrid velocity(&solver);

    for (std::size_t i = 0; i < cell_count; ++i) {
      density[i] = static_cast<Real>(state.density[i]);
      temperature[i] = static_cast<Real>(state.temperature[i]);
      pressure[i] = static_cast<Real>(state.pressure[i]);
      velocity[i] = Manta::Vec3(static_cast<Real>(state.velocity[i].x),
                                static_cast<Real>(state.velocity[i].y),
                                static_cast<Real>(state.velocity[i].z));
    }
    if (state.velocity_layout == MantaGasVelocityLayout::CellCentered) {
      /* Authored input only. The solver state itself is exchanged in the staggered layout, because
       * averaging to cell centers and back every frame low-pass filters the velocity field. */
      Manta::Grid<Manta::Vec3> centered_velocity(&solver);
      centered_velocity.copyFrom(velocity, false);
      velocity.clear();
      Manta::resampleVec3ToMac(centered_velocity, velocity);
    }

    /* Match a minimal closed gas domain: boundary cells are obstacles, interior cells are fluid. */
    flags.initDomain(0);
    flags.fillGrid();

    MantaGasStepDiagnostics local_diagnostics;
    while (!driver.finished()) {
      const float max_velocity = max_mac_velocity(velocity);
      const std::optional<float> next_dt = driver.next_dt(max_velocity);
      if (!next_dt.has_value()) {
        if (driver.substeps() >= settings.max_substeps) {
          throw std::runtime_error(
              "Gas Simulation Step exhausted its substep limit; the input state was not changed.");
        }
        throw std::runtime_error("Gas Simulation Step could not schedule a finite CFL timestep.");
      }

      /* FluidSolver::step() is intentionally not used as a physics operator. The node owns time. */
      solver.mDt = static_cast<Real>(*next_dt);
      local_diagnostics.max_velocity = std::max(local_diagnostics.max_velocity, max_velocity);

      /* Basic legacy smoke ordering, with explicit compiled operators instead of Python scripts. */
      advect(flags, velocity, density);
      advect(flags, velocity, temperature);
      advect(flags, velocity, velocity);

      if (settings.heat_buoyancy != 0.0f) {
        Manta::addBuoyancy(flags,
                           temperature,
                           velocity,
                           Manta::Vec3(settings.gravity.x, settings.gravity.y, settings.gravity.z),
                           static_cast<Real>(settings.heat_buoyancy),
                           false);
      }
      if (settings.density_buoyancy != 0.0f) {
        Manta::addBuoyancy(flags,
                           density,
                           velocity,
                           Manta::Vec3(settings.gravity.x, settings.gravity.y, settings.gravity.z),
                           static_cast<Real>(settings.density_buoyancy),
                           false);
      }

      Manta::setWallBcs(flags, velocity, nullptr, nullptr, nullptr, 0);

      const PressureSolveResult pressure_result = solve_pressure(
          solver, velocity, pressure, flags, settings);
      local_diagnostics.pressure_iterations = std::max(local_diagnostics.pressure_iterations,
                                                       pressure_result.iterations);
      local_diagnostics.pressure_residual = std::max(local_diagnostics.pressure_residual,
                                                     pressure_result.residual);
      local_diagnostics.pressure_converged = local_diagnostics.pressure_converged &&
                                             pressure_result.converged;

      if (!grids_are_finite(density, temperature, pressure, velocity)) {
        throw std::runtime_error("Mantaflow produced a non-finite gas state.");
      }
      if (!driver.commit(*next_dt)) {
        throw std::runtime_error("Gas Simulation Step rejected an internal timestep.");
      }
    }

    /* Every substep already verified that the grids are finite. */
    for (std::size_t i = 0; i < cell_count; ++i) {
      state.density[i] = static_cast<float>(density[i]);
      state.temperature[i] = static_cast<float>(temperature[i]);
      state.pressure[i] = static_cast<float>(pressure[i]);
      state.velocity[i] = {static_cast<float>(velocity[i].x),
                           static_cast<float>(velocity[i].y),
                           static_cast<float>(velocity[i].z)};
    }

    local_diagnostics.velocity_layout = MantaGasVelocityLayout::Staggered;
    local_diagnostics.substeps = driver.substeps();
    local_diagnostics.simulated_time = static_cast<float>(driver.simulated_time());
    diagnostics = local_diagnostics;
    return true;
  }
  catch (const std::exception &error) {
    error_message = error.what();
  }
  catch (...) {
    error_message = "Unknown Mantaflow gas-step failure.";
  }
  return false;
}

}  // namespace blender
