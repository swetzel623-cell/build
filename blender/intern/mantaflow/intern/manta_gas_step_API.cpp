/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "manta_gas_step_API.h"

#include "manta_gas_step_driver.h"

#include "manta.h"
#include "fluidsolver.h"
#include "grid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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

void solvePressure(MACGrid &vel,
                   Grid<Real> &pressure,
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
                   Real surfTens,
                   Grid<Real> *retRhs);

}  // namespace Manta

namespace blender {
namespace {

constexpr int pressure_preconditioner_mic = 1;  // Manta::PcMIC from plugin/pressure.cpp.

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
    Manta::Grid<Manta::Vec3> centered_velocity(&solver);
    Manta::MACGrid velocity(&solver);

    for (std::size_t i = 0; i < cell_count; ++i) {
      density[i] = static_cast<Real>(state.density[i]);
      temperature[i] = static_cast<Real>(state.temperature[i]);
      pressure[i] = static_cast<Real>(state.pressure[i]);
      centered_velocity[i] = Manta::Vec3(static_cast<Real>(state.velocity[i].x),
                                         static_cast<Real>(state.velocity[i].y),
                                         static_cast<Real>(state.velocity[i].z));
    }
    Manta::resampleVec3ToMac(centered_velocity, velocity);

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

      /* PcMIC avoids pressure.cpp's process-global static multigrid preconditioner cache. */
      Manta::solvePressure(velocity,
                           pressure,
                           flags,
                           static_cast<Real>(settings.pressure_accuracy),
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           static_cast<Real>(1.0e-4),
                           static_cast<Real>(settings.pressure_iteration_factor),
                           true,
                           pressure_preconditioner_mic,
                           false,
                           false,
                           true,
                           nullptr,
                           static_cast<Real>(0.0),
                           nullptr);

      if (!grids_are_finite(density, temperature, pressure, velocity)) {
        throw std::runtime_error("Mantaflow produced a non-finite gas state.");
      }
      if (!driver.commit(*next_dt)) {
        throw std::runtime_error("Gas Simulation Step rejected an internal timestep.");
      }
    }

    Manta::resampleMacToVec3(velocity, centered_velocity);
    for (std::size_t i = 0; i < cell_count; ++i) {
      if (!std::isfinite(static_cast<double>(centered_velocity[i].x)) ||
          !std::isfinite(static_cast<double>(centered_velocity[i].y)) ||
          !std::isfinite(static_cast<double>(centered_velocity[i].z)))
      {
        throw std::runtime_error("Mantaflow produced a non-finite centered velocity.");
      }
      state.density[i] = static_cast<float>(density[i]);
      state.temperature[i] = static_cast<float>(temperature[i]);
      state.pressure[i] = static_cast<float>(pressure[i]);
      state.velocity[i] = {static_cast<float>(centered_velocity[i].x),
                           static_cast<float>(centered_velocity[i].y),
                           static_cast<float>(centered_velocity[i].z)};
    }

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
