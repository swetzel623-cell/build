// SPDX-License-Identifier: MIT
#include "manta_gas_layout.h"
#include "manta_gas_step_API.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using blender::MantaGasState;
using blender::MantaGasStepDiagnostics;
using blender::MantaGasStepSettings;
using blender::MantaGasVec3;

constexpr int resolution = 8;
constexpr std::size_t cell_count = resolution * resolution * resolution;

void require(const bool condition, const std::string &message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(const double actual, const double expected, const std::string &message)
{
  if (std::abs(actual - expected) > 1.0e-5) {
    throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}

struct DenseState {
  std::vector<float> density = std::vector<float>(cell_count, 0.0f);
  std::vector<float> temperature = std::vector<float>(cell_count, 0.0f);
  std::vector<MantaGasVec3> velocity = std::vector<MantaGasVec3>(cell_count);
  std::vector<float> pressure = std::vector<float>(cell_count, 0.0f);

  MantaGasState state()
  {
    return {{resolution, resolution, resolution},
            density.data(),
            temperature.data(),
            velocity.data(),
            pressure.data()};
  }
};

MantaGasStepSettings settings()
{
  MantaGasStepSettings result;
  result.delta_time = 1.0f / 30.0f;
  result.cfl = 2.0f;
  result.max_substeps = 8;
  result.gravity = {0.0f, 0.0f, 0.0f};
  result.density_buoyancy = 0.0f;
  result.heat_buoyancy = 0.0f;
  return result;
}

std::size_t center_index()
{
  return blender::manta_gas::grid_index(
      resolution, resolution, resolution / 2, resolution / 2, resolution / 2);
}

bool step(DenseState &dense,
          const MantaGasStepSettings &step_settings,
          MantaGasStepDiagnostics &diagnostics,
          std::string &error)
{
  MantaGasState state = dense.state();
  return blender::manta_gas_simulation_step(state, step_settings, diagnostics, error);
}

void require_same(const DenseState &a, const DenseState &b, const std::string &message)
{
  require(a.density == b.density && a.temperature == b.temperature && a.pressure == b.pressure,
          message + " (scalar grid changed)");
  const bool same_velocity = std::equal(a.velocity.begin(),
                                        a.velocity.end(),
                                        b.velocity.begin(),
                                        [](const MantaGasVec3 &left, const MantaGasVec3 &right) {
                                          return left.x == right.x && left.y == right.y &&
                                                 left.z == right.z;
                                        });
  require(same_velocity, message + " (velocity grid changed)");
}

void require_finite(const DenseState &dense)
{
  for (std::size_t i = 0; i < cell_count; ++i) {
    require(std::isfinite(dense.density[i]) && std::isfinite(dense.temperature[i]) &&
                std::isfinite(dense.pressure[i]) && std::isfinite(dense.velocity[i].x) &&
                std::isfinite(dense.velocity[i].y) && std::isfinite(dense.velocity[i].z),
            "Engine returned a non-finite value");
  }
}

float velocity_energy(const DenseState &dense)
{
  double result = 0.0;
  for (const MantaGasVec3 &value : dense.velocity) {
    result += value.x * value.x + value.y * value.y + value.z * value.z;
  }
  return static_cast<float>(result);
}

float pressure_energy(const DenseState &dense)
{
  double result = 0.0;
  for (const float value : dense.pressure) {
    result += value * value;
  }
  return static_cast<float>(result);
}

void test_zero_time_preserves_state()
{
  DenseState dense;
  dense.density[center_index()] = 0.75f;
  dense.temperature[center_index()] = 1.25f;
  dense.pressure[center_index()] = -0.5f;
  dense.velocity[center_index()] = {1.0f, 2.0f, 3.0f};
  const DenseState original = dense;
  MantaGasStepSettings step_settings = settings();
  step_settings.delta_time = 0.0f;
  MantaGasStepDiagnostics diagnostics;
  std::string error;
  require(step(dense, step_settings, diagnostics, error), "Zero-time step failed: " + error);
  require_same(dense, original, "Zero-time step modified the state");
  require(diagnostics.substeps == 0 && diagnostics.simulated_time == 0.0f,
          "Zero-time step reported simulation work");
}

void test_stationary_uniform_state()
{
  DenseState dense;
  std::fill(dense.density.begin(), dense.density.end(), 0.5f);
  std::fill(dense.temperature.begin(), dense.temperature.end(), 0.25f);
  MantaGasStepDiagnostics diagnostics;
  std::string error;
  require(step(dense, settings(), diagnostics, error), "Stationary step failed: " + error);
  require_finite(dense);
  require(diagnostics.substeps == 1, "Stationary frame did not use one substep");
  require_near(diagnostics.simulated_time, 1.0 / 30.0, "Incorrect stationary simulated time");
  require(velocity_energy(dense) == 0.0f, "Stationary state generated velocity");
  for (int z = 1; z < resolution - 1; ++z) {
    for (int y = 1; y < resolution - 1; ++y) {
      for (int x = 1; x < resolution - 1; ++x) {
        const std::size_t i = blender::manta_gas::grid_index(resolution, resolution, x, y, z);
        require_near(dense.density[i], 0.5, "Uniform interior density was not preserved");
        require_near(dense.temperature[i], 0.25, "Uniform interior temperature was not preserved");
      }
    }
  }
}

void test_heat_buoyancy_and_pressure()
{
  DenseState dense;
  dense.density[center_index()] = 1.0f;
  dense.temperature[center_index()] = 1.0f;
  MantaGasStepSettings step_settings = settings();
  step_settings.gravity = {0.0f, 0.0f, -2.0f};
  step_settings.heat_buoyancy = 1.0f;
  MantaGasStepDiagnostics diagnostics;
  std::string error;
  require(step(dense, step_settings, diagnostics, error), "Buoyancy step failed: " + error);
  require_finite(dense);
  require(diagnostics.substeps == 1, "Buoyancy frame did not use one substep");
  require(velocity_energy(dense) > 0.0f, "Heat buoyancy produced no velocity");
  require(pressure_energy(dense) > 0.0f, "Pressure state was not retained in the output");
}

void test_cfl_substeps_complete_frame()
{
  DenseState dense;
  std::fill(dense.density.begin(), dense.density.end(), 0.5f);
  std::fill(dense.velocity.begin(), dense.velocity.end(), MantaGasVec3{4.0f, 0.0f, 0.0f});
  MantaGasStepSettings step_settings = settings();
  step_settings.delta_time = 0.5f;
  step_settings.cfl = 1.0f;
  MantaGasStepDiagnostics diagnostics;
  std::string error;
  require(step(dense, step_settings, diagnostics, error), "CFL-limited step failed: " + error);
  require_finite(dense);
  require(diagnostics.substeps >= 2, "CFL limit did not subdivide the frame");
  require_near(diagnostics.simulated_time, 0.5, "CFL substeps did not complete the frame");
  require(diagnostics.max_velocity >= 3.99f, "CFL scheduler did not sample the input velocity");
}

void test_substep_cap_is_atomic()
{
  DenseState dense;
  std::fill(dense.density.begin(), dense.density.end(), 0.5f);
  std::fill(dense.velocity.begin(), dense.velocity.end(), MantaGasVec3{4.0f, 0.0f, 0.0f});
  const DenseState original = dense;
  MantaGasStepSettings step_settings = settings();
  step_settings.delta_time = 0.5f;
  step_settings.cfl = 1.0f;
  step_settings.max_substeps = 1;
  MantaGasStepDiagnostics diagnostics;
  std::string error;
  require(!step(dense, step_settings, diagnostics, error),
          "Substep-capped frame incorrectly reported success");
  require(!error.empty(), "Substep-capped frame returned no error");
  require_same(dense, original, "Failed frame modified caller buffers");
  require(diagnostics.substeps == 0 && diagnostics.simulated_time == 0.0f &&
              diagnostics.max_velocity == 0.0f,
          "Failed frame leaked partial diagnostics");
}

void test_nonfinite_input_is_atomic()
{
  DenseState dense;
  dense.density[center_index()] = std::numeric_limits<float>::quiet_NaN();
  dense.velocity[3] = {1.0f, 2.0f, 3.0f};
  MantaGasStepDiagnostics diagnostics;
  std::string error;
  require(!step(dense, settings(), diagnostics, error), "NaN input incorrectly succeeded");
  require(!error.empty(), "NaN input returned no error");
  require(std::isnan(dense.density[center_index()]), "Rejected NaN input was modified");
  require(dense.velocity[3].x == 1.0f && dense.velocity[3].y == 2.0f &&
              dense.velocity[3].z == 3.0f,
          "Rejected input changed another grid");
}

void test_independent_sequential_states()
{
  DenseState heated;
  heated.temperature[center_index()] = 1.0f;
  MantaGasStepSettings heated_settings = settings();
  heated_settings.gravity = {0.0f, 0.0f, -2.0f};
  heated_settings.heat_buoyancy = 1.0f;
  MantaGasStepDiagnostics heated_diagnostics;
  std::string error;
  require(step(heated, heated_settings, heated_diagnostics, error),
          "First independent state failed: " + error);

  DenseState empty;
  MantaGasStepDiagnostics empty_diagnostics;
  require(step(empty, settings(), empty_diagnostics, error),
          "Second independent state failed: " + error);
  require_finite(heated);
  require_finite(empty);
  require(velocity_energy(heated) > 0.0f, "First state did not develop buoyant motion");
  require(velocity_energy(empty) == 0.0f && pressure_energy(empty) == 0.0f,
          "Second state observed data from the first solver");
}

}  // namespace

int main()
{
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"zero time preserves state", test_zero_time_preserves_state},
      {"stationary uniform state", test_stationary_uniform_state},
      {"heat buoyancy and pressure", test_heat_buoyancy_and_pressure},
      {"CFL substeps complete frame", test_cfl_substeps_complete_frame},
      {"substep cap is atomic", test_substep_cap_is_atomic},
      {"non-finite input is atomic", test_nonfinite_input_is_atomic},
      {"independent sequential states", test_independent_sequential_states},
  };

  for (const auto &[name, test] : tests) {
    test();
    std::cout << "PASS: " << name << '\n';
  }
  return 0;
}
