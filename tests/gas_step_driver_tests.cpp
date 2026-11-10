// SPDX-License-Identifier: MIT
#include "manta_gas_layout.h"
#include "manta_gas_step_driver.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using blender::manta_gas::GasStepDriver;

void require(const bool condition, const std::string &message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(const double actual, const double expected, const std::string &message)
{
  if (std::abs(actual - expected) > 1.0e-6) {
    throw std::runtime_error(message);
  }
}

void test_x_fastest_layout()
{
  using blender::manta_gas::grid_index;
  require(grid_index(17, 9, 0, 0, 0) == 0, "Origin index must be zero");
  require(grid_index(17, 9, 1, 0, 0) == 1, "X must be the fastest axis");
  require(grid_index(17, 9, 0, 1, 0) == 17, "Y stride must equal the X resolution");
  require(grid_index(17, 9, 0, 0, 1) == 153, "Z stride must equal X times Y resolution");
  require(grid_index(17, 9, 16, 8, 12) == 16 + 17 * (8 + 9 * 12),
          "Last-domain index does not match the X-fastest contract");
}

void test_invalid_settings()
{
  require(!GasStepDriver(-1.0, 2.0, 8).valid(), "Negative frame time was accepted");
  require(!GasStepDriver(1.0, 0.0, 8).valid(), "Zero CFL number was accepted");
  require(!GasStepDriver(1.0, 2.0, 0).valid(), "Zero substep limit was accepted");
  require(!GasStepDriver(std::numeric_limits<double>::quiet_NaN(), 2.0, 8).valid(),
          "NaN frame time was accepted");
}

void test_zero_length_frame()
{
  GasStepDriver driver(0.0, 2.0, 8);
  require(driver.valid(), "Zero-length frame should be valid");
  require(driver.finished(), "Zero-length frame should already be finished");
  require(!driver.next_dt(0.0f).has_value(), "Zero-length frame produced a timestep");
  require(driver.substeps() == 0, "Zero-length frame produced a substep");
}

void test_stationary_frame_uses_one_step()
{
  GasStepDriver driver(1.0 / 30.0, 2.0, 8);
  const auto dt = driver.next_dt(0.0f);
  require(dt.has_value(), "Stationary frame did not produce a timestep");
  require_near(*dt, 1.0 / 30.0, "Stationary frame should use the full frame time");
  require(driver.commit(*dt), "Valid timestep commit failed");
  require(driver.finished(), "Full frame timestep did not finish the frame");
  require(driver.substeps() == 1, "Unexpected stationary substep count");
}

void test_cfl_limit_and_final_remainder()
{
  GasStepDriver driver(1.0, 2.0, 8);
  const auto first = driver.next_dt(4.0f);
  require(first.has_value(), "CFL-limited frame did not produce a timestep");
  require_near(*first, 0.5, "Incorrect CFL timestep");
  require(driver.commit(*first), "First CFL timestep commit failed");

  const auto second = driver.next_dt(10.0f);
  require(second.has_value(), "Second CFL timestep was not produced");
  require_near(*second, 0.2, "CFL limit did not respond to increased velocity");
  require(driver.commit(*second), "Second CFL timestep commit failed");

  const auto final = driver.next_dt(1.0f);
  require(final.has_value(), "Final frame remainder was not produced");
  require_near(*final, 0.3, "Final timestep should consume exactly the remaining time");
  require(driver.commit(*final), "Final timestep commit failed");
  require(driver.finished(), "Frame did not finish after its remainder was consumed");
  require_near(driver.simulated_time(), 1.0, "Simulated time does not equal frame time");
}

void test_substep_cap_rejects_partial_frame()
{
  GasStepDriver driver(1.0, 1.0, 1);
  const auto first = driver.next_dt(4.0f);
  require(first.has_value() && std::abs(*first - 0.25f) < 1.0e-6f,
          "Unexpected capped-frame timestep");
  require(driver.commit(*first), "Capped-frame timestep commit failed");
  require(!driver.finished(), "Partial frame incorrectly reported completion");
  require(!driver.next_dt(4.0f).has_value(),
          "Scheduler allowed a partial frame to exceed its substep cap");
}

void test_invalid_velocity_is_not_scheduled()
{
  GasStepDriver driver(1.0, 2.0, 8);
  require(!driver.next_dt(std::numeric_limits<float>::quiet_NaN()).has_value(),
          "NaN velocity was scheduled");
  require(!driver.next_dt(std::numeric_limits<float>::infinity()).has_value(),
          "Infinite velocity was scheduled");
  require(!driver.next_dt(-1.0f).has_value(), "Negative velocity magnitude was scheduled");
}

void test_commit_cannot_overshoot_frame()
{
  GasStepDriver driver(0.5, 2.0, 8);
  require(!driver.commit(0.75f), "Commit accepted a timestep beyond the frame boundary");
  require(driver.substeps() == 0, "Rejected commit changed the substep count");
}

}  // namespace

int main()
{
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"X-fastest layout", test_x_fastest_layout},
      {"invalid settings", test_invalid_settings},
      {"zero-length frame", test_zero_length_frame},
      {"stationary frame", test_stationary_frame_uses_one_step},
      {"CFL limit and remainder", test_cfl_limit_and_final_remainder},
      {"substep cap", test_substep_cap_rejects_partial_frame},
      {"invalid velocity", test_invalid_velocity_is_not_scheduled},
      {"commit bounds", test_commit_cannot_overshoot_frame},
  };

  for (const auto &[name, test] : tests) {
    test();
    std::cout << "PASS: " << name << '\n';
  }
  return 0;
}
