// SPDX-License-Identifier: MIT
#include "manta_improve/activity_mask.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace manta_improve;

void require(bool condition, const std::string &message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::size_t index(Extent3 extent, int x, int y, int z)
{
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(extent.x) *
             (static_cast<std::size_t>(y) + static_cast<std::size_t>(extent.y) *
                                                  static_cast<std::size_t>(z));
}

void test_empty_mask()
{
  const Extent3 extent{24, 16, 16};
  const auto mask = build_activity_mask({}, extent);
  require(mask.block_extent() == Extent3{3, 2, 2}, "Unexpected block extent");
  require(mask.active_block_count() == 0, "Empty inputs produced active blocks");
  require(mask.active_block_fraction() == 0.0, "Empty mask fraction must be zero");
  require(mask.active_voxel_upper_bound() == 0, "Empty mask must cover no voxels");
}

void test_partial_edge_block()
{
  const Extent3 extent{17, 9, 13};
  std::vector<float> density(extent.size(), 0.0f);
  density[index(extent, 16, 8, 12)] = 1.0f;

  ActivityFields fields;
  fields.density = {density.data(), extent};
  const auto mask = build_activity_mask(fields, extent);

  require(mask.block_extent() == Extent3{3, 2, 2}, "Unexpected partial-block layout");
  require(mask.active_block_count() == 1, "Expected one active edge block");
  require(mask.block_active(2, 1, 1), "Wrong block activated");
  require(mask.active_voxel_upper_bound() == 5, "Edge-block coverage was not clipped");
  require(mask.block_coordinates(mask.active_blocks()[0]) == std::array<int, 3>{2, 1, 1},
          "Flattened block coordinate round trip failed");
}

void test_multi_field_union_and_signed_velocity()
{
  const Extent3 extent{16, 16, 8};
  std::vector<float> temperature(extent.size(), 0.0f);
  std::vector<float> fuel(extent.size(), 0.0f);
  std::vector<Vec3> velocity(extent.size());
  temperature[index(extent, 1, 1, 1)] = 0.5f;
  fuel[index(extent, 10, 1, 1)] = 0.5f;
  velocity[index(extent, 1, 9, 1)].y = -2.0f;

  ActivityFields fields;
  fields.temperature = {temperature.data(), extent};
  fields.fuel = {fuel.data(), extent};
  fields.velocity = {velocity.data(), extent};
  const auto mask = build_activity_mask(fields, extent);

  require(mask.active_block_count() == 3, "Activity union or signed velocity failed");
  require(mask.block_active(0, 0, 0), "Temperature activity missing");
  require(mask.block_active(1, 0, 0), "Fuel activity missing");
  require(mask.block_active(0, 1, 0), "Velocity activity missing");
}

void test_threshold_and_nonfinite_values()
{
  const Extent3 extent{8, 8, 8};
  std::vector<float> density(extent.size(), 0.0f);
  density[index(extent, 0, 0, 0)] = 1.0e-5f;
  ActivityFields fields;
  fields.density = {density.data(), extent};
  auto mask = build_activity_mask(fields, extent);
  require(mask.active_block_count() == 0, "A value at the threshold should remain inactive");

  density[index(extent, 0, 0, 0)] = std::numeric_limits<float>::quiet_NaN();
  mask = build_activity_mask(fields, extent);
  require(mask.active_block_count() == 1, "NaN must not be hidden by activity culling");

  density[index(extent, 0, 0, 0)] = std::numeric_limits<float>::infinity();
  mask = build_activity_mask(fields, extent);
  require(mask.active_block_count() == 1, "Infinity must not be hidden by activity culling");
}

void test_dilation_clips_at_grid_boundary()
{
  const Extent3 extent{24, 24, 24};
  std::vector<float> density(extent.size(), 0.0f);
  density[0] = 1.0f;
  ActivityFields fields;
  fields.density = {density.data(), extent};

  const auto mask = build_activity_mask(fields, extent);
  const auto dilated = dilate_activity_mask(mask, 1);
  require(mask.active_block_count() == 1, "Unexpected initial mask size");
  require(dilated.active_block_count() == 8, "Corner dilation should cover eight blocks");
  require(dilated.active_voxel_upper_bound() == 4096,
          "Dilated corner coverage should be two 8^3 blocks per axis");

  const auto unchanged = dilate_activity_mask(mask, 0);
  require(unchanged.active_block_count() == 1, "Zero-radius dilation changed the mask");
}

void test_invalid_inputs()
{
  bool threw = false;
  try {
    ActivityMask mask({8, 8, 8}, 0);
  }
  catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "Zero block size was accepted");

  threw = false;
  try {
    ActivityMask mask({-1, 8, 8}, 8);
  }
  catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "Negative grid extent was accepted");

  const Extent3 extent{16, 8, 8};
  std::vector<float> density(8 * 8 * 8, 0.0f);
  ActivityFields fields;
  fields.density = {density.data(), {8, 8, 8}};
  threw = false;
  try {
    const auto mask = build_activity_mask(fields, extent);
    (void)mask;
  }
  catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "Mismatched grid extents were accepted");

  const auto mask = build_activity_mask({}, extent);
  threw = false;
  try {
    const auto dilated = dilate_activity_mask(mask, -1);
    (void)dilated;
  }
  catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "Negative dilation radius was accepted");
}

void test_zero_sized_grid()
{
  const auto mask = build_activity_mask({}, {0, 16, 16});
  require(mask.active_block_count() == 0, "Zero-sized grid became active");
  require(mask.active_voxel_upper_bound() == 0, "Zero-sized grid covers voxels");
}

}  // namespace

int main()
{
  const std::vector<std::pair<const char *, std::function<void()>>> tests = {
      {"empty mask", test_empty_mask},
      {"partial edge block", test_partial_edge_block},
      {"multi-field union and signed velocity", test_multi_field_union_and_signed_velocity},
      {"threshold and non-finite values", test_threshold_and_nonfinite_values},
      {"dilation clipping", test_dilation_clips_at_grid_boundary},
      {"invalid inputs", test_invalid_inputs},
      {"zero-sized grid", test_zero_sized_grid},
  };

  int failures = 0;
  for (const auto &test : tests) {
    try {
      test.second();
      std::cout << "[PASS] " << test.first << '\n';
    }
    catch (const std::exception &error) {
      ++failures;
      std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
    }
  }

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " tests passed\n";
  return 0;
}
