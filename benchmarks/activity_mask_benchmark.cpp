// SPDX-License-Identifier: MIT
#include "manta_improve/activity_mask.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace manta_improve;
using Clock = std::chrono::steady_clock;

std::size_t index(Extent3 extent, int x, int y, int z)
{
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(extent.x) *
             (static_cast<std::size_t>(y) + static_cast<std::size_t>(extent.y) *
                                                  static_cast<std::size_t>(z)));
}

double seconds_since(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> make_rising_column(Extent3 extent)
{
  std::vector<float> density(extent.size(), 0.0f);
  const float center_x = 0.5f * static_cast<float>(extent.x - 1);
  const float center_y = 0.5f * static_cast<float>(extent.y - 1);
  const float radius = std::max(1.0f, static_cast<float>(extent.x) / 12.0f);
  const int begin_z = extent.z / 8;
  const int end_z = std::max(begin_z + 1, 7 * extent.z / 8);

  for (int z = begin_z; z < end_z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const float dx = static_cast<float>(x) - center_x;
        const float dy = static_cast<float>(y) - center_y;
        const float distance = std::sqrt(dx * dx + dy * dy);
        if (distance < radius) {
          density[index(extent, x, y, z)] = 1.0f - distance / radius;
        }
      }
    }
  }
  return density;
}

void stencil_cell(const std::vector<float> &input,
                  std::vector<float> &output,
                  Extent3 extent,
                  int x,
                  int y,
                  int z)
{
  auto sample = [&](int sx, int sy, int sz) {
    sx = std::clamp(sx, 0, extent.x - 1);
    sy = std::clamp(sy, 0, extent.y - 1);
    sz = std::clamp(sz, 0, extent.z - 1);
    return input[index(extent, sx, sy, sz)];
  };

  const float value = sample(x, y, z);
  const float neighbors = sample(x - 1, y, z) + sample(x + 1, y, z) +
                          sample(x, y - 1, z) + sample(x, y + 1, z) +
                          sample(x, y, z - 1) + sample(x, y, z + 1);
  output[index(extent, x, y, z)] = 0.5f * value + (1.0f / 12.0f) * neighbors;
}

void run_dense_stencil(const std::vector<float> &input,
                       std::vector<float> &output,
                       Extent3 extent)
{
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        stencil_cell(input, output, extent, x, y, z);
      }
    }
  }
}

void run_masked_stencil(const std::vector<float> &input,
                        std::vector<float> &output,
                        const ActivityMask &mask)
{
  const Extent3 extent = mask.grid_extent();
  const int block_size = mask.block_size();
  for (const int flattened : mask.active_blocks()) {
    const auto block = mask.block_coordinates(flattened);
    const int end_x = std::min((block[0] + 1) * block_size, extent.x);
    const int end_y = std::min((block[1] + 1) * block_size, extent.y);
    const int end_z = std::min((block[2] + 1) * block_size, extent.z);
    for (int z = block[2] * block_size; z < end_z; ++z) {
      for (int y = block[1] * block_size; y < end_y; ++y) {
        for (int x = block[0] * block_size; x < end_x; ++x) {
          stencil_cell(input, output, extent, x, y, z);
        }
      }
    }
  }
}

double checksum(const std::vector<float> &values)
{
  double result = 0.0;
  for (const float value : values) {
    result += value;
  }
  return result;
}

float max_difference(const std::vector<float> &a, const std::vector<float> &b)
{
  float result = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(a[i] - b[i]));
  }
  return result;
}

int positive_argument(const char *text, const char *name)
{
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    if (argc > 4) {
      std::cerr << "Usage: activity_mask_benchmark [grid_size=192] [block_size=8] [repetitions=10]\n";
      return 2;
    }
    const int grid_size = argc > 1 ? positive_argument(argv[1], "Grid size") : 192;
    const int block_size = argc > 2 ? positive_argument(argv[2], "Block size") : 8;
    const int repetitions = argc > 3 ? positive_argument(argv[3], "Repetitions") : 10;
    const Extent3 extent{grid_size, grid_size, grid_size};

    const std::vector<float> density = make_rising_column(extent);
    std::vector<float> dense_output(extent.size(), 0.0f);
    std::vector<float> masked_output(extent.size(), 0.0f);
    ActivityFields fields;
    fields.density = {density.data(), extent};

    double dense_seconds = 0.0;
    double masked_kernel_seconds = 0.0;
    double mask_rebuild_seconds = 0.0;
    double source_fraction = 0.0;
    double dilated_fraction = 0.0;

    for (int repetition = 0; repetition < repetitions; ++repetition) {
      auto start = Clock::now();
      run_dense_stencil(density, dense_output, extent);
      dense_seconds += seconds_since(start);

      start = Clock::now();
      const auto source_mask = build_activity_mask(fields, extent, block_size);
      const auto dilated_mask = dilate_activity_mask(source_mask, 1);
      mask_rebuild_seconds += seconds_since(start);

      start = Clock::now();
      run_masked_stencil(density, masked_output, dilated_mask);
      masked_kernel_seconds += seconds_since(start);
      if (repetition == 0) {
        source_fraction = source_mask.active_block_fraction();
        dilated_fraction = dilated_mask.active_block_fraction();
      }
    }

    const double masked_seconds = mask_rebuild_seconds + masked_kernel_seconds;

    const float difference = max_difference(dense_output, masked_output);
    if (difference != 0.0f) {
      throw std::runtime_error("Masked and dense stencil results differ");
    }

    std::cout << std::setprecision(9) << "{\n"
              << "  \"benchmark\": \"local_stencil_with_mask_rebuild\",\n"
              << "  \"grid\": [" << extent.x << ", " << extent.y << ", " << extent.z
              << "],\n"
              << "  \"block_size\": " << block_size << ",\n"
              << "  \"repetitions\": " << repetitions << ",\n"
              << "  \"source_active_block_fraction\": " << source_fraction << ",\n"
              << "  \"dilated_active_block_fraction\": " << dilated_fraction << ",\n"
              << "  \"dense_seconds_total\": " << dense_seconds << ",\n"
              << "  \"mask_rebuild_seconds_total\": " << mask_rebuild_seconds << ",\n"
              << "  \"masked_kernel_seconds_total\": " << masked_kernel_seconds << ",\n"
              << "  \"masked_seconds_total\": " << masked_seconds << ",\n"
              << "  \"speedup_including_mask_rebuild\": " << dense_seconds / masked_seconds << ",\n"
              << "  \"dense_checksum\": " << checksum(dense_output) << ",\n"
              << "  \"masked_checksum\": " << checksum(masked_output) << ",\n"
              << "  \"max_abs_difference\": " << difference << "\n"
              << "}\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "Benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
