// SPDX-License-Identifier: MIT
#include "manta_improve/activity_mask.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace manta_improve {
namespace {

std::size_t flatten(Extent3 extent, int x, int y, int z)
{
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(extent.x) *
             (static_cast<std::size_t>(y) + static_cast<std::size_t>(extent.y) *
                                                  static_cast<std::size_t>(z)));
}

int ceil_div(int value, int divisor)
{
  return value == 0 ? 0 : (value + divisor - 1) / divisor;
}

void validate_extent(Extent3 extent)
{
  if (extent.x < 0 || extent.y < 0 || extent.z < 0) {
    throw std::invalid_argument("Grid extents must be non-negative");
  }
  if (extent.x > 0 && extent.y > 0 &&
      static_cast<std::size_t>(extent.x) >
          std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(extent.y)) {
    throw std::overflow_error("Grid extent is too large");
  }
  const std::size_t xy = static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y);
  if (extent.z > 0 && xy > std::numeric_limits<std::size_t>::max() /
                               static_cast<std::size_t>(extent.z)) {
    throw std::overflow_error("Grid extent is too large");
  }
}

void validate_threshold(float value, const char *name)
{
  if (!std::isfinite(value) || value < 0.0f) {
    throw std::invalid_argument(std::string("Invalid activity threshold: ") + name);
  }
}

}  // namespace

std::size_t Extent3::size() const noexcept
{
  if (x <= 0 || y <= 0 || z <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(x) * static_cast<std::size_t>(y) *
         static_cast<std::size_t>(z);
}

bool Extent3::contains(int px, int py, int pz) const noexcept
{
  return px >= 0 && py >= 0 && pz >= 0 && px < x && py < y && pz < z;
}

float ScalarGridView::operator()(int x, int y, int z) const
{
  return values_[flatten(extent_, x, y, z)];
}

Vec3 VectorGridView::operator()(int x, int y, int z) const
{
  return values_[flatten(extent_, x, y, z)];
}

ActivityMask::ActivityMask(Extent3 grid_extent, int block_size)
    : grid_extent_(grid_extent), block_size_(block_size)
{
  validate_extent(grid_extent_);
  if (block_size_ <= 0) {
    throw std::invalid_argument("Block size must be positive");
  }

  block_extent_ = {ceil_div(grid_extent_.x, block_size_),
                   ceil_div(grid_extent_.y, block_size_),
                   ceil_div(grid_extent_.z, block_size_)};
  block_flags_.resize(block_extent_.size(), 0);
}

bool ActivityMask::block_active(int x, int y, int z) const
{
  if (!block_extent_.contains(x, y, z)) {
    throw std::out_of_range("Block coordinate is outside the mask");
  }
  return block_flags_[flatten(block_extent_, x, y, z)] != 0;
}

void ActivityMask::set_block_active(int x, int y, int z)
{
  block_flags_[flatten(block_extent_, x, y, z)] = 1;
}

std::array<int, 3> ActivityMask::block_coordinates(int flattened_block) const
{
  if (flattened_block < 0 ||
      static_cast<std::size_t>(flattened_block) >= block_extent_.size()) {
    throw std::out_of_range("Flattened block coordinate is outside the mask");
  }
  const int xy = block_extent_.x * block_extent_.y;
  const int z = flattened_block / xy;
  const int remainder = flattened_block % xy;
  const int y = remainder / block_extent_.x;
  const int x = remainder % block_extent_.x;
  return {x, y, z};
}

double ActivityMask::active_block_fraction() const noexcept
{
  if (block_flags_.empty()) {
    return 0.0;
  }
  return static_cast<double>(active_blocks_.size()) / static_cast<double>(block_flags_.size());
}

std::size_t ActivityMask::active_voxel_upper_bound() const
{
  std::size_t result = 0;
  for (const int flattened : active_blocks_) {
    const auto block = block_coordinates(flattened);
    const int begin_x = block[0] * block_size_;
    const int begin_y = block[1] * block_size_;
    const int begin_z = block[2] * block_size_;
    const int size_x = std::min(block_size_, grid_extent_.x - begin_x);
    const int size_y = std::min(block_size_, grid_extent_.y - begin_y);
    const int size_z = std::min(block_size_, grid_extent_.z - begin_z);
    result += static_cast<std::size_t>(size_x) * static_cast<std::size_t>(size_y) *
              static_cast<std::size_t>(size_z);
  }
  return result;
}

void ActivityMask::rebuild_active_block_list()
{
  active_blocks_.clear();
  for (std::size_t index = 0; index < block_flags_.size(); ++index) {
    if (block_flags_[index] != 0) {
      if (index > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Flattened block index exceeds the public index type");
      }
      active_blocks_.push_back(static_cast<int>(index));
    }
  }
}

ActivityMask build_activity_mask(const ActivityFields &fields,
                                 Extent3 grid_extent,
                                 int block_size,
                                 const ActivityThresholds &thresholds)
{
  validate_threshold(thresholds.density, "density");
  validate_threshold(thresholds.temperature, "temperature");
  validate_threshold(thresholds.fuel, "fuel");
  validate_threshold(thresholds.velocity, "velocity");

  ActivityMask mask(grid_extent, block_size);
  if (grid_extent.size() == 0) {
    return mask;
  }

  auto validate_grid = [&](Extent3 extent, const char *name) {
    if (!(extent == grid_extent)) {
      throw std::invalid_argument(std::string("Activity grid extent mismatch: ") + name);
    }
  };

  auto scan_scalar = [&](const ScalarGridView &grid, float threshold, const char *name) {
    if (grid.empty()) {
      return;
    }
    validate_grid(grid.extent(), name);

    for (int bz = 0; bz < mask.block_extent_.z; ++bz) {
      for (int by = 0; by < mask.block_extent_.y; ++by) {
        for (int bx = 0; bx < mask.block_extent_.x; ++bx) {
          if (mask.block_active(bx, by, bz)) {
            continue;
          }
          const int end_x = std::min((bx + 1) * block_size, grid_extent.x);
          const int end_y = std::min((by + 1) * block_size, grid_extent.y);
          const int end_z = std::min((bz + 1) * block_size, grid_extent.z);
          bool active = false;
          for (int z = bz * block_size; z < end_z && !active; ++z) {
            for (int y = by * block_size; y < end_y && !active; ++y) {
              for (int x = bx * block_size; x < end_x; ++x) {
                const float value = grid(x, y, z);
                if (!std::isfinite(value) || std::abs(value) > threshold) {
                  active = true;
                  break;
                }
              }
            }
          }
          if (active) {
            mask.set_block_active(bx, by, bz);
          }
        }
      }
    }
  };

  scan_scalar(fields.density, thresholds.density, "density");
  scan_scalar(fields.temperature, thresholds.temperature, "temperature");
  scan_scalar(fields.fuel, thresholds.fuel, "fuel");

  if (!fields.velocity.empty()) {
    validate_grid(fields.velocity.extent(), "velocity");
    const float threshold_squared = thresholds.velocity * thresholds.velocity;
    for (int bz = 0; bz < mask.block_extent_.z; ++bz) {
      for (int by = 0; by < mask.block_extent_.y; ++by) {
        for (int bx = 0; bx < mask.block_extent_.x; ++bx) {
          if (mask.block_active(bx, by, bz)) {
            continue;
          }
          const int end_x = std::min((bx + 1) * block_size, grid_extent.x);
          const int end_y = std::min((by + 1) * block_size, grid_extent.y);
          const int end_z = std::min((bz + 1) * block_size, grid_extent.z);
          bool active = false;
          for (int z = bz * block_size; z < end_z && !active; ++z) {
            for (int y = by * block_size; y < end_y && !active; ++y) {
              for (int x = bx * block_size; x < end_x; ++x) {
                const Vec3 value = fields.velocity(x, y, z);
                const float magnitude_squared = value.x * value.x + value.y * value.y +
                                                value.z * value.z;
                if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                    !std::isfinite(value.z) || magnitude_squared > threshold_squared) {
                  active = true;
                  break;
                }
              }
            }
          }
          if (active) {
            mask.set_block_active(bx, by, bz);
          }
        }
      }
    }
  }

  mask.rebuild_active_block_list();
  return mask;
}

ActivityMask dilate_activity_mask(const ActivityMask &mask, int radius)
{
  if (radius < 0) {
    throw std::invalid_argument("Dilation radius must be non-negative");
  }
  if (radius == 0) {
    return mask;
  }

  ActivityMask result(mask.grid_extent_, mask.block_size_);
  for (const int flattened : mask.active_blocks_) {
    const auto block = mask.block_coordinates(flattened);
    const int end_x = std::min(mask.block_extent_.x, block[0] + radius + 1);
    const int end_y = std::min(mask.block_extent_.y, block[1] + radius + 1);
    const int end_z = std::min(mask.block_extent_.z, block[2] + radius + 1);
    for (int z = std::max(0, block[2] - radius); z < end_z; ++z) {
      for (int y = std::max(0, block[1] - radius); y < end_y; ++y) {
        for (int x = std::max(0, block[0] - radius); x < end_x; ++x) {
          result.set_block_active(x, y, z);
        }
      }
    }
  }
  result.rebuild_active_block_list();
  return result;
}

}  // namespace manta_improve
