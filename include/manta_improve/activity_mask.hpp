// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace manta_improve {

struct Extent3 {
  int x = 0;
  int y = 0;
  int z = 0;

  std::size_t size() const noexcept;
  bool contains(int px, int py, int pz) const noexcept;

  friend bool operator==(const Extent3 &a, const Extent3 &b) noexcept
  {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

// Non-owning views keep the prototype independent of Mantaflow, OpenVDB, and
// Geometry Nodes storage. Production adapters must preserve their native layout.
class ScalarGridView {
 public:
  ScalarGridView() = default;
  ScalarGridView(const float *values, Extent3 extent) : values_(values), extent_(extent) {}

  bool empty() const noexcept { return values_ == nullptr; }
  Extent3 extent() const noexcept { return extent_; }
  float operator()(int x, int y, int z) const;

 private:
  const float *values_ = nullptr;
  Extent3 extent_;
};

class VectorGridView {
 public:
  VectorGridView() = default;
  VectorGridView(const Vec3 *values, Extent3 extent) : values_(values), extent_(extent) {}

  bool empty() const noexcept { return values_ == nullptr; }
  Extent3 extent() const noexcept { return extent_; }
  Vec3 operator()(int x, int y, int z) const;

 private:
  const Vec3 *values_ = nullptr;
  Extent3 extent_;
};

struct ActivityFields {
  ScalarGridView density;
  ScalarGridView temperature;
  ScalarGridView fuel;
  VectorGridView velocity;
};

struct ActivityThresholds {
  float density = 1.0e-5f;
  float temperature = 1.0e-5f;
  float fuel = 1.0e-5f;
  float velocity = 1.0e-5f;
};

class ActivityMask {
 public:
  ActivityMask(Extent3 grid_extent, int block_size = 8);

  Extent3 grid_extent() const noexcept { return grid_extent_; }
  Extent3 block_extent() const noexcept { return block_extent_; }
  int block_size() const noexcept { return block_size_; }
  std::size_t active_block_count() const noexcept { return active_blocks_.size(); }
  double active_block_fraction() const noexcept;

  bool block_active(int x, int y, int z) const;
  const std::vector<int> &active_blocks() const noexcept { return active_blocks_; }
  std::array<int, 3> block_coordinates(int flattened_block) const;

  // This is a block-coverage bound, not a count of exactly active voxels.
  std::size_t active_voxel_upper_bound() const;

 private:
  void set_block_active(int x, int y, int z);
  void rebuild_active_block_list();

  Extent3 grid_extent_;
  Extent3 block_extent_;
  int block_size_;
  std::vector<std::uint8_t> block_flags_;
  std::vector<int> active_blocks_;

  friend ActivityMask build_activity_mask(const ActivityFields &,
                                          Extent3,
                                          int,
                                          const ActivityThresholds &);
  friend ActivityMask dilate_activity_mask(const ActivityMask &, int);
};

// Builds the union of scalar-field and velocity activity. The build scans the
// input grids, so it is not itself an active-voxel-proportional algorithm.
// This mask is suitable for local advection/force experiments only. Pressure
// projection requires a separately validated global activity treatment.
ActivityMask build_activity_mask(const ActivityFields &fields,
                                 Extent3 grid_extent,
                                 int block_size = 8,
                                 const ActivityThresholds &thresholds = {});

ActivityMask dilate_activity_mask(const ActivityMask &mask, int radius = 1);

}  // namespace manta_improve
