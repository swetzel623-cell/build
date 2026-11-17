// SPDX-License-Identifier: MIT
#include "manta_improve/sparse_gas.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace manta_improve::sparse {
namespace {

int ceil_div(const int value, const int divisor)
{
  return (value + divisor - 1) / divisor;
}

bool is_wall(const Extent &extent, const int x, const int y, const int z)
{
  return x <= 0 || y <= 0 || z <= 0 || x >= extent.x - 1 || y >= extent.y - 1 ||
         z >= extent.z - 1;
}

std::size_t dense_index(const Extent &extent, const int x, const int y, const int z)
{
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(extent.x) *
             (static_cast<std::size_t>(y) +
              static_cast<std::size_t>(extent.y) * static_cast<std::size_t>(z));
}

/** Calls fn(index, x, y, z) for every stored voxel, with x, y, z in domain coordinates. */
template<typename Fn> void for_each_voxel(const Topology &topology, Fn &&fn)
{
  const int size = topology.brick_size;
  std::size_t index = 0;
  for (std::size_t slot = 0; slot < topology.brick_count(); ++slot) {
    const int base_x = topology.brick_coords[slot * 4 + 0] * size;
    const int base_y = topology.brick_coords[slot * 4 + 1] * size;
    const int base_z = topology.brick_coords[slot * 4 + 2] * size;
    for (int z = 0; z < size; ++z) {
      for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x, ++index) {
          fn(index, base_x + x, base_y + y, base_z + z);
        }
      }
    }
  }
}

float fetch(const Topology &topology, const std::vector<float> &channel, int x, int y, int z)
{
  const std::int64_t index = topology.index_of(x, y, z);
  return index < 0 ? 0.0f : channel[static_cast<std::size_t>(index)];
}

/** Pressure of a cell as seen by its neighbors: the unknown in fluid, zero in open air. */
float pressure_at(const Topology &topology, const std::vector<float> &pressure, int x, int y, int z)
{
  if (topology.type_at(x, y, z) != CellFluid) {
    return 0.0f;
  }
  return pressure[static_cast<std::size_t>(topology.index_of(x, y, z))];
}

constexpr int neighbor_offsets[6][3] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

/** `A x` for the pressure Poisson matrix; rows of non-fluid cells are zero. */
void apply_poisson(const Topology &topology, const std::vector<float> &x, std::vector<float> &result)
{
  for_each_voxel(topology, [&](const std::size_t index, const int gx, const int gy, const int gz) {
    if (topology.cell_type[index] != CellFluid) {
      result[index] = 0.0f;
      return;
    }
    float sum = 0.0f;
    int diagonal = 0;
    for (const auto &offset : neighbor_offsets) {
      const int nx = gx + offset[0], ny = gy + offset[1], nz = gz + offset[2];
      const CellType type = topology.type_at(nx, ny, nz);
      if (type == CellSolid) {
        continue;
      }
      ++diagonal;
      if (type == CellFluid) {
        sum += x[static_cast<std::size_t>(topology.index_of(nx, ny, nz))];
      }
    }
    result[index] = static_cast<float>(diagonal) * x[index] - sum;
  });
}

std::vector<float> poisson_diagonal(const Topology &topology)
{
  std::vector<float> diagonal(topology.voxel_count(), 0.0f);
  for_each_voxel(topology, [&](const std::size_t index, const int gx, const int gy, const int gz) {
    if (topology.cell_type[index] != CellFluid) {
      return;
    }
    int count = 0;
    for (const auto &offset : neighbor_offsets) {
      count += topology.type_at(gx + offset[0], gy + offset[1], gz + offset[2]) != CellSolid;
    }
    diagonal[index] = static_cast<float>(count);
  });
  return diagonal;
}

double dot(const std::vector<float> &a, const std::vector<float> &b)
{
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return sum;
}

float max_abs(const std::vector<float> &values)
{
  float result = 0.0f;
  for (const float value : values) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

int solve_conjugate_gradient(const Topology &topology,
                             const std::vector<float> &rhs,
                             const float accuracy,
                             const int max_iterations,
                             std::vector<float> &pressure)
{
  const std::size_t count = topology.voxel_count();
  const std::vector<float> diagonal = poisson_diagonal(topology);
  std::vector<float> residual(count), preconditioned(count), search(count), product(count);

  apply_poisson(topology, pressure, product);
  for (std::size_t i = 0; i < count; ++i) {
    residual[i] = diagonal[i] > 0.0f ? rhs[i] - product[i] : 0.0f;
    preconditioned[i] = diagonal[i] > 0.0f ? residual[i] / diagonal[i] : 0.0f;
  }
  search = preconditioned;
  double sigma = dot(residual, preconditioned);

  int iteration = 0;
  while (iteration < max_iterations && max_abs(residual) > accuracy) {
    apply_poisson(topology, search, product);
    const double curvature = dot(search, product);
    if (!(curvature > 0.0)) {
      break;
    }
    const float alpha = static_cast<float>(sigma / curvature);
    for (std::size_t i = 0; i < count; ++i) {
      pressure[i] += alpha * search[i];
      residual[i] -= alpha * product[i];
      preconditioned[i] = diagonal[i] > 0.0f ? residual[i] / diagonal[i] : 0.0f;
    }
    const double next_sigma = dot(residual, preconditioned);
    const float beta = static_cast<float>(next_sigma / sigma);
    for (std::size_t i = 0; i < count; ++i) {
      search[i] = preconditioned[i] + beta * search[i];
    }
    sigma = next_sigma;
    ++iteration;
  }
  return iteration;
}

std::vector<std::uint8_t> activity_flags(const State &state, const ActivityThresholds &thresholds)
{
  const Topology &topology = state.topology;
  std::vector<std::uint8_t> flags(topology.brick_table.size(), 0);
  const std::size_t per_brick = topology.voxels_per_brick();
  for (std::size_t slot = 0; slot < topology.brick_count(); ++slot) {
    bool active = false;
    for (std::size_t i = slot * per_brick; i < (slot + 1) * per_brick && !active; ++i) {
      active = std::abs(state.density[i]) > thresholds.scalar ||
               std::abs(state.temperature[i]) > thresholds.scalar ||
               std::abs(state.fuel[i]) > thresholds.scalar ||
               std::abs(state.flame[i]) > thresholds.scalar ||
               std::abs(state.u[i]) > thresholds.velocity ||
               std::abs(state.v[i]) > thresholds.velocity ||
               std::abs(state.w[i]) > thresholds.velocity;
    }
    if (active) {
      flags[static_cast<std::size_t>(topology.brick_coords[slot * 4 + 3])] = 1;
    }
  }
  return flags;
}

void resize_channels(State &state)
{
  const std::size_t count = state.topology.voxel_count();
  for (std::vector<float> *channel : {&state.density,
                                      &state.temperature,
                                      &state.fuel,
                                      &state.flame,
                                      &state.u,
                                      &state.v,
                                      &state.w,
                                      &state.pressure})
  {
    channel->assign(count, 0.0f);
  }
}

}  // namespace

std::int64_t Topology::index_of(const int x, const int y, const int z) const
{
  if (x < 0 || y < 0 || z < 0) {
    return -1;
  }
  const int bx = x / brick_size, by = y / brick_size, bz = z / brick_size;
  if (bx >= brick_extent.x || by >= brick_extent.y || bz >= brick_extent.z) {
    return -1;
  }
  const std::int32_t slot = brick_table[dense_index(brick_extent, bx, by, bz)];
  if (slot < 0) {
    return -1;
  }
  const int lx = x - bx * brick_size, ly = y - by * brick_size, lz = z - bz * brick_size;
  return static_cast<std::int64_t>(slot) * static_cast<std::int64_t>(voxels_per_brick()) +
         lx + brick_size * (ly + brick_size * lz);
}

CellType Topology::type_at(const int x, const int y, const int z) const
{
  if (is_wall(extent, x, y, z)) {
    return CellSolid;
  }
  const std::int64_t index = index_of(x, y, z);
  return index < 0 ? CellOpen : static_cast<CellType>(cell_type[static_cast<std::size_t>(index)]);
}

bool Topology::is_obstacle(const int x, const int y, const int z) const
{
  if (obstacle.empty()) {
    return false;
  }
  const std::size_t voxel = dense_index(extent, x, y, z);
  return (obstacle[voxel >> 5] >> (voxel & 31)) & 1u;
}

std::vector<std::uint32_t> pack_obstacle(const Extent &extent,
                                         const std::vector<std::uint8_t> &solid)
{
  const std::size_t count = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  if (solid.size() != count) {
    throw std::invalid_argument("The obstacle array does not match the extent");
  }
  std::vector<std::uint32_t> bits((count + 31) / 32, 0u);
  for (std::size_t i = 0; i < count; ++i) {
    if (solid[i]) {
      bits[i >> 5] |= 1u << (i & 31);
    }
  }
  return bits;
}

Topology build_topology(const Extent extent,
                        const int brick_size,
                        const std::vector<std::uint8_t> &brick_active,
                        const int dilation,
                        const bool classify_cells,
                        std::vector<std::uint32_t> obstacle)
{
  if (extent.x < 4 || extent.y < 4 || extent.z < 4 || brick_size < 2 || dilation < 0) {
    throw std::invalid_argument("Invalid sparse topology parameters");
  }
  const std::size_t voxels = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  if (!obstacle.empty() && obstacle.size() != (voxels + 31) / 32) {
    throw std::invalid_argument("The obstacle mask does not match the extent");
  }
  Topology topology;
  topology.extent = extent;
  topology.obstacle = std::move(obstacle);
  topology.brick_size = brick_size;
  topology.brick_extent = {ceil_div(extent.x, brick_size),
                           ceil_div(extent.y, brick_size),
                           ceil_div(extent.z, brick_size)};
  const Extent &bricks = topology.brick_extent;
  const std::size_t brick_total = static_cast<std::size_t>(bricks.x) * bricks.y * bricks.z;
  if (brick_active.size() != brick_total) {
    throw std::invalid_argument("Brick activity flags do not match the brick grid");
  }

  std::vector<std::uint8_t> active(brick_total, 0);
  for (int z = 0; z < bricks.z; ++z) {
    for (int y = 0; y < bricks.y; ++y) {
      for (int x = 0; x < bricks.x; ++x) {
        if (!brick_active[dense_index(bricks, x, y, z)]) {
          continue;
        }
        for (int nz = std::max(z - dilation, 0); nz <= std::min(z + dilation, bricks.z - 1); ++nz)
        {
          for (int ny = std::max(y - dilation, 0); ny <= std::min(y + dilation, bricks.y - 1);
               ++ny)
          {
            for (int nx = std::max(x - dilation, 0); nx <= std::min(x + dilation, bricks.x - 1);
                 ++nx)
            {
              active[dense_index(bricks, nx, ny, nz)] = 1;
            }
          }
        }
      }
    }
  }

  topology.brick_table.assign(brick_total, -1);
  for (int z = 0; z < bricks.z; ++z) {
    for (int y = 0; y < bricks.y; ++y) {
      for (int x = 0; x < bricks.x; ++x) {
        const std::size_t flat = dense_index(bricks, x, y, z);
        if (active[flat]) {
          topology.brick_table[flat] = static_cast<std::int32_t>(topology.brick_count());
          topology.brick_coords.insert(topology.brick_coords.end(),
                                       {x, y, z, static_cast<std::int32_t>(flat)});
        }
      }
    }
  }

  /* A voxel stores the velocity of its lower faces only. A cell can therefore be solved for only
   * if the cells above it along every axis are stored too, or are walls, whose faces are zero.
   * The same is required of the cells below it, although their storage is not needed: the open
   * boundary then lies one voxel inside the stored region on every side. Without that the
   * region would be one voxel larger on its lower sides and a symmetric scene would lose its
   * symmetry. The remaining stored cells form a one-voxel layer of open air. */
  if (!classify_cells) {
    return topology;
  }
  topology.cell_type.assign(topology.voxel_count(), CellOpen);
  const auto stored_or_wall = [&](const int x, const int y, const int z) {
    return is_wall(extent, x, y, z) || topology.index_of(x, y, z) >= 0;
  };
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (is_wall(extent, x, y, z) || topology.is_obstacle(x, y, z)) {
      topology.cell_type[index] = CellSolid;
    }
    else if (stored_or_wall(x + 1, y, z) && stored_or_wall(x, y + 1, z) &&
             stored_or_wall(x, y, z + 1) && stored_or_wall(x - 1, y, z) &&
             stored_or_wall(x, y - 1, z) && stored_or_wall(x, y, z - 1))
    {
      topology.cell_type[index] = CellFluid;
    }
  });
  return topology;
}

State from_dense(const DenseFields &dense,
                 const int brick_size,
                 const ActivityThresholds &thresholds,
                 const int dilation)
{
  const Extent &extent = dense.extent;
  const std::size_t count = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  if (dense.density.size() != count || dense.temperature.size() != count) {
    throw std::invalid_argument("Dense scalar fields do not match the extent");
  }
  const bool has_velocity = !dense.u.empty();
  if (has_velocity && (dense.u.size() != count || dense.v.size() != count || dense.w.size() != count))
  {
    throw std::invalid_argument("Dense velocity fields do not match the extent");
  }
  const bool has_pressure = !dense.pressure.empty();
  if (has_pressure && dense.pressure.size() != count) {
    throw std::invalid_argument("Dense pressure does not match the extent");
  }
  const bool has_fire = !dense.fuel.empty();
  if (has_fire && (dense.fuel.size() != count || dense.flame.size() != count)) {
    throw std::invalid_argument("Dense fuel and flame do not match the extent");
  }

  const Extent bricks = {ceil_div(extent.x, brick_size),
                         ceil_div(extent.y, brick_size),
                         ceil_div(extent.z, brick_size)};
  std::vector<std::uint8_t> flags(static_cast<std::size_t>(bricks.x) * bricks.y * bricks.z, 0);
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const std::size_t i = dense_index(extent, x, y, z);
        const bool active = std::abs(dense.density[i]) > thresholds.scalar ||
                            std::abs(dense.temperature[i]) > thresholds.scalar ||
                            (has_fire && (std::abs(dense.fuel[i]) > thresholds.scalar ||
                                          std::abs(dense.flame[i]) > thresholds.scalar)) ||
                            (has_velocity && (std::abs(dense.u[i]) > thresholds.velocity ||
                                              std::abs(dense.v[i]) > thresholds.velocity ||
                                              std::abs(dense.w[i]) > thresholds.velocity));
        if (active) {
          flags[dense_index(bricks, x / brick_size, y / brick_size, z / brick_size)] = 1;
        }
      }
    }
  }

  State state;
  state.topology = build_topology(
      extent, brick_size, flags, dilation, true,
      dense.obstacle.empty() ? std::vector<std::uint32_t>() :
                               pack_obstacle(extent, dense.obstacle));
  resize_channels(state);
  for_each_voxel(state.topology, [&](const std::size_t index, int x, int y, int z) {
    if (x >= extent.x || y >= extent.y || z >= extent.z) {
      return;
    }
    const std::size_t i = dense_index(extent, x, y, z);
    state.density[index] = dense.density[i];
    state.temperature[index] = dense.temperature[i];
    if (has_fire) {
      state.fuel[index] = dense.fuel[i];
      state.flame[index] = dense.flame[i];
    }
    if (has_velocity) {
      state.u[index] = dense.u[i];
      state.v[index] = dense.v[i];
      state.w[index] = dense.w[i];
    }
    if (has_pressure) {
      state.pressure[index] = dense.pressure[i];
    }
  });
  if (!dense.obstacle_u.empty()) {
    state.obstacle_velocity = brick_field_from_dense(
        extent, brick_size, {&dense.obstacle_u, &dense.obstacle_v, &dense.obstacle_w});
  }
  return state;
}

DenseFields to_dense(const State &state)
{
  const Extent &extent = state.topology.extent;
  const std::size_t count = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  DenseFields dense;
  dense.extent = extent;
  for (std::vector<float> *channel : {&dense.density,
                                      &dense.temperature,
                                      &dense.fuel,
                                      &dense.flame,
                                      &dense.u,
                                      &dense.v,
                                      &dense.w,
                                      &dense.pressure})
  {
    channel->assign(count, 0.0f);
  }
  for_each_voxel(state.topology, [&](const std::size_t index, int x, int y, int z) {
    if (x >= extent.x || y >= extent.y || z >= extent.z) {
      return;
    }
    const std::size_t i = dense_index(extent, x, y, z);
    dense.density[i] = state.density[index];
    dense.temperature[i] = state.temperature[index];
    dense.fuel[i] = state.fuel[index];
    dense.flame[i] = state.flame[index];
    dense.u[i] = state.u[index];
    dense.v[i] = state.v[index];
    dense.w[i] = state.w[index];
    dense.pressure[i] = state.pressure[index];
  });
  return dense;
}

static void mark_source_bricks(const Sources *sources, std::vector<std::uint8_t> &flags)
{
  if (!sources || sources->field.empty()) {
    return;
  }
  if (sources->field.table.size() != flags.size()) {
    throw std::invalid_argument("The sources do not match the state's brick grid");
  }
  for (std::size_t brick = 0; brick < flags.size(); ++brick) {
    if (sources->field.table[brick] >= 0) {
      flags[brick] = 1;
    }
  }
}

void retopologize(State &state,
                  const ActivityThresholds &thresholds,
                  const int dilation,
                  const Sources *sources)
{
  const Topology &old_topology = state.topology;
  std::vector<std::uint8_t> flags = activity_flags(state, thresholds);
  mark_source_bricks(sources, flags);
  Topology topology = build_topology(old_topology.extent,
                                     old_topology.brick_size,
                                     flags,
                                     dilation,
                                     true,
                                     old_topology.obstacle);
  if (topology.brick_table == old_topology.brick_table) {
    return;
  }

  State moved;
  moved.topology = std::move(topology);
  resize_channels(moved);
  const std::size_t per_brick = moved.topology.voxels_per_brick();
  for (std::size_t slot = 0; slot < moved.topology.brick_count(); ++slot) {
    const std::int32_t old_slot = old_topology.brick_table[static_cast<std::size_t>(
        moved.topology.brick_coords[slot * 4 + 3])];
    if (old_slot < 0) {
      continue;
    }
    const auto copy = [&](const std::vector<float> &from, std::vector<float> &to) {
      std::copy_n(from.begin() + static_cast<std::ptrdiff_t>(old_slot * per_brick),
                  per_brick,
                  to.begin() + static_cast<std::ptrdiff_t>(slot * per_brick));
    };
    copy(state.density, moved.density);
    copy(state.temperature, moved.temperature);
    copy(state.fuel, moved.fuel);
    copy(state.flame, moved.flame);
    copy(state.u, moved.u);
    copy(state.v, moved.v);
    copy(state.w, moved.w);
    copy(state.pressure, moved.pressure);
  }
  moved.obstacle_velocity = std::move(state.obstacle_velocity);
  state = std::move(moved);
}

void set_obstacles(State &state, std::vector<std::uint32_t> mask, BrickField velocity)
{
  const Topology &old_topology = state.topology;
  std::vector<std::uint8_t> flags(old_topology.brick_table.size(), 0);
  for (std::size_t slot = 0; slot < old_topology.brick_count(); ++slot) {
    flags[static_cast<std::size_t>(old_topology.brick_coords[slot * 4 + 3])] = 1;
  }
  Topology topology = build_topology(
      old_topology.extent, old_topology.brick_size, flags, 0, true, std::move(mask));
  if (topology.brick_table != old_topology.brick_table) {
    throw std::logic_error("Reclassifying the obstacles changed the bricks");
  }
  if (!velocity.empty() && velocity.table.size() != topology.brick_table.size()) {
    throw std::invalid_argument("The obstacle velocity does not match the state's brick grid");
  }
  state.topology = std::move(topology);
  state.obstacle_velocity = std::move(velocity);
}

void set_obstacles_from_dense(State &state, const DenseFields &dense)
{
  const Extent &extent = state.topology.extent;
  std::vector<std::uint32_t> mask;
  if (!dense.obstacle.empty()) {
    mask = pack_obstacle(extent, dense.obstacle);
  }
  BrickField velocity;
  if (!dense.obstacle_u.empty()) {
    velocity = brick_field_from_dense(extent,
                                      state.topology.brick_size,
                                      {&dense.obstacle_u, &dense.obstacle_v, &dense.obstacle_w});
  }
  set_obstacles(state, std::move(mask), std::move(velocity));
}

std::size_t BrickField::slot_count() const
{
  return channels.empty() ? 0 :
                            channels.front().size() /
                                (static_cast<std::size_t>(brick_size) * brick_size * brick_size);
}

float BrickField::at(const std::size_t channel, const int x, const int y, const int z) const
{
  if (x < 0 || y < 0 || z < 0 || channel >= channels.size()) {
    return 0.0f;
  }
  const int bx = x / brick_size, by = y / brick_size, bz = z / brick_size;
  if (bx >= brick_extent.x || by >= brick_extent.y || bz >= brick_extent.z) {
    return 0.0f;
  }
  const std::int32_t slot = table[static_cast<std::size_t>(bx) +
                                  static_cast<std::size_t>(brick_extent.x) *
                                      (by + static_cast<std::size_t>(brick_extent.y) * bz)];
  if (slot < 0) {
    return 0.0f;
  }
  const int lx = x - bx * brick_size, ly = y - by * brick_size, lz = z - bz * brick_size;
  const std::size_t per_brick = static_cast<std::size_t>(brick_size) * brick_size * brick_size;
  return channels[channel][static_cast<std::size_t>(slot) * per_brick + lx +
                           static_cast<std::size_t>(brick_size) * (ly + brick_size * lz)];
}

float &BrickField::touch(const std::size_t channel, const int x, const int y, const int z)
{
  const int bx = x / brick_size, by = y / brick_size, bz = z / brick_size;
  if (x < 0 || y < 0 || z < 0 || bx >= brick_extent.x || by >= brick_extent.y ||
      bz >= brick_extent.z || channel >= channels.size())
  {
    throw std::out_of_range("BrickField::touch outside the domain");
  }
  const std::size_t brick = static_cast<std::size_t>(bx) +
                            static_cast<std::size_t>(brick_extent.x) *
                                (by + static_cast<std::size_t>(brick_extent.y) * bz);
  const std::size_t per_brick = static_cast<std::size_t>(brick_size) * brick_size * brick_size;
  if (table[brick] < 0) {
    table[brick] = static_cast<std::int32_t>(slot_count());
    for (std::vector<float> &values : channels) {
      values.resize(values.size() + per_brick, 0.0f);
    }
  }
  const int lx = x - bx * brick_size, ly = y - by * brick_size, lz = z - bz * brick_size;
  return channels[channel][static_cast<std::size_t>(table[brick]) * per_brick + lx +
                           static_cast<std::size_t>(brick_size) * (ly + brick_size * lz)];
}

BrickField make_brick_field(const Extent &extent, const int brick_size, const std::size_t count)
{
  BrickField field;
  field.brick_size = brick_size;
  field.brick_extent = {ceil_div(extent.x, brick_size),
                        ceil_div(extent.y, brick_size),
                        ceil_div(extent.z, brick_size)};
  field.table.assign(static_cast<std::size_t>(field.brick_extent.x) * field.brick_extent.y *
                         field.brick_extent.z,
                     -1);
  field.channels.assign(count, {});
  return field;
}

State make_state(Topology topology)
{
  State state;
  state.topology = std::move(topology);
  resize_channels(state);
  return state;
}

BrickField brick_field_from_dense(const Extent &extent,
                                  const int brick_size,
                                  const std::vector<const std::vector<float> *> &channels)
{
  const std::size_t count = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  for (const std::vector<float> *channel : channels) {
    if (channel->size() != count) {
      throw std::invalid_argument("A dense field does not match the extent");
    }
  }
  BrickField field;
  field.brick_size = brick_size;
  field.brick_extent = {ceil_div(extent.x, brick_size),
                        ceil_div(extent.y, brick_size),
                        ceil_div(extent.z, brick_size)};
  const Extent &bricks = field.brick_extent;
  field.table.assign(static_cast<std::size_t>(bricks.x) * bricks.y * bricks.z, -1);
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const std::size_t i = dense_index(extent, x, y, z);
        for (const std::vector<float> *channel : channels) {
          if ((*channel)[i] != 0.0f) {
            field.table[dense_index(bricks, x / brick_size, y / brick_size, z / brick_size)] = 0;
            break;
          }
        }
      }
    }
  }
  std::int32_t slots = 0;
  for (std::int32_t &slot : field.table) {
    if (slot >= 0) {
      slot = slots++;
    }
  }
  const std::size_t per_brick = static_cast<std::size_t>(brick_size) * brick_size * brick_size;
  field.channels.assign(channels.size(), std::vector<float>(per_brick * slots, 0.0f));
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const std::int32_t slot =
            field.table[dense_index(bricks, x / brick_size, y / brick_size, z / brick_size)];
        if (slot < 0) {
          continue;
        }
        const std::size_t local = static_cast<std::size_t>(x % brick_size) +
                                  static_cast<std::size_t>(brick_size) *
                                      (y % brick_size + static_cast<std::size_t>(brick_size) *
                                                            (z % brick_size));
        for (std::size_t c = 0; c < channels.size(); ++c) {
          field.channels[c][static_cast<std::size_t>(slot) * per_brick + local] =
              (*channels[c])[dense_index(extent, x, y, z)];
        }
      }
    }
  }
  return field;
}

Sources sources_from_dense(const DenseFields &dense, const int brick_size, const SourceMode mode)
{
  Sources sources;
  sources.mode = mode;
  const std::vector<float> no_fuel(dense.fuel.empty() ? dense.density.size() : 0, 0.0f);
  std::vector<const std::vector<float> *> channels = {
      &dense.density, &dense.temperature, dense.fuel.empty() ? &no_fuel : &dense.fuel};
  if (!dense.u.empty()) {
    channels.insert(channels.end(), {&dense.u, &dense.v, &dense.w});
  }
  sources.field = brick_field_from_dense(dense.extent, brick_size, channels);
  return sources;
}

float sample(const Topology &topology,
             const std::vector<float> &channel,
             const float x,
             const float y,
             const float z,
             const float offset_x,
             const float offset_y,
             const float offset_z)
{
  const float fx = x - offset_x, fy = y - offset_y, fz = z - offset_z;
  const float bx = std::floor(fx), by = std::floor(fy), bz = std::floor(fz);
  const int ix = static_cast<int>(bx), iy = static_cast<int>(by), iz = static_cast<int>(bz);
  const float tx = fx - bx, ty = fy - by, tz = fz - bz;

  const auto lerp = [](const float a, const float b, const float t) { return a + (b - a) * t; };
  const float c00 = lerp(fetch(topology, channel, ix, iy, iz),
                         fetch(topology, channel, ix + 1, iy, iz), tx);
  const float c10 = lerp(fetch(topology, channel, ix, iy + 1, iz),
                         fetch(topology, channel, ix + 1, iy + 1, iz), tx);
  const float c01 = lerp(fetch(topology, channel, ix, iy, iz + 1),
                         fetch(topology, channel, ix + 1, iy, iz + 1), tx);
  const float c11 = lerp(fetch(topology, channel, ix, iy + 1, iz + 1),
                         fetch(topology, channel, ix + 1, iy + 1, iz + 1), tx);
  return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
}

namespace {

/**
 * Follow the velocity field from a point for `step` seconds (negative: backwards). With
 * `midpoint`, the velocity is taken half way along a first, single-stage trace.
 */
void trace(const State &state, const float px, const float py, const float pz, const float step,
           const bool midpoint, float end[3])
{
  const Topology &topology = state.topology;
  float vx = sample(topology, state.u, px, py, pz, 0.0f, 0.5f, 0.5f);
  float vy = sample(topology, state.v, px, py, pz, 0.5f, 0.0f, 0.5f);
  float vz = sample(topology, state.w, px, py, pz, 0.5f, 0.5f, 0.0f);
  if (midpoint) {
    const float half = 0.5f * step;
    const float mx = px + half * vx, my = py + half * vy, mz = pz + half * vz;
    vx = sample(topology, state.u, mx, my, mz, 0.0f, 0.5f, 0.5f);
    vy = sample(topology, state.v, mx, my, mz, 0.5f, 0.0f, 0.5f);
    vz = sample(topology, state.w, mx, my, mz, 0.5f, 0.5f, 0.0f);
  }
  end[0] = px + step * vx;
  end[1] = py + step * vy;
  end[2] = pz + step * vz;
}

}  // namespace

void advect(const State &state,
            const std::vector<float> &source,
            const float offset_x,
            const float offset_y,
            const float offset_z,
            const float dt,
            std::vector<float> &result,
            const bool midpoint)
{
  const Topology &topology = state.topology;
  result.assign(topology.voxel_count(), 0.0f);
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (topology.cell_type[index] == CellSolid) {
      return;
    }
    const float px = static_cast<float>(x) + offset_x;
    const float py = static_cast<float>(y) + offset_y;
    const float pz = static_cast<float>(z) + offset_z;
    float from[3];
    trace(state, px, py, pz, -dt, midpoint, from);
    result[index] = sample(topology, source, from[0], from[1], from[2], offset_x, offset_y,
                           offset_z);
  });
}

void advect_maccormack(const State &state,
                       const std::vector<float> &source,
                       const std::vector<float> &forward,
                       const float offset_x,
                       const float offset_y,
                       const float offset_z,
                       const float dt,
                       std::vector<float> &result,
                       const bool midpoint)
{
  const Topology &topology = state.topology;
  result.assign(topology.voxel_count(), 0.0f);
  /* A staggered component sits on the face shared with the cell below it along its axis. */
  const int axis = offset_x == 0.0f ? 0 : offset_y == 0.0f ? 1 : offset_z == 0.0f ? 2 : -1;
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    const CellType type = static_cast<CellType>(topology.cell_type[index]);
    if (type == CellSolid) {
      return;
    }
    result[index] = forward[index];
    /* Only correct values that lie inside the fluid, as Mantaflow does. */
    if (type != CellFluid ||
        (axis >= 0 &&
         topology.type_at(x - (axis == 0), y - (axis == 1), z - (axis == 2)) != CellFluid))
    {
      return;
    }
    const float px = static_cast<float>(x) + offset_x;
    const float py = static_cast<float>(y) + offset_y;
    const float pz = static_cast<float>(z) + offset_z;
    float from[3], to[3];
    trace(state, px, py, pz, -dt, midpoint, from);
    trace(state, px, py, pz, dt, midpoint, to);

    /* The values the forward trace interpolated bound the result. If any of them has no
     * storage, the trace left the active region: keep the first-order value there. The same
     * holds next to a solid, as in Mantaflow: solid cells hold zero, which would make the
     * correction add smoke along every wall. */
    const float bx = std::floor(from[0] - offset_x);
    const float by = std::floor(from[1] - offset_y);
    const float bz = std::floor(from[2] - offset_z);
    const int ix = static_cast<int>(bx), iy = static_cast<int>(by), iz = static_cast<int>(bz);
    float lowest = 0.0f, highest = 0.0f;
    for (int corner = 0; corner < 8; ++corner) {
      const std::int64_t i = topology.index_of(
          ix + (corner & 1), iy + ((corner >> 1) & 1), iz + ((corner >> 2) & 1));
      if (i < 0 || topology.cell_type[static_cast<std::size_t>(i)] == CellSolid) {
        return;
      }
      const float value = source[static_cast<std::size_t>(i)];
      lowest = corner == 0 ? value : std::min(lowest, value);
      highest = corner == 0 ? value : std::max(highest, value);
    }

    /* The backward trace must stay clear of solids and of unstored voxels too. */
    const int tx = static_cast<int>(std::floor(to[0] - offset_x));
    const int ty = static_cast<int>(std::floor(to[1] - offset_y));
    const int tz = static_cast<int>(std::floor(to[2] - offset_z));
    for (int corner = 0; corner < 8; ++corner) {
      const std::int64_t i = topology.index_of(
          tx + (corner & 1), ty + ((corner >> 1) & 1), tz + ((corner >> 2) & 1));
      if (i < 0 || topology.cell_type[static_cast<std::size_t>(i)] == CellSolid) {
        return;
      }
    }
    const float backward = sample(
        topology, forward, to[0], to[1], to[2], offset_x, offset_y, offset_z);
    const float corrected = forward[index] + 0.5f * (source[index] - backward);
    result[index] = std::min(std::max(corrected, lowest), highest);
  });
}

void conserve_correction(const std::vector<float> &source,
                         const std::vector<float> &forward,
                         std::vector<float> &result)
{
  double before = 0.0, after = 0.0, added = 0.0, removed = 0.0;
  for (std::size_t i = 0; i < source.size(); ++i) {
    before += source[i];
    after += result[i];
    const double correction = static_cast<double>(result[i]) - forward[i];
    (correction > 0.0 ? added : removed) += std::abs(correction);
  }
  const double excess = after - before;
  const double pool = excess > 0.0 ? added : removed;
  if (excess == 0.0 || pool == 0.0) {
    return;
  }
  const float keep = static_cast<float>(std::max(0.0, 1.0 - std::abs(excess) / pool));
  for (std::size_t i = 0; i < source.size(); ++i) {
    const float correction = result[i] - forward[i];
    if (excess > 0.0 ? correction > 0.0f : correction < 0.0f) {
      result[i] = forward[i] + keep * correction;
    }
  }
}

float max_speed(const State &state)
{
  float result = 0.0f;
  for (const std::vector<float> *channel : {&state.u, &state.v, &state.w}) {
    for (const float value : *channel) {
      result = std::max(result, std::abs(value));
    }
  }
  return result;
}

float cfl_time_step(const float remaining, const float cfl, const float speed)
{
  if (!(speed > 0.0f) || cfl / speed >= remaining) {
    return remaining;
  }
  return cfl / speed;
}

void add_buoyancy(State &state,
                  const std::vector<float> &factor,
                  const float coefficient,
                  const float dt,
                  const float gravity[3])
{
  const Topology &topology = state.topology;
  const float strength[3] = {-gravity[0] * dt * coefficient,
                             -gravity[1] * dt * coefficient,
                             -gravity[2] * dt * coefficient};
  std::vector<float> *faces[3] = {&state.u, &state.v, &state.w};
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (topology.cell_type[index] != CellFluid) {
      return;
    }
    for (int axis = 0; axis < 3; ++axis) {
      const int nx = x - (axis == 0), ny = y - (axis == 1), nz = z - (axis == 2);
      if (topology.type_at(nx, ny, nz) == CellFluid) {
        (*faces[axis])[index] += 0.5f * strength[axis] *
                                 (factor[index] + fetch(topology, factor, nx, ny, nz));
      }
    }
  });
}

namespace {

/* Gradient noise for the turbulence force. noise.glsl repeats every operation in this order,
 * so the CPU and the device agree to rounding. */

std::uint32_t mix_hash(std::uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

std::uint32_t corner_hash(const int x, const int y, const int z, const std::uint32_t seed)
{
  return mix_hash((static_cast<std::uint32_t>(x) * 0x8da6b343u) ^
                  mix_hash((static_cast<std::uint32_t>(y) * 0xd8163841u) ^
                           mix_hash((static_cast<std::uint32_t>(z) * 0xcb1ab31fu) ^ seed)));
}

/** Dot product of one of the 12 cube-edge directions with (x, y, z). */
float gradient_dot(const std::uint32_t hash, const float x, const float y, const float z)
{
  switch (hash % 12u) {
    case 0:
      return x + y;
    case 1:
      return -x + y;
    case 2:
      return x - y;
    case 3:
      return -x - y;
    case 4:
      return x + z;
    case 5:
      return -x + z;
    case 6:
      return x - z;
    case 7:
      return -x - z;
    case 8:
      return y + z;
    case 9:
      return -y + z;
    case 10:
      return y - z;
    default:
      return -y - z;
  }
}

float fade(const float t)
{
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float mixf(const float a, const float b, const float t)
{
  return a + (b - a) * t;
}

float gradient_noise(const float px, const float py, const float pz, const std::uint32_t seed)
{
  const float fx = std::floor(px), fy = std::floor(py), fz = std::floor(pz);
  const int ix = static_cast<int>(fx), iy = static_cast<int>(fy), iz = static_cast<int>(fz);
  const float tx = px - fx, ty = py - fy, tz = pz - fz;
  const auto corner = [&](const int dx, const int dy, const int dz) {
    return gradient_dot(corner_hash(ix + dx, iy + dy, iz + dz, seed),
                        tx - static_cast<float>(dx),
                        ty - static_cast<float>(dy),
                        tz - static_cast<float>(dz));
  };
  const float ux = fade(tx), uy = fade(ty), uz = fade(tz);
  const float x00 = mixf(corner(0, 0, 0), corner(1, 0, 0), ux);
  const float x10 = mixf(corner(0, 1, 0), corner(1, 1, 0), ux);
  const float x01 = mixf(corner(0, 0, 1), corner(1, 0, 1), ux);
  const float x11 = mixf(corner(0, 1, 1), corner(1, 1, 1), ux);
  return mixf(mixf(x00, x10, uy), mixf(x01, x11, uy), uz);
}

/** Where the turbulence pattern is in time: two slices and the blend between them. */
struct NoiseTime {
  int slice = 0;
  float blend = 0.0f;
};

NoiseTime noise_time(const Turbulence &turbulence, const float time)
{
  const float tau = time * turbulence.speed;
  const float base = std::floor(tau);
  return {static_cast<int>(base), fade(tau - base)};
}

/** Component `k` of the vector potential at noise coordinates q. */
float potential(const std::uint32_t seed,
                const NoiseTime &when,
                const int k,
                const float qx,
                const float qy,
                const float qz)
{
  const std::uint32_t channel = mix_hash(seed ^ (static_cast<std::uint32_t>(k) * 0x9e3779b9u));
  const float a = gradient_noise(
      qx, qy, qz, channel ^ mix_hash(static_cast<std::uint32_t>(when.slice)));
  const float b = gradient_noise(
      qx, qy, qz, channel ^ mix_hash(static_cast<std::uint32_t>(when.slice + 1)));
  return mixf(a, b, when.blend);
}

/** Component `axis` of the curl of the potential at noise coordinates q. */
float curl(const std::uint32_t seed,
           const NoiseTime &when,
           const int axis,
           const float qx,
           const float qy,
           const float qz)
{
  constexpr float h = 0.125f;
  /* curl_x = dPz/dy - dPy/dz, and cyclically. */
  const int b = (axis + 1) % 3, c = (axis + 2) % 3;
  const float q[3] = {qx, qy, qz};
  const auto at = [&](const int k, const int along, const float offset) {
    float p[3] = {q[0], q[1], q[2]};
    p[along] += offset;
    return potential(seed, when, k, p[0], p[1], p[2]);
  };
  const float d_c_along_b = at(c, b, h) - at(c, b, -h);
  const float d_b_along_c = at(b, c, h) - at(b, c, -h);
  return (d_c_along_b - d_b_along_c) / (2.0f * h);
}

}  // namespace

void cell_speed(const State &state, std::vector<float> &result)
{
  const Topology &topology = state.topology;
  result.assign(topology.voxel_count(), 0.0f);
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    /* The same arithmetic as speed.comp: the mean of each component's two faces. */
    const float u = 0.5f * (state.u[index] + fetch(topology, state.u, x + 1, y, z));
    const float v = 0.5f * (state.v[index] + fetch(topology, state.v, x, y + 1, z));
    const float w = 0.5f * (state.w[index] + fetch(topology, state.w, x, y, z + 1));
    result[index] = std::sqrt(u * u + v * v + w * w);
  });
}

void combust(State &state, const Fire &fire, const float dt)
{
  if (!fire.enabled) {
    return;
  }
  const Topology &topology = state.topology;
  const float capacity = fire.burn_rate * dt;
  const float fade = fire.flame_cooling * dt;
  for (std::size_t i = 0; i < topology.voxel_count(); ++i) {
    if (topology.cell_type[i] == CellSolid) {
      continue;
    }
    /* The same arithmetic as combust.comp. */
    float flame = std::max(state.flame[i] - fade, 0.0f);
    const float fuel = state.fuel[i];
    if (fuel > 0.0f && capacity > 0.0f) {
      const float burnt = std::min(fuel, capacity);
      state.fuel[i] = fuel - burnt;
      state.temperature[i] += fire.heat * burnt;
      state.density[i] += fire.smoke * burnt;
      flame = std::max(flame, std::min(fuel, 1.0f));
    }
    state.flame[i] = flame;
  }
}

void turbulence_time(const Turbulence &turbulence, const float time, int &slice, float &blend)
{
  const NoiseTime when = noise_time(turbulence, time);
  slice = when.slice;
  blend = when.blend;
}

void add_turbulence(State &state, const Turbulence &turbulence, const float time, const float dt)
{
  if (turbulence.strength == 0.0f) {
    return;
  }
  const Topology &topology = state.topology;
  const float push = turbulence.strength * dt;
  const float inverse_scale = 1.0f / turbulence.scale;
  const NoiseTime when = noise_time(turbulence, time);
  std::vector<float> speed;
  const std::vector<float> *mask = nullptr;
  switch (turbulence.mask) {
    case TurbulenceMask::None:
      break;
    case TurbulenceMask::Density:
      mask = &state.density;
      break;
    case TurbulenceMask::Temperature:
      mask = &state.temperature;
      break;
    case TurbulenceMask::Flame:
      mask = &state.flame;
      break;
    case TurbulenceMask::Speed:
      cell_speed(state, speed);
      mask = &speed;
      break;
  }
  std::vector<float> *faces[3] = {&state.u, &state.v, &state.w};
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (topology.cell_type[index] != CellFluid) {
      return;
    }
    for (int axis = 0; axis < 3; ++axis) {
      const int nx = x - (axis == 0), ny = y - (axis == 1), nz = z - (axis == 2);
      if (topology.type_at(nx, ny, nz) != CellFluid) {
        continue;
      }
      float weight = 1.0f;
      if (mask) {
        const float value = 0.5f * ((*mask)[index] + fetch(topology, *mask, nx, ny, nz));
        weight = std::min(std::max(value / turbulence.mask_level, 0.0f), 1.0f);
        if (turbulence.invert) {
          weight = 1.0f - weight;
        }
      }
      /* The face's centre: on the lower face of the voxel along `axis`. */
      const float px = static_cast<float>(x) + (axis == 0 ? 0.0f : 0.5f);
      const float py = static_cast<float>(y) + (axis == 1 ? 0.0f : 0.5f);
      const float pz = static_cast<float>(z) + (axis == 2 ? 0.0f : 0.5f);
      const float force = curl(turbulence.seed,
                               when,
                               axis,
                               px * inverse_scale,
                               py * inverse_scale,
                               pz * inverse_scale);
      (*faces[axis])[index] += push * force * weight;
    }
  });
}

void enforce_walls(State &state)
{
  const Topology &topology = state.topology;
  const BrickField &motion = state.obstacle_velocity;
  /* Velocity component `axis` of a solid cell: the domain's outer wall is at rest. */
  const auto solid_velocity = [&](const int axis, const int x, const int y, const int z) {
    if (motion.empty() || is_wall(topology.extent, x, y, z)) {
      return 0.0f;
    }
    return motion.at(static_cast<std::size_t>(axis), x, y, z);
  };
  std::vector<float> *faces[3] = {&state.u, &state.v, &state.w};
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    const bool here = topology.cell_type[index] == CellSolid;
    for (int axis = 0; axis < 3; ++axis) {
      const int nx = x - (axis == 0), ny = y - (axis == 1), nz = z - (axis == 2);
      const bool below = topology.type_at(nx, ny, nz) == CellSolid;
      if (!here && !below) {
        continue;
      }
      /* The same arithmetic as walls.comp. */
      float sum = 0.0f;
      float sides = 0.0f;
      if (here) {
        sum += solid_velocity(axis, x, y, z);
        sides += 1.0f;
      }
      if (below) {
        sum += solid_velocity(axis, nx, ny, nz);
        sides += 1.0f;
      }
      (*faces[axis])[index] = sum / sides;
    }
  });
}

void inject(State &state, const Sources *sources, const float dt)
{
  const Topology &topology = state.topology;
  const bool emit = sources && !sources->field.empty();
  if (emit && sources->field.table.size() != topology.brick_table.size()) {
    throw std::invalid_argument("The sources do not match the state's brick grid");
  }
  if (emit && sources->field.channels.size() != 3 && sources->field.channels.size() != 6) {
    throw std::invalid_argument(
        "Sources have density, temperature and fuel, and optionally velocity");
  }
  const bool with_velocity = emit && sources->field.channels.size() == 6;
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (topology.cell_type[index] == CellSolid) {
      state.density[index] = 0.0f;
      state.temperature[index] = 0.0f;
      state.fuel[index] = 0.0f;
      state.flame[index] = 0.0f;
      return;
    }
    if (!emit) {
      return;
    }
    const float density = sources->field.at(0, x, y, z);
    const float temperature = sources->field.at(1, x, y, z);
    const float fuel = sources->field.at(2, x, y, z);
    if (sources->mode == SourceMode::Raise) {
      if (density != 0.0f) {
        state.density[index] = std::max(state.density[index], density);
      }
      if (temperature != 0.0f) {
        state.temperature[index] = std::max(state.temperature[index], temperature);
      }
      if (fuel != 0.0f) {
        state.fuel[index] = std::max(state.fuel[index], fuel);
      }
    }
    else {
      state.density[index] += density * dt;
      state.temperature[index] += temperature * dt;
      state.fuel[index] += fuel * dt;
    }
    if (with_velocity && (density != 0.0f || temperature != 0.0f || fuel != 0.0f)) {
      state.u[index] = sources->field.at(3, x, y, z);
      state.v[index] = sources->field.at(4, x, y, z);
      state.w[index] = sources->field.at(5, x, y, z);
    }
  });
}

void divergence(const State &state, std::vector<float> &result)
{
  const Topology &topology = state.topology;
  result.assign(topology.voxel_count(), 0.0f);
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (topology.cell_type[index] != CellFluid) {
      return;
    }
    result[index] = fetch(topology, state.u, x + 1, y, z) - state.u[index] +
                    fetch(topology, state.v, x, y + 1, z) - state.v[index] +
                    fetch(topology, state.w, x, y, z + 1) - state.w[index];
  });
}

void jacobi_sweep(const Topology &topology,
                  const std::vector<float> &negative_divergence,
                  const std::vector<float> &pressure,
                  std::vector<float> &result)
{
  result.assign(topology.voxel_count(), 0.0f);
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (topology.cell_type[index] != CellFluid) {
      return;
    }
    float sum = 0.0f;
    int diagonal = 0;
    for (const auto &offset : neighbor_offsets) {
      const int nx = x + offset[0], ny = y + offset[1], nz = z + offset[2];
      const CellType type = topology.type_at(nx, ny, nz);
      if (type == CellSolid) {
        continue;
      }
      ++diagonal;
      if (type == CellFluid) {
        sum += pressure[static_cast<std::size_t>(topology.index_of(nx, ny, nz))];
      }
    }
    if (diagonal > 0) {
      result[index] = (negative_divergence[index] + sum) / static_cast<float>(diagonal);
    }
  });
}

void subtract_gradient(State &state)
{
  const Topology &topology = state.topology;
  std::vector<float> *faces[3] = {&state.u, &state.v, &state.w};
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    const CellType here = static_cast<CellType>(topology.cell_type[index]);
    for (int axis = 0; axis < 3; ++axis) {
      const int nx = x - (axis == 0), ny = y - (axis == 1), nz = z - (axis == 2);
      const CellType below = topology.type_at(nx, ny, nz);
      /* Faces touching a solid keep the obstacle velocity enforce_walls gave them. */
      if (here == CellSolid || below == CellSolid) {
        continue;
      }
      if (here == CellFluid || below == CellFluid) {
        const float upper = here == CellFluid ? state.pressure[index] : 0.0f;
        (*faces[axis])[index] -= upper - pressure_at(topology, state.pressure, nx, ny, nz);
      }
    }
  });
}

StepInfo project(State &state, const Settings &settings)
{
  const Topology &topology = state.topology;
  StepInfo info;
  info.active_bricks = topology.brick_count();
  if (topology.voxel_count() == 0) {
    return info;
  }

  std::vector<float> rhs;
  divergence(state, rhs);
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    rhs[i] = -rhs[i];
    /* The previous pressure is the initial guess. Cells that are no longer fluid have none. */
    if (topology.cell_type[i] != CellFluid) {
      state.pressure[i] = 0.0f;
    }
  }

  if (settings.pressure_solver == PressureSolver::ConjugateGradient) {
    info.pressure_iterations = solve_conjugate_gradient(
        topology, rhs, settings.pressure_accuracy, settings.pressure_iterations, state.pressure);
  }
  else if (settings.pressure_solver == PressureSolver::Multigrid) {
    multigrid_solve(topology, build_multigrid(topology), rhs, settings,
                    settings.pressure_iterations, state.pressure);
    info.pressure_iterations = settings.pressure_iterations;
  }
  else {
    std::vector<float> next;
    for (int i = 0; i < settings.pressure_iterations; ++i) {
      jacobi_sweep(topology, rhs, state.pressure, next);
      state.pressure.swap(next);
    }
    info.pressure_iterations = settings.pressure_iterations;
  }

  subtract_gradient(state);
  std::vector<float> remaining;
  divergence(state, remaining);
  info.max_divergence = max_abs(remaining);
  return info;
}

StepInfo step(State &state, const Settings &settings, const Sources *sources)
{
  retopologize(state, settings.thresholds, settings.dilation, sources);
  if (state.topology.voxel_count() == 0) {
    return {};
  }
  inject(state, sources, settings.dt);

  /* Advect the scalars and a velocity (`sources`) through the state's velocity for `dt`. */
  std::vector<float> density, temperature, fuel, flame, u, v, w, forward;
  const auto advect_all = [&](const float dt, const std::vector<float> &source_u,
                              const std::vector<float> &source_v,
                              const std::vector<float> &source_w) {
    const auto advect_channel = [&](const std::vector<float> &source, const float ox,
                                    const float oy, const float oz, std::vector<float> &result,
                                    const bool conserve = false) {
      if (settings.advection == Advection::MacCormack) {
        advect(state, source, ox, oy, oz, dt, forward, settings.midpoint_trace);
        advect_maccormack(state, source, forward, ox, oy, oz, dt, result,
                          settings.midpoint_trace);
        if (conserve) {
          conserve_correction(source, forward, result);
        }
      }
      else {
        advect(state, source, ox, oy, oz, dt, result, settings.midpoint_trace);
      }
    };
    advect_channel(state.density, 0.5f, 0.5f, 0.5f, density, settings.conserve_mass);
    advect_channel(state.temperature, 0.5f, 0.5f, 0.5f, temperature, settings.conserve_mass);
    if (settings.fire.enabled) {
      advect_channel(state.fuel, 0.5f, 0.5f, 0.5f, fuel, settings.conserve_mass);
      advect_channel(state.flame, 0.5f, 0.5f, 0.5f, flame, settings.conserve_mass);
      state.fuel.swap(fuel);
      state.flame.swap(flame);
    }
    advect_channel(source_u, 0.0f, 0.5f, 0.5f, u);
    advect_channel(source_v, 0.5f, 0.0f, 0.5f, v);
    advect_channel(source_w, 0.5f, 0.5f, 0.0f, w);
    state.density.swap(density);
    state.temperature.swap(temperature);
    state.u.swap(u);
    state.v.swap(v);
    state.w.swap(w);
  };
  const auto add_forces = [&](const float dt) {
    combust(state, settings.fire, dt);
    if (settings.heat_buoyancy != 0.0f) {
      add_buoyancy(state, state.temperature, settings.heat_buoyancy, dt, settings.gravity);
    }
    if (settings.density_buoyancy != 0.0f) {
      add_buoyancy(state, state.density, settings.density_buoyancy, dt, settings.gravity);
    }
    add_turbulence(state, settings.turbulence, settings.time, dt);
    for (const Turbulence &layer : settings.turbulence_layers) {
      add_turbulence(state, layer, settings.time, dt);
    }
    enforce_walls(state);
  };

  if (!settings.reflection) {
    /* Every channel is advected by the velocity from the start of the step. */
    advect_all(settings.dt, state.u, state.v, state.w);
    add_forces(settings.dt);
    StepInfo info = project(state, settings);
    info.dt = settings.dt;
    return info;
  }

  const float half = 0.5f * settings.dt;
  advect_all(half, state.u, state.v, state.w);
  add_forces(half);
  std::vector<float> reflected_u = state.u, reflected_v = state.v, reflected_w = state.w;
  project(state, settings);
  const auto reflect = [](const std::vector<float> &projected, std::vector<float> &unprojected) {
    for (std::size_t i = 0; i < projected.size(); ++i) {
      unprojected[i] = 2.0f * projected[i] - unprojected[i];
    }
  };
  reflect(state.u, reflected_u);
  reflect(state.v, reflected_v);
  reflect(state.w, reflected_w);
  /* The reflected velocity and the scalars travel through the projected field. */
  advect_all(half, reflected_u, reflected_v, reflected_w);
  add_forces(half);
  StepInfo info = project(state, settings);
  info.dt = settings.dt;
  return info;
}

FrameInfo advance(State &state,
                  const Settings &settings,
                  const float frame_time,
                  const Sources *sources)
{
  FrameInfo frame;
  float remaining = frame_time;
  Settings substep = settings;
  while (remaining > 0.0f) {
    if (frame.substeps >= settings.max_substeps) {
      throw std::runtime_error("The sparse solver exhausted its substep limit");
    }
    const float speed = max_speed(state);
    frame.max_speed = std::max(frame.max_speed, speed);
    substep.dt = cfl_time_step(remaining, settings.cfl, speed);
    substep.time = settings.time + (frame_time - remaining);
    frame.last_step = step(state, substep, sources);
    frame.last_step.dt = substep.dt;
    remaining -= substep.dt;
    ++frame.substeps;
  }
  return frame;
}

}  // namespace manta_improve::sparse
