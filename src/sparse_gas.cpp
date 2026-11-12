// SPDX-License-Identifier: MIT
#include "manta_improve/sparse_gas.hpp"

#include <algorithm>
#include <cmath>
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
  for (std::vector<float> *channel :
       {&state.density, &state.temperature, &state.u, &state.v, &state.w, &state.pressure})
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

Topology build_topology(const Extent extent,
                        const int brick_size,
                        const std::vector<std::uint8_t> &brick_active,
                        const int dilation,
                        const bool classify_cells)
{
  if (extent.x < 4 || extent.y < 4 || extent.z < 4 || brick_size < 2 || dilation < 0) {
    throw std::invalid_argument("Invalid sparse topology parameters");
  }
  Topology topology;
  topology.extent = extent;
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
   * The remaining stored cells form a one-voxel layer of open air that holds those faces. */
  if (!classify_cells) {
    return topology;
  }
  topology.cell_type.assign(topology.voxel_count(), CellOpen);
  const auto stored_or_wall = [&](const int x, const int y, const int z) {
    return is_wall(extent, x, y, z) || topology.index_of(x, y, z) >= 0;
  };
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    if (is_wall(extent, x, y, z)) {
      topology.cell_type[index] = CellSolid;
    }
    else if (stored_or_wall(x + 1, y, z) && stored_or_wall(x, y + 1, z) &&
             stored_or_wall(x, y, z + 1))
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
  state.topology = build_topology(extent, brick_size, flags, dilation);
  resize_channels(state);
  for_each_voxel(state.topology, [&](const std::size_t index, int x, int y, int z) {
    if (x >= extent.x || y >= extent.y || z >= extent.z) {
      return;
    }
    const std::size_t i = dense_index(extent, x, y, z);
    state.density[index] = dense.density[i];
    state.temperature[index] = dense.temperature[i];
    if (has_velocity) {
      state.u[index] = dense.u[i];
      state.v[index] = dense.v[i];
      state.w[index] = dense.w[i];
    }
  });
  return state;
}

DenseFields to_dense(const State &state)
{
  const Extent &extent = state.topology.extent;
  const std::size_t count = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  DenseFields dense;
  dense.extent = extent;
  for (std::vector<float> *channel :
       {&dense.density, &dense.temperature, &dense.u, &dense.v, &dense.w})
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
    dense.u[i] = state.u[index];
    dense.v[i] = state.v[index];
    dense.w[i] = state.w[index];
  });
  return dense;
}

void retopologize(State &state, const ActivityThresholds &thresholds, const int dilation)
{
  const Topology &old_topology = state.topology;
  Topology topology = build_topology(old_topology.extent,
                                     old_topology.brick_size,
                                     activity_flags(state, thresholds),
                                     dilation);
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
    copy(state.u, moved.u);
    copy(state.v, moved.v);
    copy(state.w, moved.w);
    copy(state.pressure, moved.pressure);
  }
  state = std::move(moved);
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

void advect(const State &state,
            const std::vector<float> &source,
            const float offset_x,
            const float offset_y,
            const float offset_z,
            const float dt,
            std::vector<float> &result)
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
    const float vx = sample(topology, state.u, px, py, pz, 0.0f, 0.5f, 0.5f);
    const float vy = sample(topology, state.v, px, py, pz, 0.5f, 0.0f, 0.5f);
    const float vz = sample(topology, state.w, px, py, pz, 0.5f, 0.5f, 0.0f);
    result[index] = sample(topology,
                           source,
                           px - dt * vx,
                           py - dt * vy,
                           pz - dt * vz,
                           offset_x,
                           offset_y,
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
                       std::vector<float> &result)
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
    const float vx = sample(topology, state.u, px, py, pz, 0.0f, 0.5f, 0.5f);
    const float vy = sample(topology, state.v, px, py, pz, 0.5f, 0.0f, 0.5f);
    const float vz = sample(topology, state.w, px, py, pz, 0.5f, 0.5f, 0.0f);

    /* The values the forward trace interpolated bound the result. If any of them has no
     * storage, the trace left the active region: keep the first-order value there. */
    const float bx = std::floor(px - dt * vx - offset_x);
    const float by = std::floor(py - dt * vy - offset_y);
    const float bz = std::floor(pz - dt * vz - offset_z);
    const int ix = static_cast<int>(bx), iy = static_cast<int>(by), iz = static_cast<int>(bz);
    float lowest = 0.0f, highest = 0.0f;
    for (int corner = 0; corner < 8; ++corner) {
      const std::int64_t i = topology.index_of(
          ix + (corner & 1), iy + ((corner >> 1) & 1), iz + ((corner >> 2) & 1));
      if (i < 0) {
        return;
      }
      const float value = source[static_cast<std::size_t>(i)];
      lowest = corner == 0 ? value : std::min(lowest, value);
      highest = corner == 0 ? value : std::max(highest, value);
    }

    const float backward = sample(
        topology, forward, px + dt * vx, py + dt * vy, pz + dt * vz, offset_x, offset_y, offset_z);
    const float corrected = forward[index] + 0.5f * (source[index] - backward);
    result[index] = std::min(std::max(corrected, lowest), highest);
  });
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

void enforce_walls(State &state)
{
  const Topology &topology = state.topology;
  std::vector<float> *faces[3] = {&state.u, &state.v, &state.w};
  for_each_voxel(topology, [&](const std::size_t index, const int x, const int y, const int z) {
    for (int axis = 0; axis < 3; ++axis) {
      const int nx = x - (axis == 0), ny = y - (axis == 1), nz = z - (axis == 2);
      if (topology.cell_type[index] == CellSolid || topology.type_at(nx, ny, nz) == CellSolid) {
        (*faces[axis])[index] = 0.0f;
      }
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
      if (here == CellSolid || below == CellSolid) {
        (*faces[axis])[index] = 0.0f;
      }
      else if (here == CellFluid || below == CellFluid) {
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

StepInfo step(State &state, const Settings &settings)
{
  retopologize(state, settings.thresholds, settings.dilation);
  if (state.topology.voxel_count() == 0) {
    return {};
  }

  /* Every channel is advected by the velocity from the start of the step. */
  std::vector<float> density, temperature, u, v, w, forward;
  const auto advect_channel = [&](const std::vector<float> &source, const float ox, const float oy,
                                  const float oz, std::vector<float> &result) {
    if (settings.advection == Advection::MacCormack) {
      advect(state, source, ox, oy, oz, settings.dt, forward);
      advect_maccormack(state, source, forward, ox, oy, oz, settings.dt, result);
    }
    else {
      advect(state, source, ox, oy, oz, settings.dt, result);
    }
  };
  advect_channel(state.density, 0.5f, 0.5f, 0.5f, density);
  advect_channel(state.temperature, 0.5f, 0.5f, 0.5f, temperature);
  advect_channel(state.u, 0.0f, 0.5f, 0.5f, u);
  advect_channel(state.v, 0.5f, 0.0f, 0.5f, v);
  advect_channel(state.w, 0.5f, 0.5f, 0.0f, w);
  state.density.swap(density);
  state.temperature.swap(temperature);
  state.u.swap(u);
  state.v.swap(v);
  state.w.swap(w);

  if (settings.heat_buoyancy != 0.0f) {
    add_buoyancy(state, state.temperature, settings.heat_buoyancy, settings.dt, settings.gravity);
  }
  if (settings.density_buoyancy != 0.0f) {
    add_buoyancy(state, state.density, settings.density_buoyancy, settings.dt, settings.gravity);
  }
  enforce_walls(state);
  StepInfo info = project(state, settings);
  info.dt = settings.dt;
  return info;
}

FrameInfo advance(State &state, const Settings &settings, const float frame_time)
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
    frame.last_step = step(state, substep);
    frame.last_step.dt = substep.dt;
    remaining -= substep.dt;
    ++frame.substeps;
  }
  return frame;
}

}  // namespace manta_improve::sparse
