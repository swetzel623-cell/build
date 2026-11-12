// SPDX-License-Identifier: MIT
// Multigrid for the sparse pressure Poisson system: CPU reference.
//
// Coarsening is plain aggregation of 2x2x2 cells with piecewise-constant transfer. The Galerkin
// coarse operator of a 7-point stencil is then again a 7-point stencil, described by a diagonal
// and one weight per lower face, so every level uses the same smoother and residual. Nothing here
// needs a reduction: a solve is a fixed sequence of per-cell gathers, like the other kernels.
#include "manta_improve/sparse_gas.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace manta_improve::sparse {
namespace {

constexpr int axis_offsets[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

std::size_t flat_index(const Extent &extent, const int x, const int y, const int z)
{
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(extent.x) *
             (static_cast<std::size_t>(y) +
              static_cast<std::size_t>(extent.y) * static_cast<std::size_t>(z));
}

/** The voxel level: operator entries follow from the cell types. */
struct FineView {
  const Topology &topology;

  std::int64_t index(const int x, const int y, const int z) const
  {
    return topology.index_of(x, y, z);
  }

  float diagonal(const int x, const int y, const int z) const
  {
    if (topology.type_at(x, y, z) != CellFluid) {
      return 0.0f;
    }
    int count = 0;
    for (const auto &offset : axis_offsets) {
      count += topology.type_at(x - offset[0], y - offset[1], z - offset[2]) != CellSolid;
      count += topology.type_at(x + offset[0], y + offset[1], z + offset[2]) != CellSolid;
    }
    return static_cast<float>(count);
  }

  /** Coupling between a cell and its lower neighbor along `axis`. */
  float weight(const int x, const int y, const int z, const int axis) const
  {
    const auto &offset = axis_offsets[axis];
    return topology.type_at(x, y, z) == CellFluid &&
                   topology.type_at(x - offset[0], y - offset[1], z - offset[2]) == CellFluid ?
               1.0f :
               0.0f;
  }

  template<typename Fn> void for_each(Fn &&fn) const
  {
    const int size = topology.brick_size;
    std::size_t index = 0;
    for (std::size_t slot = 0; slot < topology.brick_count(); ++slot) {
      const int bx = topology.brick_coords[slot * 4 + 0] * size;
      const int by = topology.brick_coords[slot * 4 + 1] * size;
      const int bz = topology.brick_coords[slot * 4 + 2] * size;
      for (int z = 0; z < size; ++z) {
        for (int y = 0; y < size; ++y) {
          for (int x = 0; x < size; ++x, ++index) {
            fn(index, bx + x, by + y, bz + z);
          }
        }
      }
    }
  }
};

/** A coarse level with stored operator entries. */
struct CoarseView {
  const Topology &topology;
  const MultigridLevel &level;

  std::int64_t index(const int x, const int y, const int z) const
  {
    return multigrid_index(topology, level, x, y, z);
  }

  float diagonal(const int x, const int y, const int z) const
  {
    const std::int64_t i = index(x, y, z);
    return i < 0 ? 0.0f : level.diagonal[static_cast<std::size_t>(i)];
  }

  float weight(const int x, const int y, const int z, const int axis) const
  {
    const std::int64_t i = index(x, y, z);
    if (i < 0) {
      return 0.0f;
    }
    const std::vector<float> &weights = axis == 0 ? level.weight_x :
                                        axis == 1 ? level.weight_y :
                                                    level.weight_z;
    return weights[static_cast<std::size_t>(i)];
  }

  template<typename Fn> void for_each(Fn &&fn) const
  {
    std::size_t index = 0;
    if (level.dense) {
      for (int z = 0; z < level.dims.z; ++z) {
        for (int y = 0; y < level.dims.y; ++y) {
          for (int x = 0; x < level.dims.x; ++x, ++index) {
            fn(index, x, y, z);
          }
        }
      }
      return;
    }
    const int side = level.side;
    for (std::size_t slot = 0; slot < topology.brick_count(); ++slot) {
      const int bx = topology.brick_coords[slot * 4 + 0] * side;
      const int by = topology.brick_coords[slot * 4 + 1] * side;
      const int bz = topology.brick_coords[slot * 4 + 2] * side;
      for (int z = 0; z < side; ++z) {
        for (int y = 0; y < side; ++y) {
          for (int x = 0; x < side; ++x, ++index) {
            fn(index, bx + x, by + y, bz + z);
          }
        }
      }
    }
  }
};

template<typename View> float value_at(const View &view, const std::vector<float> &values,
                                       const int x, const int y, const int z)
{
  const std::int64_t i = view.index(x, y, z);
  return i < 0 ? 0.0f : values[static_cast<std::size_t>(i)];
}

/** Sum over the six neighbors of coupling weight times neighbor value. */
template<typename View> float neighbor_sum(const View &view, const std::vector<float> &values,
                                           const int x, const int y, const int z)
{
  float sum = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    const auto &offset = axis_offsets[axis];
    const float lower = view.weight(x, y, z, axis);
    if (lower != 0.0f) {
      sum += lower * value_at(view, values, x - offset[0], y - offset[1], z - offset[2]);
    }
    const float upper = view.weight(x + offset[0], y + offset[1], z + offset[2], axis);
    if (upper != 0.0f) {
      sum += upper * value_at(view, values, x + offset[0], y + offset[1], z + offset[2]);
    }
  }
  return sum;
}

/** One red-black Gauss-Seidel sweep: both colors, in place. */
template<typename View> void smooth(const View &view, const std::vector<float> &rhs,
                                    std::vector<float> &values)
{
  for (int color = 0; color < 2; ++color) {
    view.for_each([&](const std::size_t index, const int x, const int y, const int z) {
      if (((x + y + z) & 1) != color) {
        return;
      }
      const float diagonal = view.diagonal(x, y, z);
      if (diagonal > 0.0f) {
        values[index] = (rhs[index] + neighbor_sum(view, values, x, y, z)) / diagonal;
      }
    });
  }
}

template<typename View> void residual(const View &view, const std::vector<float> &rhs,
                                      const std::vector<float> &values,
                                      std::vector<float> &result)
{
  view.for_each([&](const std::size_t index, const int x, const int y, const int z) {
    const float diagonal = view.diagonal(x, y, z);
    result[index] = diagonal > 0.0f ? rhs[index] - (diagonal * values[index] -
                                                    neighbor_sum(view, values, x, y, z)) :
                                      0.0f;
  });
}

/** Galerkin coarse operator for piecewise-constant aggregation of 2x2x2 cells of `fine`. */
template<typename View> void build_level(const View &fine, const CoarseView &coarse,
                                         MultigridLevel &level)
{
  coarse.for_each([&](const std::size_t index, const int x, const int y, const int z) {
    float diagonal = 0.0f;
    float weights[3] = {0.0f, 0.0f, 0.0f};
    for (int child = 0; child < 8; ++child) {
      const int offset[3] = {child & 1, (child >> 1) & 1, (child >> 2) & 1};
      const int cx = 2 * x + offset[0], cy = 2 * y + offset[1], cz = 2 * z + offset[2];
      diagonal += fine.diagonal(cx, cy, cz);
      for (int axis = 0; axis < 3; ++axis) {
        const float weight = fine.weight(cx, cy, cz, axis);
        if (offset[axis]) {
          /* Coupling inside the aggregate: it leaves the off-diagonals of both its cells. */
          diagonal -= 2.0f * weight;
        }
        else {
          weights[axis] += weight;
        }
      }
    }
    level.diagonal[index] = diagonal;
    level.weight_x[index] = weights[0];
    level.weight_y[index] = weights[1];
    level.weight_z[index] = weights[2];
  });
}

template<typename FineT> void restrict_residual(const FineT &fine,
                                                const std::vector<float> &fine_residual,
                                                const CoarseView &coarse,
                                                std::vector<float> &coarse_rhs)
{
  coarse.for_each([&](const std::size_t index, const int x, const int y, const int z) {
    float sum = 0.0f;
    for (int child = 0; child < 8; ++child) {
      sum += value_at(fine, fine_residual, 2 * x + (child & 1), 2 * y + ((child >> 1) & 1),
                      2 * z + ((child >> 2) & 1));
    }
    coarse_rhs[index] = sum;
  });
}

template<typename FineT> void prolong(const FineT &fine, const CoarseView &coarse,
                                      const std::vector<float> &coarse_values, const float scale,
                                      std::vector<float> &fine_values)
{
  fine.for_each([&](const std::size_t index, const int x, const int y, const int z) {
    if (fine.diagonal(x, y, z) > 0.0f) {
      fine_values[index] += scale * value_at(coarse, coarse_values, x >> 1, y >> 1, z >> 1);
    }
  });
}

struct LevelData {
  std::vector<float> values, rhs, residual;
};

template<typename View> void v_cycle(const Topology &topology,
                                     const Multigrid &multigrid,
                                     const Settings &settings,
                                     const View &view,
                                     const std::size_t coarse_level,
                                     std::vector<LevelData> &data,
                                     const std::vector<float> &rhs,
                                     std::vector<float> &values,
                                     std::vector<float> &scratch)
{
  if (coarse_level >= multigrid.levels.size()) {
    for (int i = 0; i < settings.multigrid_coarsest_sweeps; ++i) {
      smooth(view, rhs, values);
    }
    return;
  }
  for (int i = 0; i < settings.multigrid_pre_sweeps; ++i) {
    smooth(view, rhs, values);
  }
  residual(view, rhs, values, scratch);

  const CoarseView coarse{topology, multigrid.levels[coarse_level]};
  LevelData &next = data[coarse_level];
  restrict_residual(view, scratch, coarse, next.rhs);
  std::fill(next.values.begin(), next.values.end(), 0.0f);
  /* W-cycles only near the top: deeper levels are tiny, and revisiting them costs dispatches. */
  const int coarse_cycles = static_cast<int>(coarse_level) < settings.multigrid_w_depth ?
                                settings.multigrid_coarse_cycles :
                                1;
  for (int i = 0; i < coarse_cycles; ++i) {
    v_cycle(topology, multigrid, settings, coarse, coarse_level + 1, data, next.rhs, next.values,
            next.residual);
  }
  prolong(view, coarse, next.values, settings.multigrid_overcorrection, values);

  for (int i = 0; i < settings.multigrid_post_sweeps; ++i) {
    smooth(view, rhs, values);
  }
}

}  // namespace

std::int64_t multigrid_index(const Topology &topology,
                             const MultigridLevel &level,
                             const int x,
                             const int y,
                             const int z)
{
  if (x < 0 || y < 0 || z < 0) {
    return -1;
  }
  if (level.dense) {
    if (x >= level.dims.x || y >= level.dims.y || z >= level.dims.z) {
      return -1;
    }
    return static_cast<std::int64_t>(flat_index(level.dims, x, y, z));
  }
  const int side = level.side;
  const int bx = x / side, by = y / side, bz = z / side;
  const Extent &bricks = topology.brick_extent;
  if (bx >= bricks.x || by >= bricks.y || bz >= bricks.z) {
    return -1;
  }
  const std::int32_t slot = topology.brick_table[flat_index(bricks, bx, by, bz)];
  if (slot < 0) {
    return -1;
  }
  const int lx = x - bx * side, ly = y - by * side, lz = z - bz * side;
  return static_cast<std::int64_t>(slot) * side * side * side + lx + side * (ly + side * lz);
}

Multigrid build_multigrid(const Topology &topology)
{
  const int brick_size = topology.brick_size;
  if (brick_size & (brick_size - 1)) {
    throw std::invalid_argument("The multigrid pressure solver needs a power-of-two brick size");
  }
  Multigrid multigrid;
  const auto add_level = [&](MultigridLevel level, const std::size_t cells) {
    level.diagonal.assign(cells, 0.0f);
    level.weight_x.assign(cells, 0.0f);
    level.weight_y.assign(cells, 0.0f);
    level.weight_z.assign(cells, 0.0f);
    multigrid.levels.push_back(std::move(level));
    MultigridLevel &added = multigrid.levels.back();
    const CoarseView coarse{topology, added};
    if (multigrid.levels.size() == 1) {
      build_level(FineView{topology}, coarse, added);
    }
    else {
      /* Copy: the vector may have reallocated, and the finer level must stay readable. */
      const MultigridLevel finer = multigrid.levels[multigrid.levels.size() - 2];
      build_level(CoarseView{topology, finer}, coarse, added);
    }
  };

  /* Levels that still have several cells per brick are stored per brick slot. */
  for (int side = brick_size / 2; side >= 2; side /= 2) {
    MultigridLevel level;
    level.side = side;
    add_level(std::move(level), topology.brick_count() * side * side * side);
  }
  /* From one cell per brick on, levels are dense grids over the brick grid. */
  Extent dims = topology.brick_extent;
  while (true) {
    MultigridLevel level;
    level.dense = true;
    level.dims = dims;
    add_level(std::move(level), static_cast<std::size_t>(dims.x) * dims.y * dims.z);
    if (std::max({dims.x, dims.y, dims.z}) <= 2) {
      break;
    }
    dims = {(dims.x + 1) / 2, (dims.y + 1) / 2, (dims.z + 1) / 2};
  }
  return multigrid;
}

void multigrid_solve(const Topology &topology,
                     const Multigrid &multigrid,
                     const std::vector<float> &rhs,
                     const Settings &settings,
                     const int cycles,
                     std::vector<float> &pressure)
{
  std::vector<LevelData> data(multigrid.levels.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    const std::size_t cells = multigrid.levels[i].cell_count();
    data[i].values.assign(cells, 0.0f);
    data[i].rhs.assign(cells, 0.0f);
    data[i].residual.assign(cells, 0.0f);
  }
  std::vector<float> scratch(topology.voxel_count(), 0.0f);
  const FineView fine{topology};
  for (int cycle = 0; cycle < cycles; ++cycle) {
    v_cycle(topology, multigrid, settings, fine, 0, data, rhs, pressure, scratch);
  }
}

float poisson_residual(const Topology &topology,
                       const std::vector<float> &rhs,
                       const std::vector<float> &pressure)
{
  std::vector<float> result(topology.voxel_count(), 0.0f);
  residual(FineView{topology}, rhs, pressure, result);
  float largest = 0.0f;
  for (const float value : result) {
    largest = std::max(largest, std::abs(value));
  }
  return largest;
}

}  // namespace manta_improve::sparse
