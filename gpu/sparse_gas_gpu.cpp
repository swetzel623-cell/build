// SPDX-License-Identifier: MIT
#include "sparse_gas_gpu.hpp"

#include "vulkan_compute.hpp"

#include "activity.comp.h"
#include "advect.comp.h"
#include "buoyancy.comp.h"
#include "classify.comp.h"
#include "classify_links.comp.h"
#include "mask_pressure.comp.h"
#include "mg_smooth0.comp.h"
#include "mg_residual0.comp.h"
#include "mg_smooth.comp.h"
#include "mg_residual.comp.h"
#include "mg_restrict.comp.h"
#include "mg_prolong0.comp.h"
#include "mg_prolong.comp.h"
#include "mg_build0.comp.h"
#include "mg_build.comp.h"
#include "mg_dense_cycle.comp.h"
#include "divergence.comp.h"
#include "gradient.comp.h"
#include "jacobi.comp.h"
#include "maccormack.comp.h"
#include "remap.comp.h"
#include "walls.comp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace manta_improve::gpu {
namespace {

/** Push constants of the stencil kernels; matches Params in common.glsl. */
struct Params {
  std::int32_t extent[4];
  std::int32_t bricks[4];
  float a[4];
};

struct ActivityParams {
  std::int32_t voxels_per_brick;
  std::int32_t count;
  float scalar_threshold;
  float velocity_threshold;
};

/** Push constants of the multigrid kernels; matches Params in mg_common.glsl. */
struct MultigridParams {
  std::int32_t bricks[4];
  std::int32_t la[4];
  std::int32_t lb[4];
  std::int32_t count;
  std::int32_t color;
  float scale;
  std::int32_t unused;
};

/** Push constants of mg_dense_cycle.comp. */
struct DenseCycleParams {
  std::int32_t first_level;
  std::int32_t level_count;
  std::int32_t pre_sweeps;
  std::int32_t post_sweeps;
  std::int32_t coarsest_sweeps;
  float scale;
};

struct RemapParams {
  std::int32_t voxels_per_brick;
  std::int32_t count;
};

/** Logical channels. The first six are the state; the rest are scratch of the same size. */
enum Channel { Density, Temperature, U, V, W, Pressure, StateChannels = 6, AllChannels = 12 };

double milliseconds_since(const std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

std::uint32_t groups_for(const std::size_t count)
{
  return static_cast<std::uint32_t>((count + 63) / 64);
}

}  // namespace

struct SparseGasGpu::Impl {
  Compute compute;
  int advect, maccormack, buoyancy, walls, divergence, jacobi, gradient, activity, remap, classify;
  int classify_links, mask_pressure, mg_smooth0, mg_residual0, mg_smooth, mg_residual;
  int mg_restrict, mg_prolong0, mg_prolong, mg_build0, mg_build, mg_dense_cycle;

  /** One coarse multigrid level on the device. */
  struct Level {
    sparse::MultigridLevel layout;
    std::size_t cells = 0;
    /** Dense levels: first cell in the buffers all dense levels share. */
    std::size_t offset = 0;
    /** Owned by the level if it is stored per brick; copies of the shared buffers if dense. */
    Compute::Buffer op, values, rhs, residual;
  };
  std::vector<Level> levels;
  /** All dense levels back to back, so one dispatch can cycle through them. */
  Compute::Buffer dense_op, dense_values, dense_rhs, dense_residual, dense_table;
  int first_dense = 0;
  /** First level handed to the single-dispatch cycle: the dense levels from here on are tiny. */
  int first_single = 0;
  std::size_t single_dispatch_cell_limit = 512;
  /** Per voxel: cell type and neighbor links, packed by the classify_links kernel. */
  Compute::Buffer links;

  sparse::Topology topology;
  Compute::Buffer table, coords, types, slot_map, flags, speeds;
  std::size_t slot_capacity = 0;
  /** Physical buffers; `role[channel]` says which one currently plays each channel. */
  std::vector<Compute::Buffer> buffers;
  std::array<int, AllChannels> role{};
  std::size_t voxel_capacity = 0;
  Timing timing;
  /** Largest velocity component at the start of the last step. */
  float last_speed = 0.0f;

  Impl()
  {
#define KERNEL(name) compute.create_kernel(name##_spv, sizeof(name##_spv) / sizeof(std::uint32_t))
    advect = KERNEL(advect);
    maccormack = KERNEL(maccormack);
    buoyancy = KERNEL(buoyancy);
    walls = KERNEL(walls);
    divergence = KERNEL(divergence);
    jacobi = KERNEL(jacobi);
    gradient = KERNEL(gradient);
    activity = KERNEL(activity);
    remap = KERNEL(remap);
    classify = KERNEL(classify);
    classify_links = KERNEL(classify_links);
    mask_pressure = KERNEL(mask_pressure);
    mg_smooth0 = KERNEL(mg_smooth0);
    mg_residual0 = KERNEL(mg_residual0);
    mg_smooth = KERNEL(mg_smooth);
    mg_residual = KERNEL(mg_residual);
    mg_restrict = KERNEL(mg_restrict);
    mg_prolong0 = KERNEL(mg_prolong0);
    mg_prolong = KERNEL(mg_prolong);
    mg_build0 = KERNEL(mg_build0);
    mg_build = KERNEL(mg_build);
    mg_dense_cycle = KERNEL(mg_dense_cycle);
#undef KERNEL
  }

  ~Impl()
  {
    for (Compute::Buffer *buffer : {&table, &coords, &types, &slot_map, &flags, &speeds, &links,
                                    &dense_op,
                                    &dense_values, &dense_rhs, &dense_residual, &dense_table})
    {
      compute.destroy_buffer(*buffer);
    }
    for (Level &level : levels) {
      destroy_level(level);
    }
    for (Compute::Buffer &buffer : buffers) {
      compute.destroy_buffer(buffer);
    }
  }

  const Compute::Buffer &channel(const int logical) const
  {
    return buffers[static_cast<std::size_t>(role[static_cast<std::size_t>(logical)])];
  }

  void ensure(Compute::Buffer &buffer, const std::size_t bytes)
  {
    if (buffer.handle && buffer.size >= bytes) {
      return;
    }
    compute.destroy_buffer(buffer);
    buffer = compute.create_buffer(bytes + bytes / 2);
  }

  void destroy_level(Level &level)
  {
    if (level.layout.dense) {
      return;
    }
    for (Compute::Buffer *buffer : {&level.op, &level.values, &level.rhs, &level.residual}) {
      compute.destroy_buffer(*buffer);
    }
  }

  /** Make room for a topology's tables. Must be called outside recording. */
  void reserve_topology(const sparse::Topology &next)
  {
    ensure(table, next.brick_table.size() * sizeof(std::int32_t));
    if (next.brick_count() > slot_capacity || !coords.handle) {
      slot_capacity = next.brick_count() + next.brick_count() / 2 + 1;
      ensure(coords, slot_capacity * 4 * sizeof(std::int32_t));
      ensure(slot_map, slot_capacity * sizeof(std::int32_t));
      ensure(flags, slot_capacity * sizeof(std::uint32_t));
      ensure(speeds, slot_capacity * sizeof(std::uint32_t));
    }
    ensure(types, next.voxel_count() * sizeof(std::uint32_t));
    ensure(links, next.voxel_count() * sizeof(std::uint32_t));

    const std::vector<sparse::MultigridLevel> layout = sparse::multigrid_layout(next);
    bool same_shape = layout.size() == levels.size();
    for (std::size_t i = 0; same_shape && i < layout.size(); ++i) {
      same_shape = layout[i].dense == levels[i].layout.dense;
    }
    if (!same_shape) {
      for (Level &level : levels) {
        destroy_level(level);
      }
      levels.assign(layout.size(), {});
    }
    std::size_t dense_cells = 0;
    int dense_levels = 0;
    first_dense = static_cast<int>(layout.size());
    for (std::size_t i = 0; i < layout.size(); ++i) {
      Level &level = levels[i];
      level.layout = layout[i];
      level.cells = sparse::multigrid_cell_count(next, layout[i]);
      if (layout[i].dense) {
        first_dense = std::min(first_dense, static_cast<int>(i));
        level.offset = dense_cells;
        dense_cells += level.cells;
        ++dense_levels;
      }
      else {
        ensure(level.op, level.cells * 4 * sizeof(float));
        ensure(level.values, level.cells * sizeof(float));
        ensure(level.rhs, level.cells * sizeof(float));
        ensure(level.residual, level.cells * sizeof(float));
      }
    }
    first_single = static_cast<int>(layout.size());
    for (int i = static_cast<int>(layout.size()) - 1; i >= first_dense; --i) {
      if (levels[static_cast<std::size_t>(i)].cells > single_dispatch_cell_limit) {
        break;
      }
      first_single = i;
    }
    ensure(dense_op, dense_cells * 4 * sizeof(float));
    ensure(dense_values, dense_cells * sizeof(float));
    ensure(dense_rhs, dense_cells * sizeof(float));
    ensure(dense_residual, dense_cells * sizeof(float));
    ensure(dense_table, static_cast<std::size_t>(dense_levels) * 4 * sizeof(std::int32_t));
    for (Level &level : levels) {
      if (level.layout.dense) {
        level.op = dense_op;
        level.values = dense_values;
        level.rhs = dense_rhs;
        level.residual = dense_residual;
      }
    }
  }

  /** Level descriptor as the multigrid kernels expect it; -1 is the voxel level. */
  void describe(const int level, std::int32_t out[4]) const
  {
    if (level < 0) {
      out[0] = out[1] = out[2] = 0;
      out[3] = topology.brick_size;
      return;
    }
    const sparse::MultigridLevel &layout = levels[static_cast<std::size_t>(level)].layout;
    out[0] = layout.dense ? layout.dims.x : 0;
    out[1] = layout.dense ? layout.dims.y : 0;
    out[2] = layout.dense ? layout.dims.z : 0;
    out[3] = layout.dense ?
                 -static_cast<std::int32_t>(levels[static_cast<std::size_t>(level)].offset) :
                 layout.side;
  }

  std::size_t level_cells(const int level) const
  {
    return level < 0 ? topology.voxel_count() : levels[static_cast<std::size_t>(level)].cells;
  }

  MultigridParams multigrid_params(const int la, const int lb, const std::size_t count) const
  {
    MultigridParams p{};
    p.bricks[0] = topology.brick_extent.x;
    p.bricks[1] = topology.brick_extent.y;
    p.bricks[2] = topology.brick_extent.z;
    describe(la, p.la);
    describe(lb < 0 ? la : lb, p.lb);
    p.count = static_cast<std::int32_t>(count);
    return p;
  }

  /** Run V/W-cycles on the device; mirrors v_cycle in src/sparse_multigrid.cpp. */
  void record_multigrid(const sparse::Settings &settings, const int rhs_channel,
                        const int residual_channel)
  {
    const auto op = [&](const int l) -> const Compute::Buffer & {
      return l < 0 ? links : levels[static_cast<std::size_t>(l)].op;
    };
    const auto values = [&](const int l) -> const Compute::Buffer & {
      return l < 0 ? channel(Pressure) : levels[static_cast<std::size_t>(l)].values;
    };
    const auto rhs = [&](const int l) -> const Compute::Buffer & {
      return l < 0 ? channel(rhs_channel) : levels[static_cast<std::size_t>(l)].rhs;
    };
    const auto residual = [&](const int l) -> const Compute::Buffer & {
      return l < 0 ? channel(residual_channel) : levels[static_cast<std::size_t>(l)].residual;
    };
    const auto smooth = [&](const int l, const int sweeps) {
      MultigridParams p = multigrid_params(l, -1, level_cells(l));
      for (int i = 0; i < sweeps; ++i) {
        for (int color = 0; color < 2; ++color) {
          p.color = color;
          compute.dispatch(l < 0 ? mg_smooth0 : mg_smooth,
                           &p,
                           sizeof(p),
                           {&table, &coords, &op(l), &rhs(l), &values(l), nullptr, nullptr, nullptr},
                           groups_for(level_cells(l)),
                           l < 0 ? "mg smooth (voxels)" : "mg smooth (coarse)");
        }
      }
    };
    const int last = static_cast<int>(levels.size()) - 1;
    const auto cycle = [&](const auto &self, const int l) -> void {
      if (l >= last) {
        smooth(l, settings.multigrid_coarsest_sweeps);
        return;
      }
      smooth(l, settings.multigrid_pre_sweeps);
      const MultigridParams here = multigrid_params(l, -1, level_cells(l));
      compute.dispatch(l < 0 ? mg_residual0 : mg_residual,
                       &here,
                       sizeof(here),
                       {&table, &coords, &op(l), &rhs(l), &values(l), &residual(l), nullptr, nullptr},
                       groups_for(level_cells(l)),
                       l < 0 ? "mg residual (voxels)" : "mg residual (coarse)");
      const int coarse = l + 1;
      const MultigridParams down = multigrid_params(l, coarse, level_cells(coarse));
      compute.dispatch(mg_restrict,
                       &down,
                       sizeof(down),
                       {&table, &coords, nullptr, &residual(l), &rhs(coarse), &values(coarse),
                        nullptr, nullptr},
                       groups_for(level_cells(coarse)),
                       "mg restrict");
      const int coarse_cycles = coarse < settings.multigrid_w_depth ?
                                    settings.multigrid_coarse_cycles :
                                    1;
      if (coarse == first_single && settings.multigrid_w_depth <= first_single) {
        /* The remaining levels are tiny and take plain V-cycles: one dispatch runs them all. */
        const DenseCycleParams dense = {first_single - first_dense,
                                        static_cast<std::int32_t>(levels.size()) - first_single,
                                        settings.multigrid_pre_sweeps,
                                        settings.multigrid_post_sweeps,
                                        settings.multigrid_coarsest_sweeps,
                                        settings.multigrid_overcorrection};
        compute.dispatch(mg_dense_cycle,
                         &dense,
                         sizeof(dense),
                         {&dense_table, &dense_op, &dense_rhs, &dense_values, &dense_residual,
                          nullptr, nullptr, nullptr},
                         1,
                         "mg tiny levels");
      }
      else {
        for (int i = 0; i < coarse_cycles; ++i) {
          self(self, coarse);
        }
      }
      MultigridParams up = multigrid_params(l, coarse, level_cells(l));
      up.scale = settings.multigrid_overcorrection;
      compute.dispatch(l < 0 ? mg_prolong0 : mg_prolong,
                       &up,
                       sizeof(up),
                       {&table, &coords, &op(l), &values(coarse), &values(l), nullptr, nullptr,
                        nullptr},
                       groups_for(level_cells(l)),
                       "mg prolong");
      smooth(l, settings.multigrid_post_sweeps);
    };
    for (int i = 0; i < settings.pressure_iterations; ++i) {
      cycle(cycle, -1);
    }
  }

  /** Upload the brick tables of `topology` (the current one) and classify its cells. */
  void record_topology_upload()
  {
    compute.upload(
        table, topology.brick_table.data(), topology.brick_table.size() * sizeof(std::int32_t));
    compute.upload(coords,
                   topology.brick_coords.data(),
                   topology.brick_coords.size() * sizeof(std::int32_t));
    const Params p = params(0, 0, 0, 0);
    compute.dispatch(classify,
                     &p,
                     sizeof(p),
                     {&table, &coords, &types, nullptr, nullptr, nullptr, nullptr, nullptr},
                     groups_for(topology.voxel_count()),
                     "classify");
    compute.dispatch(classify_links,
                     &p,
                     sizeof(p),
                     {&table, &coords, &types, &links, nullptr, nullptr, nullptr, nullptr},
                     groups_for(topology.voxel_count()),
                     "classify");
    std::vector<std::int32_t> dense_levels;
    for (const Level &level : levels) {
      if (level.layout.dense) {
        dense_levels.insert(dense_levels.end(),
                            {level.layout.dims.x, level.layout.dims.y, level.layout.dims.z,
                             static_cast<std::int32_t>(level.offset)});
      }
    }
    compute.upload(dense_table, dense_levels.data(), dense_levels.size() * sizeof(std::int32_t));
    /* Coarse operators, each from the level below it. */
    for (int i = 0; i < static_cast<int>(levels.size()); ++i) {
      const MultigridParams build = multigrid_params(i - 1, i, level_cells(i));
      compute.dispatch(i == 0 ? mg_build0 : mg_build,
                       &build,
                       sizeof(build),
                       {&table,
                        &coords,
                        i == 0 ? &links : &levels[static_cast<std::size_t>(i - 1)].op,
                        &levels[static_cast<std::size_t>(i)].op,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr},
                       groups_for(level_cells(i)),
                       "mg build");
    }
  }

  void collect_profile()
  {
    for (const auto &[label, ms] : compute.last_profile()) {
      auto entry = std::find_if(timing.kernels.begin(),
                                timing.kernels.end(),
                                [&](const auto &item) { return item.first == label; });
      if (entry == timing.kernels.end()) {
        timing.kernels.emplace_back(label, ms);
      }
      else {
        entry->second += ms;
      }
    }
  }

  Params params(const float a0, const float a1, const float a2, const float a3) const
  {
    return {{topology.extent.x, topology.extent.y, topology.extent.z, topology.brick_size},
            {topology.brick_extent.x,
             topology.brick_extent.y,
             topology.brick_extent.z,
             static_cast<std::int32_t>(topology.voxel_count())},
            {a0, a1, a2, a3}};
  }

  /** Dispatch a stencil kernel with the topology tables at bindings 1-3 and channels after. */
  void stencil(const int kernel,
               const char *label,
               const Params &p,
               const int c4,
               const int c5 = -1,
               const int c6 = -1,
               const int c7 = -1,
               const int c8 = -1,
               const int c9 = -1)
  {
    const auto bind = [&](const int logical) { return logical < 0 ? nullptr : &channel(logical); };
    compute.dispatch(kernel,
                     &p,
                     sizeof(p),
                     {&table, &coords, &types, bind(c4), bind(c5), bind(c6), bind(c7), bind(c8),
                      bind(c9)},
                     groups_for(topology.voxel_count()),
                     label);
  }
};

SparseGasGpu::SparseGasGpu() : impl_(std::make_unique<Impl>()) {}
SparseGasGpu::~SparseGasGpu() = default;

const std::string &SparseGasGpu::device_name() const
{
  return impl_->compute.device_name();
}

bool SparseGasGpu::is_software_device() const
{
  return impl_->compute.is_software_device();
}

bool SparseGasGpu::has_timestamps() const
{
  return impl_->compute.has_timestamps();
}

void SparseGasGpu::set_single_dispatch_cell_limit(const std::size_t cells)
{
  impl_->single_dispatch_cell_limit = cells;
}

void SparseGasGpu::set_profiling(const bool enabled)
{
  impl_->compute.set_profiling(enabled);
}

const SparseGasGpu::Timing &SparseGasGpu::last_timing() const
{
  return impl_->timing;
}

void SparseGasGpu::upload(const sparse::State &state)
{
  Impl &g = *impl_;
  const std::size_t count = state.topology.voxel_count();
  g.reserve_topology(state.topology);
  if (count > g.voxel_capacity || g.buffers.empty()) {
    for (Compute::Buffer &buffer : g.buffers) {
      g.compute.destroy_buffer(buffer);
    }
    g.voxel_capacity = count + count / 2;
    g.buffers.assign(AllChannels, {});
    for (Compute::Buffer &buffer : g.buffers) {
      buffer = g.compute.create_buffer(g.voxel_capacity * sizeof(float));
    }
  }
  for (int i = 0; i < AllChannels; ++i) {
    g.role[static_cast<std::size_t>(i)] = i;
  }
  g.topology = state.topology;
  g.topology.cell_type.clear();

  g.compute.begin();
  g.record_topology_upload();
  const std::vector<float> *channels[StateChannels] = {
      &state.density, &state.temperature, &state.u, &state.v, &state.w, &state.pressure};
  for (int i = 0; i < StateChannels; ++i) {
    g.compute.upload(g.channel(i), channels[i]->data(), count * sizeof(float));
  }
  g.compute.submit();
}

void SparseGasGpu::download(sparse::State &state)
{
  Impl &g = *impl_;
  const std::size_t count = g.topology.voxel_count();
  state.topology = g.topology;
  std::vector<float> *channels[StateChannels] = {
      &state.density, &state.temperature, &state.u, &state.v, &state.w, &state.pressure};
  for (std::vector<float> *values : channels) {
    values->assign(count, 0.0f);
  }
  if (count == 0) {
    return;
  }
  state.topology.cell_type.assign(count, 0);
  g.compute.begin();
  g.compute.download(g.types, state.topology.cell_type.data(), count * sizeof(std::uint32_t));
  for (int i = 0; i < StateChannels; ++i) {
    g.compute.download(g.channel(i), channels[i]->data(), count * sizeof(float));
  }
  g.compute.submit();
}

sparse::StepInfo SparseGasGpu::step(const sparse::Settings &settings)
{
  return impl_step(settings, nullptr);
}

sparse::FrameInfo SparseGasGpu::advance(const sparse::Settings &settings, const float frame_time)
{
  sparse::FrameInfo frame;
  Timing total;
  total.gpu_ms = 0.0;
  float remaining = frame_time;
  while (remaining > 0.0f) {
    if (frame.substeps >= settings.max_substeps) {
      throw std::runtime_error("The sparse solver exhausted its substep limit");
    }
    frame.last_step = impl_step(settings, &remaining);
    frame.max_speed = std::max(frame.max_speed, impl_->last_speed);
    remaining -= frame.last_step.dt;
    ++frame.substeps;

    const Timing &timing = impl_->timing;
    total.gpu_ms = timing.gpu_ms < 0.0 || total.gpu_ms < 0.0 ? -1.0 : total.gpu_ms + timing.gpu_ms;
    total.wall_ms += timing.wall_ms;
    total.topology_ms += timing.topology_ms;
    total.topology_changed = total.topology_changed || timing.topology_changed;
    for (const auto &[label, ms] : timing.kernels) {
      auto entry = std::find_if(total.kernels.begin(), total.kernels.end(),
                                [&](const auto &item) { return item.first == label; });
      if (entry == total.kernels.end()) {
        total.kernels.emplace_back(label, ms);
      }
      else {
        entry->second += ms;
      }
    }
  }
  impl_->timing = total;
  return frame;
}

sparse::StepInfo SparseGasGpu::impl_step(const sparse::Settings &settings, const float *remaining)
{
  if (settings.pressure_solver == sparse::PressureSolver::ConjugateGradient ||
      settings.pressure_iterations < 1)
  {
    throw std::invalid_argument("The GPU path implements the Jacobi and multigrid solvers only");
  }
  Impl &g = *impl_;
  const auto start = std::chrono::steady_clock::now();
  g.timing = {};
  double gpu_ms = 0.0;
  bool gpu_timed = g.compute.has_timestamps();
  const auto add_gpu_time = [&](const double ms) {
    if (ms < 0.0) {
      gpu_timed = false;
    }
    gpu_ms += ms;
  };

  /* 1. Which bricks hold content. Only the flags leave the device. */
  const sparse::Topology &old_topology = g.topology;
  const std::size_t old_slots = old_topology.brick_count();
  std::vector<std::uint32_t> slot_flags(old_slots, 0);
  std::vector<float> slot_speeds(old_slots, 0.0f);
  if (old_slots > 0) {
    const ActivityParams params = {static_cast<std::int32_t>(old_topology.voxels_per_brick()),
                                   static_cast<std::int32_t>(old_topology.voxel_count()),
                                   settings.thresholds.scalar,
                                   settings.thresholds.velocity};
    g.compute.begin();
    g.compute.fill(g.flags, 0);
    g.compute.fill(g.speeds, 0);
    g.compute.dispatch(g.activity,
                       &params,
                       sizeof(params),
                       {&g.channel(Density),
                        &g.channel(Temperature),
                        &g.channel(U),
                        &g.channel(V),
                        &g.channel(W),
                        &g.flags,
                        &g.speeds},
                       groups_for(old_topology.voxel_count()),
                       "activity");
    g.compute.download(g.flags, slot_flags.data(), old_slots * sizeof(std::uint32_t));
    /* The speeds were accumulated as bit patterns; they are floats. */
    g.compute.download(g.speeds, slot_speeds.data(), old_slots * sizeof(float));
    add_gpu_time(g.compute.submit());
    g.collect_profile();
  }

  g.last_speed = 0.0f;
  for (const float speed : slot_speeds) {
    g.last_speed = std::max(g.last_speed, speed);
  }
  const float dt = remaining ? sparse::cfl_time_step(*remaining, settings.cfl, g.last_speed) :
                               settings.dt;

  /* 2. The next topology, built on the CPU exactly as the reference does. */
  const auto topology_start = std::chrono::steady_clock::now();
  std::vector<std::uint8_t> brick_flags(old_topology.brick_table.size(), 0);
  for (std::size_t slot = 0; slot < old_slots; ++slot) {
    if (slot_flags[slot]) {
      brick_flags[static_cast<std::size_t>(old_topology.brick_coords[slot * 4 + 3])] = 1;
    }
  }
  sparse::Topology next = sparse::build_topology(
      old_topology.extent, old_topology.brick_size, brick_flags, settings.dilation, false);
  const bool changed = next.brick_table != old_topology.brick_table;
  std::vector<std::int32_t> old_slot_of;
  if (changed) {
    old_slot_of.resize(next.brick_count());
    for (std::size_t slot = 0; slot < next.brick_count(); ++slot) {
      old_slot_of[slot] = old_topology.brick_table[static_cast<std::size_t>(
          next.brick_coords[slot * 4 + 3])];
    }
  }
  g.timing.topology_ms = milliseconds_since(topology_start);
  g.timing.topology_changed = changed;

  /* Buffers that must outlive the recording they are read in. */
  std::vector<Compute::Buffer> retired;
  const std::size_t count = next.voxel_count();
  if (changed) {
    g.reserve_topology(next);
    if (count > g.voxel_capacity) {
      /* Fresh buffers for every channel; the old ones are read by the remap, then freed. */
      std::vector<Compute::Buffer> grown(AllChannels);
      const std::size_t capacity = count + count / 2;
      for (Compute::Buffer &buffer : grown) {
        buffer = g.compute.create_buffer(capacity * sizeof(float));
      }
      retired = std::move(g.buffers);
      g.buffers.resize(AllChannels);
      std::array<int, AllChannels> retired_role = g.role;
      /* Keep the old state buffers addressable by appending them after the new ones. */
      g.buffers = std::move(grown);
      for (std::size_t i = 0; i < retired.size(); ++i) {
        g.buffers.push_back(retired[i]);
      }
      for (int i = 0; i < AllChannels; ++i) {
        g.role[static_cast<std::size_t>(i)] = AllChannels + retired_role[static_cast<std::size_t>(i)];
      }
      g.voxel_capacity = capacity;
    }
  }

  sparse::StepInfo info;
  info.active_bricks = next.brick_count();
  info.max_divergence = std::numeric_limits<float>::quiet_NaN();

  /* 3. Move onto the new topology and run the step, in one submission. */
  g.compute.begin();
  if (changed) {
    g.compute.upload(g.slot_map, old_slot_of.data(), old_slot_of.size() * sizeof(std::int32_t));
    const RemapParams params = {static_cast<std::int32_t>(next.voxels_per_brick()),
                                static_cast<std::int32_t>(count)};
    /* Targets: the scratch roles if the buffers were kept, the new buffers if they grew. */
    const bool grew = !retired.empty();
    for (int i = 0; i < StateChannels; ++i) {
      const int target = grew ? i : g.role[static_cast<std::size_t>(StateChannels + i)];
      g.compute.dispatch(g.remap,
                         &params,
                         sizeof(params),
                         {&g.slot_map,
                          &g.channel(i),
                          &g.buffers[static_cast<std::size_t>(target)],
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr},
                         groups_for(count),
                         "remap");
    }
    if (grew) {
      for (int i = 0; i < AllChannels; ++i) {
        g.role[static_cast<std::size_t>(i)] = i;
      }
    }
    else {
      for (int i = 0; i < StateChannels; ++i) {
        std::swap(g.role[static_cast<std::size_t>(i)],
                  g.role[static_cast<std::size_t>(StateChannels + i)]);
      }
    }
    g.topology = std::move(next);
    g.record_topology_upload();
  }

  if (count > 0) {
    const float offsets[5][3] = {{0.5f, 0.5f, 0.5f},
                                 {0.5f, 0.5f, 0.5f},
                                 {0.0f, 0.5f, 0.5f},
                                 {0.5f, 0.0f, 0.5f},
                                 {0.5f, 0.5f, 0.0f}};
    /* Every channel is advected by the velocity from the start of the step. */
    const bool second_order = settings.advection == sparse::Advection::MacCormack;
    const int forward = StateChannels + 5;
    for (int i = 0; i < 5; ++i) {
      const Params p = g.params(offsets[i][0], offsets[i][1], offsets[i][2], dt);
      g.stencil(g.advect, "advect", p, U, V, W, i, second_order ? forward : StateChannels + i);
      if (second_order) {
        g.stencil(g.maccormack, "maccormack", p, U, V, W, i, forward, StateChannels + i);
      }
    }
    for (int i = 0; i < 5; ++i) {
      std::swap(g.role[static_cast<std::size_t>(i)],
                g.role[static_cast<std::size_t>(StateChannels + i)]);
    }

    const auto buoyancy = [&](const int factor, const float coefficient) {
      if (coefficient != 0.0f) {
        g.stencil(g.buoyancy,
                "buoyancy",
                  g.params(-settings.gravity[0] * dt * coefficient,
                           -settings.gravity[1] * dt * coefficient,
                           -settings.gravity[2] * dt * coefficient,
                           0.0f),
                  factor,
                  U,
                  V,
                  W);
      }
    };
    buoyancy(Temperature, settings.heat_buoyancy);
    buoyancy(Density, settings.density_buoyancy);
    g.stencil(g.walls, "walls", g.params(0, 0, 0, 0), U, V, W);

    const int rhs = StateChannels, next_pressure = StateChannels + 1;
    g.stencil(g.divergence, "divergence", g.params(-1.0f, 0, 0, 0), U, V, W, rhs);
    if (settings.pressure_solver == sparse::PressureSolver::Multigrid) {
      g.stencil(g.mask_pressure, "mask pressure", g.params(0, 0, 0, 0), Pressure);
      g.record_multigrid(settings, rhs, next_pressure);
    }
    else {
      for (int i = 0; i < settings.pressure_iterations; ++i) {
        g.stencil(g.jacobi, "jacobi", g.params(0, 0, 0, 0), rhs, Pressure, next_pressure);
        std::swap(g.role[Pressure], g.role[static_cast<std::size_t>(next_pressure)]);
      }
    }
    g.stencil(g.gradient, "gradient", g.params(0, 0, 0, 0), Pressure, U, V, W);
  }
  add_gpu_time(g.compute.submit());
  g.collect_profile();

  if (!retired.empty()) {
    for (Compute::Buffer &buffer : retired) {
      g.compute.destroy_buffer(buffer);
    }
    g.buffers.resize(AllChannels);
  }

  info.pressure_iterations = count > 0 ? settings.pressure_iterations : 0;
  info.dt = dt;
  g.timing.gpu_ms = gpu_timed ? gpu_ms : -1.0;
  g.timing.wall_ms = milliseconds_since(start);
  return info;
}

}  // namespace manta_improve::gpu
