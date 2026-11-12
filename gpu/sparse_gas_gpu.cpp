// SPDX-License-Identifier: MIT
#include "sparse_gas_gpu.hpp"

#include "vulkan_compute.hpp"

#include "activity.comp.h"
#include "advect.comp.h"
#include "buoyancy.comp.h"
#include "divergence.comp.h"
#include "gradient.comp.h"
#include "jacobi.comp.h"
#include "remap.comp.h"
#include "walls.comp.h"

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
  int advect, buoyancy, walls, divergence, jacobi, gradient, activity, remap;

  sparse::Topology topology;
  Compute::Buffer table, coords, types, slot_map, flags;
  std::size_t slot_capacity = 0;
  /** Physical buffers; `role[channel]` says which one currently plays each channel. */
  std::vector<Compute::Buffer> buffers;
  std::array<int, AllChannels> role{};
  std::size_t voxel_capacity = 0;
  Timing timing;

  Impl()
  {
#define KERNEL(name) compute.create_kernel(name##_spv, sizeof(name##_spv) / sizeof(std::uint32_t))
    advect = KERNEL(advect);
    buoyancy = KERNEL(buoyancy);
    walls = KERNEL(walls);
    divergence = KERNEL(divergence);
    jacobi = KERNEL(jacobi);
    gradient = KERNEL(gradient);
    activity = KERNEL(activity);
    remap = KERNEL(remap);
#undef KERNEL
  }

  ~Impl()
  {
    for (Compute::Buffer *buffer : {&table, &coords, &types, &slot_map, &flags}) {
      compute.destroy_buffer(*buffer);
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

  /** Make room for a topology's tables. Must be called outside recording. */
  void reserve_topology(const sparse::Topology &next)
  {
    ensure(table, next.brick_table.size() * sizeof(std::int32_t));
    if (next.brick_count() > slot_capacity || !coords.handle) {
      slot_capacity = next.brick_count() + next.brick_count() / 2 + 1;
      ensure(coords, slot_capacity * 4 * sizeof(std::int32_t));
      ensure(slot_map, slot_capacity * sizeof(std::int32_t));
      ensure(flags, slot_capacity * sizeof(std::uint32_t));
    }
    ensure(types, next.voxel_count() * sizeof(std::uint32_t));
  }

  void record_topology_upload(const sparse::Topology &next)
  {
    compute.upload(table, next.brick_table.data(), next.brick_table.size() * sizeof(std::int32_t));
    compute.upload(coords, next.brick_coords.data(), next.brick_coords.size() * sizeof(std::int32_t));
    compute.upload(types, next.cell_type.data(), next.cell_type.size() * sizeof(std::uint32_t));
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
               const Params &p,
               const int c4,
               const int c5 = -1,
               const int c6 = -1,
               const int c7 = -1,
               const int c8 = -1)
  {
    const auto bind = [&](const int logical) { return logical < 0 ? nullptr : &channel(logical); };
    compute.dispatch(kernel,
                     &p,
                     sizeof(p),
                     {&table, &coords, &types, bind(c4), bind(c5), bind(c6), bind(c7), bind(c8)},
                     groups_for(topology.voxel_count()));
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

  g.compute.begin();
  g.record_topology_upload(g.topology);
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
  g.compute.begin();
  for (int i = 0; i < StateChannels; ++i) {
    g.compute.download(g.channel(i), channels[i]->data(), count * sizeof(float));
  }
  g.compute.submit();
}

sparse::StepInfo SparseGasGpu::step(const sparse::Settings &settings)
{
  if (settings.pressure_solver != sparse::PressureSolver::Jacobi ||
      settings.pressure_iterations < 1)
  {
    throw std::invalid_argument("The GPU path implements the Jacobi pressure solver only");
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
  if (old_slots > 0) {
    const ActivityParams params = {static_cast<std::int32_t>(old_topology.voxels_per_brick()),
                                   static_cast<std::int32_t>(old_topology.voxel_count()),
                                   settings.thresholds.scalar,
                                   settings.thresholds.velocity};
    g.compute.begin();
    g.compute.fill(g.flags, 0);
    g.compute.dispatch(g.activity,
                       &params,
                       sizeof(params),
                       {&g.channel(Density),
                        &g.channel(Temperature),
                        &g.channel(U),
                        &g.channel(V),
                        &g.channel(W),
                        &g.flags,
                        nullptr,
                        nullptr},
                       groups_for(old_topology.voxel_count()));
    g.compute.download(g.flags, slot_flags.data(), old_slots * sizeof(std::uint32_t));
    add_gpu_time(g.compute.submit());
  }

  /* 2. The next topology, built on the CPU exactly as the reference does. */
  const auto topology_start = std::chrono::steady_clock::now();
  std::vector<std::uint8_t> brick_flags(old_topology.brick_table.size(), 0);
  for (std::size_t slot = 0; slot < old_slots; ++slot) {
    if (slot_flags[slot]) {
      brick_flags[static_cast<std::size_t>(old_topology.brick_coords[slot * 4 + 3])] = 1;
    }
  }
  sparse::Topology next = sparse::build_topology(
      old_topology.extent, old_topology.brick_size, brick_flags, settings.dilation);
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
    g.record_topology_upload(next);
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
                         groups_for(count));
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
  }

  if (count > 0) {
    const float dt = settings.dt;
    const float offsets[5][3] = {{0.5f, 0.5f, 0.5f},
                                 {0.5f, 0.5f, 0.5f},
                                 {0.0f, 0.5f, 0.5f},
                                 {0.5f, 0.0f, 0.5f},
                                 {0.5f, 0.5f, 0.0f}};
    /* Every channel is advected by the velocity from the start of the step. */
    for (int i = 0; i < 5; ++i) {
      g.stencil(g.advect,
                g.params(offsets[i][0], offsets[i][1], offsets[i][2], dt),
                U,
                V,
                W,
                i,
                StateChannels + i);
    }
    for (int i = 0; i < 5; ++i) {
      std::swap(g.role[static_cast<std::size_t>(i)],
                g.role[static_cast<std::size_t>(StateChannels + i)]);
    }

    const auto buoyancy = [&](const int factor, const float coefficient) {
      if (coefficient != 0.0f) {
        g.stencil(g.buoyancy,
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
    g.stencil(g.walls, g.params(0, 0, 0, 0), U, V, W);

    const int rhs = StateChannels, next_pressure = StateChannels + 1;
    g.stencil(g.divergence, g.params(-1.0f, 0, 0, 0), U, V, W, rhs);
    for (int i = 0; i < settings.pressure_iterations; ++i) {
      g.stencil(g.jacobi, g.params(0, 0, 0, 0), rhs, Pressure, next_pressure);
      std::swap(g.role[Pressure], g.role[static_cast<std::size_t>(next_pressure)]);
    }
    g.stencil(g.gradient, g.params(0, 0, 0, 0), Pressure, U, V, W);
  }
  add_gpu_time(g.compute.submit());

  if (!retired.empty()) {
    for (Compute::Buffer &buffer : retired) {
      g.compute.destroy_buffer(buffer);
    }
    g.buffers.resize(AllChannels);
  }

  info.pressure_iterations = count > 0 ? settings.pressure_iterations : 0;
  g.timing.gpu_ms = gpu_timed ? gpu_ms : -1.0;
  g.timing.wall_ms = milliseconds_since(start);
  return info;
}

}  // namespace manta_improve::gpu
