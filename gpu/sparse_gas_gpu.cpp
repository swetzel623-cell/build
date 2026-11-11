// SPDX-License-Identifier: MIT
#include "sparse_gas_gpu.hpp"

#include "vulkan_compute.hpp"

#include "advect.comp.h"
#include "buoyancy.comp.h"
#include "divergence.comp.h"
#include "gradient.comp.h"
#include "jacobi.comp.h"
#include "walls.comp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace manta_improve::gpu {
namespace {

struct Params {
  std::int32_t extent[4];
  std::int32_t bricks[4];
  float a[4];
};

template<typename T> void upload(const Compute::Buffer &buffer, const std::vector<T> &values)
{
  if (!values.empty()) {
    std::memcpy(buffer.mapped, values.data(), values.size() * sizeof(T));
  }
}

void download(const Compute::Buffer &buffer, std::vector<float> &values)
{
  if (!values.empty()) {
    std::memcpy(values.data(), buffer.mapped, values.size() * sizeof(float));
  }
}

}  // namespace

struct SparseGasGpu::Impl {
  Compute compute;
  int advect, buoyancy, walls, divergence, jacobi, gradient;

  Compute::Buffer params, table, coords, types;
  /* density, temperature, u, v, w, pressure, then five scratch channels. */
  std::array<Compute::Buffer, 11> channels;
  std::size_t table_capacity = 0, voxel_capacity = 0;

  Impl()
  {
#define KERNEL(name) compute.create_kernel(name##_spv, sizeof(name##_spv) / sizeof(std::uint32_t))
    advect = KERNEL(advect);
    buoyancy = KERNEL(buoyancy);
    walls = KERNEL(walls);
    divergence = KERNEL(divergence);
    jacobi = KERNEL(jacobi);
    gradient = KERNEL(gradient);
#undef KERNEL
    params = compute.create_buffer(sizeof(Params));
  }

  ~Impl()
  {
    release();
    compute.destroy_buffer(params);
  }

  void release()
  {
    compute.destroy_buffer(table);
    compute.destroy_buffer(coords);
    compute.destroy_buffer(types);
    for (Compute::Buffer &channel : channels) {
      compute.destroy_buffer(channel);
    }
  }

  void reserve(const sparse::Topology &topology)
  {
    if (topology.brick_table.size() <= table_capacity && topology.voxel_count() <= voxel_capacity)
    {
      return;
    }
    release();
    table_capacity = topology.brick_table.size();
    voxel_capacity = topology.voxel_count();
    table = compute.create_buffer(table_capacity * sizeof(std::int32_t));
    coords = compute.create_buffer(topology.brick_count() * 4 * sizeof(std::int32_t));
    types = compute.create_buffer(voxel_capacity * sizeof(std::uint32_t));
    for (Compute::Buffer &channel : channels) {
      channel = compute.create_buffer(voxel_capacity * sizeof(float));
    }
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

sparse::StepInfo SparseGasGpu::step(sparse::State &state, const sparse::Settings &settings)
{
  if (settings.pressure_solver != sparse::PressureSolver::Jacobi) {
    throw std::invalid_argument("The GPU path only implements the Jacobi pressure solver");
  }
  Impl &g = *impl_;

  sparse::retopologize(state, settings.thresholds, settings.dilation);
  const sparse::Topology &topology = state.topology;
  sparse::StepInfo info;
  info.active_bricks = topology.brick_count();
  const std::size_t count = topology.voxel_count();
  if (count == 0) {
    return info;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (topology.cell_type[i] != sparse::CellFluid) {
      state.pressure[i] = 0.0f;
    }
  }

  g.reserve(topology);
  upload(g.table, topology.brick_table);
  upload(g.coords, topology.brick_coords);
  upload(g.types, topology.cell_type);
  enum { Density, Temperature, U, V, W, Pressure, Scratch };
  std::array<Compute::Buffer *, 11> c;
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = &g.channels[i];
  }
  upload(*c[Density], state.density);
  upload(*c[Temperature], state.temperature);
  upload(*c[U], state.u);
  upload(*c[V], state.v);
  upload(*c[W], state.w);
  upload(*c[Pressure], state.pressure);

  const std::uint32_t groups = static_cast<std::uint32_t>((count + 63) / 64);
  const auto run = [&](const int kernel,
                       const float a0,
                       const float a1,
                       const float a2,
                       const float a3,
                       const Compute::Buffer *b4,
                       const Compute::Buffer *b5,
                       const Compute::Buffer *b6,
                       const Compute::Buffer *b7 = nullptr,
                       const Compute::Buffer *b8 = nullptr) {
    const Params params = {
        {topology.extent.x, topology.extent.y, topology.extent.z, topology.brick_size},
        {topology.brick_extent.x, topology.brick_extent.y, topology.brick_extent.z,
         static_cast<std::int32_t>(count)},
        {a0, a1, a2, a3}};
    std::memcpy(g.params.mapped, &params, sizeof(params));
    g.compute.dispatch(kernel, {&g.params, &g.table, &g.coords, &g.types, b4, b5, b6, b7, b8},
                       groups);
  };

  /* Advect every channel with the velocity from the start of the step, then swap. */
  const float dt = settings.dt;
  const float offsets[5][3] = {
      {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.0f, 0.5f, 0.5f}, {0.5f, 0.0f, 0.5f},
      {0.5f, 0.5f, 0.0f}};
  for (int i = 0; i < 5; ++i) {
    run(g.advect, offsets[i][0], offsets[i][1], offsets[i][2], dt, c[U], c[V], c[W], c[i],
        c[Scratch + i]);
  }
  for (int i = 0; i < 5; ++i) {
    std::swap(c[i], c[Scratch + i]);
  }

  const auto buoyancy = [&](const Compute::Buffer *factor, const float coefficient) {
    if (coefficient != 0.0f) {
      run(g.buoyancy, -settings.gravity[0] * dt * coefficient,
          -settings.gravity[1] * dt * coefficient, -settings.gravity[2] * dt * coefficient, 0.0f,
          factor, c[U], c[V], c[W]);
    }
  };
  buoyancy(c[Temperature], settings.heat_buoyancy);
  buoyancy(c[Density], settings.density_buoyancy);
  run(g.walls, 0.0f, 0.0f, 0.0f, 0.0f, c[U], c[V], c[W]);

  Compute::Buffer *rhs = c[Scratch], *next = c[Scratch + 1];
  run(g.divergence, -1.0f, 0.0f, 0.0f, 0.0f, c[U], c[V], c[W], rhs);
  for (int i = 0; i < settings.pressure_iterations; ++i) {
    run(g.jacobi, 0.0f, 0.0f, 0.0f, 0.0f, rhs, c[Pressure], next);
    std::swap(c[Pressure], next);
  }
  run(g.gradient, 0.0f, 0.0f, 0.0f, 0.0f, c[Pressure], c[U], c[V], c[W]);

  download(*c[Density], state.density);
  download(*c[Temperature], state.temperature);
  download(*c[U], state.u);
  download(*c[V], state.v);
  download(*c[W], state.w);
  download(*c[Pressure], state.pressure);

  std::vector<float> remaining;
  sparse::divergence(state, remaining);
  for (const float value : remaining) {
    info.max_divergence = std::max(info.max_divergence, std::abs(value));
  }
  info.pressure_iterations = settings.pressure_iterations;
  return info;
}

}  // namespace manta_improve::gpu
