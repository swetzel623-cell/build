// SPDX-License-Identifier: MIT
// Runs the sparse solver's compute kernels on a Vulkan device and compares every channel with the
// CPU reference. Exits with 77 (skipped) when no Vulkan device is available.
#include "manta_improve/sparse_gas.hpp"
#include "sparse_gas_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace manta_improve;

/** Put a plate above the plume, as an obstacle. */
bool with_obstacle = false;
/**
 * Start every frame of the comparison from the CPU reference's state. A real GPU rounds
 * differently from the CPU, and a turbulent flow amplifies that over many frames; with this
 * option each frame's kernels are still checked against the reference.
 */
bool resync = false;

sparse::DenseFields blob(const sparse::Extent extent)
{
  sparse::DenseFields dense;
  dense.extent = extent;
  const std::size_t count = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  dense.density.assign(count, 0.0f);
  dense.temperature.assign(count, 0.0f);
  const float cx = extent.x / 2.0f, cy = extent.y / 2.0f, cz = extent.z / 4.0f;
  const float radius = extent.x / 8.0f;
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const float dx = x + 0.5f - cx, dy = y + 0.5f - cy, dz = z + 0.5f - cz;
        if (dx * dx + dy * dy + dz * dz <= radius * radius) {
          const std::size_t i = x + static_cast<std::size_t>(extent.x) * (y + extent.y * z);
          dense.density[i] = 1.0f;
          dense.temperature[i] = 1.0f;
        }
      }
    }
  }
  if (with_obstacle) {
    /* A square plate a little above the blob, wider than it. */
    const int bottom = static_cast<int>(0.46f * extent.z), top = bottom + std::max(3, extent.x / 16);
    const int half = extent.x * 7 / 32;
    dense.obstacle.assign(count, 0);
    for (int z = bottom; z < top; ++z) {
      for (int y = extent.y / 2 - half; y < extent.y / 2 + half; ++y) {
        for (int x = extent.x / 2 - half; x < extent.x / 2 + half; ++x) {
          dense.obstacle[x + static_cast<std::size_t>(extent.x) * (y + extent.y * z)] = 1;
        }
      }
    }
  }
  return dense;
}

float max_abs(const std::vector<float> &values)
{
  float result = 0.0f;
  for (const float value : values) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

float max_difference(const std::vector<float> &a, const std::vector<float> &b)
{
  if (a.size() != b.size()) {
    throw std::runtime_error("channel sizes differ");
  }
  float result = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(a[i] - b[i]));
  }
  return result;
}

double seconds_since(const std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

/** Write a grayscale image, `width` by `height` values in 0..1 with row 0 at the bottom. */
void write_bmp(const std::string &path, const std::vector<float> &values, const int width,
               const int height)
{
  const int scale = 2;
  const int w = width * scale, h = height * scale;
  const int row_bytes = (3 * w + 3) / 4 * 4;
  const std::uint32_t data_bytes = static_cast<std::uint32_t>(row_bytes) * h;
  std::vector<unsigned char> file(54 + data_bytes, 0);
  const auto put32 = [&](const int at, const std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
      file[static_cast<std::size_t>(at + i)] = static_cast<unsigned char>(value >> (8 * i));
    }
  };
  file[0] = 'B';
  file[1] = 'M';
  put32(2, 54 + data_bytes);
  put32(10, 54);
  put32(14, 40);
  put32(18, static_cast<std::uint32_t>(w));
  put32(22, static_cast<std::uint32_t>(h));
  file[26] = 1;
  file[28] = 24;
  put32(34, data_bytes);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float value = values[static_cast<std::size_t>(x / scale) +
                                 static_cast<std::size_t>(width) * (y / scale)];
      /* A gamma below one keeps thin smoke visible. */
      const float shade = std::pow(std::min(1.0f, std::max(0.0f, value)), 0.6f);
      const unsigned char level = static_cast<unsigned char>(255.0f * shade + 0.5f);
      const std::size_t at = 54 + static_cast<std::size_t>(y) * row_bytes + 3 * x;
      file[at] = file[at + 1] = file[at + 2] = level;
    }
  }
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
  if (!out) {
    throw std::runtime_error("could not write " + path);
  }
}

/** Write the centre slice and a side view of the density, and print the plume's measures. */
void write_plume(const sparse::State &state, const std::string &name)
{
  const sparse::DenseFields dense = sparse::to_dense(state);
  const sparse::Extent &e = dense.extent;
  std::vector<float> slice(static_cast<std::size_t>(e.x) * e.z, 0.0f);
  std::vector<float> side(slice.size(), 0.0f);
  double mass = 0.0, moment = 0.0, spread = 0.0;
  for (int z = 0; z < e.z; ++z) {
    for (int y = 0; y < e.y; ++y) {
      for (int x = 0; x < e.x; ++x) {
        const float value = dense.density[static_cast<std::size_t>(x) +
                                          static_cast<std::size_t>(e.x) *
                                              (y + static_cast<std::size_t>(e.y) * z)];
        const std::size_t pixel = static_cast<std::size_t>(x) + static_cast<std::size_t>(e.x) * z;
        side[pixel] += value;
        if (y == e.y / 2) {
          slice[pixel] = value;
        }
        const double dx = x + 0.5 - 0.5 * e.x, dy = y + 0.5 - 0.5 * e.y;
        mass += value;
        moment += static_cast<double>(value) * z;
        spread += static_cast<double>(value) * (dx * dx + dy * dy);
      }
    }
  }
  /* The side view adds the density along the line of sight; a fixed scale keeps runs comparable. */
  for (float &value : side) {
    value /= 0.2f * static_cast<float>(e.y);
  }
  write_bmp(name + "_slice.bmp", slice, e.x, e.z);
  write_bmp(name + "_side.bmp", side, e.x, e.z);
  std::cout << "Plume: mass " << mass << ", height " << (mass > 0.0 ? moment / mass : 0.0)
            << ", width (rms radius) " << (mass > 0.0 ? std::sqrt(spread / mass) : 0.0) << '\n';
  std::cout << "Wrote " << name << "_slice.bmp and " << name << "_side.bmp\n";
}

/** Largest divergence left in fluid cells, relative to the largest velocity. */
float relative_divergence(const sparse::State &state)
{
  std::vector<float> values;
  sparse::divergence(state, values);
  return max_abs(values) / std::max(1.0e-6f, max_abs(state.w));
}

double median(std::vector<double> values)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

bool use_jacobi = false;
bool first_order = false;
float cfl = -1.0f;
constexpr float frame_time = 1.0f / 30.0f;
int multigrid_w_depth = -1;
int coarse_pre_sweeps = -1, coarse_post_sweeps = -1;
int midpoint = -1;
int reflection = -1;
std::string image_name;

sparse::Settings test_settings(const int resolution)
{
  sparse::Settings settings;
  settings.gravity[2] = -9.81f * resolution / 2.0f;
  settings.heat_buoyancy = 0.5f;
  settings.density_buoyancy = 0.5f;
  if (cfl > 0.0f) {
    settings.cfl = cfl;
  }
  if (reflection >= 0) {
    settings.reflection = reflection != 0;
  }
  if (midpoint >= 0) {
    settings.midpoint_trace = midpoint != 0;
  }
  if (first_order) {
    settings.advection = sparse::Advection::SemiLagrangian;
  }
  if (use_jacobi) {
    settings.pressure_solver = sparse::PressureSolver::Jacobi;
    settings.pressure_iterations = 60;
  }
  else {
    settings.pressure_solver = sparse::PressureSolver::Multigrid;
    settings.pressure_iterations = 6;
    if (multigrid_w_depth >= 0) {
      settings.multigrid_w_depth = multigrid_w_depth;
    }
    if (coarse_pre_sweeps >= 0) {
      settings.multigrid_coarse_pre_sweeps = coarse_pre_sweeps;
      settings.multigrid_coarse_post_sweeps = coarse_post_sweeps;
    }
  }
  return settings;
}

/** Step the device and the CPU reference side by side and compare every channel each step. */
int compare(gpu::SparseGasGpu &solver, const int resolution, const int steps)
{
  const sparse::Settings settings = test_settings(resolution);
  const sparse::Extent extent = {resolution, resolution, resolution * 3 / 2};
  sparse::State cpu = sparse::from_dense(blob(extent), 8, settings.thresholds, settings.dilation);
  solver.upload(cpu);
  sparse::State device;

  double cpu_seconds = 0.0, device_seconds = 0.0;
  for (int i = 0; i < steps; ++i) {
    std::cout << "step " << i << ": running...\n";
    auto start = std::chrono::steady_clock::now();
    const sparse::FrameInfo cpu_frame = sparse::advance(cpu, settings, frame_time);
    cpu_seconds += seconds_since(start);
    start = std::chrono::steady_clock::now();
    const sparse::FrameInfo frame = solver.advance(settings, frame_time);
    const sparse::StepInfo &info = frame.last_step;
    device_seconds += seconds_since(start);
    if (cpu_frame.substeps != frame.substeps) {
      std::cout << "FAIL: substep counts differ at step " << i << ": CPU " << cpu_frame.substeps
                << ", device " << frame.substeps << '\n';
      return 1;
    }
    solver.download(device);

    if (cpu.topology.brick_table != device.topology.brick_table) {
      std::cout << "FAIL: topologies diverged at step " << i << '\n';
      return 1;
    }
    if (cpu.topology.cell_type != device.topology.cell_type) {
      std::cout << "FAIL: GPU cell classification differs from the CPU at step " << i << '\n';
      return 1;
    }
    const float scale = std::max(1.0f, max_abs(cpu.w));
    const float worst = std::max({max_difference(cpu.density, device.density),
                                  max_difference(cpu.temperature, device.temperature),
                                  max_difference(cpu.u, device.u) / scale,
                                  max_difference(cpu.v, device.v) / scale,
                                  max_difference(cpu.w, device.w) / scale,
                                  max_difference(cpu.pressure, device.pressure) / scale});
    std::cout << "step " << i << ": " << frame.substeps << " substeps, " << info.active_bricks
              << " bricks, peak |w| " << max_abs(cpu.w)
              << ", divergence left " << relative_divergence(device)
              << ", largest CPU/device difference " << worst << '\n';
    if (!(worst < 1.0e-4f)) {
      std::cout << "FAIL: device result differs from the CPU reference\n";
      return 1;
    }
    if (resync) {
      solver.upload(cpu);
    }
  }
  if (!(max_abs(cpu.w) > 0.1f)) {
    std::cout << "FAIL: the test flow did not develop\n";
    return 1;
  }
  std::cout << "CPU reference " << cpu_seconds / steps * 1000.0 << " ms/frame, device "
            << device_seconds / steps * 1000.0 << " ms/frame (state kept on the device)\n";
  std::cout << "PASS: device kernels match the CPU reference\n";
  return 0;
}

/** Time the device alone. The state stays on the device until the end. */
int benchmark(gpu::SparseGasGpu &solver, const int resolution, const int steps, const bool profile)
{
  solver.set_profiling(profile);
  std::vector<std::pair<std::string, double>> kernel_totals;
  const sparse::Settings settings = test_settings(resolution);
  const sparse::Extent extent = {resolution, resolution, resolution * 3 / 2};
  sparse::State state = sparse::from_dense(blob(extent), 8, settings.thresholds, settings.dilation);
  const std::size_t total_bricks = state.topology.brick_table.size();
  solver.upload(state);

  std::vector<double> wall, gpu_time, topology;
  std::size_t peak_bricks = 0;
  int changes = 0, substeps = 0;
  for (int i = 0; i < steps; ++i) {
    const sparse::FrameInfo frame = solver.advance(settings, frame_time);
    const sparse::StepInfo &info = frame.last_step;
    substeps += frame.substeps;
    const gpu::SparseGasGpu::Timing &timing = solver.last_timing();
    wall.push_back(timing.wall_ms);
    gpu_time.push_back(timing.gpu_ms);
    topology.push_back(timing.topology_ms);
    peak_bricks = std::max(peak_bricks, info.active_bricks);
    changes += timing.topology_changed;
    for (const auto &[label, ms] : timing.kernels) {
      auto entry = std::find_if(kernel_totals.begin(),
                                kernel_totals.end(),
                                [&](const auto &item) { return item.first == label; });
      if (entry == kernel_totals.end()) {
        kernel_totals.emplace_back(label, ms);
      }
      else {
        entry->second += ms;
      }
    }
    std::cout << "frame " << i << ": " << frame.substeps << " substeps, " << info.active_bricks
              << " bricks, wall " << timing.wall_ms
              << " ms, GPU " << timing.gpu_ms << " ms\n";
  }
  solver.download(state);
  for (const float value : state.density) {
    if (!std::isfinite(value)) {
      std::cout << "FAIL: non-finite density\n";
      return 1;
    }
  }
  std::cout << "Divergence left after the last step, relative to peak velocity: "
            << relative_divergence(state) << '\n';
  if (!image_name.empty()) {
    write_plume(state, image_name);
  }
  std::cout << "Domain " << extent.x << "x" << extent.y << "x" << extent.z << ", peak "
            << peak_bricks << " of " << total_bricks << " bricks active, topology changed in "
            << changes << " of " << steps << " steps\n";
  std::cout << substeps << " substeps in " << steps << " frames\n";
  std::cout << "Median per frame: wall " << median(wall) << " ms, GPU "
            << (solver.has_timestamps() ? median(gpu_time) : -1.0) << " ms, topology (CPU) "
            << median(topology) << " ms\n";
  if (profile) {
    double total = 0.0;
    for (const auto &item : kernel_totals) {
      total += item.second;
    }
    std::cout << "GPU time per frame by kernel (profiling adds a timestamp per command):\n";
    for (const auto &[label, ms] : kernel_totals) {
      std::cout << "  " << label << ": " << ms / steps << " ms ("
                << (total > 0.0 ? 100.0 * ms / total : 0.0) << "%)\n";
    }
  }
  std::cout << "DONE: benchmark (no correctness check in this mode)\n";
  return 0;
}

int main(int argc, char **argv)
{
  /* Write every line immediately, so a crash still shows how far the run got. */
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::cout << std::unitbuf;

  bool bench = false, profile = false;
  int tiny_limit = -1;
  std::vector<int> numbers;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--bench") {
      bench = true;
    }
    else if (std::string(argv[i]) == "--cfl" && i + 1 < argc) {
      cfl = static_cast<float>(std::atof(argv[++i]));
    }
    else if (std::string(argv[i]) == "--first-order") {
      first_order = true;
    }
    else if (std::string(argv[i]) == "--jacobi") {
      use_jacobi = true;
    }
    else if (std::string(argv[i]) == "--tiny-limit" && i + 1 < argc) {
      tiny_limit = std::atoi(argv[++i]);
    }
    else if (std::string(argv[i]) == "--w-depth" && i + 1 < argc) {
      multigrid_w_depth = std::atoi(argv[++i]);
    }
    else if (std::string(argv[i]) == "--resync") {
      resync = true;
    }
    else if (std::string(argv[i]) == "--obstacle") {
      with_obstacle = true;
    }
    else if (std::string(argv[i]) == "--image" && i + 1 < argc) {
      image_name = argv[++i];
    }
    else if (std::string(argv[i]) == "--reflection" && i + 1 < argc) {
      reflection = std::atoi(argv[++i]);
    }
    else if (std::string(argv[i]) == "--midpoint" && i + 1 < argc) {
      midpoint = std::atoi(argv[++i]);
    }
    else if (std::string(argv[i]) == "--coarse-sweeps" && i + 2 < argc) {
      coarse_pre_sweeps = std::atoi(argv[++i]);
      coarse_post_sweeps = std::atoi(argv[++i]);
    }
    else if (std::string(argv[i]) == "--profile") {
      bench = true;
      profile = true;
    }
    else {
      numbers.push_back(std::atoi(argv[i]));
    }
  }
  const int resolution = numbers.size() > 0 ? numbers[0] : 32;
  const int steps = numbers.size() > 1 ? numbers[1] : 8; /* Frames. */

  std::cout << "Starting: resolution " << resolution << ", " << steps << " frames"
            << (profile ? ", profile" : bench ? ", benchmark" : "")
            << (use_jacobi ? ", 60 Jacobi sweeps" : ", 6 multigrid cycles")
            << (first_order ? ", first-order advection" : ", MacCormack advection")
            << (test_settings(resolution).reflection ? ", advection-reflection" : ", plain steps")
            << (with_obstacle ? ", plate obstacle" : "")
            << (resync ? ", every frame started from the CPU state" : "") << '\n';
  std::cout << "Creating the Vulkan device...\n";
  std::unique_ptr<gpu::SparseGasGpu> solver;
  try {
    solver = std::make_unique<gpu::SparseGasGpu>();
  }
  catch (const std::exception &error) {
    std::cout << "SKIP: " << error.what() << '\n';
    return 77;
  }
  std::cout << "Vulkan device: " << solver->device_name()
            << (solver->is_software_device() ? " (software)" : "")
            << (solver->has_timestamps() ? "" : " (no GPU timestamps)") << '\n';

  if (tiny_limit >= 0) {
    solver->set_single_dispatch_cell_limit(static_cast<std::size_t>(tiny_limit));
    std::cout << "Single-dispatch level limit: " << tiny_limit << " cells\n";
  }
  if (multigrid_w_depth >= 0) {
    std::cout << "Multigrid W-cycle depth: " << multigrid_w_depth << '\n';
  }
  if (coarse_pre_sweeps >= 0) {
    std::cout << "Multigrid coarse sweeps: " << coarse_pre_sweeps << " before, "
              << coarse_post_sweeps << " after\n";
  }
  try {
    return bench ? benchmark(*solver, resolution, steps, profile) :
                   compare(*solver, resolution, steps);
  }
  catch (const std::exception &error) {
    std::cout << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
