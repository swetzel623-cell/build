// SPDX-License-Identifier: MIT
#pragma once

#include "manta_improve/sparse_gas.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace manta_improve::gpu {

/**
 * Runs manta_improve::sparse::step on a Vulkan device with the state kept in device memory.
 *
 * Each step is two submissions. The first marks which bricks hold content and downloads only those
 * flags. The CPU then builds the next brick table, as the reference does. The second submission
 * uploads the brick table if it changed, classifies cells on the device, moves every channel onto
 * the new bricks, and runs the whole step. The state is transferred only by upload() and download().
 *
 * Only PressureSolver::Jacobi is available, with at least one sweep.
 */
class SparseGasGpu {
 public:
  struct Timing {
    /** GPU execution time of the step's submissions, or negative if not measurable. */
    double gpu_ms = -1.0;
    /** Wall-clock time of step(), including CPU work and waiting. */
    double wall_ms = 0.0;
    /** CPU time spent building the next topology. */
    double topology_ms = 0.0;
    bool topology_changed = false;
    /** With profiling on: GPU milliseconds per kernel or transfer kind in this step. */
    std::vector<std::pair<std::string, double>> kernels;
  };

  SparseGasGpu();
  ~SparseGasGpu();

  const std::string &device_name() const;
  bool is_software_device() const;
  bool has_timestamps() const;
  /** Time each kernel separately. Adds a timestamp per command, so totals rise slightly. */
  void set_profiling(bool enabled);

  /** Replace the device state with `state`. */
  void upload(const sparse::State &state);
  /** Copy the device state, including its topology, into `state`. */
  void download(sparse::State &state);

  /**
   * Advance the device state by one step. StepInfo::max_divergence is not measured on this path
   * and is reported as NaN.
   */
  sparse::StepInfo step(const sparse::Settings &settings);
  const Timing &last_timing() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace manta_improve::gpu
