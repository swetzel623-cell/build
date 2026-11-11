// SPDX-License-Identifier: MIT
#pragma once

#include "manta_improve/sparse_gas.hpp"

#include <memory>
#include <string>

namespace manta_improve::gpu {

/**
 * Runs manta_improve::sparse::step with every per-voxel kernel on a Vulkan device.
 *
 * Topology changes still happen on the CPU, and the state is uploaded and downloaded every step.
 * Only PressureSolver::Jacobi is available: it needs no reductions, so its result can be compared
 * with the CPU reference sweep for sweep.
 */
class SparseGasGpu {
 public:
  SparseGasGpu();
  ~SparseGasGpu();

  const std::string &device_name() const;
  bool is_software_device() const;

  sparse::StepInfo step(sparse::State &state, const sparse::Settings &settings);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace manta_improve::gpu
