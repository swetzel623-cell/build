// SPDX-License-Identifier: MIT
#pragma once

// Minimal Vulkan compute runner: storage buffers and one-dispatch-at-a-time kernels.
//
// This is test scaffolding for running the sparse solver's GLSL kernels on any Vulkan device,
// including software implementations. It is not tuned: buffers are host-visible and every
// dispatch is submitted and waited for on its own.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace manta_improve::gpu {

/** Number of storage-buffer bindings every kernel declares. */
constexpr std::size_t binding_count = 9;

class Compute {
 public:
  struct Buffer {
    void *mapped = nullptr;
    std::size_t size = 0;
    std::uint64_t handle = 0;
    std::uint64_t memory = 0;
  };

  /** Throws std::runtime_error if no Vulkan device with a compute queue is available. */
  Compute();
  ~Compute();
  Compute(const Compute &) = delete;
  Compute &operator=(const Compute &) = delete;

  const std::string &device_name() const;
  /** True for software implementations such as llvmpipe. */
  bool is_software_device() const;

  Buffer create_buffer(std::size_t bytes);
  void destroy_buffer(Buffer &buffer);

  /** Compile a SPIR-V compute shader into a kernel and return its id. */
  int create_kernel(const std::uint32_t *spirv, std::size_t word_count);

  /** Run a kernel over `group_count` work groups. Null entries bind a dummy buffer. */
  void dispatch(int kernel,
                const std::array<const Buffer *, binding_count> &buffers,
                std::uint32_t group_count);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace manta_improve::gpu
