// SPDX-License-Identifier: MIT
#pragma once

// Minimal Vulkan compute runner for the sparse solver's kernels.
//
// Storage buffers live in device memory. Work is recorded between begin() and submit() into one
// command buffer, with a full memory barrier after every command, and submitted once. Uploads and
// downloads go through temporary host-visible staging buffers. The GPU time of each submission is
// measured with timestamp queries where the device supports them.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace manta_improve::gpu {

/** Kernels see storage buffers at bindings 1 to 16. */
constexpr std::size_t binding_count = 16;
/** Every kernel receives up to this many bytes of push constants. */
constexpr std::size_t push_constant_bytes = 64;

class Compute {
 public:
  struct Buffer {
    std::uint64_t handle = 0;
    std::uint64_t memory = 0;
    std::size_t size = 0;
  };
  using Bindings = std::array<const Buffer *, binding_count>;

  /** Throws std::runtime_error if no Vulkan device with a compute queue is available. */
  Compute();
  ~Compute();
  Compute(const Compute &) = delete;
  Compute &operator=(const Compute &) = delete;

  const std::string &device_name() const;
  /** True for software implementations such as llvmpipe. */
  bool is_software_device() const;
  /** True if submit() can report GPU execution time. */
  bool has_timestamps() const;

  /**
   * Time every recorded command with its own timestamp. This adds a timestamp write after each
   * command, so leave it off when measuring total time.
   */
  void set_profiling(bool enabled);
  /** Submit recorded work every `dispatches` dispatches without waiting; 0 waits for submit(). */
  void set_command_batch(std::size_t dispatches);
  /** Whether storage buffers may hold 16-bit floats on this device. */
  bool supports_half_storage() const;
  /** GPU milliseconds per label in the last submission, in order of first appearance. */
  const std::vector<std::pair<std::string, double>> &last_profile() const;

  /** A device-memory storage buffer. Must not be called between begin() and submit(). */
  Buffer create_buffer(std::size_t bytes);
  /** Abandons any recording in progress, so that cleanup after an error is safe. */
  void destroy_buffer(Buffer &buffer);

  /** Compile a SPIR-V compute shader into a kernel and return its id. */
  int create_kernel(const std::uint32_t *spirv, std::size_t word_count);

  void begin();
  /** Copy `bytes` from host memory into the start of `target` when the work is submitted. */
  void upload(const Buffer &target, const void *data, std::size_t bytes);
  /** Copy the start of `source` into `out` once submit() returns. */
  void download(const Buffer &source, void *out, std::size_t bytes);
  /** Set every 32-bit word of `target` to `value`. */
  void fill(const Buffer &target, std::uint32_t value);
  /** Run a kernel over `group_count` work groups. Null bindings receive a dummy buffer. */
  void dispatch(int kernel,
                const void *push_constants,
                std::size_t push_bytes,
                const Bindings &buffers,
                std::uint32_t group_count,
                const char *label = "dispatch");
  /**
   * Submit the recorded work, wait for it, and complete the downloads. Returns the GPU execution
   * time in milliseconds, or a negative value if the device has no usable timestamps.
   */
  double submit();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace manta_improve::gpu
