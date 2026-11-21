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
 * PressureSolver::Jacobi and PressureSolver::Multigrid are available. With multigrid, the coarse
 * operators are rebuilt on the device whenever the topology changes.
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
  /**
   * Store the channels as 16-bit floats (Settings::half_precision). Changing it drops the device
   * state: upload() must follow. Returns whether half precision is in use, which it is not on a
   * device without 16-bit storage buffers.
   */
  bool set_half_precision(bool enabled);
  bool half_precision() const;

  /**
   * Multigrid levels of at most this many cells are cycled together in one dispatch; 0 gives
   * every level its own dispatches. Takes effect at the next upload() or topology change.
   */
  void set_single_dispatch_cell_limit(std::size_t cells);

  /** Replace the device state with `state`, including its obstacles and their velocity. */
  void upload(const sparse::State &state);
  /**
   * Replace the obstacles of the device state, as sparse::set_obstacles does: the bricks stay,
   * their cells are classified again, and `velocity` moves the obstacle faces.
   */
  void set_obstacles(std::vector<std::uint32_t> mask, sparse::BrickField velocity = {});
  /** Whether set_obstacles with these would change nothing: the same mask, and no velocity
   * now or before. */
  bool same_obstacles(const std::vector<std::uint32_t> &mask,
                      const sparse::BrickField &velocity) const;
  /** Copy the device state, including its topology, into `state`. */
  void download(sparse::State &state);
  /**
   * Copy only what is shown and cached between frames into `state`: the brick layout (table
   * and coordinates), density, temperature and velocity, fuel and flame with `fire`, and
   * pressure with `pressure`. Other channels are emptied. The vectors are resized, not cleared,
   * so a `state` reused from frame to frame does not zero its memory again.
   */
  void download_shown(sparse::State &state, bool fire, bool pressure);

  /**
   * Advance the device state by one step. StepInfo::max_divergence is not measured on this path
   * and is reported as NaN.
   */
  sparse::StepInfo step(const sparse::Settings &settings, const sparse::Sources *sources = nullptr);
  /**
   * Advance by `frame_time` in CFL-limited substeps, like sparse::advance. The speed that limits
   * each substep comes back with the activity flags, so this adds no transfers.
   */
  sparse::FrameInfo advance(const sparse::Settings &settings,
                            float frame_time,
                            const sparse::Sources *sources = nullptr);
  /** Timing of the last step(), or the sum over the substeps of the last advance(). */
  const Timing &last_timing() const;

 private:
  sparse::StepInfo impl_step(const sparse::Settings &settings, const float *remaining);
  /** Upload the sources the next steps apply; null for none. */
  void use_sources(const sparse::Sources *sources);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace manta_improve::gpu
