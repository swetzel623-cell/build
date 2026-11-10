/* SPDX-License-Identifier: GPL-2.0-or-later */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace blender::manta_gas {

/**
 * Frame-local CFL substep scheduler.
 *
 * This class deliberately owns no solver state. It only accounts for time within one node call,
 * which makes the policy testable without Blender or Mantaflow and keeps all physical state in the
 * explicit grids supplied by the caller.
 */
class GasStepDriver {
 public:
  GasStepDriver(const double frame_time, const double cfl, const int max_substeps)
      : frame_time_(frame_time), cfl_(cfl), max_substeps_(max_substeps)
  {
    valid_ = std::isfinite(frame_time_) && frame_time_ >= 0.0 && std::isfinite(cfl_) && cfl_ > 0.0 &&
             max_substeps_ > 0;
  }

  bool valid() const
  {
    return valid_;
  }

  bool finished() const
  {
    return valid_ && remaining() <= time_epsilon();
  }

  int substeps() const
  {
    return substeps_;
  }

  double simulated_time() const
  {
    return simulated_time_;
  }

  double remaining() const
  {
    return std::max(0.0, frame_time_ - simulated_time_);
  }

  /**
   * Return the next timestep, or nullopt if the frame is finished, the velocity is invalid, the
   * timestep underflows, or taking another step would exceed the configured substep limit.
   */
  std::optional<float> next_dt(const float max_velocity) const
  {
    if (!valid_ || finished() || substeps_ >= max_substeps_) {
      return std::nullopt;
    }
    if (!std::isfinite(max_velocity) || max_velocity < 0.0f) {
      return std::nullopt;
    }

    double dt = remaining();
    if (max_velocity > std::numeric_limits<float>::min()) {
      dt = std::min(dt, cfl_ / static_cast<double>(max_velocity));
    }

    /* Avoid a tiny trailing substep caused by floating-point round-off at the frame boundary. */
    if (remaining() - dt <= time_epsilon()) {
      dt = remaining();
    }
    if (!(dt > 0.0) || !std::isfinite(dt)) {
      return std::nullopt;
    }

    const float result = static_cast<float>(dt);
    if (!(result > 0.0f) || !std::isfinite(result)) {
      return std::nullopt;
    }
    return result;
  }

  bool commit(const float dt)
  {
    const double float_roundoff = std::numeric_limits<float>::epsilon() *
                                  std::max(1.0, frame_time_) * 2.0;
    if (!valid_ || finished() || substeps_ >= max_substeps_ || !(dt > 0.0f) ||
        !std::isfinite(dt) ||
        static_cast<double>(dt) > remaining() + time_epsilon() + float_roundoff)
    {
      return false;
    }

    simulated_time_ = std::min(frame_time_, simulated_time_ + static_cast<double>(dt));
    ++substeps_;
    return true;
  }

 private:
  double time_epsilon() const
  {
    return std::numeric_limits<double>::epsilon() * std::max(1.0, frame_time_) * 8.0;
  }

  double frame_time_ = 0.0;
  double cfl_ = 0.0;
  int max_substeps_ = 0;
  bool valid_ = false;
  int substeps_ = 0;
  double simulated_time_ = 0.0;
};

}  // namespace blender::manta_gas
