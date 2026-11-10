/* SPDX-License-Identifier: GPL-2.0-or-later */
#pragma once

#include <cstddef>

namespace blender::manta_gas {

/** Mantaflow dense-grid indexing: X is fastest, followed by Y and then Z. */
inline std::size_t grid_index(const int size_x,
                              const int size_y,
                              const int x,
                              const int y,
                              const int z)
{
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(size_x) *
             (static_cast<std::size_t>(y) +
              static_cast<std::size_t>(size_y) * static_cast<std::size_t>(z));
}

}  // namespace blender::manta_gas
