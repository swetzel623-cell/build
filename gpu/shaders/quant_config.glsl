// SPDX-License-Identifier: MIT
// Channel storage precision, set per compiled variant (-DCHANNEL_HALF=0 or 1). CHANNEL_HALF 1:
// every per-voxel channel buffer (including the voxel-level pressure and its right-hand side)
// holds fp16; all arithmetic stays fp32. Needs storageBuffer16BitAccess (Vulkan 1.1 core).
#ifndef CHANNEL_HALF
#define CHANNEL_HALF 0
#endif
