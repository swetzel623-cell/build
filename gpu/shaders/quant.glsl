// SPDX-License-Identifier: MIT
// Storage type of the per-voxel channels. Loads convert to float; stores go through STORE.
#include "quant_config.glsl"
#if CHANNEL_HALF
#extension GL_EXT_shader_16bit_storage : require
#define CH_T float16_t
#define STORE(x) float16_t(x)
#else
#define CH_T float
#define STORE(x) (x)
#endif
#define QS(x) STORE(x)
#define QV(x) STORE(x)
#define QP(x) STORE(x)
#define QPC(x) (x)
