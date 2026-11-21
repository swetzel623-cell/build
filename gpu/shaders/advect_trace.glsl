// SPDX-License-Identifier: MIT
// Velocity sampling and the backtrace shared by the fused advection kernels. Same arithmetic as
// advect.comp and maccormack.comp, so the fused kernels give bit-identical results.
// The including kernel binds the advecting velocity at bindings 4, 5 and 6.
layout(std430, binding = 4) readonly buffer VelU { CH_T vel_u[]; };
layout(std430, binding = 5) readonly buffer VelV { CH_T vel_v[]; };
layout(std430, binding = 6) readonly buffer VelW { CH_T vel_w[]; };

#define SAMPLE(name, channel) \
  float name##_at(ivec3 c) \
  { \
    int i = index_of(c); \
    return i < 0 ? 0.0 : float(channel[i]); \
  } \
  float name(vec3 p, vec3 offset) \
  { \
    vec3 f = p - offset; \
    vec3 base = floor(f); \
    ivec3 i = ivec3(base); \
    vec3 t = f - base; \
    float f000 = name##_at(i); \
    float f100 = name##_at(i + ivec3(1, 0, 0)); \
    float f010 = name##_at(i + ivec3(0, 1, 0)); \
    float f110 = name##_at(i + ivec3(1, 1, 0)); \
    float f001 = name##_at(i + ivec3(0, 0, 1)); \
    float f101 = name##_at(i + ivec3(1, 0, 1)); \
    float f011 = name##_at(i + ivec3(0, 1, 1)); \
    float f111 = name##_at(i + ivec3(1, 1, 1)); \
    float c00 = f000 + (f100 - f000) * t.x; \
    float c10 = f010 + (f110 - f010) * t.x; \
    float c01 = f001 + (f101 - f001) * t.x; \
    float c11 = f011 + (f111 - f011) * t.x; \
    float c0 = c00 + (c10 - c00) * t.y; \
    float c1 = c01 + (c11 - c01) * t.y; \
    return c0 + (c1 - c0) * t.z; \
  }

SAMPLE(sample_u, vel_u)
SAMPLE(sample_v, vel_v)
SAMPLE(sample_w, vel_w)

vec3 velocity_at(vec3 p)
{
  return vec3(sample_u(p, vec3(0.0, 0.5, 0.5)),
              sample_v(p, vec3(0.5, 0.0, 0.5)),
              sample_w(p, vec3(0.5, 0.5, 0.0)));
}

/* P.flags.x selects the midpoint trace. */
vec3 trace(vec3 p, float step)
{
  vec3 velocity = velocity_at(p);
  if (P.flags.x != 0) {
    float half_step = 0.5 * step;
    velocity = velocity_at(p + half_step * velocity);
  }
  return p + step * velocity;
}

/* Trilinear weights of the eight corners, in the order of the SAMPLE macro. */
float lerp8(float value[8], vec3 t)
{
  float c00 = value[0] + (value[1] - value[0]) * t.x;
  float c10 = value[2] + (value[3] - value[2]) * t.x;
  float c01 = value[4] + (value[5] - value[4]) * t.x;
  float c11 = value[6] + (value[7] - value[6]) * t.x;
  float c0 = c00 + (c10 - c00) * t.y;
  float c1 = c01 + (c11 - c01) * t.y;
  return c0 + (c1 - c0) * t.z;
}
