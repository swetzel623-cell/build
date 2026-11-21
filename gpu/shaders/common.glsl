// SPDX-License-Identifier: MIT
// Shared layout of the sparse brick grid. Mirrors manta_improve::sparse::Topology: every buffer
// is one of its flat arrays, uploaded unchanged.
#include "quant.glsl"

layout(local_size_x = 64) in;

layout(push_constant) uniform Params
{
  ivec4 extent; /* Domain size in voxels; w is the brick size. */
  ivec4 bricks; /* Brick grid size; w is the number of stored voxels. */
  vec4 a;       /* Kernel-specific. */
  ivec4 flags;  /* Kernel-specific. */
}
P;
layout(std430, binding = 1) readonly buffer BrickTable { int brick_table[]; };
layout(std430, binding = 2) readonly buffer BrickCoords { ivec4 brick_coords[]; };
/* Written only by the classify kernel. */
layout(std430, binding = 3) buffer CellTypes { uint cell_type[]; };

#ifndef NO_FLOAT_CHANNELS
/* Float channels. Each kernel names the ones it uses. */
layout(std430, binding = 4) buffer B4 { CH_T b4[]; };
layout(std430, binding = 5) buffer B5 { CH_T b5[]; };
layout(std430, binding = 6) buffer B6 { CH_T b6[]; };
layout(std430, binding = 7) buffer B7 { CH_T b7[]; };
layout(std430, binding = 8) buffer B8 { CH_T b8[]; };
layout(std430, binding = 9) buffer B9 { CH_T b9[]; };
#endif

const uint CELL_OPEN = 0u;
const uint CELL_FLUID = 1u;
const uint CELL_SOLID = 2u;

/* P.extent.w: the brick size in its low 8 bits; bit 8 set for open borders. */
int brick_size()
{
  return P.extent.w & 255;
}

/* Open borders: only the floor is a wall. Otherwise every outer layer is. */
bool is_wall(ivec3 c)
{
  if ((P.extent.w & 256) != 0) {
    return c.z <= 0;
  }
  return any(lessThanEqual(c, ivec3(0))) || any(greaterThanEqual(c, P.extent.xyz - 1));
}

/* Index of the stored voxel at domain coordinates, or -1. */
int index_of(ivec3 c)
{
  if (any(lessThan(c, ivec3(0)))) {
    return -1;
  }
  int size = brick_size();
  ivec3 b = c / size;
  if (any(greaterThanEqual(b, P.bricks.xyz))) {
    return -1;
  }
  int slot = brick_table[b.x + P.bricks.x * (b.y + P.bricks.y * b.z)];
  if (slot < 0) {
    return -1;
  }
  ivec3 l = c - b * size;
  return slot * size * size * size + l.x + size * (l.y + size * l.z);
}

uint type_at(ivec3 c)
{
  if (is_wall(c)) {
    return CELL_SOLID;
  }
  int index = index_of(c);
  return index < 0 ? CELL_OPEN : cell_type[index];
}

/* Domain coordinates of a stored voxel. */
ivec3 coords_of(int index)
{
  int size = brick_size();
  int per_brick = size * size * size;
  int slot = index / per_brick;
  int local = index - slot * per_brick;
  ivec3 l = ivec3(local % size, (local / size) % size, local / (size * size));
  return brick_coords[slot].xyz * size + l;
}

/* False for the padding invocations of the last work group. */
bool voxel_in_range(out int index)
{
  index = int(gl_GlobalInvocationID.x);
  return index < P.bricks.w;
}
