// SPDX-License-Identifier: MIT
// Shared layout of the sparse brick grid. Mirrors manta_improve::sparse::Topology: every buffer
// is one of its flat arrays, uploaded unchanged.

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
layout(std430, binding = 4) buffer B4 { float b4[]; };
layout(std430, binding = 5) buffer B5 { float b5[]; };
layout(std430, binding = 6) buffer B6 { float b6[]; };
layout(std430, binding = 7) buffer B7 { float b7[]; };
layout(std430, binding = 8) buffer B8 { float b8[]; };
layout(std430, binding = 9) buffer B9 { float b9[]; };
#endif

const uint CELL_OPEN = 0u;
const uint CELL_FLUID = 1u;
const uint CELL_SOLID = 2u;

bool is_wall(ivec3 c)
{
  return any(lessThanEqual(c, ivec3(0))) || any(greaterThanEqual(c, P.extent.xyz - 1));
}

/* Index of the stored voxel at domain coordinates, or -1. */
int index_of(ivec3 c)
{
  if (any(lessThan(c, ivec3(0)))) {
    return -1;
  }
  int size = P.extent.w;
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
  int size = P.extent.w;
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
