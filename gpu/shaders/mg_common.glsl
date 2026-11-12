// SPDX-License-Identifier: MIT
// Shared by the multigrid kernels. Mirrors src/sparse_multigrid.cpp.
//
// A level is described by an ivec4: w is the number of cells per brick side for levels stored
// per brick slot (the voxel level uses the brick size), or 0 for a dense grid of size xyz.

layout(local_size_x = 64) in;

layout(push_constant) uniform Params
{
  ivec4 bricks; /* Brick grid size. */
  ivec4 la;     /* The level being worked on, or the finer level of a transfer. */
  ivec4 lb;     /* The coarser level of a transfer. */
  int count;    /* Cells to process. */
  int color;    /* Red-black color for smoothing. */
  float scale;  /* Over-correction for prolongation. */
  int unused;
}
P;
layout(std430, binding = 1) readonly buffer BrickTable { int brick_table[]; };
layout(std430, binding = 2) readonly buffer BrickCoords { ivec4 brick_coords[]; };

const ivec3 AXES[3] = ivec3[3](ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(0, 0, 1));
const uint CELL_FLUID = 1u;

/* Storage index of a cell at level coordinates, or -1. */
int level_index(ivec4 level, ivec3 c)
{
  if (any(lessThan(c, ivec3(0)))) {
    return -1;
  }
  if (level.w == 0) {
    if (any(greaterThanEqual(c, level.xyz))) {
      return -1;
    }
    return c.x + level.x * (c.y + level.y * c.z);
  }
  int side = level.w;
  ivec3 b = c / side;
  if (any(greaterThanEqual(b, P.bricks.xyz))) {
    return -1;
  }
  int slot = brick_table[b.x + P.bricks.x * (b.y + P.bricks.y * b.z)];
  if (slot < 0) {
    return -1;
  }
  ivec3 l = c - b * side;
  return slot * side * side * side + l.x + side * (l.y + side * l.z);
}

/* Level coordinates of a stored cell. */
ivec3 level_coords(ivec4 level, int index)
{
  if (level.w == 0) {
    return ivec3(index % level.x, (index / level.x) % level.y, index / (level.x * level.y));
  }
  int side = level.w;
  int per_brick = side * side * side;
  int slot = index / per_brick;
  int local = index - slot * per_brick;
  return brick_coords[slot].xyz * side +
         ivec3(local % side, (local / side) % side, local / (side * side));
}

bool cell_in_range(out int index)
{
  index = int(gl_GlobalInvocationID.x);
  return index < P.count;
}

/* Voxel level, from the packed word written by classify_links. */
bool fine_is_fluid(uint packed) { return (packed & 3u) == CELL_FLUID; }
float fine_diagonal(uint packed) { return float(bitCount((packed >> 2) & 63u)); }
bool fine_linked(uint packed, int neighbor) { return (packed & (1u << (8 + neighbor))) != 0u; }
