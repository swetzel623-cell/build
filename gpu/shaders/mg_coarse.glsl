// SPDX-License-Identifier: MIT
// Neighbor sum on a coarse level: operator entries are stored as (diagonal, wx, wy, wz), where
// each weight couples a cell to its lower neighbor along that axis.
float coarse_neighbor_sum(int index, ivec3 c)
{
  float sum = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    float lower = op[index][1 + axis];
    if (lower != 0.0) {
      sum += lower * x[level_index(P.la, c - AXES[axis])];
    }
    int above = level_index(P.la, c + AXES[axis]);
    if (above >= 0) {
      float upper = op[above][1 + axis];
      if (upper != 0.0) {
        sum += upper * x[above];
      }
    }
  }
  return sum;
}
