# SPDX-License-Identifier: GPL-2.0-or-later
"""Write a demo .blend for the Gas Simulation Step node's Sparse GPU mode.

    blender -b --factory-startup --python make_sparse_demo.py -- OUTPUT.blend [--res 64]

Two smoke plumes side by side, each in its own Simulation Zone: a free plume and one rising into
a plate obstacle. Press play to simulate. Untick "Sparse GPU" on a node to run Mantaflow instead.
"""

from __future__ import annotations

import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gas_node_testlib as lib  # noqa: E402
from gas_node_testlib import Zone  # noqa: E402


def plate_mask(res):
    nx, ny, nz = res
    bottom, top = int(0.46 * nz), int(0.46 * nz) + max(3, nx // 16)
    half = nx * 7 // 32
    lo, hi = nx // 2 - half, nx // 2 + half
    return lambda x, y, z: ((z >= bottom) & (z < top) & (x >= lo) & (x < hi) & (y >= lo) & (y < hi)) * 1.0, (bottom, top, lo, hi)


def build(name, res, origin, voxel, with_plate):
    zone = Zone(res, origin=origin, voxel=voxel, prefix=name + "_", **{
        "Sparse GPU": True,
        "CFL": 6.0,
        "Max Substeps": 64,
        "Heat Buoyancy": 0.5,
        "Density Buoyancy": 0.5,
    })
    nx, ny, nz = res
    radius = nx / 8.0
    field = lib.dense(zone, lambda x, y, z: (((x + 0.5 - nx / 2) ** 2 + (y + 0.5 - ny / 2) ** 2 +
                                              (z + 0.5 - nz / 4) ** 2) <= radius ** 2) * 1.0)
    obstacle = None
    if with_plate:
        mask, (bottom, top, lo, hi) = plate_mask(res)
        obstacle = lib.dense(zone, mask)
        # A visible stand-in for the plate; the solver only sees the grid.
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        cube.name = name + "_plate"
        cube.scale = ((hi - lo) * voxel / 2, (hi - lo) * voxel / 2, (top - bottom) * voxel / 2)
        cube.location = (origin[0] + (lo + hi) * voxel / 2, origin[1] + (lo + hi) * voxel / 2,
                         origin[2] + (bottom + top) * voxel / 2)
    data = lib.make_data_object(name + "_initial_state", zone, density=field, temperature=field,
                                obstacle=obstacle)
    data.hide_set(True)
    data.hide_render = True

    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    b = lib._Builder(tree)
    state, _diag = lib._zone(b, zone, lib._initial_state(b, zone))
    material = bpy.data.materials.get("Smoke") or make_smoke_material()
    set_material = b.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    b.link(state, set_material.inputs["Geometry"])
    out = b.new("NodeGroupOutput")
    b.link(set_material.outputs[0], out.inputs[0])

    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new("Gas", "NODES")
    modifier.node_group = tree
    return obj


def make_smoke_material():
    material = bpy.data.materials.new("Smoke")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 5.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(volume.outputs[0], output.inputs["Volume"])
    return material


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    path = os.path.abspath(argv[0] if argv else "sparse_gpu_demo.blend")
    n = int(argv[argv.index("--res") + 1]) if "--res" in argv else 64
    res = (n, n, n * 3 // 2)
    voxel = 2.0 / n
    lib.reset_scene(fps=30, frame_end=150)
    build("free_plume", res, (-2.5, -1.0, 0.0), voxel, False)
    build("plate", res, (0.5, -1.0, 0.0), voxel, True)
    scene = bpy.context.scene
    bpy.ops.object.camera_add(location=(0.0, -9.0, 1.6), rotation=(np.radians(90), 0.0, 0.0))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 6))
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
