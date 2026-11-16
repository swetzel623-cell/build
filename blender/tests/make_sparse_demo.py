# SPDX-License-Identifier: GPL-2.0-or-later
"""Write a demo .blend for the Smoke Simulation modifier.

    blender -b --factory-startup --python make_sparse_demo.py -- OUTPUT.blend

Two spheres with the Smoke Simulation modifier and turbulence: one emits freely, the other under
a plate in the "Colliders" collection. Press play to simulate. The demo carries its own copy of
the modifier's node groups, so it also opens in a build without the installed assets.
"""

from __future__ import annotations

import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets"))
import smoke_assets  # noqa: E402


def set_input(modifier, name, value):
    item = next(i for i in modifier.node_group.interface.items_tree
                if getattr(i, "in_out", None) == "INPUT" and i.name == name)
    getattr(modifier.properties.inputs, item.identifier).value = value


def emitter(name, location, group, **inputs):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=location)
    obj = bpy.context.active_object
    obj.name = name
    modifier = obj.modifiers.new("Smoke Simulation", "NODES")
    modifier.node_group = group
    for key, value in inputs.items():
        set_input(modifier, key, value)
    return obj, modifier


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    path = os.path.abspath(argv[0] if argv else "sparse_gpu_demo.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 150

    group = smoke_assets.build_assets(mark=False)
    settings = {"Voxel Size": 0.04, "Domain Size": (2.0, 2.0, 4.0), "Turbulence": 2.5,
                "Turbulence Scale": 0.3}
    emitter("Free Smoke", (-1.3, 0.0, 0.3), group, **settings)

    colliders = bpy.data.collections.new("Colliders")
    scene.collection.children.link(colliders)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.3, 0.0, 1.6))
    plate = bpy.context.active_object
    plate.name = "Plate"
    plate.scale = (1.1, 1.1, 0.08)
    for owner in list(plate.users_collection):
        owner.objects.unlink(plate)
    colliders.objects.link(plate)
    _sheltered, modifier = emitter("Smoke Under Plate", (1.3, 0.0, 0.3), group, **settings)
    set_input(modifier, "Colliders", colliders)

    bpy.ops.object.camera_add(location=(0.0, -10.0, 2.0), rotation=(np.radians(88), 0.0, 0.0))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
