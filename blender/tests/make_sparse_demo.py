# SPDX-License-Identifier: GPL-2.0-or-later
"""Write a demo .blend for the Pyro Simulation modifier.

    blender -b --factory-startup --python make_sparse_demo.py -- OUTPUT.blend

Four emitters with the Pyro Simulation modifier, one per preset: a campfire under a plate in the
"Colliders" collection, an explosion, a large plume, and a sphere flying across the scene that
leaves a trail. Press play from frame 1. The demo carries its own copy of the node groups, so it
also opens in a build without the installed assets.
"""

from __future__ import annotations

import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets"))
import smoke_assets  # noqa: E402


def set_input(modifier, name, value):
    item = smoke_assets.input_item(modifier.node_group, name)
    getattr(modifier.properties.inputs, item.identifier).value = value


def emitter(name, location, group, primitive="sphere", size=0.25, **inputs):
    if primitive == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=size, depth=0.1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    modifier = obj.modifiers.new(smoke_assets.MODIFIER, "NODES")
    modifier.node_group = group
    for key, value in inputs.items():
        set_input(modifier, key, value)
    return obj, modifier


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    path = os.path.abspath(argv[0] if argv else "pyro_demo.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 150

    group = smoke_assets.build_assets(mark=False)
    _fire, campfire = emitter("Campfire", (-4.5, 0.0, 0.1), group, primitive="cylinder",
                              size=0.3, Preset="Campfire",
                              **{"Voxel Size": 0.04, "Domain Size": (1.6, 1.6, 3.0)})
    emitter("Explosion", (-1.6, 0.0, 0.8), group, size=0.2, Preset="Explosion",
            **{"Voxel Size": 0.06, "Domain Size": (4.0, 4.0, 6.0)})
    emitter("Large Plume", (1.8, 0.0, 0.1), group, primitive="cylinder", size=0.4,
            Preset="Large Plume", **{"Voxel Size": 0.06, "Domain Size": (2.5, 2.5, 5.0)})
    trail, _ = emitter("Trail", (4.0, 0.0, 1.0), group, size=0.12, Preset="Trail",
                       **{"Voxel Size": 0.04, "Domain Size": (6.0, 2.0, 3.0)})
    # The trail's emitter flies across its domain, which is centred where it starts.
    trail.location = (4.0, 0.0, 1.0)
    trail.keyframe_insert("location", frame=1)
    trail.location = (6.5, 0.0, 1.6)
    trail.keyframe_insert("location", frame=100)

    colliders = bpy.data.collections.new("Colliders")
    scene.collection.children.link(colliders)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-4.5, 0.0, 1.6))
    plate = bpy.context.active_object
    plate.name = "Plate"
    plate.scale = (0.9, 0.9, 0.06)
    for owner in list(plate.users_collection):
        owner.objects.unlink(plate)
    colliders.objects.link(plate)
    set_input(campfire, "Colliders", colliders)

    bpy.ops.object.camera_add(location=(0.8, -15.0, 2.2), rotation=(np.radians(85), 0.0, 0.0))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
