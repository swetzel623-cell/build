# SPDX-License-Identifier: GPL-2.0-or-later
"""Write a demo .blend for the Pyro Emitter and Pyro Simulation modifiers.

    blender -b --factory-startup --python make_sparse_demo.py -- OUTPUT.blend

Four small spheres, each with a Pyro Emitter followed by a Pyro Simulation: an Explosion, a
Fireball and a Ground Blast (spiky bursts after Houdini's Pyro Burst Source, with the Explosion
look) and a Campfire under a plate in the "Colliders" collection (with the Fire look). Press
play from frame 1. The demo carries its own copy of the node groups, so it also opens in a build
without the installed assets.
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


def pyro(name, location, groups, emitter, look, **simulation):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=location)
    obj = bpy.context.active_object
    obj.name = name
    emitter_group, simulation_group = groups
    emitting = obj.modifiers.new(smoke_assets.EMITTER_MODIFIER, "NODES")
    emitting.node_group = emitter_group
    set_input(emitting, "Preset", emitter)
    simulating = obj.modifiers.new(smoke_assets.MODIFIER, "NODES")
    simulating.node_group = simulation_group
    for key, value in {"Preset": look, "Show Mesh": False, **simulation}.items():
        set_input(simulating, key, value)
    return obj, simulating


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    path = os.path.abspath(argv[0] if argv else "pyro_demo.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 120

    groups = smoke_assets.build_assets(mark=False)
    blast = {"Voxel Size": 0.05, "Domain Size": (4.5, 4.5, 5.5)}
    pyro("Explosion", (-5.0, 0.0, 0.5), groups, "Explosion", "Explosion", **blast)
    pyro("Fireball", (0.0, 0.0, 0.5), groups, "Fireball", "Explosion", **blast)
    pyro("Ground Blast", (5.0, 0.0, 0.2), groups, "Ground Blast", "Explosion", **blast)
    _fire, campfire = pyro("Campfire", (9.0, 0.0, 0.1), groups, "Campfire", "Fire",
                           **{"Voxel Size": 0.04, "Domain Size": (1.6, 1.6, 3.0),
                              "Show Mesh": True})

    colliders = bpy.data.collections.new("Colliders")
    scene.collection.children.link(colliders)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(9.0, 0.0, 1.6))
    plate = bpy.context.active_object
    plate.name = "Plate"
    plate.scale = (0.9, 0.9, 0.06)
    for owner in list(plate.users_collection):
        owner.objects.unlink(plate)
    colliders.objects.link(plate)
    set_input(campfire, "Colliders", colliders)

    bpy.ops.object.camera_add(location=(2.0, -22.0, 2.5), rotation=(np.radians(85), 0.0, 0.0))
    scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
