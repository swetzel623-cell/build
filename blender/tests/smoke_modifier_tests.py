# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of the Smoke Simulation modifier and the Smoke node-group assets.

    blender -b --factory-startup --python smoke_modifier_tests.py

Exits non-zero if any test fails.
"""

from __future__ import annotations

import os
import sys
import traceback

import bpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets"))
import smoke_assets  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = 250
    bpy.context.view_layer.update()
    return scene


def add_cube(name, location, size):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def set_input(modifier, name, value):
    item = next(i for i in modifier.node_group.interface.items_tree
                if getattr(i, "in_out", None) == "INPUT" and i.name == name)
    getattr(modifier.properties.inputs, item.identifier).value = value


def probe_group():
    """A second modifier that keeps the grid names and turns the smoke into a mesh to measure."""
    tree = bpy.data.node_groups.get("Smoke Probe")
    if tree:
        return tree
    tree = bpy.data.node_groups.new("Smoke Probe", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    get = nodes.new("GeometryNodeGetNamedGrid")
    get.data_type = "FLOAT"
    get.inputs["Name"].default_value = "density"
    links.new(group_in.outputs[0], get.inputs["Volume"])
    to_mesh = nodes.new("GeometryNodeGridToMesh")
    to_mesh.inputs["Threshold"].default_value = 0.02
    links.new(get.outputs["Grid"], to_mesh.inputs["Grid"])
    # The smoke's surface, without the emitter's mesh.
    separate = nodes.new("GeometryNodeSeparateComponents")
    links.new(group_in.outputs[0], separate.inputs[0])
    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(to_mesh.outputs[0], join.inputs[0])
    links.new(separate.outputs["Volume"], join.inputs[0])
    links.new(join.outputs[0], group_out.inputs[0])
    return tree


def smoke_emitter(**inputs):
    emitter = add_cube("Emitter", (0.3, -0.2, 0.25), 0.3)
    modifier = emitter.modifiers.new("Smoke", "NODES")
    modifier.node_group = smoke_assets.build_assets(mark=False)
    for name, value in inputs.items():
        set_input(modifier, name, value)
    probe = emitter.modifiers.new("Probe", "NODES")
    probe.node_group = probe_group()
    return emitter, modifier


class Smoke:
    """The evaluated smoke of an emitter at a frame: grid names and world bounds of the smoke."""

    def __init__(self, emitter, frame):
        bpy.context.scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = emitter.evaluated_get(depsgraph)
        self.geometry = evaluated.evaluated_geometry()
        volume = self.geometry.volume
        self.grids = sorted(grid.name for grid in volume.grids) if volume else []
        mesh = self.geometry.mesh
        world = evaluated.matrix_world
        points = [world @ v.co for v in mesh.vertices] if mesh else []
        self.low = min((p.z for p in points), default=float("nan"))
        self.top = max((p.z for p in points), default=float("nan"))
        self.left = min((p.x for p in points), default=float("nan"))
        self.right = max((p.x for p in points), default=float("nan"))
        self.warnings = [(w.type, w.message) for w in emitter.modifiers[0].node_warnings]


def run_until(emitter, last):
    smoke = None
    for frame in range(1, last + 1):
        smoke = Smoke(emitter, frame)
    return smoke


# --------------------------------------------------------------------------------------------


def test_modifier_emits_and_rises():
    reset()
    emitter, _modifier = smoke_emitter(**{"Voxel Size": 0.0625})
    early = run_until(emitter, 4)
    late = run_until(emitter, 20)
    print(f"  smoke top: frame 4 {early.top:.3f}, frame 20 {late.top:.3f}; "
          f"{late.warnings}")
    check(not [w for w in late.warnings if w[0] == "ERROR"], f"errors: {late.warnings}")
    check(late.grids == ["density", "pressure", "temperature", "velocity"],
          f"unexpected grids: {late.grids}")
    check(late.top > early.top + 0.2, "the smoke did not rise above the emitter")


def test_moving_emitter_leaves_a_trail():
    reset()
    emitter, _modifier = smoke_emitter(**{"Voxel Size": 0.0625, "Heat Buoyancy": 0.0,
                                          "Temperature": 0.0})
    emitter.location = (-0.5, 0.0, 0.5)
    emitter.keyframe_insert("location", frame=1)
    emitter.location = (0.5, 0.0, 0.5)
    emitter.keyframe_insert("location", frame=16)
    smoke = run_until(emitter, 16)
    print(f"  trail spans x {smoke.left:.2f} to {smoke.right:.2f} in world space")
    check(not [w for w in smoke.warnings if w[0] == "ERROR"], f"errors: {smoke.warnings}")
    # The smoke stays where it was emitted, so it spans the emitter's whole path.
    check(smoke.left < -0.4 and smoke.right > 0.4, "the smoke did not stay in world space")


def test_collider_shelters():
    def top_with(plate):
        reset()
        emitter, modifier = smoke_emitter(**{"Voxel Size": 0.0625})
        if plate:
            collection = bpy.data.collections.new("Colliders")
            bpy.context.scene.collection.children.link(collection)
            plate = add_cube("Plate", (0.3, -0.2, 0.7), 1.0)
            plate.scale = (1.0, 1.0, 0.15)
            for owner in plate.users_collection:
                owner.objects.unlink(plate)
            collection.objects.link(plate)
            set_input(modifier, "Colliders", collection)
        smoke = run_until(emitter, 30)
        check(not [w for w in smoke.warnings if w[0] == "ERROR"], f"errors: {smoke.warnings}")
        return smoke.top

    free, sheltered = top_with(False), top_with(True)
    print(f"  smoke top: free {free:.3f}, under a plate {sheltered:.3f}")
    check(sheltered < free - 0.2, "the collider did not hold the smoke back")


def test_assets_install():
    reset()
    folder = os.path.join(bpy.app.tempdir, "smoke_assets_datafiles")
    os.makedirs(os.path.join(folder, "assets"), exist_ok=True)
    with open(os.path.join(folder, "assets", "blender_assets.cats.txt"), "w") as stream:
        stream.write("VERSION 1\n\n")
    smoke_assets.install(folder)
    smoke_assets.install(folder)  # A second run must not add the catalogs twice.
    with open(os.path.join(folder, "assets", "blender_assets.cats.txt")) as stream:
        lines = stream.read().splitlines()
    for uuid, _path in (smoke_assets.CATALOG_SIMULATION, smoke_assets.CATALOG_PHYSICS):
        check(sum(line.startswith(uuid) for line in lines) == 1, f"catalog {uuid} not once")
    path = os.path.join(folder, "assets", "nodes", smoke_assets.ASSET_FILE)
    with bpy.data.libraries.load(path, assets_only=True) as (data_from, _data_to):
        names = sorted(data_from.node_groups)
    check(names == ["Smoke Collider", "Smoke Simulation", "Smoke Source"], f"assets: {names}")
    with bpy.data.libraries.load(path) as (_data_from, data_to):
        data_to.node_groups = ["Smoke Simulation"]
    group = data_to.node_groups[0]
    check(group.is_modifier, "Smoke Simulation is not offered as a modifier")
    check(group.asset_data.catalog_id == smoke_assets.CATALOG_PHYSICS[0], "wrong catalog")
    # What Add Modifier filters on: a geometry node tree with the modifier trait (1 << 6).
    check(group.asset_data["type"] == 3, "the asset is not a geometry node tree")
    check(group.asset_data["geometry_node_asset_traits_flag"] & (1 << 6),
          "the asset lacks the modifier trait")


def test_settings_changed_mid_simulation():
    """Blender keeps a simulation's state when an input changes part-way through; the node then
    moves the smoke onto the new grid instead of stopping with a transform error."""
    reset()
    emitter, modifier = smoke_emitter(**{"Voxel Size": 0.0625})
    before = run_until(emitter, 8)
    set_input(modifier, "Voxel Size", 0.05)
    notes = []
    for frame in range(9, 13):
        after = Smoke(emitter, frame)
        notes += after.warnings
        check(not [w for w in after.warnings if w[0] == "ERROR"],
              f"frame {frame}: {after.warnings}")
    print(f"  smoke top: frame 8 {before.top:.3f}, frame 12 {after.top:.3f}")
    check(any("moved onto the new grid" in w[1] for w in notes),
          "the node did not report moving the smoke")
    check(after.top >= before.top - 0.1, "the smoke was lost when the voxel size changed")

    set_input(modifier, "Domain Size", (1.0, 1.0, 1.0))
    smaller = Smoke(emitter, 13)
    check(not [w for w in smaller.warnings if w[0] == "ERROR"], f"{smaller.warnings}")
    check(smaller.grids == ["density", "pressure", "temperature", "velocity"],
          f"unexpected grids after shrinking the domain: {smaller.grids}")


TESTS = [
    test_assets_install,
    test_modifier_emits_and_rises,
    test_moving_emitter_leaves_a_trail,
    test_collider_shelters,
    test_settings_changed_mid_simulation,
]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    selected = [t for t in TESTS if not argv or argv[0] in t.__name__]
    failures = 0
    for test in selected:
        try:
            test()
            print(f"PASS {test.__name__}", flush=True)
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(selected) - failures}/{len(selected)} smoke modifier tests passed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
