# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of the Pyro Simulation modifier and the Pyro node-group assets.

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
    item = smoke_assets.input_item(modifier.node_group, name)
    getattr(modifier.properties.inputs, item.identifier).value = value


def probe_group(grid="density", threshold=0.02):
    """A second modifier that keeps the grid names and turns a grid into a mesh to measure."""
    name = f"Probe {grid} {threshold}"
    tree = bpy.data.node_groups.get(name)
    if tree:
        return tree
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    get = nodes.new("GeometryNodeGetNamedGrid")
    get.data_type = "FLOAT"
    get.inputs["Name"].default_value = grid
    get.inputs["Remove"].default_value = False
    links.new(group_in.outputs[0], get.inputs["Volume"])
    to_mesh = nodes.new("GeometryNodeGridToMesh")
    to_mesh.inputs["Threshold"].default_value = threshold
    links.new(get.outputs["Grid"], to_mesh.inputs["Grid"])
    # The smoke's surface, without the emitter's mesh.
    separate = nodes.new("GeometryNodeSeparateComponents")
    links.new(group_in.outputs[0], separate.inputs[0])
    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(to_mesh.outputs[0], join.inputs[0])
    links.new(separate.outputs["Volume"], join.inputs[0])
    links.new(join.outputs[0], group_out.inputs[0])
    return tree


def smoke_emitter(probe=("density", 0.02), **inputs):
    emitter = add_cube("Emitter", (0.3, -0.2, 0.25), 0.3)
    modifier = emitter.modifiers.new("Pyro", "NODES")
    modifier.node_group = smoke_assets.build_assets(mark=False)
    for name, value in inputs.items():
        set_input(modifier, name, value)
    probe_modifier = emitter.modifiers.new("Probe", "NODES")
    probe_modifier.node_group = probe_group(*probe)
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
        points = [world @ v.co for v in mesh.vertices] if mesh else []
        self.front = max((p.y for p in points), default=float("nan"))
        self.back = min((p.y for p in points), default=float("nan"))
        volume_grids = {}
        if volume:
            for grid in volume.grids:
                volume_grids[grid.name] = grid
        self.has_flame = "flame" in volume_grids


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
    for uuid, _path in (smoke_assets.CATALOG_PYRO, smoke_assets.CATALOG_PHYSICS):
        check(sum(line.startswith(uuid) for line in lines) == 1, f"catalog {uuid} not once")
    path = os.path.join(folder, "assets", "nodes", smoke_assets.ASSET_FILE)
    with bpy.data.libraries.load(path, assets_only=True) as (data_from, _data_to):
        names = sorted(data_from.node_groups)
    expected = sorted(smoke_assets.BLOCK_NAMES + (smoke_assets.MODIFIER,))
    check(names == expected, f"assets: {names}")
    with bpy.data.libraries.load(path) as (_data_from, data_to):
        data_to.node_groups = [smoke_assets.MODIFIER]
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


def test_campfire_burns():
    """The Campfire preset emits fuel that burns: flame grids, flame above the emitter, smoke."""
    reset()
    emitter, _modifier = smoke_emitter(probe=("flame", 0.3), **{"Voxel Size": 0.0625,
                                                               "Preset": "Campfire"})
    fire = run_until(emitter, 15)
    print(f"  flame from z {fire.low:.2f} to {fire.top:.2f}; {fire.warnings}")
    check(not [w for w in fire.warnings if w[0] == "ERROR"], f"errors: {fire.warnings}")
    check(fire.has_flame, "the campfire has no flame grid")
    check(fire.top > 0.4 + 0.1, "no flame rises above the emitter")


def test_explosion_spreads():
    """The Explosion preset bursts particles off the surface: its smoke spreads far sideways
    within a few frames, where plain smoke stays near the emitter."""
    def width(**inputs):
        reset()
        emitter, _modifier = smoke_emitter(**{"Voxel Size": 0.0625, **inputs})
        smoke = run_until(emitter, 8)
        check(not [w for w in smoke.warnings if w[0] == "ERROR"], f"errors: {smoke.warnings}")
        return smoke.right - smoke.left, smoke

    calm, _ = width(Preset="Smoke")
    blast, smoke = width(Preset="Explosion")
    print(f"  smoke width after 8 frames: smoke preset {calm:.2f}, explosion {blast:.2f}; "
          f"flame grid {smoke.has_flame}")
    check(blast > calm + 0.4, "the explosion did not throw smoke outwards")
    check(smoke.has_flame, "the explosion did not burn")


def test_initial_velocity_from_motion():
    """A moving emitter with Initial Velocity gives the smoke its motion: without it the smoke
    stays where it was emitted, a trail back to the start; with it the smoke travels along."""
    def back(initial):
        reset()
        emitter, _modifier = smoke_emitter(**{"Voxel Size": 0.0625, "Heat Buoyancy": 0.0,
                                              "Temperature": 0.0, "Initial Velocity": initial,
                                              "Source": 1.0})
        # Steady motion from the first frame, when the back of the trail is emitted.
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
        emitter.location = (0.0, -0.4, 0.5)
        emitter.keyframe_insert("location", frame=1)
        emitter.location = (0.0, 0.4, 0.5)
        emitter.keyframe_insert("location", frame=13)
        smoke = run_until(emitter, 13)
        check(not [w for w in smoke.warnings if w[0] == "ERROR"], f"errors: {smoke.warnings}")
        return smoke.back

    still, carried = back(False), back(True)
    print(f"  smoke trail starts at y {still:.2f} without, {carried:.2f} with initial velocity")
    check(carried > still + 0.08, "the emitter's motion did not carry the smoke")


def test_source_velocity_from_motion():
    """Pyro Source turns the mesh's motion since the previous mesh into the emitted velocity,
    adds the normal and vector velocities, and leaves velocity out unless asked."""
    reset()
    smoke_assets.build_assets(mark=False)

    def emitted(**inputs):
        tree = bpy.data.node_groups.new("Source Probe", "GeometryNodeTree")
        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        nodes, links = tree.nodes, tree.links
        cube = nodes.new("GeometryNodeMeshCube")
        cube.inputs["Size"].default_value = (0.4, 0.4, 0.4)
        moved = nodes.new("GeometryNodeTransform")
        links.new(cube.outputs[0], moved.inputs["Geometry"])
        moved.inputs["Translation"].default_value = (0.0, -0.05, 0.0)
        source = nodes.new("GeometryNodeGroup")
        source.node_tree = bpy.data.node_groups["Pyro Source"]
        links.new(cube.outputs[0], source.inputs["Mesh"])
        links.new(moved.outputs[0], source.inputs["Previous Mesh"])
        source.inputs["Delta Time"].default_value = 1.0 / 30.0
        source.inputs["Voxel Size"].default_value = 0.05
        for name, value in inputs.items():
            source.inputs[name].default_value = value
        get = nodes.new("GeometryNodeGetNamedGrid")
        get.data_type = "VECTOR"
        get.inputs["Name"].default_value = "velocity"
        links.new(source.outputs[0], get.inputs["Volume"])
        sample = nodes.new("GeometryNodeSampleGrid")
        sample.data_type = "VECTOR"
        links.new(get.outputs["Grid"], sample.inputs["Grid"])
        point = nodes.new("GeometryNodePoints")
        store = nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type = "FLOAT_VECTOR"
        store.inputs["Name"].default_value = "v"
        links.new(point.outputs[0], store.inputs["Geometry"])
        links.new(sample.outputs[0], store.inputs["Value"])
        out = nodes.new("NodeGroupOutput")
        join = nodes.new("GeometryNodeJoinGeometry")
        links.new(store.outputs[0], join.inputs[0])
        links.new(source.outputs[0], join.inputs[0])
        links.new(join.outputs[0], out.inputs[0])
        obj = bpy.data.objects.new("probe", bpy.data.meshes.new("probe"))
        bpy.context.scene.collection.objects.link(obj)
        obj.modifiers.new("Probe", "NODES").node_group = tree
        geometry = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).evaluated_geometry()
        grids = sorted(g.name for g in geometry.volume.grids) if geometry.volume else []
        value = tuple(geometry.pointcloud.attributes["v"].data[0].vector)
        bpy.data.objects.remove(obj)
        bpy.data.node_groups.remove(tree)
        return grids, value

    grids, _ = emitted()
    check("velocity" not in grids, f"velocity emitted without Initial Velocity: {grids}")
    grids, moving = emitted(**{"Initial Velocity": True, "Source": 1.0})
    print(f"  the cube moved 0.05 in y in 1/30 s; emitted velocity {moving}")
    check("velocity" in grids, "no velocity grid with Initial Velocity")
    check(abs(moving[1] - 1.5) < 0.05 and abs(moving[0]) < 0.05 and abs(moving[2]) < 0.05,
          "the emitted velocity is not the mesh's motion")
    _, both = emitted(**{"Initial Velocity": True, "Source": 0.5, "Velocity": (0.0, 0.0, 2.0)})
    check(abs(both[1] - 0.75) < 0.05 and abs(both[2] - 2.0) < 0.05,
          f"Source and Velocity do not combine: {both}")


def test_turbulence_layers_chain_in_order():
    """Turbulence Layer nodes chained together give one point per layer, in chain order, with
    their settings as attributes."""
    reset()
    smoke_assets.build_assets(mark=False)
    tree = bpy.data.node_groups.new("Chain", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    out = tree.nodes.new("NodeGroupOutput")
    layers = None
    for strength, mask in ((1.0, "None"), (2.0, "Speed"), (3.0, "Flame")):
        node = tree.nodes.new("GeometryNodeGroup")
        node.node_tree = bpy.data.node_groups["Turbulence Layer"]
        node.inputs["Strength"].default_value = strength
        node.inputs["Mask"].default_value = mask
        if layers is not None:
            tree.links.new(layers, node.inputs["Layers"])
        layers = node.outputs["Layers"]
    tree.links.new(layers, out.inputs[0])
    obj = bpy.data.objects.new("chain", bpy.data.meshes.new("chain"))
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("Chain", "NODES").node_group = tree
    geometry = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).evaluated_geometry()
    cloud = geometry.pointcloud
    strengths = [d.value for d in cloud.attributes["strength"].data]
    masks = [d.value for d in cloud.attributes["mask"].data]
    print(f"  strengths {strengths}, masks {masks}")
    check(strengths == [1.0, 2.0, 3.0], "the layers are not in chain order")
    check(masks == [0, 4, 3], "the masks were not stored")


TESTS = [
    test_assets_install,
    test_modifier_emits_and_rises,
    test_moving_emitter_leaves_a_trail,
    test_collider_shelters,
    test_settings_changed_mid_simulation,
    test_campfire_burns,
    test_explosion_spreads,
    test_initial_velocity_from_motion,
    test_source_velocity_from_motion,
    test_turbulence_layers_chain_in_order,
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
