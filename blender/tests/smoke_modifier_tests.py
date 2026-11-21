# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests of the Pyro Emitter and Pyro Simulation modifiers and the Pyro node-group assets.

    blender -b --factory-startup --python smoke_modifier_tests.py

Exits non-zero if any test fails.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import traceback

import bpy
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
sys.path.insert(0, ASSETS)
import smoke_assets  # noqa: E402

# This tree's pyro_tools.py, not a copy Blender loaded from scripts/startup.
_spec = importlib.util.spec_from_file_location("pyro_tools", os.path.join(ASSETS,
                                                                          "pyro_tools.py"))
pyro_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pyro_tools)
sys.modules["pyro_tools"] = pyro_tools
pyro_tools.register()


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
    if name == "Preset":  # Load its values now, as the Pyro tab's handler does on an update.
        pyro_tools.load_preset(modifier.id_data, modifier, force=True)


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


def smoke_emitter(probe=("density", 0.02), emitter=None, **inputs):
    """A small cube with a Pyro Emitter (settings in `emitter`, Smoke preset by default), a Pyro
    Simulation (Custom look, settings in `inputs`) and a probe that meshes a grid."""
    obj = add_cube("Emitter", (0.3, -0.2, 0.25), 0.3)
    emitter_group, simulation_group = smoke_assets.build_assets(mark=False)
    emitting = obj.modifiers.new("Pyro Emitter", "NODES")
    emitting.node_group = emitter_group
    for name, value in {"Preset": "Smoke", **(emitter or {})}.items():
        set_input(emitting, name, value)
    modifier = obj.modifiers.new("Pyro", "NODES")
    modifier.node_group = simulation_group
    for name, value in {"Preset": "Custom", **inputs}.items():
        set_input(modifier, name, value)
    probe_modifier = obj.modifiers.new("Probe", "NODES")
    probe_modifier.node_group = probe_group(*probe)
    return obj, modifier


def emitted_points(obj, frame):
    """The pyro points of an object's Pyro Emitter alone, at a frame: positions (world) and the
    named float or vector attributes."""
    for modifier in obj.modifiers[1:]:
        modifier.show_viewport = False
    bpy.context.scene.frame_set(frame)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    geometry = evaluated.evaluated_geometry()
    cloud = geometry.pointcloud
    result = {}
    if cloud is not None and len(cloud.attributes["position"].data):
        for name, width in (("position", 3), ("velocity", 3), ("density", 1), ("fuel", 1),
                            ("expansion", 1), ("radius", 1)):
            if name not in cloud.attributes:
                continue
            data = cloud.attributes[name].data
            values = np.zeros(len(data) * width)
            data.foreach_get("vector" if width == 3 else "value", values)
            result[name] = values.reshape(-1, 3) if width == 3 else values
        matrix = np.array(evaluated.matrix_world)
        result["position"] = result["position"] @ matrix[:3, :3].T + matrix[:3, 3]
    for modifier in obj.modifiers[1:]:
        modifier.show_viewport = True
    return result


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
        self.warnings = [(w.type, w.message) for w in emitter.modifiers["Pyro"].node_warnings]
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
    # Pressure stays on the GPU; the CPU fallback (no Vulkan device) still writes it.
    check([g for g in late.grids if g != "pressure"] == ["density", "temperature", "velocity"],
          f"unexpected grids: {late.grids}")
    check(late.top > early.top + 0.2, "the smoke did not rise above the emitter")


def test_moving_emitter_leaves_a_trail():
    reset()
    emitter, _modifier = smoke_emitter(emitter={"Preset": "Custom", "Emit Type": "Volume",
                                                "Temperature": 0.0, "Initial Velocity": False},
                                       **{"Voxel Size": 0.0625, "Heat Buoyancy": 0.0})
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
    check(os.path.exists(os.path.join(os.path.dirname(folder), "scripts", "startup",
                                      smoke_assets.TOOLS_FILE)), "the Pyro tab is not installed")
    path = os.path.join(folder, "assets", "nodes", smoke_assets.ASSET_FILE)
    with bpy.data.libraries.load(path, assets_only=True) as (data_from, _data_to):
        names = sorted(data_from.node_groups)
    expected = sorted(smoke_assets.BLOCK_NAMES + smoke_assets.MODIFIER_NAMES)
    check(names == expected, f"assets: {names}")
    with bpy.data.libraries.load(path) as (_data_from, data_to):
        data_to.node_groups = list(smoke_assets.MODIFIER_NAMES)
    for group in data_to.node_groups:
        check(group.is_modifier, f"{group.name} is not offered as a modifier")
        check(group.asset_data.catalog_id == smoke_assets.CATALOG_PHYSICS[0], "wrong catalog")
        # What Add Modifier filters on: a geometry node tree with the modifier trait (1 << 6).
        check(group.asset_data["type"] == 3, "the asset is not a geometry node tree")
        check(group.asset_data["geometry_node_asset_traits_flag"] & (1 << 6),
              f"{group.name} lacks the modifier trait")


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
    check([g for g in smaller.grids if g != "pressure"] == ["density", "temperature", "velocity"],
          f"unexpected grids after shrinking the domain: {smaller.grids}")


def test_campfire_burns():
    """The Campfire emitter with the Fire look: fuel burns, flame rises above the emitter."""
    reset()
    emitter, _modifier = smoke_emitter(probe=("flame", 0.3), emitter={"Preset": "Campfire"},
                                       **{"Voxel Size": 0.0625, "Preset": "Fire"})
    fire = run_until(emitter, 15)
    print(f"  flame from z {fire.low:.2f} to {fire.top:.2f}; {fire.warnings}")
    check(not [w for w in fire.warnings if w[0] == "ERROR"], f"errors: {fire.warnings}")
    check(fire.has_flame, "the campfire has no flame grid")
    check(fire.top > 0.4 + 0.1, "no flame rises above the emitter")


def test_explosion_spreads():
    """The Explosion emitter and look throw smoke out far and fast, where plain smoke stays near
    the emitter."""
    def width(emitter, look):
        reset()
        obj, _modifier = smoke_emitter(emitter={"Preset": emitter},
                                       **{"Voxel Size": 0.0625, "Preset": look,
                                          "Domain Size": (4.0, 4.0, 4.0)})
        smoke = run_until(obj, 8)
        check(not [w for w in smoke.warnings if w[0] == "ERROR"], f"errors: {smoke.warnings}")
        return smoke.right - smoke.left, smoke

    calm, _ = width("Smoke", "Smoke")
    blast, smoke = width("Explosion", "Explosion")
    print(f"  smoke width after 8 frames: smoke {calm:.2f}, explosion {blast:.2f}; "
          f"flame grid {smoke.has_flame}")
    check(blast > calm + 0.6, "the explosion did not throw smoke outwards")
    check(smoke.has_flame, "the explosion did not burn")


def test_burst_is_spiky():
    """The Explosion burst: points in about a hundred tapering spikes from a core, growing fast
    and then slower, moving outward, emitting fuel and expansion, and gone after its bursts."""
    reset()
    obj, _modifier = smoke_emitter(emitter={"Preset": "Explosion", "Seed": 3})
    # The burst's own motion: one burst, no added noise or drift (Custom keeps the rest).
    emitting = obj.modifiers["Pyro Emitter"]
    explosion = smoke_assets.EMITTER_PRESETS["Explosion"]
    set_input(emitting, "Preset", "Custom")
    for name, value in explosion.items():
        if name not in ("Emit Type",):
            set_input(emitting, name, value)
    set_input(emitting, "Emit Type", "Burst")
    for name, value in (("Bursts", 1), ("Speed Noise", 0.0), ("Velocity", (0.0, 0.0, 0.0))):
        set_input(emitting, name, value)
    centre = np.array(obj.location) + np.array([0.0, 0.0, 0.0])
    reach = {}
    for frame in (1, 3, 5, 7, 9):
        points = emitted_points(obj, frame)
        if not points:
            reach[frame] = 0.0
            continue
        offset = points["position"] - centre
        distance = np.linalg.norm(offset, axis=1)
        reach[frame] = float(distance.max())
        if frame == 3:
            # The outer half of the burst: spikes fill only a few narrow cones of directions,
            # where a ball would fill them all.
            outer = distance > 0.5 * distance.max()
            directions = offset[outer] / distance[outer][:, None]
            heading = points["velocity"][outer]
            outward = np.einsum("ij,ij->i", directions, heading / np.maximum(
                np.linalg.norm(heading, axis=1)[:, None], 1e-9))
            golden = np.pi * (3.0 - np.sqrt(5.0))
            k = np.arange(400)
            z = 1.0 - 2.0 * (k + 0.5) / 400
            bins = np.column_stack([np.sqrt(1 - z * z) * np.cos(golden * k),
                                    np.sqrt(1 - z * z) * np.sin(golden * k), z])
            occupied = len(set(np.argmax(directions @ bins.T, axis=1))) / len(bins)
            print(f"  frame 3: {len(distance)} points, the outer {len(directions)} in "
                  f"{occupied:.0%} of the directions, mean outward alignment "
                  f"{outward.mean():.2f}, fuel {points['fuel'].sum():.0f}, expansion up to "
                  f"{points['expansion'].max():.1f} 1/s")
            check(len(directions) > 200, "too few points in the spikes")
            check(occupied < 0.6, "the points are not along spikes")
            check(outward.mean() > 0.85, "the burst does not move outward")
            check(points["fuel"].sum() > 0 and points["expansion"].max() > 0,
                  "the burst emits no fuel or expansion")
    print(f"  reach by frame: {reach}")
    check(reach[3] > reach[1] and reach[5] > reach[3], "the burst does not grow")
    check(reach[3] - reach[1] > reach[5] - reach[3], "the burst does not slow down as it grows")
    check(reach[9] == 0.0, "the burst still emits after its duration")


def test_trails_fly_and_fall():
    """Debris Trails: trails thrown upward rise and then fall under world gravity, and turning
    the emitter on its side throws them sideways while they still fall; Show Paths adds one
    curve per trail."""
    for turned in (False, True):
        reset()
        obj, _modifier = smoke_emitter(emitter={"Preset": "Debris Trails", "Seed": 2})
        if turned:
            obj.rotation_euler = (math.pi / 2, 0.0, 0.0)  # Its up is now world -y.
        start = np.array(obj.location)
        heights = {}
        for frame in (6, 30):
            points = emitted_points(obj, frame)
            check(points and len(points["position"]) > 20, f"no trail points at frame {frame}")
            offset = points["position"] - start
            heights[frame] = float(offset[:, 2].mean())
            if frame == 6:
                sideways = float(-offset[:, 1].mean())
                falling = float(points["velocity"][:, 2].mean())
        print(f"  {'on its side' if turned else 'upright'}: mean height {heights[6]:.2f} at "
              f"frame 6, {heights[30]:.2f} at frame 30; out along -y {sideways:.2f}; "
              f"vertical speed at frame 6 {falling:.2f}")
        if turned:
            check(sideways > 0.15, "the turned emitter does not throw its trails sideways")
            check(heights[30] < heights[6] - 0.2, "the trails do not fall under world gravity")
        else:
            check(heights[6] > 0.15, "the trails do not rise")
            check(heights[30] < heights[6], "the trails do not come down")
    set_input(obj.modifiers["Pyro Emitter"], "Show Paths", True)
    for modifier in obj.modifiers[1:]:
        modifier.show_viewport = False
    bpy.context.scene.frame_set(6)
    geometry = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).evaluated_geometry()
    curves = geometry.curves
    count = len(curves.curves) if curves is not None else 0
    print(f"  Show Paths: {count} curves")
    check(count == smoke_assets.EMITTER_PRESETS["Debris Trails"]["Trails"],
          "Show Paths does not give one curve per trail")


def test_emitter_motion_gives_velocity():
    """A Pyro Emitter on a cube moving 0.05 per frame at 30 fps gives its points 1.5 m/s with
    Source 1, and no velocity with Initial Velocity off."""
    reset()
    obj, _modifier = smoke_emitter(emitter={"Preset": "Custom", "Emit Type": "Volume",
                                            "Initial Velocity": True, "Source": 1.0})
    for frame in range(1, 6):
        obj.location = (0.0, 0.05 * (frame - 1), 0.5)
        points = emitted_points(obj, frame)
    mean = points["velocity"].mean(axis=0)
    print(f"  emitted velocity {np.round(mean, 3)} (the cube moves 1.5 m/s in y)")
    check(abs(mean[1] - 1.5) < 0.05 and abs(mean[0]) < 0.05 and abs(mean[2]) < 0.05,
          "the emitter's motion is not its points' velocity")
    set_input(obj.modifiers["Pyro Emitter"], "Initial Velocity", False)
    points = emitted_points(obj, 5)
    check("velocity" not in points, "points carry a velocity with Initial Velocity off")


def test_initial_velocity_from_motion():
    """A moving emitter with Initial Velocity gives the smoke its motion: without it the smoke
    stays where it was emitted, a trail back to the start; with it the smoke travels along."""
    def back(initial):
        reset()
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
        emitter, _modifier = smoke_emitter(emitter={"Preset": "Custom", "Emit Type": "Volume",
                                                    "Temperature": 0.0,
                                                    "Initial Velocity": initial, "Source": 1.0},
                                           **{"Voxel Size": 0.0625, "Heat Buoyancy": 0.0})
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


def test_emitters_collection_and_plain_smoke():
    """A Pyro Simulation on another object simulates the emitters in its Emitters collection;
    on a plain mesh without emitters it emits smoke from the mesh itself."""
    reset()
    emitter_group, simulation_group = smoke_assets.build_assets(mark=False)
    source = add_cube("Source", (1.0, 0.0, 0.3), 0.3)
    emitting = source.modifiers.new("Pyro Emitter", "NODES")
    emitting.node_group = emitter_group
    set_input(emitting, "Preset", "Smoke")
    emitters = bpy.data.collections.new("Emitters")
    bpy.context.scene.collection.children.link(emitters)
    for owner in list(source.users_collection):
        owner.objects.unlink(source)
    emitters.objects.link(source)
    domain = add_cube("Domain", (0.0, 0.0, 0.0), 0.1)
    simulating = domain.modifiers.new("Pyro", "NODES")
    simulating.node_group = simulation_group
    for name, value in {"Preset": "Custom", "Voxel Size": 0.0625, "Emitters": emitters,
                        "Domain Size": (4.0, 4.0, 4.0), "Show Mesh": False}.items():
        set_input(simulating, name, value)
    domain.modifiers.new("Probe", "NODES").node_group = probe_group()
    smoke = run_until(domain, 12)
    print(f"  smoke from the collection's emitter spans x {smoke.left:.2f} to {smoke.right:.2f}")
    check(not [w for w in smoke.warnings if w[0] == "ERROR"], f"errors: {smoke.warnings}")
    check(0.7 < smoke.left < 1.0 < smoke.right < 1.3, "the smoke is not at the emitter")

    reset()
    _e, simulation_group = smoke_assets.build_assets(mark=False)
    plain = add_cube("Plain", (0.0, 0.0, 0.3), 0.3)
    modifier = plain.modifiers.new("Pyro", "NODES")
    modifier.node_group = simulation_group
    set_input(modifier, "Preset", "Smoke")  # What adding it in Blender shows.
    set_input(modifier, "Voxel Size", 0.0625)
    plain.modifiers.new("Probe", "NODES").node_group = probe_group()
    early, late = run_until(plain, 3), run_until(plain, 15)
    print(f"  plain mesh: smoke top {early.top:.2f} at frame 3, {late.top:.2f} at frame 15")
    check(late.top > early.top + 0.15, "a plain mesh did not emit rising smoke")


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


def test_viewport_shows_flame():
    """Fire shows in the viewport: a 'glow' grid for the material, density as the first grid
    (all Solid view draws), the flame as a second, orange volume for Solid view (viewport
    only), and Smoke Density and Viewport Density change only the look."""
    reset()
    emitter, modifier = smoke_emitter(emitter={"Preset": "Campfire"}, Preset="Fire",
                                      **{"Voxel Size": 0.05})
    smoke = run_until(emitter, 10)

    def solid_flame(frame):
        """The grids of the orange flame volumes, without the test's probe, and the first
        grid of the shaded volume."""
        emitter.modifiers["Probe"].show_viewport = False
        geometry = Smoke(emitter, frame).geometry
        emitter.modifiers["Probe"].show_viewport = True
        flames = [[g.name for g in ref.volume.grids] for ref in geometry.instance_references()
                  if ref.volume is not None and ref.volume.materials
                  and ref.volume.materials[0].name == "Pyro Flame Preview"]
        return flames, geometry.volume.grids[0].name, geometry.volume.materials[0].name

    flames, first, material = solid_flame(10)
    print(f"  grids {smoke.grids}; first grid {first}, material {material}; Solid view flame "
          f"volumes {flames}")
    check("glow" in smoke.grids and "flame" in smoke.grids, "no flame or glow grid")
    check(first == "density", "Solid view would not draw the smoke")
    check(material == "Pyro Simulation", "the volume has no fire material")
    check(flames == [["flame"]], "no orange flame volume for Solid view")
    set_input(modifier, "Flame In Solid View", False)
    check(solid_flame(11)[0] == [], "the Solid view flame shows with Flame In Solid View off")
    bare = Smoke(emitter, 11)
    # Smoke Density or Viewport Density 0 hides the smoke; the simulation carries on.
    set_input(modifier, "Smoke Density", 0.0)
    hidden = Smoke(emitter, 12)
    set_input(modifier, "Smoke Density", 1.0)
    set_input(modifier, "Viewport Density", 0.0)
    hidden_in_viewport = Smoke(emitter, 13)
    set_input(modifier, "Viewport Density", 1.0)
    shown = Smoke(emitter, 14)
    print(f"  smoke top: {bare.top:.2f} at frame 11, hidden at 12 and 13 ({hidden.top}, "
          f"{hidden_in_viewport.top}), {shown.top:.2f} at 14")
    check(math.isnan(hidden.top), "Smoke Density 0 does not hide the smoke")
    check(math.isnan(hidden_in_viewport.top), "Viewport Density 0 does not hide the smoke")
    check(shown.top >= bare.top - 0.05, "hiding the smoke changed the simulation")


def test_bake_from_the_pyro_tab():
    """The sidebar's Bake bakes the selected Pyro simulations into a pyro_bake folder next to
    the saved file, playback then reads the bake (new settings change nothing), and Delete
    Bake brings the live simulation back."""
    reset()
    emitter, modifier = smoke_emitter(**{"Voxel Size": 0.0625})
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 8
    folder = os.path.join(bpy.app.tempdir, "pyro_bake_test")
    os.makedirs(folder, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder, "bake.blend"))
    emitter = bpy.data.objects[emitter.name]
    modifier = emitter.modifiers["Pyro"]
    live = run_until(emitter, 8)
    bpy.ops.object.select_all(action="DESELECT")
    emitter.select_set(True)
    bpy.context.view_layer.objects.active = emitter
    result = bpy.ops.pyro.bake()
    baked_files = [name for _root, _dirs, files in os.walk(os.path.join(folder, "pyro_bake"))
                   for name in files]
    baked = Smoke(emitter, 8)
    set_input(modifier, "Voxel Size", 0.1)
    still = Smoke(emitter, 8)
    print(f"  bake {result}: {len(baked_files)} files; smoke top live {live.top:.3f}, baked "
          f"{baked.top:.3f}, baked after a new voxel size {still.top:.3f}")
    check(result == {"FINISHED"} and baked_files, "nothing was baked")
    check(abs(baked.top - live.top) < 1e-4, "the bake does not match the simulation")
    check(abs(still.top - baked.top) < 1e-4, "playback does not read the bake")
    check(bpy.ops.pyro.delete_bake() == {"FINISHED"}, "Delete Bake failed")
    again = run_until(emitter, 8)
    print(f"  after Delete Bake, with the new voxel size: smoke top {again.top:.3f}")
    check(abs(again.top - baked.top) > 1e-4, "Delete Bake did not bring the simulation back")


def test_presets_fill_the_panel():
    """Choosing a preset copies its values into the modifier's own inputs (on the next update,
    through the Pyro tab's handler); edited values then count, Custom keeps them, and the
    simulation uses what the panel shows."""
    reset()
    obj = add_cube("Emitter", (0.0, 0.0, 0.3), 0.3)
    emitter_group, simulation_group = smoke_assets.build_assets(mark=False)
    emitting = obj.modifiers.new("Pyro Emitter", "NODES")
    emitting.node_group = emitter_group
    bpy.context.view_layer.update()

    def value(modifier, name):
        return pyro_tools.modifier_input(modifier, name).value

    explosion = smoke_assets.EMITTER_PRESETS["Explosion"]
    print(f"  new Pyro Emitter: preset {value(emitting, 'Preset')}, Spikes "
          f"{value(emitting, 'Spikes')}, Long Spikes {value(emitting, 'Long Spikes'):.2f}")
    check(value(emitting, "Spikes") == explosion["Spikes"], "a new emitter shows no preset")
    pyro_tools.modifier_input(emitting, "Preset").value = "Debris Trails"
    bpy.context.view_layer.update()
    debris = smoke_assets.EMITTER_PRESETS["Debris Trails"]
    print(f"  Debris Trails: Emit Type {value(emitting, 'Emit Type')}, Trails "
          f"{value(emitting, 'Trails')}, Trail Speed {value(emitting, 'Trail Speed'):.1f}")
    check(value(emitting, "Emit Type") == "Trails" and value(emitting, "Trails") ==
          debris["Trails"], "choosing a preset did not fill in its values")
    pyro_tools.modifier_input(emitting, "Trails").value = 3
    bpy.context.view_layer.update()
    check(value(emitting, "Trails") == 3, "an edited value was put back")
    trails = smoke_assets.input_item(emitter_group, "Trails").identifier
    for modifier in obj.modifiers[1:]:
        modifier.show_viewport = False
    points = emitted_points(obj, 6)
    print(f"  with Trails edited to 3: {len(points['position'])} points at frame 6")
    check(0 < len(points["position"]) < debris["Trails"] * 60, "the edit was not used")
    pyro_tools.modifier_input(emitting, "Preset").value = "Custom"
    bpy.context.view_layer.update()
    check(getattr(emitting.properties.inputs, trails).value == 3, "Custom changed the values")

    simulating = obj.modifiers.new("Pyro", "NODES")
    simulating.node_group = simulation_group
    bpy.context.view_layer.update()
    smoke = smoke_assets.SOLVER_PRESETS["Smoke"]
    check(value(simulating, "Turbulence") == smoke["Turbulence"],
          "a new Pyro Simulation shows no preset")
    pyro_tools.modifier_input(simulating, "Preset").value = "Explosion"
    bpy.context.view_layer.update()
    print(f"  Pyro Simulation, Explosion: Gas Release {value(simulating, 'Gas Release'):.1f}, "
          f"Disturbance {value(simulating, 'Disturbance'):.1f}")
    check(abs(value(simulating, "Gas Release") -
              smoke_assets.SOLVER_PRESETS["Explosion"]["Gas Release"]) < 1e-6,
          "the simulation preset did not fill in its values")


def test_speed_settings():
    """Every Speed panel setting at once: the Explosion still simulates, rises and burns."""
    reset()
    obj, modifier = smoke_emitter(
        probe=("density", 0.02), emitter={"Preset": "Explosion"},
        **{"Voxel Size": 0.0625, "Preset": "Explosion", "Domain Size": (3.0, 3.0, 3.0),
           "Open Borders": True, "Smoke Fade": 0.5, "Heat Fade": 0.3,
           "Pressure Quality": "Medium", "Smoke Threshold": 0.001,
           "Directional Margin": True, "Fast Turbulence": True, "Half Precision": True,
           "Store Velocity": False, "Batch GPU Work": False})
    smoke = run_until(obj, 10)
    print(f"  smoke from x {smoke.left:.2f} to {smoke.right:.2f}, top {smoke.top:.2f}; "
          f"flame {smoke.has_flame}; {smoke.warnings}")
    check(not [w for w in smoke.warnings if w[0] == "ERROR"], f"errors: {smoke.warnings}")
    check(smoke.right - smoke.left > 0.3, "the explosion did not spread")
    check(smoke.has_flame, "the explosion did not burn")


TESTS = [
    test_assets_install,
    test_modifier_emits_and_rises,
    test_moving_emitter_leaves_a_trail,
    test_collider_shelters,
    test_settings_changed_mid_simulation,
    test_campfire_burns,
    test_explosion_spreads,
    test_burst_is_spiky,
    test_trails_fly_and_fall,
    test_viewport_shows_flame,
    test_bake_from_the_pyro_tab,
    test_presets_fill_the_panel,
    test_emitter_motion_gives_velocity,
    test_initial_velocity_from_motion,
    test_emitters_collection_and_plain_smoke,
    test_turbulence_layers_chain_in_order,
    test_speed_settings,
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
