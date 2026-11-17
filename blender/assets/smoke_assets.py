# SPDX-License-Identifier: GPL-2.0-or-later
"""Build the Pyro node-group assets and install them into a Blender build.

    blender -b --factory-startup --python smoke_assets.py

writes `datafiles/assets/nodes/pyro_simulation_assets.blend` next to the running Blender and
adds its catalogs to `datafiles/assets/blender_assets.cats.txt`, so that:

- Add Modifier > Physics > Pyro Simulation turns any mesh into a smoke or fire emitter, with
  presets (Smoke, Large Plume, Trail, Explosion, Campfire, Candle), and
- Add > Pyro offers the building blocks the modifier is made of, for setups built by hand around
  the Gas Simulation Step node: Pyro Preset, Pyro Particles, Pyro Emitter, Pyro Domain,
  Pyro Colliders, Pyro Solver, Pyro Display, and the lower-level Pyro Source, Pyro Point Source,
  Pyro Collider and Turbulence Layer.

The modifier's blocks pass one Settings bundle, made by Pyro Preset, instead of dozens of
links. `build_assets()` can also be imported to create the groups in the current file (the tests
do).
"""

from __future__ import annotations

import os
import sys

import bpy

CATALOG_PYRO = ("689f4607-3a1d-4e2a-a49b-8166dad896ca", "Geometry Nodes/Pyro")
CATALOG_PHYSICS = ("4c0f1d1e-6b0a-4a8e-9d43-5d1f3a6b7c21", "Geometry Nodes/Physics")
ASSET_FILE = "pyro_simulation_assets.blend"
MODIFIER = "Pyro Simulation"
GAS_NODE = "GeometryNodeMantaGasStep"


class Builder:
    def __init__(self, tree):
        self.tree = tree

    def new(self, idname, location=(0, 0), **props):
        node = self.tree.nodes.new(idname)
        node.location = location
        for key, value in props.items():
            setattr(node, key, value)
        return node

    def link(self, out_socket, in_socket):
        self.tree.links.new(out_socket, in_socket)

    def set(self, socket, value):
        if value is None:  # Left unconnected, such as an empty geometry.
            return
        if hasattr(value, "node"):
            self.link(value, socket)
        else:
            socket.default_value = value

    def math(self, operation, a, b=None):
        node = self.new("ShaderNodeMath", operation=operation)
        self.set(node.inputs[0], a)
        if b is not None:
            self.set(node.inputs[1], b)
        return node.outputs[0]

    def equal(self, a, b):
        """1.0 where two whole numbers are equal, else 0.0."""
        node = self.new("ShaderNodeMath", operation="COMPARE")
        self.set(node.inputs[0], a)
        self.set(node.inputs[1], b)
        node.inputs[2].default_value = 0.5
        return node.outputs[0]

    def vmath(self, operation, a, b=None, scale=None):
        node = self.new("ShaderNodeVectorMath", operation=operation)
        self.set(node.inputs[0], a)
        if b is not None:
            self.set(node.inputs[1], b)
        if scale is not None:
            self.set(node.inputs["Scale"], scale)
        return node.outputs[0]


def arrange(tree):
    """Lay the nodes out left to right in columns by how far they are from the group input, so
    every link is visible: each node sits one column after the furthest node that feeds it,
    group outputs in the last column, and a node fed by nothing just before its first user."""
    nodes = list(tree.nodes)
    feeds = {node: set() for node in nodes}
    users = {node: set() for node in nodes}
    for link in tree.links:
        feeds[link.to_node].add(link.from_node)
        users[link.from_node].add(link.to_node)

    column = {}

    def depth(node, visiting=()):
        if node in column:
            return column[node]
        if node in visiting:  # Not expected in a node tree; break any cycle.
            return 0
        if node.bl_idname == "NodeGroupInput":
            column[node] = 0
            return 0
        value = max((depth(f, visiting + (node,)) + 1 for f in feeds[node]), default=0)
        column[node] = value
        return value

    for node in nodes:
        depth(node)
    last = max(column.values(), default=0) + 1
    for node in nodes:
        if node.bl_idname == "NodeGroupOutput":
            column[node] = last
    # Constant sources (Self Object, Scene Time, Position, ...) one column before their first
    # user. Every user is at least in column 1, so this never shares a column with one.
    for node in nodes:
        if not feeds[node] and node.bl_idname != "NodeGroupInput" and users[node]:
            column[node] = min(column[u] for u in users[node]) - 1

    base_properties = {prop.identifier for prop in bpy.types.Node.bl_rna.properties}

    def height(node):
        """An upper estimate of the node's drawn height: header, one row per socket and one per
        property drawn in its body (operation, data type, clamp, node group, ...)."""
        sockets = [x for x in list(node.inputs) + list(node.outputs) if x.enabled and not x.hide]
        properties = [prop for prop in node.bl_rna.properties
                      if prop.identifier not in base_properties and not prop.is_readonly
                      and not prop.identifier.startswith(("active_", "is_"))]
        return 60 + 22 * len(sockets) + 28 * len(properties)

    by_column = {}
    for node in nodes:
        by_column.setdefault(column[node], []).append(node)
    x = 0.0
    placed_y = {}
    for index in sorted(by_column):
        members = by_column[index]
        # Order each column by where its feeding nodes sit, to keep links short and uncrossed.
        def key(node):
            ys = [placed_y[f] for f in feeds[node] if f in placed_y]
            return -(sum(ys) / len(ys)) if ys else 0.0
        members.sort(key=key)
        total = sum(height(n) for n in members) + 40 * (len(members) - 1)
        y = total / 2
        for node in members:
            node.location = (x, y)
            placed_y[node] = y - height(node) / 2
            y -= height(node) + 40
        x += max(n.width for n in members) + 80


def socket(tree, name, kind, in_out="INPUT", default=None, minimum=None, maximum=None,
           description="", parent=None, subtype=None):
    item = tree.interface.new_socket(name, in_out=in_out, socket_type=kind, parent=parent)
    if default is not None:
        item.default_value = default
    if minimum is not None:
        item.min_value = minimum
    if maximum is not None:
        item.max_value = maximum
    if subtype is not None:
        try:
            item.subtype = subtype
        except TypeError:
            pass  # Not every socket type offers every unit.
    item.description = description
    return item


def input_item(tree, name):
    """The group input called `name` (a panel may have the same name)."""
    return next(item for item in tree.interface.items_tree
                if getattr(item, "in_out", None) == "INPUT" and item.name == name)


def new_group(name, description):
    tree = bpy.data.node_groups.get(name)
    if tree is not None:
        bpy.data.node_groups.remove(tree)
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.description = description
    return tree


def store_grid(b, volume, name, grid, kind="FLOAT"):
    store = b.new("GeometryNodeStoreNamedGrid", data_type=kind)
    store.inputs["Name"].default_value = name
    if volume is not None:
        b.link(volume, store.inputs["Volume"])
    b.link(grid, store.inputs["Grid"])
    return store.outputs["Volume"]


def velocity_grid(b, mesh, velocity, topology):
    """A vector grid on `topology` holding the velocity field of the mesh's nearest surface."""
    capture = b.new("GeometryNodeCaptureAttribute", domain="POINT")
    capture.capture_items.new("VECTOR", "Velocity")
    b.link(mesh, capture.inputs["Geometry"])
    b.link(velocity, capture.inputs["Velocity"])
    sample = b.new("GeometryNodeSampleNearestSurface", data_type="FLOAT_VECTOR")
    b.link(capture.outputs["Geometry"], sample.inputs["Mesh"])
    b.link(capture.outputs["Velocity"], sample.inputs["Value"])
    to_grid = b.new("GeometryNodeFieldToGrid", data_type="VECTOR")
    to_grid.grid_items.clear()
    to_grid.grid_items.new("VECTOR", "Velocity")
    b.link(topology, to_grid.inputs["Topology"])
    b.link(sample.outputs["Value"], to_grid.inputs["Velocity"])
    return to_grid.outputs["Velocity"]


# --------------------------------------------------------------------------------------------
# Settings: everything a preset can set, with the Custom value the modifier panel shows.

# (name, kind, custom default, description). Kinds are bundle item types.
PRESET_ITEMS = (
    ("Emit From", "INT", 0, "0 emits from the mesh's volume, 1 from particles"),
    ("Density", "FLOAT", 1.0, "Smoke emitted"),
    ("Temperature", "FLOAT", 1.0, "Heat emitted; with heat buoyancy it makes the smoke rise"),
    ("Fuel", "FLOAT", 0.0, "Fuel emitted; it burns into flame, heat and smoke"),
    ("Initial Velocity", "BOOLEAN", False, "Give the emitted smoke a velocity"),
    ("Source", "FLOAT", 1.0, "Share of the emitter's own motion (or each particle's) given to "
                            "the smoke"),
    ("Normal", "FLOAT", 0.0, "Speed along the emitter's surface normals, world units per second"),
    ("Velocity", "VECTOR", (0.0, 0.0, 0.0), "Velocity added everywhere it emits"),
    ("Rate", "FLOAT", 200.0, "Particles: new particles per second"),
    ("Burst", "INT", 0, "Particles: particles created on the first frame"),
    ("Particle Speed", "FLOAT", 1.0, "Particles: speed along the emitter's normals"),
    ("Spread", "FLOAT", 0.5, "Particles: random speed in every direction"),
    ("Drag", "FLOAT", 1.0, "Particles: how fast they slow down, per second"),
    ("Particle Gravity", "FLOAT", 0.0, "Particles: share of gravity pulling them down"),
    ("Lifetime", "FLOAT", 0.8, "Particles: seconds each particle emits, fading out"),
    ("Particle Radius", "FLOAT", 0.08, "Particles: size of the puff each particle emits"),
    ("Burn Rate", "FLOAT", 0.75, "Fuel burnt per second. Higher burns faster, in shorter flames"),
    ("Flame Heat", "FLOAT", 4.0, "Heat per unit of fuel burnt"),
    ("Flame Smoke", "FLOAT", 1.0, "Smoke per unit of fuel burnt"),
    ("Flame Cooling", "FLOAT", 4.0, "Flame lost per second without fuel. Lower: taller flames"),
    ("Heat Buoyancy", "FLOAT", 0.5, "How strongly heat makes smoke rise against gravity"),
    ("Turbulence", "FLOAT", 0.0, "Large swirls, world units per second squared; 0 is off"),
    ("Turbulence Scale", "FLOAT", 0.25, "Size of the large swirls"),
    ("Fine Turbulence", "FLOAT", 0.0, "Small swirls, applied after the large ones and only "
                                      "where the flow moves, so they react to them"),
    ("Fine Scale", "FLOAT", 0.08, "Size of the small swirls"),
)
# Values each preset changes; everything else keeps the Custom value.
PRESETS = {
    "Smoke": {"Emit From": 0, "Density": 1.0, "Temperature": 1.0, "Fuel": 0.0,
              "Heat Buoyancy": 0.5, "Turbulence": 2.0, "Turbulence Scale": 0.3,
              "Fine Turbulence": 1.0, "Fine Scale": 0.08},
    "Large Plume": {"Emit From": 1, "Rate": 400.0, "Burst": 0, "Particle Speed": 3.0,
                    "Spread": 1.0, "Drag": 1.5, "Particle Gravity": 0.0, "Lifetime": 0.8,
                    "Particle Radius": 0.15, "Initial Velocity": True, "Source": 1.0,
                    "Velocity": (0.0, 0.0, 2.0), "Density": 1.0, "Temperature": 2.0,
                    "Fuel": 0.3, "Burn Rate": 1.0, "Flame Heat": 3.0, "Flame Smoke": 2.0,
                    "Flame Cooling": 6.0, "Heat Buoyancy": 1.0, "Turbulence": 4.0,
                    "Turbulence Scale": 0.6, "Fine Turbulence": 2.0, "Fine Scale": 0.15},
    "Trail": {"Emit From": 1, "Rate": 300.0, "Burst": 0, "Particle Speed": 0.3, "Spread": 0.3,
              "Drag": 3.0, "Particle Gravity": 0.0, "Lifetime": 0.5, "Particle Radius": 0.07,
              "Initial Velocity": False, "Density": 1.0, "Temperature": 0.5, "Fuel": 0.0,
              "Heat Buoyancy": 0.3, "Turbulence": 1.5, "Turbulence Scale": 0.2,
              "Fine Turbulence": 1.0, "Fine Scale": 0.06},
    "Explosion": {"Emit From": 1, "Rate": 0.0, "Burst": 400, "Particle Speed": 5.0,
                  "Spread": 2.0, "Drag": 4.0, "Particle Gravity": 0.5, "Lifetime": 0.45,
                  "Particle Radius": 0.12, "Initial Velocity": True, "Source": 1.0,
                  "Velocity": (0.0, 0.0, 0.0), "Density": 0.3, "Temperature": 3.0,
                  "Fuel": 1.5, "Burn Rate": 2.0, "Flame Heat": 4.0, "Flame Smoke": 1.5,
                  "Flame Cooling": 3.0, "Heat Buoyancy": 1.0, "Turbulence": 6.0,
                  "Turbulence Scale": 0.5, "Fine Turbulence": 3.0, "Fine Scale": 0.12},
    "Campfire": {"Emit From": 0, "Density": 0.0, "Temperature": 1.0, "Fuel": 1.0,
                 "Initial Velocity": True, "Source": 0.0, "Normal": 0.0,
                 "Velocity": (0.0, 0.0, 0.5), "Burn Rate": 0.75, "Flame Heat": 4.0,
                 "Flame Smoke": 0.6, "Flame Cooling": 5.0, "Heat Buoyancy": 1.0,
                 "Turbulence": 2.0, "Turbulence Scale": 0.15, "Fine Turbulence": 1.5,
                 "Fine Scale": 0.05},
    "Candle": {"Emit From": 0, "Density": 0.0, "Temperature": 0.5, "Fuel": 1.0,
               "Initial Velocity": False, "Burn Rate": 2.0, "Flame Heat": 3.0,
               "Flame Smoke": 0.1, "Flame Cooling": 10.0, "Heat Buoyancy": 1.0,
               "Turbulence": 0.3, "Turbulence Scale": 0.05, "Fine Turbulence": 0.0,
               "Fine Scale": 0.03},
}
# Settings the presets never change.
USER_ITEMS = (
    ("Add Emission", "BOOLEAN", False, "Add the emitted values per second instead of raising "
                                       "the smoke to them"),
    ("Emit Until", "FLOAT", 0.0, "Stop emitting after this many seconds; 0 emits throughout"),
    ("Particle Seed", "INT", 0, "Particles: random seed"),
    ("Density Buoyancy", "FLOAT", 0.0, "Positive values make dense smoke rise, negative sink"),
    ("Gravity", "VECTOR", (0.0, 0.0, -9.81), ""),
    ("Turbulence Speed", "FLOAT", 1.0, "How many times per second the swirls change"),
    ("Turbulence Seed", "INT", 0, ""),
    ("Turbulence Mask", "FLOAT", 0.1, "Density at which the large swirls reach full strength; "
                                      "0 stirs all air"),
    ("Sparse GPU", "BOOLEAN", True, "Simulate on the GPU. Off runs Mantaflow, which has no "
                                    "emission, colliders, turbulence or fire"),
    ("CFL", "FLOAT", 6.0, "Most voxels anything may travel in one step"),
    ("Max Substeps", "INT", 64, ""),
    ("Turbulence Layers", "GEOMETRY", None, "More turbulence layers, from Turbulence Layer "
                                            "nodes, applied after the fine swirls"),
)
# Multiplied by the preset's Strength and Speed.
STRENGTH_ITEMS = ("Density", "Temperature", "Fuel")
SPEED_ITEMS = ("Normal", "Velocity", "Particle Speed", "Spread")
SOCKET_TYPES = {"FLOAT": "NodeSocketFloat", "INT": "NodeSocketInt",
                "BOOLEAN": "NodeSocketBool", "VECTOR": "NodeSocketVector",
                "GEOMETRY": "NodeSocketGeometry", "MENU": "NodeSocketMenu",
                "COLLECTION": "NodeSocketCollection", "BUNDLE": "NodeSocketBundle"}
SETTINGS_DESCRIPTION = "All settings of a pyro simulation, from Pyro Preset"


def settings_socket(tree, in_out="INPUT"):
    return socket(tree, "Settings", "NodeSocketBundle", in_out=in_out,
                  description=SETTINGS_DESCRIPTION)


def separate(b, settings, names):
    """Separate Bundle node giving the named settings, as {name: socket}."""
    kinds = {name: kind for name, kind, *_ in PRESET_ITEMS + USER_ITEMS}
    node = b.new("NodeSeparateBundle")
    for name in names:
        node.bundle_items.new(kinds[name], name)
    b.link(settings, node.inputs[0])
    return {name: node.outputs[name] for name in names}


def menu_switch(b, kind, menu, options):
    """Menu Switch of `kind` over (name, value) options, driven by the `menu` socket."""
    node = b.new("GeometryNodeMenuSwitch", data_type=kind)
    node.enum_items.clear()
    for name, _value in options:
        node.enum_items.new(name)
    for (name, value), socket_in in zip(options, [s for s in node.inputs[1:] if s.name]):
        b.set(socket_in, value)
    b.link(menu, node.inputs["Menu"])
    return node.outputs[0]


def switch(b, kind, condition, false, true):
    node = b.new("GeometryNodeSwitch", input_type=kind)
    b.set(node.inputs["Switch"], condition)
    b.set(node.inputs["False"], false)
    b.set(node.inputs["True"], true)
    return node.outputs[0]


def named_attribute(b, name, kind="FLOAT"):
    node = b.new("GeometryNodeInputNamedAttribute", data_type=kind)
    node.inputs["Name"].default_value = name
    return node.outputs["Attribute"]


def store_attribute(b, geometry, name, value, kind="FLOAT", domain="POINT"):
    node = b.new("GeometryNodeStoreNamedAttribute", data_type=kind, domain=domain)
    node.inputs["Name"].default_value = name
    b.link(geometry, node.inputs["Geometry"])
    b.set(node.inputs["Value"], value)
    return node.outputs["Geometry"]


def field_grid(b, topology, value, kind="FLOAT"):
    node = b.new("GeometryNodeFieldToGrid", data_type=kind)
    node.grid_items.clear()
    node.grid_items.new(kind, "Value")
    b.link(topology, node.inputs["Topology"])
    b.set(node.inputs["Value"], value)
    return node.outputs["Value"]


def density_grid(b, mesh, value, voxel_size):
    node = b.new("GeometryNodeMeshToDensityGrid")
    b.link(mesh, node.inputs["Mesh"])
    b.set(node.inputs["Density"], value)
    b.link(voxel_size, node.inputs["Voxel Size"])
    return node.outputs["Density Grid"]


def store_fuel_if_any(b, volume, fuel, grid):
    """Store a fuel grid only for fuel above zero: any fuel grid turns the solver's fire on."""
    with_fuel = store_grid(b, volume, "fuel", grid)
    return switch(b, "GEOMETRY", b.math("GREATER_THAN", fuel, 0.0), volume, with_fuel)


def motion_since(b, mesh, previous_mesh, delta_time):
    """Field: each vertex's velocity since the same vertex of `previous_mesh`; zero if the two
    have different vertex counts (the first step, or a changed mesh)."""
    position = b.new("GeometryNodeInputPosition").outputs[0]
    previous = b.new("GeometryNodeSampleIndex", data_type="FLOAT_VECTOR", domain="POINT")
    b.link(previous_mesh, previous.inputs["Geometry"])
    b.link(b.new("GeometryNodeInputPosition").outputs[0], previous.inputs["Value"])
    b.link(b.new("GeometryNodeInputIndex").outputs[0], previous.inputs["Index"])
    now_count = b.new("GeometryNodeAttributeDomainSize", component="MESH")
    b.link(mesh, now_count.inputs["Geometry"])
    then_count = b.new("GeometryNodeAttributeDomainSize", component="MESH")
    b.link(previous_mesh, then_count.inputs["Geometry"])
    same = b.new("FunctionNodeCompare", data_type="INT", operation="EQUAL")
    b.link(now_count.outputs["Point Count"], same.inputs["A"])
    b.link(then_count.outputs["Point Count"], same.inputs["B"])
    motion = b.vmath("SCALE", b.vmath("SUBTRACT", position, previous.outputs[0]),
                     scale=b.math("DIVIDE", 1.0, delta_time))
    return switch(b, "VECTOR", same.outputs[0], (0.0, 0.0, 0.0), motion)


# --------------------------------------------------------------------------------------------
# Low-level blocks with explicit inputs.

def build_source():
    tree = new_group(
        "Pyro Source",
        "Turn a mesh into smoke and fire sources for the Gas Simulation Step node's Sources "
        "input, with an initial velocity from the mesh's motion, its normals and a vector")
    socket(tree, "Mesh", "NodeSocketGeometry")
    socket(tree, "Previous Mesh", "NodeSocketGeometry",
           description="The mesh one simulation step earlier, for its motion. Leave empty "
                       "for a mesh at rest")
    socket(tree, "Delta Time", "NodeSocketFloat", default=1.0 / 30.0, minimum=1.0e-6,
           description="Time since the previous mesh")
    socket(tree, "Density", "NodeSocketFloat", default=1.0, minimum=0.0,
           description="Smoke emitted inside the mesh")
    socket(tree, "Temperature", "NodeSocketFloat", default=1.0,
           description="Heat emitted inside the mesh")
    socket(tree, "Fuel", "NodeSocketFloat", default=0.0, minimum=0.0,
           description="Fuel emitted inside the mesh; it burns into flame, heat and smoke")
    socket(tree, "Initial Velocity", "NodeSocketBool", default=False,
           description="Set the velocity of the air where the mesh emits")
    socket(tree, "Source", "NodeSocketFloat", default=1.0,
           description="Share of the mesh's own motion given to the smoke")
    socket(tree, "Normal", "NodeSocketFloat", default=0.0, subtype="VELOCITY",
           description="Speed along the mesh's normals")
    socket(tree, "Velocity", "NodeSocketVector", default=(0.0, 0.0, 0.0), subtype="VELOCITY",
           description="Velocity added everywhere, world units per second")
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4,
           description="Use the simulation's voxel size")
    socket(tree, "Sources", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    mesh = inp["Mesh"]

    density = density_grid(b, mesh, inp["Density"], inp["Voxel Size"])
    volume = store_grid(b, None, "density", density)
    volume = store_grid(b, volume, "temperature",
                        density_grid(b, mesh, inp["Temperature"], inp["Voxel Size"]))
    volume = store_fuel_if_any(b, volume, inp["Fuel"],
                               density_grid(b, mesh, inp["Fuel"], inp["Voxel Size"]))

    motion = motion_since(b, mesh, inp["Previous Mesh"], inp["Delta Time"])
    normal = b.new("GeometryNodeInputNormal").outputs[0]
    velocity = b.vmath("ADD", b.vmath("ADD", b.vmath("SCALE", motion, scale=inp["Source"]),
                                      b.vmath("SCALE", normal, scale=inp["Normal"])),
                       inp["Velocity"])
    with_velocity = store_grid(b, volume, "velocity",
                               velocity_grid(b, mesh, velocity, density), kind="VECTOR_FLOAT")
    b.link(switch(b, "GEOMETRY", inp["Initial Velocity"], volume, with_velocity),
           group_out.inputs["Sources"])
    arrange(tree)
    return tree


def build_point_source():
    tree = new_group(
        "Pyro Point Source",
        "Turn points, such as Pyro Particles, into smoke and fire sources: a puff of the "
        "points' radius each, carrying their velocity")
    socket(tree, "Points", "NodeSocketGeometry",
           description="Points with a radius. Optional point attributes: 'velocity' (world "
                       "units per second) and 'emission' (a 0 to 1 multiplier)")
    socket(tree, "Density", "NodeSocketFloat", default=1.0, minimum=0.0)
    socket(tree, "Temperature", "NodeSocketFloat", default=1.0)
    socket(tree, "Fuel", "NodeSocketFloat", default=0.0, minimum=0.0)
    socket(tree, "Initial Velocity", "NodeSocketBool", default=True,
           description="Give the smoke the points' velocity")
    socket(tree, "Source", "NodeSocketFloat", default=1.0,
           description="Share of the points' velocity given to the smoke")
    socket(tree, "Velocity", "NodeSocketVector", default=(0.0, 0.0, 0.0), subtype="VELOCITY",
           description="Velocity added everywhere, world units per second")
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4,
           description="Use the simulation's voxel size")
    socket(tree, "Sources", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    points = inp["Points"]

    fog = b.new("GeometryNodePointsToVolume")
    b.link(points, fog.inputs["Points"])
    fog.inputs["Density"].default_value = 1.0
    fog.inputs["Resolution Mode"].default_value = "Size"
    b.link(inp["Voxel Size"], fog.inputs["Voxel Size"])
    b.link(b.new("GeometryNodeInputRadius").outputs[0], fog.inputs["Radius"])
    get = b.new("GeometryNodeGetNamedGrid", data_type="FLOAT")
    b.link(fog.outputs["Volume"], get.inputs["Volume"])
    get.inputs["Name"].default_value = "density"
    shape = get.outputs["Grid"]

    # Each voxel takes the emission and velocity of its nearest point.
    nearest = b.new("GeometryNodeSampleNearest", domain="POINT")
    b.link(points, nearest.inputs["Geometry"])

    def nearest_value(name, kind):
        sample = b.new("GeometryNodeSampleIndex", data_type=kind, domain="POINT")
        b.link(points, sample.inputs["Geometry"])
        exists = b.new("GeometryNodeInputNamedAttribute",
                       data_type="FLOAT" if kind == "FLOAT" else "FLOAT_VECTOR")
        exists.inputs["Name"].default_value = name
        value = exists.outputs["Attribute"]
        if name == "emission":
            value = switch(b, "FLOAT", exists.outputs["Exists"], 1.0, value)
        b.link(value, sample.inputs["Value"])
        b.link(nearest.outputs["Index"], sample.inputs["Index"])
        return sample.outputs[0]

    sample_shape = b.new("GeometryNodeSampleGrid")
    b.link(shape, sample_shape.inputs["Grid"])
    weight = b.math("MULTIPLY", sample_shape.outputs[0], nearest_value("emission", "FLOAT"))

    def emitted(amount):
        return field_grid(b, shape, b.math("MULTIPLY", weight, amount))

    volume = store_grid(b, None, "density", emitted(inp["Density"]))
    volume = store_grid(b, volume, "temperature", emitted(inp["Temperature"]))
    volume = store_fuel_if_any(b, volume, inp["Fuel"], emitted(inp["Fuel"]))
    velocity = b.vmath("ADD", b.vmath("SCALE", nearest_value("velocity", "FLOAT_VECTOR"),
                                      scale=inp["Source"]), inp["Velocity"])
    with_velocity = store_grid(b, volume, "velocity", field_grid(b, shape, velocity, "VECTOR"),
                               kind="VECTOR_FLOAT")
    result = switch(b, "GEOMETRY", inp["Initial Velocity"], volume, with_velocity)
    # No points, no sources (and no warning about a missing grid).
    count = b.new("GeometryNodeAttributeDomainSize", component="POINTCLOUD")
    b.link(points, count.inputs["Geometry"])
    has_points = b.math("GREATER_THAN", count.outputs["Point Count"], 0.0)
    b.link(switch(b, "GEOMETRY", has_points, None, result), group_out.inputs["Sources"])
    arrange(tree)
    return tree


def build_collider():
    tree = new_group(
        "Pyro Collider",
        "Turn a mesh into an obstacle for the Gas Simulation Step node's Colliders input")
    socket(tree, "Mesh", "NodeSocketGeometry")
    socket(tree, "Velocity", "NodeSocketVector", default=(0.0, 0.0, 0.0), subtype="VELOCITY",
           description="Velocity of the mesh in world units per second, evaluated on the mesh. "
                       "A moving collider pushes the smoke with it")
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4,
           description="Use the simulation's voxel size")
    socket(tree, "Colliders", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    mesh = group_in.outputs["Mesh"]
    solid = b.new("GeometryNodeMeshToDensityGrid")
    b.link(mesh, solid.inputs["Mesh"])
    solid.inputs["Density"].default_value = 1.0
    b.link(group_in.outputs["Voxel Size"], solid.inputs["Voxel Size"])
    # A gradient one voxel wide: the solver takes values above 0.5 as solid.
    b.link(group_in.outputs["Voxel Size"], solid.inputs["Gradient Width"])
    grid = solid.outputs["Density Grid"]
    volume = store_grid(b, None, "obstacle", grid)
    volume = store_grid(b, volume, "velocity",
                        velocity_grid(b, mesh, group_in.outputs["Velocity"], grid),
                        kind="VECTOR_FLOAT")
    b.link(volume, group_out.inputs["Colliders"])
    arrange(tree)
    return tree


TURBULENCE_MASKS = (("None", 0), ("Density", 1), ("Temperature", 2), ("Flame", 3), ("Speed", 4))


def build_turbulence_layer():
    tree = new_group(
        "Turbulence Layer",
        "One layer of swirling noise. Chain layers into the Turbulence Layers input of Pyro "
        "Preset or the Gas Simulation Step node; they apply in chain order, so a layer masked "
        "by Speed reacts to the layers before it")
    socket(tree, "Layers", "NodeSocketGeometry", description="Layers applied before this one")
    socket(tree, "Strength", "NodeSocketFloat", default=2.0, minimum=0.0, subtype="ACCELERATION",
           description="World units per second squared")
    socket(tree, "Scale", "NodeSocketFloat", default=0.25, minimum=1.0e-3, subtype="DISTANCE",
           description="Size of the swirls")
    socket(tree, "Speed", "NodeSocketFloat", default=1.0, minimum=0.0,
           description="How many times per second the pattern changes")
    socket(tree, "Seed", "NodeSocketInt", default=0, minimum=0)
    socket(tree, "Mask", "NodeSocketMenu",
           description="Where the layer acts: everywhere, in smoke, heat or flame, or where "
                       "the flow is fast")
    socket(tree, "Mask Level", "NodeSocketFloat", default=0.1, minimum=1.0e-4,
           description="The mask value (speed in world units per second) at full strength")
    socket(tree, "Invert", "NodeSocketBool", default=False,
           description="Act where the mask is low instead: thin smoke, or slow flow")
    socket(tree, "Layers", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    point = b.new("GeometryNodePoints")
    point.inputs["Count"].default_value = 1
    layer = point.outputs[0]
    for name, attribute, kind in (("Strength", "strength", "FLOAT"), ("Scale", "scale", "FLOAT"),
                                  ("Speed", "speed", "FLOAT"), ("Seed", "seed", "INT"),
                                  ("Mask Level", "mask_level", "FLOAT"),
                                  ("Invert", "invert", "BOOLEAN")):
        layer = store_attribute(b, layer, attribute, inp[name], kind)
    mask = menu_switch(b, "INT", inp["Mask"], TURBULENCE_MASKS)
    layer = store_attribute(b, layer, "mask", mask, "INT")
    # Join Geometry puts later links first: the new layer is linked first to end up last.
    join = b.new("GeometryNodeJoinGeometry")
    b.link(layer, join.inputs[0])
    b.link(inp["Layers"], join.inputs[0])
    b.link(join.outputs[0], group_out.inputs["Layers"])
    input_item(tree, "Mask").default_value = "Density"
    arrange(tree)
    return tree


# --------------------------------------------------------------------------------------------
# Blocks of the modifier, passing one Settings bundle.

def build_preset(layer):
    tree = new_group(
        "Pyro Preset",
        "Collect a pyro simulation's settings into one Settings bundle: a preset's values, or "
        "the Custom values below, scaled by Strength and Speed")
    socket(tree, "Preset", "NodeSocketMenu",
           description="Custom uses the values below; a preset replaces most of them")
    socket(tree, "Strength", "NodeSocketFloat", default=1.0, minimum=0.0,
           description="Multiplies the emitted density, temperature and fuel")
    socket(tree, "Speed", "NodeSocketFloat", default=1.0, minimum=0.0,
           description="Multiplies the emission and particle speeds")
    socket(tree, "Emit From", "NodeSocketMenu",
           description="The mesh's volume, or particles flying off its surface")
    for name, kind, default, description in PRESET_ITEMS + USER_ITEMS:
        if name == "Emit From":
            continue
        socket(tree, name, SOCKET_TYPES[kind], default=default, description=description)
    settings_socket(tree, "OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    emit_from = menu_switch(b, "INT", inp["Emit From"], (("Volume", 0), ("Particles", 1)))

    def preset_bundle(values):
        node = b.new("NodeCombineBundle")
        for name, kind, default, _ in PRESET_ITEMS:
            node.bundle_items.new(kind, name)
        for name, kind, default, _ in PRESET_ITEMS:
            if values is None:
                b.link(emit_from if name == "Emit From" else inp[name], node.inputs[name])
            else:
                node.inputs[name].default_value = values.get(name, default) if name != "Emit From" \
                    or name in values else 0
                if name not in values:
                    # Not set by the preset: the Custom value.
                    b.link(emit_from if name == "Emit From" else inp[name], node.inputs[name])
        return node.outputs[0]

    options = [("Custom", preset_bundle(None))]
    options += [(name, preset_bundle(values)) for name, values in PRESETS.items()]
    chosen = menu_switch(b, "BUNDLE", inp["Preset"], options)
    values = separate(b, chosen, [name for name, *_ in PRESET_ITEMS])
    for name in STRENGTH_ITEMS:
        values[name] = b.math("MULTIPLY", values[name], inp["Strength"])
    for name in SPEED_ITEMS:
        scale = b.vmath if name == "Velocity" else None
        values[name] = (b.vmath("SCALE", values[name], scale=inp["Speed"]) if scale else
                        b.math("MULTIPLY", values[name], inp["Speed"]))
    # The fine swirls are a speed-masked layer before any layers given.
    fine = b.new("GeometryNodeGroup", node_tree=layer)
    b.link(values["Fine Turbulence"], fine.inputs["Strength"])
    b.link(values["Fine Scale"], fine.inputs["Scale"])
    b.link(inp["Turbulence Speed"], fine.inputs["Speed"])
    b.link(b.math("ADD", inp["Turbulence Seed"], 1.0), fine.inputs["Seed"])
    fine.inputs["Mask"].default_value = "Speed"
    fine.inputs["Mask Level"].default_value = 0.5
    no_fine = b.math("LESS_THAN", values["Fine Turbulence"], 1.0e-6)
    fine_layer = switch(b, "GEOMETRY", no_fine, fine.outputs[0], None)
    join = b.new("GeometryNodeJoinGeometry")
    b.link(inp["Turbulence Layers"], join.inputs[0])
    b.link(fine_layer, join.inputs[0])
    values["Turbulence Layers"] = join.outputs[0]

    combine = b.new("NodeCombineBundle")
    for name, kind, *_ in PRESET_ITEMS + USER_ITEMS:
        combine.bundle_items.new(kind, name)
        source = values.get(name, inp[name] if name in inp else None)
        b.link(source, combine.inputs[name])
    b.link(combine.outputs[0], group_out.inputs["Settings"])
    input_item(tree, "Preset").default_value = "Custom"
    input_item(tree, "Emit From").default_value = "Volume"
    arrange(tree)
    return tree


def build_particles():
    tree = new_group(
        "Pyro Particles",
        "A particle simulation for Pyro Point Source: particles leave the emitter's surface, "
        "slow down, fall and fade. Feed it the emitter in world space")
    socket(tree, "Emitter", "NodeSocketGeometry")
    settings_socket(tree)
    socket(tree, "Points", "NodeSocketGeometry", in_out="OUTPUT",
           description="Points with radius, 'velocity' and 'emission' (fading to 0)")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    s = separate(b, inp["Settings"], ["Emit From", "Rate", "Burst", "Particle Speed", "Spread",
                                      "Drag", "Particle Gravity", "Lifetime", "Particle Radius",
                                      "Particle Seed", "Emit Until", "Gravity"])
    sim_in = b.new("GeometryNodeSimulationInput")
    sim_out = b.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    sim_out.state_items.new("FLOAT", "Elapsed")
    dt = sim_in.outputs["Delta Time"]
    elapsed = sim_in.outputs["Elapsed"]
    emitter = inp["Emitter"]

    # Move the particles that exist: drag, gravity, then age and fade.
    points = sim_in.outputs["Geometry"]
    velocity = named_attribute(b, "velocity", "FLOAT_VECTOR")
    keep = b.math("MAXIMUM", b.math("SUBTRACT", 1.0, b.math("MULTIPLY", s["Drag"], dt)), 0.0)
    gravity = b.vmath("SCALE", s["Gravity"], scale=b.math("MULTIPLY", s["Particle Gravity"], dt))
    velocity = b.vmath("ADD", b.vmath("SCALE", velocity, scale=keep), gravity)
    points = store_attribute(b, points, "velocity", velocity, "FLOAT_VECTOR")
    moved = b.new("GeometryNodeSetPosition")
    b.link(points, moved.inputs["Geometry"])
    b.link(b.vmath("SCALE", named_attribute(b, "velocity", "FLOAT_VECTOR"), scale=dt),
           moved.inputs["Offset"])
    age = b.math("ADD", named_attribute(b, "age"), dt)
    points = store_attribute(b, moved.outputs[0], "age", age)
    dead = b.new("GeometryNodeDeleteGeometry", domain="POINT")
    b.link(points, dead.inputs["Geometry"])
    b.link(b.math("GREATER_THAN", named_attribute(b, "age"), s["Lifetime"]),
           dead.inputs["Selection"])
    fade = b.math("MAXIMUM", b.math("SUBTRACT", 1.0, b.math(
        "DIVIDE", named_attribute(b, "age"), b.math("MAXIMUM", s["Lifetime"], 1.0e-3))), 0.0)
    alive = store_attribute(b, dead.outputs[0], "emission", fade)

    # New particles: the rate over this step (while emitting), and the burst on the first step.
    area = b.new("GeometryNodeAttributeStatistic", data_type="FLOAT", domain="FACE")
    b.link(emitter, area.inputs["Geometry"])
    b.link(b.new("GeometryNodeInputMeshFaceArea").outputs[0], area.inputs["Attribute"])
    emitting = b.math("MAXIMUM", b.math("LESS_THAN", elapsed, s["Emit Until"]),
                      b.math("LESS_THAN", s["Emit Until"], 1.0e-6))
    first = b.math("LESS_THAN", elapsed, b.math("MULTIPLY", dt, 0.5))
    count = b.math("ADD", b.math("MULTIPLY", b.math("MULTIPLY", s["Rate"], dt), emitting),
                   b.math("MULTIPLY", s["Burst"], first))
    count = b.math("MULTIPLY", count, b.equal(s["Emit From"], 1.0))
    spawn = b.new("GeometryNodeDistributePointsOnFaces")
    b.link(emitter, spawn.inputs["Mesh"])
    b.link(b.math("DIVIDE", count, b.math("MAXIMUM", area.outputs["Sum"], 1.0e-6)),
           spawn.inputs["Density"])
    b.link(b.math("ADD", s["Particle Seed"], b.math("FLOOR", b.math("MULTIPLY", elapsed, 977.0))),
           spawn.inputs["Seed"])
    random = b.new("FunctionNodeRandomValue", data_type="FLOAT_VECTOR")
    random.inputs["Min"].default_value = (-1.0, -1.0, -1.0)
    random.inputs["Max"].default_value = (1.0, 1.0, 1.0)
    b.link(b.math("ADD", s["Particle Seed"], b.math("FLOOR", b.math("MULTIPLY", elapsed, 389.0))),
           random.inputs["Seed"])
    start = b.vmath("ADD", b.vmath("SCALE", spawn.outputs["Normal"], scale=s["Particle Speed"]),
                    b.vmath("SCALE", random.outputs["Value"], scale=s["Spread"]))
    born = store_attribute(b, spawn.outputs["Points"], "velocity", start, "FLOAT_VECTOR")
    born = store_attribute(b, born, "age", 0.0)
    born = store_attribute(b, born, "emission", 1.0)
    radius = b.new("GeometryNodeSetPointRadius")
    b.link(born, radius.inputs["Points"])
    b.link(s["Particle Radius"], radius.inputs["Radius"])
    join = b.new("GeometryNodeJoinGeometry")
    b.link(radius.outputs[0], join.inputs[0])
    b.link(alive, join.inputs[0])
    b.link(join.outputs[0], sim_out.inputs["Geometry"])
    b.link(b.math("ADD", elapsed, dt), sim_out.inputs["Elapsed"])
    b.link(sim_out.outputs["Geometry"], group_out.inputs["Points"])
    arrange(tree)
    return tree


def build_emitter(source, point_source):
    tree = new_group(
        "Pyro Emitter",
        "Sources of one simulation step from the emitter mesh or its particles, per the Settings")
    socket(tree, "Emitter", "NodeSocketGeometry", description="The emitter mesh in world space")
    socket(tree, "Previous Emitter", "NodeSocketGeometry",
           description="The emitter one step earlier, for its motion")
    socket(tree, "Particles", "NodeSocketGeometry", description="From Pyro Particles")
    settings_socket(tree)
    socket(tree, "Elapsed", "NodeSocketFloat", default=0.0,
           description="Seconds since the simulation started, for Emit Until")
    socket(tree, "Delta Time", "NodeSocketFloat", default=1.0 / 30.0, minimum=1.0e-6)
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4)
    socket(tree, "Sources", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    s = separate(b, inp["Settings"], ["Emit From", "Density", "Temperature", "Fuel",
                                      "Initial Velocity", "Source", "Normal", "Velocity",
                                      "Emit Until"])
    mesh = b.new("GeometryNodeGroup", node_tree=source)
    b.link(inp["Emitter"], mesh.inputs["Mesh"])
    b.link(inp["Previous Emitter"], mesh.inputs["Previous Mesh"])
    b.link(inp["Delta Time"], mesh.inputs["Delta Time"])
    b.link(inp["Voxel Size"], mesh.inputs["Voxel Size"])
    points = b.new("GeometryNodeGroup", node_tree=point_source)
    b.link(inp["Particles"], points.inputs["Points"])
    b.link(inp["Voxel Size"], points.inputs["Voxel Size"])
    for name in ("Density", "Temperature", "Fuel", "Initial Velocity", "Source", "Velocity"):
        b.link(s[name], mesh.inputs[name])
        b.link(s[name], points.inputs[name])
    b.link(s["Normal"], mesh.inputs["Normal"])
    chosen = switch(b, "GEOMETRY", b.equal(s["Emit From"], 1.0), mesh.outputs[0],
                    points.outputs[0])
    emitting = b.math("MAXIMUM", b.math("LESS_THAN", inp["Elapsed"], s["Emit Until"]),
                      b.math("LESS_THAN", s["Emit Until"], 1.0e-6))
    b.link(switch(b, "GEOMETRY", emitting, None, chosen), group_out.inputs["Sources"])
    arrange(tree)
    return tree


def build_domain():
    tree = new_group(
        "Pyro Domain",
        "The simulation box around an emitter: its lowest corner and voxel counts. Centred on "
        "the emitter, its floor a tenth of its height below it")
    socket(tree, "Emitter", "NodeSocketGeometry", description="In world space")
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-3)
    socket(tree, "Domain Size", "NodeSocketVector", default=(2.0, 2.0, 4.0), subtype="XYZ")
    socket(tree, "Domain Min", "NodeSocketVector", in_out="OUTPUT")
    for axis in "XYZ":
        socket(tree, "Resolution " + axis, "NodeSocketInt", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    bounds = b.new("GeometryNodeBoundBox")
    b.link(inp["Emitter"], bounds.inputs["Geometry"])
    centre = b.vmath("SCALE", b.vmath("ADD", bounds.outputs["Min"], bounds.outputs["Max"]),
                     scale=0.5)
    corner = b.vmath("SUBTRACT", centre, b.vmath("SCALE", inp["Domain Size"], scale=0.5))
    corner_xyz = b.new("ShaderNodeSeparateXYZ")
    b.link(corner, corner_xyz.inputs[0])
    low_xyz = b.new("ShaderNodeSeparateXYZ")
    b.link(bounds.outputs["Min"], low_xyz.inputs[0])
    size_xyz = b.new("ShaderNodeSeparateXYZ")
    b.link(inp["Domain Size"], size_xyz.inputs[0])
    domain_min = b.new("ShaderNodeCombineXYZ")
    b.link(corner_xyz.outputs[0], domain_min.inputs[0])
    b.link(corner_xyz.outputs[1], domain_min.inputs[1])
    b.link(b.math("SUBTRACT", low_xyz.outputs[2], b.math("MULTIPLY", size_xyz.outputs[2], 0.1)),
           domain_min.inputs[2])
    b.link(domain_min.outputs[0], group_out.inputs["Domain Min"])
    for axis, name in enumerate("XYZ"):
        cells = b.math("CEIL", b.math("DIVIDE", size_xyz.outputs[axis], inp["Voxel Size"]))
        b.link(b.math("MINIMUM", b.math("MAXIMUM", cells, 4.0), 1024.0),
               group_out.inputs["Resolution " + name])
    arrange(tree)
    return tree


def build_colliders(collider):
    tree = new_group(
        "Pyro Colliders",
        "Obstacles from a collection, with their velocity from the previous step")
    socket(tree, "Collection", "NodeSocketCollection")
    socket(tree, "Previous", "NodeSocketGeometry",
           description="This node's Geometry output one step earlier")
    socket(tree, "Delta Time", "NodeSocketFloat", default=1.0 / 30.0, minimum=1.0e-6)
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4)
    socket(tree, "Colliders", "NodeSocketGeometry", in_out="OUTPUT")
    socket(tree, "Geometry", "NodeSocketGeometry", in_out="OUTPUT",
           description="The colliders now; keep it for the next step's Previous")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    info = b.new("GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    b.link(inp["Collection"], info.inputs["Collection"])
    realize = b.new("GeometryNodeRealizeInstances")
    b.link(info.outputs[0], realize.inputs["Geometry"])
    meshes = realize.outputs[0]
    node = b.new("GeometryNodeGroup", node_tree=collider)
    b.link(meshes, node.inputs["Mesh"])
    b.link(motion_since(b, meshes, inp["Previous"], inp["Delta Time"]), node.inputs["Velocity"])
    b.link(inp["Voxel Size"], node.inputs["Voxel Size"])
    b.link(node.outputs[0], group_out.inputs["Colliders"])
    b.link(meshes, group_out.inputs["Geometry"])
    arrange(tree)
    return tree


def build_solver():
    tree = new_group(
        "Pyro Solver",
        "One step of the Gas Simulation Step node with the Settings. Use it inside a "
        "Simulation Zone")
    socket(tree, "State", "NodeSocketGeometry")
    socket(tree, "Sources", "NodeSocketGeometry")
    socket(tree, "Colliders", "NodeSocketGeometry")
    settings_socket(tree)
    socket(tree, "Delta Time", "NodeSocketFloat", default=1.0 / 30.0, minimum=0.0)
    socket(tree, "Domain Min", "NodeSocketVector", default=(-1.0, -1.0, -1.0))
    for axis in "XYZ":
        socket(tree, "Resolution " + axis, "NodeSocketInt", default=32, minimum=4)
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4)
    socket(tree, "State", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    names = ["Gravity", "Heat Buoyancy", "Density Buoyancy", "CFL", "Max Substeps",
             "Sparse GPU", "Turbulence", "Turbulence Scale", "Turbulence Speed",
             "Turbulence Seed", "Turbulence Mask", "Turbulence Layers", "Burn Rate",
             "Flame Heat", "Flame Smoke", "Flame Cooling", "Add Emission"]
    s = separate(b, inp["Settings"], names)
    gas = b.new(GAS_NODE)
    b.link(inp["State"], gas.inputs["State"])
    b.link(inp["Delta Time"], gas.inputs["Delta Time"])
    b.link(inp["Domain Min"], gas.inputs["Origin"])
    for axis in "XYZ":
        b.link(inp["Resolution " + axis], gas.inputs["Resolution " + axis])
    b.link(inp["Voxel Size"], gas.inputs["Voxel Size"])
    b.link(inp["Sources"], gas.inputs["Sources"])
    b.link(inp["Colliders"], gas.inputs["Colliders"])
    b.link(b.new("GeometryNodeInputSceneTime").outputs["Seconds"], gas.inputs["Time"])
    for name in names:
        b.link(s[name], gas.inputs["Add Sources" if name == "Add Emission" else name])
    b.link(gas.outputs["State"], group_out.inputs["State"])
    arrange(tree)
    return tree


def build_material():
    material = bpy.data.materials.get("Pyro Simulation")
    if material is None:
        material = bpy.data.materials.new("Pyro Simulation")
    if material.node_tree is None:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 5.0
    # Flame: the 'flame' grid lights the volume, from dark red through orange to yellow.
    flame = nodes.new("ShaderNodeAttribute")
    flame.attribute_name = "flame"
    ramp = nodes.new("ShaderNodeValToRGB")
    elements = ramp.color_ramp.elements
    elements[0].position, elements[0].color = 0.0, (0.0, 0.0, 0.0, 1.0)
    elements[1].position, elements[1].color = 1.0, (1.0, 0.85, 0.35, 1.0)
    middle = elements.new(0.35)
    middle.color = (1.0, 0.25, 0.02, 1.0)
    links.new(flame.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], volume.inputs["Emission Color"])
    strength = nodes.new("ShaderNodeMath")
    strength.operation = "MULTIPLY"
    strength.inputs[1].default_value = 8.0
    strength.label = "Flame Brightness"
    links.new(flame.outputs["Fac"], strength.inputs[0])
    links.new(strength.outputs[0], volume.inputs["Emission Strength"])
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(volume.outputs[0], output.inputs["Volume"])
    return material


def build_display(material):
    tree = new_group(
        "Pyro Display",
        "Shade the simulated smoke and fire, move it into the object's space, and keep the "
        "emitter if asked")
    socket(tree, "State", "NodeSocketGeometry")
    socket(tree, "Emitter", "NodeSocketGeometry", description="The emitter in its own space")
    socket(tree, "To World", "NodeSocketMatrix", description="The emitter's world matrix")
    socket(tree, "Show Emitter", "NodeSocketBool", default=True)
    socket(tree, "Material", "NodeSocketMaterial")
    socket(tree, "Geometry", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    shaded = b.new("GeometryNodeSetMaterial")
    b.link(inp["State"], shaded.inputs["Geometry"])
    b.link(inp["Material"], shaded.inputs["Material"])
    to_local = b.new("GeometryNodeTransform")
    to_local.inputs["Mode"].default_value = "Matrix"
    invert = b.new("FunctionNodeInvertMatrix")
    b.link(inp["To World"], invert.inputs["Matrix"])
    b.link(shaded.outputs[0], to_local.inputs["Geometry"])
    b.link(invert.outputs["Matrix"], to_local.inputs["Transform"])
    join = b.new("GeometryNodeJoinGeometry")
    b.link(switch(b, "GEOMETRY", inp["Show Emitter"], None, inp["Emitter"]), join.inputs[0])
    b.link(to_local.outputs[0], join.inputs[0])
    b.link(join.outputs[0], group_out.inputs["Geometry"])
    group_in.outputs  # keep
    input_item(tree, "Material").default_value = material
    arrange(tree)
    return tree


def build_modifier(blocks, material):
    tree = new_group(
        MODIFIER,
        "Emit smoke and fire from this mesh and simulate them on the GPU. Pick a preset, or "
        "Custom to set everything. The domain is placed around the mesh on the first frame")
    tree.is_modifier = True
    socket(tree, "Geometry", "NodeSocketGeometry")
    socket(tree, "Geometry", "NodeSocketGeometry", in_out="OUTPUT")
    preset = blocks["Pyro Preset"]
    preset_items = {item.name: item for item in preset.interface.items_tree
                    if getattr(item, "in_out", None) == "INPUT"}
    panels = (
        ("Preset", ["Preset", "Strength", "Speed"]),
        ("Emission", ["Emit From", "Density", "Temperature", "Fuel", "Add Emission",
                      "Emit Until"]),
        ("Initial Velocity", ["Initial Velocity", "Source", "Normal", "Velocity"]),
        ("Particles", ["Rate", "Burst", "Particle Speed", "Spread", "Drag", "Particle Gravity",
                       "Lifetime", "Particle Radius", "Particle Seed"]),
        ("Fire", ["Burn Rate", "Flame Heat", "Flame Smoke", "Flame Cooling"]),
        ("Domain", ["Voxel Size", "Domain Size", "Colliders"]),
        ("Forces", ["Heat Buoyancy", "Density Buoyancy", "Gravity"]),
        ("Turbulence", ["Turbulence", "Turbulence Scale", "Turbulence Speed", "Turbulence Seed",
                        "Turbulence Mask", "Fine Turbulence", "Fine Scale"]),
        ("Solver", ["Sparse GPU", "CFL", "Max Substeps"]),
        ("Display", ["Show Emitter"]),
    )
    extra = {
        "Voxel Size": ("NodeSocketFloat", 0.05, 1.0e-3,
                       "Size of a voxel in world units. Halving it gives eight times the voxels"),
        "Domain Size": ("NodeSocketVector", (2.0, 2.0, 4.0), None,
                        "Size of the simulated box in world units, centred on the mesh and "
                        "starting a little below it. Only the parts with smoke cost anything"),
        "Colliders": ("NodeSocketCollection", None, None,
                      "Objects the smoke flows around. They may be animated"),
        "Show Emitter": ("NodeSocketBool", True, None,
                         "Keep the emitting mesh visible next to the smoke"),
    }
    for title, names in panels:
        panel = tree.interface.new_panel(title, default_closed=title not in ("Preset",))
        for name in names:
            if name in extra:
                kind, default, minimum, description = extra[name]
                item = socket(tree, name, kind, default=default, minimum=minimum, parent=panel,
                              description=description)
                if name == "Domain Size":
                    item.subtype = "XYZ"
                continue
            source = preset_items[name]
            item = socket(tree, name, source.socket_type, parent=panel,
                          description=source.description)
            for attribute in ("default_value", "min_value", "max_value", "subtype"):
                if hasattr(source, attribute):
                    try:
                        setattr(item, attribute, getattr(source, attribute))
                    except (AttributeError, TypeError):
                        pass

    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs

    def block(name, **links):
        node = b.new("GeometryNodeGroup", node_tree=blocks[name])
        for socket_name, value in links.items():
            b.set(node.inputs[socket_name.replace("_", " ")], value)
        return node

    settings = block("Pyro Preset")
    for item in preset_items:
        if item in inp:
            b.link(inp[item], settings.inputs[item])
    settings = settings.outputs["Settings"]

    # Simulated in world space; Pyro Display moves the result back into the object's space.
    self_info = b.new("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    b.link(b.new("GeometryNodeSelfObject").outputs[0], self_info.inputs["Object"])
    to_world = b.new("GeometryNodeTransform")
    to_world.inputs["Mode"].default_value = "Matrix"
    b.link(inp["Geometry"], to_world.inputs["Geometry"])
    b.link(self_info.outputs["Transform"], to_world.inputs["Transform"])
    emitter = to_world.outputs["Geometry"]

    domain = block("Pyro Domain", Emitter=emitter, Voxel_Size=inp["Voxel Size"],
                   Domain_Size=inp["Domain Size"])
    particles = block("Pyro Particles", Emitter=emitter, Settings=settings)

    # The zone keeps the domain's corner from its first frame, the emitter and colliders of the
    # previous step for their motion, and the time since the start.
    sim_in = b.new("GeometryNodeSimulationInput")
    sim_out = b.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    for kind, name in (("VECTOR", "Domain Min"), ("GEOMETRY", "Previous Emitter"),
                       ("GEOMETRY", "Previous Colliders"), ("FLOAT", "Elapsed")):
        sim_out.state_items.new(kind, name)
    b.link(domain.outputs["Domain Min"], sim_in.inputs["Domain Min"])
    b.link(emitter, sim_in.inputs["Previous Emitter"])
    delta = sim_in.outputs["Delta Time"]
    colliders = block("Pyro Colliders", Collection=inp["Colliders"],
                      Previous=sim_in.outputs["Previous Colliders"], Delta_Time=delta,
                      Voxel_Size=inp["Voxel Size"])
    sources = block("Pyro Emitter", Emitter=emitter,
                    Previous_Emitter=sim_in.outputs["Previous Emitter"],
                    Particles=particles.outputs["Points"], Settings=settings,
                    Elapsed=sim_in.outputs["Elapsed"], Delta_Time=delta,
                    Voxel_Size=inp["Voxel Size"])
    solver = block("Pyro Solver", State=sim_in.outputs["Geometry"],
                   Sources=sources.outputs["Sources"], Colliders=colliders.outputs["Colliders"],
                   Settings=settings, Delta_Time=delta, Domain_Min=sim_in.outputs["Domain Min"],
                   Voxel_Size=inp["Voxel Size"])
    for axis in "XYZ":
        b.link(domain.outputs["Resolution " + axis], solver.inputs["Resolution " + axis])
    b.link(solver.outputs["State"], sim_out.inputs["Geometry"])
    b.link(sim_in.outputs["Domain Min"], sim_out.inputs["Domain Min"])
    b.link(emitter, sim_out.inputs["Previous Emitter"])
    b.link(colliders.outputs["Geometry"], sim_out.inputs["Previous Colliders"])
    b.link(b.math("ADD", sim_in.outputs["Elapsed"], delta), sim_out.inputs["Elapsed"])

    display = block("Pyro Display", State=sim_out.outputs["Geometry"], Emitter=inp["Geometry"],
                    To_World=self_info.outputs["Transform"], Show_Emitter=inp["Show Emitter"])
    display.inputs["Material"].default_value = material
    b.link(display.outputs[0], group_out.inputs["Geometry"])
    input_item(tree, "Preset").default_value = "Custom"
    input_item(tree, "Emit From").default_value = "Volume"
    arrange(tree)
    return tree


BLOCK_NAMES = ("Pyro Preset", "Pyro Particles", "Pyro Emitter", "Pyro Domain", "Pyro Colliders",
               "Pyro Solver", "Pyro Display", "Pyro Source", "Pyro Point Source",
               "Pyro Collider", "Turbulence Layer")


def build_assets(mark=True):
    """Create the groups and material in the current file; returns the modifier group."""
    source = build_source()
    point_source = build_point_source()
    collider = build_collider()
    layer = build_turbulence_layer()
    material = build_material()
    blocks = {
        "Pyro Source": source,
        "Pyro Point Source": point_source,
        "Pyro Collider": collider,
        "Turbulence Layer": layer,
        "Pyro Preset": build_preset(layer),
        "Pyro Particles": build_particles(),
        "Pyro Emitter": build_emitter(source, point_source),
        "Pyro Domain": build_domain(),
        "Pyro Colliders": build_colliders(collider),
        "Pyro Solver": build_solver(),
        "Pyro Display": build_display(material),
    }
    modifier = build_modifier(blocks, material)
    if mark:
        for tree, catalog in [(blocks[name], CATALOG_PYRO) for name in BLOCK_NAMES] + [
                (modifier, CATALOG_PHYSICS)]:
            tree.asset_mark()
            tree.asset_data.catalog_id = catalog[0]
            tree.asset_data.description = tree.description
    return modifier


def install(datafiles):
    """Write the asset file and register its catalogs in a Blender's datafiles folder."""
    assets = os.path.join(datafiles, "assets")
    nodes = os.path.join(assets, "nodes")
    os.makedirs(nodes, exist_ok=True)
    build_assets()
    path = os.path.join(nodes, ASSET_FILE)
    groups = {bpy.data.node_groups[name] for name in BLOCK_NAMES + (MODIFIER,)}
    bpy.data.libraries.write(path, groups, fake_user=True)

    catalogs = os.path.join(assets, "blender_assets.cats.txt")
    lines = ["VERSION 1", ""]
    if os.path.exists(catalogs):
        with open(catalogs, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    for uuid, path_name in (CATALOG_PYRO, CATALOG_PHYSICS):
        entry = f"{uuid}:{path_name}:{path_name.replace('/', '-')}"
        if entry in lines:
            continue
        # Replace an older path of the same catalog.
        lines = [line for line in lines if not line.startswith(uuid + ":")]
        lines.append(entry)
    with open(catalogs, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"Installed the pyro assets in {path}")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    datafiles = argv[0] if argv else bpy.utils.system_resource("DATAFILES")
    install(datafiles)


if __name__ == "__main__":
    main()
