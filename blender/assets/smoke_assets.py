# SPDX-License-Identifier: GPL-2.0-or-later
"""Build the Pyro node-group assets and install them into a Blender build.

    blender -b --factory-startup --python smoke_assets.py

writes `datafiles/assets/nodes/pyro_simulation_assets.blend` next to the running Blender and
adds its catalogs to `datafiles/assets/blender_assets.cats.txt`, so that:

- Add Modifier > Physics > Pyro Emitter turns any mesh into pyro source points: spiky bursts
  in the manner of Houdini's Pyro Burst Source (Explosion, Fireball, Ground Blast, Muzzle Flash,
  Smoke Puff), the mesh's volume or surface (Smoke, Campfire, Candle), or particles (Trail,
  Large Plume);
- Add Modifier > Physics > Pyro Simulation simulates them: on the same object after a Pyro
  Emitter, or on any object with emitters in its Emitters collection; its presets set the look
  of the simulation (Smoke, Fire, Explosion, Large Plume); and
- Add > Pyro offers the building blocks both are made of, for setups built by hand.

Each modifier's blocks pass one Settings bundle instead of dozens of links. `build_assets()` can also be imported to create the groups in the current file (the tests
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
EMITTER_MODIFIER = "Pyro Emitter"
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


SOCKET_TYPES = {"FLOAT": "NodeSocketFloat", "INT": "NodeSocketInt",
                "BOOLEAN": "NodeSocketBool", "VECTOR": "NodeSocketVector",
                "GEOMETRY": "NodeSocketGeometry", "MENU": "NodeSocketMenu",
                "COLLECTION": "NodeSocketCollection", "BUNDLE": "NodeSocketBundle"}


# --------------------------------------------------------------------------------------------
# Small node helpers.

def separate(b, bundle, items):
    """Separate Bundle node giving the named (name, kind) items, as {name: socket}."""
    node = b.new("NodeSeparateBundle")
    for name, kind in items:
        node.bundle_items.new(kind, name)
    b.link(bundle, node.inputs[0])
    return {name: node.outputs[name] for name, _ in items}


def combine(b, items, values):
    """Combine Bundle node of (name, kind) items from {name: socket or value}."""
    node = b.new("NodeCombineBundle")
    for name, kind in items:
        node.bundle_items.new(kind, name)
    for name, _ in items:
        b.set(node.inputs[name], values[name])
    return node.outputs[0]


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


def random(b, kind, minimum, maximum, ident, seed):
    """Random Value of `kind` (FLOAT or FLOAT_VECTOR) for the ID field `ident`."""
    node = b.new("FunctionNodeRandomValue", data_type=kind)
    b.set(node.inputs["Min"], minimum)
    b.set(node.inputs["Max"], maximum)
    b.set(node.inputs["ID"], ident)
    b.set(node.inputs["Seed"], seed)
    return node.outputs["Value"]


def xyz(b, vector):
    node = b.new("ShaderNodeSeparateXYZ")
    b.link(vector, node.inputs[0])
    return node.outputs[0], node.outputs[1], node.outputs[2]


def vector(b, x, y, z):
    node = b.new("ShaderNodeCombineXYZ")
    b.set(node.inputs[0], x)
    b.set(node.inputs[1], y)
    b.set(node.inputs[2], z)
    return node.outputs[0]


def clamp01(b, value):
    return b.math("MINIMUM", b.math("MAXIMUM", value, 0.0), 1.0)


def mix(b, a, c, t):
    """a + (c - a) * t for floats."""
    return b.math("ADD", a, b.math("MULTIPLY", b.math("SUBTRACT", c, a), t))


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
                     scale=b.math("DIVIDE", 1.0, b.math("MAXIMUM", delta_time, 1.0e-6)))
    return switch(b, "VECTOR", same.outputs[0], (0.0, 0.0, 0.0), motion)


def point_count(b, geometry, component="POINTCLOUD"):
    node = b.new("GeometryNodeAttributeDomainSize", component=component)
    b.link(geometry, node.inputs["Geometry"])
    return node.outputs["Point Count"]


# --------------------------------------------------------------------------------------------
# Pyro points: what an emitter hands to a simulation. A point cloud whose points carry the
# amounts they emit and their world-oriented velocity.

PYRO_ATTRIBUTES = (("density", "FLOAT"), ("temperature", "FLOAT"), ("fuel", "FLOAT"),
                   ("expansion", "FLOAT"), ("velocity", "FLOAT_VECTOR"))


def tag_pyro_points(b, points, density, temperature, fuel, expansion, velocity):
    for (name, kind), value in zip(PYRO_ATTRIBUTES,
                                   (density, temperature, fuel, expansion, velocity)):
        points = store_attribute(b, points, name, value, kind)
    return store_attribute(b, points, "pyro", True, "BOOLEAN")


# --------------------------------------------------------------------------------------------
# Emitter settings: everything the Pyro Emitter modifier and its presets set.

# (name, kind, custom default, description, minimum)
EMITTER_ITEMS = (
    ("Emit Type", "INT", 0, "0 burst, 1 the mesh's volume, 2 its surface, 3 particles", None),
    ("Density", "FLOAT", 1.0, "Smoke emitted", 0.0),
    ("Temperature", "FLOAT", 1.0, "Heat emitted; with heat buoyancy it makes the smoke rise",
     None),
    ("Fuel", "FLOAT", 0.0, "Fuel emitted; it burns into flame, heat and smoke", 0.0),
    ("Expansion", "FLOAT", 0.0, "How fast the emitted gas grows, per second: the push of an "
                                "explosion", 0.0),
    ("Start Frame", "FLOAT", 1.0, "Frame the emission starts on", None),
    ("Duration", "FLOAT", 0.0, "Frames it emits for; a burst expands over them. 0 emits on "
                               "(bursts last 8 frames)", 0.0),
    ("Fade", "FLOAT", 1.0, "How the emission dies down over the duration: 0 stays full, 1 "
                           "fades evenly, higher fades sooner", 0.0),
    ("Size", "FLOAT", 0.25, "Burst: radius at the start", 0.001),
    ("Outward Expansion", "FLOAT", 4.0, "Burst: how many times its starting size it grows "
                                        "to", 1.0),
    ("Expansion Curve", "FLOAT", 3.0, "Burst: above 1 grows fast at first and then slows, as "
                                      "a blast does", 0.1),
    ("Spikes", "INT", 40, "Burst: number of spikes (Houdini's trailings)", 1),
    ("Roundness", "FLOAT", 0.4, "Burst: 1 gives spikes of equal length, lower gives ragged, "
                                "random ones", 0.0),
    ("Spike Length", "FLOAT", 0.7, "Burst: how far each spike reaches in from its tip; 1 runs "
                                   "to the centre", 0.0),
    ("Thickness", "FLOAT", 0.3, "Burst: puff radius at the tips, relative to the burst's "
                                "radius", 0.01),
    ("Core Strength", "FLOAT", 1.5, "Burst: emission at the inner end of a spike relative to "
                                    "its tip", 0.0),
    ("Shape Noise", "FLOAT", 0.4, "Burst: lumpiness of the outline", 0.0),
    ("Direction", "VECTOR", (0.0, 0.0, 1.0), "Burst: the axis it blows along", None),
    ("Spread", "FLOAT", 180.0, "Burst: half-angle around the direction, in degrees: 180 is "
                               "every way, 90 a hemisphere, small a jet", 0.0),
    ("Bursts", "INT", 1, "Burst: number of bursts, each a little later and offset", 1),
    ("Burst Delay", "FLOAT", 2.0, "Burst: frames between bursts", 0.0),
    ("Burst Scatter", "FLOAT", 0.15, "Burst: how far apart the bursts are", 0.0),
    ("Outward Speed", "FLOAT", 1.0, "Burst: share of the burst's own expansion speed given to "
                                    "the gas", 0.0),
    ("Speed Noise", "FLOAT", 1.0, "Random velocity added to every point, world units per "
                                  "second", 0.0),
    ("Point Spacing", "FLOAT", 0.05, "Distance between emitting points; use about the "
                                     "simulation's voxel size", 0.001),
    ("Initial Velocity", "BOOLEAN", True, "Give the gas the velocity of the emission", None),
    ("Source", "FLOAT", 1.0, "Share of the emitter's own motion given to the gas", None),
    ("Normal", "FLOAT", 0.0, "Volume and surface: speed along the mesh's normals", None),
    ("Velocity", "VECTOR", (0.0, 0.0, 0.0), "Velocity added everywhere it emits", None),
    ("Rate", "FLOAT", 200.0, "Particles: new particles per second", 0.0),
    ("Particle Burst", "INT", 0, "Particles: particles created on the first frame", 0),
    ("Particle Speed", "FLOAT", 1.0, "Particles: speed along the emitter's normals", None),
    ("Particle Spread", "FLOAT", 0.5, "Particles: random speed in every direction", 0.0),
    ("Drag", "FLOAT", 1.0, "Particles: how fast they slow down, per second", 0.0),
    ("Particle Gravity", "FLOAT", 0.0, "Particles: share of gravity pulling them down", None),
    ("Lifetime", "FLOAT", 0.8, "Particles: seconds each particle emits, fading out", 0.01),
    ("Particle Radius", "FLOAT", 0.08, "Particles: size of the puff each emits", 0.001),
)
# Never set by a preset.
EMITTER_USER_ITEMS = (
    ("Seed", "INT", 0, "Random seed of the burst's spikes and the particles", 0),
    ("Gravity", "VECTOR", (0.0, 0.0, -9.81), "", None),
)
EMITTER_PRESETS = {
    "Explosion": {"Emit Type": 0, "Density": 0.3, "Temperature": 2.5, "Fuel": 2.0,
                  "Expansion": 6.0, "Start Frame": 1.0, "Duration": 6.0, "Fade": 1.5,
                  "Size": 0.15, "Outward Expansion": 6.0, "Expansion Curve": 3.0,
                  "Spikes": 90, "Roundness": 0.15, "Spike Length": 0.6, "Thickness": 0.11,
                  "Core Strength": 1.4, "Shape Noise": 0.7, "Direction": (0.0, 0.0, 1.0),
                  "Spread": 180.0, "Bursts": 3, "Burst Delay": 2.0, "Burst Scatter": 0.2,
                  "Outward Speed": 1.0, "Speed Noise": 1.5, "Initial Velocity": True,
                  "Source": 1.0, "Velocity": (0.0, 0.0, 1.0)},
    "Fireball": {"Emit Type": 0, "Density": 0.3, "Temperature": 3.0, "Fuel": 2.0,
                 "Expansion": 4.0, "Start Frame": 1.0, "Duration": 8.0, "Fade": 1.5,
                 "Size": 0.25, "Outward Expansion": 3.0, "Expansion Curve": 2.5, "Spikes": 40,
                 "Roundness": 0.6, "Spike Length": 1.0, "Thickness": 0.3,
                 "Core Strength": 1.2, "Shape Noise": 0.4, "Direction": (0.0, 0.0, 1.0),
                 "Spread": 180.0, "Bursts": 1, "Outward Speed": 0.7, "Speed Noise": 0.8,
                 "Initial Velocity": True, "Source": 1.0, "Velocity": (0.0, 0.0, 2.0)},
    "Ground Blast": {"Emit Type": 0, "Density": 1.0, "Temperature": 1.5, "Fuel": 1.2,
                     "Expansion": 6.0, "Start Frame": 1.0, "Duration": 7.0, "Fade": 1.5,
                     "Size": 0.15, "Outward Expansion": 6.0, "Expansion Curve": 3.0,
                     "Spikes": 90, "Roundness": 0.2, "Spike Length": 0.7, "Thickness": 0.12,
                     "Core Strength": 1.6, "Shape Noise": 0.7, "Direction": (0.0, 0.0, 1.0),
                     "Spread": 75.0, "Bursts": 2, "Burst Delay": 2.0, "Burst Scatter": 0.25,
                     "Outward Speed": 1.2, "Speed Noise": 2.0, "Initial Velocity": True,
                     "Source": 1.0, "Velocity": (0.0, 0.0, 1.0)},
    "Muzzle Flash": {"Emit Type": 0, "Density": 0.2, "Temperature": 2.5, "Fuel": 2.0,
                     "Expansion": 8.0, "Start Frame": 1.0, "Duration": 3.0, "Fade": 1.0,
                     "Size": 0.06, "Outward Expansion": 8.0, "Expansion Curve": 2.0,
                     "Spikes": 20, "Roundness": 0.3, "Spike Length": 1.0, "Thickness": 0.35,
                     "Core Strength": 1.0, "Shape Noise": 0.3, "Direction": (0.0, 1.0, 0.0),
                     "Spread": 15.0, "Bursts": 1, "Outward Speed": 1.5, "Speed Noise": 1.0,
                     "Initial Velocity": True, "Source": 1.0, "Velocity": (0.0, 0.0, 0.0)},
    "Smoke Puff": {"Emit Type": 0, "Density": 1.5, "Temperature": 0.3, "Fuel": 0.0,
                   "Expansion": 3.0, "Start Frame": 1.0, "Duration": 6.0, "Fade": 1.0,
                   "Size": 0.2, "Outward Expansion": 3.0, "Expansion Curve": 2.5, "Spikes": 50,
                   "Roundness": 0.4, "Spike Length": 0.9, "Thickness": 0.22,
                   "Core Strength": 1.2, "Shape Noise": 0.5, "Direction": (0.0, 0.0, 1.0),
                   "Spread": 180.0, "Bursts": 1, "Outward Speed": 0.8, "Speed Noise": 0.5,
                   "Initial Velocity": True, "Source": 1.0, "Velocity": (0.0, 0.0, 0.3)},
    "Smoke": {"Emit Type": 1, "Density": 1.0, "Temperature": 1.0, "Fuel": 0.0,
              "Expansion": 0.0, "Duration": 0.0, "Initial Velocity": False},
    "Campfire": {"Emit Type": 1, "Density": 0.0, "Temperature": 1.0, "Fuel": 1.0,
                 "Expansion": 0.0, "Duration": 0.0, "Initial Velocity": True, "Source": 0.0,
                 "Normal": 0.0, "Velocity": (0.0, 0.0, 0.5)},
    "Candle": {"Emit Type": 1, "Density": 0.0, "Temperature": 0.5, "Fuel": 1.0,
               "Expansion": 0.0, "Duration": 0.0, "Initial Velocity": False},
    "Trail": {"Emit Type": 3, "Rate": 300.0, "Particle Burst": 0, "Particle Speed": 0.3,
              "Particle Spread": 0.3, "Drag": 3.0, "Particle Gravity": 0.0, "Lifetime": 0.5,
              "Particle Radius": 0.07, "Initial Velocity": False, "Density": 1.0,
              "Temperature": 0.5, "Fuel": 0.0, "Expansion": 0.0, "Duration": 0.0},
    "Large Plume": {"Emit Type": 3, "Rate": 400.0, "Particle Burst": 0, "Particle Speed": 3.0,
                    "Particle Spread": 1.0, "Drag": 1.5, "Particle Gravity": 0.0,
                    "Lifetime": 0.8, "Particle Radius": 0.15, "Initial Velocity": True,
                    "Source": 1.0, "Velocity": (0.0, 0.0, 2.0), "Density": 1.0,
                    "Temperature": 2.0, "Fuel": 0.3, "Expansion": 0.0, "Duration": 0.0},
}
EMIT_TYPES = (("Burst", 0), ("Volume", 1), ("Surface", 2), ("Particles", 3))
EMITTER_STRENGTH = ("Density", "Temperature", "Fuel", "Expansion")
EMITTER_SPEED = ("Normal", "Velocity", "Particle Speed", "Particle Spread", "Outward Speed",
                 "Speed Noise")

# Solver settings: the Pyro Simulation modifier and its look presets.
SOLVER_ITEMS = (
    ("Heat Buoyancy", "FLOAT", 0.5, "How strongly heat makes smoke rise against gravity", None),
    ("Burn Rate", "FLOAT", 0.75, "Fuel burnt per second. Higher burns faster, in shorter "
                                 "flames", 0.0),
    ("Flame Heat", "FLOAT", 4.0, "Heat per unit of fuel burnt", 0.0),
    ("Flame Smoke", "FLOAT", 1.0, "Smoke per unit of fuel burnt", 0.0),
    ("Flame Cooling", "FLOAT", 4.0, "Flame lost per second without fuel. Lower: taller flames",
     0.0),
    ("Gas Release", "FLOAT", 0.0, "How much burning fuel expands; the push of a fireball", 0.0),
    ("Turbulence", "FLOAT", 0.0, "Large swirls, world units per second squared; 0 is off", 0.0),
    ("Turbulence Scale", "FLOAT", 0.25, "Size of the large swirls", 0.001),
    ("Fine Turbulence", "FLOAT", 0.0, "Small swirls, applied after the large ones and only "
                                      "where the flow moves, so they react to them", 0.0),
    ("Fine Scale", "FLOAT", 0.08, "Size of the small swirls", 0.001),
    ("Disturbance", "FLOAT", 0.0, "Small, fast noise in the thin smoke at the edges, which "
                                  "breaks up smooth caps (Houdini's disturbance)", 0.0),
    ("Disturbance Scale", "FLOAT", 0.05, "Size of the disturbance", 0.001),
)
SOLVER_USER_ITEMS = (
    ("Add Emission", "BOOLEAN", False, "Add the emitted amounts per second instead of raising "
                                       "the gas to them", None),
    ("Density Buoyancy", "FLOAT", 0.0, "Positive values make dense smoke rise, negative sink",
     None),
    ("Gravity", "VECTOR", (0.0, 0.0, -9.81), "", None),
    ("Turbulence Speed", "FLOAT", 1.0, "How many times per second the swirls change", 0.0),
    ("Turbulence Seed", "INT", 0, "", 0),
    ("Turbulence Mask", "FLOAT", 0.1, "Density at which the large swirls reach full strength; "
                                      "0 stirs all air", 0.0),
    ("Sparse GPU", "BOOLEAN", True, "Simulate on the GPU. Off runs Mantaflow, which has no "
                                    "emission, colliders, turbulence or fire", None),
    ("CFL", "FLOAT", 6.0, "Most voxels anything may travel in one step", 0.1),
    ("Max Substeps", "INT", 64, "", 1),
    ("Turbulence Layers", "GEOMETRY", None, "More turbulence layers, from Turbulence Layer "
                                            "nodes, applied after the others", None),
)
SOLVER_PRESETS = {
    "Smoke": {"Heat Buoyancy": 0.5, "Burn Rate": 0.75, "Flame Heat": 4.0, "Flame Smoke": 1.0,
              "Flame Cooling": 4.0, "Gas Release": 0.0, "Turbulence": 2.0,
              "Turbulence Scale": 0.3, "Fine Turbulence": 1.0, "Fine Scale": 0.08,
              "Disturbance": 0.0},
    "Fire": {"Heat Buoyancy": 1.0, "Burn Rate": 0.75, "Flame Heat": 4.0, "Flame Smoke": 0.6,
             "Flame Cooling": 5.0, "Gas Release": 0.5, "Turbulence": 2.0,
             "Turbulence Scale": 0.15, "Fine Turbulence": 1.5, "Fine Scale": 0.05,
             "Disturbance": 1.0, "Disturbance Scale": 0.04},
    "Explosion": {"Heat Buoyancy": 1.2, "Burn Rate": 1.2, "Flame Heat": 3.0,
                  "Flame Smoke": 1.0, "Flame Cooling": 3.0, "Gas Release": 1.5,
                  "Turbulence": 5.0, "Turbulence Scale": 0.4, "Fine Turbulence": 3.0,
                  "Fine Scale": 0.1, "Disturbance": 5.0, "Disturbance Scale": 0.06},
    "Large Plume": {"Heat Buoyancy": 1.0, "Burn Rate": 1.0, "Flame Heat": 3.0,
                    "Flame Smoke": 2.0, "Flame Cooling": 6.0, "Gas Release": 0.5,
                    "Turbulence": 4.0, "Turbulence Scale": 0.6, "Fine Turbulence": 2.0,
                    "Fine Scale": 0.15, "Disturbance": 2.0, "Disturbance Scale": 0.08},
}


def items_of(table):
    return [(name, kind) for name, kind, *_ in table]


def settings_socket(tree, in_out="INPUT", description="All settings, from a preset block"):
    return socket(tree, "Settings", "NodeSocketBundle", in_out=in_out, description=description)


def add_inputs(tree, table, skip=(), parent=None):
    for name, kind, default, description, minimum in table:
        if name in skip:
            continue
        socket(tree, name, SOCKET_TYPES[kind], default=default, minimum=minimum,
               description=description, parent=parent)


def preset_block(name, description, table, user_table, presets, strength=(), speed=(),
                 menus=(), extra=None):
    """A group that picks a preset's values (or the Custom inputs), scales them by Strength and
    Speed and bundles them with the user-only settings. `menus` are (item, options) inputs
    shown as menus and turned into integers."""
    tree = new_group(name, description)
    socket(tree, "Preset", "NodeSocketMenu",
           description="Custom uses the values below; a preset replaces most of them")
    socket(tree, "Strength", "NodeSocketFloat", default=1.0, minimum=0.0,
           description="Multiplies what is emitted" if strength else "Multiplies the forces")
    socket(tree, "Speed", "NodeSocketFloat", default=1.0, minimum=0.0,
           description="Multiplies the speeds")
    menu_items = dict(menus)
    for item, _options in menus:
        socket(tree, item, "NodeSocketMenu")
    add_inputs(tree, table, skip=menu_items)
    add_inputs(tree, user_table)
    settings_socket(tree, "OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    custom = {}
    for name, kind, *_ in table:
        custom[name] = (menu_switch(b, "INT", inp[name], menu_items[name]) if name in menu_items
                        else inp[name])
    options = [("Custom", combine(b, items_of(table), custom))]
    for preset, values in presets.items():
        options.append((preset, combine(b, items_of(table),
                                        {name: values.get(name, custom[name])
                                         for name, *_ in table})))
    chosen = separate(b, menu_switch(b, "BUNDLE", inp["Preset"], options), items_of(table))
    values = dict(chosen)
    for name in strength:
        values[name] = b.math("MULTIPLY", values[name], inp["Strength"])
    for name in speed:
        kind = dict(items_of(table))[name]
        values[name] = (b.vmath("SCALE", values[name], scale=inp["Speed"]) if kind == "VECTOR"
                        else b.math("MULTIPLY", values[name], inp["Speed"]))
    for name, *_ in user_table:
        values[name] = inp[name]
    if extra:
        extra(b, inp, values)
    b.link(combine(b, items_of(table) + items_of(user_table), values),
           group_out.inputs["Settings"])
    input_item(tree, "Preset").default_value = "Custom"
    for item, options_list in menus:
        input_item(tree, item).default_value = options_list[0][0]
    arrange(tree)
    return tree


# --------------------------------------------------------------------------------------------
# Emitter blocks.

def build_burst():
    tree = new_group(
        "Pyro Burst",
        "Spiky burst points, after Houdini's Pyro Burst Source: random spikes from a centre "
        "that grow fast and then slow, each a trail of puffs towards the centre")
    socket(tree, "Centre", "NodeSocketVector", default=(0.0, 0.0, 0.0),
           description="Where the burst starts, in the emitter's space")
    settings_socket(tree)
    socket(tree, "Frame", "NodeSocketFloat", default=1.0)
    socket(tree, "Seconds Per Frame", "NodeSocketFloat", default=1.0 / 24.0, minimum=1.0e-6)
    socket(tree, "Points", "NodeSocketGeometry", in_out="OUTPUT",
           description="Points with radius and the pyro attributes; velocity in the emitter's "
                       "space")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    s = separate(b, inp["Settings"], items_of(EMITTER_ITEMS) + items_of(EMITTER_USER_ITEMS))

    # Points: burst b, spike k, point j along the spike from the tip inward.
    per_spike = b.math("MAXIMUM", b.math("CEIL", b.math(
        "DIVIDE", b.math("MULTIPLY", b.math("MULTIPLY", s["Size"], s["Outward Expansion"]),
                         s["Spike Length"]),
        s["Point Spacing"])), 1.0)
    per_spike = b.math("MINIMUM", per_spike, 64.0)
    per_burst = b.math("MULTIPLY", s["Spikes"], per_spike)
    count = b.math("MULTIPLY", per_burst, s["Bursts"])
    points = b.new("GeometryNodePoints")
    b.link(b.math("MINIMUM", count, 200000.0), points.inputs["Count"])
    index = b.new("GeometryNodeInputIndex").outputs[0]
    burst = b.math("FLOOR", b.math("DIVIDE", index, per_burst))
    spike_id = b.math("FLOOR", b.math("DIVIDE", index, per_spike))
    along = b.math("SUBTRACT", index, b.math("MULTIPLY", spike_id, per_spike))
    seed = s["Seed"]

    # When: each burst starts Burst Delay frames after the one before, and lasts Duration.
    duration = switch(b, "FLOAT", b.math("GREATER_THAN", s["Duration"], 0.0), 8.0, s["Duration"])
    start = b.math("ADD", s["Start Frame"], b.math("MULTIPLY", burst, s["Burst Delay"]))
    age = b.math("SUBTRACT", inp["Frame"], start)
    progress = clamp01(b, b.math("DIVIDE", age, duration))
    live = b.math("MULTIPLY", b.math("GREATER_THAN", age, -1.0e-4),
                  b.math("LESS_THAN", age, duration))
    # Size: Size × (1 + (Outward Expansion - 1) × (1 - (1 - p)^curve)), fast at first.
    remaining = b.math("SUBTRACT", 1.0, progress)
    grow = b.math("SUBTRACT", 1.0, b.math("POWER", remaining, s["Expansion Curve"]))
    extra = b.math("MULTIPLY", s["Size"], b.math("SUBTRACT", s["Outward Expansion"], 1.0))
    radius_now = b.math("ADD", s["Size"], b.math("MULTIPLY", extra, grow))
    # Its rate in world units per second: extra × curve × (1 - p)^(curve - 1) / duration.
    rate = b.math("DIVIDE", b.math("MULTIPLY", b.math("MULTIPLY", extra, s["Expansion Curve"]),
                                   b.math("POWER", remaining,
                                          b.math("SUBTRACT", s["Expansion Curve"], 1.0))),
                  b.math("MULTIPLY", duration, inp["Seconds Per Frame"]))

    # Where: a random direction in the cone around Direction, per spike.
    cos_spread = b.math("COSINE", b.math("RADIANS", b.math("MINIMUM", s["Spread"], 180.0)))
    u = random(b, "FLOAT", 0.0, 1.0, spike_id, b.math("ADD", seed, 11.0))
    v = random(b, "FLOAT", 0.0, 1.0, spike_id, b.math("ADD", seed, 23.0))
    cos_theta = b.math("SUBTRACT", 1.0, b.math("MULTIPLY", u, b.math("SUBTRACT", 1.0, cos_spread)))
    sin_theta = b.math("SQRT", b.math("MAXIMUM", b.math(
        "SUBTRACT", 1.0, b.math("MULTIPLY", cos_theta, cos_theta)), 0.0))
    phi = b.math("MULTIPLY", v, 6.283185)
    local = vector(b, b.math("MULTIPLY", sin_theta, b.math("COSINE", phi)),
                   b.math("MULTIPLY", sin_theta, b.math("SINE", phi)), cos_theta)
    align = b.new("FunctionNodeAlignRotationToVector", axis="Z")
    b.link(b.vmath("NORMALIZE", s["Direction"]), align.inputs["Vector"])
    rotate = b.new("FunctionNodeRotateVector")
    b.link(local, rotate.inputs["Vector"])
    b.link(align.outputs["Rotation"], rotate.inputs["Rotation"])
    direction = rotate.outputs[0]
    # How far: Roundness 1 gives equal spikes; shape noise varies them smoothly over the
    # directions, which gives lobes rather than salt and pepper.
    ragged = random(b, "FLOAT", 0.0, 1.0, spike_id, b.math("ADD", seed, 37.0))
    length = b.math("SUBTRACT", 1.0, b.math("MULTIPLY", b.math("SUBTRACT", 1.0, s["Roundness"]),
                                            ragged))
    noise = b.new("ShaderNodeTexNoise", noise_dimensions="4D")
    b.link(b.vmath("SCALE", direction, scale=1.7), noise.inputs["Vector"])
    b.link(b.math("ADD", b.math("MULTIPLY", seed, 7.31), burst), noise.inputs["W"])
    noise.inputs["Detail"].default_value = 2.0
    lumps = b.math("MULTIPLY", b.math("SUBTRACT", noise.outputs["Fac"], 0.5), 2.0)
    length = b.math("MAXIMUM", b.math("ADD", length, b.math("MULTIPLY", lumps, s["Shape Noise"])),
                    0.15)
    # Along the spike: 1 at the tip, 1 - Spike Length at the inner end.
    step = b.math("DIVIDE", along, b.math("MAXIMUM", b.math("SUBTRACT", per_spike, 1.0), 1.0))
    fraction = b.math("SUBTRACT", 1.0, b.math("MULTIPLY", step, s["Spike Length"]))
    reach = b.math("MULTIPLY", length, fraction)
    # Bursts after the first start a little away from the centre.
    scatter = random(b, "FLOAT_VECTOR", (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), burst,
                     b.math("ADD", seed, 53.0))
    first = b.math("LESS_THAN", burst, 0.5)
    offset = b.vmath("SCALE", scatter, scale=b.math("MULTIPLY", s["Burst Scatter"],
                                                    b.math("SUBTRACT", 1.0, first)))
    centre = b.vmath("ADD", inp["Centre"], offset)
    position = b.vmath("ADD", centre, b.vmath("SCALE", direction,
                                              scale=b.math("MULTIPLY", radius_now, reach)))
    # Velocity: the point's own outward motion, plus noise.
    jitter = random(b, "FLOAT_VECTOR", (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), index,
                    b.math("ADD", seed, 71.0))
    outward = b.vmath("SCALE", direction, scale=b.math("MULTIPLY", b.math(
        "MULTIPLY", rate, reach), s["Outward Speed"]))
    velocity = b.vmath("ADD", outward, b.vmath("SCALE", jitter, scale=s["Speed Noise"]))
    # How much: fades over the duration, stronger towards the core of each spike.
    fade = b.math("POWER", remaining, s["Fade"])
    core = mix(b, s["Core Strength"], 1.0, fraction)
    weight = b.math("MULTIPLY", b.math("MULTIPLY", fade, core), live)
    radius = b.math("MULTIPLY", b.math("MULTIPLY", s["Thickness"], radius_now),
                    b.math("ADD", 0.5, b.math("MULTIPLY", 0.5, fraction)))
    # Never thinner than the points are apart, so a puff covers at least a voxel.
    radius = b.math("MAXIMUM", radius, b.math("MULTIPLY", s["Point Spacing"], 0.6))

    placed = b.new("GeometryNodeSetPosition")
    b.link(points.outputs[0], placed.inputs["Geometry"])
    b.link(position, placed.inputs["Position"])
    sized = b.new("GeometryNodeSetPointRadius")
    b.link(placed.outputs[0], sized.inputs["Points"])
    b.link(radius, sized.inputs["Radius"])
    tagged = tag_pyro_points(b, sized.outputs[0],
                             b.math("MULTIPLY", s["Density"], weight),
                             b.math("MULTIPLY", s["Temperature"], weight),
                             b.math("MULTIPLY", s["Fuel"], weight),
                             b.math("MULTIPLY", s["Expansion"], weight), velocity)
    dead = b.new("GeometryNodeDeleteGeometry", domain="POINT")
    b.link(tagged, dead.inputs["Geometry"])
    b.link(b.math("LESS_THAN", live, 0.5), dead.inputs["Selection"])
    b.link(dead.outputs[0], group_out.inputs["Points"])
    arrange(tree)
    return tree


def build_fill():
    tree = new_group(
        "Pyro Fill",
        "Points inside a mesh's volume or on its surface, spaced evenly, with the pyro "
        "attributes and a velocity from the mesh's motion, normals and a vector")
    socket(tree, "Mesh", "NodeSocketGeometry")
    socket(tree, "Motion", "NodeSocketVector", default=(0.0, 0.0, 0.0),
           description="Field: each vertex's velocity, in the emitter's space")
    settings_socket(tree)
    socket(tree, "Surface", "NodeSocketBool", default=False,
           description="Points on the surface instead of inside")
    socket(tree, "Weight", "NodeSocketFloat", default=1.0,
           description="Multiplies what is emitted (the fade over the duration)")
    socket(tree, "Points", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    s = separate(b, inp["Settings"], items_of(EMITTER_ITEMS) + items_of(EMITTER_USER_ITEMS))
    spacing = s["Point Spacing"]
    mesh = inp["Mesh"]
    # The vertices' motion and normals, carried to the points from the nearest surface.
    capture = b.new("GeometryNodeCaptureAttribute", domain="POINT")
    capture.capture_items.new("VECTOR", "Motion")
    b.link(mesh, capture.inputs["Geometry"])
    b.link(inp["Motion"], capture.inputs["Motion"])
    captured = capture.outputs["Geometry"]

    fog = b.new("GeometryNodeMeshToVolume")
    b.link(captured, fog.inputs["Mesh"])
    fog.inputs["Resolution Mode"].default_value = "Size"
    b.link(b.math("MULTIPLY", spacing, 0.5), fog.inputs["Voxel Size"])
    inside = b.new("GeometryNodeDistributePointsInVolume")
    inside.inputs["Mode"].default_value = "Grid"
    b.link(fog.outputs[0], inside.inputs["Volume"])
    b.link(spacing, inside.inputs["Spacing"])
    inside.inputs["Threshold"].default_value = 0.1
    on_faces = b.new("GeometryNodeDistributePointsOnFaces")
    b.link(captured, on_faces.inputs["Mesh"])
    b.link(b.math("DIVIDE", 1.0, b.math("MULTIPLY", spacing, spacing)), on_faces.inputs["Density"])
    b.link(s["Seed"], on_faces.inputs["Seed"])
    chosen = switch(b, "GEOMETRY", inp["Surface"], inside.outputs[0], on_faces.outputs["Points"])

    def nearest(value, kind="FLOAT_VECTOR"):
        sample = b.new("GeometryNodeSampleNearestSurface", data_type=kind)
        b.link(captured, sample.inputs["Mesh"])
        b.link(value, sample.inputs["Value"])
        return sample.outputs["Value"]

    normal = nearest(b.new("GeometryNodeInputNormal").outputs[0])
    motion = nearest(capture.outputs["Motion"])
    velocity = b.vmath("ADD", b.vmath("ADD", b.vmath("SCALE", motion, scale=s["Source"]),
                                      b.vmath("SCALE", normal, scale=s["Normal"])),
                       s["Velocity"])
    sized = b.new("GeometryNodeSetPointRadius")
    b.link(chosen, sized.inputs["Points"])
    b.link(b.math("MULTIPLY", spacing, 0.75), sized.inputs["Radius"])
    weight = inp["Weight"]
    tagged = tag_pyro_points(b, sized.outputs[0],
                             b.math("MULTIPLY", s["Density"], weight),
                             b.math("MULTIPLY", s["Temperature"], weight),
                             b.math("MULTIPLY", s["Fuel"], weight),
                             b.math("MULTIPLY", s["Expansion"], weight), velocity)
    b.link(tagged, group_out.inputs["Points"])
    arrange(tree)
    return tree


def build_particles():
    tree = new_group(
        "Pyro Particles",
        "A particle simulation off the emitter's surface: particles leave it, slow down, fall "
        "and fade, each emitting a puff that carries its velocity")
    socket(tree, "Emitter", "NodeSocketGeometry")
    settings_socket(tree)
    socket(tree, "Points", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    s = separate(b, inp["Settings"], items_of(EMITTER_ITEMS) + items_of(EMITTER_USER_ITEMS))
    sim_in = b.new("GeometryNodeSimulationInput")
    sim_out = b.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    sim_out.state_items.new("FLOAT", "Elapsed")
    dt = sim_in.outputs["Delta Time"]
    elapsed = sim_in.outputs["Elapsed"]
    emitter = inp["Emitter"]

    points = sim_in.outputs["Geometry"]
    velocity = named_attribute(b, "particle_velocity", "FLOAT_VECTOR")
    keep = b.math("MAXIMUM", b.math("SUBTRACT", 1.0, b.math("MULTIPLY", s["Drag"], dt)), 0.0)
    gravity = b.vmath("SCALE", s["Gravity"], scale=b.math("MULTIPLY", s["Particle Gravity"], dt))
    velocity = b.vmath("ADD", b.vmath("SCALE", velocity, scale=keep), gravity)
    points = store_attribute(b, points, "particle_velocity", velocity, "FLOAT_VECTOR")
    moved = b.new("GeometryNodeSetPosition")
    b.link(points, moved.inputs["Geometry"])
    b.link(b.vmath("SCALE", named_attribute(b, "particle_velocity", "FLOAT_VECTOR"), scale=dt),
           moved.inputs["Offset"])
    points = store_attribute(b, moved.outputs[0], "age", b.math("ADD", named_attribute(b, "age"),
                                                                 dt))
    dead = b.new("GeometryNodeDeleteGeometry", domain="POINT")
    b.link(points, dead.inputs["Geometry"])
    b.link(b.math("GREATER_THAN", named_attribute(b, "age"), s["Lifetime"]),
           dead.inputs["Selection"])
    alive = dead.outputs[0]

    area = b.new("GeometryNodeAttributeStatistic", data_type="FLOAT", domain="FACE")
    b.link(emitter, area.inputs["Geometry"])
    b.link(b.new("GeometryNodeInputMeshFaceArea").outputs[0], area.inputs["Attribute"])
    first = b.math("LESS_THAN", elapsed, b.math("MULTIPLY", dt, 0.5))
    count = b.math("ADD", b.math("MULTIPLY", s["Rate"], dt),
                   b.math("MULTIPLY", s["Particle Burst"], first))
    count = b.math("MULTIPLY", count, b.equal(s["Emit Type"], 3.0))
    spawn = b.new("GeometryNodeDistributePointsOnFaces")
    b.link(emitter, spawn.inputs["Mesh"])
    b.link(b.math("DIVIDE", count, b.math("MAXIMUM", area.outputs["Sum"], 1.0e-6)),
           spawn.inputs["Density"])
    b.link(b.math("ADD", s["Seed"], b.math("FLOOR", b.math("MULTIPLY", elapsed, 977.0))),
           spawn.inputs["Seed"])
    jitter = random(b, "FLOAT_VECTOR", (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), None,
                    b.math("ADD", s["Seed"], b.math("FLOOR", b.math("MULTIPLY", elapsed, 389.0))))
    start = b.vmath("ADD", b.vmath("SCALE", spawn.outputs["Normal"], scale=s["Particle Speed"]),
                    b.vmath("SCALE", jitter, scale=s["Particle Spread"]))
    born = store_attribute(b, spawn.outputs["Points"], "particle_velocity", start, "FLOAT_VECTOR")
    born = store_attribute(b, born, "age", 0.0)
    join = b.new("GeometryNodeJoinGeometry")
    b.link(born, join.inputs[0])
    b.link(alive, join.inputs[0])
    b.link(join.outputs[0], sim_out.inputs["Geometry"])
    b.link(b.math("ADD", elapsed, dt), sim_out.inputs["Elapsed"])

    # Each particle emits, fading over its life, with its own velocity.
    particles = sim_out.outputs["Geometry"]
    fade = clamp01(b, b.math("SUBTRACT", 1.0, b.math(
        "DIVIDE", named_attribute(b, "age"), b.math("MAXIMUM", s["Lifetime"], 1.0e-3))))
    sized = b.new("GeometryNodeSetPointRadius")
    b.link(particles, sized.inputs["Points"])
    b.link(s["Particle Radius"], sized.inputs["Radius"])
    velocity = b.vmath("ADD", b.vmath("SCALE", named_attribute(b, "particle_velocity",
                                                                "FLOAT_VECTOR"),
                                      scale=s["Source"]), s["Velocity"])
    tagged = tag_pyro_points(b, sized.outputs[0], b.math("MULTIPLY", s["Density"], fade),
                             b.math("MULTIPLY", s["Temperature"], fade),
                             b.math("MULTIPLY", s["Fuel"], fade),
                             b.math("MULTIPLY", s["Expansion"], fade), velocity)
    b.link(tagged, group_out.inputs["Points"])
    arrange(tree)
    return tree


def build_emitter_modifier(blocks):
    tree = new_group(
        EMITTER_MODIFIER,
        "Turn this mesh into pyro source points: spiky bursts, its volume or surface, or "
        "particles. Follow it with Pyro Simulation, or put this object in another Pyro "
        "Simulation's Emitters collection")
    tree.is_modifier = True
    socket(tree, "Geometry", "NodeSocketGeometry")
    socket(tree, "Geometry", "NodeSocketGeometry", in_out="OUTPUT")
    preset = blocks["Pyro Emitter Preset"]
    preset_inputs = {item.name: item for item in preset.interface.items_tree
                     if getattr(item, "in_out", None) == "INPUT"}
    panels = (
        ("Preset", ["Preset", "Strength", "Speed", "Emit Type"]),
        ("Emission", ["Density", "Temperature", "Fuel", "Expansion", "Start Frame", "Duration",
                      "Fade"]),
        ("Burst Shape", ["Size", "Outward Expansion", "Expansion Curve", "Spikes", "Roundness",
                         "Spike Length", "Thickness", "Core Strength", "Shape Noise",
                         "Direction", "Spread"]),
        ("Bursts", ["Bursts", "Burst Delay", "Burst Scatter"]),
        ("Velocity", ["Initial Velocity", "Outward Speed", "Speed Noise", "Source", "Normal",
                      "Velocity"]),
        ("Particles", ["Rate", "Particle Burst", "Particle Speed", "Particle Spread", "Drag",
                       "Particle Gravity", "Lifetime", "Particle Radius", "Gravity"]),
        ("Points", ["Point Spacing", "Seed", "Show Mesh"]),
    )
    extra = {"Show Mesh": ("NodeSocketBool", True, "Keep the mesh visible")}
    copy_panels(tree, panels, preset_inputs, extra)
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    settings_node = b.new("GeometryNodeGroup", node_tree=preset)
    for name in preset_inputs:
        if name in inp:
            b.link(inp[name], settings_node.inputs[name])
    settings = settings_node.outputs["Settings"]
    s = separate(b, settings, items_of(EMITTER_ITEMS) + items_of(EMITTER_USER_ITEMS))

    # Time, and the emitter's world placement (velocities are world-oriented).
    time = b.new("GeometryNodeInputSceneTime")
    frame = time.outputs["Frame"]
    seconds_per_frame = switch(b, "FLOAT", b.math("GREATER_THAN", b.math("ABSOLUTE", frame), 0.5),
                               1.0 / 24.0, b.math("DIVIDE", time.outputs["Seconds"], frame))
    self_info = b.new("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    b.link(b.new("GeometryNodeSelfObject").outputs[0], self_info.inputs["Object"])
    world = self_info.outputs["Transform"]
    mesh = inp["Geometry"]
    # The mesh's motion: a zone remembers its world positions of the previous frame.
    to_world = b.new("GeometryNodeTransform")
    to_world.inputs["Mode"].default_value = "Matrix"
    b.link(mesh, to_world.inputs["Geometry"])
    b.link(world, to_world.inputs["Transform"])
    sim_in = b.new("GeometryNodeSimulationInput")
    sim_out = b.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    sim_out.state_items.new("GEOMETRY", "Moving")
    b.link(to_world.outputs[0], sim_out.inputs["Geometry"])
    motion_world = motion_since(b, to_world.outputs[0], sim_in.outputs["Geometry"],
                                sim_in.outputs["Delta Time"])
    # The world mesh with each vertex's motion, out of the zone.
    b.link(store_attribute(b, to_world.outputs[0], "pyro_motion", motion_world, "FLOAT_VECTOR"),
           sim_out.inputs["Moving"])
    with_motion = sim_out.outputs["Moving"]
    # Back onto the mesh's own vertices as a field (the counts match).
    sample = b.new("GeometryNodeSampleIndex", data_type="FLOAT_VECTOR", domain="POINT")
    b.link(with_motion, sample.inputs["Geometry"])
    b.link(named_attribute(b, "pyro_motion", "FLOAT_VECTOR"), sample.inputs["Value"])
    b.link(b.new("GeometryNodeInputIndex").outputs[0], sample.inputs["Index"])
    motion = sample.outputs[0]

    # Emission window for volume, surface and particles: from Start Frame for Duration.
    age = b.math("SUBTRACT", frame, s["Start Frame"])
    endless = b.math("LESS_THAN", s["Duration"], 1.0e-6)
    progress = clamp01(b, b.math("DIVIDE", age, b.math("MAXIMUM", s["Duration"], 1.0e-6)))
    window = b.math("MULTIPLY", b.math("GREATER_THAN", age, -1.0e-4),
                    b.math("MAXIMUM", endless, b.math("LESS_THAN", age, s["Duration"])))
    fade = switch(b, "FLOAT", endless,
                  b.math("POWER", b.math("SUBTRACT", 1.0, progress), s["Fade"]), 1.0)
    weight = b.math("MULTIPLY", window, fade)

    bounds = b.new("GeometryNodeBoundBox")
    b.link(mesh, bounds.inputs["Geometry"])
    centre = b.vmath("SCALE", b.vmath("ADD", bounds.outputs["Min"], bounds.outputs["Max"]),
                     scale=0.5)
    burst = b.new("GeometryNodeGroup", node_tree=blocks["Pyro Burst"])
    b.link(centre, burst.inputs["Centre"])
    b.link(settings, burst.inputs["Settings"])
    b.link(frame, burst.inputs["Frame"])
    b.link(seconds_per_frame, burst.inputs["Seconds Per Frame"])
    fill = b.new("GeometryNodeGroup", node_tree=blocks["Pyro Fill"])
    b.link(mesh, fill.inputs["Mesh"])
    b.link(motion, fill.inputs["Motion"])
    b.link(settings, fill.inputs["Settings"])
    b.link(b.equal(s["Emit Type"], 2.0), fill.inputs["Surface"])
    b.link(weight, fill.inputs["Weight"])
    particles = b.new("GeometryNodeGroup", node_tree=blocks["Pyro Particles"])
    b.link(mesh, particles.inputs["Emitter"])
    b.link(settings, particles.inputs["Settings"])
    by_type = b.new("GeometryNodeIndexSwitch", data_type="GEOMETRY")
    while len([x for x in by_type.inputs if x.name not in ("Index", "")]) < 4:
        by_type.index_switch_items.new()
    b.link(s["Emit Type"], by_type.inputs["Index"])
    item_inputs = [x for x in by_type.inputs if x.name not in ("Index", "")]
    b.link(burst.outputs[0], item_inputs[0])
    b.link(fill.outputs[0], item_inputs[1])
    b.link(fill.outputs[0], item_inputs[2])
    b.link(switch(b, "GEOMETRY", window, None, particles.outputs[0]), item_inputs[3])
    points = by_type.outputs[0]
    # Bursts move with the object: add its motion at the burst's centre.
    centre_motion = b.new("GeometryNodeSampleIndex", data_type="FLOAT_VECTOR", domain="POINT")
    b.link(with_motion, centre_motion.inputs["Geometry"])
    b.link(named_attribute(b, "pyro_motion", "FLOAT_VECTOR"), centre_motion.inputs["Value"])
    centre_motion.inputs["Index"].default_value = 0
    is_burst = b.equal(s["Emit Type"], 0.0)
    carried = b.vmath("SCALE", centre_motion.outputs[0],
                      scale=b.math("MULTIPLY", s["Source"], is_burst))
    # Velocities so far are in the emitter's space; the simulation wants them world-oriented.
    turn = b.new("FunctionNodeTransformDirection")
    b.link(b.vmath("ADD", named_attribute(b, "velocity", "FLOAT_VECTOR"),
                   b.vmath("SCALE", s["Velocity"], scale=is_burst)), turn.inputs["Direction"])
    b.link(world, turn.inputs["Transform"])
    oriented = b.vmath("ADD", turn.outputs[0], carried)
    points = store_attribute(b, points, "velocity", oriented, "FLOAT_VECTOR")
    # Without Initial Velocity the points carry none, and the gas keeps its own.
    still = b.new("GeometryNodeRemoveAttribute")
    b.link(points, still.inputs["Geometry"])
    still.inputs["Name"].default_value = "velocity"
    points = switch(b, "GEOMETRY", s["Initial Velocity"], still.outputs[0], points)

    join = b.new("GeometryNodeJoinGeometry")
    b.link(switch(b, "GEOMETRY", inp["Show Mesh"], None, mesh), join.inputs[0])
    b.link(points, join.inputs[0])
    b.link(join.outputs[0], group_out.inputs["Geometry"])
    input_item(tree, "Preset").default_value = "Explosion"
    input_item(tree, "Emit Type").default_value = "Burst"
    arrange(tree)
    return tree


def copy_panels(tree, panels, preset_inputs, extra):
    """Modifier inputs in panels, copied from a preset block's inputs (or `extra`)."""
    for title, names in panels:
        panel = tree.interface.new_panel(title, default_closed=title != "Preset")
        for name in names:
            if name in extra:
                kind, default, description = extra[name][:3]
                socket(tree, name, kind, default=default, parent=panel, description=description)
                continue
            source = preset_inputs[name]
            item = socket(tree, name, source.socket_type, parent=panel,
                          description=source.description)
            for attribute in ("default_value", "min_value", "max_value", "subtype"):
                if hasattr(source, attribute):
                    try:
                        setattr(item, attribute, getattr(source, attribute))
                    except (AttributeError, TypeError):
                        pass


# --------------------------------------------------------------------------------------------
# Simulation blocks.

def build_point_source():
    tree = new_group(
        "Pyro Point Source",
        "Turn pyro points into the Gas Simulation Step node's Sources: a puff of each point's "
        "radius emitting its density, temperature, fuel and expansion, with its velocity")
    socket(tree, "Points", "NodeSocketGeometry",
           description="Points with a radius and the attributes density, temperature, fuel, "
                       "expansion and velocity (world units per second)")
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
    # Each voxel takes the values of its nearest point, inside the points' puffs.
    nearest = b.new("GeometryNodeSampleNearest", domain="POINT")
    b.link(points, nearest.inputs["Geometry"])

    def nearest_value(name, kind):
        sample = b.new("GeometryNodeSampleIndex", data_type=kind, domain="POINT")
        b.link(points, sample.inputs["Geometry"])
        b.link(named_attribute(b, name, kind), sample.inputs["Value"])
        b.link(nearest.outputs["Index"], sample.inputs["Index"])
        return sample.outputs[0]

    inside = b.new("GeometryNodeSampleGrid")
    b.link(shape, inside.inputs["Grid"])
    weight = inside.outputs[0]
    def most(name):
        stat = b.new("GeometryNodeAttributeStatistic", data_type="FLOAT", domain="POINT")
        b.link(points, stat.inputs["Geometry"])
        b.link(named_attribute(b, name), stat.inputs["Attribute"])
        return stat.outputs["Max"]

    volume = None
    for name in ("density", "temperature"):
        volume = store_grid(b, volume, name, field_grid(
            b, shape, b.math("MULTIPLY", weight, nearest_value(name, "FLOAT"))))
    # Fuel and expansion grids only where some point emits them: any fuel grid turns fire on.
    for name, grid_name in (("fuel", "fuel"), ("expansion", "divergence")):
        with_grid = store_grid(b, volume, grid_name, field_grid(
            b, shape, b.math("MULTIPLY", weight, nearest_value(name, "FLOAT"))))
        volume = switch(b, "GEOMETRY", b.math("GREATER_THAN", most(name), 0.0), volume, with_grid)
    moving = store_grid(b, volume, "velocity",
                        field_grid(b, shape, nearest_value("velocity", "FLOAT_VECTOR"), "VECTOR"),
                        kind="VECTOR_FLOAT")
    # A velocity grid only for points that carry a velocity.
    exists = b.new("GeometryNodeInputNamedAttribute", data_type="FLOAT_VECTOR")
    exists.inputs["Name"].default_value = "velocity"
    carried = b.new("GeometryNodeAttributeStatistic", data_type="FLOAT", domain="POINT")
    b.link(points, carried.inputs["Geometry"])
    b.link(exists.outputs["Exists"], carried.inputs["Attribute"])
    volume = switch(b, "GEOMETRY", b.math("GREATER_THAN", carried.outputs["Max"], 0.5), volume,
                    moving)
    has_points = b.math("GREATER_THAN", point_count(b, points), 0.0)
    b.link(switch(b, "GEOMETRY", has_points, None, volume), group_out.inputs["Sources"])
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
        "Simulation, Pyro Preset or the Gas Simulation Step node; they apply in chain order, so "
        "a layer masked by Speed reacts to the layers before it")
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


def layer_node(b, layer, strength, scale, speed, seed, mask, level, invert=False):
    node = b.new("GeometryNodeGroup", node_tree=layer)
    b.set(node.inputs["Strength"], strength)
    b.set(node.inputs["Scale"], scale)
    b.set(node.inputs["Speed"], speed)
    b.set(node.inputs["Seed"], seed)
    node.inputs["Mask"].default_value = mask
    node.inputs["Mask Level"].default_value = level
    node.inputs["Invert"].default_value = invert
    off = b.math("LESS_THAN", strength, 1.0e-6)
    return switch(b, "GEOMETRY", off, node.outputs[0], None)


def build_solver_preset(layer):
    def add_layers(b, inp, values):
        # Fine swirls masked by speed, then disturbance in the thin smoke, then any layers given.
        fine = layer_node(b, layer, values["Fine Turbulence"], values["Fine Scale"],
                          inp["Turbulence Speed"], b.math("ADD", inp["Turbulence Seed"], 1.0),
                          "Speed", 0.5)
        disturbance = layer_node(b, layer, values["Disturbance"], values["Disturbance Scale"],
                                 b.math("MULTIPLY", inp["Turbulence Speed"], 4.0),
                                 b.math("ADD", inp["Turbulence Seed"], 2.0), "Density", 0.15,
                                 invert=True)
        join = b.new("GeometryNodeJoinGeometry")
        b.link(inp["Turbulence Layers"], join.inputs[0])
        b.link(disturbance, join.inputs[0])
        b.link(fine, join.inputs[0])
        values["Turbulence Layers"] = join.outputs[0]

    return preset_block(
        "Pyro Preset",
        "Collect a pyro simulation's settings into one Settings bundle: a look preset's values, "
        "or the Custom values below, scaled by Strength and Speed",
        SOLVER_ITEMS, SOLVER_USER_ITEMS, SOLVER_PRESETS,
        strength=("Turbulence", "Fine Turbulence", "Disturbance", "Gas Release"),
        extra=add_layers)


def build_emitter_preset():
    return preset_block(
        "Pyro Emitter Preset",
        "Collect a pyro emitter's settings into one Settings bundle: a preset's values, or the "
        "Custom values below, scaled by Strength and Speed",
        EMITTER_ITEMS, EMITTER_USER_ITEMS, EMITTER_PRESETS, strength=EMITTER_STRENGTH,
        speed=EMITTER_SPEED, menus=(("Emit Type", EMIT_TYPES),))


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
    corner_x, corner_y, _ = xyz(b, corner)
    _, _, low_z = xyz(b, bounds.outputs["Min"])
    size = xyz(b, inp["Domain Size"])
    b.link(vector(b, corner_x, corner_y, b.math("SUBTRACT", low_z, b.math("MULTIPLY", size[2], 0.1))),
           group_out.inputs["Domain Min"])
    for axis, name in enumerate("XYZ"):
        cells = b.math("CEIL", b.math("DIVIDE", size[axis], inp["Voxel Size"]))
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


def build_gather():
    tree = new_group(
        "Pyro Gather",
        "The pyro points to simulate: from this object (after a Pyro Emitter) and from the "
        "objects in the Emitters collection, in world space. A mesh without any becomes a "
        "plain smoke emitter")
    socket(tree, "Geometry", "NodeSocketGeometry", description="This object's geometry, in world "
                                                              "space")
    socket(tree, "Emitters", "NodeSocketCollection")
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4)
    socket(tree, "Points", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    inp = group_in.outputs
    own = b.new("GeometryNodeSeparateComponents")
    b.link(inp["Geometry"], own.inputs[0])
    info = b.new("GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    b.link(inp["Emitters"], info.inputs["Collection"])
    realize = b.new("GeometryNodeRealizeInstances")
    b.link(info.outputs[0], realize.inputs["Geometry"])
    others = b.new("GeometryNodeSeparateComponents")
    b.link(realize.outputs[0], others.inputs[0])
    join = b.new("GeometryNodeJoinGeometry")
    b.link(own.outputs["Point Cloud"], join.inputs[0])
    b.link(others.outputs["Point Cloud"], join.inputs[0])
    keep = b.new("GeometryNodeDeleteGeometry", domain="POINT")
    b.link(join.outputs[0], keep.inputs["Geometry"])
    not_pyro = b.new("FunctionNodeBooleanMath", operation="NOT")
    b.link(named_attribute(b, "pyro", "BOOLEAN"), not_pyro.inputs[0])
    b.link(not_pyro.outputs[0], keep.inputs["Selection"])
    found = keep.outputs[0]
    # No pyro points at all: fill this object's mesh with plain smoke.
    plain = b.new("GeometryNodeMeshToVolume")
    b.link(own.outputs["Mesh"], plain.inputs["Mesh"])
    plain.inputs["Resolution Mode"].default_value = "Size"
    b.link(inp["Voxel Size"], plain.inputs["Voxel Size"])
    fill = b.new("GeometryNodeDistributePointsInVolume")
    fill.inputs["Mode"].default_value = "Grid"
    b.link(plain.outputs[0], fill.inputs["Volume"])
    b.link(inp["Voxel Size"], fill.inputs["Spacing"])
    sized = b.new("GeometryNodeSetPointRadius")
    b.link(fill.outputs[0], sized.inputs["Points"])
    b.link(b.math("MULTIPLY", inp["Voxel Size"], 0.75), sized.inputs["Radius"])
    smoke = tag_pyro_points(b, sized.outputs[0], 1.0, 1.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    any_found = b.math("GREATER_THAN", point_count(b, found), 0.0)
    b.link(switch(b, "GEOMETRY", any_found, smoke, found), group_out.inputs["Points"])
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
             "Flame Heat", "Flame Smoke", "Flame Cooling", "Gas Release", "Add Emission"]
    kinds = dict(items_of(SOLVER_ITEMS) + items_of(SOLVER_USER_ITEMS))
    s = separate(b, inp["Settings"], [(name, kinds[name]) for name in names])
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
        "object's mesh if asked")
    socket(tree, "State", "NodeSocketGeometry")
    socket(tree, "Mesh", "NodeSocketGeometry", description="The object's own mesh")
    socket(tree, "To World", "NodeSocketMatrix", description="The object's world matrix")
    socket(tree, "Show Mesh", "NodeSocketBool", default=True)
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
    mesh_only = b.new("GeometryNodeSeparateComponents")
    b.link(inp["Mesh"], mesh_only.inputs[0])
    join = b.new("GeometryNodeJoinGeometry")
    b.link(switch(b, "GEOMETRY", inp["Show Mesh"], None, mesh_only.outputs["Mesh"]),
           join.inputs[0])
    b.link(to_local.outputs[0], join.inputs[0])
    b.link(join.outputs[0], group_out.inputs["Geometry"])
    input_item(tree, "Material").default_value = material
    arrange(tree)
    return tree


def build_simulation_modifier(blocks, material):
    tree = new_group(
        MODIFIER,
        "Simulate smoke and fire from pyro points: this object's (after a Pyro Emitter) and "
        "those of the objects in Emitters. Pick a look preset, or Custom to set everything. The "
        "domain is placed around the emitters on the first frame")
    tree.is_modifier = True
    socket(tree, "Geometry", "NodeSocketGeometry")
    socket(tree, "Geometry", "NodeSocketGeometry", in_out="OUTPUT")
    preset = blocks["Pyro Preset"]
    preset_inputs = {item.name: item for item in preset.interface.items_tree
                     if getattr(item, "in_out", None) == "INPUT"}
    panels = (
        ("Preset", ["Preset", "Strength", "Speed"]),
        ("Domain", ["Voxel Size", "Domain Size", "Emitters", "Colliders"]),
        ("Fire", ["Burn Rate", "Flame Heat", "Flame Smoke", "Flame Cooling", "Gas Release"]),
        ("Forces", ["Heat Buoyancy", "Density Buoyancy", "Gravity", "Add Emission"]),
        ("Turbulence", ["Turbulence", "Turbulence Scale", "Turbulence Speed", "Turbulence Seed",
                        "Turbulence Mask", "Fine Turbulence", "Fine Scale", "Disturbance",
                        "Disturbance Scale"]),
        ("Solver", ["Sparse GPU", "CFL", "Max Substeps"]),
        ("Display", ["Show Mesh"]),
    )
    extra = {
        "Voxel Size": ("NodeSocketFloat", 0.05,
                       "Size of a voxel in world units. Halving it gives eight times the voxels"),
        "Domain Size": ("NodeSocketVector", (2.0, 2.0, 4.0),
                        "Size of the simulated box in world units, centred on the emitters and "
                        "starting a little below them. Only the parts with smoke cost anything"),
        "Emitters": ("NodeSocketCollection", None,
                     "Objects with a Pyro Emitter modifier to simulate as well"),
        "Colliders": ("NodeSocketCollection", None,
                      "Objects the smoke flows around. They may be animated"),
        "Show Mesh": ("NodeSocketBool", True, "Keep this object's mesh visible"),
    }
    copy_panels(tree, panels, preset_inputs, extra)
    input_item(tree, "Domain Size").subtype = "XYZ"
    input_item(tree, "Voxel Size").min_value = 1.0e-3
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
    for item in preset_inputs:
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
    world_geometry = to_world.outputs["Geometry"]
    points = block("Pyro Gather", Geometry=world_geometry, Emitters=inp["Emitters"],
                   Voxel_Size=inp["Voxel Size"]).outputs[0]
    placement = b.new("GeometryNodeJoinGeometry")
    b.link(world_geometry, placement.inputs[0])
    b.link(points, placement.inputs[0])
    domain = block("Pyro Domain", Emitter=placement.outputs[0], Voxel_Size=inp["Voxel Size"],
                   Domain_Size=inp["Domain Size"])

    # The zone keeps the domain's corner from its first frame and the colliders of the previous
    # step for their motion.
    sim_in = b.new("GeometryNodeSimulationInput")
    sim_out = b.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    for kind, name in (("VECTOR", "Domain Min"), ("GEOMETRY", "Previous Colliders")):
        sim_out.state_items.new(kind, name)
    b.link(domain.outputs["Domain Min"], sim_in.inputs["Domain Min"])
    delta = sim_in.outputs["Delta Time"]
    colliders = block("Pyro Colliders", Collection=inp["Colliders"],
                      Previous=sim_in.outputs["Previous Colliders"], Delta_Time=delta,
                      Voxel_Size=inp["Voxel Size"])
    sources = block("Pyro Point Source", Points=points, Voxel_Size=inp["Voxel Size"])
    solver = block("Pyro Solver", State=sim_in.outputs["Geometry"],
                   Sources=sources.outputs["Sources"], Colliders=colliders.outputs["Colliders"],
                   Settings=settings, Delta_Time=delta, Domain_Min=sim_in.outputs["Domain Min"],
                   Voxel_Size=inp["Voxel Size"])
    for axis in "XYZ":
        b.link(domain.outputs["Resolution " + axis], solver.inputs["Resolution " + axis])
    b.link(solver.outputs["State"], sim_out.inputs["Geometry"])
    b.link(sim_in.outputs["Domain Min"], sim_out.inputs["Domain Min"])
    b.link(colliders.outputs["Geometry"], sim_out.inputs["Previous Colliders"])

    display = block("Pyro Display", State=sim_out.outputs["Geometry"], Mesh=inp["Geometry"],
                    To_World=self_info.outputs["Transform"], Show_Mesh=inp["Show Mesh"])
    display.inputs["Material"].default_value = material
    b.link(display.outputs[0], group_out.inputs["Geometry"])
    input_item(tree, "Preset").default_value = "Smoke"
    arrange(tree)
    return tree


BLOCK_NAMES = ("Pyro Emitter Preset", "Pyro Burst", "Pyro Fill", "Pyro Particles",
               "Pyro Preset", "Pyro Gather", "Pyro Domain", "Pyro Colliders", "Pyro Solver",
               "Pyro Display", "Pyro Point Source", "Pyro Collider", "Turbulence Layer")
MODIFIER_NAMES = (EMITTER_MODIFIER, MODIFIER)


def build_assets(mark=True):
    """Create the groups and material in the current file; returns the (emitter, simulation)
    modifier groups."""
    layer = build_turbulence_layer()
    collider = build_collider()
    material = build_material()
    blocks = {
        "Turbulence Layer": layer,
        "Pyro Collider": collider,
        "Pyro Emitter Preset": build_emitter_preset(),
        "Pyro Burst": build_burst(),
        "Pyro Fill": build_fill(),
        "Pyro Particles": build_particles(),
        "Pyro Preset": build_solver_preset(layer),
        "Pyro Gather": build_gather(),
        "Pyro Domain": build_domain(),
        "Pyro Colliders": build_colliders(collider),
        "Pyro Solver": build_solver(),
        "Pyro Display": build_display(material),
        "Pyro Point Source": build_point_source(),
    }
    emitter = build_emitter_modifier(blocks)
    simulation = build_simulation_modifier(blocks, material)
    if mark:
        for tree, catalog in [(blocks[name], CATALOG_PYRO) for name in BLOCK_NAMES] + [
                (emitter, CATALOG_PHYSICS), (simulation, CATALOG_PHYSICS)]:
            tree.asset_mark()
            tree.asset_data.catalog_id = catalog[0]
            tree.asset_data.description = tree.description
    return emitter, simulation


def install(datafiles):
    """Write the asset file and register its catalogs in a Blender's datafiles folder."""
    assets = os.path.join(datafiles, "assets")
    nodes = os.path.join(assets, "nodes")
    os.makedirs(nodes, exist_ok=True)
    build_assets()
    path = os.path.join(nodes, ASSET_FILE)
    groups = {bpy.data.node_groups[name] for name in BLOCK_NAMES + MODIFIER_NAMES}
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
