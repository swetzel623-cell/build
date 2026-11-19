# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared helpers for running the Gas Simulation Step node inside a patched Blender build.

The helpers build Geometry Nodes trees from Python, feed arbitrary dense initial state through a
data mesh, and read the simulated grids back through a point cloud so tests can compare voxels with
NumPy. They are only meaningful inside Blender (`blender -b --python ...`).
"""

from __future__ import annotations

import bpy
import numpy as np

GAS_NODE = "GeometryNodeMantaGasStep"
GRID_NAMES = ("density", "temperature", "velocity", "pressure")
# The Sparse GPU path keeps pressure on the device; its output has no pressure grid (the CPU
# fallback of that path still writes one).
SPARSE_GRID_NAMES = ("density", "temperature", "velocity")


def state_grids_match(names, zone, extra=()):
    """Whether a state volume's grid names are the ones `zone`'s solver path writes."""
    expected = set(zone.grid_names) | set(extra)
    return set(names) - ({"pressure"} if zone.sparse else set()) == expected
DIAG_NAMES = ("substeps", "max_velocity", "import_ms", "solve_ms", "export_ms",
              "pressure_iterations", "pressure_residual", "active_bricks")


class Zone:
    """Configuration of one Simulation Zone containing one Gas Simulation Step node."""

    def __init__(self, res, origin=(-1.0, -1.0, -1.0), voxel=0.0625, prefix="", **params):
        self.res = tuple(int(v) for v in res)
        self.origin = tuple(float(v) for v in origin)
        self.voxel = float(voxel)
        self.prefix = prefix
        # Transform used for the *initial* grids; defaults to the node's own transform.
        self.init_origin = params.pop("init_origin", self.origin)
        self.init_voxel = params.pop("init_voxel", self.voxel)
        self.init_density_type = params.pop("init_density_type", "FLOAT")
        # Resolution of the initial grids; None means the node's own resolution.
        self.init_res = params.pop("init_res", None)
        self.readback = params.pop("readback", True)
        # More state grids to read back, such as ("fuel", "flame").
        self.extra_grids = tuple(params.pop("extra_grids", ()))
        # Callable (builder, zone) -> geometry socket for the node's Turbulence Layers.
        self.layers = params.pop("layers", None)
        # Callables (builder, zone) -> volume socket for the node's Sources and Colliders.
        self.sources = params.pop("sources", None)
        self.colliders = params.pop("colliders", None)
        self.params = params
        self.data_object = None

    @property
    def sparse(self):
        return bool(self.params.get("Sparse GPU", False))

    @property
    def grid_names(self):
        """The state grids read back: no pressure from the Sparse GPU path."""
        return SPARSE_GRID_NAMES if self.sparse else GRID_NAMES

    @property
    def cells(self):
        return self.res[0] * self.res[1] * self.res[2]


def reset_scene(fps=30, frame_end=250):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = frame_end
    # Node warnings are only logged for the active dependency graph; in background mode nothing
    # activates it until the view layer is updated explicitly.
    bpy.context.view_layer.update()
    return scene


def make_data_object(name, zone, density=None, temperature=None, velocity=None):
    """Create a mesh whose vertices carry one dense initial-state sample per voxel (X fastest)."""
    count = zone.cells
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(count)
    for attr, kind, values, width in (
        ("density", "FLOAT", density, 1),
        ("temperature", "FLOAT", temperature, 1),
        ("velocity", "FLOAT_VECTOR", velocity, 3),
    ):
        layer = mesh.attributes.new(attr, kind, "POINT")
        array = np.zeros(count * width, dtype=np.float32)
        if values is not None:
            array[:] = np.asarray(values, dtype=np.float32).reshape(-1)
        layer.data.foreach_set("vector" if width == 3 else "value", array)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    zone.data_object = obj
    return obj


def dense(zone, fn):
    """Evaluate fn(ix, iy, iz) on the voxel lattice and return a flat X-fastest float32 array."""
    nx, ny, nz = zone.res
    iz, iy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return np.asarray(fn(ix, iy, iz), dtype=np.float32).reshape(-1)


class _Builder:
    def __init__(self, tree):
        self.tree = tree
        self.nodes = tree.nodes
        self.links = tree.links

    def new(self, idname, **props):
        node = self.nodes.new(idname)
        for key, value in props.items():
            setattr(node, key, value)
        return node

    def link(self, out_socket, in_socket):
        self.links.new(out_socket, in_socket)

    def set(self, socket, value):
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

    def voxel_index(self, position, origin, voxel, res):
        """Flat X-fastest voxel index of a world-space position field."""
        sub = self.new("ShaderNodeVectorMath", operation="SUBTRACT")
        self.link(position, sub.inputs[0])
        sub.inputs[1].default_value = origin
        scale = self.new("ShaderNodeVectorMath", operation="SCALE")
        self.link(sub.outputs[0], scale.inputs[0])
        scale.inputs["Scale"].default_value = 1.0 / voxel
        sep = self.new("ShaderNodeSeparateXYZ")
        self.link(scale.outputs[0], sep.inputs[0])
        ix, iy, iz = (self.math("ROUND", sep.outputs[i]) for i in range(3))
        inner = self.math("ADD", iy, self.math("MULTIPLY", iz, float(res[1])))
        return self.math("ADD", ix, self.math("MULTIPLY", inner, float(res[0])))

    def index_xyz(self, res):
        """Integer voxel coordinates of the point index (X fastest)."""
        index = self.new("GeometryNodeInputIndex").outputs[0]
        nx, ny = float(res[0]), float(res[1])
        ix = self.math("FLOORED_MODULO", index, nx)
        rest = self.math("FLOOR", self.math("DIVIDE", index, nx))
        iy = self.math("FLOORED_MODULO", rest, ny)
        iz = self.math("FLOOR", self.math("DIVIDE", rest, ny))
        return ix, iy, iz


def _initial_state(b, zone):
    """Volume with density/temperature/velocity grids sampled from the zone's data mesh."""
    topology = b.new("GeometryNodeCubeGridTopology")
    o, v, r = zone.init_origin, zone.init_voxel, zone.init_res or zone.res
    topology.inputs["Bounds Min"].default_value = o
    topology.inputs["Bounds Max"].default_value = tuple(o[i] + v * r[i] for i in range(3))
    for axis, count in zip("XYZ", r):
        topology.inputs["Resolution " + axis].default_value = count

    info = b.new("GeometryNodeObjectInfo")
    info.inputs["Object"].default_value = zone.data_object
    position = b.new("GeometryNodeInputPosition").outputs[0]
    index = b.voxel_index(position, o, v, r)

    volume = None
    for name in ("density", "temperature", "velocity"):
        vector = name == "velocity"
        kind = "FLOAT_VECTOR" if vector else "FLOAT"
        sock = "VECTOR" if vector else "FLOAT"
        if name == "density" and zone.init_density_type != "FLOAT":
            kind, sock = "INT", "INT"
        attr = b.new("GeometryNodeInputNamedAttribute", data_type=kind)
        attr.inputs["Name"].default_value = name
        sample = b.new("GeometryNodeSampleIndex", data_type=kind, domain="POINT")
        b.link(info.outputs["Geometry"], sample.inputs["Geometry"])
        b.link(attr.outputs["Attribute"], sample.inputs["Value"])
        b.link(index, sample.inputs["Index"])

        to_grid = b.new("GeometryNodeFieldToGrid", data_type=sock)
        to_grid.grid_items.clear()
        to_grid.grid_items.new(sock, "Value")
        b.link(topology.outputs["Topology"], to_grid.inputs["Topology"])
        b.link(sample.outputs["Value"], to_grid.inputs["Value"])

        store = b.new("GeometryNodeStoreNamedGrid", data_type=kind if kind != "FLOAT_VECTOR" else "VECTOR_FLOAT")
        store.inputs["Name"].default_value = name
        b.link(to_grid.outputs["Value"], store.inputs["Grid"])
        if volume is not None:
            b.link(volume, store.inputs["Volume"])
        volume = store.outputs["Volume"]
    return volume


def grid_volume(b, zone, items):
    """Volume on the zone's own voxels with one grid per (name, kind, value) item.

    `kind` is "FLOAT" or "VECTOR"; `value` is a constant or a field socket, evaluated at the
    voxel positions (b.voxel_coords gives integer voxel coordinates as a field).
    """
    topology = b.new("GeometryNodeCubeGridTopology")
    o, v, r = zone.origin, zone.voxel, zone.res
    topology.inputs["Bounds Min"].default_value = o
    topology.inputs["Bounds Max"].default_value = tuple(o[i] + v * r[i] for i in range(3))
    for axis, count in zip("XYZ", r):
        topology.inputs["Resolution " + axis].default_value = count
    volume = None
    for name, kind, value in items:
        to_grid = b.new("GeometryNodeFieldToGrid", data_type=kind)
        to_grid.grid_items.clear()
        to_grid.grid_items.new(kind, "Value")
        b.link(topology.outputs["Topology"], to_grid.inputs["Topology"])
        b.set(to_grid.inputs["Value"], value)
        store = b.new("GeometryNodeStoreNamedGrid",
                      data_type="FLOAT" if kind == "FLOAT" else "VECTOR_FLOAT")
        store.inputs["Name"].default_value = name
        b.link(to_grid.outputs["Value"], store.inputs["Grid"])
        if volume is not None:
            b.link(volume, store.inputs["Volume"])
        volume = store.outputs["Volume"]
    return volume


def voxel_coords(b, zone):
    """Field of the voxel coordinates (rounded) of each grid voxel's position, as x, y, z sockets."""
    position = b.new("GeometryNodeInputPosition").outputs[0]
    sub = b.new("ShaderNodeVectorMath", operation="SUBTRACT")
    b.link(position, sub.inputs[0])
    sub.inputs[1].default_value = zone.origin
    scale = b.new("ShaderNodeVectorMath", operation="SCALE")
    b.link(sub.outputs[0], scale.inputs[0])
    scale.inputs["Scale"].default_value = 1.0 / zone.voxel
    sep = b.new("ShaderNodeSeparateXYZ")
    b.link(scale.outputs[0], sep.inputs[0])
    return tuple(b.math("ROUND", sep.outputs[i]) for i in range(3))


def box_field(b, zone, lo, hi):
    """1.0 inside the voxel box lo <= (x, y, z) < hi, else 0.0. Bounds may be sockets."""
    coords = voxel_coords(b, zone)
    inside = None
    for axis in range(3):
        above = b.math("GREATER_THAN", coords[axis], b.math("SUBTRACT", lo[axis], 0.5))
        below = b.math("LESS_THAN", coords[axis], b.math("SUBTRACT", hi[axis], 0.5))
        term = b.math("MULTIPLY", above, below)
        inside = term if inside is None else b.math("MULTIPLY", inside, term)
    return inside


def _zone(b, zone, initial):
    sim_in = b.new("GeometryNodeSimulationInput")
    sim_out = b.new("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    sim_out.state_items.new("GEOMETRY", "Diag")
    b.link(initial, sim_in.inputs["Geometry"])

    gas = b.new(GAS_NODE)
    gas.name = zone.prefix + "gas"
    b.link(sim_in.outputs["Geometry"], gas.inputs["State"])
    b.link(sim_in.outputs["Delta Time"], gas.inputs["Delta Time"])
    for axis, count in zip("XYZ", zone.res):
        gas.inputs["Resolution " + axis].default_value = count
    gas.inputs["Origin"].default_value = zone.origin
    gas.inputs["Voxel Size"].default_value = zone.voxel
    for key, value in zone.params.items():
        socket = gas.inputs[key]
        socket.default_value = bool(value) if socket.type == "BOOLEAN" else value
    if "Time" in gas.inputs and "Time" not in zone.params:
        b.link(b.new("GeometryNodeInputSceneTime").outputs["Seconds"], gas.inputs["Time"])
    if zone.sources is not None:
        b.link(zone.sources(b, zone), gas.inputs["Sources"])
    if zone.colliders is not None:
        b.link(zone.colliders(b, zone), gas.inputs["Colliders"])
    if zone.layers is not None:
        b.link(zone.layers(b, zone), gas.inputs["Turbulence Layers"])
    b.link(gas.outputs["State"], sim_out.inputs["Geometry"])

    diag = b.new("GeometryNodePoints").outputs[0]
    outputs = ("Substeps", "Max Velocity", "Import Time", "Solve Time", "Export Time",
               "Pressure Iterations", "Pressure Residual", "Active Bricks")
    for name, socket in zip(DIAG_NAMES, outputs):
        store = b.new("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
        store.inputs["Name"].default_value = name
        b.link(diag, store.inputs["Geometry"])
        b.link(gas.outputs[socket], store.inputs["Value"])
        diag = store.outputs["Geometry"]
    b.link(diag, sim_out.inputs["Diag"])
    return sim_out.outputs["Geometry"], sim_out.outputs["Diag"]


def _readback(b, zone, state, diag):
    points = b.new("GeometryNodePoints")
    points.inputs["Count"].default_value = zone.cells if zone.readback else 1
    cloud = points.outputs[0]
    ix, iy, iz = b.index_xyz(zone.res)

    def store(name, kind, value):
        nonlocal cloud
        node = b.new("GeometryNodeStoreNamedAttribute", data_type=kind, domain="POINT")
        node.inputs["Name"].default_value = zone.prefix + name
        b.link(cloud, node.inputs["Geometry"])
        b.link(value, node.inputs["Value"])
        cloud = node.outputs["Geometry"]

    if zone.readback:
        for name in zone.grid_names + zone.extra_grids:
            vector = name == "velocity"
            get = b.new("GeometryNodeGetNamedGrid", data_type="VECTOR" if vector else "FLOAT")
            b.link(state, get.inputs["Volume"])
            get.inputs["Name"].default_value = name
            get.inputs["Remove"].default_value = False
            sample = b.new("GeometryNodeSampleGridIndex", data_type="VECTOR" if vector else "FLOAT")
            b.link(get.outputs["Grid"], sample.inputs["Grid"])
            for socket, value in zip("XYZ", (ix, iy, iz)):
                b.link(value, sample.inputs[socket])
            store(name, "FLOAT_VECTOR" if vector else "FLOAT", sample.outputs["Value"])

    for name in DIAG_NAMES:
        attr = b.new("GeometryNodeInputNamedAttribute", data_type="FLOAT")
        attr.inputs["Name"].default_value = name
        sample = b.new("GeometryNodeSampleIndex", data_type="FLOAT", domain="POINT")
        b.link(diag, sample.inputs["Geometry"])
        b.link(attr.outputs["Attribute"], sample.inputs["Value"])
        sample.inputs["Index"].default_value = 0
        store(name, "FLOAT", sample.outputs["Value"])
    return cloud


def build_object(name, zones):
    """Create an object whose Geometry Nodes modifier runs every zone and outputs state + readback."""
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    b = _Builder(tree)
    join = b.new("GeometryNodeJoinGeometry")
    for zone in zones:
        if zone.data_object is None:
            make_data_object(name + "_" + (zone.prefix or "data"), zone)
        state, diag = _zone(b, zone, _initial_state(b, zone))
        b.link(_readback(b, zone, state, diag), join.inputs[0])
        b.link(state, join.inputs[0])
    out = b.new("NodeGroupOutput")
    b.link(join.outputs[0], out.inputs[0])

    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new("Gas", "NODES")
    modifier.node_group = tree
    return obj


def _attribute(cloud, name, width):
    data = cloud.attributes[name].data
    array = np.empty(len(data) * width, dtype=np.float32)
    data.foreach_get("vector" if width == 3 else "value", array)
    return array.reshape(-1, 3) if width == 3 else array


def evaluate(obj, zones, frame=None):
    """Evaluate obj (optionally after jumping to frame) and return per-zone grids and diagnostics."""
    scene = bpy.context.scene
    if frame is not None:
        scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    geometry = obj.evaluated_get(depsgraph).evaluated_geometry()
    cloud = geometry.pointcloud
    results = []
    # Join Geometry places later links first, so zone point clouds appear in reverse order.
    offsets, start = {}, 0
    for zone in reversed(zones):
        count = zone.cells if zone.readback else 1
        offsets[zone.prefix] = (start, start + count)
        start += count
    if cloud is None or len(cloud.points) != start:
        raise RuntimeError("Unexpected readback point count")
    for zone in zones:
        lo, hi = offsets[zone.prefix]
        result = {}
        if zone.readback:
            for name in zone.grid_names + zone.extra_grids:
                result[name] = _attribute(cloud, zone.prefix + name, 3 if name == "velocity" else 1)[lo:hi].copy()
        for name in DIAG_NAMES:
            result[name] = float(_attribute(cloud, zone.prefix + name, 1)[lo])
        results.append(result)
    volume = geometry.volume
    grid_names = sorted(grid.name for grid in volume.grids) if volume else []
    return results, grid_names


def warnings(obj):
    """Messages reported by nodes during the last evaluation of obj's Geometry Nodes modifier."""
    return [(w.type, w.message) for w in obj.modifiers[0].node_warnings]


def run_frames(obj, zones, first, last):
    """Step the scene frame by frame and return the evaluation at the last frame."""
    result = None
    for frame in range(first, last + 1):
        result = evaluate(obj, zones, frame)
    return result


def as_grid(zone, flat):
    nx, ny, nz = zone.res
    return flat.reshape((nz, ny, nx) + flat.shape[1:])


def interior(zone, flat):
    return as_grid(zone, flat)[1:-1, 1:-1, 1:-1]


def turbulence_layers(b, layers):
    """Point cloud for the Turbulence Layers input: one point per dict of layer attributes."""
    points = b.new("GeometryNodePoints")
    points.inputs["Count"].default_value = len(layers)
    cloud = points.outputs[0]
    index = b.new("GeometryNodeInputIndex").outputs[0]
    names = sorted({key for layer in layers for key in layer})
    for name in names:
        kind = {"seed": "INT", "mask": "INT", "invert": "BOOLEAN"}.get(name, "FLOAT")
        # The value of point i: a chain of switches on the index.
        value = None
        for i, layer in enumerate(layers):
            constant = layer.get(name, 0)
            if value is None:
                value = constant
                continue
            compare = b.new("FunctionNodeCompare", data_type="INT", operation="EQUAL")
            b.link(index, compare.inputs["A"])
            compare.inputs["B"].default_value = i
            switch = b.new("GeometryNodeSwitch", input_type=kind)
            b.link(compare.outputs[0], switch.inputs["Switch"])
            b.set(switch.inputs["False"], value)
            switch.inputs["True"].default_value = constant
            value = switch.outputs[0]
        store = b.new("GeometryNodeStoreNamedAttribute", data_type=kind, domain="POINT")
        store.inputs["Name"].default_value = name
        b.link(cloud, store.inputs["Geometry"])
        b.set(store.inputs["Value"], value)
        cloud = store.outputs["Geometry"]
    return cloud
