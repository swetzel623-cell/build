# SPDX-License-Identifier: GPL-2.0-or-later
"""Build the Smoke node-group assets and install them into a Blender build.

    blender -b --factory-startup --python smoke_assets.py

writes `datafiles/assets/nodes/smoke_simulation_assets.blend` next to the running Blender and
adds its catalogs to `datafiles/assets/blender_assets.cats.txt`, so that:

- Add Modifier > Physics > Smoke Simulation turns any mesh into a smoke emitter, and
- Add > Simulation offers Smoke Source and Smoke Collider for building a setup by hand around
  the Gas Simulation Step node.

`build_assets()` can also be imported to create the groups in the current file (the tests do).
"""

from __future__ import annotations

import os
import sys

import bpy

CATALOG_SIMULATION = ("689f4607-3a1d-4e2a-a49b-8166dad896ca", "Geometry Nodes/Simulation")
CATALOG_PHYSICS = ("4c0f1d1e-6b0a-4a8e-9d43-5d1f3a6b7c21", "Geometry Nodes/Physics")
ASSET_FILE = "smoke_simulation_assets.blend"
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

    def vmath(self, operation, a, b=None, scale=None):
        node = self.new("ShaderNodeVectorMath", operation=operation)
        self.set(node.inputs[0], a)
        if b is not None:
            self.set(node.inputs[1], b)
        if scale is not None:
            self.set(node.inputs["Scale"], scale)
        return node.outputs[0]


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


def build_source():
    tree = new_group(
        "Smoke Source",
        "Turn a mesh into smoke sources for the Gas Simulation Step node's Sources input")
    socket(tree, "Mesh", "NodeSocketGeometry")
    socket(tree, "Density", "NodeSocketFloat", default=1.0, minimum=0.0,
           description="Smoke density emitted inside the mesh")
    socket(tree, "Temperature", "NodeSocketFloat", default=1.0,
           description="Heat emitted inside the mesh; with heat buoyancy it makes the smoke rise")
    socket(tree, "Set Velocity", "NodeSocketBool", default=False,
           description="Set the velocity of the air where the mesh emits")
    socket(tree, "Velocity", "NodeSocketVector", default=(0.0, 0.0, 0.0), subtype="VELOCITY",
           description="Velocity in world units per second, evaluated on the mesh")
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-4,
           description="Use the simulation's voxel size")
    socket(tree, "Sources", "NodeSocketGeometry", in_out="OUTPUT")
    b = Builder(tree)
    group_in = b.new("NodeGroupInput")
    group_out = b.new("NodeGroupOutput")
    mesh = group_in.outputs["Mesh"]

    def density_grid(value):
        node = b.new("GeometryNodeMeshToDensityGrid")
        b.link(mesh, node.inputs["Mesh"])
        b.link(value, node.inputs["Density"])
        b.link(group_in.outputs["Voxel Size"], node.inputs["Voxel Size"])
        return node.outputs["Density Grid"]

    density = density_grid(group_in.outputs["Density"])
    volume = store_grid(b, None, "density", density)
    volume = store_grid(b, volume, "temperature", density_grid(group_in.outputs["Temperature"]))
    with_velocity = store_grid(
        b, volume, "velocity", velocity_grid(b, mesh, group_in.outputs["Velocity"], density),
        kind="VECTOR_FLOAT")
    switch = b.new("GeometryNodeSwitch", input_type="GEOMETRY")
    b.link(group_in.outputs["Set Velocity"], switch.inputs["Switch"])
    b.link(volume, switch.inputs["False"])
    b.link(with_velocity, switch.inputs["True"])
    b.link(switch.outputs[0], group_out.inputs["Sources"])
    return tree


def build_collider():
    tree = new_group(
        "Smoke Collider",
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
    return tree


def build_material():
    material = bpy.data.materials.get("Smoke Simulation")
    if material is None:
        material = bpy.data.materials.new("Smoke Simulation")
    if material.node_tree is None:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 5.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(volume.outputs[0], output.inputs["Volume"])
    return material


def build_modifier(source, collider, material):
    tree = new_group(
        "Smoke Simulation",
        "Emit smoke from this mesh and simulate it on the GPU. The domain is placed around the "
        "mesh on the first frame of the simulation")
    tree.is_modifier = True
    socket(tree, "Geometry", "NodeSocketGeometry")
    socket(tree, "Geometry", "NodeSocketGeometry", in_out="OUTPUT")
    panel = tree.interface.new_panel("Emission")
    socket(tree, "Density", "NodeSocketFloat", default=1.0, minimum=0.0, parent=panel,
           description="Smoke density emitted inside the mesh")
    socket(tree, "Temperature", "NodeSocketFloat", default=1.0, parent=panel,
           description="Heat emitted inside the mesh; with heat buoyancy it makes the smoke rise")
    socket(tree, "Add Emission", "NodeSocketBool", default=False, parent=panel,
           description="Add density and temperature per second instead of raising them to the "
                       "emitted values")
    socket(tree, "Set Velocity", "NodeSocketBool", default=False, parent=panel,
           description="Push the air where the mesh emits")
    socket(tree, "Velocity", "NodeSocketVector", default=(0.0, 0.0, 0.0), subtype="VELOCITY",
           parent=panel, description="World units per second")
    panel = tree.interface.new_panel("Domain")
    socket(tree, "Voxel Size", "NodeSocketFloat", default=0.05, minimum=1.0e-3, parent=panel,
           description="Size of a voxel in world units. Halving it gives eight times the voxels")
    socket(tree, "Domain Size", "NodeSocketVector", default=(2.0, 2.0, 4.0), subtype="XYZ",
           parent=panel,
           description="Size of the simulated box in world units, centred on the mesh and "
                       "starting a little below it. Only the parts with smoke cost anything")
    socket(tree, "Colliders", "NodeSocketCollection", parent=panel,
           description="Objects the smoke flows around. They may be animated")
    panel = tree.interface.new_panel("Forces")
    socket(tree, "Heat Buoyancy", "NodeSocketFloat", default=0.5, parent=panel,
           description="How strongly heat makes smoke rise against gravity")
    socket(tree, "Density Buoyancy", "NodeSocketFloat", default=0.0, parent=panel,
           description="Positive values make dense smoke rise, negative values make it sink")
    socket(tree, "Gravity", "NodeSocketVector", default=(0.0, 0.0, -9.81),
           subtype="ACCELERATION", parent=panel)
    panel = tree.interface.new_panel("Turbulence")
    socket(tree, "Turbulence", "NodeSocketFloat", default=0.0, minimum=0.0, parent=panel,
           subtype="ACCELERATION",
           description="Strength of swirling noise; try about half of gravity times the heat "
                       "buoyancy. 0 turns it off")
    socket(tree, "Turbulence Scale", "NodeSocketFloat", default=0.25, minimum=1.0e-3,
           subtype="DISTANCE", parent=panel, description="Size of the swirls")
    socket(tree, "Turbulence Speed", "NodeSocketFloat", default=1.0, minimum=0.0,
           parent=panel, description="How many times per second the pattern changes")
    socket(tree, "Turbulence Seed", "NodeSocketInt", default=0, minimum=0, parent=panel)
    socket(tree, "Turbulence Mask", "NodeSocketFloat", default=0.1, minimum=0.0, parent=panel,
           description="Density at which turbulence reaches full strength; 0 stirs all air")
    panel = tree.interface.new_panel("Solver")
    socket(tree, "Sparse GPU", "NodeSocketBool", default=True, parent=panel,
           description="Simulate on the GPU. Off runs Mantaflow on the CPU, which does not "
                       "support emission, colliders or turbulence yet, so nothing appears")
    socket(tree, "CFL", "NodeSocketFloat", default=6.0, minimum=0.1, maximum=50.0, parent=panel,
           description="Most voxels anything may travel in one step")
    socket(tree, "Max Substeps", "NodeSocketInt", default=64, minimum=1, maximum=1024,
           parent=panel)
    panel = tree.interface.new_panel("Display")
    socket(tree, "Show Emitter", "NodeSocketBool", default=True, parent=panel,
           description="Keep the emitting mesh visible next to the smoke")

    b = Builder(tree)
    group_in = b.new("NodeGroupInput", location=(-1400, 0))
    group_out = b.new("NodeGroupOutput", location=(1400, 0))
    inp = group_in.outputs

    # Everything is simulated in world space; the result is moved back into this object's space.
    self_info = b.new("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    b.link(b.new("GeometryNodeSelfObject").outputs[0], self_info.inputs["Object"])
    world = self_info.outputs["Transform"]
    to_world = b.new("GeometryNodeTransform")
    to_world.inputs["Mode"].default_value = "Matrix"
    b.link(inp["Geometry"], to_world.inputs["Geometry"])
    b.link(world, to_world.inputs["Transform"])
    emitter = to_world.outputs["Geometry"]

    colliders_info = b.new("GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    b.link(inp["Colliders"], colliders_info.inputs["Collection"])
    realize = b.new("GeometryNodeRealizeInstances")
    b.link(colliders_info.outputs[0], realize.inputs["Geometry"])
    colliders = realize.outputs[0]

    # The domain: centred on the emitter, its floor a tenth of its height below the emitter.
    bounds = b.new("GeometryNodeBoundBox")
    b.link(emitter, bounds.inputs["Geometry"])
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
    resolution = []
    for axis in range(3):
        cells = b.math("CEIL", b.math("DIVIDE", size_xyz.outputs[axis], inp["Voxel Size"]))
        resolution.append(b.math("MINIMUM", b.math("MAXIMUM", cells, 4.0), 1024.0))

    # The zone keeps the domain's corner from its first frame, and the colliders of the
    # previous frame to find their velocity.
    sim_in = b.new("GeometryNodeSimulationInput", location=(-200, 0))
    sim_out = b.new("GeometryNodeSimulationOutput", location=(800, 0))
    sim_in.pair_with_output(sim_out)
    sim_out.state_items.new("VECTOR", "Domain Min")
    sim_out.state_items.new("GEOMETRY", "Previous Colliders")
    b.link(domain_min.outputs[0], sim_in.inputs["Domain Min"])
    b.link(colliders, sim_in.inputs["Previous Colliders"])
    delta = sim_in.outputs["Delta Time"]

    previous = b.new("GeometryNodeSampleIndex", data_type="FLOAT_VECTOR", domain="POINT")
    b.link(sim_in.outputs["Previous Colliders"], previous.inputs["Geometry"])
    b.link(b.new("GeometryNodeInputPosition").outputs[0], previous.inputs["Value"])
    b.link(b.new("GeometryNodeInputIndex").outputs[0], previous.inputs["Index"])
    moved = b.vmath("SUBTRACT", b.new("GeometryNodeInputPosition").outputs[0],
                    previous.outputs[0])
    collider_velocity = b.vmath("SCALE", moved, scale=b.math("DIVIDE", 1.0, delta))

    source_node = b.new("GeometryNodeGroup", node_tree=source)
    b.link(emitter, source_node.inputs["Mesh"])
    for name in ("Density", "Temperature", "Set Velocity", "Velocity", "Voxel Size"):
        b.link(inp[name], source_node.inputs[name])
    collider_node = b.new("GeometryNodeGroup", node_tree=collider)
    b.link(colliders, collider_node.inputs["Mesh"])
    b.link(collider_velocity, collider_node.inputs["Velocity"])
    b.link(inp["Voxel Size"], collider_node.inputs["Voxel Size"])

    gas = b.new(GAS_NODE, location=(300, 0))
    b.link(sim_in.outputs["Geometry"], gas.inputs["State"])
    b.link(delta, gas.inputs["Delta Time"])
    for axis, value in zip("XYZ", resolution):
        b.link(value, gas.inputs["Resolution " + axis])
    b.link(sim_in.outputs["Domain Min"], gas.inputs["Origin"])
    b.link(source_node.outputs["Sources"], gas.inputs["Sources"])
    b.link(collider_node.outputs["Colliders"], gas.inputs["Colliders"])
    b.link(inp["Add Emission"], gas.inputs["Add Sources"])
    b.link(b.new("GeometryNodeInputSceneTime").outputs["Seconds"], gas.inputs["Time"])
    for name in ("Voxel Size", "Gravity", "Heat Buoyancy", "Density Buoyancy", "CFL",
                 "Max Substeps", "Sparse GPU", "Turbulence", "Turbulence Scale",
                 "Turbulence Speed", "Turbulence Seed", "Turbulence Mask"):
        b.link(inp[name], gas.inputs[name])

    b.link(gas.outputs["State"], sim_out.inputs["Geometry"])
    b.link(sim_in.outputs["Domain Min"], sim_out.inputs["Domain Min"])
    b.link(colliders, sim_out.inputs["Previous Colliders"])

    shaded = b.new("GeometryNodeSetMaterial")
    shaded.inputs["Material"].default_value = material
    b.link(sim_out.outputs["Geometry"], shaded.inputs["Geometry"])
    to_local = b.new("GeometryNodeTransform")
    to_local.inputs["Mode"].default_value = "Matrix"
    invert = b.new("FunctionNodeInvertMatrix")
    b.link(world, invert.inputs["Matrix"])
    b.link(shaded.outputs[0], to_local.inputs["Geometry"])
    b.link(invert.outputs["Matrix"], to_local.inputs["Transform"])

    show = b.new("GeometryNodeSwitch", input_type="GEOMETRY")
    b.link(inp["Show Emitter"], show.inputs["Switch"])
    b.link(inp["Geometry"], show.inputs["True"])
    join = b.new("GeometryNodeJoinGeometry", location=(1200, 0))
    b.link(show.outputs[0], join.inputs[0])
    b.link(to_local.outputs[0], join.inputs[0])
    b.link(join.outputs[0], group_out.inputs["Geometry"])
    return tree


def build_assets(mark=True):
    """Create the groups and material in the current file; returns the modifier group."""
    source = build_source()
    collider = build_collider()
    material = build_material()
    modifier = build_modifier(source, collider, material)
    if mark:
        for tree, catalog in ((source, CATALOG_SIMULATION), (collider, CATALOG_SIMULATION),
                              (modifier, CATALOG_PHYSICS)):
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
    groups = {bpy.data.node_groups[name]
              for name in ("Smoke Source", "Smoke Collider", "Smoke Simulation")}
    bpy.data.libraries.write(path, groups, fake_user=True)

    catalogs = os.path.join(assets, "blender_assets.cats.txt")
    lines = ["VERSION 1", ""]
    if os.path.exists(catalogs):
        with open(catalogs, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    for uuid, path_name in (CATALOG_SIMULATION, CATALOG_PHYSICS):
        entry = f"{uuid}:{path_name}:{path_name.replace('/', '-')}"
        if entry in lines:
            continue
        # Replace an older path of the same catalog.
        lines = [line for line in lines if not line.startswith(uuid + ":")]
        lines.append(entry)
    with open(catalogs, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"Installed the smoke assets in {path}")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    datafiles = argv[0] if argv else bpy.utils.system_resource("DATAFILES")
    install(datafiles)


if __name__ == "__main__":
    main()
