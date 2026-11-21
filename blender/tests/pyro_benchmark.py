# SPDX-License-Identifier: GPL-2.0-or-later
"""Time the Pyro Emitter + Pyro Simulation scene frame by frame, for speed work.

    blender -b --factory-startup --python pyro_benchmark.py -- [--voxel 0.06] [--frames 40]
        [--emitter Explosion] [--sim Explosion] [--variant full|nodisplay] [--out FILE.json]
        [--set "Open Borders=1" --set "Pressure Quality=Medium" ...]

Builds a fresh scene from this tree's assets (a small sphere with a Pyro Emitter and a Pyro
Simulation, both on the chosen presets) and plays it from frame 1. Each frame it prints, as one
JSON line: the wall time of the frame, the gas node's own Import/Solve/Export times, substeps,
active bricks and the process memory; at the end a summary, including the device the solver ran
on (or why it fell back to the CPU). `nodisplay` skips Pyro Display, to separate the display's
cost from the solver's. --set changes a Pyro Simulation input after the preset is loaded
(booleans as 0/1, menus by name), for example the Speed panel's settings. Works on Windows and
Linux.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import statistics
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
sys.path.insert(0, ASSETS)
import smoke_assets as sa  # noqa: E402

_spec = importlib.util.spec_from_file_location("pyro_tools", os.path.join(ASSETS, "pyro_tools.py"))
pyro_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pyro_tools)

DIAG = ("Substeps", "Max Velocity", "Import Time", "Solve Time", "Export Time", "Active Bricks")


def memory_mb():
    """Resident memory of this process, in MB."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel32.GetCurrentProcess()),
                                                 ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 2**20
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def set_inputs(modifier, values):
    group = modifier.node_group
    for name, value in values.items():
        getattr(modifier.properties.inputs, sa.input_item(group, name).identifier).value = value
        if name == "Preset":
            pyro_tools.load_preset(modifier.id_data, modifier, force=True)


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_end = max(250, args.frames)
    emitter_group, sim_group = sa.build_assets(mark=False)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(0, 0, 0.5))
    obj = bpy.context.active_object
    emitter = obj.modifiers.new("Emitter", "NODES")
    emitter.node_group = emitter_group
    set_inputs(emitter, {"Preset": args.emitter})
    sim = obj.modifiers.new("Sim", "NODES")
    sim.node_group = sim_group
    set_inputs(sim, {"Voxel Size": args.voxel, "Domain Size": (5.0, 5.0, 6.0),
                     "Show Mesh": False, "Preset": args.sim})
    for assignment in args.set:
        name, _, text = assignment.partition("=")
        target = getattr(sim.properties.inputs, sa.input_item(sim_group, name.strip()).identifier)
        current = target.value
        if isinstance(current, bool):
            value = text.strip() not in ("0", "false", "False", "")
        elif isinstance(current, int):
            value = int(text)
        elif isinstance(current, float):
            value = float(text)
        else:
            value = text.strip()
        target.value = value
    tap_diagnostics(sim_group, args.variant)
    return scene, obj, sim


def tap_diagnostics(sim_group, variant):
    """Carry the gas node's diagnostic outputs out of the simulation zone on one point."""
    solver = bpy.data.node_groups["Pyro Solver"]
    solver.interface.new_socket("Diag", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gas = next(n for n in solver.nodes if n.bl_idname == sa.GAS_NODE)
    group_out = next(n for n in solver.nodes if n.bl_idname == "NodeGroupOutput")
    geometry = solver.nodes.new("GeometryNodePoints").outputs[0]
    for name in DIAG:
        store = solver.nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type = "FLOAT"
        store.inputs["Name"].default_value = "diag_" + name.replace(" ", "_")
        solver.links.new(geometry, store.inputs["Geometry"])
        solver.links.new(gas.outputs[name], store.inputs["Value"])
        geometry = store.outputs["Geometry"]
    solver.links.new(geometry, group_out.inputs["Diag"])

    tree = sim_group
    sim_out = next(n for n in tree.nodes if n.bl_idname == "GeometryNodeSimulationOutput")
    solver_node = next(n for n in tree.nodes
                       if n.bl_idname == "GeometryNodeGroup" and n.node_tree == solver)
    display = next(n for n in tree.nodes if n.bl_idname == "GeometryNodeGroup"
                   and n.node_tree.name.startswith("Pyro Display"))
    out = next(n for n in tree.nodes if n.bl_idname == "NodeGroupOutput")
    sim_out.state_items.new("GEOMETRY", "Diag")
    tree.links.new(solver_node.outputs["Diag"], sim_out.inputs["Diag"])
    join = tree.nodes.new("GeometryNodeJoinGeometry")
    if variant != "nodisplay":
        tree.links.new(display.outputs[0], join.inputs[0])
    tree.links.new(sim_out.outputs["Diag"], join.inputs[0])
    tree.links.new(join.outputs[0], out.inputs[0])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="pyro_benchmark.py")
    parser.add_argument("--voxel", type=float, default=0.06)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--emitter", default="Explosion")
    parser.add_argument("--sim", default="Explosion")
    parser.add_argument("--variant", choices=("full", "nodisplay"), default="full")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", action="append", default=[],
                        help='"Input Name=value" for the Pyro Simulation modifier')
    args = parser.parse_args(argv)

    scene, obj, sim = build_scene(args)
    rows, messages = [], set()
    start = time.perf_counter()
    for frame in range(1, args.frames + 1):
        t0 = time.perf_counter()
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        geometry = obj.evaluated_get(depsgraph).evaluated_geometry()
        row = {"frame": frame, "wall_ms": (time.perf_counter() - t0) * 1000.0,
               "memory_mb": memory_mb()}
        points = geometry.pointcloud
        if points is not None and len(points.points) >= 1:
            for name in DIAG:
                attribute = points.attributes.get("diag_" + name.replace(" ", "_"))
                if attribute is not None:
                    row[name] = attribute.data[0].value
        messages.update(w.message for w in sim.node_warnings)
        rows.append(row)
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}),
              flush=True)

    def mean(key, first=1):
        values = [r[key] for r in rows[first:] if key in r]
        return round(statistics.mean(values), 2) if values else None

    summary = {
        "voxel": args.voxel, "frames": args.frames, "variant": args.variant,
        "emitter": args.emitter, "sim": args.sim, "set": args.set, "total_s": round(time.perf_counter() - start, 2),
        "mean_wall_ms": mean("wall_ms"), "median_wall_ms":
            round(statistics.median(r["wall_ms"] for r in rows[1:]), 2) if len(rows) > 1 else None,
        "mean_solve_ms": mean("Solve Time"), "mean_import_ms": mean("Import Time"),
        "mean_export_ms": mean("Export Time"), "mean_substeps": mean("Substeps"),
        "last_active_bricks": rows[-1].get("Active Bricks"),
        "memory_growth_mb_per_frame": round((rows[-1]["memory_mb"] - rows[0]["memory_mb"])
                                            / max(1, len(rows) - 1), 2),
        "node_messages": sorted(messages),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"summary": summary, "rows": rows}, handle, indent=1)


main()
