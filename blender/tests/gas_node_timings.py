# SPDX-License-Identifier: GPL-2.0-or-later
"""Record the Gas Simulation Step node's import, solve, and export timings in a Simulation Zone.

    blender -b --factory-startup --python gas_node_timings.py -- [--res 32 64 96] [--json FILE]

The scene is a buoyant smoke blob in a closed cubic domain. For each resolution the script reports
the node's own diagnostics per frame and the wall-clock time of the whole frame evaluation, and
optionally the wall-clock frame time of the legacy Fluid modifier at the same resolution. The
legacy number is context only: that solver uses a different pressure preconditioner and does work
(emission, caching) the node does not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gas_node_testlib as lib  # noqa: E402
from gas_node_testlib import Zone  # noqa: E402


def node_run(res, frames):
    lib.reset_scene()
    zone = Zone((res, res, res), origin=(-1.0, -1.0, -1.0), voxel=2.0 / res, readback=False)
    c, r = res / 2, res / 8
    field = lib.dense(
        zone, lambda x, y, z: ((x - c) ** 2 + (y - c) ** 2 + (z - res / 4) ** 2 <= r * r) * 1.0
    )
    lib.make_data_object("data", zone, density=field, temperature=field)
    obj = lib.build_object("timing", [zone])
    lib.evaluate(obj, [zone], 1)
    rows = []
    for frame in range(2, frames + 2):
        begin = time.perf_counter()
        (result,), _ = lib.evaluate(obj, [zone], frame)
        result["frame_wall_ms"] = (time.perf_counter() - begin) * 1000.0
        rows.append(result)
    problems = [m for kind, m in lib.warnings(obj) if kind == "ERROR"]
    if problems:
        raise RuntimeError(f"node reported errors: {problems}")
    return rows


def legacy_run(res, frames, work):
    scene = lib.reset_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    domain = bpy.context.object
    ds = domain.modifiers.new("Fluid", "FLUID")
    ds.fluid_type = "DOMAIN"
    ds = ds.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = res
    ds.use_adaptive_timesteps = True
    ds.vorticity = 0.0
    ds.use_dissolve_smoke = False
    ds.cache_type = "REPLAY"
    ds.cache_directory = os.path.join(work, f"legacy_{res}")
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.25, location=(0.0, 0.0, -0.5))
    fs = bpy.context.object.modifiers.new("Fluid", "FLUID")
    fs.fluid_type = "FLOW"
    fs = fs.flow_settings
    fs.flow_type = "SMOKE"
    fs.flow_behavior = "GEOMETRY"
    fs.temperature = 1.0
    scene.frame_set(1)
    bpy.context.evaluated_depsgraph_get()
    walls = []
    for frame in range(2, frames + 2):
        begin = time.perf_counter()
        scene.frame_set(frame)
        walls.append((time.perf_counter() - begin) * 1000.0)
    return walls


def summarize(rows, key):
    values = np.array([row[key] for row in rows], dtype=np.float64)
    return {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, nargs="+", default=[32, 64, 96])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    work = tempfile.mkdtemp(prefix="gas_timing_")

    report = {"threads": bpy.context.scene.render.threads, "frames": args.frames, "runs": []}
    for res in args.res:
        rows = node_run(res, args.frames)
        entry = {"resolution": res, "cells": res**3}
        for key in ("import_ms", "solve_ms", "export_ms", "frame_wall_ms", "substeps"):
            entry[key] = summarize(rows, key)
        if args.legacy:
            walls = legacy_run(res, args.frames, work)
            entry["legacy_frame_wall_ms"] = summarize([{"w": w} for w in walls], "w")
        report["runs"].append(entry)
        print(f"{res}^3: " + ", ".join(
            f"{key} {value['median']:.1f}" for key, value in entry.items() if isinstance(value, dict)
        ), flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
