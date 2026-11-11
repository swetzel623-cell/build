# SPDX-License-Identifier: GPL-2.0-or-later
"""Record the Gas Simulation Step node's import, solve, and export timings in a Simulation Zone.

    blender -b --factory-startup --python gas_node_timings.py -- [--res 32 64 96] [--json FILE]

The scene is a buoyant smoke blob in a closed cubic domain. For each resolution the script reports
the node's own diagnostics per frame and the wall-clock time of the whole frame evaluation.

With `--legacy` the legacy Fluid modifier runs first and the node starts from the legacy state
after its first frame, with equivalent parameters and the same CFL number, so both integrate the
same flow. The legacy frame time still includes work the node does not do (flow-object
processing, cache writes, shadow grid), and the node frame time includes work the legacy solver
does not do (OpenVDB conversion, solver reconstruction).
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
import gas_node_legacy_compare as legacy  # noqa: E402
from gas_node_testlib import Zone  # noqa: E402


def node_run(res, frames, params):
    lib.reset_scene()
    zone = Zone((res, res, res), origin=(-1.0, -1.0, -1.0), voxel=2.0 / res, readback=False,
                **params)
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


def matched_run(res, frames, work, params):
    """Time the legacy solver, then the node on the same flow.

    The legacy solver emits once and runs `frames` frames after its first. The node starts from
    the legacy state after that first frame with the equivalent parameters and the legacy CFL
    number, so both integrate the same problem and their substep counts are comparable.
    """
    walls = {}
    states, info = legacy.run_legacy(res, frames + 1, os.path.join(work, str(res)), keep={1},
                                     walls=walls)
    legacy_walls = [walls[frame] for frame in range(2, frames + 2)]
    legacy_steps = [info["legacy_steps_by_frame"].get(frame, 0) for frame in range(2, frames + 2)]
    overrides = {"CFL": info["cfl"], "Max Substeps": 64, "readback": False}
    overrides.update(params)
    node = legacy.run_node(states[1], info, frames, match_step_padding=False, **overrides)
    rows = [node[frame] for frame in range(1, frames + 1)]
    return rows, legacy_walls, legacy_steps


def summarize(rows, key):
    values = np.array([row[key] for row in rows], dtype=np.float64)
    return {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, nargs="+", default=[32, 64, 96])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="INPUT=VALUE",
                        help="Override a float node input, e.g. 'Pressure Accuracy=1.0'")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    work = tempfile.mkdtemp(prefix="gas_timing_")

    params = {name: float(value) for name, value in (item.split("=", 1) for item in args.set)}
    report = {"threads": bpy.context.scene.render.threads, "frames": args.frames,
              "overrides": params, "runs": []}
    for res in args.res:
        if args.legacy:
            rows, legacy_walls, legacy_steps = matched_run(res, args.frames, work, params)
        else:
            rows = node_run(res, args.frames, params)
        entry = {"resolution": res, "cells": res**3}
        for key in ("import_ms", "solve_ms", "export_ms", "frame_wall_ms", "substeps",
                    "pressure_iterations"):
            entry[key] = summarize(rows, key)
        entry["total_frame_wall_ms"] = float(sum(row["frame_wall_ms"] for row in rows))
        entry["total_substeps"] = float(sum(row["substeps"] for row in rows))
        if args.legacy:
            entry["legacy_frame_wall_ms"] = summarize([{"w": w} for w in legacy_walls], "w")
            entry["legacy_substeps"] = summarize([{"n": n} for n in legacy_steps], "n")
            entry["legacy_total_frame_wall_ms"] = float(sum(legacy_walls))
            entry["legacy_total_substeps"] = float(sum(legacy_steps))
        report["runs"].append(entry)
        print(f"{res}^3: " + ", ".join(
            f"{key} {value['median'] if isinstance(value, dict) else value:.1f}"
            for key, value in entry.items() if key not in ("resolution", "cells")
        ), flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
