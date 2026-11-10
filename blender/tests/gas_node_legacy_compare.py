# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare the Gas Simulation Step node with the legacy Fluid modifier gas solver.

The legacy domain is configured to the feature subset the node supports (closed walls, no
vorticity, no dissolve, no noise). Adaptive time steps stay enabled, as by default: with them
disabled the pinned legacy loop takes two full solver steps per frame, because the time step it
passes to Mantaflow is rounded to six decimals and falls just short of the frame length. The
script counts the legacy solver steps per frame and requires exactly one, so both solvers take
the same single step. A flow object emits once, the legacy
solver runs in Replay mode, and its state at a start frame is handed to the node as initial state.
Both solvers then advance the same number of frames and the grids are compared.

    blender -b --factory-startup --python gas_node_legacy_compare.py -- [--res 32] [--json FILE]

This reports differences; it does not assert equivalence. Known structural differences are the
pressure preconditioner (MIC here, static multigrid in the legacy script) and the pressure initial
guess. The node's first frame also receives the legacy velocity as an authored cell-centered
field constructed to resample onto the legacy staggered field (see centered_preimage).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gas_node_testlib as lib  # noqa: E402
from gas_node_testlib import Zone  # noqa: E402

MANTA_TIME_PER_SECOND = 2.5  # Legacy script: dt = 0.1 at 25 fps.


def run_legacy(res, frames, work):
    scene = lib.reset_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    domain = bpy.context.object
    settings = domain.modifiers.new("Fluid", "FLUID")
    settings.fluid_type = "DOMAIN"
    ds = settings.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = res
    ds.use_adaptive_domain = False
    ds.use_adaptive_timesteps = True
    ds.use_noise = False
    ds.use_dissolve_smoke = False
    ds.vorticity = 0.0
    ds.time_scale = 1.0
    ds.alpha = 1.0
    ds.beta = 1.0
    for side in ("front", "back", "left", "right", "top", "bottom"):
        setattr(ds, "use_collision_border_" + side, True)
    ds.cache_type = "REPLAY"
    ds.cache_directory = os.path.join(work, "legacy_cache")

    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.3, location=(0.1, 0.0, -0.45), subdivisions=3)
    flow = bpy.context.object
    fm = flow.modifiers.new("Fluid", "FLUID")
    fm.fluid_type = "FLOW"
    fs = fm.flow_settings
    fs.flow_type = "SMOKE"
    fs.flow_behavior = "GEOMETRY"
    fs.density = 1.0
    fs.temperature = 1.0
    fs.use_initial_velocity = False

    states = {}
    steps = {}
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
        if frame == 1:
            count_solver_steps(steps)
        # The solver lives on the evaluated domain object.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ds = domain.evaluated_get(depsgraph).modifiers[0].domain_settings
        dims = tuple(ds.domain_resolution)
        count = dims[0] * dims[1] * dims[2]
        density = np.array(ds.density_grid, dtype=np.float32)
        if density.size != count:
            continue
        states[frame] = {
            "density": density,
            # The Python API halves the solver's heat grid ("-2..2 to -1..1").
            "temperature": np.array(ds.heat_grid, dtype=np.float32) * 2.0,
            "mac": np.array(ds.velocity_grid, dtype=np.float32).reshape(count, 3),
        }
    info = {
        "res": tuple(ds.domain_resolution),
        "voxel": float(ds.cell_size[0]),
        "domain_size": float(max(domain.dimensions)),
        "alpha": float(ds.alpha),
        "beta": float(ds.beta),
        "gravity": tuple(scene.gravity),
        "fps": scene.render.fps,
        "legacy_steps_per_frame": sorted(set(steps.values())),
        "legacy_dt": steps.get("dt"),
        "legacy_frame_length": steps.get("frame_length"),
    }
    steps.pop("dt", None)
    steps.pop("frame_length", None)
    info["legacy_steps_per_frame"] = sorted(set(steps.values()))
    return states, info


def count_solver_steps(steps):
    """Wrap the legacy solver's Python step functions to count calls per scene frame.

    The legacy scripts run in a private module, which is only reachable through the garbage
    collector. This is test instrumentation, not a supported interface.
    """
    import gc

    for namespace in gc.get_objects():
        if not isinstance(namespace, dict):
            continue
        names = [k for k in namespace if isinstance(k, str) and k.startswith("smoke_step_")
                 and "noise" not in k]
        for name in names:
            original = namespace[name]
            solver = namespace.get("s" + name.rsplit("_", 1)[-1])

            def counted(original=original, solver=solver):
                frame = bpy.context.scene.frame_current
                steps[frame] = steps.get(frame, 0) + 1
                if solver is not None:
                    steps["dt"] = float(solver.timestep)
                    steps["frame_length"] = float(solver.frameLength)
                original()

            namespace[name] = counted


def legacy_mac_world(mac, voxel):
    """Legacy staggered velocity (cells per Manta time unit) in world units per second."""
    return (mac.astype(np.float64) * MANTA_TIME_PER_SECOND * voxel).astype(np.float32)


def centered_preimage(mac_world, res):
    """Cell-centered field that Mantaflow's resampleVec3ToMac maps back onto `mac_world`.

    The node can only be handed an authored (cell-centered) velocity, which it averages onto
    faces: face[i] = (center[i - 1] + center[i]) / 2. Solving that recurrence from the wall
    (center[0] = 0) lets the node start from the legacy staggered field itself, so the comparison
    measures the solvers and not a lossy handover.
    """
    nx, ny, nz = res
    face = mac_world.reshape(nz, ny, nx, 3).astype(np.float64)
    center = np.zeros_like(face)
    for i in range(1, nx):
        center[:, :, i, 0] = 2.0 * face[:, :, i, 0] - center[:, :, i - 1, 0]
    for j in range(1, ny):
        center[:, j, :, 1] = 2.0 * face[:, j, :, 1] - center[:, j - 1, :, 1]
    for k in range(1, nz):
        center[k, :, :, 2] = 2.0 * face[k, :, :, 2] - center[k - 1, :, :, 2]
    return center.reshape(-1, 3).astype(np.float32)


def run_node(start, info, frames, **overrides):
    scene = lib.reset_scene(fps=info["fps"])
    if info.get("legacy_dt") and info.get("legacy_frame_length"):
        # Mantaflow's adaptive scheduler pads the final step of a frame by 1e-4 time units.
        scene.render.fps_base = info["legacy_dt"] / info["legacy_frame_length"]
    res, voxel = info["res"], info["voxel"]
    params = {
        "Gravity": info["gravity"],
        "Density Buoyancy": info["alpha"] / info["domain_size"],
        "Heat Buoyancy": info["beta"] / info["domain_size"],
        "CFL": 100.0,  # One step per frame, as in the legacy run without adaptive time steps.
        "Max Substeps": 1,
    }
    params.update(overrides)
    zone = Zone(res, origin=(-1.0, -1.0, -1.0), voxel=voxel, **params)
    lib.make_data_object(
        "legacy_state", zone,
        density=start["density"], temperature=start["temperature"],
        velocity=centered_preimage(legacy_mac_world(start["mac"], voxel), res),
    )
    obj = lib.build_object("node", [zone])
    out = {}
    for frame in range(1, frames + 2):
        (result,), _ = lib.evaluate(obj, [zone], frame)
        out[frame - 1] = result
    problems = [m for kind, m in lib.warnings(obj) if kind == "ERROR"]
    if problems:
        raise RuntimeError(f"node reported errors: {problems}")
    return out


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def com_z(density, res):
    grid = density.reshape(res[2], res[1], res[0]).astype(np.float64)
    return float((grid * np.arange(res[2])[:, None, None]).sum() / max(grid.sum(), 1.0e-30))


def compare(node, legacy, info):
    res, voxel = info["res"], info["voxel"]
    legacy_velocity = legacy_mac_world(legacy["mac"], voxel)
    kinetic = lambda v: float((v.astype(np.float64) ** 2).sum())  # noqa: E731
    return {
        "density_rel_l2": rel_l2(node["density"], legacy["density"]),
        "density_max_abs": float(np.abs(node["density"] - legacy["density"]).max()),
        "temperature_rel_l2": rel_l2(node["temperature"], legacy["temperature"]),
        "velocity_rel_l2": rel_l2(node["velocity"], legacy_velocity),
        "kinetic_energy_ratio": kinetic(node["velocity"]) / max(kinetic(legacy_velocity), 1.0e-30),
        "mass_ratio": float(node["density"].sum() / max(legacy["density"].sum(), 1.0e-30)),
        "com_z_node": com_z(node["density"], res),
        "com_z_legacy": com_z(legacy["density"], res),
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=32)
    parser.add_argument("--start", type=int, default=6)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--work", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    work = args.work or tempfile.mkdtemp(prefix="gas_legacy_")

    last = args.start + max(args.horizons)
    legacy, info = run_legacy(args.res, last, work)
    if args.start not in legacy or last not in legacy:
        raise RuntimeError(f"legacy solver produced frames {sorted(legacy)}")
    print(f"legacy domain {info}")
    if info["legacy_steps_per_frame"] != [1]:
        raise RuntimeError(
            f"legacy solver took {info['legacy_steps_per_frame']} steps per frame; the comparison "
            "requires exactly one"
        )

    node = run_node(legacy[args.start], info, max(args.horizons))
    report = {"info": info, "start_frame": args.start, "horizons": {}}
    for horizon in args.horizons:
        row = compare(node[horizon], legacy[args.start + horizon], info)
        row["legacy_mass_change"] = float(
            legacy[args.start + horizon]["density"].sum() / legacy[args.start]["density"].sum()
        )
        report["horizons"][horizon] = row
        print(f"+{horizon:3d} frames: " + ", ".join(f"{k} {v:.4g}" for k, v in row.items()))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
