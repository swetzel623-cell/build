# SPDX-License-Identifier: GPL-2.0-or-later
"""Runtime tests for the Gas Simulation Step node inside a real Simulation Zone.

Run with a Blender binary built from the patched pinned source:

    blender -b --factory-startup --python gas_node_runtime_tests.py -- [--work DIR] [-k NAME]

The process exits non-zero if any test fails.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gas_node_testlib as lib  # noqa: E402
from gas_node_testlib import Zone  # noqa: E402

RES = (24, 24, 32)
DT = 1.0 / 30.0


def blob(zone, center, radius):
    cx, cy, cz = center
    return lib.dense(
        zone, lambda x, y, z: ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius**2) * 1.0
    )


def smoke_zone(prefix="", res=RES, center=None, **params):
    zone = Zone(res, prefix=prefix, **params)
    center = center or (res[0] / 2, res[1] / 2, res[2] / 4)
    field = blob(zone, center, 4.0)
    lib.make_data_object((prefix or "smoke") + "_data", zone, density=field, temperature=field)
    return zone


def center_of_mass_z(zone, density):
    grid = lib.as_grid(zone, density).astype(np.float64)
    z = np.arange(zone.res[2], dtype=np.float64)[:, None, None]
    return float((grid * z).sum() / grid.sum())


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def errors(obj):
    return [message for kind, message in lib.warnings(obj) if kind == "ERROR"]


# --------------------------------------------------------------------------------------------


def test_node_is_registered(_work):
    lib.reset_scene()
    tree = bpy.data.node_groups.new("probe", "GeometryNodeTree")
    node = tree.nodes.new(lib.GAS_NODE)
    check(node.bl_label == "Gas Simulation Step", "unexpected node label")
    check([s.name for s in node.outputs][0] == "State", "State must be the first output")


def test_zero_time_start_frame(_work):
    lib.reset_scene()
    zone = smoke_zone()
    obj = lib.build_object("zero", [zone])
    (result,), names = lib.evaluate(obj, [zone], 1)
    initial = blob(zone, (12, 12, 8), 4.0)
    check(np.array_equal(result["density"], initial), "start frame changed density")
    check(np.array_equal(result["temperature"], initial), "start frame changed temperature")
    check(not result["velocity"].any(), "start frame created velocity")
    check(result["substeps"] == 0 and result["solve_ms"] == 0.0, "zero time ran the solver")
    check(names == ["density", "temperature", "velocity"], "zero time must return the input volume")
    check(not errors(obj), "zero time reported an error")


def test_smoke_advances_and_rises(_work):
    lib.reset_scene()
    zone = smoke_zone()
    obj = lib.build_object("smoke", [zone])
    (start,), _ = lib.evaluate(obj, [zone], 1)
    (end,), names = lib.run_frames(obj, [zone], 2, 25)
    check(names == sorted(lib.GRID_NAMES), f"missing state grids: {names}")
    check(not errors(obj), f"unexpected node errors: {errors(obj)}")
    for name in lib.GRID_NAMES:
        check(np.isfinite(end[name]).all(), f"{name} is not finite")
    rise = center_of_mass_z(zone, end["density"]) - center_of_mass_z(zone, start["density"])
    check(rise > 1.0, f"smoke rose only {rise:.3f} cells against -Z gravity")
    check(end["substeps"] >= 1 and end["max_velocity"] > 0.0, "diagnostics were not reported")
    check(np.abs(end["pressure"]).max() > 0.0, "pressure state was not retained")
    # MacCormack advection is not conservative (the legacy solver drifts the same way; see
    # test_matches_legacy_solver), so this only guards against gross loss or gain.
    mass_ratio = float(end["density"].sum() / start["density"].sum())
    check(0.6 < mass_ratio < 1.4, f"density mass drifted by a factor of {mass_ratio:.3f}")
    # Closed walls. Velocity is staggered: component c of a cell lives on its lower face along c,
    # so the Z faces of layers 0 and 1 bound the bottom wall and the top wall layer's face is the
    # upper boundary of the fluid.
    grid = lib.as_grid(zone, end["velocity"])
    wall_normal = max(np.abs(grid[k, :, :, 2]).max() for k in (0, 1, zone.res[2] - 1))
    check(wall_normal == 0.0, f"velocity {wall_normal} through a closed wall")
    interior_faces = grid[2:-1, 1:-1, 1:-1, 2]
    divergence = (
        grid[1:-1, 1:-1, 2:, 0] - grid[1:-1, 1:-1, 1:-1, 0]
        + grid[1:-1, 2:, 1:-1, 1] - grid[1:-1, 1:-1, 1:-1, 1]
        + grid[2:, 1:-1, 1:-1, 2] - grid[1:-1, 1:-1, 1:-1, 2]
    )
    check(np.abs(divergence).max() < 1.0e-3 * np.abs(interior_faces).max(),
          f"state velocity is not divergence-free: {np.abs(divergence).max()}")


def test_no_buoyancy_keeps_interior_at_rest(_work):
    lib.reset_scene()
    zone = smoke_zone(**{"Heat Buoyancy": 0.0, "Density Buoyancy": 0.0})
    obj = lib.build_object("rest", [zone])
    (start,), _ = lib.evaluate(obj, [zone], 1)
    (end,), _ = lib.run_frames(obj, [zone], 2, 6)
    check(not end["velocity"].any(), "velocity appeared without forces")
    check(
        np.array_equal(lib.interior(zone, end["density"]), lib.interior(zone, start["density"])),
        "density changed in a fluid at rest",
    )
    check(end["substeps"] == 1, "zero velocity must take one full-frame step")


def test_buoyancy_opposes_gravity_direction(_work):
    lib.reset_scene()
    zone = smoke_zone(center=(12, 12, 16), Gravity=(9.81, 0.0, 0.0))
    obj = lib.build_object("sideways", [zone])
    (end,), _ = lib.run_frames(obj, [zone], 1, 6)
    weights = end["density"].astype(np.float64)
    mean = (end["velocity"].astype(np.float64) * weights[:, None]).sum(axis=0) / weights.sum()
    check(mean[0] < -1.0e-3, f"heat buoyancy did not oppose +X gravity: {mean}")
    check(abs(mean[1]) < 0.05 * abs(mean[0]) and abs(mean[2]) < 0.05 * abs(mean[0]),
          f"buoyancy leaked into other axes: {mean}")

    lib.reset_scene()
    zone = smoke_zone(center=(12, 12, 16), **{"Heat Buoyancy": 0.0, "Density Buoyancy": -1.0})
    obj = lib.build_object("sink", [zone])
    (end,), _ = lib.run_frames(obj, [zone], 1, 6)
    weights = end["density"].astype(np.float64)
    mean = (end["velocity"].astype(np.float64) * weights[:, None]).sum(axis=0) / weights.sum()
    check(mean[2] < -1.0e-3, f"negative density buoyancy did not sink: {mean}")


def swirl_zone(speed, **params):
    """Divergence-free rotation about Z with the given peak speed in world units per second."""
    zone = Zone(RES, **params)
    cx, cy = (RES[0] - 1) / 2, (RES[1] - 1) / 2
    radius = min(RES[0], RES[1]) / 2 - 1.5

    def component(axis):
        def fn(x, y, z):
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            falloff = np.clip(1.0 - r / radius, 0.0, None) * (r / radius)
            tangent = (-(y - cy), (x - cx), 0.0 * x)[axis]
            return 4.0 * speed * falloff * tangent / np.maximum(r, 1.0e-6)
        return lib.dense(zone, fn)

    velocity = np.stack([component(0), component(1), component(2)], axis=1)
    density = blob(zone, (cx + 5, cy, 16), 3.0)
    lib.make_data_object("swirl_data", zone, density=density, velocity=velocity)
    return zone


def test_cfl_substeps(_work):
    speed = 6.0  # world units per second: 96 cells/s, 3.2 cells per 1/30 s frame.
    forces = {"Heat Buoyancy": 0.0, "Density Buoyancy": 0.0}
    lib.reset_scene()
    zone = swirl_zone(speed, CFL=1.0, **forces)
    obj = lib.build_object("cfl", [zone])
    (start,), _ = lib.evaluate(obj, [zone], 1)
    (step,), _ = lib.evaluate(obj, [zone], 2)
    check(not errors(obj), f"unexpected node errors: {errors(obj)}")
    peak = float(np.linalg.norm(start["velocity"], axis=1).max())
    cells_per_frame = peak / zone.voxel * DT
    check(step["substeps"] >= int(np.ceil(cells_per_frame * 0.7)),
          f"{step['substeps']} substeps for {cells_per_frame:.2f} cells per frame at CFL 1")
    check(step["substeps"] <= int(np.ceil(cells_per_frame * 1.6)) + 1,
          f"too many substeps: {step['substeps']}")
    check(0.5 * peak < step["max_velocity"] < 2.0 * peak,
          f"Max Velocity {step['max_velocity']} is not in world units (input peak {peak})")
    check(not np.array_equal(step["density"], start["density"]), "density was not advected")

    lib.reset_scene()
    relaxed = swirl_zone(speed, CFL=100.0, **forces)
    obj = lib.build_object("cfl_relaxed", [relaxed])
    (one,), _ = lib.run_frames(obj, [relaxed], 1, 2)
    check(one["substeps"] == 1, "a large CFL limit must permit one full-frame step")


def test_substep_limit_passes_state_through(_work):
    lib.reset_scene()
    zone = swirl_zone(6.0, CFL=0.25, **{"Max Substeps": 2, "Heat Buoyancy": 0.0})
    obj = lib.build_object("limit", [zone])
    (start,), start_names = lib.evaluate(obj, [zone], 1)
    (after,), names = lib.run_frames(obj, [zone], 2, 4)
    messages = errors(obj)
    check(any("substep limit" in m for m in messages), f"missing substep-limit error: {messages}")
    check(names == start_names, "failure must return the original input volume")
    for name in ("density", "temperature", "velocity"):
        check(np.array_equal(after[name], start[name]), f"failed step modified {name}")
    check(after["substeps"] == 0 and after["solve_ms"] == 0.0, "failure must zero diagnostics")


def _rejected(zone, expected):
    obj = lib.build_object("reject", [zone])
    (start,), start_names = lib.evaluate(obj, [zone], 1)
    (after,), names = lib.run_frames(obj, [zone], 2, 3)
    messages = errors(obj)
    check(any(expected in m for m in messages), f"expected '{expected}' in {messages}")
    check(names == start_names, "rejection must return the original input volume")
    check(np.array_equal(after["density"], start["density"]), "rejection modified the state")
    check(after["substeps"] == 0, "rejection must zero diagnostics")


def test_transform_and_type_rejection(_work):
    lib.reset_scene()
    _rejected(smoke_zone(init_voxel=0.125), "does not match the requested origin and voxel-size")
    lib.reset_scene()
    _rejected(smoke_zone(init_origin=(-1.0, -1.0, -0.9)),
              "does not match the requested origin and voxel-size")
    lib.reset_scene()
    _rejected(smoke_zone(init_density_type="INT"), "must be a float grid")

    # Same transform but more voxels than the node's domain.
    lib.reset_scene()
    big = Zone((RES[0], RES[1], RES[2] + 8))
    ones = np.ones(big.cells, dtype=np.float32)
    lib.make_data_object("big_data", big, density=ones, temperature=ones)
    big_node = Zone(RES)
    big_node.data_object = big.data_object
    big_node.init_res = big.res
    _rejected(big_node, "active voxels outside the simulation domain")


def _bake_round_trip(work, target):
    lib.reset_scene()
    scene = bpy.context.scene
    scene.frame_end = 12
    zone = smoke_zone()
    obj = lib.build_object("bake", [zone])
    live = {}
    for frame in range(1, 13):
        (result,), _ = lib.evaluate(obj, [zone], frame)
        live[frame] = result

    blend = os.path.join(work, f"bake_{target.lower()}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    modifier = obj.modifiers[0]
    modifier.bake_target = target
    directory = os.path.join(work, f"bake_{target.lower()}_cache")
    if target == "DISK":
        modifier.bake_directory = directory
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    scene.frame_set(1)
    check("FINISHED" in bpy.ops.object.simulation_nodes_cache_bake(selected=False), "bake failed")
    if target == "DISK":
        files = [f for _, _, names in os.walk(directory) for f in names]
        check(files, "disk bake wrote no files")
    bpy.ops.wm.save_mainfile()

    bpy.ops.wm.open_mainfile(filepath=blend)
    bpy.context.view_layer.update()
    obj = bpy.data.objects["bake"]
    # Random access proves the frames come from the bake, not from re-simulation.
    for frame in (12, 7, 2, 9):
        (result,), names = lib.evaluate(obj, [zone], frame)
        check(names == sorted(lib.GRID_NAMES), f"baked frame {frame} lost grids: {names}")
        for name in lib.GRID_NAMES:
            check(np.array_equal(result[name], live[frame][name]),
                  f"{target} bake: {name} differs from the live simulation at frame {frame}")
    check(not errors(obj), f"baked playback reported errors: {errors(obj)}")


def test_bake_save_reload(work):
    _bake_round_trip(work, "PACKED")
    _bake_round_trip(work, "DISK")


def test_two_independent_zones(_work):
    def zones():
        a = smoke_zone(prefix="a_", center=(8, 12, 8))
        b = smoke_zone(prefix="b_", res=(20, 28, 24), center=(10, 18, 6), origin=(2.0, 0.0, 0.0),
                       voxel=0.125, Gravity=(0.0, 0.0, -4.0), CFL=1.0)
        return a, b

    def run(selection, repeat=1):
        lib.reset_scene()
        chosen = [zones()[i] for i in selection]
        obj = lib.build_object("zones", chosen)
        results = None
        for _ in range(repeat):
            results, _ = lib.run_frames(obj, chosen, 1, 16)
        check(not errors(obj), f"unexpected node errors: {errors(obj)}")
        return results

    both = run((0, 1))
    solo_a, = run((0,))
    solo_b, = run((1,))
    rerun_a, = run((0,), repeat=2)

    def worst(lhs, rhs):
        return max(float(np.abs(lhs[n] - rhs[n]).max()) for n in lib.GRID_NAMES)

    # Threaded reductions in the pressure solver are not bitwise reproducible, so the baseline is
    # the run-to-run difference of an isolated zone.
    noise = max(worst(solo_a, rerun_a), 1.0e-5)
    check(worst(both[0], solo_a) <= 50.0 * noise,
          f"zone A differs when zone B is present: {worst(both[0], solo_a)} (noise {noise})")
    check(worst(both[1], solo_b) <= 50.0 * noise,
          f"zone B differs when zone A is present: {worst(both[1], solo_b)} (noise {noise})")
    check(float(np.abs(both[1]["density"]).sum()) > 0 and both[0]["density"].shape != both[1]["density"].shape,
          "zones were not distinct")
    print(f"    run-to-run noise {worst(solo_a, rerun_a):.3e}, "
          f"A {worst(both[0], solo_a):.3e}, B {worst(both[1], solo_b):.3e}")


def test_matches_legacy_solver(work):
    """Same single step per frame from the same state must reproduce the legacy gas solver."""
    import gas_node_legacy_compare as legacy

    states, info = legacy.run_legacy(32, 12, os.path.join(work, "legacy"))
    check(info["legacy_steps_per_frame"] == [1], f"legacy steps: {info['legacy_steps_per_frame']}")
    node = legacy.run_node(states[6], info, 6)
    for horizon, limit in ((1, 1.0e-3), (6, 1.0e-2)):
        row = legacy.compare(node[horizon], states[6 + horizon], info)
        for key in ("density_rel_l2", "temperature_rel_l2", "velocity_rel_l2"):
            check(row[key] < limit, f"+{horizon} frames: {key} = {row[key]:.3e} (limit {limit})")
        print(f"    +{horizon} frames: density {row['density_rel_l2']:.2e}, "
              f"velocity {row['velocity_rel_l2']:.2e}")


def test_pressure_preconditioners_and_diagnostics(_work):
    def run(**params):
        lib.reset_scene()
        zone = smoke_zone(**params)
        obj = lib.build_object("pressure", [zone])
        (result,), _ = lib.run_frames(obj, [zone], 1, 8)
        return result, lib.warnings(obj)

    tight = {"Pressure Accuracy": 1.0e-6, "Pressure Iteration Factor": 20.0}
    multigrid, multigrid_warnings = run(Multigrid=True, **tight)
    mic, mic_warnings = run(Multigrid=False, **tight)
    for result, messages in ((multigrid, multigrid_warnings), (mic, mic_warnings)):
        check(result["pressure_iterations"] >= 1, "pressure iterations were not reported")
        check(result["pressure_residual"] <= 1.0e-6, f"residual {result['pressure_residual']}")
        check(not any("iteration limit" in m for _, m in messages), f"unexpected: {messages}")
    check(multigrid["pressure_iterations"] < mic["pressure_iterations"],
          "multigrid did not need fewer iterations than MIC")
    scale = float(np.abs(mic["velocity"]).max())
    difference = float(np.abs(multigrid["velocity"] - mic["velocity"]).max())
    check(difference < 1.0e-3 * scale, f"preconditioners disagree: {difference} of {scale}")
    print(f"    iterations: multigrid {multigrid['pressure_iterations']:.0f}, "
          f"MIC {mic['pressure_iterations']:.0f}; velocity difference {difference / scale:.1e}")

    starved, starved_warnings = run(
        Multigrid=False, **{"Pressure Accuracy": 1.0e-8, "Pressure Iteration Factor": 0.1})
    check(any(kind == "WARNING" and "iteration limit" in m for kind, m in starved_warnings),
          f"missing convergence warning: {starved_warnings}")
    check(starved["pressure_residual"] > 1.0e-8, "starved solve reported convergence")
    check(starved["substeps"] >= 1, "an unconverged solve must still advance the state")


def test_reruns_are_reproducible(_work):
    lib.reset_scene()
    zone = smoke_zone()
    obj = lib.build_object("resume", [zone])
    whole = lib.run_frames(obj, [zone], 1, 9)[0][0]
    # Re-run after jumping back: the zone restarts from frame 1 and must reproduce the state.
    again = lib.run_frames(obj, [zone], 1, 9)[0][0]
    for name in lib.GRID_NAMES:
        check(np.array_equal(whole[name], again[name]), f"{name} is not reproducible")


TESTS = [
    test_node_is_registered,
    test_zero_time_start_frame,
    test_smoke_advances_and_rises,
    test_no_buoyancy_keeps_interior_at_rest,
    test_buoyancy_opposes_gravity_direction,
    test_cfl_substeps,
    test_substep_limit_passes_state_through,
    test_transform_and_type_rejection,
    test_bake_save_reload,
    test_two_independent_zones,
    test_pressure_preconditioners_and_diagnostics,
    test_reruns_are_reproducible,
    test_matches_legacy_solver,
]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=None)
    parser.add_argument("-k", default="")
    args = parser.parse_args(argv)
    work = args.work or tempfile.mkdtemp(prefix="gas_node_tests_")
    os.makedirs(work, exist_ok=True)

    failures = 0
    selected = [t for t in TESTS if args.k in t.__name__]
    for test in selected:
        try:
            test(work)
            print(f"PASS {test.__name__}", flush=True)
        except Exception:  # noqa: BLE001 - report every failure and continue.
            failures += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(selected) - failures}/{len(selected)} runtime tests passed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
