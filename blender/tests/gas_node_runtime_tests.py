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


def sparse_zone(prefix="", res=(32, 32, 48), **params):
    params.setdefault("Sparse GPU", True)
    params.setdefault("CFL", 6.0)
    zone = Zone(res, prefix=prefix, **params)
    field = blob(zone, (res[0] / 2, res[1] / 2, res[2] / 4), 4.0)
    lib.make_data_object((prefix or "sparse") + "_data", zone, density=field, temperature=field)
    return zone


def sparse_note(obj):
    """Which device ran the sparse solver, for the log: a CPU fallback is a warning."""
    notes = [message for kind, message in lib.warnings(obj) if kind != "ERROR"]
    return notes[0] if notes else "GPU"


def test_sparse_gpu_advances_and_rises(_work):
    lib.reset_scene()
    zone = sparse_zone()
    obj = lib.build_object("sparse", [zone])
    (start,), _ = lib.evaluate(obj, [zone], 1)
    (end,), names = lib.run_frames(obj, [zone], 2, 16)
    print(f"  sparse solver: {sparse_note(obj)}")
    check(lib.state_grids_match(names, zone), f"missing state grids: {names}")
    check(not errors(obj), f"unexpected node errors: {errors(obj)}")
    for name in zone.grid_names:
        check(np.isfinite(end[name]).all(), f"{name} is not finite")
    rise = center_of_mass_z(zone, end["density"]) - center_of_mass_z(zone, start["density"])
    check(rise > 1.0, f"smoke rose only {rise:.3f} cells")
    check(end["substeps"] >= 1 and end["max_velocity"] > 0.0, "diagnostics were not reported")
    check(end["active_bricks"] >= 1, "no active bricks were reported")
    # Advection holds the total; only values below the activity threshold are dropped.
    mass_ratio = float(end["density"].sum() / start["density"].sum())
    check(abs(mass_ratio - 1.0) < 1.0e-3, f"density mass changed by a factor of {mass_ratio:.5f}")
    grid = lib.as_grid(zone, end["velocity"])
    wall_normal = max(np.abs(grid[k, :, :, 2]).max() for k in (0, 1, zone.res[2] - 1))
    check(wall_normal == 0.0, f"velocity {wall_normal} through a closed wall")


def plate_colliders(bottom0, speed_cells):
    """A 14x14x3 plate whose bottom is at voxel bottom0 + (frame - 1) * speed_cells / 30."""
    def build(b, zone):
        frame = b.new("GeometryNodeInputSceneTime").outputs["Frame"]
        bottom = b.math("ADD", float(bottom0),
                        b.math("FLOOR", b.math("MULTIPLY", b.math("SUBTRACT", frame, 1.0),
                                               speed_cells / 30.0)))
        top = b.math("ADD", bottom, 3.0)
        mask = lib.box_field(b, zone, (9.0, 9.0, bottom), (23.0, 23.0, top))
        speed = speed_cells * zone.voxel
        return lib.grid_volume(b, zone, [("obstacle", "FLOAT", mask),
                                         ("velocity", "VECTOR", (0.0, 0.0, speed))])
    return build


def plate_mask(zone, bottom):
    return lib.dense(zone, lambda x, y, z: ((z >= bottom) & (z < bottom + 3) & (x >= 9) & (x < 23)
                                            & (y >= 9) & (y < 23)) * 1.0).astype(bool)


def test_sparse_gpu_obstacle(_work):
    lib.reset_scene()
    zone = sparse_zone(colliders=plate_colliders(22, 0.0))
    obj = lib.build_object("plate", [zone])
    (start,), _ = lib.evaluate(obj, [zone], 1)
    (end,), names = lib.run_frames(obj, [zone], 2, 13)
    print(f"  sparse solver: {sparse_note(obj)}")
    check(lib.state_grids_match(names, zone), f"unexpected grids: {names}")
    check(not errors(obj), f"unexpected node errors: {errors(obj)}")
    solid = plate_mask(zone, 22)
    check(np.abs(end["density"][solid]).max() == 0.0, "smoke entered the obstacle")
    above = lib.as_grid(zone, end["density"])[25:, 11:21, 11:21].sum()
    check(above < 0.25 * lib.as_grid(zone, start["density"]).sum(), "the plate did not shelter the space above it")
    mass_ratio = float(end["density"].sum() / start["density"].sum())
    check(abs(mass_ratio - 1.0) < 1.0e-3, f"density mass changed by a factor of {mass_ratio:.5f}")


def test_sparse_gpu_moving_collider(_work):
    lib.reset_scene()
    still = {"Heat Buoyancy": 0.0, "Density Buoyancy": 0.0}
    zone = sparse_zone(colliders=plate_colliders(10, 30.0), **still)
    obj = lib.build_object("moving", [zone])
    (start,), _ = lib.evaluate(obj, [zone], 1)
    (end,), _ = lib.run_frames(obj, [zone], 2, 9)
    print(f"  sparse solver: {sparse_note(obj)}")
    check(not errors(obj), f"unexpected node errors: {errors(obj)}")
    # At frame 9 the plate's bottom is at voxel 18.
    check(np.abs(end["density"][plate_mask(zone, 18)]).max() == 0.0, "smoke inside the moving plate")
    grid = lib.as_grid(zone, end["velocity"])
    face = grid[18, 9:23, 9:23, 2]
    check(np.abs(face - 30.0 * zone.voxel).max() < 1.0e-4, "the plate's faces do not move with it")
    rise = center_of_mass_z(zone, end["density"]) - center_of_mass_z(zone, start["density"])
    check(rise > 0.3, f"the moving plate did not push the smoke (rose {rise:.3f})")


def test_sparse_gpu_sources(_work):
    lib.reset_scene()
    still = {"Heat Buoyancy": 0.0, "Density Buoyancy": 0.0}

    def sphere(level):
        def build(b, zone):
            x, y, z = lib.voxel_coords(b, zone)
            d2 = b.math("ADD", b.math("ADD", b.math("POWER", b.math("SUBTRACT", x, 16.0), 2.0),
                                      b.math("POWER", b.math("SUBTRACT", y, 16.0), 2.0)),
                        b.math("POWER", b.math("SUBTRACT", z, 12.0), 2.0))
            value = b.math("MULTIPLY", b.math("LESS_THAN", d2, 9.5), level)
            return lib.grid_volume(b, zone, [("density", "FLOAT", value),
                                             ("temperature", "FLOAT", value)])
        return build

    res = (32, 32, 48)
    probe = Zone(res)
    inside = lib.dense(probe, lambda x, y, z: ((x - 16) ** 2 + (y - 16) ** 2 + (z - 12) ** 2 < 9.5) * 1.0)

    def run(add):
        lib.reset_scene()
        zone = Zone(res, prefix="", sources=sphere(0.8), **{"Sparse GPU": True, "CFL": 6.0,
                                                             "Add Sources": add, **still})
        empty = np.zeros(zone.cells, dtype=np.float32)
        lib.make_data_object("empty_data", zone, density=empty, temperature=empty)
        obj = lib.build_object("emit", [zone])
        (end,), _ = lib.run_frames(obj, [zone], 1, 7)
        check(not errors(obj), f"unexpected node errors: {errors(obj)}")
        print(f"  sparse solver: {sparse_note(obj)}")
        return end

    raised = run(False)
    worst = np.abs(raised["density"] - 0.8 * inside).max()
    check(worst < 1.0e-6, f"raising did not reach the source's level (off by {worst})")
    added = run(True)
    expected = 0.8 * inside.sum() * 6.0 / 30.0
    total = float(added["density"].sum())
    check(abs(total / expected - 1.0) < 1.0e-3, f"adding gave {total}, expected {expected}")


def test_sparse_gpu_turbulence(_work):
    def run(strength):
        lib.reset_scene()
        zone = sparse_zone(**{"Turbulence": strength, "Turbulence Scale": 0.5})
        obj = lib.build_object("stir", [zone])
        (start,), _ = lib.evaluate(obj, [zone], 1)
        (end,), _ = lib.run_frames(obj, [zone], 2, 13)
        check(not errors(obj), f"unexpected node errors: {errors(obj)}")
        for name in zone.grid_names:
            check(np.isfinite(end[name]).all(), f"{name} is not finite")
        print(f"  sparse solver: {sparse_note(obj)}")
        return start, end

    _, calm = run(0.0)
    start, stirred = run(150.0 * 0.0625)
    change = np.linalg.norm(stirred["density"] - calm["density"]) / np.linalg.norm(calm["density"])
    print(f"  turbulence changed the density by {change:.3f} (relative L2)")
    check(change > 0.05, "turbulence made no visible difference")
    mass_ratio = float(stirred["density"].sum() / start["density"].sum())
    check(abs(mass_ratio - 1.0) < 1.0e-3, f"density mass changed by a factor of {mass_ratio:.5f}")


def test_sparse_gpu_keeps_state(_work):
    def run(zones, last):
        notes = []
        obj = lib.build_object("kept", zones)
        lib.evaluate(obj, zones, 1)
        result = None
        for frame in range(2, last + 1):
            result, _ = lib.evaluate(obj, zones, frame)
            notes.append([m for k, m in lib.warnings(obj) if k != "ERROR"])
        check(not errors(obj), f"unexpected node errors: {errors(obj)}")
        return result, notes

    lib.reset_scene()
    alone, notes = run([sparse_zone(prefix="a_")], 10)
    print(f"  frame 2: {notes[0]}; frame 10: {notes[-1]}")
    if any("ran on the CPU" in m for m in notes[-1]):
        print("  no Vulkan device: nothing is kept on a GPU")
        return
    check(all("state uploaded" in m for m in notes[0]), "the first step should upload")
    check(all(any("kept on the GPU" in m for m in frame) for frame in notes[1:]),
          "the state was not kept on the GPU between frames")

    # A second zone with another resolution runs in between, on its own solver.
    lib.reset_scene()
    together, _ = run([sparse_zone(prefix="a_"), sparse_zone(prefix="b_", res=(24, 24, 40))], 10)
    for name in lib.SPARSE_GRID_NAMES:
        check(np.array_equal(together[0][name], alone[0][name]),
              f"{name} changed when another zone ran in between")


def test_voxel_size_change_moves_the_state(_work):
    """A simulation zone keeps its state when the voxel size changes part-way through. The node
    moves its own state onto the new grid: the smoke's mass and upward momentum carry over, in
    the Mantaflow path and the Sparse GPU path."""
    old, new = 0.0625, 0.05
    for sparse in (False, True):
        lib.reset_scene()
        zone = smoke_zone(prefix="s" if sparse else "d", **({"Sparse GPU": True, "CFL": 6.0}
                                                         if sparse else {}))
        obj = lib.build_object("refit", [zone])
        lib.evaluate(obj, [zone], 1)
        (before,), _ = lib.run_frames(obj, [zone], 2, 6)
        gas = obj.modifiers[0].node_group.nodes[zone.prefix + "gas"]
        gas.inputs["Voxel Size"].default_value = new
        (after,), names = lib.evaluate(obj, [zone], 7)
        notes = lib.warnings(obj)
        label = "Sparse GPU" if sparse else "Mantaflow"
        check(not errors(obj), f"{label}: unexpected node errors: {errors(obj)}")
        check(any("moved onto the new grid" in message for _kind, message in notes),
              f"{label}: the move was not reported: {notes}")
        check(lib.state_grids_match(names, zone), f"{label}: missing state grids: {names}")

        mass = [float(before["density"].sum()) * old**3, float(after["density"].sum()) * new**3]
        # In a closed box the air's total momentum is zero; the smoke's is what rises.
        lift = [float((before["velocity"][:, 2] * before["density"]).sum()) * old**3,
                float((after["velocity"][:, 2] * after["density"]).sum()) * new**3]
        height = [center_of_mass_z(zone, before["density"]) * old,
                  center_of_mass_z(zone, after["density"]) * new]
        print(f"  {label}: mass {mass[0]:.5f} -> {mass[1]:.5f}, upward momentum "
              f"{lift[0]:.5f} -> {lift[1]:.5f}, smoke height {height[0]:.3f} -> {height[1]:.3f}")
        check(abs(mass[1] / mass[0] - 1.0) < 0.05, f"{label}: the smoke's mass changed")
        check(lift[0] > 0.0 and 0.7 < lift[1] / lift[0] < 1.6,
              f"{label}: the upward momentum did not carry over")
        # One frame of rise, not a jump: within one old voxel of where it was.
        check(0.0 <= height[1] - height[0] < old, f"{label}: the smoke moved on the new grid")


def test_sparse_gpu_fire(_work):
    """A source of fuel burns into flame, heat and smoke that rise; without fuel the state has no
    fire grids."""
    lib.reset_scene()
    res = (32, 32, 48)

    def burner(b, zone):
        x, y, z = lib.voxel_coords(b, zone)
        d2 = b.math("ADD", b.math("ADD", b.math("POWER", b.math("SUBTRACT", x, 16.0), 2.0),
                                  b.math("POWER", b.math("SUBTRACT", y, 16.0), 2.0)),
                    b.math("POWER", b.math("SUBTRACT", z, 8.0), 2.0))
        inside = b.math("LESS_THAN", d2, 9.5)
        return lib.grid_volume(b, zone, [("fuel", "FLOAT", inside),
                                         ("temperature", "FLOAT", b.math("MULTIPLY", inside, 0.3))])

    zone = Zone(res, sources=burner, extra_grids=("fuel", "flame"),
                **{"Sparse GPU": True, "CFL": 6.0, "Burn Rate": 1.5, "Flame Heat": 3.0,
                   "Flame Smoke": 0.8, "Flame Cooling": 3.0})
    empty = np.zeros(zone.cells, dtype=np.float32)
    lib.make_data_object("fire_data", zone, density=empty, temperature=empty)
    obj = lib.build_object("fire", [zone])
    (end,), names = lib.run_frames(obj, [zone], 1, 16)
    check(not errors(obj), f"unexpected node errors: {errors(obj)}")
    print(f"  sparse solver: {sparse_note(obj)}")
    check(lib.state_grids_match(names, zone, ("fuel", "flame")), f"state grids: {names}")
    flame, smoke = end["flame"], end["density"]
    smoke_height = center_of_mass_z(zone, smoke)
    flame_height = center_of_mass_z(zone, flame)
    print(f"  flame total {flame.sum():.1f} (peak {flame.max():.2f}) centred at z "
          f"{flame_height:.1f}; smoke total {smoke.sum():.1f} centred at z {smoke_height:.1f}; "
          f"fuel left {end['fuel'].sum():.1f}")
    check(flame.max() > 0.9, "the fuel did not burn at the full rate anywhere")
    check(smoke.sum() > 1.0, "the fire left no smoke")
    check(smoke_height > 8.0 + 3.0, "the fire's smoke did not rise above the burner")
    check(np.isfinite(end["velocity"]).all(), "velocity is not finite")


def test_sparse_gpu_turbulence_layers(_work):
    """A layer with the same settings as the single turbulence inputs gives the same result, and
    the order of layers matters when one is masked by speed."""
    strength = 150.0 * 0.0625

    def run(single=None, layers=None):
        lib.reset_scene()
        params = {"Turbulence Scale": 0.5, "Turbulence Seed": 4, "Turbulence Mask": 0.0}
        if single:
            params["Turbulence"] = single
        if layers:
            params["layers"] = lambda b, zone: lib.turbulence_layers(b, layers)
        zone = sparse_zone(**params)
        obj = lib.build_object("layers", [zone])
        lib.evaluate(obj, [zone], 1)
        (end,), _ = lib.run_frames(obj, [zone], 2, 9)
        check(not errors(obj), f"unexpected node errors: {errors(obj)}")
        return end

    as_input = run(single=strength)
    as_layer = run(layers=[{"strength": strength, "scale": 0.5, "speed": 1.0, "seed": 4,
                            "mask": 0, "mask_level": 1.0}])
    same = np.abs(as_input["density"] - as_layer["density"]).max()
    print(f"  one layer against the single turbulence inputs: largest difference {same:.2e}")
    check(same < 1.0e-6, "a layer did not match the same turbulence given as inputs")

    broad = {"strength": strength, "scale": 0.5, "seed": 1, "mask": 0, "mask_level": 1.0}
    fast = {"strength": 2.0 * strength, "scale": 0.15, "seed": 2, "mask": 4, "mask_level": 0.3}
    first = run(layers=[broad, fast])
    second = run(layers=[fast, broad])
    order = np.linalg.norm(first["density"] - second["density"]) / np.linalg.norm(first["density"])
    print(f"  swapping a speed-masked layer and a broad one changed the density by {order:.3f}")
    check(order > 1.0e-3, "the order of the layers made no difference")


def test_sparse_gpu_expansion(_work):
    """A 'divergence' source grid makes the gas expand there, pushing the smoke around it
    outward."""
    res = (32, 32, 32)

    def core(level):
        def build(b, zone):
            x, y, z = lib.voxel_coords(b, zone)
            d2 = b.math("ADD", b.math("ADD", b.math("POWER", b.math("SUBTRACT", x, 16.0), 2.0),
                                      b.math("POWER", b.math("SUBTRACT", y, 16.0), 2.0)),
                        b.math("POWER", b.math("SUBTRACT", z, 16.0), 2.0))
            return lib.grid_volume(b, zone, [("divergence", "FLOAT",
                                              b.math("MULTIPLY", b.math("LESS_THAN", d2, 9.5),
                                                     level))])
        return build

    def spread(level):
        lib.reset_scene()
        zone = Zone(res, sources=core(level), **{"Sparse GPU": True, "CFL": 6.0,
                                                 "Heat Buoyancy": 0.0, "Density Buoyancy": 0.0})
        field = blob(zone, (16, 16, 16), 8.0)
        lib.make_data_object("expand_data", zone, density=field, temperature=field * 0.0)
        obj = lib.build_object("expand", [zone])
        lib.evaluate(obj, [zone], 1)
        (end,), _ = lib.run_frames(obj, [zone], 2, 7)
        check(not errors(obj), f"unexpected node errors: {errors(obj)}")
        grid = lib.as_grid(zone, end["density"]).astype(np.float64)
        z, y, x = np.meshgrid(*(np.arange(n) for n in grid.shape), indexing="ij")
        r = np.sqrt((x - 16.0) ** 2 + (y - 16.0) ** 2 + (z - 16.0) ** 2)
        return float((grid * r).sum() / grid.sum())

    still, pushed = spread(0.0), spread(20.0)
    print(f"  mean distance of smoke from the core: {still:.2f} voxels without expansion, "
          f"{pushed:.2f} with 20/s")
    check(pushed > still * 1.03, "the divergence source did not push the smoke outward")


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
    test_sparse_gpu_advances_and_rises,
    test_sparse_gpu_obstacle,
    test_sparse_gpu_moving_collider,
    test_sparse_gpu_sources,
    test_sparse_gpu_turbulence,
    test_sparse_gpu_keeps_state,
    test_voxel_size_change_moves_the_state,
    test_sparse_gpu_fire,
    test_sparse_gpu_turbulence_layers,
    test_sparse_gpu_expansion,
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
