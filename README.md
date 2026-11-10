# Blender Mantaflow Improvement

Experimental engine-side work toward a faster Mantaflow gas solver controlled from Blender Geometry Nodes.

The intended architecture is a frontend-independent engine with explicit, serializable state. Geometry Nodes and the legacy Fluid modifier should call the same engine rather than becoming separate simulation implementations.

## Current increments

### Standalone activity-mask prototype

- Dense-grid adapters with no OpenVDB or Blender dependency.
- Block activity masks formed from density, temperature, fuel, and velocity.
- Configurable block dilation for local stencil experiments.
- Unit tests and an A/B microbenchmark.

The mask is intentionally **not** used as a pressure-projection mask. Smoke density alone does not establish which cells participate in the globally coupled velocity solve. Building the mask still scans the input grids; it does not provide active-voxel-proportional memory or a complete sparse solver.

### Geometry Nodes CPU gas-step prototype

The `blender/` overlay adds a compiled `Gas Simulation Step` node and a small node-local Mantaflow API. It provides:

- Explicit, serializable `density`, `temperature`, `velocity`, and `pressure` volume-grid state.
- Per-call solver ownership suitable for independent Simulation Zone instances.
- Velocity exchanged in the solver's own staggered (MAC) layout, so the state survives a frame
  boundary without loss; authored cell-centered velocities are accepted and resampled once.
- MacCormack advection, buoyancy, closed wall boundaries, and MIC-preconditioned pressure projection using compiled Mantaflow operators.
- Frame-local CFL substep scheduling, failure without partial state mutation, and finite-value checks.
- Runtime import, solve, and export timing outputs.

The complete frame remains in Mantaflow storage between node entry and exit. No legacy `FluidModifierData`, generated Python script, global solver singleton, or separate cache mechanism is used.

See [`blender/README.md`](blender/README.md) for the pinned Blender commit, installer usage, node
contract, tests, measurements, and limitations.

### Validation status

The patched pinned Blender source has been built and linked in full (headless, without Cycles),
and the node has been run inside real Simulation Zones in that binary:

- 12 in-Blender runtime tests pass: smoke advancement, zero time, buoyancy, CFL substepping,
  substep-limit and invalid-grid pass-through, packed and disk bake with save and reload, and two
  independent zones.
- From identical state the node reproduces the legacy Fluid modifier's gas solver on the supported
  feature subset: velocity within 2e-5 (relative L2) after 5 frames and 1e-3 after 20 on a 32³
  scene.
- 9 engine tests pass against the pinned Mantaflow library, also under ASan/UBSan.

It is **not yet faster** than the legacy solver. OpenVDB import and export are under 1% of a
frame, but the solve is 1.7 to 4.7 times slower than the legacy modifier at 64³ to 128³, mostly
in the MIC-preconditioned pressure solve. A full-feature (GUI, Cycles) build, interactive use, and
sanitizer runs of the Blender process are still outstanding.

## Standalone build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the activity-mask microbenchmark with defaults:

```sh
./build/activity_mask_benchmark
```

Or provide grid size, block size, and repetition count:

```sh
./build/activity_mask_benchmark 192 8 20
```

The benchmark reports JSON and measures a local stencil kernel, not an entire Mantaflow bake. Gas-step timing outputs similarly require representative-scene measurements before making performance claims.

## Next steps

1. Give the engine a multigrid pressure solve it owns. Mantaflow's `solvePressureSystem` keeps its
   multigrid in a process-global map for both multigrid modes, so the engine would construct
   `GridMg` and the CG solver itself for the duration of one call.
2. Align the substep schedule with Mantaflow's scheduler, or justify the difference, and report
   pressure iterations and convergence as diagnostics.
3. Repeat the legacy comparison with several substeps per frame, at higher resolution, and on
   non-cubic domains.
4. Build with a window system and Cycles, use the node interactively, and run the Geometry Nodes
   adapter under ASan.

## License

The standalone prototype is MIT licensed. Files in the `blender/` integration overlay are GPL-2.0-or-later, and third-party dependencies retain their own licensing requirements.
