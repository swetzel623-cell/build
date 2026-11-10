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
- Cell-centered OpenVDB velocity interchange and internal Mantaflow MAC storage.
- MacCormack advection, buoyancy, closed wall boundaries, and MIC-preconditioned pressure projection using compiled Mantaflow operators.
- Frame-local CFL substep scheduling, failure without partial state mutation, and finite-value checks.
- Runtime import, solve, and export timing outputs.

The complete frame remains in Mantaflow storage between node entry and exit. No legacy `FluidModifierData`, generated Python script, global solver singleton, or separate cache mechanism is used.

See [`blender/README.md`](blender/README.md) for the pinned Blender commit, installer usage, node contract, test harness, and limitations. The compiled engine and complete node source file have been checked against the pinned sources, and seven actual-solver runtime tests pass. A full integrated Blender build and in-Blender runtime simulation are still required.

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

## Next validation step

Build the patched pinned Blender source, then test smoke advancement, bake/reload behavior, two independent Simulation Zones, substep-limit failure, and grid-transform rejection. Compare against the legacy gas solver where the supported feature sets overlap and record the node timing outputs before attempting retained-state or GPU optimizations.

## License

The standalone prototype is MIT licensed. Files in the `blender/` integration overlay are GPL-2.0-or-later, and third-party dependencies retain their own licensing requirements.
