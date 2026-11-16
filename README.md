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

### GPU sparse solver (started)

A brick-sparse gas solver whose data layout is flat arrays a GPU consumes unchanged, with a CPU
reference implementation and GLSL compute kernels run through Vulkan. The kernels match the CPU
reference on a software Vulkan device and on an NVIDIA RTX 3070. It has no meaningful performance
measurement yet, uses simpler numerics than the Mantaflow node, and is not connected to Blender.
See [`docs/gpu-sparse-solver.md`](docs/gpu-sparse-solver.md) for the research, design decisions,
and what is and is not verified.

### Geometry Nodes CPU gas-step prototype

The `blender/` overlay adds a compiled `Gas Simulation Step` node and a small node-local Mantaflow API. It provides:

- Explicit, serializable `density`, `temperature`, `velocity`, and `pressure` volume-grid state.
- Per-call solver ownership suitable for independent Simulation Zone instances.
- Velocity exchanged in the solver's own staggered (MAC) layout, so the state survives a frame
  boundary without loss; authored cell-centered velocities are accepted and resampled once.
- MacCormack advection, buoyancy, closed wall boundaries, and multigrid-preconditioned pressure projection using compiled Mantaflow operators, with the multigrid owned by each call.
- Frame-local CFL substep scheduling, failure without partial state mutation, and finite-value checks.
- Runtime import, solve, and export timing outputs.

The complete frame remains in Mantaflow storage between node entry and exit. No legacy `FluidModifierData`, generated Python script, global solver singleton, or separate cache mechanism is used.

See [`blender/README.md`](blender/README.md) for the pinned Blender commit, installer usage, node
contract, tests, measurements, and limitations.

### Validation status

The patched pinned Blender source has been built and linked in full (headless, without Cycles),
and the node has been run inside real Simulation Zones in that binary:

- 13 in-Blender runtime tests pass: smoke advancement, zero time, buoyancy, CFL substepping,
  substep-limit and invalid-grid pass-through, packed and disk bake with save and reload, two
  independent zones, and both pressure preconditioners.
- From identical state the node reproduces the legacy Fluid modifier's gas solver on the supported
  feature subset: velocity within 4e-6 (relative L2) after 5 frames and 2e-5 after 20 on a 32³
  scene.
- 12 engine tests pass against the pinned Mantaflow library, also under ASan/UBSan.

It is **at parity with, not faster than,** the legacy solver. Integrating the same flow with the
same CFL number, the node takes the same number of substeps and 0.98 to 1.07 times the legacy
wall time at 64³ to 128³. OpenVDB import and export are about 2% of a frame. A full-feature (GUI,
Cycles) build, interactive use, and sanitizer runs of the Blender process are still outstanding.

## Standalone build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The GPU kernels are built when the Vulkan headers and loader and `glslangValidator` are found. On
a machine with a Vulkan-capable GPU, this runs them against the CPU reference and prints timings:

```sh
./build/sparse_gas_gpu_tests 64 20
```

Add `--bench` to time the GPU path alone, with the state kept on the device and no CPU
comparison:

```sh
./build/sparse_gas_gpu_tests 128 30 --bench
```

`--profile` does the same and also reports the GPU time of each kernel. The pressure solve is
six multigrid cycles by default; `--jacobi` selects 60 Jacobi sweeps instead.

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

1. Profile a node call at 128³ and remove what the legacy solver does not pay for: rebuilding the
   multigrid hierarchy every substep, reallocating the solver and its grids every call, and the
   serial finite-value and maximum-velocity scans.
2. Repeat the legacy grid comparison with several substeps per frame, at higher resolution, and on
   non-cubic domains.
3. Build with a window system and Cycles, use the node interactively, and run the Geometry Nodes
   adapter under ASan.

## License

The standalone prototype is MIT licensed. Files in the `blender/` integration overlay are GPL-2.0-or-later, and third-party dependencies retain their own licensing requirements.
