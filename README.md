# Blender Mantaflow Improvement

Experimental engine-side work toward a faster Mantaflow gas solver controlled from Blender Geometry Nodes.

The intended architecture is a frontend-independent engine with explicit, serializable state. Geometry Nodes and the legacy Fluid modifier should call the same engine rather than becoming separate simulation implementations.

## Current increment

This branch contains the first standalone Phase 1 prototype:

- Dense-grid adapters with no OpenVDB or Blender dependency.
- Block activity masks formed from density, temperature, fuel, and velocity.
- Configurable block dilation for local stencil experiments.
- Unit tests and an A/B microbenchmark.
- Cross-platform CMake/CTest build configuration.

The mask is intentionally **not** used as a pressure-projection mask. Smoke density alone does not establish which cells participate in the globally coupled velocity solve.

Building the mask still scans the input grids. The immediate question is therefore whether skipping expensive local kernels repays that scan and dilation overhead; this prototype does not claim active-voxel-proportional memory or a complete sparse solver.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the microbenchmark with defaults:

```sh
./build/activity_mask_benchmark
```

Or provide grid size, block size, and repetition count:

```sh
./build/activity_mask_benchmark 192 8 20
```

The benchmark reports JSON so results can later be collected by a regression dashboard. It measures a local stencil kernel, not an entire Mantaflow bake.

## Next integration step

Implement a CPU `Gas Simulation Step` prototype inside a Geometry Nodes Simulation Zone. That prototype must establish solver-state ownership, named-grid outputs, cache invalidation, and the cost of converting between Mantaflow and Geometry Nodes/OpenVDB storage before GPU execution is introduced.

## License

The standalone prototype is MIT licensed. Future Blender integration and third-party dependencies retain their own licensing requirements.
