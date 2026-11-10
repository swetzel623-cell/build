# Blender Geometry Nodes gas-step integration

This directory contains a source overlay and installer for Blender commit
`377d577d520472f2702e456b26c17dffc663dc52`.

## Apply

From a clean Blender checkout at the pinned commit:

```sh
python3 /path/to/Blender-Mantaflow-Improvement/blender/apply_integration.py /path/to/blender --check
python3 /path/to/Blender-Mantaflow-Improvement/blender/apply_integration.py /path/to/blender
```

The installer verifies the checked-out commit and expected source anchors before writing. Use
`--allow-unpinned` only when intentionally porting the overlay to another revision.

Build Blender with `WITH_FLUID=ON` and `WITH_OPENVDB=ON`. The installer adds the engine sources to
`intern/mantaflow`, adds the node to Geometry Nodes node discovery, and links `bf_nodes_geometry`
to `bf::intern::mantaflow`. The node is not built when either dependency is disabled.

The files are also mirrored under `blender/intern` and `blender/source` for review. No upstream
Blender files other than the two CMake lists are modified.

## Node contract

`Gas Simulation Step` is intended for a Geometry Nodes Simulation Zone:

1. Connect the zone's initial/previous volume state to `State`.
2. Connect Simulation Input's `Delta Time` to the node's `Delta Time`.
3. Connect the node's `State` to Simulation Output.
4. Use `Get Named Grid` to inspect or render `density`, `temperature`, `velocity`, and `pressure`.

The input volume may omit any of those grids; missing grids are initialized to zero. Present grids
must have the expected float or float-vector type, exactly match the requested origin and uniform
voxel-size transform, and have no active voxels outside the configured domain. The node does not
silently resample mismatched grids.

Velocity is cell-centered and expressed in world units per second at the node boundary. The adapter
converts it to grid cells per second and then to Mantaflow MAC form. It converts back only after all
internal substeps have completed. Gravity is similarly converted from world units per second
squared to cells per second squared.

The output is a new volume containing the four named grids. It is explicit Geometry Nodes state,
so the Simulation Zone owns baking, reuse, interpolation, and invalidation. There is no separate
solver cache path and no hidden solver singleton. Each call reconstructs its solver and grids from
the input state; independent zones therefore do not share solver ownership.

## Solver scope

The frame-local loop uses:

- CFL-limited substeps with a hard maximum and exact frame-time accounting.
- MacCormack advection for density, temperature, and velocity.
- Density and heat buoyancy.
- Closed wall boundaries.
- Pressure projection with Mantaflow's MIC preconditioner and closed-domain pressure fixing.
- Finite-value checks and pass-through of the original state on failure.

`PcMIC` is intentional. Mantaflow's static multigrid preconditioner path retains data in a
process-global map keyed by solver pointer, which is not an acceptable ownership model for
independent Simulation Zone instances.

The node reports substeps, sampled maximum velocity, OpenVDB import time, engine solve time, and
OpenVDB export time. The timing outputs are instrumentation, not benchmark results; representative
scenes still need to be measured in a real Blender build.

## Deliberate limitations

This is a correctness-first CPU prototype, not a feature-complete replacement for the Fluid
modifier:

- No emission or source operators, obstacles beyond the closed boundary, fire, colors, liquid,
  vorticity confinement, secondary particles, or effector/force-field compatibility.
- No moving adaptive domain, static multigrid preconditioner, pressure-convergence diagnostics, or
  retained solver allocation between calls.
- No activity-mask pressure culling. Pressure remains globally coupled.
- No GPU execution or claim that NanoVDB is a mutable solver layout.
- No custom disk cache. Simulation Zone baking owns persistence.
- Solver reconstruction and OpenVDB conversion costs are expected to be significant and are
  explicitly exposed for measurement.

## Compiled-engine tests

The reusable engine harness builds Blender's actual pinned `extern/mantaflow` library through its
official CMake source list, then builds and tests the new node-local engine API. It requires CMake
3.22 or newer, Git, a C++20 compiler, TBB, Python development files, and Zlib. OpenVDB is disabled
in this engine-only harness because the dense engine boundary does not depend on it.

From this repository, with a Blender checkout at the pinned commit:

```sh
cmake -S blender/tests -B build-gas-engine \
  -DBLENDER_SOURCE_DIR=/path/to/pinned/blender \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-gas-engine
ctest --test-dir build-gas-engine --output-on-failure
```

The seven runtime tests cover zero-time pass-through, stationary fluid interiors, heat buoyancy
and retained pressure, CFL frame completion, atomic substep-limit failure, non-finite input
rejection, and independent sequential states.

## Validation performed

- The actual pinned Mantaflow library and gas engine compile, and all seven engine tests pass in
  both the normal Debug harness and an ASan/UBSan-instrumented harness. Strict warnings are errors
  for the new engine and test sources.
- The complete Geometry Nodes adapter translation unit compiles as C++20 against the actual pinned
  Blender headers and system OpenVDB headers. This checks source/API compatibility, not final
  Blender target linking or node execution.
- The installer passes a no-write dry run and application to a fresh checkout at the pinned commit.
  All five copied files match the overlay exactly, and the CMake edits pass `git diff --check`.
- The standalone activity-mask and gas-step driver/layout suites pass in both the default build and
  a strict ASan/UBSan build.

## Remaining validation

These checks do not establish a successful full Blender application build, numerical equivalence to
the legacy modifier, in-Blender node execution, cache round-trip correctness, thread safety under
every node evaluation schedule, or runtime performance. The sanitizer result covers the standalone
engine tests, not the complete Blender process or Geometry Nodes adapter. Build the patched pinned
Blender source and run gas advancement, transform-rejection, cache-reload, and
multiple-independent-zone tests before treating the integration as usable.
