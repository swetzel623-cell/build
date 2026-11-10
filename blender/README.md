# Blender Geometry Nodes gas-step integration

This directory contains a source overlay and patch for Blender commit
`377d577d520472f2702e456b26c17dffc663dc52`.

## Apply

From a clean Blender checkout at the pinned commit:

```sh
git apply --check /path/to/Blender-Mantaflow-Improvement/blender/patches/0001-geometry-nodes-gas-step.patch
git apply /path/to/Blender-Mantaflow-Improvement/blender/patches/0001-geometry-nodes-gas-step.patch
```

Build Blender with `WITH_FLUID=ON` and `WITH_OPENVDB=ON`. The patch adds the engine sources to
`intern/mantaflow`, adds the node to Geometry Nodes node discovery, and links `bf_nodes_geometry`
to `bf::intern::mantaflow`. The node is not built when either dependency is disabled.

The files are also mirrored under `blender/intern` and `blender/source` for review. The patch is
the authoritative apply mechanism; no upstream Blender files other than the two CMake lists are
modified.

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

## Validation boundary

The standalone scheduler/layout tests and patch-application check can run without Blender. They do
not establish compilation of the complete Blender target, numerical equivalence to the legacy
modifier, cache round-trip correctness, thread safety under every node evaluation schedule, or
runtime performance. Those require building the patched pinned Blender source and running gas,
cache-reload, and multiple-independent-zone tests.
