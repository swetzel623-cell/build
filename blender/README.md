# Blender Geometry Nodes gas-step integration

This directory contains a source overlay and installer for Blender commit
`377d577d520472f2702e456b26c17dffc663dc52`.

## Apply

From a clean Blender checkout at the pinned commit:

```sh
python3 /path/to/repository/blender/apply_integration.py /path/to/blender --check
python3 /path/to/repository/blender/apply_integration.py /path/to/blender
```

The installer verifies the checked-out commit and expected source anchors before writing. Use
`--allow-unpinned` only when intentionally porting the overlay to another revision.

It copies five source files (mirrored under `blender/intern` and `blender/source` for review) and
edits four upstream files:

| Upstream file | Edit |
| --- | --- |
| `intern/mantaflow/CMakeLists.txt` | Adds the engine sources and defines `TBB=1` for the engine source, which includes Mantaflow's kernel headers. |
| `source/blender/nodes/geometry/CMakeLists.txt` | Adds the node source; with `WITH_MOD_FLUID` and `WITH_OPENVDB`, defines `WITH_FLUID` and links `bf_intern_mantaflow`. |
| `source/blender/makesrna/intern/rna_nodes_geometry.cc` | Defines the node's RNA struct. Built-in node registration asserts without it. |
| `scripts/startup/bl_ui/node_add_menu_geometry.py` | Adds the node to Add > Simulation. |

Build Blender with `WITH_MOD_FLUID=ON` and `WITH_OPENVDB=ON`. The node type is registered in every
configuration; without Mantaflow or OpenVDB it reports an error and passes its input through.

## Node contract

`Gas Simulation Step` is intended for a Geometry Nodes Simulation Zone:

1. Connect the zone's initial/previous volume state to `State`.
2. Connect Simulation Input's `Delta Time` to the node's `Delta Time`.
3. Connect the node's `State` to Simulation Output.
4. Use `Get Named Grid` to inspect or render `density`, `temperature`, `velocity`, and `pressure`.

The input volume may omit any of those grids; missing grids are initialized to zero. Present grids
must have the expected float or float-vector type, match the requested origin and uniform
voxel-size transform (OpenVDB's transform comparison, which tolerates float rounding), and have no
active voxels outside the configured domain. The node does not resample mismatched grids.

The output is a new volume containing the four named grids. It is explicit Geometry Nodes state,
so the Simulation Zone owns baking, reuse, interpolation, and invalidation. There is no separate
solver cache path and no hidden solver singleton. Each call reconstructs its solver and grids from
the input state; independent zones therefore do not share solver ownership.

### Velocity layout

The output `velocity` grid is the solver's own **staggered** (MAC) field in world units per second,
tagged with OpenVDB's `GRID_STAGGERED` class, the same convention as the legacy fluid cache.
Component `c` of voxel `(i, j, k)` is the velocity through that voxel's lower face along axis `c`,
so it is offset by half a voxel from the voxel center along `c`.

On input, a `velocity` grid tagged as staggered is taken as solver state unchanged. Any other
vector grid is treated as an authored cell-centered velocity and averaged onto faces once. That
path is for initial conditions and for velocity edited inside the zone; it is lossy by nature.

The state was originally exchanged cell-centered in both directions. Averaging faces to centers
and back every frame is a low-pass filter on the velocity field: measured against the legacy
solver from identical state, it removed about 14% of the kinetic energy in one frame and 42% in 20.
With the staggered state the node reproduces the legacy solver (see below), and two calls of half
a frame are bitwise identical to one call taking the same two substeps.

Gravity is converted from world units per second squared to cells per second squared.

## Solver scope

The frame-local loop uses:

- CFL-limited substeps with a hard maximum and exact frame-time accounting.
- MacCormack advection for density, temperature, and velocity.
- Density and heat buoyancy.
- Closed wall boundaries.
- Pressure projection with Mantaflow's MIC preconditioner and closed-domain pressure fixing.
- Finite-value checks and pass-through of the original state on failure.

`PcMIC` is intentional. Both of Mantaflow's multigrid preconditioner modes go through
`solvePressureSystem`, which reads and writes a process-global `std::map` keyed by solver pointer
without synchronization. That is not an acceptable ownership model for independent Simulation
Zone instances that may evaluate concurrently. It is also the dominant cost; see the timings.

The node reports substeps, sampled maximum velocity, OpenVDB import time, engine solve time, and
OpenVDB export time.

### Mapping from legacy Fluid modifier settings

For a domain whose longest side is `L` world units, the legacy solver's smoke buoyancy matches
`Density Buoyancy = Buoyancy Density / L` and `Heat Buoyancy = Buoyancy Heat / L`, with `Gravity`
set to the scene gravity. The legacy solver's heat grid is twice the value its Python `heat_grid`
property returns.

## Deliberate limitations

This is a correctness-first CPU prototype, not a feature-complete replacement for the Fluid
modifier:

- No emission or source operators, obstacles beyond the closed boundary, fire, colors, liquid,
  vorticity confinement, secondary particles, or effector/force-field compatibility.
- No moving adaptive domain, multigrid preconditioner, pressure-convergence diagnostics, or
  retained solver allocation between calls.
- No activity-mask pressure culling. Pressure remains globally coupled.
- No GPU execution or claim that NanoVDB is a mutable solver layout.
- No custom disk cache. Simulation Zone baking owns persistence.

## Tests

### Compiled-engine tests

The reusable engine harness builds Blender's actual pinned `extern/mantaflow` library through its
official CMake source list, then builds and tests the node-local engine API. It requires CMake
3.22 or newer, Git, a C++20 compiler, TBB, Python development files, and Zlib.

```sh
cmake -S blender/tests -B build-gas-engine \
  -DBLENDER_SOURCE_DIR=/path/to/pinned/blender \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-gas-engine
ctest --test-dir build-gas-engine --output-on-failure
```

Nine runtime tests cover zero-time pass-through, stationary fluid interiors, heat buoyancy and
retained pressure, CFL frame completion, atomic substep-limit failure, non-finite input rejection,
independent sequential states, exact resumption from the staggered state, and the velocity layout
reported after a failure.

### In-Blender runtime tests

These need a Blender binary built from the patched pinned source. They build Simulation Zone node
trees from Python, inject dense initial state through a data mesh, and read the grids back for
NumPy comparison.

```sh
blender -b --factory-startup --python blender/tests/gas_node_runtime_tests.py
blender -b --factory-startup --python blender/tests/gas_node_legacy_compare.py -- --horizons 1 5 20 60
blender -b --factory-startup --python blender/tests/gas_node_timings.py -- --res 32 64 96 128 --legacy
```

`gas_node_runtime_tests.py` runs twelve tests and exits non-zero on failure:

| Test | What it establishes |
| --- | --- |
| node is registered | The node type exists with the expected sockets. |
| zero time start frame | The zone's start frame (`Delta Time` 0) returns the input volume bit for bit, no solver work. |
| smoke advances and rises | 24 frames of buoyant smoke: four finite grids, rises against gravity, pressure retained, no flow through closed walls, divergence-free state. |
| no buoyancy keeps interior at rest | Without forces velocity stays exactly zero and interior density is unchanged; one full-frame step. |
| buoyancy opposes gravity direction | Heat buoyancy with +X gravity accelerates along -X only; negative density buoyancy sinks. |
| CFL substeps | A 3.2 cells-per-frame swirl at CFL 1 is subdivided; `Max Velocity` is in world units; a large CFL takes one step. |
| substep limit passes state through | Exhausting `Max Substeps` reports the error, returns the input volume unchanged, zeroes diagnostics. |
| transform and type rejection | Wrong voxel size, wrong origin, integer density grid, and active voxels outside the domain are each rejected with pass-through. |
| bake save reload | Packed and disk bakes: after saving and reopening, frames read in random order equal the live simulation bit for bit. |
| two independent zones | Two zones with different resolution, transform, gravity, and CFL in one tree equal their isolated runs bit for bit. |
| reruns are reproducible | Restarting the zone reproduces the state bit for bit. |
| matches legacy solver | See the next section; asserts relative errors below 1e-3 after 1 frame and 1e-2 after 6. |

## Validation performed

Environment: Ubuntu 24.04, 4 cores (Intel Xeon 2.1 GHz), Clang 18.1.3 (the pinned source rejects
GCC older than 14), Blender's precompiled `lib/linux_x64` libraries at the pinned submodule
commit.

- **Full patched Blender build and link.** `blender_lite.cmake` with `WITH_HEADLESS`,
  `WITH_MOD_FLUID`, `WITH_OPENVDB`, `WITH_OPENVDB_BLOSC`, `WITH_TBB`, and `WITH_PYTHON_INSTALL`
  enabled, `Release`. All 4878 build steps completed, the `blender` target linked and installed,
  and the binary reports `Blender 5.4.0 Alpha`. No compiler warnings came from the overlay
  sources. This is a headless build without Cycles or a window system.
- **In-Blender runtime tests.** 12 of 12 pass in that binary. Separately, the `.vdb` files of
  the disk bake were inspected and carry the `staggered` grid class for the velocity grid.
- **Engine tests.** 9 of 9 pass against the pinned Mantaflow library, in a Debug build with GCC
  13.3 and in an ASan/UBSan build, with `-Wall -Wextra -Werror` on the engine and test sources.
- **Standalone suites.** Activity-mask and gas-step driver suites pass.
- **Installer.** Dry run and application to the pinned checkout succeed; a second application is
  refused; `git diff --check` is clean; the installed tree is byte-identical to the tree that was
  built and tested.

### Agreement with the legacy gas solver

`gas_node_legacy_compare.py` runs the legacy Fluid modifier on a 32³ closed domain restricted to
the node's feature set (no vorticity, dissolve, or noise), hands its state at frame 6 to the node,
and advances both. Both take one solver step per frame, which the script verifies by counting
legacy steps.

| Frames after handover | Density rel. L2 | Temperature rel. L2 | Velocity rel. L2 | Kinetic energy ratio |
| --- | --- | --- | --- | --- |
| 1 | 8.3e-07 | 7.1e-07 | 2.2e-05 | 1.000 |
| 5 | 8.5e-06 | 7.9e-06 | 2.3e-05 | 1.000 |
| 20 | 1.3e-04 | 7.1e-04 | 9.6e-04 | 1.000 |
| 60 | 5.4e-02 | 4.2e-02 | 2.7e-02 | 0.998 |

The remaining differences are consistent with the different pressure preconditioner and initial
guess at a 1e-3 solver tolerance, amplified over time by the flow. This is agreement on one scene
at one resolution for the supported feature subset, not a general equivalence proof.

Two behaviors of the legacy solver matter when comparing:

- With **Adaptive Time Steps disabled** at 30 fps the pinned legacy loop takes **two** full solver
  steps per frame. It passes Mantaflow a time step rounded to six decimals (`0.083333`), which is
  just short of the frame length, so its `while` loop runs again. The comparison therefore keeps
  adaptive time steps enabled.
- MacCormack advection does not conserve mass. In the legacy run total density changes by -4%
  over 20 frames and +31% over 60; the node tracks it (mass ratio 1.00 and 1.02).

### Timings

`gas_node_timings.py`, buoyant blob in a closed cube, 20 frames, 4 threads, medians in
milliseconds per frame. These are measurements of one small scene on one machine.

| Resolution | Import | Solve | Export | Node frame (wall) | Node substeps | Legacy frame (wall) | Legacy steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32³ | 0.9 | 30 | 1.3 | 34 | 1 | 42 | 1 |
| 64³ | 5.8 | 488 | 5.2 | 502 | 2 | 304 | 2 |
| 96³ | 19 | 3333 | 16 | 3380 | 3 | 953 | 2 |
| 128³ | 48 | 13775 | 39 | 13878 | 3.5 | 2954 | 2.5 |

Import and export, which were the suspected cost, are under 1% of the frame. The solve dominates,
and at 64³ and above the node is slower than the legacy modifier: 1.7 times at 64³, 3.5 times at
96³, and 4.7 times at 128³. Setting `Pressure Accuracy` to its loosest value (1.0), which ends the
pressure iteration almost immediately, cut the solve time to 188 ms at 64³ and 1135 ms at 96³, so
the pressure solve accounts for at least 60% of it. The legacy solver uses the static multigrid
preconditioner; the node uses MIC with a CG iteration limit proportional to resolution. The node
also schedules more substeps at the same CFL number, because it applies `CFL / max velocity`
directly where Mantaflow's scheduler clamps and rounds the step. The legacy column includes
emission processing and cache writes the node does not do, so it is context rather than a
like-for-like benchmark.

## Remaining validation

Not done, and not claimed:

- A full-feature Blender build (window system, Cycles). The node has not been used interactively
  or rendered.
- Sanitizer validation of the Blender process or the Geometry Nodes adapter. One memory bug in
  the adapter (an unowned grid reference being released) was found and fixed through Blender's
  own assertion; the adapter has not been run under ASan.
- Thread-safety under a concurrent legacy Fluid bake. Mantaflow's `PbClass` instance list is a
  process-global vector that the engine's grid destructors read.
- Evidence that two zones in one tree were actually evaluated concurrently in the test.
- Legacy agreement at other resolutions, with multiple substeps per frame, or on non-cubic
  domains.
- Platforms other than Linux x86-64, and compilers other than Clang 18 (GCC 13 for the engine
  harness).
