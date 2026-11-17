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

It copies six source files (mirrored under `blender/intern` and `blender/source` for review),
copies the sparse solver's sources from this repository's `include`, `src` and `gpu` directories
into `intern/sparse_gas`, and edits five upstream files:

| Upstream file | Edit |
| --- | --- |
| `intern/CMakeLists.txt` | Adds `intern/sparse_gas` next to `intern/mantaflow` (both with `WITH_MOD_FLUID`). |
| `intern/mantaflow/CMakeLists.txt` | Adds the engine sources and defines `TBB=1` for the engine source, which includes Mantaflow's kernel headers. |
| `source/blender/nodes/geometry/CMakeLists.txt` | Adds the node source; with `WITH_MOD_FLUID` and `WITH_OPENVDB`, defines `WITH_FLUID` and links `bf_intern_mantaflow` and `bf_intern_sparse_gas`. |
| `source/blender/makesrna/intern/rna_nodes_geometry.cc` | Defines the node's RNA struct. Built-in node registration asserts without it. |
| `scripts/startup/bl_ui/node_add_menu_geometry.py` | Adds the node to Add > Simulation. |

Build Blender with `WITH_MOD_FLUID=ON` and `WITH_OPENVDB=ON`. The node type is registered in every
configuration; without Mantaflow or OpenVDB it reports an error and passes its input through.

The sparse solver's GPU kernels are compiled when CMake finds the Vulkan headers (Blender's own
`lib/<platform>/vulkan`, or the Vulkan SDK) and `glslangValidator` (the Vulkan SDK; set
`VULKAN_SDK` or put it on `PATH`). Configure prints `Sparse gas solver: Vulkan kernels enabled`
when it found both. Blender links no Vulkan library for them: the loader (`vulkan-1.dll`,
`libvulkan.so.1`) is opened when the node first needs a device.

## Pyro Simulation modifier

`blender/assets/smoke_assets.py` builds the Pyro node-group assets and installs them into a
Blender build's own Essentials library (`datafiles/assets/nodes/pyro_simulation_assets.blend`,
plus two catalogs in `blender_assets.cats.txt`):

```sh
blender -b --factory-startup --python blender/assets/smoke_assets.py
```

The Windows build workflow runs it, so its zip already has them.

**Add Modifier > Physics > Pyro Simulation** turns the selected mesh into a smoke or fire emitter.

- **Preset**: Custom, Smoke, Large Plume, Trail, Explosion, Campfire or Candle. A preset sets the
  emission, particles, fire, buoyancy and turbulence; the panels below it then show greyed out.
  `Strength` scales what is emitted and `Speed` the emission and particle speeds of any preset.
  Custom uses the panels.
- **Emission**: from the mesh's volume or from particles; density, temperature, fuel; adding
  instead of raising; `Emit Until` stops emitting after so many seconds.
- **Initial Velocity**: like Blender's flow setting. `Source` is the share of the emitter's own
  motion (or each particle's) given to the smoke, `Normal` a speed along the surface normals,
  and `Velocity` a vector added to both.
- **Particles**: a point simulation off the emitter's surface: rate and first-frame burst,
  speed along the normals and random spread, drag, gravity, lifetime (they fade out), radius.
  Each particle emits a puff of its radius carrying its velocity. The Trail, Explosion and
  Large Plume presets use them.
- **Fire**: fuel burns at `Burn Rate` into heat (`Flame Heat`) and smoke (`Flame Smoke`); the
  flame, as bright as the fuel burning, fades at `Flame Cooling` once the fuel is gone. The
  material shows the `flame` grid as light from dark red through orange to yellow.
- **Domain**: voxel size, domain size, and a Colliders collection (may be animated).
- **Forces**, **Turbulence** (large swirls, then fine swirls that act only where the flow moves,
  so they follow the large ones), **Solver**, **Display**.

The simulation runs in world space. The domain is centred on the emitter on the first frame of
the simulation, its floor a tenth of its height below it, and stays there: give a moving emitter
a domain that covers its path. Only bricks with smoke cost anything. Turning the GPU off runs
Mantaflow, which has no emission, colliders, turbulence or fire, so nothing appears.

Inside (Tab on the modifier's group) it is one row of blocks: **Pyro Preset** makes one
`Settings` bundle; **Pyro Domain** places the box; **Pyro Particles** runs the particles; in the
simulation zone **Pyro Emitter**, **Pyro Colliders** and **Pyro Solver** make one step; **Pyro
Display** shades the result. All of them, and the lower-level **Pyro Source** (mesh), **Pyro
Point Source** (points with radius, `velocity` and `emission`), **Pyro Collider** and
**Turbulence Layer**, are in the node editor's **Add > Pyro** menu for setups built by hand.
Chain Turbulence Layer nodes into Pyro Preset's `Turbulence Layers` (or the Gas Simulation Step
node's) for more layers: they apply in chain order, and a layer masked by Speed reacts to the
ones before it.

`blender/tests/smoke_modifier_tests.py` checks the installed assets, smoke rising, a moving
emitter's trail, a collider holding smoke back, a setting changed mid-simulation, the Campfire
preset's flame, the Explosion preset's spread, initial velocity from the emitter's motion (and
exactly 1.5 m/s from a cube that moved 5 cm in 1/30 s), and Turbulence Layer chains:

```sh
blender -b --factory-startup --python blender/tests/smoke_modifier_tests.py
```

Adding the modifier from the menu cannot be exercised in background mode, where the asset
library does not load; the test checks the metadata the menu reads instead.

## Sparse GPU mode

With the `Sparse GPU` input on, the node runs this repository's sparse brick-grid solver instead
of Mantaflow (see `docs/gpu-sparse-solver.md`):

- Only bricks of 8×8×8 voxels near smoke or heat are simulated. Air elsewhere is open (pressure
  zero), not part of a closed box; the outermost voxel layer of the domain is still a wall.
- Advection-reflection steps, MacCormack advection with a midpoint trace, and six multigrid
  cycles per projection. Advection keeps the totals of density and temperature. A `CFL` of 6
  suits it; the Mantaflow default of 2 works but costs three times as many substeps.
- `Sources` takes a volume whose float grids `density`, `temperature` and `fuel` are emitted
  every step, for example made with Mesh to Density Grid and Store Named Grid. By default they
  are raised to the source's values; `Add Sources` adds them per second instead. An optional
  vector grid `velocity` (world units per second) is set where they emit.
- Fire: while the state has `fuel` or `flame` grids, or the sources have fuel, fuel burns at
  `Burn Rate` per second into heat (`Flame Heat` per unit) and smoke (`Flame Smoke` per unit).
  The `flame` grid is the fuel burning (up to 1); it rises with the hot gas and fades at
  `Flame Cooling` per second once the fuel is gone. The output then carries `fuel` and `flame`
  grids too.
- `Colliders` takes a volume whose float grid `obstacle` marks solid voxels (above 0.5) and an
  optional vector grid `velocity` (world units per second) for their motion. It is read every
  frame, so an animated collider moves; smoke inside voxels that become solid is dropped.
- `Turbulence` adds a swirling noise force (world units per second squared; compare it with
  gravity times the buoyancy), with `Turbulence Scale` (swirl size), `Turbulence Speed`
  (pattern changes per second), `Turbulence Seed`, and `Turbulence Mask`, the density at which it
  reaches full strength (0 stirs all simulated air, which spreads the smoke through the domain).
  Connect Scene Time's Seconds to `Time` so the pattern moves. `Turbulence Layers` takes more
  layers, one point each (attributes `strength`, `scale`, `speed`, `seed`, `mask`: 0 none,
  1 density, 2 temperature, 3 flame, 4 speed; `mask_level`, `invert`), applied in point order
  after the first; Turbulence Layer node groups build it.
- Grids in `Sources` and `Colliders` may use any transform: each active voxel goes to the
  nearest domain voxel, so a voxel size equal to the domain's gives the cleanest result. In
  Mantaflow mode both inputs are ignored with a warning.
- `Pressure Accuracy`, `Multigrid` and `Pressure Iteration Factor` are unused; `Pressure
  Iterations` reports the six cycles and `Active Bricks` the bricks of the last substep.
- GPU work holds a lock, so zones run one at a time on the GPU and never share state.
- Without a usable Vulkan device, or in a build without the kernels, the node runs the solver's
  CPU reference instead (single-threaded and slow) and says so in a warning.

`blender/tests/make_sparse_demo.py` writes a demo file with four Pyro Simulation emitters, one
per preset: a campfire under a plate, an explosion, a large plume and a moving trail:

```sh
blender -b --factory-startup --python blender/tests/make_sparse_demo.py -- pyro_demo.blend
```

Only the blocks with smoke cross OpenVDB: the state is read and written brick by brick (8³, like
OpenVDB's leaves), and sources and colliders go straight into bricks. Up to four GPU solvers
live for the whole process; when a node's input is exactly the state one of them wrote last
(tagged with an id in the density grid's metadata, same grid trees, same resolution), that
solver continues without an upload. The info line says "state kept on the GPU" or "state
uploaded". The output still holds the whole state, so display and baking are unchanged.

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
- Pressure projection by multigrid-preconditioned conjugate gradients with closed-domain
  pressure fixing. MIC preconditioning is available through the `Multigrid` input.
- Finite-value checks and pass-through of the original state on failure.

The engine does not call Mantaflow's `solvePressure`. Both of that function's multigrid modes go
through `solvePressureSystem`, which reads and writes a process-global `std::map` keyed by solver
pointer without synchronization, which is not an acceptable ownership model for independent
Simulation Zone instances that may evaluate concurrently. Instead the engine performs the same
steps itself (right-hand side, Laplace matrix, pressure pinning, preconditioned CG, velocity
correction) with Mantaflow's `GridMg` and `GridCg` classes, and constructs and destroys the
multigrid inside the call. The multigrid hierarchy is therefore rebuilt every substep.

The node reports substeps, sampled maximum velocity, OpenVDB import time, engine solve time,
OpenVDB export time, and the largest pressure iteration count and final residual of any substep.
If a substep stops at its iteration limit before reaching `Pressure Accuracy`, the state still
advances and the node adds a warning. The limit is 100 iterations with multigrid, as in Mantaflow,
and `Pressure Iteration Factor` times the largest resolution with MIC.

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
- No moving adaptive domain and no solver allocation or multigrid hierarchy retained between
  calls.
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

Twelve runtime tests cover zero-time pass-through, stationary fluid interiors, heat buoyancy and
retained pressure, CFL frame completion, atomic substep-limit failure, non-finite input rejection,
independent sequential states, exact resumption from the staggered state, interpretation of the
velocity layout flag, agreement of the two preconditioners with reported convergence, reporting of
an exhausted iteration limit, and the velocity layout reported after a failure.

### In-Blender runtime tests

These need a Blender binary built from the patched pinned source. They build Simulation Zone node
trees from Python, inject dense initial state through a data mesh, and read the grids back for
NumPy comparison.

```sh
blender -b --factory-startup --python blender/tests/gas_node_runtime_tests.py
blender -b --factory-startup --python blender/tests/gas_node_legacy_compare.py -- --horizons 1 5 20 60
blender -b --factory-startup --python blender/tests/gas_node_timings.py -- --res 32 64 96 128 --legacy
```

`gas_node_runtime_tests.py` runs nineteen tests and exits non-zero on failure:

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
| pressure preconditioners and diagnostics | Multigrid and MIC agree to 1e-3 of peak velocity and both report iterations and a converged residual; multigrid needs fewer iterations; a starved MIC solve advances the state and raises the convergence warning. |
| reruns are reproducible | Restarting the zone reproduces the state bit for bit. |
| matches legacy solver | See the next section; asserts relative errors below 1e-3 after 1 frame and 1e-2 after 6. |
| sparse gpu advances and rises | Sparse GPU mode for 15 frames: finite grids, smoke rises, active bricks reported, density total held to 1e-3, no flow through the closed walls. Logs whether the GPU or the CPU fallback ran. |
| sparse gpu obstacle | A plate in `Colliders` above the smoke: no smoke enters it, it shelters the space above, the density total is held. |
| sparse gpu moving collider | A plate animated by the frame number rises through the smoke: no smoke inside it, its faces move at its velocity, the smoke is pushed up. |
| sparse gpu sources | A sphere in `Sources` emits into an empty scene: raised values equal the source exactly, and with `Add Sources` the total matches the rate. |
| sparse gpu turbulence | Turbulence visibly changes the plume (relative L2 above 0.05) and the density total is held. |
| sparse gpu keeps state | The first step uploads, later ones keep the state on the GPU; a second zone with another resolution in between leaves the first zone's result bit for bit the same. |

## Validation performed

Environment: Ubuntu 24.04, 4 cores (Intel Xeon 2.1 GHz), Clang 18.1.3 (the pinned source rejects
GCC older than 14), Blender's precompiled `lib/linux_x64` libraries at the pinned submodule
commit.

- **Full patched Blender build and link.** `blender_lite.cmake` with `WITH_HEADLESS`,
  `WITH_MOD_FLUID`, `WITH_OPENVDB`, `WITH_OPENVDB_BLOSC`, `WITH_TBB`, and `WITH_PYTHON_INSTALL`
  enabled, `Release`. All 4878 build steps completed, the `blender` target linked and installed,
  and the binary reports `Blender 5.4.0 Alpha`. No compiler warnings came from the overlay
  sources. This is a headless build without Cycles or a window system.
- **In-Blender runtime tests.** 13 of 13 pass in that binary. Separately, the `.vdb` files of
  the disk bake were inspected and carry the `staggered` grid class for the velocity grid.
- **Engine tests.** 12 of 12 pass against the pinned Mantaflow library, in a Debug build with GCC
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
| 1 | 8.3e-07 | 7.1e-07 | 7.5e-07 | 1.000 |
| 5 | 5.2e-06 | 4.4e-06 | 3.5e-06 | 1.000 |
| 20 | 3.0e-05 | 2.4e-05 | 1.5e-05 | 1.000 |
| 60 | 2.6e-02 | 2.7e-02 | 1.4e-02 | 1.000 |

The first frames agree to single-precision round-off. The legacy solver keeps its multigrid
hierarchy between steps and the node rebuilds it, and the flow amplifies that difference over
time. With the MIC preconditioner the velocity error is 2e-5 after 5 frames and 1e-3 after 20.
This is agreement on one scene at one resolution for the supported feature subset, with one
solver step per frame; it is not a general equivalence proof.

Two behaviors of the legacy solver matter when comparing:

- With **Adaptive Time Steps disabled** at 30 fps the pinned legacy loop takes **two** full solver
  steps per frame. It passes Mantaflow a time step rounded to six decimals (`0.083333`), which is
  just short of the frame length, so its `while` loop runs again. The comparison therefore keeps
  adaptive time steps enabled.
- MacCormack advection does not conserve mass. In the legacy run total density changes by -4%
  over 20 frames and +31% over 60; the node tracks it (mass ratio 1.00 and 1.02).

### Timings

`gas_node_timings.py --legacy`, 4 threads. The legacy modifier emits once and runs 20 frames; the
node starts from the legacy state with equivalent parameters and the same CFL number (2.0), so
both integrate the same flow. Per-frame columns are medians; totals are over the 20 frames.

| Resolution | Import (ms) | Solve (ms) | Export (ms) | CG iterations | Node total (s) | Node substeps | Legacy total (s) | Legacy steps | Node with MIC total (s) | MIC CG iterations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32³ | 0.8 | 32 | 1.2 | 5 | 0.7 | 20 | 1.0 | 20 | – | – |
| 64³ | 4.9 | 288 | 4.7 | 6 | 5.5 | 34 | 5.6 | 33 | 9.1 | 60 |
| 96³ | 17.5 | 976 | 14.2 | 6 | 18.5 | 36 | 17.2 | 36 | 38.8 | 48 |
| 128³ | 38.8 | 3312 | 33.1 | 6 | 59.7 | 51 | 57.5 | 51 | 150.5 | 61 |

- OpenVDB import and export together are about 2% of a frame.
- With multigrid the node takes the same number of substeps as the legacy solver and about the
  same time: 0.7 times the legacy total at 32³ and 0.98 to 1.07 times at 64³ to 128³.
- With MIC, the previous default, it took 1.7 to 2.7 times the legacy total on the same flow.

The node is therefore at parity with the legacy solver on this scene, not faster. The two frame
times also do not contain the same work: the legacy frame includes flow-object processing, cache
writes, and a shadow grid; the node frame includes OpenVDB conversion and rebuilding the solver,
its grids, and the multigrid hierarchy on every call. These are measurements of one small scene
on one machine.

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
  domains. The matched timing runs take the same number of substeps as the legacy solver, but
  their grids were not compared.
- Platforms other than Linux x86-64, and compilers other than Clang 18 (GCC 13 for the engine
  harness).
