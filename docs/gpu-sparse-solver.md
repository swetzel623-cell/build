# GPU sparse gas solver: research and first increment

This document records what was researched, what was decided and why, what exists in the
repository now, and what has and has not been verified.

## Goal

A smoke solver whose cost follows the smoke instead of the bounding box, running on the GPU,
usable from the Geometry Nodes gas node. The dense Mantaflow-based node in `blender/` stays the
correctness reference for the physics.

## Findings

**Sparse layouts used for simulation on GPUs**

- NanoVDB stores a VDB tree without pointers so it can be copied to a GPU unchanged. Until
  recently it assumed static topology, which rules it out as a simulation container. Index grids,
  introduced with fVDB, separate topology from data: the tree maps a voxel to an index into flat
  side arrays. A 2026 NVIDIA paper adds GPU topology updates (dilation, merge, prune) and a
  "voxel-sequential" access scheme for stencils, and links to the OpenVDB repository for code.
  [NanoVDB](https://research.nvidia.com/labs/prl/publication/nanovdb),
  [Dynamic GPU Sparse Grids](https://research.nvidia.com/labs/prl/publication/sifakis2026dynamic/)
- GVDB Voxels ran FLIP fluids on a sparse brick hierarchy with a matrix-free conjugate gradient
  pressure solve on the GPU. [Fast Fluid Simulations with Sparse Volumes on the GPU](https://diglib.eg.org/handle/10.1111/cgf13350)
- SPGrid pairs a sparse paged grid with an adaptive multigrid-preconditioned CG solver, on the
  CPU. DCGrid positions itself as combining GPU execution with a multigrid hierarchy.
  [SPGrid](https://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf),
  [DCGrid](https://graphics.tudelft.nl/~klaus/papers/DCGrid.pdf)

**How production sparse smoke solvers treat the inactive region**

- Houdini's sparse pyro keeps 16³ tiles active where reference fields (density, flame) are
  non-zero, pads that set, and rebuilds it every step. Inactive space is treated as vacuum that
  smoke can move into, and two separate puffs do not see each other until their active regions
  merge. Its pressure projection uses unsmoothed aggregation multigrid with internal boundaries
  treated as open. [Sparse pyro](https://www.sidefx.com/docs/houdini/pyro/sparsity.html),
  [Sparse Smoke Simulations in Houdini](https://history.siggraph.org/?p=41379)
- Axiom is a commercial GPU sparse solver for Houdini with OpenCL, CUDA, and Metal backends and
  a CPU fallback when a simulation exceeds GPU memory.
  [Axiom](https://www.cgchannel.com/2024/09/matt-puchala-releases-axiom-3-for-houdini/)

**What Blender offers**

- Checked in the pinned source: Blender's GPU module has compute dispatch
  (`GPU_compute_dispatch`), storage buffers with read-back (`GPU_storagebuf_read`), and is used
  for compute by EEVEE, the draw manager, and GPU subdivision. Its shaders are GLSL.
- Vulkan became an officially supported backend in Blender 4.5.
  [Projects update](https://code.blender.org/2025/05/projects-update-q2-2025/)
- The pinned Blender libraries bundle NanoVDB 32.9, including `PNanoVDB.h`, the portable
  C/GLSL/HLSL accessor header, and a `cuda` directory.
- Geometry Nodes evaluate on worker threads. Nothing found shows a geometry node using the GPU
  module, so how a node obtains a GPU context is unresolved (see open questions).

## Decisions

1. **Portable compute shaders, not CUDA.** The reference implementations with GPU topology
   updates are CUDA. Blender cannot make a core node depend on one vendor, and its own GPU module
   is GLSL on Vulkan, Metal, and OpenGL. The kernels are therefore plain GLSL compute shaders over
   storage buffers, which is what Blender's GPU module consumes.
2. **Fixed-size bricks with a flat brick table, not a VDB tree, for now.** A dense table over the
   brick grid maps a brick to a storage slot; channels are flat float arrays indexed by
   `slot * brick_volume + local`. A lookup is one table read, there is no tree traversal in the
   shaders, and the same arrays are uploaded unchanged. With 8³ bricks a brick is exactly an
   OpenVDB leaf node, so import and export can become per-leaf copies. The cost is the dense
   table: 4 bytes per brick of the bounding box, so 16 MB for a 1024³ domain. A NanoVDB index
   grid would remove that and is the likely long-term container; the kernels only depend on
   "coordinates to index", so the swap is contained.
3. **Topology on the CPU.** Deciding which bricks are active is a small per-brick computation. It
   stays on the CPU until profiling says otherwise.
4. **Open air outside the active region.** Like Houdini, inactive space has pressure zero and
   smoke may flow into it. The domain's outer voxel layer remains a closed wall, as in Mantaflow.
   This is a different boundary condition from the dense closed-box solve, so the sparse and
   dense results are not expected to be equal. Velocity alone does not keep a brick active: the
   pressure solve induces a small velocity everywhere, so it would activate the whole domain.
5. **A CPU reference with the same layout comes first.** Without it GPU results cannot be
   checked, and Blender needs a CPU fallback in any case (background renders, no GPU). Every CPU
   kernel is a gather over stored voxels and corresponds to one compute shader.

## What exists

| Piece | Location |
| --- | --- |
| Topology, state, dense conversion, CPU reference kernels and step | `include/manta_improve/sparse_gas.hpp`, `src/sparse_gas.cpp` |
| GLSL compute kernels: advect, buoyancy, walls, divergence, jacobi, gradient | `gpu/shaders/` |
| GLSL kernels for the active-region test, cell classification, and moving channels onto a new topology | `gpu/shaders/activity.comp`, `classify.comp`, `remap.comp` |
| Vulkan compute runner: device-memory buffers, batched recording, GPU timestamps | `gpu/vulkan_compute.*` |
| Driver that keeps the state on the device and runs each step | `gpu/sparse_gas_gpu.*` |
| CPU tests | `tests/sparse_gas_tests.cpp` |
| GPU-versus-CPU test and timing | `tests/sparse_gas_gpu_tests.cpp` |

**Staggered storage and the ghost layer.** A voxel stores the velocity of its three lower faces.
A cell can only be solved for if the cells above it along each axis are stored too, or are walls.
The remaining stored cells, a one-voxel layer on the upper sides of the active set, are classified
as open air and hold the faces between fluid and open air. Cell types are computed once per
topology and uploaded.

**The step.** Rebuild topology from the current values, advect density, temperature, and the
three velocity components, add heat and density buoyancy, zero wall faces, solve pressure, subtract
its gradient. Buoyancy and the wall rule follow Mantaflow's operators.

**Advection.** MacCormack by default: a forward and a backward semi-Lagrangian trace, the error
estimate added back for values inside the fluid, and the result clamped to the eight values the
forward trace interpolated. Where that trace reads a voxel without storage, the first-order value
is kept. This is the scheme the Mantaflow node uses, but not the same code; the two have not been
compared.

**Frames.** `advance()` takes a frame in substeps of at most `cfl` voxels of travel, measured by
the largest velocity component at the start of each substep, and fails past a substep limit. On
the device that speed is gathered per brick by the activity kernel and comes back with the
activity flags, so substepping adds no transfer.

**The device driver.** The state stays in device memory between steps; `upload()` and
`download()` are the only full transfers. A step is two submissions. The first runs the activity
kernel and downloads one flag per brick, 4 bytes per active brick. The CPU builds the next brick
table from those flags with the same code as the reference, but does not classify cells. The
second submission uploads the brick table and a slot map if the topology changed, moves the six
state channels onto it with the remap kernel, classifies every stored voxel with the classify
kernel, and runs the whole step. With profiling on, every command is followed by a timestamp so
the GPU time of each kernel can be reported. Every command is followed by a full memory barrier.
Descriptor sets are cached per combination of buffers, and parameters are push constants. Channel
buffers are allocated with 50% headroom and replaced when the active set outgrows them. GPU time
is measured with timestamp queries around each submission.

**Pressure.** The system is the 7-point Poisson matrix over fluid cells with Neumann conditions at
walls and Dirichlet zero at open cells. The CPU reference has Jacobi-preconditioned CG, plain
Jacobi sweeps, and multigrid. The GPU path has Jacobi sweeps and multigrid: neither needs a
reduction, so the GPU result can be compared with the reference operation for operation.

**Multigrid.** Coarse levels aggregate 2×2×2 cells with piecewise-constant transfer. The Galerkin
coarse operator of a 7-point stencil is then again a 7-point stencil, a diagonal and one weight
per lower face, so every level shares one smoother and one residual, and nothing special is
needed at walls or at the edge of the active region. Levels that still have several cells per
brick are stored per brick slot like the voxels; from one cell per brick on, levels are small
dense grids over the brick grid, down to at most 2 cells per side. Smoothing is red-black
Gauss-Seidel, two sweeps before and after each coarse correction and eight on the last level.
Plain aggregation underestimates the coarse correction, so it is scaled by 1.4, and the two
finest coarse levels are visited twice per cycle (a W-cycle near the top, V-cycles below). These
values were chosen by measurement: the residual then falls by about 0.31 per cycle on sparse and
closed domains from 32 to 128 voxels wide, against 0.55 for plain V-cycles; larger scaling
factors diverge. On the device, the coarse operators are rebuilt by kernels whenever the topology
changes, and the voxel level reads a packed word per voxel (cell type and which neighbors are
fluid or not solid) instead of looking up neighbor types.

## Verified

On Linux x86-64, GCC 13, Release.

CPU reference, 7 tests in `sparse_gas_tests`:

- Brick addressing, cell classification, dilation, overhanging bricks.
- Dense round trip is exact.
- CG projection leaves less than 1e-3 divergence, both with all bricks active and with only the
  bricks around a blob active; no flow through closed walls.
- With every brick active, brick sizes 4, 5, 8, and 16 give bit-identical results after 8 steps.
- A buoyant blob rises; without forces nothing moves; an empty state stores nothing.
- Sparse against all-bricks-active on 48×48×96 over 30 steps: peak 176 of 432 bricks (41%),
  plume rise 17.6 against 17.2 voxels, total density within 2%, density relative L2 difference
  0.18.
- 3000 Jacobi sweeps agree with CG to 3e-7 on a 16³ box.
- Multigrid reduces the residual by at least a factor of 0.31 per cycle and by 3e-4 in six
  cycles, on a sparse and on a closed 48×48×64 domain; six steps of a plume solved with ten cycles
  agree with CG to 8e-7 in velocity.

- MacCormack keeps a plume's peak density at 0.97 after 25 steps where first-order advection
  leaves 0.72, in fewer active bricks, and stays inside the initial value range.
- `advance()` subdivides frames at CFL 1 and fails when the substep limit is exhausted.

GPU MacCormack and substepping on llvmpipe: every channel equals the CPU reference bit for bit,
with the same substep counts, including a run forced to several substeps per frame.

GPU multigrid on llvmpipe: every channel equals the CPU reference bit for bit (32×32×48 over 8
steps and 48×48×72 over 10). After six cycles the remaining divergence relative to peak velocity
is about 5e-6, against 2e-3 after 60 Jacobi sweeps on the same flow.

GPU kernels, `sparse_gas_gpu_tests`: on llvmpipe, a software Vulkan implementation, every
channel equals the CPU reference bit for bit after each of 8 steps of a rising plume (32×32×48,
60 Jacobi sweeps per step, topology growing from 48 to 64 bricks). This verifies the shaders, the
buffer layout, and the driver.

**First run on a real GPU.** `sparse_gas_gpu_tests 64 20` on an NVIDIA GeForce RTX 3070
(Windows, the CI-built binary), 64×64×96 domain, 60 Jacobi sweeps per step:

- Every channel matched the CPU reference within the test tolerance on all 20 steps; the largest
  difference was 7.3e-6 (scaled by peak velocity), against 72 voxels per second peak velocity. The
  results are not bit-identical as on llvmpipe; the difference stays at round-off level and does
  not grow over the run.
- The active set grew from 64 to 244 of the domain's 768 bricks (about 30%) as the plume rose.
- 19.1 ms per step on the device path against 482 ms per step for the CPU reference.

That timing ratio is **not** a measure of the solver's speed. The CPU reference is deliberately
simple and single-threaded, and is not the Mantaflow engine. That run used the first driver, which
uploaded and downloaded the whole state every step and submitted and awaited about 70 dispatches
separately. The resident driver above replaces it; its timings on a real GPU are still to be
measured with `sparse_gas_gpu_tests <resolution> <steps> --bench`.

**Resident driver on the RTX 3070** (same machine, CI-built binary, 60 Jacobi sweeps per step):

- Comparison mode, 64×64×96, 20 steps: passes with the same round-off-level differences as
  before (largest 7.3e-6). 7.3 ms per step including a full download every step for the
  comparison, against 19.1 ms for the first driver.
- `--bench`, 40 steps each, median per step:

| Domain | Peak active bricks | Wall | GPU | Topology (CPU) |
| --- | --- | --- | --- | --- |
| 64×64×96 | 452 of 768 (59%) | 8.9 ms | 3.9 ms | 2.5 ms |
| 128×128×192 | 1896 of 6144 (31%) | 18.3 ms | 8.0 ms | 8.1 ms |
| 192×192×288 | 4517 of 20736 (22%) | 45.3 ms | 23.8 ms | 17.6 ms |

  The last 192 step, with 2.3 million stored voxels, took 114 ms wall and 62 ms GPU.

What this shows:

- **The CPU topology rebuild is now as large as the GPU work.** It classifies every stored voxel
  on the CPU, and the per-voxel cell types are then uploaded. The topology changed in 24 to 38 of
  40 steps, so this cost is paid almost every step.
- **The GPU work is several times above what memory bandwidth allows.** A rough estimate for one
  Jacobi sweep over 2.3 million voxels is about 0.2 ms at the 3070's bandwidth, about 12 ms for
  60 sweeps, against 62 ms measured for the whole step. Neighbor lookups through the brick table
  and a full barrier after every dispatch are the likely causes; per-kernel timing is needed to
  confirm.
- **The active region grows quickly.** First-order advection diffuses smoke into neighboring
  bricks, and the activity threshold is low. Over 40 steps the plume at 192 reached 22% of the
  box.
- These figures are not comparable with the Mantaflow node's timings: different machine,
  different numerics (60 unconverged Jacobi sweeps against a converged multigrid solve,
  first-order against MacCormack advection), and a different scene.

**After moving cell classification to the GPU and removing duplicate lookups** (RTX 3070, same
runs):

- Comparison mode still passes with the same differences (largest 7.3e-6), and the GPU's cell
  types equal the CPU reference's on every step. 3.2 ms per step including the per-step download,
  against 7.3 ms before.
- `--bench`, median per step:

| Domain | Wall | GPU | Topology (CPU) | Before: wall / GPU / topology |
| --- | --- | --- | --- | --- |
| 64×64×96 | 3.2 ms | 1.8 ms | 0.02 ms | 8.9 / 3.9 / 2.5 ms |
| 128×128×192 | 7.3 ms | 4.9 ms | 0.05 ms | 18.3 / 8.0 / 8.1 ms |
| 192×192×288 | 15.4 ms | 13.1 ms | 0.12 ms | 45.3 / 23.8 / 17.6 ms |

  The last 192 step, with 2.3 million stored voxels, took 29.5 ms wall and 27.1 ms GPU, against
  114 ms and 62 ms before.
- `--profile` at 192×192×288, average GPU time per step over 40 steps: Jacobi sweeps 9.2 ms
  (69%), advection 3.1 ms (23%), buoyancy 0.33 ms, remap 0.16 ms, gradient 0.13 ms, walls
  0.11 ms, classify 0.09 ms, divergence 0.07 ms, activity 0.04 ms, transfers under 0.04 ms. At
  128×128×192 the split is 74% Jacobi and 18% advection.

What this shows:

- **CPU overhead is no longer the problem.** Wall time is now within a few milliseconds of GPU
  time; the remaining gap is largest in steps where buffers grow and are reallocated.
- **The pressure solve dominates, and it is the weakest part numerically too.** 60 Jacobi sweeps
  are about 70% of GPU time and do not converge the pressure at these resolutions. A rough
  estimate puts the sweeps at about 30% of the 3070's memory bandwidth, so kernel tuning could
  gain perhaps two to three times. A solver that converges in far fewer passes (multigrid) is the
  larger gain and also fixes the accuracy problem.

**Multigrid on the RTX 3070, first version** (one dispatch per sweep on every level):

- Comparison mode, 64×64×96, 20 steps: passes; largest difference 9.1e-6.
- 192×192×288, 40 steps, six cycles against 60 Jacobi sweeps:

| | Multigrid | Jacobi |
| --- | --- | --- |
| Divergence left, relative to peak velocity | 4.5e-5 | 4.2e-2 |
| Peak active bricks | 1111 | 4517 |
| Median step, wall / GPU | 19.8 / 15.0 ms | 18.3 / 14.2 ms |
| Last step, wall | 32.7 ms at 1099 bricks | 38.2 ms at 4517 bricks |

  With the pressure converged the plume stays compact: a quarter of the bricks that the
  unconverged Jacobi run activates. Per brick, though, this version was several times slower.
- `--profile` at 192: smoothing on coarse levels 11.1 ms (56%), on the voxel level 3.3 ms (17%),
  advection 1.3 ms. The coarse levels hold about a seventh of the cells, so nearly all of their
  time is overhead: about 1,800 small dispatches per step at roughly 6 µs each.

In response, all dense levels now live back to back in shared buffers and one dispatch can run
a V-cycle over several of them: a single work group of 128 invocations shares each pass over a
level and meets at a barrier between passes.

Handing it **all** dense levels was a mistake that llvmpipe hid. On the RTX 3070 that dispatch
took 28 ms per step at 192 (24 calls of about 1.2 ms), against the 11 ms it replaced, and the
median step rose from 19.8 to 29.2 ms: one work group is slow on the first dense level's 20,736
cells. Results were still correct (comparison passed, same differences). The driver now hands
over only the trailing levels of at most 512 cells and gives larger levels their own dispatches
again; `--tiny-limit` changes the limit and `--w-depth` the W-cycle depth for experiments.

**Variants measured on the RTX 3070**, 192×192×288, 40 steps, six cycles, median per step:

| Variant | Wall | GPU | Divergence left |
| --- | --- | --- | --- |
| Every level its own dispatches (limit 0) | 18.7 ms | 13.8 ms | 4.5e-5 |
| Single dispatch for levels up to 64 cells | 17.4 ms | 13.4 ms | 4.5e-5 |
| Up to 512 cells (default) | 16.8 ms | 13.5 ms | 4.5e-5 |
| Up to 4096 cells | 17.2 ms | 13.8 ms | 4.5e-5 |
| Up to 512 cells, W-cycle on one level only (`--w-depth 1`) | 14.5 ms | 11.1 ms | 7.3e-5 |
| 60 Jacobi sweeps, for reference | 19.3 ms | 14.4 ms | 4.2e-2 |

The comparison run passed again (largest difference 9.1e-6). What the numbers say:

- **The size limit barely matters**; the differences between the first four rows are within the
  run-to-run noise. Individual steps at the same brick count vary by a factor of two (for
  example 5 and 11 ms GPU), so these medians are rough.
- **The solve is bound by the number of dispatches, not by the work.** A 128×128×192 run with
  679 bricks takes about as long (13.1 ms GPU) as the 192×192×288 run with 1111 bricks
  (12.9 ms). In the 192 profile the voxel-level smoothing, where most of the cells are, is
  2.4 ms; smoothing on the coarse levels, with a seventh of the cells, is 4.2 ms.
- Visiting fewer levels twice (`--w-depth 1`) removes about 40% of the dispatches and is the
  only variant that is clearly faster, for a somewhat larger remaining divergence.
- Multigrid now takes about the same time per step as 60 Jacobi sweeps on this scene, leaves
  about 1000 times less divergence, and keeps the plume in a quarter of the bricks.

Because the cost is per dispatch, it should weigh less on larger problems; that is untested.

With the resident driver, on llvmpipe, every channel still equals the CPU reference bit for bit:
32×32×48 over 12 steps, and 48×48×72 over 14 steps, where the active set grows from 64 to 160
bricks and the channel buffers are replaced once.

## Not verified, and known gaps

- **One real GPU, one scene.** Only the RTX 3070 run above. No AMD, Intel, or Apple device.
- **No comparison with the Mantaflow node's speed.** The solvers also differ in physics (below),
  so a fair comparison needs matched numerics first. The driver is also not written for speed: buffers
  are host-visible, every dispatch is submitted and waited for separately, and the whole state is
  uploaded and downloaded every step.
- **The driver is not tuned.** Every command is followed by a full barrier, kernels use 64-thread
  groups over a flat voxel range, neighbors are found through the brick table, and staging buffers
  are allocated per transfer. None of this has been profiled.
- **Not compared with the Mantaflow engine.** The scheme now matches in kind (MacCormack
  advection, CFL substeps, converged pressure), but nothing has been checked against it.
- **Multigrid is dispatch-bound at these sizes.** About 1,000 dispatches per step, most of them
  on coarse levels with little work. Larger problems, where the work should dominate, have not
  been measured.
- **Aggregation multigrid converges slowly for a multigrid method** (0.31 per cycle). As a
  preconditioner for conjugate gradients it would do better, but that needs reductions on the
  device.
- **The CPU reference is deliberately simple and slow** (single-threaded, a table lookup per
  neighbor read).
- **Not connected to Blender.**

## Open questions

- **GPU context from Geometry Nodes.** Node evaluation happens on worker threads, possibly in a
  background render with no window. Using Blender's GPU module there needs a context that is
  valid on that thread. Blender has a system GPU context that other background work uses; whether
  a node may use it, and how it is locked, needs to be established in the Blender source before
  any integration design is fixed.
- **Where the state lives between frames.** Simulation Zone state is CPU geometry. Keeping grids
  on the GPU across frames conflicts with that; downloading every frame costs transfer time.
- **Sparse OpenVDB interchange.** The node currently exchanges dense-topology grids.

## Next steps

1. Measure multigrid at larger sizes, where dispatch overhead should matter less, before
   tuning it further.
2. Compare with the Mantaflow engine on a case where the active region covers the domain.
3. Resolve the Blender GPU-context question, then port the driver to Blender's GPU module behind
   the existing node, with the CPU reference as fallback.
