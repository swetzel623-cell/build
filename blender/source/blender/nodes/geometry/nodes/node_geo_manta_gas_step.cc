/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "node_geometry_util.hh"

#include "BKE_geometry_set.hh"
#include "BKE_lib_id.hh"
#include "BKE_volume.hh"
#include "BKE_volume_grid.hh"
#include "BKE_volume_openvdb.hh"

#include "DNA_pointcloud_types.h"

#include "BLT_translation.hh"

#if defined(WITH_FLUID) && defined(WITH_OPENVDB)
#  include "manta_gas_layout.h"
#  include "manta_gas_step_API.h"

#  include "manta_improve/sparse_gas.hpp"
#  ifdef WITH_SPARSE_GAS_GPU
#    include "sparse_gas_gpu.hpp"
#  endif

#  include "openvdb/openvdb.h"
#  include "openvdb/tools/GridTransformer.h"
#  include "openvdb/tools/Interpolation.h"

#  include "BLI_task.hh"

#  include <algorithm>
#  include <array>
#  include <chrono>
#  include <cmath>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#  include <memory>
#  include <mutex>
#  include <optional>
#  include <random>
#  include <string>
#  include <type_traits>
#  include <vector>
#endif

namespace blender::nodes::node_geo_manta_gas_step_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("State"_ustr)
      .description(
          "Volume containing density, temperature, velocity, and pressure grids, and fuel and "
          "flame grids for fire. Missing grids start at zero. A velocity grid that is not tagged as staggered is treated as "
          "cell-centered and resampled to faces");
  b.add_output<decl::Geometry>("State"_ustr)
      .description(
          "Serializable next simulation state: density, temperature, velocity and pressure "
          "grids, and fuel and flame grids while there is fire. Velocity is the solver's "
          "staggered (face-centered) field");

  b.add_input<decl::Float>("Delta Time"_ustr)
      .default_value(1.0f / 30.0f)
      .min(0.0f)
      .subtype(PROP_TIME_ABSOLUTE)
      .description("Time to advance in seconds, normally supplied by the Simulation Input node");

  b.add_input<decl::Int>("Resolution X"_ustr).default_value(32).min(4).max(1024);
  b.add_input<decl::Int>("Resolution Y"_ustr).default_value(32).min(4).max(1024);
  b.add_input<decl::Int>("Resolution Z"_ustr).default_value(32).min(4).max(1024);
  b.add_input<decl::Vector>("Origin"_ustr)
      .default_value(float3(-1.0f))
      .description("World-space position of grid index (0, 0, 0)");
  b.add_input<decl::Float>("Voxel Size"_ustr)
      .default_value(0.0625f)
      .min(1.0e-6f)
      .description("Uniform world-space voxel size; input grids must use this exact transform");

  b.add_input<decl::Vector>("Gravity"_ustr)
      .default_value(float3(0.0f, 0.0f, -9.81f))
      .subtype(PROP_ACCELERATION)
      .description("World-space acceleration used by the buoyancy operators");
  b.add_input<decl::Float>("Density Buoyancy"_ustr)
      .default_value(0.0f)
      .min(-100.0f)
      .max(100.0f)
      .description("Buoyancy multiplier for density; positive values rise against gravity");
  b.add_input<decl::Float>("Heat Buoyancy"_ustr)
      .default_value(1.0f)
      .min(-100.0f)
      .max(100.0f)
      .description("Buoyancy multiplier for temperature; positive values rise against gravity");

  b.add_input<decl::Float>("CFL"_ustr)
      .default_value(2.0f)
      .min(0.001f)
      .max(100.0f)
      .description("Maximum velocity times timestep in grid cells");
  b.add_input<decl::Int>("Max Substeps"_ustr)
      .default_value(32)
      .min(1)
      .max(1024)
      .description("Fail without changing the state if the frame cannot finish within this limit");
  b.add_input<decl::Float>("Pressure Accuracy"_ustr)
      .default_value(1.0e-3f)
      .min(1.0e-8f)
      .max(1.0f)
      .description("Residual accuracy requested from Mantaflow's MIC-preconditioned solver");
  b.add_input<decl::Bool>("Multigrid"_ustr)
      .default_value(true)
      .description(
          "Precondition the pressure solve with a multigrid owned by this node evaluation. "
          "Disable to use the MIC preconditioner");
  b.add_input<decl::Float>("Pressure Iteration Factor"_ustr)
      .default_value(1.5f)
      .min(0.1f)
      .max(20.0f)
      .description(
          "Multiplier for the resolution-dependent pressure iteration limit of the MIC "
          "preconditioner. Unused with Multigrid");
  b.add_input<decl::Bool>("Sparse GPU"_ustr)
      .default_value(false)
      .description(
          "Use the sparse brick-grid solver on the GPU (Vulkan) instead of Mantaflow. Only "
          "bricks near smoke are simulated; air elsewhere is open. A CFL of 6 suits this "
          "solver. Without a usable GPU it runs its slow CPU reference and warns");
  b.add_input<decl::Geometry>("Sources"_ustr)
      .description(
          "Sparse GPU only. Volume whose float grids 'density', 'temperature' and 'fuel' are "
          "emitted every step, for example from Mesh to Density Grid. Fuel burns into flame, "
          "heat and smoke. A float grid 'divergence' (1/s) makes the gas expand there, as in an "
          "explosion. An optional vector grid 'velocity' (world units per second) is set "
          "where they emit. Grids may use any "
          "transform; each active voxel goes to the nearest domain voxel");
  b.add_input<decl::Bool>("Add Sources"_ustr)
      .default_value(false)
      .description(
          "Add the sources' values per second instead of raising density, temperature and fuel "
          "to them");
  b.add_input<decl::Geometry>("Colliders"_ustr)
      .description(
          "Sparse GPU only. Volume whose float grid 'obstacle' marks solid voxels (value above "
          "0.5), for example from Mesh to Density Grid, evaluated every frame so colliders may "
          "move. An optional vector grid 'velocity' (world units per second) moves the "
          "collider's faces");
  b.add_input<decl::Float>("Time"_ustr)
      .default_value(0.0f)
      .subtype(PROP_TIME_ABSOLUTE)
      .description("Sparse GPU only. Scene time in seconds, which the turbulence pattern follows. "
                   "Connect Scene Time's Seconds");
  b.add_input<decl::Float>("Turbulence"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_ACCELERATION)
      .description("Sparse GPU only. Strength of a swirling noise force in world units per "
                   "second squared; 0 turns it off. Compare it with gravity times the buoyancy");
  b.add_input<decl::Float>("Turbulence Scale"_ustr)
      .default_value(0.5f)
      .min(1.0e-4f)
      .subtype(PROP_DISTANCE)
      .description("Size of the turbulence swirls in world units");
  b.add_input<decl::Float>("Turbulence Speed"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("How many times per second the turbulence pattern changes; 0 freezes it");
  b.add_input<decl::Int>("Turbulence Seed"_ustr).default_value(0).min(0);
  b.add_input<decl::Float>("Turbulence Mask"_ustr)
      .default_value(0.1f)
      .min(0.0f)
      .description("Density at which turbulence reaches full strength; it fades out in thinner "
                   "smoke. 0 stirs all simulated air, which spreads the smoke through the "
                   "domain");
  b.add_input<decl::Geometry>("Turbulence Layers"_ustr)
      .description(
          "Sparse GPU only. More turbulence, one point per layer, applied in point order after "
          "the turbulence above, so a layer masked by speed reacts to the layers before it. "
          "Point attributes: 'strength', 'scale', 'speed', 'mask_level' (floats, world units), "
          "'seed' and 'mask' (integers; mask 0 none, 1 density, 2 temperature, 3 flame, "
          "4 speed) and 'invert' (boolean). Turbulence Layer node groups build it");
  b.add_input<decl::Float>("Burn Rate"_ustr)
      .default_value(0.75f)
      .min(0.0f)
      .description("Sparse GPU only. Fuel burnt per second where there is any. Higher burns "
                   "faster, in shorter flames");
  b.add_input<decl::Float>("Flame Heat"_ustr)
      .default_value(4.0f)
      .min(0.0f)
      .description("Temperature added per unit of fuel burnt, which makes the fire rise");
  b.add_input<decl::Float>("Flame Smoke"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("Smoke density left per unit of fuel burnt");
  b.add_input<decl::Float>("Flame Cooling"_ustr)
      .default_value(4.0f)
      .min(0.0f)
      .description("Flame lost per second once its fuel is gone. Lower leaves taller flames");
  b.add_input<decl::Float>("Gas Release"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description("Sparse GPU only. How much burning fuel expands: the volume grows by this "
                   "much per unit of fuel burnt, pushing the fire outward like an explosion");
  b.add_input<decl::Bool>("Open Borders"_ustr)
      .default_value(false)
      .description("Sparse GPU only. Let smoke and heat leave through the sides and top of the "
                   "domain; the floor stays solid. Closed, smoke gathers in the box until it "
                   "fills it, and every filled brick costs time");
  b.add_input<decl::Float>("Smoke Fade"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description("Sparse GPU only. Fraction of the smoke lost per second (exponentially); thin "
                   "smoke then disappears and stops being simulated. 0 keeps it all");
  b.add_input<decl::Float>("Heat Fade"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description("Sparse GPU only. Fraction of the heat lost per second, so hot gas cools and "
                   "stops accelerating upward. 0 keeps it all");
  b.add_input<decl::Int>("Pressure Cycles"_ustr)
      .default_value(6)
      .min(1)
      .max(16)
      .description("Sparse GPU only. Multigrid cycles per pressure solve. Fewer is faster: 4 cuts "
                   "the solve by a third with differences of under 1%; 3 or fewer start to show");
  b.add_input<decl::Float>("Smoke Threshold"_ustr)
      .default_value(1.0e-5f)
      .min(0.0f)
      .max(1.0f)
      .description("Sparse GPU only. Smoke, heat, fuel or flame below this is treated as empty, "
                   "and bricks holding nothing more are not simulated. Higher drops faint smoke "
                   "sooner (0.001 is hardly visible; 0.01 loses the faintest wisps)");
  b.add_input<decl::Bool>("Directional Margin"_ustr)
      .default_value(false)
      .description("Sparse GPU only. Pad the simulated region only on the sides smoke can reach "
                   "within a step, instead of all around: 10-20% fewer bricks while a plume "
                   "grows. The fine swirls differ slightly");
  b.add_input<decl::Bool>("Fast Turbulence"_ustr)
      .default_value(false)
      .description("Sparse GPU only. Apply the turbulence once per step instead of twice; half "
                   "its cost. The fine detail changes");
  b.add_input<decl::Bool>("Half Precision"_ustr)
      .default_value(false)
      .description("Sparse GPU only. Store the simulation on the GPU in 16-bit numbers: half the "
                   "GPU memory and less memory traffic, with differences like one pressure cycle "
                   "fewer. More than about 4 Pressure Cycles then stop helping. Ignored on GPUs "
                   "without 16-bit storage. Changing it restarts the simulation from its cache");
  b.add_input<decl::Bool>("Store Velocity"_ustr)
      .default_value(true)
      .description("Sparse GPU only. Keep the velocity grid in every simulated frame. Off, the "
                   "frame cache needs about half the memory, but render motion blur of the "
                   "volume has no velocity");
  b.add_input<decl::Bool>("Batch GPU Work"_ustr)
      .default_value(true)
      .description("Sparse GPU only. Start the GPU on recorded work while the rest of the step is "
                   "still recorded. Same results; turn off only to compare timings");

  b.add_output<decl::Int>("Substeps"_ustr).description("Number of internal solver steps executed");
  b.add_output<decl::Float>("Max Velocity"_ustr)
      .description("Largest sampled solver velocity in world units per second");
  b.add_output<decl::Float>("Import Time"_ustr)
      .description("OpenVDB-to-dense conversion time in milliseconds");
  b.add_output<decl::Float>("Solve Time"_ustr)
      .description("Mantaflow allocation and simulation time in milliseconds");
  b.add_output<decl::Float>("Export Time"_ustr)
      .description("Dense-to-OpenVDB conversion time in milliseconds");
  b.add_output<decl::Int>("Pressure Iterations"_ustr)
      .description("Largest pressure solver iteration count of any substep");
  b.add_output<decl::Float>("Pressure Residual"_ustr)
      .description("Largest final pressure solver residual of any substep");
  b.add_output<decl::Int>("Active Bricks"_ustr)
      .description("Sparse GPU only: bricks of 8x8x8 voxels simulated in the last substep");
}

#if defined(WITH_FLUID) && defined(WITH_OPENVDB)

static void set_diagnostics(GeoNodeExecParams &params,
                            const int substeps,
                            const float max_velocity,
                            const float import_ms,
                            const float solve_ms,
                            const float export_ms,
                            const int pressure_iterations = 0,
                            const float pressure_residual = 0.0f,
                            const int active_bricks = 0)
{
  params.set_output("Active Bricks"_ustr, active_bricks);
  params.set_output("Pressure Iterations"_ustr, pressure_iterations);
  params.set_output("Pressure Residual"_ustr, pressure_residual);
  params.set_output("Substeps"_ustr, substeps);
  params.set_output("Max Velocity"_ustr, max_velocity);
  params.set_output("Import Time"_ustr, import_ms);
  params.set_output("Solve Time"_ustr, solve_ms);
  params.set_output("Export Time"_ustr, export_ms);
}

static void fail(GeoNodeExecParams &params, GeometrySet &state, const std::string &message)
{
  params.error_message_add(NodeWarningType::Error, message);
  params.set_output("State"_ustr, std::move(state));
  set_diagnostics(params, 0, 0.0f, 0.0f, 0.0f, 0.0f);
}

static bool active_values_fit_domain(const openvdb::GridBase &grid,
                                     const int3 &resolution,
                                     const char *name,
                                     std::string &error)
{
  if (grid.activeVoxelCount() == 0) {
    return true;
  }
  const openvdb::CoordBBox bounds = grid.evalActiveVoxelBoundingBox();
  const openvdb::Coord &min = bounds.min();
  const openvdb::Coord &max = bounds.max();
  if (min.x() < 0 || min.y() < 0 || min.z() < 0 || max.x() >= resolution.x ||
      max.y() >= resolution.y || max.z() >= resolution.z)
  {
    error = std::string("Grid '") + name + "' has active voxels outside the simulation domain.";
    return false;
  }
  return true;
}

static bool import_scalar_grid(const Volume *volume,
                               const char *name,
                               const openvdb::math::Transform &expected_transform,
                               const int3 &resolution,
                               const bool checked,
                               float *output,
                               std::string &error)
{
  if (!volume) {
    return true;
  }
  const bke::VolumeGridData *grid_data = BKE_volume_grid_find(volume, name);
  if (!grid_data) {
    return true;
  }
  if (grid_data->grid_type() != VOLUME_GRID_FLOAT) {
    error = std::string("Grid '") + name + "' must be a float grid.";
    return false;
  }

  /* Borrow the grid. Wrapping the pointer in a #bke::VolumeGrid would release a user that this
   * function does not own. The type was checked above. */
  bke::VolumeTreeAccessToken token;
  const openvdb::FloatGrid &grid = static_cast<const openvdb::FloatGrid &>(
      grid_data->grid(token));
  if (!checked) {
    if (!(grid.transform() == expected_transform)) {
      error = std::string("Grid '") + name +
              "' does not match the requested origin and voxel-size transform.";
      return false;
    }
    if (!active_values_fit_domain(grid, resolution, name, error)) {
      return false;
    }
  }

  openvdb::FloatGrid::ConstAccessor accessor = grid.getConstAccessor();
  for (int z = 0; z < resolution.z; ++z) {
    for (int y = 0; y < resolution.y; ++y) {
      for (int x = 0; x < resolution.x; ++x) {
        const std::size_t index = manta_gas::grid_index(resolution.x, resolution.y, x, y, z);
        output[index] = accessor.getValue(openvdb::Coord(x, y, z));
      }
    }
  }
  return true;
}

static bool import_velocity_grid(const Volume *volume,
                                 const openvdb::math::Transform &expected_transform,
                                 const int3 &resolution,
                                 const float voxel_size,
                                 const bool checked,
                                 MantaGasVec3 *output,
                                 MantaGasVelocityLayout &r_layout,
                                 std::string &error)
{
  r_layout = MantaGasVelocityLayout::CellCentered;
  if (!volume) {
    return true;
  }
  constexpr const char *name = "velocity";
  const bke::VolumeGridData *grid_data = BKE_volume_grid_find(volume, name);
  if (!grid_data) {
    return true;
  }
  if (grid_data->grid_type() != VOLUME_GRID_VECTOR_FLOAT) {
    error = "Grid 'velocity' must be a float vector grid.";
    return false;
  }

  bke::VolumeTreeAccessToken token;
  const openvdb::Vec3SGrid &grid = static_cast<const openvdb::Vec3SGrid &>(
      grid_data->grid(token));
  if (!checked) {
    if (!(grid.transform() == expected_transform)) {
      error = "Grid 'velocity' does not match the requested origin and voxel-size transform.";
      return false;
    }
    if (!active_values_fit_domain(grid, resolution, name, error)) {
      return false;
    }
  }

  /* The node's own output is tagged as staggered; any other vector grid is an authored
   * cell-centered velocity. */
  if (grid.getGridClass() == openvdb::GRID_STAGGERED) {
    r_layout = MantaGasVelocityLayout::Staggered;
  }

  openvdb::Vec3SGrid::ConstAccessor accessor = grid.getConstAccessor();
  for (int z = 0; z < resolution.z; ++z) {
    for (int y = 0; y < resolution.y; ++y) {
      for (int x = 0; x < resolution.x; ++x) {
        const std::size_t index = manta_gas::grid_index(resolution.x, resolution.y, x, y, z);
        const openvdb::Vec3f value = accessor.getValue(openvdb::Coord(x, y, z));
        output[index] = {value[0] / voxel_size, value[1] / voxel_size, value[2] / voxel_size};
      }
    }
  }
  return true;
}

/** Metadata on the density grid that names the solver state a node wrote. */
static constexpr const char *sparse_state_key = "gas_step_state_id";
/** Metadata on every grid this node writes. Only a state carrying it is moved onto a changed
 * domain; an authored initial state must match the domain exactly. */
static constexpr const char *output_key = "gas_step_output";

static void tag_output(openvdb::GridBase &grid)
{
  grid.insertMeta(output_key, openvdb::BoolMetadata(true));
}

/**
 * State grids are written to bakes as 16-bit floats: half the file size, with differences of
 * about 5e-4 relative. MANTA_BAKE_HALF=0 writes full precision.
 */
static bool bake_half()
{
  static const bool on = [] {
    const char *value = std::getenv("MANTA_BAKE_HALF");
    return !(value && value[0] == '0');
  }();
  return on;
}

static openvdb::FloatGrid::Ptr make_scalar_grid(const std::vector<float> &values,
                                                const int3 &resolution,
                                                const openvdb::math::Transform &transform,
                                                const bool fog_volume)
{
  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
  grid->setTransform(transform.copy());
  grid->setGridClass(fog_volume ? openvdb::GRID_FOG_VOLUME : openvdb::GRID_UNKNOWN);
  tag_output(*grid);
  openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
  for (int z = 0; z < resolution.z; ++z) {
    for (int y = 0; y < resolution.y; ++y) {
      for (int x = 0; x < resolution.x; ++x) {
        const std::size_t index = manta_gas::grid_index(resolution.x, resolution.y, x, y, z);
        if (values[index] != 0.0f) {
          accessor.setValue(openvdb::Coord(x, y, z), values[index]);
        }
      }
    }
  }
  return grid;
}

static openvdb::Vec3SGrid::Ptr make_velocity_grid(const std::vector<MantaGasVec3> &values,
                                                  const int3 &resolution,
                                                  const openvdb::math::Transform &transform,
                                                  const float voxel_size)
{
  openvdb::Vec3SGrid::Ptr grid = openvdb::Vec3SGrid::create(openvdb::Vec3f::zero());
  grid->setTransform(transform.copy());
  /* Face-centered solver state, the same convention as the legacy fluid cache. */
  grid->setGridClass(openvdb::GRID_STAGGERED);
  tag_output(*grid);
  openvdb::Vec3SGrid::Accessor accessor = grid->getAccessor();
  for (int z = 0; z < resolution.z; ++z) {
    for (int y = 0; y < resolution.y; ++y) {
      for (int x = 0; x < resolution.x; ++x) {
        const std::size_t index = manta_gas::grid_index(resolution.x, resolution.y, x, y, z);
        const MantaGasVec3 &value = values[index];
        if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f) {
          accessor.setValue(openvdb::Coord(x, y, z),
                            openvdb::Vec3f(value.x * voxel_size,
                                           value.y * voxel_size,
                                           value.z * voxel_size));
        }
      }
    }
  }
  return grid;
}

namespace sparse = manta_improve::sparse;

struct SparseRun {
  int substeps = 0;
  /** Largest velocity component seen, in cells per second. */
  float max_speed = 0.0f;
  int active_bricks = 0;
  /** Set when the GPU could not be used and the CPU reference ran instead. */
  std::string fallback_reason;
  std::string device_name;
};

#  ifdef WITH_SPARSE_GAS_GPU
/**
 * GPU solvers kept for the whole process, created on first use, each with its own Vulkan device
 * and the state of the simulation it advanced last. When a node's input is exactly the state one
 * of them wrote (the same grid trees, tagged with the same id), that solver continues without
 * an upload; any other input is uploaded in full. Every use holds the lock, so zones never share
 * state. They are never destroyed, because tearing down Vulkan objects during process exit is
 * unsafe with some drivers.
 */
struct SparseSlot {
  manta_improve::gpu::SparseGasGpu *gpu = nullptr;
  std::int64_t state_id = 0;
  std::array<const void *, 6> trees{};
  std::array<int, 3> resolution{};
  std::uint64_t last_used = 0;
  /** The last frame's downloaded state, kept so its vectors are reused. */
  sparse::State shown;
  /** Where `shown` lives, to rebuild its velocity grid when the slot is given away. */
  openvdb::math::Transform::Ptr transform;
  float voxel_size = 0.0f;
};

struct SparseDevice {
  std::mutex mutex;
  std::vector<SparseSlot> slots;
  bool tried = false;
  std::string error;
  std::uint64_t clock = 0;
  /** Without Store Velocity: the velocity of states whose solver was given to another simulation,
   * by state id, oldest first. */
  std::vector<std::pair<std::int64_t, openvdb::GridBase::Ptr>> held;
};

static constexpr std::size_t sparse_slot_limit = 4;

static SparseDevice &sparse_device()
{
  static SparseDevice device;
  return device;
}
#  endif

static std::int64_t next_state_id()
{
  static std::mutex mutex;
  static std::int64_t next = [] {
    std::random_device random;
    return (std::int64_t(random()) << 32 | std::int64_t(random())) & 0x7fffffffffff0000;
  }();
  std::lock_guard<std::mutex> lock(mutex);
  return ++next;
}

/** Float grid `name` of a geometry's volume, or null. Throws if it has another type. */
static const openvdb::FloatGrid *find_float_grid(const Volume *volume,
                                                 const char *name,
                                                 bke::VolumeTreeAccessToken &token)
{
  const bke::VolumeGridData *data = volume ? BKE_volume_grid_find(volume, name) : nullptr;
  if (!data) {
    return nullptr;
  }
  if (data->grid_type() != VOLUME_GRID_FLOAT) {
    throw std::runtime_error(std::string("Grid '") + name + "' must be a float grid");
  }
  return &static_cast<const openvdb::FloatGrid &>(data->grid(token));
}

static const openvdb::Vec3SGrid *find_vector_grid(const Volume *volume,
                                                  const char *name,
                                                  bke::VolumeTreeAccessToken &token)
{
  const bke::VolumeGridData *data = volume ? BKE_volume_grid_find(volume, name) : nullptr;
  if (!data) {
    return nullptr;
  }
  if (data->grid_type() != VOLUME_GRID_VECTOR_FLOAT) {
    throw std::runtime_error(std::string("Grid '") + name + "' must be a float vector grid");
  }
  return &static_cast<const openvdb::Vec3SGrid &>(data->grid(token));
}

/**
 * Visit the domain voxels that the active values of a grid with any transform land on: each
 * active voxel goes to the nearest domain voxel. Cost follows the active voxels, not the domain.
 */
template<typename Visit>
static void for_each_active_voxel(const openvdb::FloatGrid &grid,
                                  const openvdb::math::Transform &domain,
                                  const int3 &resolution,
                                  const Visit &visit)
{
  for (openvdb::FloatGrid::ValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
    const float value = *iter;
    if (value == 0.0f) {
      continue;
    }
    openvdb::CoordBBox box;
    iter.getBoundingBox(box);
    for (auto coord = box.begin(); coord; ++coord) {
      const openvdb::Vec3d index = domain.worldToIndex(grid.transform().indexToWorld(*coord));
      const int x = int(std::lround(index.x())), y = int(std::lround(index.y())),
                z = int(std::lround(index.z()));
      if (x < 0 || y < 0 || z < 0 || x >= resolution.x || y >= resolution.y ||
          z >= resolution.z)
      {
        continue;
      }
      visit(x, y, z, value);
    }
  }
}

/** A world-space velocity sample at a domain voxel, in cells per second. */
static openvdb::Vec3f velocity_at(
    const openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler> &sampler,
    const openvdb::math::Transform &domain,
    const float voxel_size,
    const int x,
    const int y,
    const int z)
{
  return sampler.wsSample(domain.indexToWorld(openvdb::Vec3d(x, y, z))) / voxel_size;
}

/* ---- Leaf-parallel state export and source import ----
 * A solver brick is 8x8x8 with origins at multiples of 8, exactly one OpenVDB LeafNode<float,3>,
 * so leaves are built and read in parallel. For checking, the environment variable
 * MANTA_GAS_FAST_VDB=0 runs the original voxel-by-voxel code instead, and =2 runs both and
 * prints whether they give identical grids. */

static int fast_vdb_mode()
{
  static const int mode = [] {
    const char *value = std::getenv("MANTA_GAS_FAST_VDB");
    return value ? std::atoi(value) : 1;
  }();
  return mode;
}

using FloatLeaf = openvdb::FloatTree::LeafNodeType;
using VectorLeaf = openvdb::Vec3STree::LeafNodeType;

/** Per brick, the voxels inside the domain as leaf mask words (word = local x, bit = y * 8 + z). */
static void brick_inside_mask(const openvdb::Coord &origin, const int3 &limit, std::uint64_t inside[8])
{
  const int nx = std::min(8, limit.x - origin.x()), ny = std::min(8, limit.y - origin.y()),
            nz = std::min(8, limit.z - origin.z());
  std::uint64_t word = 0;
  for (int y = 0; y < ny; ++y) {
    word |= (nz >= 8 ? 0xffull : ((1ull << std::max(nz, 0)) - 1)) << (8 * y);
  }
  for (int x = 0; x < 8; ++x) {
    inside[x] = x < nx ? word : 0;
  }
}

/**
 * The leaf of one brick's values (512, X-fastest) with the voxels `get` reports as set active,
 * or null when there are none. Inactive voxels hold the zero background, as with setValue.
 */
template<typename LeafT, typename Get>
static LeafT *build_brick_leaf(const openvdb::Coord &origin,
                               const std::uint64_t inside[8],
                               const Get &get)
{
  using ValueT = typename LeafT::ValueType;
  const ValueT zero = openvdb::zeroVal<ValueT>();
  alignas(64) ValueT values[512];
  std::uint64_t words[8];
  std::uint64_t any = 0;
  for (int x = 0; x < 8; ++x) {
    std::uint64_t word = 0;
    for (int bit = 0; bit < 64; ++bit) {
      bool on = false;
      const ValueT value = get(x + 8 * (bit >> 3) + 64 * (bit & 7), on);
      on = on && ((inside[x] >> bit) & 1u);
      values[x * 64 + bit] = on ? value : zero;
      word |= std::uint64_t(on) << bit;
    }
    words[x] = word;
    any |= word;
  }
  if (!any) {
    return nullptr;
  }
  LeafT *leaf = new LeafT(origin, zero, false);
  std::memcpy(leaf->buffer().data(), values, sizeof(values));
  for (int x = 0; x < 8; ++x) {
    leaf->getValueMask().template getWord<std::uint64_t>(x) = words[x];
  }
  return leaf;
}

/** Fill the (empty) output trees from the state, leaves built in parallel. */
static void export_leaves(const sparse::State &state,
                          const int3 &resolution,
                          const float voxel_size,
                          const bool fire,
                          const bool with_pressure,
                          openvdb::FloatTree *const scalars[5],
                          openvdb::Vec3STree &velocity)
{
  const std::size_t count = state.topology.brick_count();
  /* Per slot: density, temperature, pressure, fuel, flame. */
  std::vector<FloatLeaf *> leaves(count * 5, nullptr);
  std::vector<VectorLeaf *> vectors(count, nullptr);
  threading::parallel_for(IndexRange(count), 16, [&](const IndexRange range) {
    for (const int64_t slot : range) {
      const std::int32_t *c = &state.topology.brick_coords[slot * 4];
      const openvdb::Coord origin(c[0] * 8, c[1] * 8, c[2] * 8);
      std::uint64_t inside[8];
      brick_inside_mask(origin, resolution, inside);
      const std::size_t base = std::size_t(slot) * 512;
      const auto scalar = [&](const std::vector<float> &channel) {
        const float *values = &channel[base];
        /* Most bricks of a plume are empty (the dilation shell): skip them with a cheap scan. */
        if (std::all_of(values, values + 512, [](const float v) { return v == 0.0f; })) {
          return (FloatLeaf *)nullptr;
        }
        return build_brick_leaf<FloatLeaf>(origin, inside, [values](const int i, bool &on) {
          on = values[i] != 0.0f;
          return values[i];
        });
      };
      leaves[slot * 5 + 0] = scalar(state.density);
      leaves[slot * 5 + 1] = scalar(state.temperature);
      if (with_pressure) {
        leaves[slot * 5 + 2] = scalar(state.pressure);
      }
      if (fire) {
        leaves[slot * 5 + 3] = scalar(state.fuel);
        leaves[slot * 5 + 4] = scalar(state.flame);
      }
      const float *u = &state.u[base], *v = &state.v[base], *w = &state.w[base];
      bool moving = false;
      for (int i = 0; i < 512; ++i) {
        moving |= (u[i] != 0.0f) | (v[i] != 0.0f) | (w[i] != 0.0f);
      }
      if (!moving) {
        continue;
      }
      vectors[slot] = build_brick_leaf<VectorLeaf>(origin, inside, [&](const int i, bool &on) {
        on = u[i] != 0.0f || v[i] != 0.0f || w[i] != 0.0f;
        return openvdb::Vec3f(u[i] * voxel_size, v[i] * voxel_size, w[i] * voxel_size);
      });
    }
  });
  for (std::size_t slot = 0; slot < count; ++slot) {
    for (int grid = 0; grid < 5; ++grid) {
      if (FloatLeaf *leaf = leaves[slot * 5 + grid]) {
        scalars[grid]->addLeaf(leaf);
      }
    }
    if (vectors[slot]) {
      velocity.addLeaf(vectors[slot]);
    }
  }
}

/** Whether two trees hold the same leaves, masks and bit-identical values (and no tiles). */
template<typename TreeT> static bool same_tree(const TreeT &a, const TreeT &b)
{
  if (a.leafCount() != b.leafCount() || a.activeVoxelCount() != b.activeVoxelCount() ||
      a.activeTileCount() != b.activeTileCount())
  {
    return false;
  }
  for (auto leaf = a.cbeginLeaf(); leaf; ++leaf) {
    const auto *other = b.probeConstLeaf(leaf->origin());
    if (!other || other->getValueMask() != leaf->getValueMask() ||
        std::memcmp(leaf->buffer().data(),
                    other->buffer().data(),
                    sizeof(typename TreeT::ValueType) * 512) != 0)
    {
      return false;
    }
  }
  return true;
}

/**
 * The integer offset from a grid's index space to the domain's when every grid voxel lands
 * exactly on a domain voxel (same voxel size, no rotation, integer translation), else nullopt.
 */
static std::optional<openvdb::Coord> aligned_offset(const openvdb::math::Transform &grid,
                                                    const openvdb::math::Transform &domain)
{
  if (!grid.isLinear() || !domain.isLinear()) {
    return std::nullopt;
  }
  const auto map = [&](const openvdb::Vec3d &c) {
    return domain.worldToIndex(grid.indexToWorld(c));
  };
  const openvdb::Vec3d origin = map(openvdb::Vec3d(0.0));
  const openvdb::Vec3d axes[3] = {map(openvdb::Vec3d(1, 0, 0)) - origin,
                                  map(openvdb::Vec3d(0, 1, 0)) - origin,
                                  map(openvdb::Vec3d(0, 0, 1)) - origin};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      if (std::abs(axes[a][b] - (a == b ? 1.0 : 0.0)) > 1e-9) {
        return std::nullopt;
      }
    }
  }
  openvdb::Coord offset;
  for (int a = 0; a < 3; ++a) {
    const double rounded = std::round(origin[a]);
    /* Far from a rounding tie, so every voxel rounds as the general path rounds it. */
    if (std::abs(origin[a] - rounded) > 1e-4 || std::abs(rounded) > 1e8) {
      return std::nullopt;
    }
    offset[a] = int(rounded);
  }
  return offset;
}

/**
 * Raise one channel of a source field to a grid's values, leaf by leaf in parallel when the
 * grid is aligned with the domain; else the general per-voxel path. Same result as that path.
 */
static void add_source_channel_fast(sparse::BrickField &field,
                                    const std::size_t channel,
                                    const openvdb::FloatGrid &grid,
                                    const openvdb::math::Transform &domain,
                                    const int3 &resolution)
{
  const std::optional<openvdb::Coord> offset = aligned_offset(grid.transform(), domain);
  if (!offset) {
    for_each_active_voxel(grid, domain, resolution, [&](int x, int y, int z, float value) {
      float &target = field.touch(channel, x, y, z);
      target = std::max(target, value);
    });
    return;
  }
  const openvdb::Coord off = *offset;
  const sparse::Extent bricks = field.brick_extent;
  const auto inside = [&](const openvdb::Coord &c) {
    return c.x() >= 0 && c.y() >= 0 && c.z() >= 0 && c.x() < resolution.x &&
           c.y() < resolution.y && c.z() < resolution.z;
  };
  const auto brick_of = [&](const openvdb::Coord &c) {
    return std::size_t(c.x() >> 3) +
           std::size_t(bricks.x) * (std::size_t(c.y() >> 3) + std::size_t(bricks.y) *
                                                                  std::size_t(c.z() >> 3));
  };
  /* Active tiles (rare): voxel by voxel. */
  {
    openvdb::FloatGrid::ValueOnCIter iter = grid.cbeginValueOn();
    iter.setMaxDepth(openvdb::FloatGrid::ValueOnCIter::LEAF_DEPTH - 1);
    for (; iter; ++iter) {
      const float value = *iter;
      if (value == 0.0f) {
        continue;
      }
      openvdb::CoordBBox box;
      iter.getBoundingBox(box);
      box.translate(off);
      box.intersect(openvdb::CoordBBox(
          openvdb::Coord(0),
          openvdb::Coord(resolution.x - 1, resolution.y - 1, resolution.z - 1)));
      for (auto c = box.begin(); c; ++c) {
        float &target = field.touch(channel, (*c).x(), (*c).y(), (*c).z());
        target = std::max(target, value);
      }
    }
  }
  std::vector<const FloatLeaf *> leaves;
  leaves.reserve(grid.tree().leafCount());
  for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
    leaves.push_back(&*leaf);
  }
  /* 1. Per leaf, the bricks its nonzero values land in (at most 8), in first-touch order. */
  std::vector<std::array<std::int64_t, 8>> touched(leaves.size());
  threading::parallel_for(IndexRange(int64_t(leaves.size())), 32, [&](const IndexRange range) {
    for (const int64_t l : range) {
      std::array<std::int64_t, 8> &list = touched[l];
      list.fill(-1);
      int n = 0;
      for (auto value = leaves[l]->cbeginValueOn(); value && n < 8; ++value) {
        const openvdb::Coord c = value.getCoord() + off;
        if (*value == 0.0f || !inside(c)) {
          continue;
        }
        const std::int64_t brick = std::int64_t(brick_of(c));
        if (std::find(list.begin(), list.begin() + n, brick) == list.begin() + n) {
          list[n++] = brick;
        }
      }
    }
  });
  /* 2. Storage in the order the general path would allocate it. */
  std::size_t slots = field.slot_count();
  for (const std::array<std::int64_t, 8> &list : touched) {
    for (const std::int64_t brick : list) {
      if (brick < 0) {
        break;
      }
      if (field.table[brick] < 0) {
        field.table[brick] = std::int32_t(slots++);
      }
    }
  }
  for (std::vector<float> &values : field.channels) {
    values.resize(slots * 512, 0.0f);
  }
  /* 3. The mapping is one to one, so no two leaves write the same voxel. */
  float *target = field.channels[channel].data();
  const std::int32_t *table = field.table.data();
  threading::parallel_for(IndexRange(int64_t(leaves.size())), 32, [&](const IndexRange range) {
    for (const int64_t l : range) {
      for (auto value = leaves[l]->cbeginValueOn(); value; ++value) {
        const float v = *value;
        const openvdb::Coord c = value.getCoord() + off;
        if (v == 0.0f || !inside(c)) {
          continue;
        }
        float &t = target[std::size_t(table[brick_of(c)]) * 512 + std::size_t(c.x() & 7) +
                          8 * (std::size_t(c.y() & 7) + 8 * std::size_t(c.z() & 7))];
        t = std::max(t, v);
      }
    }
  });
}

/** Whether two source fields hold the same bricks and bit-identical values. */
static bool same_field(const sparse::BrickField &a, const sparse::BrickField &b)
{
  if (a.slot_count() != b.slot_count() || a.channels.size() != b.channels.size() ||
      a.table.size() != b.table.size())
  {
    return false;
  }
  for (std::size_t brick = 0; brick < a.table.size(); ++brick) {
    if ((a.table[brick] < 0) != (b.table[brick] < 0)) {
      return false;
    }
    if (a.table[brick] < 0) {
      continue;
    }
    for (std::size_t c = 0; c < a.channels.size(); ++c) {
      if (std::memcmp(&a.channels[c][std::size_t(a.table[brick]) * 512],
                      &b.channels[c][std::size_t(b.table[brick]) * 512],
                      512 * sizeof(float)) != 0)
      {
        return false;
      }
    }
  }
  return true;
}

/** The sources of one frame, on bricks, straight from their grids. */
static sparse::Sources make_sources_impl(const openvdb::FloatGrid *density,
                                    const openvdb::FloatGrid *temperature,
                                    const openvdb::FloatGrid *fuel,
                                    const openvdb::FloatGrid *expansion,
                                    const openvdb::Vec3SGrid *velocity,
                                    const openvdb::math::Transform &domain,
                                    const int3 &resolution,
                                    const float voxel_size,
                                    const bool add,
                                    const bool fast)
{
  sparse::Sources sources;
  sources.mode = add ? sparse::SourceMode::Add : sparse::SourceMode::Raise;
  if (!density && !temperature && !fuel && !expansion) {
    return sources;
  }
  sources.field = sparse::make_brick_field(
      {resolution.x, resolution.y, resolution.z}, 8, velocity ? 7 : 4);
  sparse::BrickField &field = sources.field;
  const openvdb::FloatGrid *channels[4] = {density, temperature, fuel, expansion};
  for (int channel = 0; channel < 4; ++channel) {
    const openvdb::FloatGrid *grid = channels[channel];
    if (!grid) {
      continue;
    }
    if (fast) {
      add_source_channel_fast(field, std::size_t(channel), *grid, domain, resolution);
      continue;
    }
    for_each_active_voxel(*grid, domain, resolution, [&](int x, int y, int z, float value) {
      float &target = field.touch(std::size_t(channel), x, y, z);
      target = std::max(target, value);
    });
  }
  if (velocity) {
    /* The bricks with storage, sampled in parallel with the fast path (each task its own
     * sampler, whose accessor is not thread safe). */
    std::vector<std::size_t> stored;
    for (std::size_t brick = 0; brick < field.table.size(); ++brick) {
      if (field.table[brick] >= 0) {
        stored.push_back(brick);
      }
    }
    const auto sample = [&](const IndexRange range) {
    openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler> sampler(
        *velocity);
    for (const int64_t k : range) {
      const std::size_t brick = stored[k];
      const std::int32_t slot = field.table[brick];
      const int bx = int(brick % std::size_t(field.brick_extent.x));
      const int by = int(brick / std::size_t(field.brick_extent.x) %
                         std::size_t(field.brick_extent.y));
      const int bz = int(brick / (std::size_t(field.brick_extent.x) * field.brick_extent.y));
      for (int local = 0; local < 512; ++local) {
        const std::size_t i = std::size_t(slot) * 512 + std::size_t(local);
        if (field.channels[0][i] == 0.0f && field.channels[1][i] == 0.0f &&
            field.channels[2][i] == 0.0f && field.channels[3][i] == 0.0f)
        {
          continue;
        }
        const openvdb::Vec3f value = velocity_at(sampler,
                                                 domain,
                                                 voxel_size,
                                                 bx * 8 + (local & 7),
                                                 by * 8 + ((local >> 3) & 7),
                                                 bz * 8 + (local >> 6));
        field.channels[4][i] = value[0];
        field.channels[5][i] = value[1];
        field.channels[6][i] = value[2];
      }
    }
    };
    if (fast) {
      threading::parallel_for(IndexRange(int64_t(stored.size())), 8, sample);
    }
    else {
      sample(IndexRange(int64_t(stored.size())));
    }
  }
  return sources;
}

static sparse::Sources make_sources(const openvdb::FloatGrid *density,
                                    const openvdb::FloatGrid *temperature,
                                    const openvdb::FloatGrid *fuel,
                                    const openvdb::FloatGrid *expansion,
                                    const openvdb::Vec3SGrid *velocity,
                                    const openvdb::math::Transform &domain,
                                    const int3 &resolution,
                                    const float voxel_size,
                                    const bool add)
{
  const int mode = fast_vdb_mode();
  if (mode != 2) {
    return make_sources_impl(density, temperature, fuel, expansion, velocity, domain,
                             resolution, voxel_size, add, mode == 1);
  }
  auto start = std::chrono::steady_clock::now();
  sparse::Sources reference = make_sources_impl(
      density, temperature, fuel, expansion, velocity, domain, resolution, voxel_size, add, false);
  const double reference_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  start = std::chrono::steady_clock::now();
  sparse::Sources fast = make_sources_impl(
      density, temperature, fuel, expansion, velocity, domain, resolution, voxel_size, add, true);
  const double fast_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::fprintf(stderr,
               "MANTA_GAS_FAST_VDB sources: %zu bricks, original %.2f ms, fast %.2f ms, "
               "identical %d, same order %d\n",
               reference.field.slot_count(),
               reference_ms,
               fast_ms,
               int(same_field(reference.field, fast.field)),
               int(reference.field.table == fast.field.table));
  return fast;
}

/** The colliders of one frame: an obstacle bit per domain voxel and their velocity on bricks. */
static void make_colliders(const openvdb::FloatGrid *solid,
                           const openvdb::Vec3SGrid *velocity,
                           const openvdb::math::Transform &domain,
                           const int3 &resolution,
                           const float voxel_size,
                           std::vector<std::uint32_t> &mask,
                           sparse::BrickField &motion)
{
  mask.clear();
  motion = {};
  if (!solid) {
    return;
  }
  const std::size_t count = std::size_t(resolution.x) * resolution.y * resolution.z;
  mask.assign((count + 31) / 32, 0u);
  std::vector<int3> voxels;
  for_each_active_voxel(*solid, domain, resolution, [&](int x, int y, int z, float value) {
    if (!(value > 0.5f)) {
      return;
    }
    const std::size_t i = manta_gas::grid_index(resolution.x, resolution.y, x, y, z);
    if (!((mask[i >> 5] >> (i & 31)) & 1u)) {
      mask[i >> 5] |= 1u << (i & 31);
      voxels.push_back({x, y, z});
    }
  });
  if (velocity && !voxels.empty()) {
    motion = sparse::make_brick_field({resolution.x, resolution.y, resolution.z}, 8, 3);
    openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler> sampler(
        *velocity);
    for (const int3 &c : voxels) {
      const openvdb::Vec3f value = velocity_at(sampler, domain, voxel_size, c.x, c.y, c.z);
      for (int axis = 0; axis < 3; ++axis) {
        motion.touch(std::size_t(axis), c.x, c.y, c.z) = value[axis];
      }
    }
  }
}

/** The state grids of a node's input volume. */
struct StateGrids {
  const openvdb::FloatGrid *density = nullptr;
  const openvdb::FloatGrid *temperature = nullptr;
  const openvdb::Vec3SGrid *velocity = nullptr;
  const openvdb::FloatGrid *pressure = nullptr;
  const openvdb::FloatGrid *fuel = nullptr;
  const openvdb::FloatGrid *flame = nullptr;

  std::array<const void *, 6> trees() const
  {
    return {density ? &density->tree() : nullptr,
            temperature ? &temperature->tree() : nullptr,
            velocity ? &velocity->tree() : nullptr,
            pressure ? &pressure->tree() : nullptr,
            fuel ? &fuel->tree() : nullptr,
            flame ? &flame->tree() : nullptr};
  }

  /** The id a node wrote on its output, or 0. */
  std::int64_t state_id() const
  {
    if (!density) {
      return 0;
    }
    const auto meta = density->getMetadata<openvdb::Int64Metadata>(sparse_state_key);
    return meta ? meta->value() : 0;
  }
};

/**
 * Build a solver state from the state grids, reading only the bricks around their active
 * values. Equivalent to dense import followed by sparse::from_dense.
 */
static sparse::State import_sparse_state(const StateGrids &grids,
                                         const int3 &resolution,
                                         const float voxel_size,
                                         const bool staggered,
                                         const sparse::Settings &settings,
                                         std::vector<std::uint32_t> obstacle,
                                         sparse::BrickField obstacle_velocity)
{
  const sparse::Extent extent = {resolution.x, resolution.y, resolution.z};
  const sparse::Extent bricks = {(extent.x + 7) / 8, (extent.y + 7) / 8, (extent.z + 7) / 8};
  std::vector<std::uint8_t> flags(std::size_t(bricks.x) * bricks.y * bricks.z, 0);
  for (const openvdb::FloatGrid *grid :
       {grids.density, grids.temperature, grids.fuel, grids.flame})
  {
    if (!grid) {
      continue;
    }
    for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
      if (!(std::abs(*iter) > settings.thresholds.scalar)) {
        continue;
      }
      openvdb::CoordBBox box;
      iter.getBoundingBox(box);
      const openvdb::Coord lo = openvdb::Coord::maxComponent(box.min(), openvdb::Coord(0));
      const openvdb::Coord hi = openvdb::Coord::minComponent(
          box.max(), openvdb::Coord(extent.x - 1, extent.y - 1, extent.z - 1));
      for (int bz = lo.z() / 8; bz <= hi.z() / 8; ++bz) {
        for (int by = lo.y() / 8; by <= hi.y() / 8; ++by) {
          for (int bx = lo.x() / 8; bx <= hi.x() / 8; ++bx) {
            flags[std::size_t(bx) + std::size_t(bricks.x) * (by + std::size_t(bricks.y) * bz)] =
                1;
          }
        }
      }
    }
  }
  sparse::State state = sparse::make_state(sparse::build_topology(
      extent, 8, flags, settings.dilation, true, std::move(obstacle)));
  state.obstacle_velocity = std::move(obstacle_velocity);

  const float zero = 0.0f;
  std::optional<openvdb::FloatGrid::ConstAccessor> density, temperature, pressure, fuel, flame;
  std::optional<openvdb::Vec3SGrid::ConstAccessor> velocity;
  if (grids.density) {
    density.emplace(grids.density->getConstAccessor());
  }
  if (grids.temperature) {
    temperature.emplace(grids.temperature->getConstAccessor());
  }
  if (grids.pressure) {
    pressure.emplace(grids.pressure->getConstAccessor());
  }
  if (grids.fuel) {
    fuel.emplace(grids.fuel->getConstAccessor());
  }
  if (grids.flame) {
    flame.emplace(grids.flame->getConstAccessor());
  }
  if (grids.velocity) {
    velocity.emplace(grids.velocity->getConstAccessor());
  }
  const auto vel = [&](int x, int y, int z) {
    return velocity->getValue(openvdb::Coord(x, y, z)) / voxel_size;
  };
  const sparse::Topology &topology = state.topology;
  for (std::size_t slot = 0; slot < topology.brick_count(); ++slot) {
    const int ox = topology.brick_coords[slot * 4] * 8;
    const int oy = topology.brick_coords[slot * 4 + 1] * 8;
    const int oz = topology.brick_coords[slot * 4 + 2] * 8;
    for (int local = 0; local < 512; ++local) {
      const int x = ox + (local & 7), y = oy + ((local >> 3) & 7), z = oz + (local >> 6);
      if (x >= extent.x || y >= extent.y || z >= extent.z) {
        continue;
      }
      const std::size_t i = slot * 512 + std::size_t(local);
      const openvdb::Coord c(x, y, z);
      state.density[i] = density ? density->getValue(c) : zero;
      state.temperature[i] = temperature ? temperature->getValue(c) : zero;
      state.pressure[i] = pressure ? pressure->getValue(c) : zero;
      state.fuel[i] = fuel ? fuel->getValue(c) : zero;
      state.flame[i] = flame ? flame->getValue(c) : zero;
      if (!velocity) {
        continue;
      }
      if (staggered) {
        const openvdb::Vec3f value = vel(x, y, z);
        state.u[i] = value[0];
        state.v[i] = value[1];
        state.w[i] = value[2];
        continue;
      }
      /* Authored cell-centered input: average onto the lower faces once, as Mantaflow's
       * resampleVec3ToMac does. */
      const openvdb::Vec3f here = vel(x, y, z);
      state.u[i] = x > 0 ? 0.5f * (vel(x - 1, y, z)[0] + here[0]) : 0.0f;
      state.v[i] = y > 0 ? 0.5f * (vel(x, y - 1, z)[1] + here[1]) : 0.0f;
      state.w[i] = z > 0 ? 0.5f * (vel(x, y, z - 1)[2] + here[2]) : 0.0f;
    }
  }
  return state;
}

/**
 * Write a solver state into a new volume, brick by brick, tagging it with `state_id`. Fuel and
 * flame grids are written only with `fire`.
 */
static Volume *export_sparse_state(const sparse::State &state,
                                   const int3 &resolution,
                                   const openvdb::math::Transform &transform,
                                   const float voxel_size,
                                   const std::int64_t state_id,
                                   const bool fire,
                                   const bool with_velocity,
                                   std::array<const void *, 6> &trees)
{
  const auto scalar = [&](const bool fog) {
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(transform.copy());
    grid->setGridClass(fog ? openvdb::GRID_FOG_VOLUME : openvdb::GRID_UNKNOWN);
    tag_output(*grid);
    grid->setSaveFloatAsHalf(bake_half());
    return grid;
  };
  /* Pressure only when the state has it: the GPU path keeps it on the device. */
  const bool with_pressure = !state.pressure.empty();
  openvdb::FloatGrid::Ptr density = scalar(true), temperature = scalar(true),
                          pressure = scalar(false), fuel = scalar(true), flame = scalar(true);
  openvdb::Vec3SGrid::Ptr velocity = openvdb::Vec3SGrid::create(openvdb::Vec3f::zero());
  velocity->setTransform(transform.copy());
  velocity->setGridClass(openvdb::GRID_STAGGERED);
  velocity->setSaveFloatAsHalf(bake_half());
  tag_output(*velocity);
  density->insertMeta(sparse_state_key, openvdb::Int64Metadata(state_id));

  const int mode = fast_vdb_mode();
  const auto fast_export = [&](openvdb::FloatTree *const scalars[5], openvdb::Vec3STree &vectors) {
    export_leaves(state, resolution, voxel_size, fire, with_pressure, scalars, vectors);
  };
  if (mode == 1) {
    openvdb::FloatTree *scalars[5] = {&density->tree(),
                                      &temperature->tree(),
                                      &pressure->tree(),
                                      &fuel->tree(),
                                      &flame->tree()};
    fast_export(scalars, velocity->tree());
  }
  else {
    const auto start = std::chrono::steady_clock::now();
    openvdb::FloatGrid::Accessor density_access = density->getAccessor();
    openvdb::FloatGrid::Accessor temperature_access = temperature->getAccessor();
    openvdb::FloatGrid::Accessor pressure_access = pressure->getAccessor();
    openvdb::FloatGrid::Accessor fuel_access = fuel->getAccessor();
    openvdb::FloatGrid::Accessor flame_access = flame->getAccessor();
    openvdb::Vec3SGrid::Accessor velocity_access = velocity->getAccessor();
    const sparse::Topology &topology = state.topology;
    for (std::size_t slot = 0; slot < topology.brick_count(); ++slot) {
      const int ox = topology.brick_coords[slot * 4] * 8;
      const int oy = topology.brick_coords[slot * 4 + 1] * 8;
      const int oz = topology.brick_coords[slot * 4 + 2] * 8;
      for (int local = 0; local < 512; ++local) {
        const int x = ox + (local & 7), y = oy + ((local >> 3) & 7), z = oz + (local >> 6);
        if (x >= resolution.x || y >= resolution.y || z >= resolution.z) {
          continue;
        }
        const std::size_t i = slot * 512 + std::size_t(local);
        const openvdb::Coord c(x, y, z);
        if (state.density[i] != 0.0f) {
          density_access.setValue(c, state.density[i]);
        }
        if (state.temperature[i] != 0.0f) {
          temperature_access.setValue(c, state.temperature[i]);
        }
        if (with_pressure && state.pressure[i] != 0.0f) {
          pressure_access.setValue(c, state.pressure[i]);
        }
        if (fire && state.fuel[i] != 0.0f) {
          fuel_access.setValue(c, state.fuel[i]);
        }
        if (fire && state.flame[i] != 0.0f) {
          flame_access.setValue(c, state.flame[i]);
        }
        if (with_velocity && (state.u[i] != 0.0f || state.v[i] != 0.0f || state.w[i] != 0.0f)) {
          velocity_access.setValue(c,
                                   openvdb::Vec3f(state.u[i] * voxel_size,
                                                  state.v[i] * voxel_size,
                                                  state.w[i] * voxel_size));
        }
      }
    }
    if (mode == 2) {
      const double reference_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
      openvdb::FloatTree check[5];
      openvdb::Vec3STree check_velocity(openvdb::Vec3f::zero());
      openvdb::FloatTree *scalars[5] = {&check[0], &check[1], &check[2], &check[3], &check[4]};
      const auto fast_start = std::chrono::steady_clock::now();
      fast_export(scalars, check_velocity);
      const double fast_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - fast_start)
                                 .count();
      const bool identical = same_tree(density->tree(), check[0]) &&
                             same_tree(temperature->tree(), check[1]) &&
                             same_tree(pressure->tree(), check[2]) &&
                             same_tree(fuel->tree(), check[3]) &&
                             same_tree(flame->tree(), check[4]) &&
                             same_tree(velocity->tree(), check_velocity);
      std::fprintf(stderr,
                   "MANTA_GAS_FAST_VDB export: %zu bricks, original %.2f ms, fast %.2f ms, "
                   "identical %d\n",
                   state.topology.brick_count(),
                   reference_ms,
                   fast_ms,
                   int(identical));
    }
  }
  trees = {&density->tree(),
           &temperature->tree(),
           with_velocity ? &velocity->tree() : nullptr,
           with_pressure ? &pressure->tree() : nullptr,
           fire ? &fuel->tree() : nullptr,
           fire ? &flame->tree() : nullptr};
  Volume *volume = BKE_id_new_nomain<Volume>("Gas Simulation Step State");
  /* The volume must own each grid alone. */
  BKE_volume_grid_add_vdb(*volume, "density", std::move(density));
  BKE_volume_grid_add_vdb(*volume, "temperature", std::move(temperature));
  if (with_velocity) {
    BKE_volume_grid_add_vdb(*volume, "velocity", std::move(velocity));
  }
  if (with_pressure) {
    BKE_volume_grid_add_vdb(*volume, "pressure", std::move(pressure));
  }
  if (fire) {
    BKE_volume_grid_add_vdb(*volume, "fuel", std::move(fuel));
    BKE_volume_grid_add_vdb(*volume, "flame", std::move(flame));
  }
  return volume;
}

static double milliseconds_since(const std::chrono::steady_clock::time_point &start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

/** Checks shared by both solvers' input grids: transform and extent. */
static void check_state_grid(const openvdb::GridBase *grid,
                             const char *name,
                             const openvdb::math::Transform &transform,
                             const int3 &resolution,
                             const bool checked)
{
  if (!grid || checked) {
    return;
  }
  if (!(grid->transform() == transform)) {
    throw std::runtime_error(std::string("Grid '") + name +
                             "' does not match the requested origin and voxel-size transform.");
  }
  std::string error;
  if (!active_values_fit_domain(*grid, resolution, name, error)) {
    throw std::runtime_error(error);
  }
}

/** Turbulence layers from a point cloud, one point per layer in point order; see the input. */
static std::vector<sparse::Turbulence> make_turbulence_layers(const GeometrySet &geometry,
                                                              const float voxel_size)
{
  std::vector<sparse::Turbulence> layers;
  const PointCloud *points = geometry.get_pointcloud();
  if (!points || points->totpoint == 0) {
    return layers;
  }
  const bke::AttributeAccessor attributes = points->attributes();
  const bke::AttrDomain domain = bke::AttrDomain::Point;
  const VArray<float> strength = *attributes.lookup_or_default<float>("strength", domain, 0.0f);
  const VArray<float> scale = *attributes.lookup_or_default<float>("scale", domain, 0.5f);
  const VArray<float> speed = *attributes.lookup_or_default<float>("speed", domain, 1.0f);
  const VArray<float> level = *attributes.lookup_or_default<float>("mask_level", domain, 0.1f);
  const VArray<int> seed = *attributes.lookup_or_default<int>("seed", domain, 0);
  const VArray<int> mask = *attributes.lookup_or_default<int>("mask", domain, 1);
  const VArray<bool> invert = *attributes.lookup_or_default<bool>("invert", domain, false);
  for (const int i : IndexRange(points->totpoint)) {
    sparse::Turbulence layer;
    layer.strength = strength[i] / voxel_size;
    layer.scale = scale[i] / voxel_size;
    layer.speed = speed[i];
    layer.seed = std::uint32_t(std::max(0, seed[i]));
    switch (mask[i]) {
      case 0:
        layer.mask = sparse::TurbulenceMask::None;
        break;
      case 2:
        layer.mask = sparse::TurbulenceMask::Temperature;
        break;
      case 3:
        layer.mask = sparse::TurbulenceMask::Flame;
        break;
      case 4:
        layer.mask = sparse::TurbulenceMask::Speed;
        break;
      default:
        layer.mask = sparse::TurbulenceMask::Density;
        break;
    }
    /* Speed is in world units per second; the solver's in voxels per second. */
    layer.mask_level = layer.mask == sparse::TurbulenceMask::Speed ? level[i] / voxel_size :
                                                                     level[i];
    layer.invert = invert[i];
    if (!std::isfinite(layer.strength) || !std::isfinite(layer.speed) ||
        !(layer.scale > 0.0f) || !std::isfinite(layer.scale) ||
        !(layer.mask_level > 0.0f) || !std::isfinite(layer.mask_level))
    {
      throw std::runtime_error(
          "Turbulence layers need finite values, with positive scale and mask level");
    }
    layers.push_back(layer);
  }
  return layers;
}

/**
 * The Sparse GPU path. The state crosses OpenVDB only where it has bricks, and a solver that
 * wrote this node's input last frame continues from its own device state.
 */
template<typename GridT>
static openvdb::GridBase::Ptr fit_grid(const GridT &grid,
                                       const char *name,
                                       const openvdb::math::Transform &transform,
                                       const int3 &resolution);

static void exec_sparse(GeoNodeExecParams &params,
                        GeometrySet &state,
                        const int3 &resolution,
                        const openvdb::math::Transform &transform,
                        const float voxel_size,
                        const float3 &gravity,
                        const float delta_time,
                        const bool checked)
{
  const auto import_start = std::chrono::steady_clock::now();
  const GeometrySet source_geometry = params.extract_input<GeometrySet>("Sources"_ustr);
  const bool add_sources = params.extract_input<bool>("Add Sources"_ustr);
  const GeometrySet collider_geometry = params.extract_input<GeometrySet>("Colliders"_ustr);

  sparse::Settings settings;
  settings.cfl = params.extract_input<float>("CFL"_ustr);
  settings.max_substeps = params.extract_input<int>("Max Substeps"_ustr);
  settings.gravity[0] = gravity.x / voxel_size;
  settings.gravity[1] = gravity.y / voxel_size;
  settings.gravity[2] = gravity.z / voxel_size;
  settings.density_buoyancy = params.extract_input<float>("Density Buoyancy"_ustr);
  settings.heat_buoyancy = params.extract_input<float>("Heat Buoyancy"_ustr);
  settings.pressure_solver = sparse::PressureSolver::Multigrid;
  settings.pressure_iterations = 6;
  settings.pressure_iterations = std::clamp(
      params.extract_input<int>("Pressure Cycles"_ustr), 1, 16);
  settings.open_borders = params.extract_input<bool>("Open Borders"_ustr);
  settings.density_fade = std::max(0.0f, params.extract_input<float>("Smoke Fade"_ustr));
  settings.temperature_fade = std::max(0.0f, params.extract_input<float>("Heat Fade"_ustr));
  settings.thresholds.scalar = std::max(0.0f, params.extract_input<float>("Smoke Threshold"_ustr));
  settings.directional_dilation = params.extract_input<bool>("Directional Margin"_ustr);
  settings.turbulence_once = params.extract_input<bool>("Fast Turbulence"_ustr);
  settings.half_precision = params.extract_input<bool>("Half Precision"_ustr) ? 1 : 0;
  settings.command_batch = params.extract_input<bool>("Batch GPU Work"_ustr) ? 64 : 0;
  const bool store_velocity = params.extract_input<bool>("Store Velocity"_ustr);
  if (!std::isfinite(settings.density_fade) || !std::isfinite(settings.temperature_fade) ||
      !std::isfinite(settings.thresholds.scalar))
  {
    fail(params, state, "Smoke Fade, Heat Fade and Smoke Threshold must be finite.");
    return;
  }
  settings.time = params.extract_input<float>("Time"_ustr);
  sparse::Turbulence &turbulence = settings.turbulence;
  turbulence.strength = params.extract_input<float>("Turbulence"_ustr) / voxel_size;
  turbulence.scale = params.extract_input<float>("Turbulence Scale"_ustr) / voxel_size;
  turbulence.speed = params.extract_input<float>("Turbulence Speed"_ustr);
  turbulence.seed = static_cast<std::uint32_t>(
      std::max(0, params.extract_input<int>("Turbulence Seed"_ustr)));
  const float turbulence_mask = params.extract_input<float>("Turbulence Mask"_ustr);
  turbulence.mask = turbulence_mask > 0.0f ? sparse::TurbulenceMask::Density :
                                             sparse::TurbulenceMask::None;
  turbulence.mask_level = turbulence_mask > 0.0f ? turbulence_mask : 1.0f;
  const GeometrySet layer_geometry = params.extract_input<GeometrySet>("Turbulence Layers"_ustr);
  sparse::Fire &fire = settings.fire;
  fire.burn_rate = params.extract_input<float>("Burn Rate"_ustr);
  fire.heat = params.extract_input<float>("Flame Heat"_ustr);
  fire.smoke = params.extract_input<float>("Flame Smoke"_ustr);
  fire.flame_cooling = params.extract_input<float>("Flame Cooling"_ustr);
  fire.gas_release = params.extract_input<float>("Gas Release"_ustr);
  if (!std::isfinite(fire.burn_rate) || !std::isfinite(fire.heat) ||
      !std::isfinite(fire.smoke) || !std::isfinite(fire.flame_cooling) ||
      !std::isfinite(fire.gas_release) || fire.burn_rate < 0.0f || fire.heat < 0.0f ||
      fire.smoke < 0.0f || fire.flame_cooling < 0.0f || fire.gas_release < 0.0f)
  {
    fail(params, state, "The fire settings must be finite and not negative.");
    return;
  }
  if (!(settings.cfl > 0.0f) || !std::isfinite(settings.cfl) || settings.max_substeps < 1 ||
      !std::isfinite(settings.time) || !std::isfinite(turbulence.strength) ||
      !(turbulence.scale > 0.0f) || !std::isfinite(turbulence.scale) ||
      !std::isfinite(turbulence.speed))
  {
    fail(params, state,
         "CFL, Max Substeps, Time and the turbulence settings must be finite, with positive "
         "CFL, substeps and scale.");
    return;
  }

  bke::VolumeTreeAccessToken token;
  StateGrids grids;
  sparse::Sources sources;
  std::vector<std::uint32_t> obstacle;
  sparse::BrickField obstacle_velocity;
  bool staggered = false;
  try {
    const Volume *input = state.get_volume();
    grids.density = find_float_grid(input, "density", token);
    grids.temperature = find_float_grid(input, "temperature", token);
    grids.velocity = find_vector_grid(input, "velocity", token);
    grids.pressure = find_float_grid(input, "pressure", token);
    grids.fuel = find_float_grid(input, "fuel", token);
    grids.flame = find_float_grid(input, "flame", token);
    check_state_grid(grids.fuel, "fuel", transform, resolution, checked);
    check_state_grid(grids.flame, "flame", transform, resolution, checked);
    check_state_grid(grids.density, "density", transform, resolution, checked);
    check_state_grid(grids.temperature, "temperature", transform, resolution, checked);
    check_state_grid(grids.velocity, "velocity", transform, resolution, checked);
    check_state_grid(grids.pressure, "pressure", transform, resolution, checked);
    /* The node's own output is tagged as staggered; any other vector grid is an authored
     * cell-centered velocity. */
    staggered = grids.velocity && grids.velocity->getGridClass() == openvdb::GRID_STAGGERED;

    const Volume *source_volume = source_geometry.get_volume();
    const openvdb::FloatGrid *source_fuel = find_float_grid(source_volume, "fuel", token);
    sources = make_sources(find_float_grid(source_volume, "density", token),
                           find_float_grid(source_volume, "temperature", token),
                           source_fuel,
                           find_float_grid(source_volume, "divergence", token),
                           find_vector_grid(source_volume, "velocity", token),
                           transform,
                           resolution,
                           voxel_size,
                           add_sources);
    const Volume *collider_volume = collider_geometry.get_volume();
    make_colliders(find_float_grid(collider_volume, "obstacle", token),
                   find_vector_grid(collider_volume, "velocity", token),
                   transform,
                   resolution,
                   voxel_size,
                   obstacle,
                   obstacle_velocity);
    /* Fire runs while there is fuel or flame in the state, or fuel to emit. */
    fire.enabled = grids.fuel || grids.flame || source_fuel;
    settings.turbulence_layers = make_turbulence_layers(layer_geometry, voxel_size);
  }
  catch (const std::exception &exception) {
    fail(params, state, exception.what());
    return;
  }
  const std::int64_t input_id = grids.state_id();
  const std::array<const void *, 6> input_trees = grids.trees();
  double import_ms = milliseconds_since(import_start);

  const std::int64_t output_id = next_state_id();
  std::array<const void *, 6> output_trees{};
  Volume *output_volume = nullptr;
  SparseRun run;
  double solve_ms = 0.0, export_ms = 0.0;
  bool continued = false;
  try {
    const auto import_state = [&]() {
      const auto start = std::chrono::steady_clock::now();
      sparse::State result = import_sparse_state(grids,
                                                 resolution,
                                                 voxel_size,
                                                 staggered,
                                                 settings,
                                                 std::move(obstacle),
                                                 std::move(obstacle_velocity));
      import_ms += milliseconds_since(start);
      return result;
    };
    const auto finish = [&](const sparse::State &result,
                            const sparse::FrameInfo &frame,
                            const bool with_velocity) {
      run.substeps = frame.substeps;
      run.max_speed = frame.max_speed;
      run.active_bricks = static_cast<int>(result.topology.brick_count());
      const auto start = std::chrono::steady_clock::now();
      output_volume = export_sparse_state(
          result, resolution, transform, voxel_size, output_id, fire.enabled, with_velocity,
          output_trees);
      export_ms = milliseconds_since(start);
    };
    bool done = false;
#  ifdef WITH_SPARSE_GAS_GPU
    {
      SparseDevice &device = sparse_device();
      std::lock_guard<std::mutex> lock(device.mutex);
      if (!device.tried) {
        device.tried = true;
        try {
          device.slots.push_back({new manta_improve::gpu::SparseGasGpu()});
        }
        catch (const std::exception &exception) {
          device.error = exception.what();
        }
      }
      if (!device.slots.empty()) {
        /* The solver that wrote this input, or else a free or the least recently used one. */
        SparseSlot *slot = nullptr;
        for (SparseSlot &candidate : device.slots) {
          if (input_id != 0 && candidate.state_id == input_id && candidate.trees == input_trees &&
              candidate.resolution == std::array<int, 3>{resolution.x, resolution.y, resolution.z})
          {
            slot = &candidate;
            continued = true;
          }
        }
        if (!slot && input_id != 0) {
          /* The same simulation with changed grids (moved onto a new domain, or edited): its
           * own solver, rather than a fresh one while the old one holds a stale state. */
          for (SparseSlot &candidate : device.slots) {
            if (candidate.state_id == input_id) {
              slot = &candidate;
              break;
            }
          }
        }
        if (!slot) {
          for (SparseSlot &candidate : device.slots) {
            if (candidate.last_used == 0) {
              slot = &candidate;
              break;
            }
          }
        }
        if (!slot && device.slots.size() < sparse_slot_limit) {
          try {
            device.slots.push_back({new manta_improve::gpu::SparseGasGpu()});
            slot = &device.slots.back();
          }
          catch (const std::exception &) {
            /* Reuse an existing solver below. */
          }
        }
        if (!slot) {
          slot = &*std::min_element(
              device.slots.begin(), device.slots.end(), [](const auto &a, const auto &b) {
                return a.last_used < b.last_used;
              });
        }
        slot->last_used = ++device.clock;
        openvdb::GridBase::Ptr resumed_velocity;
        /* A change of precision drops the device state: start again from this input. */
        const bool was_half = slot->gpu->half_precision();
        if (slot->gpu->set_half_precision(settings.half_precision != 0) != was_half) {
          continued = false;
        }
        if (!store_velocity && !continued) {
          /* The slot's state is about to be dropped: keep its velocity, which its cached frame
           * does not have, in case its simulation comes back. */
          if (slot->state_id != 0 && slot->transform && !slot->shown.u.empty()) {
            sparse::State &old = slot->shown;
            std::array<const void *, 6> unused;
            const int3 old_resolution(slot->resolution[0], slot->resolution[1], slot->resolution[2]);
            Volume *held = export_sparse_state(
                old, old_resolution, *slot->transform, slot->voxel_size, 0, false, true, unused);
            bke::VolumeTreeAccessToken held_token;
            const openvdb::GridBase &held_grid =
                BKE_volume_grid_find(held, "velocity")->grid(held_token);
            device.held.emplace_back(slot->state_id, held_grid.deepCopyGrid());
            BKE_id_free(nullptr, held);
            if (device.held.size() > 2 * sparse_slot_limit) {
              device.held.erase(device.held.begin());
            }
          }
          /* A lean input continued elsewhere than on its own solver: its velocity from above. */
          if (!grids.velocity && input_id != 0) {
            for (const auto &[id, grid] : device.held) {
              if (id == input_id) {
                const auto &held = static_cast<const openvdb::Vec3SGrid &>(*grid);
                resumed_velocity = fit_grid(held, "velocity", transform, resolution);
              }
            }
            if (resumed_velocity) {
              grids.velocity = static_cast<const openvdb::Vec3SGrid *>(resumed_velocity.get());
              staggered = true;
            }
          }
        }
        /* Until this evaluation finishes, the slot holds no state anyone may continue. */
        slot->state_id = 0;
        sparse::State fresh;
        if (!continued) {
          fresh = import_state();
        }
        const auto solve_start = std::chrono::steady_clock::now();
        if (!continued) {
          slot->gpu->upload(fresh);
        }
        else if (!slot->gpu->same_obstacles(obstacle, obstacle_velocity)) {
          /* Unchanged colliders (or none) need no new classification or multigrid setup. */
          slot->gpu->set_obstacles(std::move(obstacle), std::move(obstacle_velocity));
        }
        const sparse::FrameInfo frame = slot->gpu->advance(settings, delta_time, &sources);
        /* Only what is shown and needed to continue comes back; pressure stays on the device
         * (a frame that cannot continue there restarts it from zero). The slot keeps the
         * vectors so their memory is reused from frame to frame. */
        sparse::State &result = slot->shown;
        slot->gpu->download_shown(result, fire.enabled, false);
        solve_ms = milliseconds_since(solve_start);
        finish(result, frame, store_velocity);
        slot->transform = transform.copy();
        slot->voxel_size = voxel_size;
        slot->state_id = output_id;
        slot->trees = output_trees;
        slot->resolution = {resolution.x, resolution.y, resolution.z};
        run.device_name = slot->gpu->device_name();
        done = true;
      }
      else {
        run.fallback_reason = "no usable Vulkan device (" + device.error + ")";
      }
    }
#  else
    run.fallback_reason = "this Blender was built without the Vulkan kernels";
#  endif
    if (!done) {
      sparse::State result = import_state();
      const auto solve_start = std::chrono::steady_clock::now();
      const sparse::FrameInfo frame = sparse::advance(result, settings, delta_time, &sources);
      solve_ms = milliseconds_since(solve_start);
      finish(result, frame, true);
    }
  }
  catch (const std::exception &exception) {
    fail(params, state, std::string("Sparse solver failed: ") + exception.what());
    return;
  }

  params.set_output("State"_ustr, GeometrySet::from_volume(output_volume));
  set_diagnostics(params,
                  run.substeps,
                  run.max_speed * voxel_size,
                  float(import_ms),
                  float(solve_ms),
                  float(export_ms),
                  settings.pressure_iterations,
                  0.0f,
                  run.active_bricks);
  if (!run.fallback_reason.empty()) {
    params.error_message_add(NodeWarningType::Warning,
                             "Sparse solver ran on the CPU, which is slow: " +
                                 run.fallback_reason);
  }
  else {
    params.error_message_add(NodeWarningType::Info,
                             "Sparse solver ran on " + run.device_name +
                                 (continued ? " (state kept on the GPU)" :
                                              " (state uploaded)"));
  }
}

static void copy_metadata(const openvdb::GridBase &from, openvdb::GridBase &to)
{
  for (auto it = from.beginMeta(); it != from.endMeta(); ++it) {
    to.insertMeta(it->first, *it->second);
  }
}

/** The domain's voxels that the active values of `grid` can reach once moved onto `transform`:
 * their bounds, one voxel wider, clipped to the domain. */
static openvdb::CoordBBox fit_region(const openvdb::GridBase &grid,
                                     const openvdb::math::Transform &transform,
                                     const int3 &resolution)
{
  openvdb::CoordBBox source = grid.evalActiveVoxelBoundingBox();
  source.expand(1);
  const openvdb::BBoxd world = grid.transform().indexToWorld(source);
  const openvdb::BBoxd index = transform.worldToIndex(world);
  openvdb::CoordBBox region(openvdb::Coord::floor(index.min()) - openvdb::Coord(1),
                            openvdb::Coord::ceil(index.max()) + openvdb::Coord(1));
  region.intersect(openvdb::CoordBBox(
      openvdb::Coord(0), openvdb::Coord(resolution.x - 1, resolution.y - 1, resolution.z - 1)));
  return region;
}

/** The node's staggered velocity on `transform`: each component sampled where its lower face
 * lies, which is where both grids store it. Values are in world units and need no scaling. */
static openvdb::Vec3SGrid::Ptr resample_staggered(const openvdb::Vec3SGrid &grid,
                                                  const openvdb::math::Transform &transform,
                                                  const int3 &resolution)
{
  openvdb::Vec3SGrid::Ptr result = openvdb::Vec3SGrid::create(grid.background());
  result->setTransform(transform.copy());
  if (grid.activeVoxelCount() == 0) {
    return result;
  }
  const openvdb::CoordBBox region = fit_region(grid, transform, resolution);
  if (region.empty()) {
    return result;
  }
  /* The sampler keeps a reference to the accessor. */
  const openvdb::Vec3SGrid::ConstAccessor input = grid.getConstAccessor();
  const openvdb::tools::GridSampler<openvdb::Vec3SGrid::ConstAccessor,
                                    openvdb::tools::StaggeredBoxSampler>
      sampler(input, grid.transform());
  openvdb::Vec3SGrid::Accessor output = result->getAccessor();
  for (auto it = region.begin(); it; ++it) {
    const openvdb::Coord &c = *it;
    openvdb::Vec3f value;
    for (int axis = 0; axis < 3; ++axis) {
      openvdb::Vec3d face(c.x(), c.y(), c.z());
      face[axis] -= 0.5;
      value[axis] = sampler.wsSample(transform.indexToWorld(face))[axis];
    }
    if (value != openvdb::Vec3f::zero()) {
      output.setValue(c, value);
    }
  }
  return result;
}

/**
 * A copy of the node-written `grid` on `transform` and inside the domain. Velocity keeps its
 * staggering, and pressure, which the solvers keep in voxel units, is rescaled to the new voxel
 * size. The metadata stays, so the GPU solver that wrote the state is reused for it.
 */
template<typename GridT>
static openvdb::GridBase::Ptr fit_grid(const GridT &grid,
                                       const char *name,
                                       const openvdb::math::Transform &transform,
                                       const int3 &resolution)
{
  typename GridT::Ptr result;
  if (grid.transform() == transform) {
    result = grid.deepCopy();
  }
  else {
    if constexpr (std::is_same_v<GridT, openvdb::Vec3SGrid>) {
      if (grid.getGridClass() == openvdb::GRID_STAGGERED) {
        result = resample_staggered(grid, transform, resolution);
      }
    }
    if (!result) {
      result = GridT::create(grid.background());
      result->setTransform(transform.copy());
      openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(grid, *result);
    }
    copy_metadata(grid, *result);
    result->setGridClass(grid.getGridClass());
    result->setVectorType(grid.getVectorType());
    if constexpr (std::is_same_v<GridT, openvdb::FloatGrid>) {
      if (STREQ(name, "pressure")) {
        const double ratio = grid.voxelSize()[0] / transform.voxelSize()[0];
        const float scale = float(ratio * ratio);
        for (auto value = result->beginValueOn(); value; ++value) {
          value.setValue(*value * scale);
        }
      }
    }
  }
  result->tree().clip(openvdb::CoordBBox(
      openvdb::Coord(0), openvdb::Coord(resolution.x - 1, resolution.y - 1, resolution.z - 1)));
  return result;
}

static constexpr const char *state_grid_names[] = {
    "density", "temperature", "velocity", "pressure", "fuel", "flame"};

/**
 * A simulation zone keeps its state when the domain's settings change part-way through, so the
 * state can arrive on another voxel size or origin, or reach past a smaller domain. Moves each
 * grid of such a state that does not fit onto the requested domain, if this node wrote the
 * state. Sets `r_moved` if it moved any.
 *
 * Returns true if the state was written by this node and now fits the domain, so the solvers
 * need not check its grids' transforms and extents again.
 */
static bool fit_state_to_domain(GeometrySet &state,
                                const openvdb::math::Transform &transform,
                                const int3 &resolution,
                                bool &r_moved)
{
  r_moved = false;
  const Volume *volume = state.get_volume();
  if (!volume) {
    return false;
  }
  /* A node downstream may rebuild one grid without the tag; any tagged grid marks the state. */
  bool tagged = false;
  constexpr int grid_count = int(std::size(state_grid_names));
  std::array<bool, grid_count> misfit{};
  for (int i = 0; i < grid_count; ++i) {
    const bke::VolumeGridData *grid_data = BKE_volume_grid_find(volume, state_grid_names[i]);
    if (!grid_data) {
      continue;
    }
    bke::VolumeTreeAccessToken token;
    const openvdb::GridBase &grid = grid_data->grid(token);
    if (const auto tag = grid.getMetadata<openvdb::BoolMetadata>(output_key)) {
      tagged = tagged || tag->value();
    }
    if (!ELEM(grid_data->grid_type(), VOLUME_GRID_FLOAT, VOLUME_GRID_VECTOR_FLOAT)) {
      /* Left for the solvers to reject. */
      return false;
    }
    std::string unused;
    misfit[i] = !(grid.transform() == transform) ||
                !active_values_fit_domain(grid, resolution, state_grid_names[i], unused);
  }
  if (!tagged) {
    return false;
  }
  if (std::none_of(misfit.begin(), misfit.end(), [](const bool b) { return b; })) {
    return true;
  }

  Volume *edit = state.get_volume_for_write();
  for (int i = 0; i < grid_count; ++i) {
    if (!misfit[i]) {
      continue;
    }
    const char *name = state_grid_names[i];
    const bke::VolumeGridData *grid_data = BKE_volume_grid_find(edit, name);
    openvdb::GridBase::Ptr fitted;
    {
      bke::VolumeTreeAccessToken token;
      const openvdb::GridBase &grid = grid_data->grid(token);
      if (grid_data->grid_type() == VOLUME_GRID_FLOAT) {
        fitted = fit_grid(
            static_cast<const openvdb::FloatGrid &>(grid), name, transform, resolution);
      }
      else {
        fitted = fit_grid(
            static_cast<const openvdb::Vec3SGrid &>(grid), name, transform, resolution);
      }
    }
    BKE_volume_grid_remove(edit, grid_data);
    BKE_volume_grid_add_vdb(*edit, name, std::move(fitted));
  }
  r_moved = true;
  return true;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet state = params.extract_input<GeometrySet>("State"_ustr);
  const float delta_time = params.extract_input<float>("Delta Time"_ustr);
  const int3 resolution(params.extract_input<int>("Resolution X"_ustr),
                        params.extract_input<int>("Resolution Y"_ustr),
                        params.extract_input<int>("Resolution Z"_ustr));
  const float3 origin = params.extract_input<float3>("Origin"_ustr);
  const float voxel_size = params.extract_input<float>("Voxel Size"_ustr);
  const float3 gravity = params.extract_input<float3>("Gravity"_ustr);

  if (delta_time == 0.0f) {
    params.set_output("State"_ustr, std::move(state));
    set_diagnostics(params, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }
  if (delta_time < 0.0f || !std::isfinite(delta_time)) {
    fail(params, state, "Delta Time must be finite and non-negative.");
    return;
  }
  if (!(voxel_size > 0.0f) || !std::isfinite(voxel_size) ||
      !BKE_volume_voxel_size_valid(float3(voxel_size)))
  {
    fail(params, state, "Voxel Size must be finite, positive, and supported by OpenVDB.");
    return;
  }
  if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) ||
      !std::isfinite(gravity.x) || !std::isfinite(gravity.y) || !std::isfinite(gravity.z))
  {
    fail(params, state, "Origin and Gravity must contain only finite values.");
    return;
  }

  try {
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(
        voxel_size);
    transform->postTranslate(openvdb::math::Vec3<double>(origin.x, origin.y, origin.z));

    bool moved = false;
    const bool checked = fit_state_to_domain(state, *transform, resolution, moved);
    if (moved) {
      params.error_message_add(NodeWarningType::Info,
                               TIP_("The voxel size or domain changed during the simulation, so "
                                    "the smoke was moved onto the new grid. Go back to the start "
                                    "frame for a clean restart"));
    }

    if (params.extract_input<bool>("Sparse GPU"_ustr)) {
      exec_sparse(
          params, state, resolution, *transform, voxel_size, gravity, delta_time, checked);
      return;
    }

    const std::size_t cell_count = static_cast<std::size_t>(resolution.x) *
                                   static_cast<std::size_t>(resolution.y) *
                                   static_cast<std::size_t>(resolution.z);
    std::vector<float> density(cell_count, 0.0f);
    std::vector<float> temperature(cell_count, 0.0f);
    std::vector<float> pressure(cell_count, 0.0f);
    std::vector<MantaGasVec3> velocity(cell_count);

    const auto import_start = std::chrono::steady_clock::now();
    const Volume *input_volume = state.get_volume();
    std::string error;
    MantaGasVelocityLayout velocity_layout = MantaGasVelocityLayout::CellCentered;
    if (!import_scalar_grid(
            input_volume, "density", *transform, resolution, checked, density.data(), error) ||
        !import_scalar_grid(input_volume,
                            "temperature",
                            *transform,
                            resolution,
                            checked,
                            temperature.data(),
                            error) ||
        !import_velocity_grid(input_volume,
                              *transform,
                              resolution,
                              voxel_size,
                              checked,
                              velocity.data(),
                              velocity_layout,
                              error) ||
        !import_scalar_grid(
            input_volume, "pressure", *transform, resolution, checked, pressure.data(), error))
    {
      fail(params, state, error);
      return;
    }
    const float import_ms = static_cast<float>(milliseconds_since(import_start));
    const GeometrySet source_geometry = params.extract_input<GeometrySet>("Sources"_ustr);
    const GeometrySet collider_geometry = params.extract_input<GeometrySet>("Colliders"_ustr);
    const bool sparse_only_inputs =
        source_geometry.has_volume() || collider_geometry.has_volume() ||
        params.extract_input<float>("Turbulence"_ustr) != 0.0f ||
        params.extract_input<GeometrySet>("Turbulence Layers"_ustr).has_pointcloud();

    MantaGasState manta_state;
    manta_state.resolution = {resolution.x, resolution.y, resolution.z};
    manta_state.density = density.data();
    manta_state.temperature = temperature.data();
    manta_state.velocity = velocity.data();
    manta_state.pressure = pressure.data();
    manta_state.velocity_layout = velocity_layout;

    MantaGasStepSettings settings;
    settings.delta_time = delta_time;
    settings.cfl = params.extract_input<float>("CFL"_ustr);
    settings.max_substeps = params.extract_input<int>("Max Substeps"_ustr);
    settings.gravity = {gravity.x / voxel_size, gravity.y / voxel_size, gravity.z / voxel_size};
    settings.density_buoyancy = params.extract_input<float>("Density Buoyancy"_ustr);
    settings.heat_buoyancy = params.extract_input<float>("Heat Buoyancy"_ustr);
    settings.pressure_accuracy = params.extract_input<float>("Pressure Accuracy"_ustr);
    settings.pressure_iteration_factor = params.extract_input<float>(
        "Pressure Iteration Factor"_ustr);
    settings.preconditioner = params.extract_input<bool>("Multigrid"_ustr) ?
                                  MantaGasPreconditioner::Multigrid :
                                  MantaGasPreconditioner::MIC;

    MantaGasStepDiagnostics diagnostics;
    const auto solve_start = std::chrono::steady_clock::now();
    if (!manta_gas_simulation_step(manta_state, settings, diagnostics, error)) {
      fail(params, state, error);
      return;
    }
    const float solve_ms = static_cast<float>(milliseconds_since(solve_start));

    const auto export_start = std::chrono::steady_clock::now();
    Volume *output_volume = BKE_id_new_nomain<Volume>("Gas Simulation Step State");
    BKE_volume_grid_add_vdb(*output_volume,
                            "density",
                            make_scalar_grid(density, resolution, *transform, true));
    BKE_volume_grid_add_vdb(*output_volume,
                            "temperature",
                            make_scalar_grid(temperature, resolution, *transform, true));
    BKE_volume_grid_add_vdb(*output_volume,
                            "velocity",
                            make_velocity_grid(velocity, resolution, *transform, voxel_size));
    BKE_volume_grid_add_vdb(*output_volume,
                            "pressure",
                            make_scalar_grid(pressure, resolution, *transform, false));
    const float export_ms = static_cast<float>(milliseconds_since(export_start));

    GeometrySet output_state = GeometrySet::from_volume(output_volume);
    params.set_output("State"_ustr, std::move(output_state));
    set_diagnostics(params,
                    diagnostics.substeps,
                    diagnostics.max_velocity * voxel_size,
                    import_ms,
                    solve_ms,
                    export_ms,
                    diagnostics.pressure_iterations,
                    diagnostics.pressure_residual);
    if (sparse_only_inputs) {
      params.error_message_add(
          NodeWarningType::Warning,
          TIP_("Sources, Colliders, Turbulence and fire are used only by the Sparse GPU "
               "solver"));
    }
    if (!diagnostics.pressure_converged) {
      params.error_message_add(
          NodeWarningType::Warning,
          TIP_("The pressure solve stopped at its iteration limit before reaching the requested "
               "accuracy"));
    }
  }
  catch (const std::exception &exception) {
    fail(params, state, std::string("Gas Simulation Step failed: ") + exception.what());
  }
  catch (...) {
    fail(params, state, "Gas Simulation Step failed with an unknown error.");
  }
}

#else

/* The node type stays registered so files and menus remain valid in every build configuration. */
static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet state = params.extract_input<GeometrySet>("State"_ustr);
  params.error_message_add(NodeWarningType::Error,
                           TIP_("Disabled, Blender was compiled without Mantaflow or OpenVDB"));
  params.set_output("State"_ustr, std::move(state));
  params.set_default_remaining_outputs();
}

#endif

static void node_register()
{
  static bke::bNodeType ntype;
  geo_node_type_base(&ntype, "GeometryNodeMantaGasStep"_ustr);
  ntype.ui_name = "Gas Simulation Step";
  ntype.ui_description =
      "Advance a closed-domain smoke state with compiled Mantaflow operators, or with the sparse "
      "GPU solver. Use inside a Geometry Nodes Simulation Zone";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_manta_gas_step_cc
