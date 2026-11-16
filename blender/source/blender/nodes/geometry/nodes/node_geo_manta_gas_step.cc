/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "node_geometry_util.hh"

#include "BKE_geometry_set.hh"
#include "BKE_lib_id.hh"
#include "BKE_volume.hh"
#include "BKE_volume_grid.hh"
#include "BKE_volume_openvdb.hh"

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

#  include <algorithm>
#  include <array>
#  include <chrono>
#  include <cmath>
#  include <memory>
#  include <mutex>
#  include <optional>
#  include <random>
#  include <string>
#  include <vector>
#endif

namespace blender::nodes::node_geo_manta_gas_step_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("State"_ustr)
      .description(
          "Volume containing density, temperature, velocity, and pressure grids. Missing grids "
          "start at zero. A velocity grid that is not tagged as staggered is treated as "
          "cell-centered and resampled to faces");
  b.add_output<decl::Geometry>("State"_ustr)
      .description(
          "Serializable next simulation state containing the four named volume grids. Velocity "
          "is the solver's staggered (face-centered) field");

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
          "Sparse GPU only. Volume whose float grids 'density' and 'temperature' are emitted "
          "every step, for example from Mesh to Density Grid. An optional vector grid "
          "'velocity' (world units per second) is set where they emit. Grids may use any "
          "transform; each active voxel goes to the nearest domain voxel");
  b.add_input<decl::Bool>("Add Sources"_ustr)
      .default_value(false)
      .description(
          "Add the sources' values per second instead of raising density and temperature to "
          "them");
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
  if (!(grid.transform() == expected_transform)) {
    error = std::string("Grid '") + name +
            "' does not match the requested origin and voxel-size transform.";
    return false;
  }
  if (!active_values_fit_domain(grid, resolution, name, error)) {
    return false;
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
  if (!(grid.transform() == expected_transform)) {
    error = "Grid 'velocity' does not match the requested origin and voxel-size transform.";
    return false;
  }
  if (!active_values_fit_domain(grid, resolution, name, error)) {
    return false;
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

static openvdb::FloatGrid::Ptr make_scalar_grid(const std::vector<float> &values,
                                                const int3 &resolution,
                                                const openvdb::math::Transform &transform,
                                                const bool fog_volume)
{
  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
  grid->setTransform(transform.copy());
  grid->setGridClass(fog_volume ? openvdb::GRID_FOG_VOLUME : openvdb::GRID_UNKNOWN);
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
  std::array<const void *, 4> trees{};
  std::array<int, 3> resolution{};
  std::uint64_t last_used = 0;
};

struct SparseDevice {
  std::mutex mutex;
  std::vector<SparseSlot> slots;
  bool tried = false;
  std::string error;
  std::uint64_t clock = 0;
};

static constexpr std::size_t sparse_slot_limit = 4;

static SparseDevice &sparse_device()
{
  static SparseDevice device;
  return device;
}
#  endif

/** Metadata on the density grid that names the solver state a node wrote. */
static constexpr const char *sparse_state_key = "gas_step_state_id";
/** Metadata on the density grid of every state this node wrote. Only such a state is moved onto
 * a changed domain; an authored initial state must match the domain exactly. */
static constexpr const char *output_key = "gas_step_output";

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

/** The sources of one frame, on bricks, straight from their grids. */
static sparse::Sources make_sources(const openvdb::FloatGrid *density,
                                    const openvdb::FloatGrid *temperature,
                                    const openvdb::Vec3SGrid *velocity,
                                    const openvdb::math::Transform &domain,
                                    const int3 &resolution,
                                    const float voxel_size,
                                    const bool add)
{
  sparse::Sources sources;
  sources.mode = add ? sparse::SourceMode::Add : sparse::SourceMode::Raise;
  if (!density && !temperature) {
    return sources;
  }
  sources.field = sparse::make_brick_field(
      {resolution.x, resolution.y, resolution.z}, 8, velocity ? 5 : 2);
  sparse::BrickField &field = sources.field;
  for (int channel = 0; channel < 2; ++channel) {
    const openvdb::FloatGrid *grid = channel == 0 ? density : temperature;
    if (!grid) {
      continue;
    }
    for_each_active_voxel(*grid, domain, resolution, [&](int x, int y, int z, float value) {
      float &target = field.touch(std::size_t(channel), x, y, z);
      target = std::max(target, value);
    });
  }
  if (velocity) {
    openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler> sampler(
        *velocity);
    for (std::size_t brick = 0; brick < field.table.size(); ++brick) {
      const std::int32_t slot = field.table[brick];
      if (slot < 0) {
        continue;
      }
      const int bx = int(brick % std::size_t(field.brick_extent.x));
      const int by = int(brick / std::size_t(field.brick_extent.x) %
                         std::size_t(field.brick_extent.y));
      const int bz = int(brick / (std::size_t(field.brick_extent.x) * field.brick_extent.y));
      for (int local = 0; local < 512; ++local) {
        const std::size_t i = std::size_t(slot) * 512 + std::size_t(local);
        if (field.channels[0][i] == 0.0f && field.channels[1][i] == 0.0f) {
          continue;
        }
        const openvdb::Vec3f value = velocity_at(sampler,
                                                 domain,
                                                 voxel_size,
                                                 bx * 8 + (local & 7),
                                                 by * 8 + ((local >> 3) & 7),
                                                 bz * 8 + (local >> 6));
        field.channels[2][i] = value[0];
        field.channels[3][i] = value[1];
        field.channels[4][i] = value[2];
      }
    }
  }
  return sources;
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

  std::array<const void *, 4> trees() const
  {
    return {density ? &density->tree() : nullptr,
            temperature ? &temperature->tree() : nullptr,
            velocity ? &velocity->tree() : nullptr,
            pressure ? &pressure->tree() : nullptr};
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
  for (const openvdb::FloatGrid *grid : {grids.density, grids.temperature}) {
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
  std::optional<openvdb::FloatGrid::ConstAccessor> density, temperature, pressure;
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

/** Write a solver state into a new volume, brick by brick, tagging it with `state_id`. */
static Volume *export_sparse_state(const sparse::State &state,
                                   const int3 &resolution,
                                   const openvdb::math::Transform &transform,
                                   const float voxel_size,
                                   const std::int64_t state_id,
                                   std::array<const void *, 4> &trees)
{
  const auto scalar = [&](const bool fog) {
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(transform.copy());
    grid->setGridClass(fog ? openvdb::GRID_FOG_VOLUME : openvdb::GRID_UNKNOWN);
    return grid;
  };
  openvdb::FloatGrid::Ptr density = scalar(true), temperature = scalar(true),
                          pressure = scalar(false);
  openvdb::Vec3SGrid::Ptr velocity = openvdb::Vec3SGrid::create(openvdb::Vec3f::zero());
  velocity->setTransform(transform.copy());
  velocity->setGridClass(openvdb::GRID_STAGGERED);
  density->insertMeta(sparse_state_key, openvdb::Int64Metadata(state_id));
  density->insertMeta(output_key, openvdb::BoolMetadata(true));

  {
    openvdb::FloatGrid::Accessor density_access = density->getAccessor();
    openvdb::FloatGrid::Accessor temperature_access = temperature->getAccessor();
    openvdb::FloatGrid::Accessor pressure_access = pressure->getAccessor();
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
        if (state.pressure[i] != 0.0f) {
          pressure_access.setValue(c, state.pressure[i]);
        }
        if (state.u[i] != 0.0f || state.v[i] != 0.0f || state.w[i] != 0.0f) {
          velocity_access.setValue(c,
                                   openvdb::Vec3f(state.u[i] * voxel_size,
                                                  state.v[i] * voxel_size,
                                                  state.w[i] * voxel_size));
        }
      }
    }
  }
  trees = {&density->tree(), &temperature->tree(), &velocity->tree(), &pressure->tree()};
  Volume *volume = BKE_id_new_nomain<Volume>("Gas Simulation Step State");
  /* The volume must own each grid alone. */
  BKE_volume_grid_add_vdb(*volume, "density", std::move(density));
  BKE_volume_grid_add_vdb(*volume, "temperature", std::move(temperature));
  BKE_volume_grid_add_vdb(*volume, "velocity", std::move(velocity));
  BKE_volume_grid_add_vdb(*volume, "pressure", std::move(pressure));
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
                             const int3 &resolution)
{
  if (!grid) {
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

/**
 * The Sparse GPU path. The state crosses OpenVDB only where it has bricks, and a solver that
 * wrote this node's input last frame continues from its own device state.
 */
static void exec_sparse(GeoNodeExecParams &params,
                        GeometrySet &state,
                        const int3 &resolution,
                        const openvdb::math::Transform &transform,
                        const float voxel_size,
                        const float3 &gravity,
                        const float delta_time)
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
    check_state_grid(grids.density, "density", transform, resolution);
    check_state_grid(grids.temperature, "temperature", transform, resolution);
    check_state_grid(grids.velocity, "velocity", transform, resolution);
    check_state_grid(grids.pressure, "pressure", transform, resolution);
    /* The node's own output is tagged as staggered; any other vector grid is an authored
     * cell-centered velocity. */
    staggered = grids.velocity && grids.velocity->getGridClass() == openvdb::GRID_STAGGERED;

    const Volume *source_volume = source_geometry.get_volume();
    sources = make_sources(find_float_grid(source_volume, "density", token),
                           find_float_grid(source_volume, "temperature", token),
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
  }
  catch (const std::exception &exception) {
    fail(params, state, exception.what());
    return;
  }
  const std::int64_t input_id = grids.state_id();
  const std::array<const void *, 4> input_trees = grids.trees();
  double import_ms = milliseconds_since(import_start);

  const std::int64_t output_id = next_state_id();
  std::array<const void *, 4> output_trees{};
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
    const auto finish = [&](const sparse::State &result, const sparse::FrameInfo &frame) {
      run.substeps = frame.substeps;
      run.max_speed = frame.max_speed;
      run.active_bricks = static_cast<int>(result.topology.brick_count());
      const auto start = std::chrono::steady_clock::now();
      output_volume = export_sparse_state(
          result, resolution, transform, voxel_size, output_id, output_trees);
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
        /* Until this evaluation finishes, the slot holds no state anyone may continue. */
        slot->state_id = 0;
        sparse::State fresh;
        if (!continued) {
          fresh = import_state();
        }
        const auto solve_start = std::chrono::steady_clock::now();
        if (continued) {
          slot->gpu->set_obstacles(std::move(obstacle), std::move(obstacle_velocity));
        }
        else {
          slot->gpu->upload(fresh);
        }
        const sparse::FrameInfo frame = slot->gpu->advance(settings, delta_time, &sources);
        sparse::State result;
        slot->gpu->download(result);
        solve_ms = milliseconds_since(solve_start);
        finish(result, frame);
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
      finish(result, frame);
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

/** A copy of `grid` on `transform`, resampled if it was on another one, and without values
 * outside the domain. */
template<typename GridT>
static openvdb::GridBase::Ptr fit_grid(const GridT &grid,
                                       const openvdb::math::Transform &transform,
                                       const int3 &resolution)
{
  typename GridT::Ptr result;
  if (grid.transform() == transform) {
    result = grid.deepCopy();
  }
  else {
    result = GridT::create(grid.background());
    result->setTransform(transform.copy());
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(grid, *result);
    result->setGridClass(grid.getGridClass());
    result->setVectorType(grid.getVectorType());
  }
  result->removeMeta(sparse_state_key);
  result->tree().clip(openvdb::CoordBBox(
      openvdb::Coord(0), openvdb::Coord(resolution.x - 1, resolution.y - 1, resolution.z - 1)));
  return result;
}

/**
 * A simulation zone keeps its state when the domain's settings change part-way through, so the
 * state can arrive on another voxel size or origin, or reach past a smaller domain. Moves such
 * a state, if this node wrote it, onto the requested domain. Returns true if anything was
 * moved.
 */
static bool fit_state_to_domain(GeometrySet &state,
                                const openvdb::math::Transform &transform,
                                const int3 &resolution)
{
  static constexpr const char *names[] = {"density", "temperature", "velocity", "pressure"};
  const Volume *volume = state.get_volume();
  if (!volume) {
    return false;
  }
  if (const bke::VolumeGridData *density = BKE_volume_grid_find(volume, "density")) {
    bke::VolumeTreeAccessToken token;
    const openvdb::GridBase &grid = density->grid(token);
    const auto tag = grid.getMetadata<openvdb::BoolMetadata>(output_key);
    if (!tag || !tag->value()) {
      return false;
    }
  }
  else {
    return false;
  }
  bool needed = false;
  for (const char *name : names) {
    const bke::VolumeGridData *grid_data = BKE_volume_grid_find(volume, name);
    if (!grid_data || !ELEM(grid_data->grid_type(), VOLUME_GRID_FLOAT, VOLUME_GRID_VECTOR_FLOAT))
    {
      continue;
    }
    bke::VolumeTreeAccessToken token;
    const openvdb::GridBase &grid = grid_data->grid(token);
    std::string unused;
    if (!(grid.transform() == transform) ||
        !active_values_fit_domain(grid, resolution, name, unused))
    {
      needed = true;
    }
  }
  if (!needed) {
    return false;
  }

  Volume *edit = state.get_volume_for_write();
  for (const char *name : names) {
    const bke::VolumeGridData *grid_data = BKE_volume_grid_find(edit, name);
    if (!grid_data) {
      continue;
    }
    openvdb::GridBase::Ptr fitted;
    {
      bke::VolumeTreeAccessToken token;
      const openvdb::GridBase &grid = grid_data->grid(token);
      if (grid_data->grid_type() == VOLUME_GRID_FLOAT) {
        fitted = fit_grid(static_cast<const openvdb::FloatGrid &>(grid), transform, resolution);
      }
      else if (grid_data->grid_type() == VOLUME_GRID_VECTOR_FLOAT) {
        fitted = fit_grid(static_cast<const openvdb::Vec3SGrid &>(grid), transform, resolution);
      }
    }
    if (fitted) {
      BKE_volume_grid_remove(edit, grid_data);
      BKE_volume_grid_add_vdb(*edit, name, std::move(fitted));
    }
  }
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

    if (fit_state_to_domain(state, *transform, resolution)) {
      params.error_message_add(NodeWarningType::Info,
                               TIP_("The voxel size or domain changed during the simulation, so "
                                    "the smoke was moved onto the new grid. Go back to the start "
                                    "frame for a clean restart"));
    }

    if (params.extract_input<bool>("Sparse GPU"_ustr)) {
      exec_sparse(params, state, resolution, *transform, voxel_size, gravity, delta_time);
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
    if (!import_scalar_grid(input_volume, "density", *transform, resolution, density.data(), error) ||
        !import_scalar_grid(
            input_volume, "temperature", *transform, resolution, temperature.data(), error) ||
        !import_velocity_grid(input_volume,
                              *transform,
                              resolution,
                              voxel_size,
                              velocity.data(),
                              velocity_layout,
                              error) ||
        !import_scalar_grid(
            input_volume, "pressure", *transform, resolution, pressure.data(), error))
    {
      fail(params, state, error);
      return;
    }
    const float import_ms = static_cast<float>(milliseconds_since(import_start));
    const GeometrySet source_geometry = params.extract_input<GeometrySet>("Sources"_ustr);
    const GeometrySet collider_geometry = params.extract_input<GeometrySet>("Colliders"_ustr);
    const bool sparse_only_inputs = source_geometry.has_volume() ||
                                    collider_geometry.has_volume() ||
                                    params.extract_input<float>("Turbulence"_ustr) != 0.0f;

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
    openvdb::FloatGrid::Ptr density_grid = make_scalar_grid(
        density, resolution, *transform, true);
    density_grid->insertMeta(output_key, openvdb::BoolMetadata(true));
    BKE_volume_grid_add_vdb(*output_volume, "density", std::move(density_grid));
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
          TIP_("Sources, Colliders and Turbulence are used only by the Sparse GPU solver"));
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
