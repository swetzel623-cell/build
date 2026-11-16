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
#  include "openvdb/tools/Interpolation.h"

#  include <chrono>
#  include <cmath>
#  include <memory>
#  include <mutex>
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
 * One Vulkan device for the whole process, created on first use. It holds compiled kernels and
 * buffers, never simulation state: every evaluation uploads its own state and downloads the
 * result while holding the lock, so independent zones do not share solver state. It is never
 * destroyed, because tearing down Vulkan objects during process exit is unsafe with some
 * drivers.
 */
struct SparseDevice {
  std::mutex mutex;
  manta_improve::gpu::SparseGasGpu *gpu = nullptr;
  bool tried = false;
  std::string error;
};

static SparseDevice &sparse_device()
{
  static SparseDevice device;
  return device;
}
#  endif

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
 * Write the active values of a grid with any transform into a dense domain array: each active
 * voxel goes to the nearest domain voxel, and where several land on one, the largest wins.
 */
static void rasterize(const openvdb::FloatGrid &grid,
                      const openvdb::math::Transform &domain,
                      const int3 &resolution,
                      std::vector<float> &output)
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
      float &target = output[manta_gas::grid_index(resolution.x, resolution.y, x, y, z)];
      target = std::max(target, value);
    }
  }
}

/** Sample a world-space velocity grid at the domain voxels where `mask` is set, in cells/s. */
static void sample_velocity(const openvdb::Vec3SGrid &grid,
                            const openvdb::math::Transform &domain,
                            const int3 &resolution,
                            const float voxel_size,
                            const std::vector<bool> &mask,
                            std::vector<float> &u,
                            std::vector<float> &v,
                            std::vector<float> &w)
{
  const std::size_t count = mask.size();
  u.assign(count, 0.0f);
  v.assign(count, 0.0f);
  w.assign(count, 0.0f);
  openvdb::tools::GridSampler<openvdb::Vec3SGrid, openvdb::tools::BoxSampler> sampler(grid);
  for (int z = 0; z < resolution.z; ++z) {
    for (int y = 0; y < resolution.y; ++y) {
      for (int x = 0; x < resolution.x; ++x) {
        const std::size_t i = manta_gas::grid_index(resolution.x, resolution.y, x, y, z);
        if (!mask[i]) {
          continue;
        }
        const openvdb::Vec3f value = sampler.wsSample(domain.indexToWorld(openvdb::Vec3d(x, y, z)));
        u[i] = value[0] / voxel_size;
        v[i] = value[1] / voxel_size;
        w[i] = value[2] / voxel_size;
      }
    }
  }
}

/**
 * Advance the dense arrays in place with the sparse solver. Velocity is in cells per second and
 * comes back staggered.
 */
static void run_sparse_step(const int3 &resolution,
                            const MantaGasStepSettings &step_settings,
                            std::vector<float> &density,
                            std::vector<float> &temperature,
                            std::vector<float> &pressure,
                            std::vector<MantaGasVec3> &velocity,
                            const MantaGasVelocityLayout velocity_layout,
                            const sparse::Sources &sources,
                            std::vector<std::uint8_t> obstacle,
                            std::vector<float> obstacle_u,
                            std::vector<float> obstacle_v,
                            std::vector<float> obstacle_w,
                            SparseRun &run)
{
  const int nx = resolution.x, ny = resolution.y, nz = resolution.z;
  sparse::DenseFields fields;
  fields.extent = {nx, ny, nz};
  fields.density = density;
  fields.temperature = temperature;
  fields.pressure = pressure;
  fields.obstacle = std::move(obstacle);
  fields.obstacle_u = std::move(obstacle_u);
  fields.obstacle_v = std::move(obstacle_v);
  fields.obstacle_w = std::move(obstacle_w);
  const std::size_t count = density.size();
  fields.u.assign(count, 0.0f);
  fields.v.assign(count, 0.0f);
  fields.w.assign(count, 0.0f);
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const std::size_t i = manta_gas::grid_index(nx, ny, x, y, z);
        if (velocity_layout == MantaGasVelocityLayout::Staggered) {
          fields.u[i] = velocity[i].x;
          fields.v[i] = velocity[i].y;
          fields.w[i] = velocity[i].z;
          continue;
        }
        /* Authored cell-centered input: average onto the lower faces once, as Mantaflow's
         * resampleVec3ToMac does. */
        if (x > 0) {
          fields.u[i] = 0.5f * (velocity[manta_gas::grid_index(nx, ny, x - 1, y, z)].x +
                                velocity[i].x);
        }
        if (y > 0) {
          fields.v[i] = 0.5f * (velocity[manta_gas::grid_index(nx, ny, x, y - 1, z)].y +
                                velocity[i].y);
        }
        if (z > 0) {
          fields.w[i] = 0.5f * (velocity[manta_gas::grid_index(nx, ny, x, y, z - 1)].z +
                                velocity[i].z);
        }
      }
    }
  }

  sparse::Settings settings;
  settings.cfl = step_settings.cfl;
  settings.max_substeps = step_settings.max_substeps;
  settings.gravity[0] = step_settings.gravity.x;
  settings.gravity[1] = step_settings.gravity.y;
  settings.gravity[2] = step_settings.gravity.z;
  settings.density_buoyancy = step_settings.density_buoyancy;
  settings.heat_buoyancy = step_settings.heat_buoyancy;
  settings.pressure_solver = sparse::PressureSolver::Multigrid;
  settings.pressure_iterations = 6;

  sparse::State state = sparse::from_dense(fields, 8, settings.thresholds, settings.dilation);
  fields = {};
  sparse::FrameInfo frame;
  bool done = false;
#  ifdef WITH_SPARSE_GAS_GPU
  {
    SparseDevice &device = sparse_device();
    std::lock_guard<std::mutex> lock(device.mutex);
    if (!device.tried) {
      device.tried = true;
      try {
        device.gpu = new manta_improve::gpu::SparseGasGpu();
      }
      catch (const std::exception &exception) {
        device.error = exception.what();
      }
    }
    if (device.gpu) {
      device.gpu->upload(state);
      frame = device.gpu->advance(settings, step_settings.delta_time, &sources);
      device.gpu->download(state);
      run.device_name = device.gpu->device_name();
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
    frame = sparse::advance(state, settings, step_settings.delta_time, &sources);
  }

  run.substeps = frame.substeps;
  run.max_speed = frame.max_speed;
  run.active_bricks = static_cast<int>(state.topology.brick_count());
  const sparse::DenseFields result = sparse::to_dense(state);
  density = result.density;
  temperature = result.temperature;
  pressure = result.pressure;
  for (std::size_t i = 0; i < count; ++i) {
    velocity[i] = {result.u[i], result.v[i], result.w[i]};
  }
}

static double milliseconds_since(const std::chrono::steady_clock::time_point &start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
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
    const std::size_t cell_count = static_cast<std::size_t>(resolution.x) *
                                   static_cast<std::size_t>(resolution.y) *
                                   static_cast<std::size_t>(resolution.z);
    std::vector<float> density(cell_count, 0.0f);
    std::vector<float> temperature(cell_count, 0.0f);
    std::vector<float> pressure(cell_count, 0.0f);
    std::vector<MantaGasVec3> velocity(cell_count);

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(
        voxel_size);
    transform->postTranslate(openvdb::math::Vec3<double>(origin.x, origin.y, origin.z));

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
    /* Sources and colliders, evaluated every frame, so they may be animated. */
    const bool sparse_gpu = params.extract_input<bool>("Sparse GPU"_ustr);
    const GeometrySet source_geometry = params.extract_input<GeometrySet>("Sources"_ustr);
    const bool add_sources = params.extract_input<bool>("Add Sources"_ustr);
    const GeometrySet collider_geometry = params.extract_input<GeometrySet>("Colliders"_ustr);
    const Volume *source_volume = source_geometry.get_volume();
    const Volume *collider_volume = collider_geometry.get_volume();
    sparse::DenseFields emit;
    std::vector<std::uint8_t> solid;
    std::vector<float> obstacle_u, obstacle_v, obstacle_w;
    bool has_sources = false, has_colliders = false;
    try {
      bke::VolumeTreeAccessToken token;
      const openvdb::FloatGrid *source_density = find_float_grid(source_volume, "density", token);
      const openvdb::FloatGrid *source_temperature = find_float_grid(
          source_volume, "temperature", token);
      const openvdb::Vec3SGrid *source_velocity = find_vector_grid(
          source_volume, "velocity", token);
      const openvdb::FloatGrid *collider = find_float_grid(collider_volume, "obstacle", token);
      const openvdb::Vec3SGrid *collider_velocity = find_vector_grid(
          collider_volume, "velocity", token);
      has_sources = source_density || source_temperature;
      has_colliders = collider != nullptr;
      if (sparse_gpu && has_sources) {
        emit.extent = {resolution.x, resolution.y, resolution.z};
        emit.density.assign(cell_count, 0.0f);
        emit.temperature.assign(cell_count, 0.0f);
        if (source_density) {
          rasterize(*source_density, *transform, resolution, emit.density);
        }
        if (source_temperature) {
          rasterize(*source_temperature, *transform, resolution, emit.temperature);
        }
        if (source_velocity) {
          std::vector<bool> emitting(cell_count);
          for (std::size_t i = 0; i < cell_count; ++i) {
            emitting[i] = emit.density[i] != 0.0f || emit.temperature[i] != 0.0f;
          }
          sample_velocity(*source_velocity, *transform, resolution, voxel_size, emitting,
                          emit.u, emit.v, emit.w);
        }
      }
      if (sparse_gpu && has_colliders) {
        std::vector<float> values(cell_count, 0.0f);
        rasterize(*collider, *transform, resolution, values);
        solid.resize(cell_count);
        std::vector<bool> mask(cell_count);
        for (std::size_t i = 0; i < cell_count; ++i) {
          solid[i] = values[i] > 0.5f;
          mask[i] = solid[i];
        }
        if (collider_velocity) {
          sample_velocity(*collider_velocity, *transform, resolution, voxel_size, mask,
                          obstacle_u, obstacle_v, obstacle_w);
        }
      }
    }
    catch (const std::exception &exception) {
      fail(params, state, exception.what());
      return;
    }
    const float import_ms = static_cast<float>(milliseconds_since(import_start));

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
    SparseRun sparse_run;
    const auto solve_start = std::chrono::steady_clock::now();
    if (sparse_gpu) {
      try {
        sparse::Sources sources;
        sources.mode = add_sources ? sparse::SourceMode::Add : sparse::SourceMode::Raise;
        if (has_sources) {
          sources = sparse::sources_from_dense(emit, 8, sources.mode);
        }
        run_sparse_step(resolution,
                        settings,
                        density,
                        temperature,
                        pressure,
                        velocity,
                        velocity_layout,
                        sources,
                        std::move(solid),
                        std::move(obstacle_u),
                        std::move(obstacle_v),
                        std::move(obstacle_w),
                        sparse_run);
      }
      catch (const std::exception &exception) {
        fail(params, state, std::string("Sparse solver failed: ") + exception.what());
        return;
      }
      diagnostics.substeps = sparse_run.substeps;
      diagnostics.max_velocity = sparse_run.max_speed;
      diagnostics.pressure_iterations = 6;
    }
    else {
      const bool success = manta_gas_simulation_step(manta_state, settings, diagnostics, error);
      if (!success) {
        fail(params, state, error);
        return;
      }
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
                    diagnostics.pressure_residual,
                    sparse_run.active_bricks);
    if (sparse_gpu && !sparse_run.fallback_reason.empty()) {
      params.error_message_add(NodeWarningType::Warning,
                               "Sparse solver ran on the CPU, which is slow: " +
                                   sparse_run.fallback_reason);
    }
    else if (sparse_gpu) {
      params.error_message_add(NodeWarningType::Info,
                               "Sparse solver ran on " + sparse_run.device_name);
    }
    if (!sparse_gpu && (has_sources || has_colliders)) {
      params.error_message_add(NodeWarningType::Warning,
                               TIP_("Sources and Colliders are used only by the Sparse GPU solver"));
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
