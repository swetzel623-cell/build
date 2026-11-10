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

#  include "openvdb/openvdb.h"

#  include <chrono>
#  include <cmath>
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
  b.add_input<decl::Float>("Pressure Iteration Factor"_ustr)
      .default_value(1.5f)
      .min(0.1f)
      .max(20.0f)
      .description("Multiplier for Mantaflow's resolution-dependent pressure iteration limit");

  b.add_output<decl::Int>("Substeps"_ustr).description("Number of internal solver steps executed");
  b.add_output<decl::Float>("Max Velocity"_ustr)
      .description("Largest sampled solver velocity in world units per second");
  b.add_output<decl::Float>("Import Time"_ustr)
      .description("OpenVDB-to-dense conversion time in milliseconds");
  b.add_output<decl::Float>("Solve Time"_ustr)
      .description("Mantaflow allocation and simulation time in milliseconds");
  b.add_output<decl::Float>("Export Time"_ustr)
      .description("Dense-to-OpenVDB conversion time in milliseconds");
}

#if defined(WITH_FLUID) && defined(WITH_OPENVDB)

static void set_diagnostics(GeoNodeExecParams &params,
                            const int substeps,
                            const float max_velocity,
                            const float import_ms,
                            const float solve_ms,
                            const float export_ms)
{
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

    MantaGasStepDiagnostics diagnostics;
    const auto solve_start = std::chrono::steady_clock::now();
    const bool success = manta_gas_simulation_step(manta_state, settings, diagnostics, error);
    const float solve_ms = static_cast<float>(milliseconds_since(solve_start));
    if (!success) {
      fail(params, state, error);
      return;
    }

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
                    export_ms);
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
      "Advance a closed-domain smoke state with compiled Mantaflow operators. Use inside a "
      "Geometry Nodes Simulation Zone";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_manta_gas_step_cc
