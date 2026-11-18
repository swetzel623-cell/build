// SPDX-License-Identifier: MIT
#include "manta_improve/sparse_gas.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace manta_improve::sparse;

constexpr int everywhere = 1000; /* Dilation that activates every brick. */

void require(const bool condition, const std::string &message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::size_t dense_index(const Extent &extent, int x, int y, int z)
{
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(extent.x) *
             (static_cast<std::size_t>(y) + static_cast<std::size_t>(extent.y) * z);
}

DenseFields blob(const Extent extent, const float cx, const float cy, const float cz, const float r)
{
  DenseFields dense;
  dense.extent = extent;
  const std::size_t count = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
  dense.density.assign(count, 0.0f);
  dense.temperature.assign(count, 0.0f);
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const float dx = x + 0.5f - cx, dy = y + 0.5f - cy, dz = z + 0.5f - cz;
        if (dx * dx + dy * dy + dz * dz <= r * r) {
          dense.density[dense_index(extent, x, y, z)] = 1.0f;
          dense.temperature[dense_index(extent, x, y, z)] = 1.0f;
        }
      }
    }
  }
  return dense;
}

double sum(const std::vector<float> &values)
{
  double total = 0.0;
  for (const float value : values) {
    total += value;
  }
  return total;
}

double center_of_mass_z(const DenseFields &dense)
{
  double weighted = 0.0;
  for (int z = 0; z < dense.extent.z; ++z) {
    for (int y = 0; y < dense.extent.y; ++y) {
      for (int x = 0; x < dense.extent.x; ++x) {
        weighted += (z + 0.5) * dense.density[dense_index(dense.extent, x, y, z)];
      }
    }
  }
  return weighted / sum(dense.density);
}

double relative_l2(const std::vector<float> &a, const std::vector<float> &b)
{
  double difference = 0.0, norm = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    difference += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
    norm += double(b[i]) * b[i];
  }
  return std::sqrt(difference / std::max(norm, 1.0e-30));
}

Settings plume_settings()
{
  Settings settings;
  settings.dt = 1.0f / 30.0f;
  settings.gravity[2] = -9.81f * 16.0f; /* 16 voxels per world unit. */
  settings.heat_buoyancy = 0.5f;
  settings.density_buoyancy = 0.5f;
  return settings;
}

void test_topology_and_cell_types()
{
  const Extent extent = {32, 32, 32};
  std::vector<std::uint8_t> flags(4 * 4 * 4, 0);
  flags[1 + 4 * (1 + 4 * 1)] = 1;
  const Topology isolated = build_topology(extent, 8, flags, 0);
  require(isolated.brick_count() == 1 && isolated.voxel_count() == 512, "one brick expected");
  require(isolated.index_of(8, 8, 8) == 0 && isolated.index_of(15, 15, 15) == 511,
          "voxel addressing is not X-fastest inside the brick");
  require(isolated.index_of(7, 8, 8) == -1 && isolated.index_of(16, 8, 8) == -1,
          "voxels outside the brick must have no storage");
  require(isolated.type_at(10, 10, 10) == CellFluid, "interior voxel must be fluid");
  require(isolated.type_at(15, 10, 10) == CellOpen && isolated.type_at(10, 10, 15) == CellOpen,
          "the upper layer of an isolated brick holds open-air faces");
  require(isolated.type_at(8, 10, 10) == CellOpen && isolated.type_at(10, 10, 8) == CellOpen,
          "the lower layer must be open air too, so that the boundary is the same on every side");
  require(isolated.type_at(9, 9, 9) == CellFluid && isolated.type_at(14, 14, 14) == CellFluid,
          "one voxel inside the brick's surface must be fluid");
  require(isolated.type_at(20, 20, 20) == CellOpen, "unstored interior is open air");
  require(isolated.type_at(0, 5, 5) == CellSolid && isolated.type_at(31, 5, 5) == CellSolid,
          "domain boundary layer must be solid");

  const Topology dilated = build_topology(extent, 8, flags, 1);
  require(dilated.brick_count() == 27, "dilation by one brick must give 27 bricks");
  require(dilated.type_at(15, 10, 10) == CellFluid, "dilation must turn the ghost layer fluid");

  const Topology full = build_topology(extent, 8, flags, everywhere);
  require(full.brick_count() == 64, "full dilation must activate every brick");
  require(full.type_at(30, 30, 30) == CellFluid, "cells below a wall need no ghost layer");
  std::size_t open = 0;
  for (const std::uint32_t type : full.cell_type) {
    open += type == CellOpen;
  }
  require(open == 0, "a fully active closed box has no open cells");

  /* Bricks may overhang a domain that is not a multiple of the brick size. */
  const Topology ragged = build_topology({20, 12, 9}, 8, std::vector<std::uint8_t>(3 * 2 * 2, 1), 0);
  require(ragged.type_at(19, 5, 5) == CellSolid && ragged.type_at(21, 5, 5) == CellSolid,
          "overhanging voxels must be solid");
}

void test_dense_round_trip()
{
  DenseFields dense = blob({32, 32, 32}, 12.0f, 14.0f, 10.0f, 4.0f);
  dense.u.assign(dense.density.size(), 0.0f);
  dense.v = dense.u;
  dense.w = dense.u;
  dense.w[dense_index(dense.extent, 12, 14, 10)] = 3.0f;
  ActivityThresholds thresholds;
  thresholds.velocity = 1.0e-4f;
  const State state = from_dense(dense, 8, thresholds, 0);
  require(state.topology.brick_count() < 64, "the blob must not activate the whole domain");
  const DenseFields back = to_dense(state);
  require(back.density == dense.density && back.temperature == dense.temperature &&
              back.w == dense.w,
          "dense round trip changed values");
}

void test_projection_removes_divergence()
{
  const Extent extent = {24, 24, 24};
  DenseFields dense = blob(extent, 12.0f, 12.0f, 12.0f, 3.0f);
  dense.u.assign(dense.density.size(), 0.0f);
  dense.v = dense.u;
  dense.w = dense.u;
  std::srand(7);
  for (float &value : dense.w) {
    value = static_cast<float>(std::rand() % 2001 - 1000) / 250.0f;
  }
  for (float &value : dense.u) {
    value = static_cast<float>(std::rand() % 2001 - 1000) / 250.0f;
  }

  for (const int dilation : {0, everywhere}) {
    State state = from_dense(dense, 8, {}, dilation);
    enforce_walls(state);
    std::vector<float> before;
    divergence(state, before);
    Settings settings;
    settings.pressure_accuracy = 1.0e-4f;
    settings.pressure_iterations = 2000;
    const StepInfo info = project(state, settings);
    float initial = 0.0f;
    for (const float value : before) {
      initial = std::max(initial, std::abs(value));
    }
    require(initial > 1.0f, "test velocity has no divergence");
    require(info.max_divergence < 1.0e-3f,
            "projection left divergence " + std::to_string(info.max_divergence));
    const DenseFields result = to_dense(state);
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        require(result.w[dense_index(extent, x, y, 1)] == 0.0f &&
                    result.w[dense_index(extent, x, y, extent.z - 1)] == 0.0f,
                "flow through a closed wall");
      }
    }
  }
}

DenseFields run(const DenseFields &initial, int brick_size, int dilation, int steps,
                Settings settings, std::size_t *max_bricks = nullptr)
{
  settings.dilation = dilation;
  State state = from_dense(initial, brick_size, settings.thresholds, dilation);
  for (int i = 0; i < steps; ++i) {
    const StepInfo info = step(state, settings);
    if (max_bricks) {
      *max_bricks = std::max(*max_bricks, info.active_bricks);
    }
  }
  return to_dense(state);
}

/** The brick decomposition is bookkeeping: with every brick active it must not affect results. */
void test_brick_size_does_not_change_the_solution()
{
  const DenseFields initial = blob({32, 32, 32}, 16.0f, 16.0f, 9.0f, 4.0f);
  Settings settings = plume_settings();
  settings.pressure_accuracy = 1.0e-5f;
  const DenseFields reference = run(initial, 8, everywhere, 8, settings);
  require(sum(reference.w) != 0.0, "reference run produced no motion");
  for (const int brick_size : {4, 16, 5}) {
    const DenseFields other = run(initial, brick_size, everywhere, 8, settings);
    const double density_error = relative_l2(other.density, reference.density);
    const double velocity_error = relative_l2(other.w, reference.w);
    std::cout << "  brick " << brick_size << ": density " << density_error << ", velocity "
              << velocity_error << '\n';
    require(density_error < 1.0e-4 && velocity_error < 1.0e-3,
            "brick size " + std::to_string(brick_size) + " changed the solution");
  }
}

void test_plume_rises_and_rest_stays_at_rest()
{
  const DenseFields initial = blob({32, 32, 48}, 16.0f, 16.0f, 10.0f, 4.0f);
  const DenseFields after = run(initial, 8, 1, 20, plume_settings());
  const double rise = center_of_mass_z(after) - center_of_mass_z(initial);
  require(rise > 1.0, "sparse plume rose only " + std::to_string(rise) + " voxels");
  for (const float value : after.density) {
    require(std::isfinite(value), "non-finite density");
  }

  Settings still = plume_settings();
  still.heat_buoyancy = 0.0f;
  still.density_buoyancy = 0.0f;
  const DenseFields rest = run(initial, 8, 1, 5, still);
  require(sum(rest.w) == 0.0 && sum(rest.u) == 0.0, "velocity appeared without forces");
  require(rest.density == initial.density, "density moved in a fluid at rest");

  DenseFields empty = initial;
  std::fill(empty.density.begin(), empty.density.end(), 0.0f);
  std::fill(empty.temperature.begin(), empty.temperature.end(), 0.0f);
  State nothing = from_dense(empty, 8, {}, 1);
  require(nothing.topology.brick_count() == 0, "an empty state must store no bricks");
  require(step(nothing, plume_settings()).active_bricks == 0, "empty step must be a no-op");
}

/**
 * The point of the sparse solver: follow the smoke with a fraction of the storage. Outside the
 * active bricks it assumes open air instead of solving the whole closed box, so it is not
 * expected to equal the dense solve; this checks that it stays close while the smoke is far from
 * the walls, and that nothing is lost at the moving boundary of the active region.
 */
void test_sparse_follows_the_plume()
{
  const Extent extent = {48, 48, 96};
  const DenseFields initial = blob(extent, 24.0f, 24.0f, 14.0f, 5.0f);
  const Settings settings = plume_settings();
  const int steps = 30;

  std::size_t sparse_bricks = 0, full_bricks = 0;
  const DenseFields sparse = run(initial, 8, 1, steps, settings, &sparse_bricks);
  const DenseFields full = run(initial, 8, everywhere, steps, settings, &full_bricks);

  const double fraction = static_cast<double>(sparse_bricks) / static_cast<double>(full_bricks);
  const double mass_ratio = sum(sparse.density) / sum(full.density);
  const double density_error = relative_l2(sparse.density, full.density);
  const double rise_sparse = center_of_mass_z(sparse) - center_of_mass_z(initial);
  const double rise_full = center_of_mass_z(full) - center_of_mass_z(initial);
  std::cout << "  peak active bricks " << sparse_bricks << " of " << full_bricks << " (" << fraction
            << "), mass ratio " << mass_ratio << ", density rel L2 " << density_error
            << ", rise sparse " << rise_sparse << " full " << rise_full << '\n';
  require(fraction < 0.5, "sparse run activated too much of the domain");
  /* Neither scheme conserves mass exactly, so the two runs drift apart by a few percent. */
  require(std::abs(mass_ratio - 1.0) < 0.05, "sparse run lost or gained smoke");
  require(std::abs(rise_sparse - rise_full) < 0.15 * rise_full,
          "sparse plume height differs from the dense solve");
}

void test_jacobi_matches_conjugate_gradient()
{
  const DenseFields initial = blob({16, 16, 16}, 8.0f, 8.0f, 6.0f, 2.5f);
  Settings exact = plume_settings();
  exact.pressure_accuracy = 1.0e-6f;
  Settings jacobi = exact;
  jacobi.pressure_solver = PressureSolver::Jacobi;
  jacobi.pressure_iterations = 3000;
  const DenseFields a = run(initial, 8, everywhere, 3, exact);
  const DenseFields b = run(initial, 8, everywhere, 3, jacobi);
  const double error = relative_l2(b.w, a.w);
  std::cout << "  Jacobi vs CG velocity rel L2 " << error << '\n';
  require(error < 1.0e-3, "Jacobi and CG disagree");
}

/** Multigrid must cut the residual by a steady factor per cycle and agree with CG. */
void test_multigrid_converges()
{
  const DenseFields initial = blob({48, 48, 64}, 24.0f, 24.0f, 14.0f, 6.0f);
  for (const int dilation : {1, everywhere}) {
    State state = from_dense(initial, 8, {}, dilation);
    const Settings settings = plume_settings();
    add_buoyancy(state, state.temperature, 0.5f, settings.dt, settings.gravity);
    enforce_walls(state);
    std::vector<float> rhs;
    divergence(state, rhs);
    for (float &value : rhs) {
      value = -value;
    }
    const Multigrid multigrid = build_multigrid(state.topology);
    std::vector<float> pressure(state.topology.voxel_count(), 0.0f);
    const float initial_residual = poisson_residual(state.topology, rhs, pressure);
    float previous = initial_residual;
    float worst_factor = 0.0f;
    for (int cycle = 0; cycle < 6; ++cycle) {
      multigrid_solve(state.topology, multigrid, rhs, settings, 1, pressure);
      const float current = poisson_residual(state.topology, rhs, pressure);
      worst_factor = std::max(worst_factor, current / previous);
      previous = current;
    }
    std::cout << "  dilation " << dilation << ": " << multigrid.levels.size()
              << " coarse levels, worst factor per cycle " << worst_factor << ", total "
              << previous / initial_residual << '\n';
    require(worst_factor < 0.5f, "a multigrid cycle reduced the residual too little");
    require(previous < 2.0e-3f * initial_residual, "six multigrid cycles did not converge");
  }

  Settings exact = plume_settings();
  exact.pressure_accuracy = 1.0e-5f;
  exact.pressure_iterations = 2000;
  Settings cycles = plume_settings();
  cycles.pressure_solver = PressureSolver::Multigrid;
  cycles.pressure_iterations = 10;
  const DenseFields small = blob({32, 32, 48}, 16.0f, 16.0f, 10.0f, 4.0f);
  const DenseFields a = run(small, 8, 1, 6, exact);
  const DenseFields b = run(small, 8, 1, 6, cycles);
  const double error = relative_l2(b.w, a.w);
  std::cout << "  multigrid vs CG velocity rel L2 after 6 steps " << error << '\n';
  require(error < 1.0e-3, "multigrid and CG disagree");

  bool rejected = false;
  try {
    build_multigrid(build_topology({20, 20, 20}, 5, std::vector<std::uint8_t>(64, 1), 0));
  }
  catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "a brick size that is not a power of two must be rejected");
}

/** MacCormack must keep smoke sharper than semi-Lagrangian advection without overshooting. */
void test_maccormack_is_sharper_and_bounded()
{
  const DenseFields initial = blob({40, 40, 64}, 20.0f, 20.0f, 12.0f, 5.0f);
  Settings first_order = plume_settings();
  first_order.advection = Advection::SemiLagrangian;
  Settings second_order = plume_settings();
  second_order.advection = Advection::MacCormack;

  std::size_t diffuse_bricks = 0, sharp_bricks = 0;
  const DenseFields diffuse = run(initial, 8, 1, 25, first_order, &diffuse_bricks);
  const DenseFields sharp = run(initial, 8, 1, 25, second_order, &sharp_bricks);
  const auto peak = [](const DenseFields &fields) {
    return *std::max_element(fields.density.begin(), fields.density.end());
  };
  const auto lowest = [](const DenseFields &fields) {
    return *std::min_element(fields.density.begin(), fields.density.end());
  };
  std::cout << "  peak density: MacCormack " << peak(sharp) << ", semi-Lagrangian "
            << peak(diffuse) << "; peak bricks " << sharp_bricks << " and " << diffuse_bricks
            << '\n';
  require(peak(sharp) > peak(diffuse) + 0.05f, "MacCormack did not preserve the peak better");
  require(sharp_bricks <= diffuse_bricks, "MacCormack spread smoke into more bricks");
  require(peak(sharp) <= 1.0f + 1.0e-5f && lowest(sharp) >= -1.0e-5f,
          "MacCormack left the range of the initial values");
}

void test_advance_takes_cfl_substeps()
{
  const DenseFields initial = blob({32, 32, 48}, 16.0f, 16.0f, 10.0f, 4.0f);
  Settings settings = plume_settings();
  settings.cfl = 1.0f;
  State state = from_dense(initial, 8, settings.thresholds, settings.dilation);
  const float frame = 1.0f / 30.0f;
  int most_substeps = 0;
  for (int i = 0; i < 12; ++i) {
    const float speed_before = max_speed(state);
    const FrameInfo info = advance(state, settings, frame);
    most_substeps = std::max(most_substeps, info.substeps);
    /* The first substep is limited by the speed at the start of the frame. */
    const int expected = speed_before > 0.0f ?
                             static_cast<int>(std::ceil(frame * speed_before / settings.cfl)) :
                             1;
    require(info.substeps >= std::min(expected, 2) && info.substeps >= 1,
            "too few substeps for the CFL limit");
    require(info.max_speed >= speed_before, "frame did not report the starting speed");
  }
  std::cout << "  most substeps in a frame: " << most_substeps << ", final speed "
            << max_speed(state) << '\n';
  require(most_substeps >= 2, "the plume never needed substeps at CFL 1");
  require(cfl_time_step(1.0f, 2.0f, 0.0f) == 1.0f && cfl_time_step(1.0f, 2.0f, 8.0f) == 0.25f &&
              cfl_time_step(0.1f, 2.0f, 8.0f) == 0.1f,
          "cfl_time_step");

  settings.max_substeps = 1;
  bool failed = false;
  try {
    advance(state, settings, frame);
  }
  catch (const std::runtime_error &) {
    failed = true;
  }
  require(failed, "exhausting the substep limit must fail");
}

/**
 * The default settings must keep most of a fast plume's smoke. With plain steps, a single-stage
 * backward trace loses mass when a step moves smoke several voxels through an accelerating
 * flow; the midpoint trace is what makes large steps usable. See "Fewer substeps" in
 * docs/gpu-sparse-solver.md.
 */
void test_large_steps_keep_mass()
{
  const DenseFields initial = blob({32, 32, 48}, 16.0f, 16.0f, 8.0f, 4.0f);
  const double initial_mass = sum(initial.density);
  const auto kept = [&](const char *label, Settings settings) {
    settings.pressure_solver = PressureSolver::Multigrid;
    settings.pressure_iterations = 6;
    State state = from_dense(initial, 8, settings.thresholds, settings.dilation);
    int substeps = 0;
    for (int i = 0; i < 6; ++i) {
      /* Long frames, so that the limit and not the frame length sets the step. */
      substeps += advance(state, settings, 1.0f / 8.0f).substeps;
    }
    const double fraction = sum(to_dense(state).density) / initial_mass;
    std::cout << "  " << label << ", CFL " << settings.cfl << ": " << substeps
              << " steps, mass kept " << fraction << std::endl;
    return fraction;
  };
  const double at_default = kept("defaults", plume_settings());
  require(std::abs(at_default - 1.0) < 1.0e-3, "the default settings changed the amount of smoke");

  /* Without the limiter, the trace alone decides how much smoke survives. */
  Settings plain = plume_settings();
  plain.reflection = false;
  plain.conserve_mass = false;
  plain.cfl = 4.0f;
  const double midpoint = kept("plain steps, midpoint trace", plain);
  plain.midpoint_trace = false;
  const double single_stage = kept("plain steps, single-stage trace", plain);
  require(midpoint > 0.65, "plain steps with the midpoint trace lost too much smoke");
  require(single_stage < midpoint - 0.1, "the midpoint trace was expected to keep more smoke");
}

/**
 * A scene that is mirror-symmetric must stay so. The open boundary of the active region once lay
 * a voxel further out on its lower sides, which made a centred plume lean within a few frames.
 */
void test_symmetric_scene_stays_symmetric()
{
  const Extent extent{32, 32, 48};
  const DenseFields initial = blob(extent, 16.0f, 16.0f, 12.0f, 4.0f);
  Settings settings = plume_settings();
  settings.pressure_solver = PressureSolver::Multigrid;
  settings.pressure_iterations = 6;
  State state = from_dense(initial, 8, settings.thresholds, settings.dilation);
  for (int i = 0; i < 5; ++i) {
    advance(state, settings, 1.0f / 15.0f);
  }
  const DenseFields result = to_dense(state);
  float asymmetry = 0.0f;
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const float value = result.density[dense_index(extent, x, y, z)];
        asymmetry = std::max(
            asymmetry, std::abs(value - result.density[dense_index(extent, extent.x - 1 - x, y, z)]));
        asymmetry = std::max(
            asymmetry, std::abs(value - result.density[dense_index(extent, x, extent.y - 1 - y, z)]));
      }
    }
  }
  std::cout << "  largest density difference between mirrored voxels: " << asymmetry << std::endl;
  require(asymmetry < 2.0e-4f, "a symmetric plume lost its symmetry");
}

/** A plate above the plume must stay free of smoke and of flow through its faces. */
void test_obstacle_blocks_the_plume()
{
  const Extent extent{32, 32, 48};
  DenseFields initial = blob(extent, 16.0f, 16.0f, 12.0f, 4.0f);
  const auto in_plate = [](const int x, const int y, const int z) {
    return z >= 22 && z < 25 && x >= 9 && x < 23 && y >= 9 && y < 23;
  };
  Settings settings = plume_settings();
  settings.pressure_solver = PressureSolver::Multigrid;
  settings.pressure_iterations = 6;
  State free_state = from_dense(initial, 8, settings.thresholds, settings.dilation);
  initial.obstacle.assign(initial.density.size(), 0);
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        initial.obstacle[dense_index(extent, x, y, z)] = in_plate(x, y, z);
      }
    }
  }
  State state = from_dense(initial, 8, settings.thresholds, settings.dilation);
  const double initial_mass = sum(initial.density);
  float worst_divergence = 0.0f;
  for (int i = 0; i < 12; ++i) {
    const FrameInfo frame = advance(state, settings, 1.0f / 15.0f);
    worst_divergence = std::max(worst_divergence, frame.last_step.max_divergence);
    advance(free_state, settings, 1.0f / 15.0f);
  }
  require(state.topology.type_at(16, 16, 23) == CellSolid, "the plate must be stored as solid");
  const DenseFields result = to_dense(state);
  const DenseFields free_result = to_dense(free_state);
  float inside = 0.0f, through = 0.0f;
  double above = 0.0, free_above = 0.0, total = 0.0;
  for (int z = 0; z < extent.z; ++z) {
    for (int y = 0; y < extent.y; ++y) {
      for (int x = 0; x < extent.x; ++x) {
        const std::size_t i = dense_index(extent, x, y, z);
        total += result.density[i];
        if (z >= 25 && x >= 11 && x < 21 && y >= 11 && y < 21) {
          above += result.density[i];
          free_above += free_result.density[i];
        }
        const bool solid = in_plate(x, y, z);
        if (solid) {
          inside = std::max(inside, std::abs(result.density[i]));
        }
        /* Every face of a solid voxel is at rest. */
        if (solid || in_plate(x - 1, y, z)) {
          through = std::max(through, std::abs(result.u[i]));
        }
        if (solid || in_plate(x, y - 1, z)) {
          through = std::max(through, std::abs(result.v[i]));
        }
        if (solid || in_plate(x, y, z - 1)) {
          through = std::max(through, std::abs(result.w[i]));
        }
      }
    }
  }
  std::cout << "  smoke inside the plate " << inside << ", flow through it " << through
            << ", mass " << total << ", above its middle " << above << " (without plate "
            << free_above << "), worst divergence " << worst_divergence << std::endl;
  require(inside == 0.0f, "smoke entered the obstacle");
  require(through == 0.0f, "flow crossed an obstacle face");
  /* Without conserve_mass, MacCormack adds smoke along the plate; the limiter holds the total. */
  require(std::abs(total / initial_mass - 1.0) < 1.0e-3, "the obstacle changed the amount of smoke");
  require(free_above > 10.0 && above < 0.25 * free_above, "the plate did not shelter the space above it");
  require(worst_divergence < 0.05f, "the projection failed next to the obstacle");
}

/** Sources raise or add smoke and heat, keep their bricks active, and set their velocity. */
void test_sources_emit()
{
  const Extent extent{32, 32, 48};
  const DenseFields shape = blob(extent, 16.0f, 16.0f, 12.0f, 3.0f);
  DenseFields emit = shape;
  for (float &value : emit.density) {
    value *= 0.8f;
  }
  for (float &value : emit.temperature) {
    value *= 0.5f;
  }
  const double voxels = sum(shape.density);

  DenseFields empty = shape;
  std::fill(empty.density.begin(), empty.density.end(), 0.0f);
  std::fill(empty.temperature.begin(), empty.temperature.end(), 0.0f);
  Settings still;
  still.heat_buoyancy = 0.0f;
  still.density_buoyancy = 0.0f;
  still.pressure_solver = PressureSolver::Multigrid;
  still.pressure_iterations = 6;

  /* Nothing moves, so raised values sit exactly at the source's level. */
  State state = from_dense(empty, 8, still.thresholds, still.dilation);
  require(state.topology.brick_count() == 0, "an empty state has no bricks");
  const Sources raise = sources_from_dense(emit, 8, SourceMode::Raise);
  for (int frame = 0; frame < 3; ++frame) {
    advance(state, still, 1.0f / 30.0f, &raise);
  }
  DenseFields result = to_dense(state);
  double worst = 0.0;
  for (std::size_t i = 0; i < result.density.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::abs(result.density[i] - emit.density[i])));
    worst = std::max(worst,
                     static_cast<double>(std::abs(result.temperature[i] - emit.temperature[i])));
  }
  std::cout << "  raise: " << state.topology.brick_count() << " bricks, largest difference "
            << worst << std::endl;
  require(state.topology.brick_count() > 0, "the source did not activate its bricks");
  require(worst == 0.0, "raising did not reach the source's level exactly");

  /* Adding accumulates the source's rate. */
  state = from_dense(empty, 8, still.thresholds, still.dilation);
  const Sources add = sources_from_dense(emit, 8, SourceMode::Add);
  for (int frame = 0; frame < 6; ++frame) {
    advance(state, still, 1.0f / 30.0f, &add);
  }
  const double added = sum(to_dense(state).density);
  const double expected = 0.8 * voxels * 6.0 / 30.0;
  std::cout << "  add: density total " << added << ", expected " << expected << std::endl;
  require(std::abs(added / expected - 1.0) < 1.0e-4, "adding did not accumulate the rate");

  /* A source velocity carries the smoke along. */
  emit.u.assign(emit.density.size(), 0.0f);
  emit.v.assign(emit.density.size(), 0.0f);
  emit.w.assign(emit.density.size(), 0.0f);
  for (std::size_t i = 0; i < emit.w.size(); ++i) {
    emit.w[i] = shape.density[i] > 0.0f ? 40.0f : 0.0f;
  }
  state = from_dense(empty, 8, still.thresholds, still.dilation);
  const Sources jet = sources_from_dense(emit, 8, SourceMode::Raise);
  for (int frame = 0; frame < 8; ++frame) {
    advance(state, still, 1.0f / 30.0f, &jet);
  }
  result = to_dense(state);
  const double rise = center_of_mass_z(result) - center_of_mass_z(shape);
  std::cout << "  jet: smoke centre " << rise << " voxels above the source" << std::endl;
  require(rise > 2.0, "the source velocity did not carry the smoke");
}

/** A plate moving up pushes the smoke above it and carries no flow through its faces. */
void test_moving_obstacle_pushes_smoke()
{
  const Extent extent{32, 32, 48};
  DenseFields initial = blob(extent, 16.0f, 16.0f, 24.0f, 8.0f);
  const auto plate_at = [&](const int bottom) {
    std::vector<std::uint8_t> solid(initial.density.size(), 0);
    for (int z = bottom; z < bottom + 3; ++z) {
      for (int y = 10; y < 22; ++y) {
        for (int x = 10; x < 22; ++x) {
          solid[dense_index(extent, x, y, z)] = 1;
        }
      }
    }
    return solid;
  };
  const float speed = 30.0f; /* Voxels per second: one voxel per frame. */
  Settings settings;
  settings.heat_buoyancy = 0.0f;
  settings.density_buoyancy = 0.0f;
  settings.pressure_solver = PressureSolver::Multigrid;
  settings.pressure_iterations = 6;
  State state = from_dense(initial, 8, settings.thresholds, settings.dilation);
  const double start_z = center_of_mass_z(initial);
  int bottom = 14;
  float worst_face = 0.0f, worst_divergence = 0.0f;
  for (int frame = 0; frame < 6; ++frame) {
    DenseFields obstacles;
    obstacles.extent = extent;
    obstacles.obstacle = plate_at(bottom);
    obstacles.obstacle_u.assign(initial.density.size(), 0.0f);
    obstacles.obstacle_v.assign(initial.density.size(), 0.0f);
    obstacles.obstacle_w.assign(initial.density.size(), 0.0f);
    for (std::size_t i = 0; i < obstacles.obstacle.size(); ++i) {
      obstacles.obstacle_w[i] = obstacles.obstacle[i] ? speed : 0.0f;
    }
    set_obstacles_from_dense(state, obstacles);
    const FrameInfo info = advance(state, settings, 1.0f / 30.0f);
    worst_divergence = std::max(worst_divergence, info.last_step.max_divergence);
    const DenseFields now = to_dense(state);
    /* Faces on the plate's top and bottom move with it; no smoke stays inside. */
    for (int y = 10; y < 22; ++y) {
      for (int x = 10; x < 22; ++x) {
        worst_face = std::max(worst_face,
                              std::abs(now.w[dense_index(extent, x, y, bottom)] - speed));
        worst_face = std::max(worst_face,
                              std::abs(now.w[dense_index(extent, x, y, bottom + 3)] - speed));
        for (int z = bottom; z < bottom + 3; ++z) {
          require(now.density[dense_index(extent, x, y, z)] == 0.0f, "smoke inside the plate");
        }
      }
    }
    ++bottom;
  }
  const double rise = center_of_mass_z(to_dense(state)) - start_z;
  std::cout << "  plate face error " << worst_face << ", worst divergence " << worst_divergence
            << ", smoke centre rose " << rise << " voxels" << std::endl;
  require(worst_face < 1.0e-4f, "the plate's faces do not move with it");
  require(worst_divergence < 0.05f, "the projection failed around the moving plate");
  require(rise > 0.5, "the moving plate did not push the smoke");
}

/** The turbulence force stirs without compressing, follows its seed and time, and its mask. */
void test_turbulence_stirs()
{
  const Extent extent{32, 32, 48};
  const DenseFields initial = blob(extent, 16.0f, 16.0f, 24.0f, 6.0f);
  Settings settings;
  settings.dilation = 1000; /* Every brick, so the mask has empty air to leave alone. */
  const State start = from_dense(initial, 8, settings.thresholds, settings.dilation);
  Turbulence turbulence;
  turbulence.strength = 300.0f;
  turbulence.scale = 8.0f;
  turbulence.seed = 3;
  turbulence.mask = TurbulenceMask::None;

  const auto stirred = [&](const Turbulence &t, const float time) {
    State state = start;
    add_turbulence(state, t, time, 1.0f / 30.0f);
    return state;
  };
  const auto difference = [](const State &a, const State &b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.u.size(); ++i) {
      result = std::max({result, static_cast<double>(std::abs(a.u[i] - b.u[i])),
                         static_cast<double>(std::abs(a.v[i] - b.v[i])),
                         static_cast<double>(std::abs(a.w[i] - b.w[i]))});
    }
    return result;
  };

  const State once = stirred(turbulence, 0.0f);
  const double magnitude = difference(once, start);
  /* Away from the walls, where faces touching a wall get no push. */
  std::vector<float> div;
  divergence(once, div);
  DenseFields div_dense = to_dense(once);
  {
    State probe = once;
    probe.density = div;
    div_dense = to_dense(probe);
  }
  float worst = 0.0f;
  for (int z = 2; z < extent.z - 2; ++z) {
    for (int y = 2; y < extent.y - 2; ++y) {
      for (int x = 2; x < extent.x - 2; ++x) {
        worst = std::max(worst, std::abs(div_dense.density[dense_index(extent, x, y, z)]));
      }
    }
  }
  std::cout << "  largest push " << magnitude << " voxels/s, largest divergence " << worst
            << " 1/s" << std::endl;
  require(magnitude > 1.0, "turbulence did not move the air");
  /* A curl is divergence-free; differencing on the faces leaves a small part. */
  require(worst < 0.25 * magnitude, "turbulence compresses the air");

  require(difference(stirred(turbulence, 0.0f), once) == 0.0, "turbulence is not repeatable");
  Turbulence other = turbulence;
  other.seed = 4;
  require(difference(stirred(other, 0.0f), once) > 0.5 * magnitude, "the seed changes nothing");
  require(difference(stirred(turbulence, 0.5f), once) > 0.1 * magnitude,
          "the pattern does not change over time");
  Turbulence frozen = turbulence;
  frozen.speed = 0.0f;
  require(difference(stirred(frozen, 0.5f), stirred(frozen, 0.0f)) == 0.0,
          "a frozen pattern changed over time");

  /* Masked by density: faces of empty air keep their velocity. */
  Turbulence masked = turbulence;
  masked.mask = TurbulenceMask::Density;
  masked.mask_level = 0.1f;
  const State in_smoke = stirred(masked, 0.0f);
  const DenseFields dense = to_dense(in_smoke);
  float empty_push = 0.0f;
  for (int z = 2; z < 6; ++z) {
    for (int y = 2; y < 30; ++y) {
      for (int x = 2; x < 30; ++x) {
        const std::size_t i = dense_index(extent, x, y, z);
        empty_push = std::max({empty_push, std::abs(dense.u[i]), std::abs(dense.v[i]),
                               std::abs(dense.w[i])});
      }
    }
  }
  require(empty_push == 0.0f, "the density mask let turbulence move empty air");
  require(difference(in_smoke, start) > 0.5 * magnitude, "the masked force vanished in smoke");

  /* Through whole frames the total amount of smoke is still held. */
  State state = from_dense(initial, 8, settings.thresholds, 1);
  Settings stir = plume_settings();
  stir.pressure_solver = PressureSolver::Multigrid;
  stir.pressure_iterations = 6;
  stir.turbulence = turbulence;
  stir.turbulence.mask = TurbulenceMask::Density;
  for (int frame = 0; frame < 6; ++frame) {
    stir.time = static_cast<float>(frame) / 30.0f;
    advance(state, stir, 1.0f / 30.0f);
  }
  const double kept = sum(to_dense(state).density) / sum(initial.density);
  std::cout << "  mass kept after 6 stirred frames " << kept << std::endl;
  require(std::abs(kept - 1.0) < 1.0e-3, "turbulence changed the amount of smoke");
}

}  // namespace

/** Fuel burns into flame, heat and smoke at the burn rate, and the flame fades once it is gone. */
void test_fire_burns()
{
  const Extent extent{32, 32, 48};
  const DenseFields shape = blob(extent, 16.0f, 16.0f, 12.0f, 4.0f);
  DenseFields initial = shape;
  std::fill(initial.density.begin(), initial.density.end(), 0.0f);
  std::fill(initial.temperature.begin(), initial.temperature.end(), 0.0f);
  initial.fuel = shape.density;
  initial.flame.assign(shape.density.size(), 0.0f);
  const double fuel = sum(initial.fuel);

  Settings settings;
  settings.fire.enabled = true;
  settings.fire.burn_rate = 0.6f;
  settings.fire.heat = 3.0f;
  settings.fire.smoke = 0.5f;
  settings.fire.flame_cooling = 5.0f;
  settings.gravity[2] = -196.0f; /* 9.81 m/s^2 at 5 cm voxels, as in Blender. */
  settings.pressure_solver = PressureSolver::Multigrid;
  settings.pressure_iterations = 6;

  /* At rest, burning is exact: one step burns burn_rate * dt of each voxel's fuel. */
  State still = from_dense(initial, 8, settings.thresholds, settings.dilation);
  combust(still, settings.fire, 0.5f);
  DenseFields burnt = to_dense(still);
  const double gone = fuel - sum(burnt.fuel);
  std::cout << "  half a second: fuel " << fuel << " -> " << sum(burnt.fuel) << ", smoke "
            << sum(burnt.density) << ", heat " << sum(burnt.temperature) << std::endl;
  require(std::abs(sum(burnt.density) - 0.5 * gone) < 1.0e-3 * fuel, "smoke is not 0.5 per fuel");
  require(std::abs(sum(burnt.temperature) - 3.0 * gone) < 1.0e-3 * fuel,
          "heat is not 3 per fuel");
  float most_flame = 0.0f;
  for (const float value : burnt.flame) {
    most_flame = std::max(most_flame, value);
  }
  require(most_flame == 1.0f, "a voxel full of fuel does not flame at 1");
  for (std::size_t i = 0; i < burnt.flame.size(); ++i) {
    require(burnt.flame[i] == std::min(initial.fuel[i], 1.0f), "flame is not the fuel burning");
  }
  for (std::size_t i = 0; i < burnt.fuel.size(); ++i) {
    const float expected = std::max(initial.fuel[i] - 0.3f, 0.0f);
    require(std::abs(burnt.fuel[i] - expected) < 1.0e-6f, "fuel did not burn at the rate");
  }
  /* Without fuel the flame fades at flame_cooling: from 1 to 0.5 in 0.1 s, gone by 0.2 s. */
  State faded = still;
  std::fill(faded.fuel.begin(), faded.fuel.end(), 0.0f);
  combust(faded, settings.fire, 0.1f);
  float left = 0.0f;
  for (const float value : faded.flame) {
    left = std::max(left, value);
  }
  require(std::abs(left - 0.5f) < 1.0e-6f, "the flame did not fade at flame_cooling");
  combust(faded, settings.fire, 0.1f);
  left = 0.0f;
  for (const float value : faded.flame) {
    left = std::max(left, value);
  }
  require(left == 0.0f, "the flame did not fade without fuel");

  /* Burning heats the gas, so a fire rises; its smoke conserves what the fuel became. */
  State fire = from_dense(initial, 8, settings.thresholds, settings.dilation);
  for (int frame = 0; frame < 12; ++frame) {
    advance(fire, settings, 1.0f / 30.0f);
  }
  const DenseFields result = to_dense(fire);
  const double rise = center_of_mass_z(result) - center_of_mass_z(shape);
  const double total = sum(result.fuel) + sum(result.density) / settings.fire.smoke;
  std::cout << "  12 frames: smoke centre " << rise << " voxels up, fuel + smoke/0.5 = " << total
            << " of " << fuel << std::endl;
  require(rise > 1.0, "the fire did not rise");
  require(std::abs(total / fuel - 1.0) < 1.0e-3, "burning lost or made material");

  /* Disabled, the fuel just sits and nothing burns. */
  Settings off = settings;
  off.fire.enabled = false;
  State cold = from_dense(initial, 8, off.thresholds, off.dilation);
  advance(cold, off, 1.0f / 30.0f);
  require(sum(to_dense(cold).density) == 0.0, "fuel burned with the fire off");

  /* A source emits fuel like density. */
  DenseFields emit = initial;
  std::fill(emit.fuel.begin(), emit.fuel.end(), 0.0f);
  emit.fuel = shape.density;
  DenseFields empty = initial;
  std::fill(empty.fuel.begin(), empty.fuel.end(), 0.0f);
  State lit = from_dense(empty, 8, settings.thresholds, settings.dilation);
  const Sources source = sources_from_dense(emit, 8, SourceMode::Raise);
  require(source.field.channels.size() == 4, "sources do not carry fuel and expansion channels");
  for (int frame = 0; frame < 6; ++frame) {
    advance(lit, settings, 1.0f / 30.0f, &source);
  }
  const DenseFields lit_dense = to_dense(lit);
  std::cout << "  source: flame " << sum(lit_dense.flame) << ", smoke " << sum(lit_dense.density)
            << std::endl;
  require(sum(lit_dense.flame) > 0.0 && sum(lit_dense.density) > 0.0,
          "an emitted fuel did not burn");
}

/**
 * Expansion: a source's expansion becomes the flow's divergence there, pushing smoke outward,
 * and burning fuel with a gas release does the same.
 */
void test_expansion_pushes_outward()
{
  const Extent extent{32, 32, 32};
  const DenseFields shape = blob(extent, 16.0f, 16.0f, 16.0f, 3.0f);
  Settings settings;
  settings.heat_buoyancy = 0.0f;
  settings.density_buoyancy = 0.0f;
  settings.reflection = false;
  settings.pressure_solver = PressureSolver::Multigrid;
  settings.pressure_iterations = 10;

  /* A ring of smoke around an expanding core moves outward. */
  DenseFields initial = blob(extent, 16.0f, 16.0f, 16.0f, 8.0f);
  DenseFields emit = shape;
  std::fill(emit.density.begin(), emit.density.end(), 0.0f);
  std::fill(emit.temperature.begin(), emit.temperature.end(), 0.0f);
  emit.expansion = shape.density;
  for (float &value : emit.expansion) {
    value *= 20.0f;
  }
  const Sources source = sources_from_dense(emit, 8, SourceMode::Raise);
  const auto spread = [&](const DenseFields &dense) {
    double weighted = 0.0, total = 0.0;
    for (int z = 0; z < extent.z; ++z) {
      for (int y = 0; y < extent.y; ++y) {
        for (int x = 0; x < extent.x; ++x) {
          const double d = dense.density[dense_index(extent, x, y, z)];
          const double r2 = (x + 0.5 - 16.0) * (x + 0.5 - 16.0) +
                            (y + 0.5 - 16.0) * (y + 0.5 - 16.0) +
                            (z + 0.5 - 16.0) * (z + 0.5 - 16.0);
          weighted += d * std::sqrt(r2);
          total += d;
        }
      }
    }
    return weighted / total;
  };
  State state = from_dense(initial, 8, settings.thresholds, settings.dilation);
  const StepInfo info = step(state, settings, &source);
  std::vector<float> div;
  divergence(state, div);
  const DenseFields after_step = [&] {
    State probe = state;
    probe.density = div;
    return to_dense(probe);
  }();
  const float centre = after_step.density[dense_index(extent, 16, 16, 16)];
  std::cout << "  divergence at the centre " << centre << " 1/s (asked for 20), left over "
            << info.max_divergence << std::endl;
  require(std::abs(centre - 20.0f) < 0.5f, "the flow does not take the source's expansion");
  for (int i = 0; i < 5; ++i) {
    step(state, settings, &source);
  }
  const double before = spread(initial), after = spread(to_dense(state));
  std::cout << "  mean distance of smoke from the centre: " << before << " -> " << after
            << " voxels" << std::endl;
  /* The core grows by about 450 voxels (20/s over 113 voxels for 0.2 s): a 6% larger radius
   * for the smoke around it. */
  require(after > before * 1.03, "the expansion did not push the smoke outward");

  /* Burning fuel with a gas release expands where it burns; without, it does not. */
  DenseFields fuel = shape;
  std::fill(fuel.density.begin(), fuel.density.end(), 0.0f);
  std::fill(fuel.temperature.begin(), fuel.temperature.end(), 0.0f);
  fuel.fuel = shape.density;
  fuel.flame.assign(shape.density.size(), 0.0f);
  const auto burn_divergence = [&](const float gas_release) {
    Settings fire = settings;
    fire.fire.enabled = true;
    fire.fire.burn_rate = 2.0f;
    fire.fire.gas_release = gas_release;
    State burning = from_dense(fuel, 8, fire.thresholds, fire.dilation);
    step(burning, fire);
    std::vector<float> d;
    divergence(burning, d);
    State probe = burning;
    probe.density = d;
    return to_dense(probe).density[dense_index(extent, 16, 16, 16)];
  };
  const float released = burn_divergence(5.0f), quiet = burn_divergence(0.0f);
  std::cout << "  burning core divergence: " << released << " 1/s with gas release 5 (burning "
            << "2/s), " << quiet << " without" << std::endl;
  require(std::abs(released - 10.0f) < 0.5f, "gas release is not 5 times the fuel burnt");
  require(std::abs(quiet) < 1.0e-3f, "fuel expanded without a gas release");
}

/** Turbulence layers apply in order; a speed mask makes a layer react to the ones before it. */
void test_turbulence_layers_in_order()
{
  const Extent extent{32, 32, 48};
  const DenseFields initial = blob(extent, 16.0f, 16.0f, 24.0f, 6.0f);
  Settings settings;
  settings.dilation = 1000;
  const State start = from_dense(initial, 8, settings.thresholds, settings.dilation);

  Turbulence large;
  large.strength = 400.0f;
  large.scale = 12.0f;
  large.seed = 1;
  large.mask = TurbulenceMask::None;
  Turbulence fine;
  fine.strength = 300.0f;
  fine.scale = 3.0f;
  fine.seed = 2;
  fine.mask = TurbulenceMask::Speed;
  fine.mask_level = 4.0f;

  const auto run = [&](const std::vector<Turbulence> &layers) {
    State state = start;
    for (const Turbulence &layer : layers) {
      add_turbulence(state, layer, 0.0f, 1.0f / 30.0f);
    }
    return state;
  };
  const auto difference = [](const State &a, const State &b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.u.size(); ++i) {
      result = std::max({result, static_cast<double>(std::abs(a.u[i] - b.u[i])),
                         static_cast<double>(std::abs(a.v[i] - b.v[i])),
                         static_cast<double>(std::abs(a.w[i] - b.w[i]))});
    }
    return result;
  };
  /* The air starts at rest, so the fine layer alone does nothing; after the large layer it
   * stirs where the large one made the air move. */
  require(difference(run({fine}), start) == 0.0, "a speed mask stirred air at rest");
  const double order = difference(run({large, fine}), run({fine, large}));
  std::cout << "  large then fine differs from fine then large by " << order << " voxels/s"
            << std::endl;
  require(order > 1.0, "the order of the layers did not matter");
  Turbulence inverted = fine;
  inverted.invert = true;
  require(difference(run({inverted}), start) > 1.0, "an inverted speed mask did not stir still air");
  /* step() applies Settings::turbulence, then the layers, in order. */
  Settings layered = settings;
  layered.heat_buoyancy = 0.0f;
  layered.turbulence = large;
  layered.turbulence_layers = {fine};
  layered.reflection = false;
  layered.pressure_solver = PressureSolver::Multigrid;
  layered.pressure_iterations = 4;
  Settings swapped = layered;
  swapped.turbulence = fine;
  swapped.turbulence_layers = {large};
  State a = start, b = start;
  step(a, layered);
  step(b, swapped);
  require(difference(a, b) > 0.5, "step() does not apply the layers in order");
}

int main()
{
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"topology and cell types", test_topology_and_cell_types},
      {"dense round trip", test_dense_round_trip},
      {"projection removes divergence", test_projection_removes_divergence},
      {"brick size does not change the solution", test_brick_size_does_not_change_the_solution},
      {"plume rises and rest stays at rest", test_plume_rises_and_rest_stays_at_rest},
      {"sparse follows the plume", test_sparse_follows_the_plume},
      {"Jacobi matches conjugate gradient", test_jacobi_matches_conjugate_gradient},
      {"multigrid converges", test_multigrid_converges},
      {"MacCormack is sharper and bounded", test_maccormack_is_sharper_and_bounded},
      {"advance takes CFL substeps", test_advance_takes_cfl_substeps},
      {"large steps keep mass", test_large_steps_keep_mass},
      {"symmetric scene stays symmetric", test_symmetric_scene_stays_symmetric},
      {"obstacle blocks the plume", test_obstacle_blocks_the_plume},
      {"sources emit", test_sources_emit},
      {"moving obstacle pushes smoke", test_moving_obstacle_pushes_smoke},
      {"turbulence stirs", test_turbulence_stirs},
      {"fire burns", test_fire_burns},
      {"turbulence layers in order", test_turbulence_layers_in_order},
      {"expansion pushes outward", test_expansion_pushes_outward},
  };
  const char *only = std::getenv("SPARSE_GAS_TEST");
  for (const auto &[name, test] : tests) {
    if (only && name != only) {
      continue;
    }
    test();
    std::cout << "PASS: " << name << std::endl;
  }
  return 0;
}
