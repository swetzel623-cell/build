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
  require(isolated.type_at(8, 10, 10) == CellFluid, "the lower layer needs no ghost cells");
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
  require(std::abs(mass_ratio - 1.0) < 0.02, "sparse run lost or gained smoke");
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

}  // namespace

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
  };
  for (const auto &[name, test] : tests) {
    test();
    std::cout << "PASS: " << name << std::endl;
  }
  return 0;
}
