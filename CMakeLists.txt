cmake_minimum_required(VERSION 3.20)
project(manta_improve LANGUAGES CXX)

option(MANTA_IMPROVE_BUILD_BENCHMARKS "Build microbenchmarks" ON)

add_library(manta_improve_activity src/activity_mask.cpp)
target_include_directories(
  manta_improve_activity
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(manta_improve_activity PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(manta_improve_activity PRIVATE /W4)
else()
  target_compile_options(manta_improve_activity PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(activity_mask_tests tests/activity_mask_tests.cpp)
  target_link_libraries(activity_mask_tests PRIVATE manta_improve_activity)
  add_test(NAME activity_mask_tests COMMAND activity_mask_tests)

  add_executable(gas_step_driver_tests tests/gas_step_driver_tests.cpp)
  target_include_directories(
    gas_step_driver_tests
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/blender/intern/mantaflow/intern
  )
  target_compile_features(gas_step_driver_tests PRIVATE cxx_std_17)
  add_test(NAME gas_step_driver_tests COMMAND gas_step_driver_tests)
endif()

if(MANTA_IMPROVE_BUILD_BENCHMARKS)
  add_executable(activity_mask_benchmark benchmarks/activity_mask_benchmark.cpp)
  target_link_libraries(activity_mask_benchmark PRIVATE manta_improve_activity)
endif()
