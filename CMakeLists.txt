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

add_library(manta_improve_sparse src/sparse_gas.cpp src/sparse_multigrid.cpp)
target_include_directories(
  manta_improve_sparse
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(manta_improve_sparse PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(manta_improve_sparse PRIVATE /W4)
else()
  target_compile_options(manta_improve_sparse PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(sparse_gas_tests tests/sparse_gas_tests.cpp)
  target_link_libraries(sparse_gas_tests PRIVATE manta_improve_sparse)
  add_test(NAME sparse_gas_tests COMMAND sparse_gas_tests)

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

# GPU kernels of the sparse solver. They need the Vulkan headers and loader and glslangValidator
# to build, and any Vulkan device, including a software one, to run.
option(MANTA_IMPROVE_BUILD_GPU "Build the Vulkan compute kernels when their dependencies exist" ON)
if(MANTA_IMPROVE_BUILD_GPU)
  find_package(Vulkan QUIET)
  find_program(MANTA_IMPROVE_GLSLANG NAMES glslangValidator glslang HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
  if(Vulkan_FOUND AND MANTA_IMPROVE_GLSLANG)
    set(MANTA_IMPROVE_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${MANTA_IMPROVE_SHADER_DIR}")
    set(MANTA_IMPROVE_SHADER_HEADERS)
    foreach(KERNEL advect maccormack buoyancy walls divergence jacobi gradient activity remap classify
            classify_links mask_pressure mg_smooth0 mg_restrict0 mg_smooth mg_residual mg_restrict copy3 reflect mg_prolong0 mg_prolong mg_build0 mg_build mg_dense_cycle)
      set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/${KERNEL}.comp")
      set(SHADER_HEADER "${MANTA_IMPROVE_SHADER_DIR}/${KERNEL}.comp.h")
      add_custom_command(
        OUTPUT "${SHADER_HEADER}"
        COMMAND "${MANTA_IMPROVE_GLSLANG}" -V "-I${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders"
                --vn "${KERNEL}_spv" -o "${SHADER_HEADER}" "${SHADER_SOURCE}"
        DEPENDS "${SHADER_SOURCE}" "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/mg_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/mg_coarse.glsl"
        VERBATIM
      )
      list(APPEND MANTA_IMPROVE_SHADER_HEADERS "${SHADER_HEADER}")
    endforeach()

    add_library(manta_improve_sparse_gpu
      gpu/vulkan_compute.cpp
      gpu/sparse_gas_gpu.cpp
      ${MANTA_IMPROVE_SHADER_HEADERS}
    )
    target_include_directories(manta_improve_sparse_gpu
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/gpu"
      PRIVATE "${MANTA_IMPROVE_SHADER_DIR}"
    )
    target_link_libraries(manta_improve_sparse_gpu PUBLIC manta_improve_sparse Vulkan::Vulkan)
    if(NOT MSVC)
      target_compile_options(manta_improve_sparse_gpu PRIVATE -Wall -Wextra)
    endif()

    if(BUILD_TESTING)
      add_executable(sparse_gas_gpu_tests tests/sparse_gas_gpu_tests.cpp)
      target_link_libraries(sparse_gas_gpu_tests PRIVATE manta_improve_sparse_gpu)
      # 48 voxels and 10 steps make the active set outgrow its first allocation.
      add_test(NAME sparse_gas_gpu_tests COMMAND sparse_gas_gpu_tests 48 10)
      add_test(NAME sparse_gas_gpu_jacobi_tests
               COMMAND sparse_gas_gpu_tests 32 6 --jacobi --first-order)
      # A low CFL number forces several substeps per frame.
      add_test(NAME sparse_gas_gpu_substep_tests COMMAND sparse_gas_gpu_tests 32 8 --cfl 0.5)
      # The deeper W-cycle and uneven coarse sweeps are not the defaults; keep them covered.
      add_test(NAME sparse_gas_gpu_wcycle_tests
               COMMAND sparse_gas_gpu_tests 32 6 --w-depth 2 --coarse-sweeps 1 2)
      # Plain steps are not the default; keep them covered.
      add_test(NAME sparse_gas_gpu_plain_step_tests
               COMMAND sparse_gas_gpu_tests 32 8 --reflection 0 --cfl 0.5)
      set_tests_properties(
        sparse_gas_gpu_tests sparse_gas_gpu_jacobi_tests sparse_gas_gpu_substep_tests
        sparse_gas_gpu_wcycle_tests sparse_gas_gpu_plain_step_tests
        PROPERTIES SKIP_RETURN_CODE 77
      )
    endif()
  else()
    message(STATUS "Vulkan or glslangValidator not found: skipping the GPU kernels")
  endif()
endif()

if(MANTA_IMPROVE_BUILD_BENCHMARKS)
  add_executable(activity_mask_benchmark benchmarks/activity_mask_benchmark.cpp)
  target_link_libraries(activity_mask_benchmark PRIVATE manta_improve_activity)
endif()
