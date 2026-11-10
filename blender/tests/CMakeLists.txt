# SPDX-License-Identifier: GPL-2.0-or-later
# Standalone runtime tests for the gas engine against Blender's pinned Mantaflow sources.
cmake_minimum_required(VERSION 3.22)
project(manta_gas_engine_tests LANGUAGES CXX)

set(BLENDER_SOURCE_DIR "" CACHE PATH "Blender checkout at the gas-step integration commit")
if(NOT BLENDER_SOURCE_DIR)
  message(FATAL_ERROR "Set BLENDER_SOURCE_DIR to a pinned Blender source checkout")
endif()

set(MANTA_GAS_PINNED_COMMIT "377d577d520472f2702e456b26c17dffc663dc52")
find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${BLENDER_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE BLENDER_COMMIT_RESULT
  OUTPUT_VARIABLE BLENDER_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT BLENDER_COMMIT_RESULT EQUAL 0 OR NOT BLENDER_COMMIT STREQUAL MANTA_GAS_PINNED_COMMIT)
  message(FATAL_ERROR
    "BLENDER_SOURCE_DIR must be at ${MANTA_GAS_PINNED_COMMIT}; found '${BLENDER_COMMIT}'")
endif()

get_filename_component(MANTA_IMPROVE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The engine tests do not use OpenVDB. Build the real Mantaflow library with its official CMake
# source list, but without unrelated optional file-I/O dependencies.
set(WITH_TBB ON)
set(WITH_OPENVDB OFF)
set(WITH_PYTHON_NUMPY OFF)
find_package(TBB REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(ZLIB REQUIRED)

add_library(manta_test_dependency_tbb INTERFACE)
target_link_libraries(manta_test_dependency_tbb INTERFACE TBB::tbb)
add_library(bf::dependencies::optional::tbb ALIAS manta_test_dependency_tbb)

add_library(manta_test_dependency_python INTERFACE)
target_link_libraries(manta_test_dependency_python INTERFACE Python3::Python)
add_library(bf::dependencies::optional::python ALIAS manta_test_dependency_python)

add_library(manta_test_dependency_zlib INTERFACE)
target_link_libraries(manta_test_dependency_zlib INTERFACE ZLIB::ZLIB)
add_library(bf::dependencies::zlib ALIAS manta_test_dependency_zlib)

add_library(manta_test_dependency_openvdb INTERFACE)
add_library(bf::dependencies::optional::openvdb ALIAS manta_test_dependency_openvdb)

function(add_cxx_flag FLAG)
  add_compile_options("${FLAG}")
endfunction()

function(blender_add_lib TARGET SOURCES INC INC_SYS LIB)
  add_library("${TARGET}" STATIC ${SOURCES})
  target_include_directories("${TARGET}" PUBLIC ${INC})
  target_include_directories("${TARGET}" SYSTEM PUBLIC ${INC_SYS})
  target_link_libraries("${TARGET}" PUBLIC ${LIB})
endfunction()

add_subdirectory("${BLENDER_SOURCE_DIR}/extern/mantaflow" extern_mantaflow)

add_library(manta_gas_step_engine STATIC
  "${MANTA_IMPROVE_ROOT}/blender/intern/mantaflow/intern/manta_gas_step_API.cpp"
)
target_compile_definitions(manta_gas_step_engine PRIVATE
  WITH_FLUID=1
  WITH_PYTHON
  TBB=1
  OPENVDB=0
  NO_CNPY=1
)
target_include_directories(manta_gas_step_engine PUBLIC
  "${MANTA_IMPROVE_ROOT}/blender/intern/mantaflow/extern"
  "${MANTA_IMPROVE_ROOT}/blender/intern/mantaflow/intern"
)
target_include_directories(manta_gas_step_engine SYSTEM PRIVATE
  "${BLENDER_SOURCE_DIR}/extern/mantaflow/preprocessed"
  "${BLENDER_SOURCE_DIR}/extern/mantaflow/preprocessed/plugin"
  "${BLENDER_SOURCE_DIR}/extern/mantaflow/helper/pwrapper"
  "${BLENDER_SOURCE_DIR}/extern/mantaflow/helper/util"
)
target_link_libraries(manta_gas_step_engine PUBLIC extern_mantaflow)

if(MSVC)
  target_compile_options(manta_gas_step_engine PRIVATE /W4 /WX)
else()
  target_compile_options(manta_gas_step_engine PRIVATE -Wall -Wextra -Werror)
endif()

add_executable(manta_gas_step_engine_tests
  "${MANTA_IMPROVE_ROOT}/tests/gas_step_engine_tests.cpp"
)
target_link_libraries(manta_gas_step_engine_tests PRIVATE manta_gas_step_engine)
if(MSVC)
  target_compile_options(manta_gas_step_engine_tests PRIVATE /W4 /WX)
else()
  target_compile_options(manta_gas_step_engine_tests PRIVATE -Wall -Wextra -Werror)
endif()

enable_testing()
add_test(NAME manta_gas_step_engine_tests COMMAND manta_gas_step_engine_tests)
