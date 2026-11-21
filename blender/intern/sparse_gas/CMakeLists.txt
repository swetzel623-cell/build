# SPDX-License-Identifier: GPL-2.0-or-later
#
# The sparse brick-grid gas solver of the manta_improve repository (MIT), built as a Blender
# library. apply_integration.py copies its sources next to this file. The CPU reference always
# builds. The Vulkan kernels build when the Vulkan headers and glslangValidator are found; the
# Vulkan loader itself is opened at run time, so Blender links no Vulkan library for them.

set(INC
  include
  gpu
)

set(INC_SYS
)

set(SRC
  src/sparse_gas.cpp
  src/sparse_multigrid.cpp

  include/manta_improve/sparse_gas.hpp
)

set(LIB
)

set(SPARSE_GAS_KERNELS
  advect maccormack buoyancy walls divergence jacobi gradient activity remap classify
  classify_links mask_pressure mg_smooth0 mg_restrict0 mg_smooth mg_residual mg_restrict copy3
  reflect conserve_reduce conserve_apply advect4 maccormack4 conserve_reduce4 conserve_apply4
  advect3 maccormack3 inject turbulence combust speed mg_prolong0 mg_prolong mg_build0 mg_build
  mg_dense_cycle fade
)

if(VULKAN_INCLUDE_DIRS)
  set(SPARSE_GAS_VULKAN_INCLUDE ${VULKAN_INCLUDE_DIRS})
else()
  find_path(SPARSE_GAS_VULKAN_INCLUDE vulkan/vulkan.h
    HINTS "${LIBDIR}/vulkan/include" "$ENV{VULKAN_SDK}/Include" "$ENV{VULKAN_SDK}/include"
  )
endif()
find_program(SPARSE_GAS_GLSLANG NAMES glslangValidator glslang
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
)

set(SPARSE_GAS_GPU OFF)
if(SPARSE_GAS_VULKAN_INCLUDE AND SPARSE_GAS_GLSLANG)
  set(SPARSE_GAS_GPU ON)
  set(SPARSE_GAS_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  file(MAKE_DIRECTORY "${SPARSE_GAS_SHADER_DIR}")
  foreach(KERNEL ${SPARSE_GAS_KERNELS})
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/${KERNEL}.comp")
    # Each kernel twice: 32-bit channels, and 16-bit channels (the Half Precision input).
    foreach(PRECISION full half)
      if(PRECISION STREQUAL "half")
        set(VARIANT "_half")
        set(SHADER_DEFINES "-DCHANNEL_HALF=1")
      else()
        set(VARIANT "")
        set(SHADER_DEFINES "-DCHANNEL_HALF=0")
      endif()
      set(SHADER_HEADER "${SPARSE_GAS_SHADER_DIR}/${KERNEL}${VARIANT}.comp.h")
      add_custom_command(
        OUTPUT "${SHADER_HEADER}"
        COMMAND "${SPARSE_GAS_GLSLANG}" -V "-I${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders"
                ${SHADER_DEFINES} --vn "${KERNEL}${VARIANT}_spv" -o "${SHADER_HEADER}"
                "${SHADER_SOURCE}"
        DEPENDS "${SHADER_SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/mg_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/mg_coarse.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/advect_trace.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/quant.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders/quant_config.glsl"
        VERBATIM
      )
      list(APPEND SRC "${SHADER_HEADER}")
    endforeach()
  endforeach()
  list(APPEND SRC
    gpu/vulkan_compute.cpp
    gpu/sparse_gas_gpu.cpp

    gpu/vulkan_compute.hpp
    gpu/sparse_gas_gpu.hpp
  )
  list(APPEND INC_SYS
    ${SPARSE_GAS_VULKAN_INCLUDE}
    ${SPARSE_GAS_SHADER_DIR}
  )
  list(APPEND LIB
    ${CMAKE_DL_LIBS}
  )
  message(STATUS "Sparse gas solver: Vulkan kernels enabled (${SPARSE_GAS_GLSLANG})")
else()
  message(STATUS "Sparse gas solver: Vulkan headers or glslangValidator not found, CPU only")
endif()

blender_add_lib(bf_intern_sparse_gas "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")

# Users include the headers by these paths and learn whether the kernels exist.
target_include_directories(bf_intern_sparse_gas INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/gpu"
)
if(SPARSE_GAS_GPU)
  target_compile_definitions(bf_intern_sparse_gas INTERFACE WITH_SPARSE_GAS_GPU)
endif()
