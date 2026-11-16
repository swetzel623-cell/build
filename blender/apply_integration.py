#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Apply the reviewed Geometry Nodes gas-step overlay to a pinned Blender source tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "377d577d520472f2702e456b26c17dffc663dc52"

OVERLAY_FILES = (
    "intern/mantaflow/extern/manta_gas_step_API.h",
    "intern/mantaflow/intern/manta_gas_layout.h",
    "intern/mantaflow/intern/manta_gas_step_driver.h",
    "intern/mantaflow/intern/manta_gas_step_API.cpp",
    "source/blender/nodes/geometry/nodes/node_geo_manta_gas_step.cc",
    "intern/sparse_gas/CMakeLists.txt",
)

# The sparse solver's sources, copied from the repository root into intern/sparse_gas.
SPARSE_DESTINATION = "intern/sparse_gas"
SPARSE_FILES = (
    "include/manta_improve/sparse_gas.hpp",
    "src/sparse_gas.cpp",
    "src/sparse_multigrid.cpp",
    "gpu/vulkan_compute.hpp",
    "gpu/vulkan_compute.cpp",
    "gpu/sparse_gas_gpu.hpp",
    "gpu/sparse_gas_gpu.cpp",
)
SPARSE_SHADER_DIR = "gpu/shaders"

INTERN_LIST = "intern/CMakeLists.txt"
INTERN_LIST_OLD = """if(WITH_MOD_FLUID)
  add_subdirectory(mantaflow)
endif()
"""
INTERN_LIST_NEW = """if(WITH_MOD_FLUID)
  add_subdirectory(mantaflow)
  add_subdirectory(sparse_gas)
endif()
"""

INTERN_CMAKE = "intern/mantaflow/CMakeLists.txt"
GEOMETRY_CMAKE = "source/blender/nodes/geometry/CMakeLists.txt"

INTERN_OLD = """  intern/manta_python_API.cpp
  intern/manta_fluid_API.cpp
  intern/MANTA_main.cpp

  extern/manta_python_API.h
  extern/manta_fluid_API.h
  intern/MANTA_main.h
"""
INTERN_NEW = """  intern/manta_python_API.cpp
  intern/manta_fluid_API.cpp
  intern/manta_gas_step_API.cpp
  intern/MANTA_main.cpp

  extern/manta_python_API.h
  extern/manta_fluid_API.h
  extern/manta_gas_step_API.h
  intern/MANTA_main.h
  intern/manta_gas_layout.h
  intern/manta_gas_step_driver.h
"""

# The engine includes Mantaflow's kernel headers directly, so it must see the same kernel
# configuration as `extern_mantaflow`. Upstream sources in this target keep their definitions.
INTERN_LIB_ANCHOR = """blender_add_lib(bf_intern_mantaflow "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
"""
INTERN_LIB_INSERT = """if(WITH_TBB)
  set_source_files_properties(
    intern/manta_gas_step_API.cpp
    PROPERTIES COMPILE_DEFINITIONS "TBB=1"
  )
endif()

"""

GEOMETRY_ANCHOR = """add_node_discovery(
  bf_nodes_geometry_generated
"""
GEOMETRY_INSERT = """list(APPEND SRC
  nodes/node_geo_manta_gas_step.cc
)
if(WITH_MOD_FLUID AND WITH_OPENVDB)
  add_definitions(-DWITH_FLUID)
  list(APPEND INC
    ../../../../intern/mantaflow/extern
    ../../../../intern/mantaflow/intern
  )
  list(APPEND LIB
    PRIVATE bf_intern_mantaflow
    PRIVATE bf_intern_sparse_gas
  )
endif()

"""


# Every built-in node needs an RNA struct; registration asserts without it.
RNA_SOURCE = "source/blender/makesrna/intern/rna_nodes_geometry.cc"
RNA_ANCHOR = """  define("GeometryNode", "GeometryNodeMaterialSelection");
"""
RNA_INSERT = """  define("GeometryNode", "GeometryNodeMantaGasStep");
"""

MENU_SOURCE = "scripts/startup/bl_ui/node_add_menu_geometry.py"
MENU_ANCHOR = """        self.node_operator(layout, "GeometryNodeXPBDSolver")
"""
MENU_INSERT = """        self.node_operator(layout, "GeometryNodeMantaGasStep")
"""


def replace_once(text: str, old: str, new: str, path: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one integration anchor, found {count}")
    return text.replace(old, new, 1)


def verify_commit(blender_root: Path, allow_unpinned: bool) -> None:
    if allow_unpinned:
        return
    result = subprocess.run(
        ["git", "-C", str(blender_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not read the Blender checkout's Git commit")
    actual = result.stdout.strip()
    if actual != PINNED_COMMIT:
        raise RuntimeError(f"Expected Blender commit {PINNED_COMMIT}, found {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blender_root", type=Path, help="Blender source checkout")
    parser.add_argument("--check", action="store_true", help="Validate without writing files")
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Allow a different source revision; API compatibility remains the integrator's responsibility",
    )
    args = parser.parse_args()

    overlay_root = Path(__file__).resolve().parent
    repository_root = overlay_root.parent
    blender_root = args.blender_root.resolve()
    try:
        verify_commit(blender_root, args.allow_unpinned)
        for relative in OVERLAY_FILES:
            if not (overlay_root / relative).is_file():
                raise RuntimeError(f"Missing overlay source: {relative}")
        sparse_sources = [repository_root / relative for relative in SPARSE_FILES]
        shader_dir = repository_root / SPARSE_SHADER_DIR
        sparse_sources += sorted(shader_dir.glob("*.comp")) + sorted(shader_dir.glob("*.glsl"))
        for source in sparse_sources:
            if not source.is_file():
                raise RuntimeError(f"Missing sparse solver source: {source}")
        if (blender_root / SPARSE_DESTINATION / "src").exists():
            raise RuntimeError(f"{SPARSE_DESTINATION} already holds solver sources")

        intern_list_path = blender_root / INTERN_LIST
        intern_list_text = replace_once(
            intern_list_path.read_text(), INTERN_LIST_OLD, INTERN_LIST_NEW, INTERN_LIST
        )

        intern_path = blender_root / INTERN_CMAKE
        geometry_path = blender_root / GEOMETRY_CMAKE
        intern_text = intern_path.read_text()
        geometry_text = geometry_path.read_text()
        intern_text = replace_once(intern_text, INTERN_OLD, INTERN_NEW, INTERN_CMAKE)
        intern_text = replace_once(
            intern_text, INTERN_LIB_ANCHOR, INTERN_LIB_INSERT + INTERN_LIB_ANCHOR, INTERN_CMAKE
        )
        geometry_text = replace_once(
            geometry_text, GEOMETRY_ANCHOR, GEOMETRY_INSERT + GEOMETRY_ANCHOR, GEOMETRY_CMAKE
        )

        rna_path = blender_root / RNA_SOURCE
        menu_path = blender_root / MENU_SOURCE
        rna_text = replace_once(rna_path.read_text(), RNA_ANCHOR, RNA_INSERT + RNA_ANCHOR, RNA_SOURCE)
        menu_text = replace_once(
            menu_path.read_text(), MENU_ANCHOR, MENU_ANCHOR + MENU_INSERT, MENU_SOURCE
        )

        if args.check:
            print(f"Integration dry run passed for {blender_root}")
            return 0

        for relative in OVERLAY_FILES:
            destination = blender_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((overlay_root / relative).read_bytes())
        for source in sparse_sources:
            destination = blender_root / SPARSE_DESTINATION / source.relative_to(repository_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        intern_list_path.write_text(intern_list_text)
        intern_path.write_text(intern_text)
        geometry_path.write_text(geometry_text)
        rna_path.write_text(rna_text)
        menu_path.write_text(menu_text)
        print(f"Applied Geometry Nodes gas-step integration to {blender_root}")
        print("Build with WITH_MOD_FLUID=ON and WITH_OPENVDB=ON; run runtime validation before use.")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"Integration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
