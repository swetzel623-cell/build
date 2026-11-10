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
)

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

GEOMETRY_ANCHOR = """add_node_discovery(
  bf_nodes_geometry_generated
"""
GEOMETRY_INSERT = """if(WITH_FLUID AND WITH_OPENVDB)
  list(APPEND SRC
    nodes/node_geo_manta_gas_step.cc
  )
  list(APPEND INC
    ../../../../intern/mantaflow/extern
    ../../../../intern/mantaflow/intern
  )
  list(APPEND LIB
    PRIVATE bf::intern::mantaflow
  )
endif()

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
    blender_root = args.blender_root.resolve()
    try:
        verify_commit(blender_root, args.allow_unpinned)
        for relative in OVERLAY_FILES:
            if not (overlay_root / relative).is_file():
                raise RuntimeError(f"Missing overlay source: {relative}")

        intern_path = blender_root / INTERN_CMAKE
        geometry_path = blender_root / GEOMETRY_CMAKE
        intern_text = intern_path.read_text()
        geometry_text = geometry_path.read_text()
        intern_text = replace_once(intern_text, INTERN_OLD, INTERN_NEW, INTERN_CMAKE)
        geometry_text = replace_once(
            geometry_text, GEOMETRY_ANCHOR, GEOMETRY_INSERT + GEOMETRY_ANCHOR, GEOMETRY_CMAKE
        )

        if args.check:
            print(f"Integration dry run passed for {blender_root}")
            return 0

        for relative in OVERLAY_FILES:
            destination = blender_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((overlay_root / relative).read_bytes())
        intern_path.write_text(intern_text)
        geometry_path.write_text(geometry_text)
        print(f"Applied Geometry Nodes gas-step integration to {blender_root}")
        print("Build with WITH_FLUID=ON and WITH_OPENVDB=ON; run runtime validation before use.")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"Integration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
