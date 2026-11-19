# SPDX-License-Identifier: GPL-2.0-or-later
"""The Pyro tab in the 3D viewport's sidebar (N): bake the Pyro simulations of the selected
objects to disk once they look right, delete the bake to change them again, and switch the view
to one that shows the flames.

smoke_assets.py installs this file into Blender's scripts/startup folder, so it loads with
Blender; no add-on needs enabling.
"""

from __future__ import annotations

import bpy

SIMULATION = "Pyro Simulation"
EMITTER = "Pyro Emitter"
BAKE_FOLDER = "//pyro_bake"


def pyro_modifiers(obj):
    """The object's Pyro Emitter and Pyro Simulation modifiers."""
    return [modifier for modifier in getattr(obj, "modifiers", ())
            if modifier.type == "NODES" and modifier.node_group is not None
            and modifier.node_group.name.split(".")[0] in (SIMULATION, EMITTER)]


def simulations(objects):
    return [obj for obj in objects
            if any(m.node_group.name.split(".")[0] == SIMULATION for m in pyro_modifiers(obj))]


def bake_folder(obj, modifier):
    return f"{BAKE_FOLDER}/{bpy.path.clean_name(obj.name)}_{bpy.path.clean_name(modifier.name)}"


class PYRO_OT_bake(bpy.types.Operator):
    """Bake the Pyro simulation of the selected objects over the scene's frame range, to a
    pyro_bake folder next to the .blend file. Playback then reads the bake"""

    bl_idname = "pyro.bake"
    bl_label = "Bake Pyro"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(simulations(context.selected_objects))

    def execute(self, context):
        if not bpy.data.is_saved:
            self.report({"ERROR"}, "Save the .blend file first: the bake goes in a folder "
                                   "next to it")
            return {"CANCELLED"}
        for obj in simulations(context.selected_objects):
            for modifier in pyro_modifiers(obj):
                modifier.bake_target = "DISK"
                modifier.bake_directory = bake_folder(obj, modifier)
        context.scene.frame_set(context.scene.frame_start)
        if context.window is not None:
            # Runs as a job with a progress bar; Esc stops it.
            return bpy.ops.object.simulation_nodes_cache_bake("INVOKE_DEFAULT", selected=True)
        return bpy.ops.object.simulation_nodes_cache_bake(selected=True)


class PYRO_OT_delete_bake(bpy.types.Operator):
    """Delete the selected objects' Pyro bake, so the simulation runs again with any new
    settings"""

    bl_idname = "pyro.delete_bake"
    bl_label = "Delete Bake"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(simulations(context.selected_objects))

    def execute(self, context):
        result = bpy.ops.object.simulation_nodes_cache_delete(selected=True)
        context.scene.frame_set(context.scene.frame_start)
        return result


class PYRO_OT_show_flames(bpy.types.Operator):
    """Show the smoke lit and the flames glowing: Material Preview, with sharper volumes and
    shadows in the smoke"""

    bl_idname = "pyro.show_flames"
    bl_label = "Show Flames"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        eevee = context.scene.eevee
        eevee.volumetric_tile_size = "4"
        eevee.use_volumetric_shadows = True
        eevee.volumetric_end = max(eevee.volumetric_end, 200.0)
        screen = context.screen
        for area in screen.areas if screen else ():
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
        return {"FINISHED"}


class VIEW3D_PT_pyro(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pyro"
    bl_label = "Pyro"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="View", icon="LIGHT_SUN")
        box.operator(PYRO_OT_show_flames.bl_idname, icon="SHADING_TEXTURE")
        box.label(text="Solid view: flames show as orange shapes")

        box = layout.box()
        box.label(text="Bake", icon="FILE_CACHE")
        selected = simulations(context.selected_objects)
        if not selected:
            box.label(text="Select objects with a Pyro Simulation")
        else:
            box.label(text=", ".join(obj.name for obj in selected[:3])
                      + (" ..." if len(selected) > 3 else ""))
        row = box.row(align=True)
        row.prop(scene, "frame_start", text="Start")
        row.prop(scene, "frame_end", text="End")
        if not bpy.data.is_saved:
            box.label(text="Save the .blend file first", icon="ERROR")
        else:
            box.label(text="Into a pyro_bake folder next to the file")
        row = box.row(align=True)
        row.scale_y = 1.4
        row.operator(PYRO_OT_bake.bl_idname, icon="RENDER_ANIMATION")
        row.operator(PYRO_OT_delete_bake.bl_idname, icon="TRASH")
        col = box.column(align=True)
        col.label(text="Happy with it? Bake, then play or render.")
        col.label(text="To change settings, delete the bake first.")


CLASSES = (PYRO_OT_bake, PYRO_OT_delete_bake, PYRO_OT_show_flames, VIEW3D_PT_pyro)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
