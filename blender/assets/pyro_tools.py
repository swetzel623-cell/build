# SPDX-License-Identifier: GPL-2.0-or-later
"""The Pyro tab in the 3D viewport's sidebar (N): bake the Pyro simulations of the selected
objects to disk once they look right, delete the bake to change them again, and switch the view
to one that shows the flames.

It also makes the Pyro modifiers' Preset menus load values: choosing a preset copies its values
into the modifier's own inputs, so every preset shows the same panel with that preset's
numbers, and any of them can be changed afterwards. Custom leaves them as they are.

smoke_assets.py installs this file into Blender's scripts/startup folder, so it loads with
Blender; no add-on needs enabling.
"""

from __future__ import annotations

import json

import bpy
from bpy.app.handlers import persistent

SIMULATION = "Pyro Simulation"
EMITTER = "Pyro Emitter"
BAKE_FOLDER = "//pyro_bake"
PRESET_TABLE = "pyro_presets"  # On a modifier's node group: its presets, as JSON.
APPLIED = "pyro_preset_applied"  # On an object: the preset last loaded into each modifier.


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


def preset_table(group):
    raw = group.get(PRESET_TABLE) if group is not None else None
    return json.loads(raw) if raw else None


def modifier_input(modifier, name):
    for item in modifier.node_group.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name == name:
            return getattr(modifier.properties.inputs, item.identifier, None)
    return None


def load_preset(obj, modifier, force=False):
    """Copy the chosen preset's values into the modifier's inputs, if it changed since the last
    time (or `force`). Returns True when it loaded values."""
    table = preset_table(modifier.node_group)
    menu = modifier_input(modifier, "Preset") if table is not None else None
    if menu is None:
        return False
    chosen = menu.value
    key = str(modifier.persistent_uid)
    applied = obj.get(APPLIED)
    applied = applied.to_dict() if applied is not None else {}
    if applied.get(key) == chosen and not force:
        return False
    applied[key] = chosen
    obj[APPLIED] = applied
    if chosen not in table:  # Custom: keep the values.
        return False
    for name, value in table[chosen].items():
        target = modifier_input(modifier, name)
        if target is not None:
            target.value = value
    return True


@persistent
def load_changed_presets(scene, depsgraph):
    for update in depsgraph.updates:
        obj = update.id.original if isinstance(update.id, bpy.types.Object) else None
        if obj is None:
            continue
        for modifier in pyro_modifiers(obj):
            load_preset(obj, modifier)


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
        col.label(text="Played frames stay in memory, baked or not.", icon="INFO")
        col.label(text="To use less: Speed panel, Store Velocity off,")
        col.label(text="Open Borders, Smoke Fade.")


CLASSES = (PYRO_OT_bake, PYRO_OT_delete_bake, PYRO_OT_show_flames, VIEW3D_PT_pyro)


def register():
    for cls in CLASSES:
        # A copy loaded earlier (from scripts/startup, or by a test) gives way to this one.
        existing = cls.__mro__[1].bl_rna_get_subclass_py(cls.__name__)
        if existing is not None:
            bpy.utils.unregister_class(existing)
        bpy.utils.register_class(cls)
    handlers = bpy.app.handlers.depsgraph_update_post
    for old in [h for h in handlers if getattr(h, "__name__", "") == "load_changed_presets"]:
        handlers.remove(old)
    handlers.append(load_changed_presets)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    handlers = bpy.app.handlers.depsgraph_update_post
    for old in [h for h in handlers if getattr(h, "__name__", "") == "load_changed_presets"]:
        handlers.remove(old)


if __name__ == "__main__":
    register()
