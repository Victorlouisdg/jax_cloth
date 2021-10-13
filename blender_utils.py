import os
import bpy
import jax.numpy as jnp


def select_only(blender_object):
    """Selects and actives a Blender object and deselects all others"""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = blender_object
    blender_object.select_set(True)


def get_mesh(object_name):
    ob = bpy.data.objects[object_name]
    mesh = ob.data
    positions = [(ob.matrix_world @ v.co) for v in mesh.vertices]
    mesh.calc_loop_triangles()
    triangles = [tri.vertices for tri in mesh.loop_triangles]
    return jnp.array(positions), jnp.array(triangles)


def save_animation(obj_name, positions, history, frames, substeps, output_dir):
    mesh = bpy.data.objects[obj_name].data

    # Set positions back to original at frame 1
    for j, v in enumerate(mesh.vertices):
        v.co = positions[j]
        v.keyframe_insert("co", frame=1)

    for i in range(frames):
        for j, v in enumerate(mesh.vertices):
            v.co = history[substeps - 1 + (substeps * i), j, :]
            v.keyframe_insert("co", frame=i + 2)

    bpy.context.scene.frame_set(frames + 1)
    bpy.context.scene.frame_end = frames + 25  # pause one second on final result

    output_file = os.path.join(output_dir, "result.blend")
    bpy.ops.wm.save_as_mainfile(filepath=output_file)
