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
