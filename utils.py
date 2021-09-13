import bpy


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
    return positions, triangles


def bezier_from_points(p0, p1, p2, p3):
    bpy.ops.curve.primitive_bezier_curve_add()
    curve = bpy.context.active_object.data
    b0, b1 = curve.splines[0].bezier_points

    b0.handle_left_type = "FREE"
    b0.handle_right_type = "FREE"
    b1.handle_left_type = "FREE"
    b1.handle_right_type = "FREE"

    b0.co = p0
    b0.handle_right = p1
    b0.handle_left = p0 - p1
    b1.handle_left = p2
    b1.handle_right = p3 - p2
    b1.co = p3
