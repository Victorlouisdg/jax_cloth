import bpy


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


###
# import bmesh

# verts = [(1, 1, 1), (0, 0, 0)]  # 2 verts made with XYZ coords
# mesh2 = bpy.data.meshes.new("mesh")  # add a new mesh
# object = bpy.data.objects.new("MESH", mesh2)
# bpy.context.collection.objects.link(object)

# bm2 = bmesh.new()

# for v in bezier_sampled:
#     bm2.verts.new(v)  # add a new vert

# # make the bmesh the object's mesh
# bm2.to_mesh(mesh2)
# bm2.free()  # always do this when finished
###
