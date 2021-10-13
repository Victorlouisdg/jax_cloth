import bpy
import jax.numpy as jnp


def lerp(p0, p1, t):
    return (1 - t) * p0 + t * p1


def evaluate(p0, p1, p2, p3, t):
    a = lerp(p0, p1, t)
    b = lerp(p1, p2, t)
    c = lerp(p2, p3, t)
    d = lerp(a, b, t)
    e = lerp(b, c, t)
    p = lerp(d, e, t)
    return p


def visualize(p0, p1, p2, p3):
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

    bpy.context.object.data.bevel_depth = 0.01

    material = bpy.data.materials.new(name="Bezier")
    material.diffuse_color = (0.92, 1.0, 0.03, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.92, 1.0, 0.03, 1.0)
    bsdf.inputs["Roughness"].default_value = 1
    curve.materials.append(material)


def sample(p0, p1, p2, p3, steps):
    samples = jnp.array(
        [evaluate(p0, p1, p2, p3, (i + 1) / steps) for i in range(steps)]
    )
    samples = samples.reshape(steps, 1, 3)
    return samples


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
