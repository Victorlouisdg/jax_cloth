from jax_cloth import utils, sim
from functools import partial
import jax.numpy as jnp
from jax import grad, vmap
from jax.ops import index
import time
import bpy

bpy.context.scene.frame_set(0)

object_name = "Shirt"
ob = bpy.data.objects[object_name]
ob.animation_data_clear()

positions, triangles = utils.get_mesh(object_name)
positions = jnp.array(positions)
triangles = jnp.array(triangles)

positions_uv = positions[:, :2]

velocities = jnp.zeros_like(positions)
initial_state = positions, velocities

triangle_energy_fn = lambda x, y: partial(
    sim.stretch_energy_BW, ku=10000.0, kv=10000.0
)(x, y) + partial(sim.shear_energy_BW, k=100.0)(x, y)

# TODO think about how I can simplify this.
def mesh_energy(positions_flat, positions_uv, triangles, triangle_energy_fn):
    positions = positions_flat.reshape(-1, 3)
    energies = vmap(triangle_energy_fn)(positions[triangles], positions_uv[triangles])
    total_energy = jnp.sum(energies)
    return -total_energy


mesh_energy_fn = partial(
    mesh_energy,
    positions_uv=positions_uv,
    triangles=triangles,
    triangle_energy_fn=triangle_energy_fn,
)

forces_fn = grad(mesh_energy_fn)

amount_of_vertices = positions_uv.shape[0]
system_size = 3 * amount_of_vertices

m = 1.0 / amount_of_vertices
masses = jnp.full(system_size, m)

g = [0.0, 0.0, -9.81] * amount_of_vertices
gravity = masses * jnp.array(g)

M = jnp.diag(masses)

forces_fn_grav = lambda x: forces_fn(x) + gravity

fps = 24
substeps = 10
dt = 1.0 / (fps * substeps)

build_fn = partial(sim.build_system_BW, forces_fn=forces_fn_grav, M=M)

S = jnp.identity(system_size)

pinned = jnp.array([0, 24, 50, 91, 103])

for i in pinned:
    S = S.at[index[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]].set(0.0)

z = jnp.zeros(system_size)


step_fn = partial(sim.step_PPCG, build_fn=build_fn, S=S, z=z, dt=dt)

seconds = 5
frames = fps * seconds
steps = frames * substeps


mesh = ob.data

t0 = time.time()

grasped_vertex = 0

p0 = positions[grasped_vertex]
p3 = p0.at[0].multiply(-1.0)  # mirror over YZ-plane
p1 = 2 / 3 * p0 + 1 / 3 * p3 + jnp.array([0, 0, 1])
p2 = 1 / 3 * p0 + 2 / 3 * p3 + jnp.array([0, 0, 1])

bezier_sampled = [
    sim.bezier_evaluate(p0, p1, p2, p3, (i + 1) / steps) for i in range(steps)
]

trajectory = jnp.array(bezier_sampled)

# trajectory = jnp.tile(p0, (steps, 1))

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

utils.bezier_from_points(p0, p1, p2, p3)

history = sim.simulate(step_fn, initial_state, trajectory, steps)

for i in range(frames):
    for j, v in enumerate(mesh.vertices):
        v.co = history[substeps * i, j, :]
        v.keyframe_insert("co", frame=i + 1)

bpy.context.scene.frame_set(0)
bpy.context.scene.frame_end = frames

# Set positions back to original
for j, v in enumerate(mesh.vertices):
    v.co = positions[j]


t1 = time.time()

print(t1 - t0, "seconds elapsed to animate mesh")
