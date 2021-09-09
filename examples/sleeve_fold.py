from jax_cloth import utils, sim
from functools import partial
import jax.numpy as jnp
from jax import grad, vmap
from jax.ops import index

object_name = "Shirt"

positions, triangles = utils.get_mesh(object_name)
positions = jnp.array(positions)
triangles = jnp.array(triangles)

positions_uv = positions[:, :2]


velocities = jnp.zeros_like(positions)
initial_state = positions, velocities

# from sim import triangle_stretch_energy

energy_fn = partial(sim.triangle_stretch_energy, ku=10000.0, kv=10000.0)

# TODO think about how I can simplify this.
def mesh_energy(positions_flat, positions_uv, triangles, energy_fn):
    positions = positions_flat.reshape(-1, 3)
    energies = vmap(energy_fn)(positions[triangles], positions_uv[triangles])
    total_energy = jnp.sum(energies)
    return -total_energy


mesh_energy_fn = partial(
    mesh_energy, positions_uv=positions_uv, triangles=triangles, energy_fn=energy_fn
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

dt = 0.01

build_fn = partial(sim.build_system_BW, forces_fn=forces_fn_grav, M=M)

S = jnp.identity(system_size)

pinned = jnp.array([0, 4])

for i in pinned:
    S = S.at[index[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]].set(0.0)

z = jnp.zeros(system_size)


step_fn = partial(sim.step_PPCG, build_fn=build_fn, S=S, z=z, dt=dt)


def step_dummy(carry, input):
    positions, velocities = carry

    positions_new = positions.at[:, 2].add(-0.01)

    carry = (positions_new, velocities)
    output = positions_new
    return (carry, output)


history = sim.simulate(step_dummy, initial_state, 1000)

import bpy

# bpy.context.scene.frame_set(0)
# for i in range(99):
#     bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
#     ob = bpy.data.objects[object_name]
#     mesh = ob.data
#     for j, v in enumerate(mesh.vertices):
#         v.co = history[i, j, :]

ob = bpy.data.objects[object_name]
mesh = ob.data

sk_basis = ob.shape_key_add(name="Basis")

import time
import bmesh

t0 = time.time()

sk = ob.shape_key_add(name="Deform")

for j, v in enumerate(mesh.vertices):
    v.co.x += 0.2

# bpy.ops.object.mode_set(mode="EDIT")
# bm = bmesh.from_edit_mesh(mesh)
# for j, v in enumerate(bm.verts):
#     v.co.x += 0.2
# bmesh.update_edit_mesh(mesh, True)
# bpy.ops.object.mode_set(mode="OBJECT")


# bm = bmesh.new()
# bm.from_mesh(mesh)

# print(bm.verts)

# for i in range(10):
#     # Method 1: key each vertex
#     # for j, v in enumerate(mesh.vertices):
#     #     v.co = history[i, j, :]
#     #     v.keyframe_insert("co", frame=i)

#     # Method 2: add shapekeys
#     # sk = ob.shape_key_add(name="Deform")
#     # for j in range(len(mesh.vertices)):
#     #     sk.data[j].co = history[i, j, :]

#     # Method 3: edit bmesh
#     sk = ob.shape_key_add(name="Deform")

#     for j, v in enumerate(bm.verts):
#         v.co = history[i, j, :]
#         # print(v.co)

#     # print(type(sk.data[0]))
#     bm.to_mesh(mesh)

#     # sk.keyframe_insert(data_path="value", frame=i)

#     # Does not work: keyframe all vertices at once
#     # mesh.keyframe_insert("vertices", frame=i)


t1 = time.time()

print(t1 - t0, "seconds elapsed")

print(history.shape)
