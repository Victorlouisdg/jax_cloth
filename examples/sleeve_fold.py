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

energy_fn = partial(sim.triangle_stretch_energy, ku=1000.0, kv=1000.0)

print("start energy", energy_fn(positions[triangles[0]], positions_uv[triangles[0]]))

# TODO think about how I can simplify this.
def mesh_energy(positions_flat, positions_uv, triangles, energy_fn):
    positions = positions_flat.reshape(-1, 3)
    energies = vmap(energy_fn)(positions[triangles], positions_uv[triangles])
    total_energy = jnp.sum(energies)
    return -total_energy


print("mesh energy", mesh_energy(positions, positions_uv, triangles, energy_fn))

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

forces_fn_grav = lambda x: forces_fn(x) + gravity  # implicit

# forces_fn_grav = lambda x: forces_fn(x) + gravity.reshape(-1, 3) # explicit
# forces_fn_grav = lambda x: gravity.reshape(-1, 3)

print("gravity", gravity.shape)
print("masses", masses.shape)
print("positions", positions.shape)
print("positions flat", positions.flatten().shape)


fps = 24
substeps = 10
dt = 1.0 / (fps * substeps)

build_fn = partial(sim.build_system_BW, forces_fn=forces_fn_grav, M=M)

S = jnp.identity(system_size)

pinned = jnp.array([91, 103])

for i in pinned:
    S = S.at[index[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]].set(0.0)

z = jnp.zeros(system_size)


step_fn = partial(sim.step_PPCG, build_fn=build_fn, S=S, z=z, dt=dt)

# step_fn = partial(
#     sim.step_explicit_euler,
#     forces_fn=forces_fn_grav,
#     masses=masses,
#     pinned=pinned,
#     dt=dt,
# )

# step_fn(initial_state, None)


# def step_dummy(carry, input):
#     positions, velocities = carry

#     positions_new = positions.at[:, 2].add(-dt)

#     carry = (positions_new, velocities)
#     output = positions_new
#     return (carry, output)


seconds = 1
frames = fps * seconds
steps = frames * substeps

history = sim.simulate(step_fn, initial_state, steps)

mesh = ob.data

t0 = time.time()

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
