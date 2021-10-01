from jax_cloth import utils, sim, viz
from functools import partial
import jax.numpy as jnp
from jax import grad, vmap, jit
import time
from jax.lax import cond
import bpy

bpy.context.scene.frame_set(0)

object_name = "Shirt"
goal_object_name = "Shirt Folded"

ob = bpy.data.objects[object_name]
ob.animation_data_clear()

ob_goal = bpy.data.objects[goal_object_name]
ob_goal.animation_data_clear()

positions, triangles = utils.get_mesh(object_name)
positions = jnp.array(positions)
triangles = jnp.array(triangles)

positions_goal, _ = utils.get_mesh(goal_object_name)
positions_goal = jnp.array(positions_goal)


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


def rayleigh_drag(position, velocity):
    return cond(
        position[2] < 0.0,
        lambda v: -10.0 * v,
        lambda _: jnp.zeros(3),
        operand=velocity,
    )


def drag_fn(positions, velocities):
    positions = positions.reshape(-1, 3)
    velocities = velocities.reshape(-1, 3)
    return vmap(rayleigh_drag)(positions, velocities).flatten()


forces_fn_grav = lambda x, v: forces_fn(x) + gravity + drag_fn(x, v)

fps = 24
substeps = 10
dt = 1.0 / (fps * substeps)

build_fn = partial(sim.build_system_BW, forces_fn=forces_fn_grav, M=M)

animated = jnp.array([0, 1, 2])


step_fn = jit(partial(sim.step_PPCG, build_fn=build_fn, animated=animated, dt=dt))

seconds = 5
frames = fps * seconds
steps = frames * substeps


mesh = ob.data

t0 = time.time()


# def generate_bezier(start_position):
#     p0 = start_position
#     p3 = p0.at[0].multiply(-1.0)  # mirror over YZ-plane
#     h = jnp.linalg.norm(p3 - p0) / 2
#     p1 = 2 / 3 * p0 + 1 / 3 * p3 + jnp.array([0, 0, h])
#     p2 = 1 / 3 * p0 + 2 / 3 * p3 + jnp.array([0, 0, h])

#     bezier_samples = [
#         sim.bezier_evaluate(p0, p1, p2, p3, (i + 1) / steps) for i in range(steps)
#     ]

#     # For debugging
#     viz.bezier_from_points(p0, p1, p2, p3)

#     return jnp.array(bezier_samples)


def initialize_control_points(start_position):
    p0 = start_position
    p3 = p0.at[0].multiply(-1.0)  # mirror over YZ-plane
    h = jnp.linalg.norm(p3 - p0) / 2
    p1 = 2 / 3 * p0 + 1 / 3 * p3 + jnp.array([0, 0, h])
    p2 = 1 / 3 * p0 + 2 / 3 * p3 + jnp.array([0, 0, h])

    return p0, p1, p2, p3


# trajectory = jnp.stack([generate_bezier(positions[a]) for a in animated], axis=1)

# history = sim.simulate(step_fn, initial_state, trajectory, steps)

# p0, p3


def loss(free_control_points):

    p1, p2 = free_control_points

    # generate trajectory
    # sim with trajectory
    # calculate loss

    return jnp.sum(jnp.square(positions_goal - positions)) / amount_of_vertices


# grad(loss) = grad_loss
# optimize loss

print("loss", loss(history[-1]))
# for i in range(frames):
#     for j, v in enumerate(mesh.vertices):
#         v.co = history[substeps * i, j, :]
#         v.keyframe_insert("co", frame=i + 1)

# bpy.context.scene.frame_set(0)
# bpy.context.scene.frame_end = frames

# Set positions back to original
for j, v in enumerate(mesh.vertices):
    v.co = positions[j]


t1 = time.time()

print(t1 - t0, "seconds elapsed to animate mesh")
