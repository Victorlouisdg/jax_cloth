import bpy
from jax_cloth import utils, sims, sim, viz
from jax import jit, value_and_grad, jacfwd
from functools import partial
import jax.numpy as jnp
import time
import datetime
import matplotlib.pyplot as plt

t0 = time.time()

object_name = "Shirt"
goal_object_name = "Shirt Folded"

ob = bpy.data.objects[object_name]
ob_goal = bpy.data.objects[goal_object_name]

ob.animation_data_clear()
ob_goal.animation_data_clear()

mesh = ob.data

positions, triangles = utils.get_mesh(object_name)
positions_goal, _ = utils.get_mesh(goal_object_name)
positions_uv = positions[:, :2]
velocities = jnp.zeros_like(positions)
initial_state = positions, velocities

amount_of_vertices = positions_uv.shape[0]


fps = 24
substeps = 10
dt = 1.0 / (fps * substeps)
seconds = 1
frames = fps * seconds
steps = frames * substeps

animated = jnp.array([0])

sim_trajectory = partial(
    sims.bw,
    initial_state=initial_state,
    steps=steps,
    positions_uv=positions_uv,
    triangles=triangles,
    animated=animated,
    dt=dt,
)

sim_trajectory2 = lambda trajectory: sims.bw(
    initial_state, trajectory, steps, positions_uv, triangles, animated, dt
)


def initialize_control_points(start_position):
    p0 = start_position
    # p3 = p0.at[0].multiply(-1.0)  # mirror over YZ-plane
    p3 = p0.at[0].multiply(-0.5)  # mirror over YZ-plane

    h = jnp.linalg.norm(p3 - p0) / 2
    p1 = 2 / 3 * p0 + 1 / 3 * p3 + jnp.array([0, 0, h])
    p2 = 1 / 3 * p0 + 2 / 3 * p3 + jnp.array([0, 0, h])

    return p0, p1, p2, p3


p0, p1, p2, p3 = initialize_control_points(positions[0])
viz.bezier_from_points(p0, p1, p2, p3)


def bezier(p0, p1, p2, p3, steps):
    samples = jnp.array(
        [sim.bezier_evaluate(p0, p1, p2, p3, (i + 1) / steps) for i in range(steps)]
    )
    samples = samples.reshape(steps, 1, 3)
    return samples


trajectory = bezier(p0, p1, p2, p3, steps)

# history = sim_trajectory(trajectory=trajectory)

z = 0.09
y = 1.02


def loss(x):
    p3 = jnp.array([x, y, z])
    trajectory = bezier(p0, p1, p2, p3, steps)
    history = sim_trajectory(trajectory=trajectory)
    return jnp.sum(jnp.square(positions_goal - history[-1])) / amount_of_vertices


# step = 0.3
# step = 0.025
step = 0.005
# step = 0.001

xs = jnp.arange(0.1, 1.0 + step, step)
losses = []
grads = []

for x in xs:
    # Reverse mode
    # loss_value, loss_grad = jit(value_and_grad(loss))(x)

    # Forward mode
    jit_loss = jit(loss)
    loss_value = jit_loss(x)
    loss_grad = jit(jacfwd(jit_loss))(x)

    losses.append(loss_value)
    grads.append(loss_grad)
    print(f"Loss for x = {x} is {loss_value}, grad is {loss_grad}")


fig = plt.figure()

timestamp = datetime.datetime.now()

plt.title("Loss Landscape for Toy Task: 1D Sleeve Folding")
plt.xlabel("x of trajectory endpoint")
plt.ylabel("Loss MSE (deviation from goal)")
plt.plot(xs, losses)
plt.savefig(f"loss_landscape_{timestamp}.png")

fig = plt.figure()

plt.title("Gradient of Loss for Toy Task: 1D Sleeve Folding")
plt.xlabel("x of trajectory endpoint")
plt.ylabel("Gradient of the Loss")
plt.plot(xs, grads)
plt.hlines(0.0, min(xs), max(xs), color="black", linewidth=0.5)
plt.savefig(f"grad_landscape_{timestamp}.png")

# print(len(history))

# for i in range(frames):
#     for j, v in enumerate(mesh.vertices):
#         v.co = history[substeps * i, j, :]
#         v.keyframe_insert("co", frame=i + 1)

# bpy.context.scene.frame_set(0)
# bpy.context.scene.frame_end = frames

# # Set positions back to original
# for j, v in enumerate(mesh.vertices):
#     v.co = positions[j]

t1 = time.time()
print(f"Execution took {t1 - t0} seconds.")
