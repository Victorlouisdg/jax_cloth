""" The experiment file is responsible for several things. 
    1) Reading in the meshes
    2) Choosing the simulator
    3) Initializing parameters to optimize
    4) Definig the loss
    5) Running the experiment
    6) Saving results
"""
import bpy

from jax_cloth import sims, bezier
from jax_cloth import blender_utils as bu

import jax.numpy as jnp
from jax import jit, value_and_grad, jacfwd
from functools import partial

import os
import time
import datetime
import matplotlib.pyplot as plt

t0 = time.time()

# 1) Reading in the meshes
obj = "Start"
obj_goal = "Goal"

positions, triangles = bu.get_mesh(obj)
positions_goal, _ = bu.get_mesh(obj_goal)

positions_uv = positions[:, :2]
velocities = jnp.zeros_like(positions)
initial_state = positions, velocities

amount_of_vertices = positions.shape[0]

# 2) Choosing the simulator
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

# 3) Initializing the parameters to optimize
def initialize_control_points(start_position):
    p0 = start_position
    # p3 = p0.at[0].multiply(-1.0)  # mirror over YZ-plane
    p3 = p0.at[0].multiply(-0.5)

    h = jnp.linalg.norm(p3 - p0) / 2
    p1 = 2 / 3 * p0 + 1 / 3 * p3 + jnp.array([0, 0, h])
    p2 = 1 / 3 * p0 + 2 / 3 * p3 + jnp.array([0, 0, h])

    return p0, p1, p2, p3


p0, p1, p2, p3 = initialize_control_points(positions[0])

y = 1.02
z = 0.09

# 4) Defining the loss
def loss(x):
    p3 = jnp.array([x, y, z])
    trajectory = bezier.sample(p0, p1, p2, p3, steps)
    history = sim_trajectory(trajectory=trajectory)
    return jnp.sum(jnp.square(positions_goal - history[-1])) / amount_of_vertices


# 5) Running the experiment
x_step = 1.0
xs = [0.5]  # jnp.arange(0.1, 1.0 + x_step, x_step)
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

# 6) Saving the results
jax_cloth_dir = os.path.dirname(os.path.dirname(__file__))
timestamp = datetime.datetime.now()
filename = os.path.basename(__file__).split(".")[0]

output_dir = os.path.join(jax_cloth_dir, "output", f"{filename} {timestamp}")

os.mkdir(output_dir)

# 6.1) Creating plots

fig = plt.figure()
plt.title("Loss Landscape for Toy Task: 1D Sleeve Folding")
plt.xlabel("x of trajectory endpoint")
plt.ylabel("Loss MSE (deviation from goal)")
plt.plot(xs, losses)
plt.savefig(os.path.join(output_dir, f"loss_landscape_{timestamp}.png"))

fig = plt.figure()
plt.title("Gradient of Loss for Toy Task: 1D Sleeve Folding")
plt.xlabel("x of trajectory endpoint")
plt.ylabel("Gradient of the Loss")
plt.plot(xs, grads)
plt.hlines(0.0, min(xs), max(xs), color="black", linewidth=0.5)
plt.savefig(os.path.join(output_dir, f"grad_landscape_{timestamp}.png"))

# 6.2) Resim best result and save animation
trajectory = bezier.sample(p0, p1, p2, p3, steps)

bezier.visualize(p0, p1, p2, p3)

history = sim_trajectory(trajectory=trajectory)

bu.save_animation(obj, positions, history, frames, substeps, output_dir)

t1 = time.time()
print(f"Execution took {t1 - t0} seconds.")
