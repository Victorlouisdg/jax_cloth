from jax_cloth import sim
from jax import grad, vmap
from jax.lax import cond
import jax.numpy as jnp

from functools import partial

triangle_energy_fn = lambda x, y: partial(
    sim.stretch_energy_BW, ku=10000.0, kv=10000.0
)(x, y) + partial(sim.shear_energy_BW, k=100.0)(x, y)

# TODO think about how I can simplify this.
def mesh_energy(positions_flat, positions_uv, triangles, triangle_energy_fn):
    positions = positions_flat.reshape(-1, 3)
    energies = vmap(triangle_energy_fn)(positions[triangles], positions_uv[triangles])
    total_energy = jnp.sum(energies)
    return -total_energy


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


def bw(initial_state, trajectory, steps, positions_uv, triangles, animated, dt):
    mesh_energy_fn = partial(
        mesh_energy,
        positions_uv=positions_uv,
        triangles=triangles,
        triangle_energy_fn=triangle_energy_fn,
    )

    amount_of_vertices = positions_uv.shape[0]
    system_size = 3 * amount_of_vertices

    m = 1.0 / amount_of_vertices
    masses = jnp.full(system_size, m)
    M = jnp.diag(masses)

    g = [0.0, 0.0, -9.81] * amount_of_vertices
    gravity = masses * jnp.array(g)

    forces_fn = grad(mesh_energy_fn)
    forces_fn_grav = lambda x, v: forces_fn(x) + drag_fn(x, v) + gravity

    build_fn = partial(sim.build_system_BW, forces_fn=forces_fn_grav, M=M)

    step_fn = partial(sim.step_PPCG, build_fn=build_fn, animated=animated, dt=dt)

    return sim.simulate(step_fn, initial_state, trajectory, steps)
