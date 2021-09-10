import jax.numpy as jnp
from jax import grad, jacobian, vmap
from jax.lax import scan
from jax.ops import index
import jax.scipy.sparse


def deformation_gradient(positions, positions_uv):
    u0, u1, u2 = positions_uv
    Dm = jnp.column_stack([u1 - u0, u2 - u0])
    Dm_inv = jnp.linalg.inv(Dm)

    x0, x1, x2 = positions
    Ds = jnp.column_stack([x1 - x0, x2 - x0])

    F = Ds @ Dm_inv
    return F


def area(triangle_vertices):
    v0, v1, v2 = triangle_vertices
    return jnp.linalg.norm(jnp.cross(v1 - v0, v2 - v0)) / 2.0


def triangle_stretch_energy(positions, positions_uv, ku, kv):
    a = area(positions_uv)

    F = deformation_gradient(positions, positions_uv)
    wu, wv = jnp.hsplit(F, 2)
    Cu = jnp.linalg.norm(wu) - 1
    Cv = jnp.linalg.norm(wv) - 1

    Eu = 0.5 * a * ku * (Cu ** 2)
    Ev = 0.5 * a * kv * (Cv ** 2)
    return Eu + Ev


def mesh_energy(positions_flat, positions_uv, triangles, energy_fn):
    positions = positions_flat.reshape(-1, 3)
    energies = vmap(energy_fn)(positions[triangles], positions_uv[triangles])
    total_energy = jnp.sum(energies)
    return -total_energy


def simulate(step_fn, initial_state, amount_of_steps):
    carry, outputs = scan(step_fn, initial_state, xs=None, length=amount_of_steps)
    return outputs


def build_system_BW(positions, velocities, dt, forces_fn, M):
    h = dt
    x0 = positions.flatten()
    v0 = velocities.flatten()

    f0 = forces_fn(x0)
    dfdx = jacobian(forces_fn)(x0)

    # Equation (16) in Baraff-Witkin.
    A = M - (h * h) * dfdx
    b = h * (f0 + h * (dfdx @ v0))

    return A, b


def filter_system(A, b, S, z):
    # Prefiltering the system (see also: Dynamic Deformables section 10.1.1)
    I = jnp.identity(A.shape[0])
    LHS = (S @ A @ S) + I - S
    c = b - A @ z
    rhs = S @ c
    return LHS, rhs


def PPCG(A, b, S, z):
    LHS, rhs = filter_system(A, b, S, z)
    LHS_fn = lambda x: LHS @ x

    y = jax.scipy.sparse.linalg.cg(LHS_fn, rhs)[0]
    x = y + z
    return x


def step_PPCG(carry, input, build_fn, S, z, dt):
    positions, velocities = carry

    A, b = build_fn(positions, velocities, dt)

    x = PPCG(A, b, S, z)

    delta_v = x.reshape(-1, 3)

    velocities_new = velocities + delta_v
    positions_new = positions + velocities_new * dt

    carry = (positions_new, velocities_new)
    output = positions_new
    return (carry, output)


def step_explicit_euler(carry, input, forces_fn, masses, pinned, dt):
    positions, velocities = carry

    forces = forces_fn(positions)

    accelerations = forces / masses.reshape(-1, 3)
    accelerations = accelerations.at[pinned].set(0.0)

    velocities_new = velocities + accelerations * dt
    positions_new = positions + velocities * dt

    carry = (positions_new, velocities_new)
    output = positions_new
    return (carry, output)
