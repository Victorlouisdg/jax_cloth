import bpy
from jax_cloth import utils, sim

#from jax import grad
import jax.numpy as jnp

#from jax_cloth import utils

#def f(x):
#  y = jnp.sin(x)
#  def g(z):
#    return jnp.cos(z) * jnp.tan(y.sum()) * jnp.tanh(x).sum()
#  return grad(lambda w: jnp.sum(g(w)))(x)


#x = 1.0

#print(f(x))

#print(grad(lambda x: jnp.sum(f(x)))(x))

#utils.test_util()

positions, triangles = utils.get_mesh('Shirt')