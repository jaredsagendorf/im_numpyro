import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt

def harmonic_restraint(x1, x2, d, k):
    '''
    This is probably a dumb way to implement the restraint.
    '''
    dx = jnp.linalg.norm(x1-x2)
    return -k*(d - dx)**2 

def model(distance=1.5, kappa=2.0, box_size=4.0):
    '''
    This function defines the computational graph that numpyro will "trace" 
    behind the scenes and convert to XLA compiled code which the sampler will use.
    '''
    num_particles = 2
    num_dims = 3
    box_vectors = box_size*jnp.ones((num_particles, num_dims))
    x_prior = dist.Uniform(high=box_vectors).to_event(1)

    # two 3D particle coordinates array
    X = numpyro.sample("X", x_prior)

    # a very simple harmonic restraint
    numpyro.factor("harmonic_restraint", harmonic_restraint(X[0], X[1], distance, kappa))


numpyro.render_model(model, 
    render_distributions=True,
    filename="model.png"
)
kernel = NUTS(model, step_size=1e-5)
numpyro.set_host_device_count(4)
mcmc = MCMC(
    kernel,
    num_samples=500,
    num_warmup=1000,
    num_chains=4,
)
rng_key = random.PRNGKey(8)
mcmc.run(rng_key)
samples = mcmc.get_samples() # a dict of variables defined in model(), keyed by their declared name

Xs = samples["X"]
print(Xs.shape) # [num_chains*num_samples, batch_shape, event_shape]

sampled_distances = jnp.sqrt(jnp.sum((Xs[:,0] - Xs[:,1])**2, axis=-1))

# look at distribution of distances
plt.hist(sampled_distances)
plt.axvline(1.5, color='k') # the target distance
plt.title("interparticle distances")
plt.show()

# look at marginal distribution of coordinate values
fig, axs = plt.subplots(nrows=2, ncols=3)
for i in range(2):
    for j in range(3):
        axs[i,j].hist(Xs[:,i,j], density=True)
        axs[i,j].set_title(f"Particle {i}, Coordinate {j}")
plt.tight_layout()
plt.show()