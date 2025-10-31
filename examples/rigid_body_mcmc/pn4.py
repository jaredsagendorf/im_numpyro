import numpyro.distributions as dist
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.scipy.special import erf

import math

# This implements a 4D projected normal distribution (e.g., a distribution over unit quaternions)
class ProjectedNormal4(dist.ProjectedNormal):
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        if self._validate_args:
            event_shape = value.shape[-1:]
            if event_shape != self.event_shape:
                raise ValueError(
                    f"Expected event shape {self.event_shape}, but got {event_shape}"
                )
            self._validate_sample(value)
        dim = int(self.concentration.shape[-1])
        if dim == 4:
            return _projected_normal_log_prob_4(self.concentration, value)
        raise NotImplementedError(
            f"ProjectedNormal4.log_prob() is not implemented for dim = {dim}. "
            "Consider using handlers.reparam with ProjectedNormalReparam."
        )

# This function was taken from pyro.dist codebase and converted to JAX
def _projected_normal_log_prob_4(concentration, value):
    def _dot(x: Array, y: Array) -> ArrayLike:
        return (x[..., None, :] @ y[..., None])[..., 0, 0]

    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a bivariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t*t
    r2 = _dot(concentration, concentration) - t2
    perp_part = -0.5*r2 - 1.5 * math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x^3/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = (2 + t^2)/(E^(t^2/2) Sqrt[2 Pi]) + (t (3 + t^2) (1 + Erf[t/Sqrt[2]]))/2
    # = (2 + t^2)/(E^(t^2/2) Sqrt[2 Pi]) + (t (3 + t^2) Erfc[-t/Sqrt[2]])/2
    para_part = jnp.log(
        (2 + t2) * jnp.exp(-0.5*t2) / (2 * math.pi) ** 0.5
        + t * (3 + t2) * (1 + erf(t * (0.5**0.5))) / 2
    )

    return para_part + perp_part