import typing

import jax
import jax.numpy as jnp
from flax import nnx


class ModelProtocol(typing.Protocol):
    def __call__(self, noisy: jax.Array, time: jax.Array) -> jax.Array: ...


def loss(model: ModelProtocol, data: jax.Array, rngs: nnx.Rngs) -> jax.Array:
    time = jax.nn.sigmoid(rngs.normal((data.shape[0],)))
    time_reshaped = time.reshape((-1,) + (1,) * (data.ndim - 1))
    noise = rngs.normal(data.shape)
    noisy = (1 - time_reshaped) * noise + time_reshaped * data

    return jnp.mean((model(noisy, time) - (data - noise)) ** 2)


def sample(model: ModelProtocol, noise: jax.Array, *, steps: int = 20) -> jax.Array:
    noisy = noise

    for step in range(steps):
        time = jnp.full((noisy.shape[0],), step / steps)
        noisy = noisy + model(noisy, time) / steps

    return noisy
