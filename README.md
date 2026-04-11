# flow-matching

An implementation of Flow Matching from [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2022).

## Installation

```sh
uv add git+https://github.com/flawley/flow-matching
```

## Usage

Train.

```python
import jax
import optax
from flax import nnx

import flow_matching


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, data: jax.Array, rngs: nnx.Rngs) -> jax.Array:
    loss, grads = nnx.value_and_grad(flow_matching.loss)(model, data, rngs)
    optimizer.update(model, grads)

    return loss

model = ...  # An nnx.Module that takes xt, t and returns v(xt, t).
optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
data = jax.numpy.array(...)  # x_1 ~ p_data.
rngs = nnx.Rngs(0)
loss = train_step(model, optimizer, data, rngs)
```

Sample.

```python
noise = rngs.normal(...)  # x_0 ~ N(0, 1).
data = flow_matching.sample(model, noise)  # x_1 ~ p_data.
```

