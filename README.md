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


model = ...  # An nnx.Module that takes x_t, t and returns v(x_t, t).
data = ...  # x_1 ~ p_data.
rngs = nnx.Rngs(0)
loss, grads = nnx.value_and_grad(flow_matching.loss)(model, data, rngs)
```

Sample.

```python
noise = rngs.normal(...)  # x_0 ~ N(0, 1).
data = flow_matching.sample(model, noise)  # x_1 ~ p_data.
```

