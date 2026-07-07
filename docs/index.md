# gtotr

**gtotr** is a Python package for generalized tensor-on-tensor regression.

It currently focuses on:

- CP-based generalized tensor regression models
- GLM-like family and link support
- alternating-optimization fit methods
- a statsmodels-inspired interface

## Highlights

- Tensor-valued responses and covariates using `pyttb`
- `GToTR_CP` model class
- Gaussian, Binomial, and Poisson family support
- Identity, Log, and Logit links
- generic and specialized CP alternating-optimization fit methods

## Example

```python
import numpy as np
import pyttb as ttb
import gtotr

rng = np.random.default_rng(1234)

Y = ttb.tensor(rng.normal(size=(4, 5, 20)))
X = ttb.tensor(rng.normal(size=(3, 5, 20)))

model = gtotr.gtotr_cp(
    responses=Y,
    covariates=X,
    family="gaussian",
    link="identity",
)

results = model.fit(rank=2)
print(results.summary())
```
