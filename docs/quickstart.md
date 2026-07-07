# Quickstart

## Gaussian / Identity example

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

results = model.fit(rank=2, printitn=1)
Yhat = results.predict()
```

## Choosing a fit method

```python
results = model.fit(
    method="cp_ao_glm",
    rank=2,
)
```

## Choosing the inner GLM method

The `cp_ao_glm` fit method supports alternate statsmodels GLM fitting methods:

```python
results = model.fit(
    method="cp_ao_glm",
    rank=2,
    glm_method="newton",
    glm_method_options={
        "maxiter": 10,
        "tol": 1e-6,
        "disp": 0,
    },
)
```

## Inspecting results

```python
print(results.llf)
print(results.method)
print(results.fit_info["glm_method"])
coef = results.coef_
```