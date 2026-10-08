# Quickstart

## Installation

Install the package from PyPI:

```bash
pip install gtotr
```

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

## Poisson / Identity example (PToTR)

Poisson-response tensor-on-tensor regression uses the Poisson family with an Identity
link. Use the `ptotr_cp` constructor, which returns a
[`PToTR_CP`][gtotr.models.gtotr_cp.PToTR_CP] model.

The specialized `cp_ao_poisson_identity` method is used by default. It requires
non-negative covariates.

```python
import numpy as np
import pyttb as ttb
import gtotr

rng = np.random.default_rng(1234)

X = ttb.tensor(rng.random(size=(3, 20)))  # non-negative covariates
Y = ttb.tensor(rng.poisson(lam=2.0, size=(4, 20)).astype(float))

model = gtotr.ptotr_cp(responses=Y, covariates=X)

results = model.fit(rank=2, printitn=0)
Yhat = results.predict()  # predicted means (mu = <X|B> under the identity link)
```

## Sparse Poisson / Identity example

`ptotr_cp` also accepts sparse `pyttb.sptensor` responses and covariates. Sparse
support is intentionally scoped to the PToTR model and sparse-aware fit methods,
currently `cp_ao_poisson_identity`.

The following example starts from small dense arrays only to construct sparse pyttb
objects. In a real application, you can construct the sparse tensors directly from
your nonzero indices and values.

```python
import numpy as np
import pyttb as ttb
import gtotr


def dense_to_sptensor(data):
    """Convert a dense NumPy array to pyttb.sptensor."""
    subs = np.argwhere(data != 0)
    vals = data[tuple(subs.T)].astype(float) if subs.size else np.array([], dtype=float)
    return ttb.sptensor(subs, vals, shape=data.shape)


rng = np.random.default_rng(1234)

# Sample mode is the last mode. X and Y must share this sample size.
n = 20

# Non-negative sparse covariates.
X_data = rng.random(size=(3, n))
X_data *= rng.random(size=(3, n)) < 0.35

# Sparse count responses.
Y_data = rng.poisson(lam=1.0, size=(4, n)).astype(float)
Y_data *= rng.random(size=(4, n)) < 0.40

X_sparse = dense_to_sptensor(X_data)
Y_sparse = dense_to_sptensor(Y_data)

model = gtotr.ptotr_cp(
    responses=Y_sparse,
    covariates=X_sparse,
)

# cp_ao_poisson_identity is the default method for PToTR.
results = model.fit(
    rank=2,
    maxiters=25,
    printitn=0,
)

# Predictions are currently returned as dense pyttb.tensor objects.
Yhat = results.predict(which="mean")
print(type(Yhat))
print(results.llf)
```

For sparse PToTR inputs, generic dense-only fit methods such as `cp_ao_glm` are not
advertised by `model.fit_methods()`.

```python
print(model.fit_methods())
# ['cp_ao_poisson_identity']
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
