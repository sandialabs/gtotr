# Models

## CP model

The primary general-purpose model constructor is
[`gtotr_cp`][gtotr.models.gtotr_cp.gtotr_cp].

```python
model = gtotr.gtotr_cp(
    responses=Y,
    covariates=X,
    family="poisson",
    link="log",
)
```

This returns a [GToTR_CP][gtotr.models.gtotr_cp.GToTR_CP] instance.

The general `GToTR_CP` API is dense-oriented. Sparse tensor support is provided only by
model classes and fit methods that explicitly document sparse support.

## PToTR model

Poisson-response Tensor-on-Tensor Regression (PToTR) is the Poisson + Identity-link
specialization of GToTR, with mean

\[
\mu = \langle X \mid B \rangle .
\]

Use the [`ptotr_cp`][gtotr.models.gtotr_cp.ptotr_cp] constructor:

```python
model = gtotr.ptotr_cp(
    responses=Y,
    covariates=X,
)
```

This returns a [PToTR_CP][gtotr.models.gtotr_cp.PToTR_CP] instance. Passing `family` or
`link` to `ptotr_cp` raises `TypeError`, since these are preset to `family="poisson"`
and `link="identity"`.

`PToTR_CP` accepts either dense `pyttb.tensor` or sparse `pyttb.sptensor` responses and
covariates. Sparse tensor support is scoped to sparse-aware fit methods, currently
[`cp_ao_poisson_identity`](fitmethods.md#cp_ao_poisson_identity), which is the default
fit method for PToTR models.

## Sparse PToTR inputs

Sparse inputs can be passed directly to `ptotr_cp`:

```python
import numpy as np
import pyttb as ttb
import gtotr


def dense_to_sptensor(data):
    subs = np.argwhere(data != 0)
    vals = data[tuple(subs.T)].astype(float) if subs.size else np.array([], dtype=float)
    return ttb.sptensor(subs, vals, shape=data.shape)


rng = np.random.default_rng(0)
n = 20

# Last mode is the sample mode.
X_data = rng.random(size=(3, n))
X_data *= rng.random(size=(3, n)) < 0.35

Y_data = rng.poisson(lam=1.0, size=(4, n)).astype(float)
Y_data *= rng.random(size=(4, n)) < 0.40

X = dense_to_sptensor(X_data)
Y = dense_to_sptensor(Y_data)

model = gtotr.ptotr_cp(
    responses=Y,
    covariates=X,
)

results = model.fit(rank=2)
Yhat = results.predict(which="mean")
```

Notes:

- `responses` and `covariates` may be `pyttb.tensor` or `pyttb.sptensor`.
- Covariates must be non-negative for the default `cp_ao_poisson_identity` solver.
- Predictions are currently returned as dense `pyttb.tensor` objects.
- Dense-only fit methods are not listed by `model.fit_methods()` when sparse inputs are
  used.

The method is described in Llosa-Vite, C., & Dunlavy, D. M. (2026). *Poisson-response
Tensor-on-Tensor Regression and Applications.* arXiv:2604.07377 [stat.ME].
[https://arxiv.org/abs/2604.07377](https://arxiv.org/abs/2604.07377)

# Base classes

[GToTRBase][gtotr.models.gtotr_base.GToTRBase] defines the shared interface for GToTR
model implementations. It is intended for developers implementing new model types and
should not be instantiated directly.
