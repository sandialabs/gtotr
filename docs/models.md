# Models

## CP model

The primary user-facing model constructor is
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

## PToTR alias

Poisson-response Tensor-on-Tensor Regression (PToTR) is a special case of `gtotr`
using the Poisson family with an Identity link (mean `mu = <X|B>`). The
[`ptotr_cp`][gtotr.models.gtotr_cp.ptotr_cp] constructor preserves that identity:

```python
model = gtotr.ptotr_cp(
    responses=Y,
    covariates=X,
)
```

This is identical to `gtotr.gtotr_cp(responses=Y, covariates=X, family="poisson",
link="identity")`. Passing `family` or `link` to `ptotr_cp` raises `TypeError`, since
they are preset. Poisson + Identity models fit with the specialized
[`cp_ao_poisson_identity`](fitmethods.md#cp_ao_poisson_identity) method by default.

The method is described in Llosa-Vite, C., & Dunlavy, D. M. (2026). *Poisson-response
Tensor-on-Tensor Regression and Applications.* arXiv:2604.07377 [stat.ME].
[https://arxiv.org/abs/2604.07377](https://arxiv.org/abs/2604.07377)

# Base classes

[GToTRBase][gtotr.models.gtotr_base.GToTRBase] defines the shared interface for GToTR
model implementations. It is intended for developers implementing new model types and
should not be instantiated directly.
