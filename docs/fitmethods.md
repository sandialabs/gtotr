# Fit Methods

Fit methods are selected with:

```python
results = model.fit(method="...")
```

## `cp_ao_glm`

`cp_ao_glm` uses CP alternating optimization with GLM-based inner updates.

```python
results = model.fit(
    method="cp_ao_glm",
    rank=2,
)
```

### Inner GLM method selection

The inner GLM fitting method can be selected with `glm_method`.

```python
results = model.fit(
    method="cp_ao_glm",
    rank=2,
    glm_method="irls",
    glm_method_options={
        "maxiter": 50,
        "tol": 1e-8,
        "disp": 0,
    },
)
```

Currently supported `glm_method` values include:

- `irls`
- `newton`
- `nm`
- `bfgs`
- `lbfgs`
- `powell`
- `cg`
- `ncg`

## `cp_ao_gaussian_identity`

This is a specialized CP alternating-optimization method for the Gaussian family with
identity link.

It is typically faster than the generic GLM-based method when applicable.

```python
results = model.fit(
    method="cp_ao_gaussian_identity",
    rank=2,
)
```

## `cp_ao_poisson_identity`

This is a specialized CP alternating-optimization method for the Poisson family with
identity link (mean `mu = <X|B>`), ported from the standalone `ptotr` package. It uses
multiplicative (majorize-minimize) updates and is the **default** fit method for
Poisson + Identity models (the generic `cp_ao_glm` remains available explicitly).

```python
results = model.fit(
    method="cp_ao_poisson_identity",
    rank=2,
)
```

The multiplicative updates carry a convergence guarantee that assumes a non-negative
initial coefficient `B` and non-negative covariates `X`. By default the solver
validates both (`check_inputs=True`) and raises `ValueError` on violation. For large or
sparse inputs you can skip the checks (which voids the guarantee):

```python
results = model.fit(
    method="cp_ao_poisson_identity",
    rank=2,
    check_inputs=False,
)
```

This method is described in Llosa-Vite, C., & Dunlavy, D. M. (2026). *Poisson-response
Tensor-on-Tensor Regression and Applications.* arXiv:2604.07377 [stat.ME].
[https://arxiv.org/abs/2604.07377](https://arxiv.org/abs/2604.07377)

## Notes

- `maxiters` controls the **outer** alternating-optimization iterations.
- `glm_method_options["maxiter"]` controls the **inner** GLM fitting iterations for
`cp_ao_glm`.