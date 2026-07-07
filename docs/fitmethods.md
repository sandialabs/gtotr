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

## Notes

- `maxiters` controls the **outer** alternating-optimization iterations.
- `glm_method_options["maxiter"]` controls the **inner** GLM fitting iterations for
`cp_ao_glm`.