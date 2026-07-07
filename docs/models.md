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

# Base classes

[GToTRBase][gtotr.models.gtotr_base.GToTRBase] defines the shared interface for GToTR
model implementations. It is intended for developers implementing new model types and
should not be instantiated directly.
