# Families and Links

`gtotr` uses a GLM-like family/link design.

## Families

Currently supported:

- Gaussian
- Binomial
- Poisson

## Links

Currently supported:

- Identity
- Log
- Logit

## String-based construction

Families and links can be specified using strings:

```python
model = gtotr.gtotr_cp(
    responses=Y,
    covariates=X,
    family="binomial",
    link="logit",
)
```

## Explicit family/link objects

You can also pass explicit `gtotr` family and link objects to customize settings such as clipping parameters:

```python
from gtotr.families import Binomial
from gtotr.families.links import Logit

family = Binomial(link=Logit(eps=1e-8), eps=1e-8)
```

## Numerical stability

For numerically sensitive families:

- Binomial clips predicted probabilities away from 0 and 1
- Poisson clips predicted means away from 0

These clipping controls are available through the family/link objects.