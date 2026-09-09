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

### Identity link for Poisson

Identity is a blessed (supported) non-default link for the Poisson family. With the
Identity link the mean is modeled directly as `mu = <X|B>` (rather than
`mu = exp(<X|B>)` under the default Log link). This is the combination used by the
specialized [`cp_ao_poisson_identity`](fitmethods.md#cp_ao_poisson_identity) fit method
and the [`ptotr_cp`](models.md#ptotr-alias) constructor. The Poisson default link
remains Log.

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