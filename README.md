[![testing](https://github.com/sandialabs/gtotr/actions/workflows/run-tests.yml/badge.svg)](https://github.com/sandialabs/gtotr/actions/workflows/run-tests.yml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type%20checked-mypy-039dfc)](https://mypy-lang.org)

# Generalized Tensor-on-Tensor Regression(GToTR)

`gtotr` is a Python package for generalized tensor-on-tensor regression, 
extending [`statsmodels.GLM`](https://www.statsmodels.org/stable/glm.html) to cases 
with tensor response and tensor covariates. The model parameters in `gtotr` are 
estimated using maximum likelihood estimation associated with a low-rank model of the 
parameter tensor. Currently, `gtotr` provides estimators using low-rank Canonical 
Polyadic (CP) models.

## Getting Started

### Installing 

```bash
$ python -m pip install .
```

Test the install:

```bash
$ python
>>> import gtotr
>>> help(gtotr)
```

## Documentation

- Documentation: [gtotr.readthedocs.io](https://gtotr.readthedocs.io)
- Tutorials: [Jupyter notebook tutorials](tutorials/)
2. Open `gtotr-01-getting-started.ipynb`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on participating as a developer.
 
