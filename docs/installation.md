# Installation

## Basic install

Install the package from source:

```bash
pip install -e .
```

## Install with documentation dependencies

To build and preview the documentation locally:

```bash
pip install -e .[docs]
```

## Local docs preview

Run:

```bash
mkdocs serve
```

Then open the local URL shown in the terminal.

## Development notes

Depending on your platform and environment, you may also need the package dependencies
used by:

- `numpy`
- `scipy`
- `statsmodels`
- `pyttb`