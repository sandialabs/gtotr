# Generalized Tensor-on-Tensor Regression(GToTR) Contributor Guide

## Issues
If you are looking to get started or want to propose a change please start by checking
current or filing a new [issue](https://github.com/sandialabs/gtotr/issues).

## Working on GToTR locally
1. Clone the project and enter the directory
   ```
   git clone https://github.com/sandialabs/gtotr.git
   # OR git clone git@github.com:sandialabs/gtotr.git
   ```

1. Setup your desired python environment as appropriate

1. Install GToTR and dependencies

   Most changes only require dev options:
   ```commandline
   cd gtotr
   python -m pip install -e ".[dev]"
   ```

   If you are adding tutorials or making changes to the docs, you will also need the
   doc dependencies:
   ```commandline
   cd gtotr
   python -m pip install -e ".[dev,doc]"
   ```

1. Checkout a branch and make your changes
    ```
    git checkout -b my-new-feature-branch
    ```

1. Formatters and linting (These are checked in the full test suite as well)
   1. Run autoformatters and linting from root of project (they will change your code)
      ```commandline
      ruff check . --fix
      ruff format
      ```
      1. Ruff's `--fix` won't necessarily address everything and may point out issues that need manual attention
      1. [We](./.pre-commit-config.yaml) optionally support [pre-commit hooks](https://pre-commit.com/) for this
         1. Alternatively, you can run `pre-commit run --all-files` from the command line if you don't want to install the hooks.
   1. Check typing
      ```commandline
      mypy gtotr/
      ```
      1. Not included in our pre-commit hooks because of slow runtime.
   1. Check spelling
      ```commandline
      codespell
      ```
      1. This is also included in the optional pre-commit hooks.

1. Run tests (at desired fidelity)
   1. Tests and doctests
        ```commandline
        pytest .
        ```
   1. Just tests
        ```commandline
        pytest tests
        ```
   1. With coverage
        ```commandline
        pytest . --cov=gtotr --cov-report=term-missing
        ```

### Adding tutorials

1. Follow general setup from above
   1. Checkout a branch to make your changes
   1. Install from source

1. Create a new Jupyter notebook in [./tutorials](./tutorials)
   1. Our current convention is to prefix the filename with `gtotr-##-` and use lower case and hyphens

1. Rebuild the docs, review locally, and iterate on changes until ready for review

1. Strip all output and metadata from the notebook by running the following from the top level directory:
   ```
   nbstripout --extra-keys "metadata.language_info metadata.vscode metadata.kernelspec" tutorials/*.ipynb
   ``` 
