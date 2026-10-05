# Contributing

Thank you for contributing to `gtotr`.

This document describes common development workflows for the package, including how to
build and serve the documentation locally.

## Development environment

From the repository root, create and activate a Python environment, then install the
package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

To include documentation dependencies as well, install the `doc` optional dependency
group:

```bash
pip install -e ".[dev,doc]"
```

The documentation stack uses MkDocs, Material for MkDocs, and mkdocstrings.

## Running tests

Run the test suite from the repository root:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=gtotr
```

## Building and serving the documentation

The documentation source files live in the `docs/` directory and are configured by
`mkdocs.yml`.

### Install documentation dependencies

If you have not already installed the documentation dependencies, run:

```bash
pip install -e ".[doc]"
```

or, for a development environment with both test/lint and documentation tools:

```bash
pip install -e ".[dev,doc]"
```

The quotes are recommended because some shells, such as `zsh`, interpret square
brackets specially.

### Serve the documentation locally

To build the documentation and serve it locally with live reload:

```bash
mkdocs serve
```

MkDocs will print a local URL, typically:

```text
http://127.0.0.1:8000/
```

Open that URL in a browser to view the HTML documentation. While `mkdocs serve` is
running, edits to files in `docs/`, docstrings, or `mkdocs.yml` should automatically
trigger a rebuild.

### Build static HTML documentation

To build the static HTML documentation without starting a web server:

```bash
mkdocs build
```

The generated HTML files are written to:

```text
site/
```

The main page is:

```text
site/index.html
```

For a stricter documentation build that treats warnings as errors, use:

```bash
mkdocs build --strict
```

This is useful before submitting changes that affect docs or public API docstrings.

### Clean and rebuild

To remove the generated documentation and rebuild from scratch:

```bash
rm -rf site
mkdocs build --strict
```

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force site
mkdocs build --strict
```

## Documentation style notes

- Put user-facing narrative documentation in `docs/*.md`.
- Keep API reference entries in `docs/api.md`.
- Public classes, methods, and functions should have NumPy-style docstrings.
- If a new page is added under `docs/`, also add it to the `nav:` section in
  `mkdocs.yml`.
- Prefer small, executable examples that use public APIs such as `gtotr_cp` and
  `ptotr_cp`.
- When documenting sparse PToTR examples, note that sparse input support is currently
  scoped to `PToTR_CP` and sparse-aware fit methods such as
  `cp_ao_poisson_identity`.

## Linting and formatting

If linting tools are installed through the `dev` dependency group, run:

```bash
ruff check .
```

To apply automatic fixes where possible:

```bash
ruff check . --fix
```

## Pull request checklist

Before submitting a pull request, consider running:

```bash
pytest
ruff check .
mkdocs build --strict
```

Also confirm that any new public API has corresponding documentation and tests.
