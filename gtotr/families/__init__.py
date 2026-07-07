# gtotr/families/__init__.py
"""GToTR families and related utilities."""

from __future__ import annotations

from .base import Family, Link
from .binomial import Binomial
from .gaussian import Gaussian
from .links import Identity, Log, Logit
from .poisson import Poisson
from .setup import family_setup

__all__ = [  # noqa: RUF022
    # base types
    "Family",
    "Link",
    # links
    "Identity",
    "Log",
    "Logit",
    # families
    "Gaussian",
    "Binomial",
    "Poisson",
    # factory
    "family_setup",
]
