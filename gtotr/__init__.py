"""gtotr: Generalized Tensor-on-Tensor Regression."""

from __future__ import annotations

from . import fitmethods as _fitmethods  # noqa: F401
from .models import GToTR_CP, GToTRBase, gtotr_cp
from .models.gtotr_base import ResultsBase

__all__ = [
    "GToTRBase",
    "GToTR_CP",
    "ResultsBase",
    "gtotr_cp",
]

__version__ = "0.1.0"
