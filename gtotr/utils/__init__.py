# gtotr/utils/__init__.py
"""Utility functions for GToTR models and fit methods."""

from __future__ import annotations

from .likelihood import poisson_deviance_tensor, poisson_loglike_tensor
from .tensor_ops import contract_xb_cp

__all__ = [
    "contract_xb_cp",
    "poisson_deviance_tensor",
    "poisson_loglike_tensor",
]
