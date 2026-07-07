# gtotr/fitmethods/base.py
"""Base class for GToTR fit methods."""

from __future__ import annotations

from typing import Any


class FitMethodBase:
    """Base class for GToTR fit methods."""

    method: str = "base"
    description: str = ""

    @classmethod
    def supports(cls, _model) -> bool:
        """Check if the model is compatible with this fit method."""
        return True

    def fit(self, _model, **fit_options) -> tuple[Any, dict[str, Any]]:
        """Fit the model and return parameters and fit info."""
        raise NotImplementedError
