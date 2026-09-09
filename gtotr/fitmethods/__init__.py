# gtotr/fitmethods/__init__.py
"""GToTR fit methods and related utilities."""

from __future__ import annotations

from gtotr.models.gtotr_cp import GToTR_CP

from .cp_ao_gaussian_identity import CPAOGaussianIdentity
from .cp_ao_glm import CPAOGLM
from .cp_ao_poisson_identity import CPAOPoissonIdentity

GToTR_CP.register_fit_method(CPAOGLM)
GToTR_CP.register_fit_method(CPAOGaussianIdentity)
GToTR_CP.register_fit_method(CPAOPoissonIdentity)

__all__ = ["CPAOGLM", "CPAOGaussianIdentity", "CPAOPoissonIdentity"]
