# gtotr/fitmethods/cp_ao_gaussian_identity.py
"""CP alternating optimization for Gaussian family with Identity link."""

from __future__ import annotations

from gtotr.fitmethods.base import FitMethodBase
from gtotr.solvers.cp_ao_backends.gaussian_identity import CPAOGaussianIdentityBackend
from gtotr.solvers.cp_ao_solver import cp_ao_solver


class CPAOGaussianIdentity(FitMethodBase):
    """Fast CP alternating optimization for Gaussian family with Identity link."""

    method = "cp_ao_gaussian_identity"
    description = (
        "Fast CP alternating optimization for Gaussian family with Identity link."
    )

    @classmethod
    def supports(cls, model) -> bool:
        """Check if the model is compatible with this fit method."""
        fam = model.family
        return (fam.family_name == "gaussian") and (fam.link.link_name == "identity")

    def fit(self, model, **fit_options):
        """Fit using CP alternating optimization with Gaussian-Identity updates."""
        backend = CPAOGaussianIdentityBackend()
        return cp_ao_solver(model, backend, **fit_options)
