# gtotr/fitmethods/cp_ao_glm.py
"""CP alternating optimization using statsmodels-GLM updates."""

from __future__ import annotations

from gtotr.fitmethods.base import FitMethodBase
from gtotr.models.gtotr_cp import GToTR_CP
from gtotr.solvers.cp_ao_backends.glm import CPAOGLMBackend
from gtotr.solvers.cp_ao_solver import cp_ao_solver


class CPAOGLM(FitMethodBase):
    """CP alternating optimization using statsmodels-GLM updates."""

    method = "cp_ao_glm"
    description = "CP alternating optimization using statsmodels-GLM updates."

    @classmethod
    def supports(cls, model) -> bool:
        """Check if the model is compatible with this fit method."""
        return isinstance(model, GToTR_CP)

    def fit(self, model, **fit_options):
        """Fit the model using CP alternating optimization with GLM updates."""
        backend = CPAOGLMBackend()
        return cp_ao_solver(model, backend, **fit_options)
