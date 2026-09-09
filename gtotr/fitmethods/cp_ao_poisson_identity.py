# gtotr/fitmethods/cp_ao_poisson_identity.py
"""CP alternating optimization for Poisson family with Identity link."""

from __future__ import annotations

from gtotr.fitmethods.base import FitMethodBase
from gtotr.solvers.cp_ao_backends.poisson_identity import cp_ao_poisson_identity_solve


class CPAOPoissonIdentity(FitMethodBase):
    """Multiplicative CP alternating optimization for Poisson + Identity link.

    Fits a CP-decomposed regression coefficient for a Poisson-response
    tensor-on-tensor regression under the Identity link (mean ``mu = <X|B>``) using
    a multiplicative (majorize-minimize) alternating-optimization scheme.

    This method applies to models with the Poisson family and Identity link
    (``supports()`` gates on ``family_name == "poisson"`` and
    ``link.link_name == "identity"``). It is the **default** fit method for such models
    (the generic ``cp_ao_glm`` remains reachable via ``method="cp_ao_glm"``).

    The multiplicative updates come with a convergence guarantee that assumes a
    non-negative initial ``B`` and non-negative covariates ``X``. The underlying solver
    validates both by default (``check_inputs=True``) and raises ``ValueError`` on
    violation; pass ``check_inputs=False`` to skip the checks for large or sparse
    inputs (which voids the guarantee).

    See Also
    --------
    gtotr.solvers.cp_ao_backends.poisson_identity.cp_ao_poisson_identity_solve :
        The underlying self-contained solver.
    gtotr.models.gtotr_cp.ptotr_cp : Convenience constructor for Poisson + Identity
        models.

    References
    ----------
    Llosa-Vite, C., & Dunlavy, D. M. (2026). *Poisson-response Tensor-on-Tensor
    Regression and Applications.* arXiv:2604.07377 [stat.ME].
    [https://arxiv.org/abs/2604.07377](https://arxiv.org/abs/2604.07377)
    """

    method = "cp_ao_poisson_identity"
    description = (
        "Multiplicative CP alternating optimization for Poisson family with Identity "
        "link."
    )

    @classmethod
    def supports(cls, model) -> bool:
        """Check if the model is compatible with this fit method."""
        fam = model.family
        return (fam.family_name == "poisson") and (fam.link.link_name == "identity")

    def fit(self, model, **fit_options):
        """Fit using the self-contained Poisson-Identity multiplicative solver."""
        return cp_ao_poisson_identity_solve(model, **fit_options)
