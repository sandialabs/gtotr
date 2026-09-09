# gtotr/families/poisson.py
"""GToTR Poisson family."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.special import gammaln

from .base import Family
from .links import Log


class Poisson(Family, sm.families.Poisson):
    """
    GToTR Poisson family.

    Notes
    -----
    - Default link is Log.
    - Identity is a blessed (supported) non-default link for Poisson: with the Identity
      link the mean is modeled directly as ``mu = <X|B>``. This is the link used by the
      ``cp_ao_poisson_identity`` fit method (ported from ptotr). Select it with
      ``link="identity"`` or via the ``ptotr_cp`` convenience constructor.
    - `eps` is an optional clipping threshold used to avoid log(0) in log-likelihood
      calculations (if you override `loglike`).

    References
    ----------
    Llosa-Vite, C., & Dunlavy, D. M. (2026). *Poisson-response Tensor-on-Tensor
    Regression and Applications.* arXiv:2604.07377 [stat.ME].
    [https://arxiv.org/abs/2604.07377](https://arxiv.org/abs/2604.07377)
    """

    family_name = "poisson"

    def __init__(self, link=None, check_link: bool = False, eps: float = 1e-12):
        self.eps = float(eps)
        if link is None:
            link = Log()
        super().__init__(link=link, check_link=check_link)

    def loglike(self, responses, mu, var_weights=1.0, freq_weights=1.0, _scale=1.0):
        """Compute Poisson log-likelihood with clipping for numerical stability."""
        lam = np.clip(mu, self.eps, np.inf)
        w = var_weights * freq_weights

        ll_obs = responses * np.log(lam) - lam - gammaln(responses + 1.0)
        return float(np.sum(w * ll_obs))

    def deviance(self, responses, mu, var_weights=1.0, freq_weights=1.0, _scale=1.0):
        """Compute Poisson deviance with clipping for numerical stability."""
        lam = np.clip(mu, self.eps, np.inf)
        w = var_weights * freq_weights

        term = np.zeros_like(lam, dtype=float)
        mask = responses > 0
        term[mask] = responses[mask] * np.log(responses[mask] / lam[mask])

        dev_obs = 2.0 * (term - (responses - lam))
        return float(np.sum(w * dev_obs))
