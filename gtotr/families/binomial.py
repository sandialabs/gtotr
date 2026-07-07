# gtotr/families/binomial.py
"""GToTR Binomial family."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from .base import Family
from .links import Logit


class Binomial(Family, sm.families.Binomial):
    """
    GTOTR Binomial family.

    Notes
    -----
    - Default link is Logit.
    - If eps > 0.0, predicted probabilities are clipped to [eps, 1-eps] to avoid log(0)
    issues in log-likelihood computation.
    """

    family_name = "binomial"

    def __init__(self, link=None, check_link: bool = False, eps: float = 1e-12):
        self.eps = float(eps)
        if link is None:
            link = Logit(eps=self.eps)
        super().__init__(link=link, check_link=check_link)

    def loglike(self, responses, mu, var_weights=1.0, freq_weights=1.0, _scale=1.0):
        """
        Compute Binomial log-likelihood with clipping for numerical stability.

        Assumes Bernoulli responses (0/1).
        """
        p = np.clip(mu, self.eps, 1.0 - self.eps)
        w = var_weights * freq_weights

        ll_obs = responses * np.log(p) + (1.0 - responses) * np.log(1.0 - p)
        return float(np.sum(w * ll_obs))

    def deviance(self, responses, mu, var_weights=1.0, freq_weights=1.0, _scale=1.0):
        """
        Compute Binomial deviance with clipping for numerical stability.

        Assumes Bernoulli responses (0/1).
        """
        p = np.clip(mu, self.eps, 1.0 - self.eps)
        w = var_weights * freq_weights

        dev_obs = np.zeros_like(p, dtype=float)

        mask1 = responses == 1
        dev_obs[mask1] = 2.0 * np.log(1.0 / p[mask1])

        mask0 = responses == 0
        dev_obs[mask0] = 2.0 * np.log(1.0 / (1.0 - p[mask0]))

        return float(np.sum(w * dev_obs))
