# gtotr/families/_links_impl.py
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from .base import Link


class Identity(Link, sm.families.links.Identity):
    link_name = "identity"

    def __init__(self) -> None:
        super().__init__()


class Log(Link, sm.families.links.Log):
    link_name = "log"

    def __init__(self) -> None:
        super().__init__()


class Logit(Link, sm.families.links.Logit):
    link_name = "logit"

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    # This helps prevent warnings for 1 / (1 + exp(-z)) with large negative z values
    def inverse(self, z):
        p = expit(z)
        if self.eps > 0.0:
            p = np.clip(p, self.eps, 1.0 - self.eps)
        return p
