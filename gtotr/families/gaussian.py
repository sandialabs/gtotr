# gtotr/families/gaussian.py
"""GToTR Gaussian family."""

from __future__ import annotations

import statsmodels.api as sm

from .base import Family
from .links import Identity


class Gaussian(Family, sm.families.Gaussian):
    """
    GToTR Gaussian family.

    Notes
    -----
    - Default link is Identity.
    """

    family_name = "gaussian"

    def __init__(self, link=None, check_link: bool = False):
        if link is None:
            link = Identity()
        super().__init__(link=link, check_link=check_link)
