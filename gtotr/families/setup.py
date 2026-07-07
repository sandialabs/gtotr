# gtotr/families/setup.py
"""GToTR families, links, and related utilities."""

from __future__ import annotations

import statsmodels.api as sm

from .base import Family, Link
from .binomial import Binomial
from .gaussian import Gaussian
from .links import Identity, Log, Logit
from .poisson import Poisson

FamilySpec = str | Family | None
LinkSpec = str | Link | None


def family_setup(family: FamilySpec = None, link: LinkSpec = None) -> Family:
    """
    Construct a GToTR Family and (optionally) attach/override a GToTR Link.

    Supported inputs:
      - family: None | str | gtotr.families.Family
      - link:   None | str | gtotr.families.links.Link

    Policy:
      - Raw statsmodels families/links are intentionally not accepted (even though
        GToTR families/links subclass statsmodels under the hood).
      - If `link is None`, the family's own default link is used.
      - If `link` is provided (string or Link), it overrides the family's link.
    """
    fam = _get_family(family)

    if link is not None:
        fam.set_link(_get_link(link))

    return fam


def _get_family(family: FamilySpec) -> Family:
    """Construct a GToTR Family.

    Note: this is only used for string->Family mapping. If the user passes a Family
    object directly, we assume it's already a GToTR Family and return it as-is (after a
    type check). The string->Family mapping supports "gaussian", "binomial"/"bernoulli",
    and "poisson", which are the most common families we have. If the user wants to use
    a different family, they can create a GToTR Family object themselves and pass it in
    directly.
    """
    if family is None:
        return Gaussian()  # let Gaussian choose its own default link

    # Explicitly reject raw statsmodels families for a clearer error.
    if isinstance(family, sm.families.Family) and not isinstance(family, Family):
        raise TypeError(
            "Raw statsmodels Family objects are not supported. "
            "Pass a GToTR family (e.g., gtotr.families.Gaussian()) "
            "or a string like 'gaussian'."
        )

    if isinstance(family, Family):
        return family

    if isinstance(family, str):
        key = family.strip().lower()
        if key == "gaussian":
            return Gaussian()
        if key in ("binomial", "bernoulli"):
            return Binomial()
        if key == "poisson":
            return Poisson()
        raise ValueError(f"Unknown family '{family}'.")

    raise TypeError(
        f"family must be None, a string, or a GToTR Family. Got {type(family)!r}."
    )


def _get_link(link: LinkSpec) -> Link:
    """Construct a GToTR Link.

    Note: this is only used for string->Link mapping. If the user passes a Link object
    directly, we assume it's already a GToTR Link and return it as-is (after a type
    check). The string->Link mapping supports "identity", "log", and "logit", which
    are the most common links for the families we have. If the user wants to use a
    different link, they can create a GToTR Link object themselves and pass it in
    directly.
    """
    if link is None:
        return Identity()

    # Explicitly reject raw statsmodels links for a clearer error.
    if isinstance(link, sm.families.links.Link) and not isinstance(link, Link):
        raise TypeError(
            "Raw statsmodels Link objects are not supported. "
            "Pass a GToTR link (e.g., gtotr.families.links.Logit()) "
            "or a string like 'logit'."
        )

    if isinstance(link, Link):
        return link

    if isinstance(link, str):
        key = link.strip().lower()
        if key == "identity":
            return Identity()
        if key == "log":
            return Log()
        if key == "logit":
            return Logit()
        raise ValueError(f"Unknown link '{link}'.")

    raise TypeError(
        f"link must be None, a string, or a GToTR Link. Got {type(link)!r}."
    )
