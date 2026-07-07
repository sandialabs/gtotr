# gtotr/families/base.py
"""Base classes for GToTR families and links."""

from __future__ import annotations

import statsmodels.api as sm


class Link(sm.families.links.Link):
    """Base class for GToTR links, extending statsmodels Link."""

    link_name: str = "base"


class Family(sm.families.Family):
    """Base class for GToTR families, extending statsmodels Family."""

    family_name: str = "base"

    def set_link(self, link) -> None:
        """Set the link function for this family, with type checks."""
        # Special-case raw statsmodels links for a clearer error
        if isinstance(link, sm.families.links.Link) and not isinstance(link, Link):
            raise TypeError(
                "Raw statsmodels Link objects are not supported. "
                "Pass a GToTR link (e.g., gtotr.families.links.Logit()) "
                "or a string like 'logit'."
            )

        if not isinstance(link, Link):
            raise TypeError(f"link must be a GToTR Link, got {type(link)!r}")

        self._setlink(link)
