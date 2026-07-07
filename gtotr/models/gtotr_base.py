# gtotr/models/gtotr_base.py
"""GToTR base classes and utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import pyttb as ttb

    from gtotr.fitmethods.base import FitMethodBase

from gtotr.families import Family, Link, family_setup


class GToTRBase(ABC):
    """
    Abstract base class for Generalized Tensor-on-Tensor Regression models.

    This class defines the common model interface used by derived model classes
    such as [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP]. It is not intended to
    be instantiated directly.
    """

    ResultsClass = None  # derived models may override

    def __init__(
        self,
        *,
        responses: ttb.tensor,
        covariates: ttb.tensor,
        family: str | Family | None = None,
        link: str | Link | None = None,
        **model_options: Any,
    ):
        """
        Initialize model with responses and covariates.

        Parameters
        ----------
        responses : pyttb.tensor
            Tensor of response variables, with sample size along the last mode.

        covariates : pyttb.tensor
            Tensor of covariates, with sample size along the last mode.

        family : str or gtotr.families.Family, optional
            Family specification. May be a string or a GToTR family object. Raw
            statsmodels family objects are rejected by
            [`family_setup`][gtotr.families.setup.family_setup].

        link : str or gtotr.families.Link, optional
            Link specification. May be a string or a GToTR link object. Raw
            statsmodels link objects are rejected by
            [`family_setup`][gtotr.families.setup.family_setup].

        **model_options : Any
            Additional model-specific configuration.
        """
        self.responses = responses
        self.covariates = covariates
        self.family: Family = family_setup(family=family, link=link)
        self.model_options = dict(model_options)

    @property
    def link(self) -> Link:
        """Convenience: the link is always stored on the family."""
        return self.family.link

    # ----- Core model API (to be implemented by derived classes) -----

    @abstractmethod
    def predict(
        self, params: Any, *, covariates: ttb.tensor | None = None, which: str = "mean"
    ) -> Any:
        """
        Predict responses from model parameters and covariates.

        which="linear": return eta
        which="mean":   return mu (inverse link of eta)
        """
        ...

    @abstractmethod
    def loglike(self, params: Any) -> float:
        """Calculate log-likelihood given parameters."""
        ...

    @abstractmethod
    def contract_xb(
        self, coef: Any, *, covariates: ttb.tensor | None = None, normtype: int = 2
    ) -> Any:
        """Compute the linear predictor eta = <covariates | coef>."""
        ...

    def get_coef(self, params: Any):
        """
        Return the coefficient tensor object (ktensor/ttensor/...) from params.

        Default convention: params is a dict with key "coef".
        Override in derived models if needed.
        """
        return params["coef"]

    # ----- Fit protocol -----

    _fit_methods: ClassVar[dict[str, type[FitMethodBase]]] = {}

    @classmethod
    def register_fit_method(cls, method_cls) -> None:
        """Register a fit method class for this model."""
        cls._fit_methods[method_cls.method] = method_cls

    def fit_methods(self) -> tuple[str, ...]:
        """Return names of fit methods that support this model."""
        return tuple(
            name for name, mcls in self._fit_methods.items() if mcls.supports(self)
        )

    def get_default_method(self) -> str:
        """Return the default fit method for this model."""
        methods = self.fit_methods()
        if not methods:
            raise RuntimeError("No fit methods are registered for this model.")
        return methods[0]

    def fit(self, method: str | None = None, **fit_options: Any):
        """
        Fit the model using the specified fit method.

        Parameters
        ----------
        method : str, optional
            Name of the fit method to use. If omitted, the model's default fit
            method is selected.

        **fit_options : Any
            Options passed to the selected fit method. For CP alternating
            optimization methods, common options include ``rank``, ``maxiters``,
            ``tolerance``, ``normtype``, ``printitn``, ``trace``, and ``seed``.

        Returns
        -------
        ResultsBase
            Fitted model results containing fitted parameters, diagnostics, and
            prediction methods.
        """
        method = self.get_default_method() if method is None else method

        if method not in self._fit_methods:
            raise ValueError(
                f"Unknown method '{method}'. Available: {self.fit_methods()}"
            )

        mcls = self._fit_methods[method]
        if not mcls.supports(self):
            raise ValueError(
                f"Method '{method}' not supported for this model. Available: ",
                f"{self.fit_methods()}",
            )

        fitter = mcls()
        params, fit_info = fitter.fit(self, **fit_options)

        Results = self.ResultsClass or ResultsBase
        return Results(model=self, params=params, fit_info=fit_info)


class ResultsBase:
    """Base class for results of fitted GToTR models."""

    def __init__(
        self, *, model: GToTRBase, params: Any, fit_info: dict[str, Any] | None = None
    ):
        self.model = model
        self.params = params
        self.fit_info = {} if fit_info is None else dict(fit_info)

    def predict(self, covariates: ttb.tensor | None = None, which="mean"):
        """Predict responses from the fitted model."""
        if which not in ("mean", "linear"):
            raise ValueError(
                f"Invalid value for 'which': {which}. Must be 'mean' or 'linear'."
            )
        covariates = covariates if covariates is not None else self.model.covariates
        return self.model.predict(self.params, covariates=covariates, which=which)

    @property
    def converged(self) -> bool | None:
        """Whether the fitting procedure converged, if provided in fit_info."""
        return self.fit_info.get("converged")

    @property
    def method(self) -> str | None:
        """Fitting method used, if provided in fit_info."""
        return self.fit_info.get("method")

    @property
    def coef_(self) -> Any:
        """GToTR coefficients."""
        return self.model.get_coef(self.params)

    @property
    def llf(self) -> float:
        """
        Log-likelihood at the fitted parameters.

        Prefer cached value if provided in fit_info (keys: 'llf' or 'loglike').
        """
        if "llf" in self.fit_info:
            return float(self.fit_info["llf"])
        if "loglike" in self.fit_info:
            return float(self.fit_info["loglike"])
        return float(self.model.loglike(self.params))

    def summary(self) -> str:
        """Return a string summary of the fitted model."""
        return f"{type(self.model).__name__} Results: llf={self.llf}"
