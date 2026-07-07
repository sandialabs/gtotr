# gtotr/models/gtotr_cp.py
"""GToTR with CP decomposition of regression coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyttb as ttb

if TYPE_CHECKING:
    from gtotr.families import Family, Link
from gtotr.utils import contract_xb_cp

from .gtotr_base import GToTRBase


class GToTR_CP(GToTRBase):
    """Model class for GToTR with CP decomposition."""

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
        Initialize a CP-based GToTR model.

        Most users should construct this model using
        [`gtotr_cp`][gtotr.models.gtotr_cp.gtotr_cp], which provides the same
        model with a function-style interface.
        """
        if responses.shape[-1] != covariates.shape[-1]:
            raise ValueError(
                "responses and covariates must share the same sample mode size"
            )
        super().__init__(
            responses=responses,
            covariates=covariates,
            family=family,
            link=link,
            **model_options,
        )

    def get_default_method(self) -> str:
        """Determine default fit method based on available methods."""
        methods = set(self.fit_methods())
        if "cp_ao_gaussian_identity" in methods:
            return "cp_ao_gaussian_identity"
        if "cp_ao_glm" in methods:
            return "cp_ao_glm"
        # fallback to base behavior (will raise if none)
        return super().get_default_method()

    def predict(
        self,
        params: dict[str, ttb.ktensor],
        *,
        covariates: ttb.tensor | None = None,
        which: str = "mean",
    ) -> ttb.tensor:
        """
        Predict responses from model parameters and covariates.

        Parameters
        ----------
        params : dict[str, pyttb.ktensor]
            Dictionary containing model parameters, including:
            "coef": regression coefficient in Kruskal form

        covariates : pyttb.tensor, optional
            Optional covariate tensor to use for prediction; if None, will use the
            covariates provided at model initialization. This allows for making
            predictions on new data if desired, but by default will make predictions on
            the training data used for fitting.

        which : {"mean", "linear"}, default="mean"
            If ``"linear"``, return the linear predictor. If ``"mean"``, return
            the inverse-link-transformed mean response.

        Returns
        -------
        pyttb.tensor
            Predicted responses.
        """
        B = params["coef"]
        eta = self.contract_xb(B, covariates=covariates).to_tensor()
        if which == "linear":
            return eta
        elif which == "mean":
            return ttb.tensor(self.family.link.inverse(eta.data))
        else:
            raise ValueError(
                f"Invalid value for 'which': {which}. Must be 'mean' or 'linear'."
            )

    def loglike(self, params: dict[str, ttb.ktensor]) -> float:
        """
        Calculate log-likelihood of the model given parameters.

        Parameters
        ----------
        params: dict[str, ttb.ktensor]
            Dictionary containing model parameters, including:
            ``"coef"``: regression coefficient in Kruskal form

        Returns
        -------
        float
            The log-likelihood of the model given the current parameters and covariates.
        """
        mu = self.predict(params, which="mean")
        # var_weights / freq_weights can be expanded later
        return float(self.family.loglike(self.responses.data, mu.data))

    def contract_xb(
        self,
        coef: ttb.ktensor,
        *,
        covariates: ttb.tensor | None = None,
        normtype: int = 2,
    ) -> ttb.ktensor:
        """
        Compute the CP regression coefficient tensor applied to covariates, i.e. <X|B>.

        pyttb.mttkrp allows a length-1 vector here to indicate a rank-1 "all-ones"
        factor for the sample mode in this contraction. This is a convenient way to
        handle the fact that the regression coefficient tensor has one more mode than
        the covariate tensor, and that the sample mode is shared between them. By using
        np.array([1]) as the factor for the sample mode in the MTTKRP, we effectively
        sum over the sample mode without needing to explicitly form an all-ones vector
        of the appropriate length, which can be memory-efficient for large sample sizes.

        Parameters
        ----------
        coef : ttb.ktensor
            The CP regression coefficient tensor in Kruskal form.

        covariates : pyttb.tensor, optional
            Covariate tensor to use for contraction; if None, will use the covariates
            provided at model initialization. This allows for making predictions on new
            data if desired, but by default will make predictions on the training data
            used for fitting.

        Returns
        -------
        pyttb.ktensor
             The CP regression coefficient tensor applied to covariates, i.e. <X|B>.
        """
        covariates = self.covariates if covariates is None else covariates
        return contract_xb_cp(coef, covariates, normtype=normtype)

    def _init_params(self, rank: int, seed: int = 0) -> ttb.ktensor:
        """Initialize model parameters.

        Parameters
        ----------
        rank : int
            Rank of the CP decomposition of the regression coefficient tensor.

        seed : int
            Random number generator seed (for reproducibility).

        Returns
        -------
        pyttb.ktensor
            Initial guess for regression coefficient.
        """
        rng = np.random.default_rng(seed)
        Binit = []
        for k in range(len(self.covariates.shape) - 1):
            Binit.append(rng.random((self.covariates.shape[k], rank)))
        for k in range(len(self.responses.shape) - 1):
            Binit.append(rng.random((self.responses.shape[k], rank)))
        return ttb.ktensor(Binit).normalize()


def gtotr_cp(
    *,
    responses: ttb.tensor,
    covariates: ttb.tensor,
    family: str | Family | None = None,
    link: str | Link | None = None,
    **model_options: Any,
) -> GToTR_CP:
    """
    Initialize a `GToTR_CP` model.

    This is a convenience wrapper that simply initializes a
    [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP] model instance. The actual fitting
    logic is implemented in the [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP] class,
    which allows for more flexible usage if users want to call the fit method directly
    with custom parameters or access other methods of the
    [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP] class.

    Parameters
    ----------
    responses : pyttb.tensor
        Tensor of response variables, with sample size at the last mode.

    covariates : pyttb.tensor
        Tensor of covariates, with sample size at the last mode

    family : str or gtotr.families.Family, optional
        The family of the model, which determines the likelihood function and link
        function used in the regression. This can be specified as a string, such as
        ``"poisson"`` or ``"gaussian"``, or as a GToTR family object such as
        [`Gaussian`][gtotr.families.Gaussian],
        [`Binomial`][gtotr.families.Binomial], or
        [`Poisson`][gtotr.families.Poisson]. If not specified, the default is
        Gaussian.

    link : str or gtotr.families.links.Link, optional
        The link function to use in the regression. This can be specified as a
        string, such as ``"log"``, ``"identity"``, or ``"logit"``, or as a GToTR
        link object such as [`Identity`][gtotr.families.links.Identity],
        [`Log`][gtotr.families.links.Log], or
        [`Logit`][gtotr.families.links.Logit]. If not specified, the default link
        function for the selected family is used.

    **model_options : Any
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    GToTR_CP
        An instance of the [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP] model
        initialized with the provided responses and covariates, ready to be fitted
        using the fit method.

    Examples
    --------
    Import required packages.

    >>> import numpy as np
    >>> import pyttb as ttb
    >>> from gtotr import gtotr_cp
    >>> from gtotr.utils import contract_xb_cp

    Create data for family="gaussian", link="identity".

    >>> # metadata
    >>> nobs = 20
    >>> cov_shape = (3, 5, nobs)
    >>> resp_shape = (4, 5, 6, nobs)
    >>> rank = 2
    >>> rng = np.random.default_rng(1234)
    >>> # covariates
    >>> X = ttb.tensor(rng.normal(size=cov_shape))
    >>> # coefficient tensor; low-rank CP structure, normtype=2 for Gaussian data
    >>> Vs = [rng.normal(size=(n, rank)) for n in cov_shape[:-1]]
    >>> Us = [rng.normal(size=(m, rank)) for m in resp_shape[:-1]]
    >>> B_true = ttb.ktensor(Vs + Us).normalize(normtype=2)
    >>> # responses; mu = eta for identity link
    >>> eta = contract_xb_cp(B_true, X, normtype=2).to_tensor()
    >>> mu = eta
    >>> noise_sd = 0.05
    >>> Y = ttb.tensor(mu.data + rng.normal(scale=noise_sd, size=mu.shape))

    Create and fit model.

    >>> model = gtotr_cp(responses=Y, covariates=X, family="gaussian", link="identity")
    >>> results = model.fit(rank=rank, printitn=0)

    Make predictions and evaluate fit.

    >>> Yhat = results.predict()
    >>> residual = (Y.data - Yhat.data).flatten()
    >>> rmse = np.sqrt(np.mean(residual**2))
    """
    return GToTR_CP(
        responses=responses,
        covariates=covariates,
        family=family,
        link=link,
        **model_options,
    )
