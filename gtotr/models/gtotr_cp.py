# gtotr/models/gtotr_cp.py
"""GToTR with CP decomposition of regression coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyttb as ttb

if TYPE_CHECKING:
    from gtotr.families import Family, Link

from gtotr.utils import contract_xb_cp
from gtotr.utils.likelihood import poisson_deviance_tensor, poisson_loglike_tensor

from .gtotr_base import GToTRBase


class GToTR_CP(GToTRBase):
    """Model class for GToTR with CP decomposition.

    Notes
    -----
    The general ``GToTR_CP`` API is dense-oriented. Sparse tensor support is provided
    by specialized subclasses and fit methods when explicitly documented.
    """

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
        if "cp_ao_poisson_identity" in methods:
            return "cp_ao_poisson_identity"
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


class PToTR_CP(GToTR_CP):
    """CP Poisson-response tensor-on-tensor regression with Identity link.

    ``PToTR_CP`` is the sparse-aware Poisson/Identity specialization of
    [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP]. It accepts dense
    ``pyttb.tensor`` or sparse ``pyttb.sptensor`` responses and covariates.

    Sparse tensor support is intentionally scoped to this model class and to fit
    methods that explicitly support it, currently ``cp_ao_poisson_identity``.
    General GToTR model classes and fit methods should not be assumed sparse-safe.
    """

    def __init__(
        self,
        *,
        responses: ttb.sptensor | ttb.tensor,
        covariates: ttb.sptensor | ttb.tensor,
        **model_options: Any,
    ):
        """Initialize a PToTR CP model with Poisson family and Identity link."""
        if responses.shape[-1] != covariates.shape[-1]:
            raise ValueError(
                "responses and covariates must share the same sample mode size"
            )
        super().__init__(
            responses=responses,  # type: ignore[arg-type]
            covariates=covariates,  # type: ignore[arg-type]
            family="poisson",
            link="identity",
            **model_options,
        )

    def predict(
        self,
        params: dict[str, ttb.ktensor],
        *,
        covariates: ttb.sptensor | ttb.tensor | None = None,
        which: str = "mean",
    ) -> ttb.tensor:
        """
        Predict responses from model parameters and dense or sparse covariates.

        Parameters
        ----------
        params : dict[str, pyttb.ktensor]
            Dictionary containing model parameters, including ``"coef"``.

        covariates : pyttb.sptensor or pyttb.tensor, optional
            Optional covariate tensor to use for prediction. If None, use the
            covariates supplied at model initialization.

        which : {"mean", "linear"}, default="mean"
            If ``"linear"``, return the linear predictor. If ``"mean"``, return the
            inverse-link-transformed mean response. For Poisson/Identity, these are the
            same up to clipping/validation performed elsewhere.

        Returns
        -------
        pyttb.tensor
            Dense predicted responses. Predictions are currently returned dense even
            when the input covariates are sparse.
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

    def loglike(
        self,
        params: dict[str, ttb.ktensor],
        *,
        responses: ttb.sptensor | ttb.tensor | None = None,
        covariates: ttb.sptensor | ttb.tensor | None = None,
    ) -> float:
        """
        Calculate the Poisson log-likelihood for dense or sparse responses.

        Parameters
        ----------
        params : dict[str, pyttb.ktensor]
            Dictionary containing model parameters, including ``"coef"``.

        responses : pyttb.sptensor or pyttb.tensor, optional
            Optional response tensor. If None, use the responses supplied at model
            initialization.

        covariates : pyttb.sptensor or pyttb.tensor, optional
            Optional covariate tensor. If None, use the covariates supplied at model
            initialization.

        Returns
        -------
        float
            Poisson log-likelihood under the Identity-link mean
            ``mu = <covariates | coef>``.
        """
        responses = self.responses if responses is None else responses
        mu = self.predict(params, covariates=covariates, which="mean")
        return poisson_loglike_tensor(responses, mu, eps=self.family.eps)

    def deviance(
        self,
        params: dict[str, ttb.ktensor],
        *,
        responses: ttb.sptensor | ttb.tensor | None = None,
        covariates: ttb.sptensor | ttb.tensor | None = None,
    ) -> float:
        """
        Calculate the Poisson deviance for dense or sparse responses.

        Parameters
        ----------
        params : dict[str, pyttb.ktensor]
            Dictionary containing model parameters, including ``"coef"``.

        responses : pyttb.sptensor or pyttb.tensor, optional
            Optional response tensor. If None, use the responses supplied at model
            initialization.

        covariates : pyttb.sptensor or pyttb.tensor, optional
            Optional covariate tensor. If None, use the covariates supplied at model
            initialization.

        Returns
        -------
        float
            Poisson deviance under the Identity-link mean
            ``mu = <covariates | coef>``.
        """
        responses = self.responses if responses is None else responses
        mu = self.predict(params, covariates=covariates, which="mean")
        return poisson_deviance_tensor(responses, mu, eps=self.family.eps)

    def contract_xb(
        self,
        coef: ttb.ktensor,
        *,
        covariates: ttb.sptensor | ttb.tensor | None = None,
        normtype: int = 2,
    ) -> ttb.ktensor:
        """
        Compute the CP regression coefficient tensor applied to covariates.

        Parameters
        ----------
        coef : pyttb.ktensor
            The CP regression coefficient tensor in Kruskal form.

        covariates : pyttb.sptensor or pyttb.tensor, optional
            Covariate tensor to use for contraction. If None, use the covariates
            supplied at model initialization.

        normtype : int, default=2
            Normalization type passed to ``pyttb.ktensor.normalize``.

        Returns
        -------
        pyttb.ktensor
            The CP regression coefficient tensor applied to covariates.
        """
        covariates = self.covariates if covariates is None else covariates
        return contract_xb_cp(coef, covariates, normtype=normtype)


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
    [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP] model instance.

    Parameters
    ----------
    responses : pyttb.tensor
        Tensor of response variables, with sample size at the last mode.

    covariates : pyttb.tensor
        Tensor of covariates, with sample size at the last mode.

    family : str or gtotr.families.Family, optional
        The family of the model, which determines the likelihood function and link
        function used in the regression.

    link : str or gtotr.families.links.Link, optional
        The link function to use in the regression.

    **model_options : Any
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    GToTR_CP
        A dense-oriented CP GToTR model.
    """
    return GToTR_CP(
        responses=responses,
        covariates=covariates,
        family=family,
        link=link,
        **model_options,
    )


def ptotr_cp(
    *,
    responses: ttb.sptensor | ttb.tensor,
    covariates: ttb.sptensor | ttb.tensor,
    **model_options: Any,
) -> PToTR_CP:
    """
    Initialize a Poisson-response Tensor-on-Tensor Regression (PToTR) model.

    This constructor returns a [`PToTR_CP`][gtotr.models.gtotr_cp.PToTR_CP] model,
    which is the Poisson/Identity specialization of
    [`GToTR_CP`][gtotr.models.gtotr_cp.GToTR_CP].

    Unlike the general ``gtotr_cp`` constructor, ``ptotr_cp`` accepts either dense
    ``pyttb.tensor`` or sparse ``pyttb.sptensor`` responses and covariates. Sparse
    tensor support is scoped to fit methods that explicitly support it, currently
    ``cp_ao_poisson_identity``.

    Parameters
    ----------
    responses : pyttb.sptensor or pyttb.tensor
        Tensor of response variables, with sample size at the last mode.

    covariates : pyttb.sptensor or pyttb.tensor
        Tensor of covariates, with sample size at the last mode.

    **model_options : Any
        Additional keyword arguments passed to the model constructor. Passing
        ``family`` or ``link`` is not allowed because they are preset to
        ``family="poisson"`` and ``link="identity"``.

    Returns
    -------
    PToTR_CP
        A Poisson/Identity CP tensor-on-tensor regression model.

    Raises
    ------
    TypeError
        If ``family`` or ``link`` is passed.

    See Also
    --------
    gtotr.models.gtotr_cp.gtotr_cp : General dense GToTR CP model constructor.
    gtotr.fitmethods.cp_ao_poisson_identity.CPAOPoissonIdentity : The sparse-aware
        Poisson/Identity fit method.

    References
    ----------
    Llosa-Vite, C., & Dunlavy, D. M. (2026). *Poisson-response Tensor-on-Tensor
    Regression and Applications.* arXiv:2604.07377 [stat.ME].
    [https://arxiv.org/abs/2604.07377](https://arxiv.org/abs/2604.07377)
    """
    if "family" in model_options or "link" in model_options:
        raise TypeError(
            "ptotr_cp presets family='poisson' and link='identity'; "
            "do not pass 'family'/'link'."
        )
    return PToTR_CP(
        responses=responses,
        covariates=covariates,
        **model_options,
    )
