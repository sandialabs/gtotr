# gtotr/solvers/cp_ao_backends/glm.py
"""CP alternating-optimization backend with GLM factor updates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pyttb as ttb
import statsmodels.api as sm

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .base import CPAOBackendBase


class CPAOGLMBackend(CPAOBackendBase):
    """CP alternating-optimization backend with GLM factor updates."""

    name = "cp_ao_glm"

    _valid_glm_methods: ClassVar[set[str]] = {
        "irls",
        "newton",
        "nm",
        "bfgs",
        "lbfgs",
        "powell",
        "cg",
        "ncg",
    }

    def initialize(self, model: Any, *, rank: int, **opts: Any) -> None:
        """Initialize shared objects and controls for GLM-based CP AO."""
        super().initialize(model, rank=rank, **opts)

        self.glm_method = str(opts.get("glm_method", "irls")).lower()
        if self.glm_method not in self._valid_glm_methods:
            raise ValueError(
                f"Unknown glm_method '{self.glm_method}'. "
                f"Valid options are: {sorted(self._valid_glm_methods)}"
            )

        self.glm_method_options = dict(opts.get("glm_method_options", {}))
        self.glm_method_options.setdefault("maxiter", 50)
        self.glm_method_options.setdefault("tol", 1e-8)
        self.glm_method_options.setdefault("disp", 0)

        # reshape / ordering conventions used in factor updates
        self.covariate_return_order = opts.get("covariate_return_order", "F")

        # cached objects updated once per outer AO sweep
        self.covariate_constant_U: NDArray[np.float64] | None = None
        self.response_constant_W: NDArray[np.float64] | None = None

        # flattened / reshaped data views used repeatedly
        self.responses_vec = self.responses.data.reshape(-1, order="F")

    def compute_covariate_constants(self, coef: ttb.ktensor) -> None:
        """Compute constants used in factor updates associated with covariate modes."""
        self.covariate_constant_U = ttb.khatrirao(
            *(coef.factor_matrices[self.covariates_ndims :]),
            reverse=True,
        )

    def compute_covariate_factor(self, coef: ttb.ktensor, mode: int) -> np.ndarray:
        """Compute factor associated with covariate mode by solving a GLM."""
        assert self.covariate_constant_U is not None, (
            "Must call compute_covariate_constants before compute_covariate_factor"
        )
        H = _compute_h_matrix(
            responses_params=self.covariate_constant_U,
            mode=mode,
            covariates=self.covariates,
            covariates_params=coef.factor_matrices[: self.covariates_ndims],
            responses_dims=self.responses_shape,
            covariates_dims=self.covariates_shape,
            num_samples=self.num_samples,
            rank=self.rank,
            mode_end=self.covariates_ndims,
        )

        params = self._fit_glm(responses=self.responses_vec, covariates=H)

        # aggressive garbage collection to free memory from large intermediate objects
        del H
        gc.collect()

        result = np.reshape(
            params,
            coef.factor_matrices[mode].shape,
            order=self.covariate_return_order,
        )
        return result

    def compute_response_constants(self, coef: ttb.ktensor) -> None:
        """Compute constants used in factor updates associated with response modes."""
        self.response_constant_W = self.covariates.mttkrp(
            [*coef.factor_matrices[: self.covariates_ndims], 1],
            self.covariates_ndims,
        )

    def compute_response_factor(self, coef: ttb.ktensor, mode: int) -> np.ndarray:
        """Compute factor associated with response mode by solving a GLM."""
        k = mode - self.covariates_ndims

        assert self.response_constant_W is not None, (
            "Must call compute_response_constants before compute_response_factor"
        )
        G = _compute_g_matrix(
            responses_params=coef.factor_matrices[self.covariates_ndims :],
            mode=k,
            covariates_mttkrp_covs=self.response_constant_W,
        )

        cols = np.flip(np.delete(np.arange(len(self.responses_shape) + 1), k))
        Y = self.responses.to_tenmat(rdims=np.array([k]), cdims=cols).data

        params = np.zeros_like(coef.factor_matrices[mode], order="F")
        for row in range(Y.shape[0]):
            row_params = self._fit_glm(responses=Y[row, :], covariates=G)
            params[row, :] = row_params

        # aggressive garbage collection to free memory from large intermediate objects
        del G
        del Y
        gc.collect()

        return params

    def _fit_glm(self, responses: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """Solve a GLM subproblem and return fitted coefficients."""
        res = sm.GLM(
            responses,
            covariates,
            family=self.family,
        ).fit(
            method=self.glm_method,
            **self.glm_method_options,
        )

        params = np.asarray(res.params).copy()
        del res
        gc.collect()
        return params


def _compute_g_matrix(
    responses_params: list[np.ndarray], mode: int, covariates_mttkrp_covs: np.ndarray
):
    return ttb.khatrirao(
        *(
            responses_params[:mode]
            + responses_params[mode + 1 :]
            + [covariates_mttkrp_covs]
        )
    )


def _compute_h_matrix(
    responses_params: np.ndarray,
    mode: int,
    covariates: ttb.tensor,
    covariates_params: list[np.ndarray],
    responses_dims: list[int] | tuple[int],
    covariates_dims: list[int] | tuple[int],
    num_samples: int,
    rank: int,
    mode_end: int,
):
    if mode_end == 1:
        # TODO: more efficient: this or np.kron(covariates.double().T, responses.params)
        return np.kron(covariates.double(), responses_params.T).T
    A = covariates.to_tenmat(rdims=np.array([mode_end, mode]))
    B = ttb.khatrirao(
        *(covariates_params[:mode] + covariates_params[mode + 1 :]), reverse=True
    )
    W = np.matmul(A.double(), B)
    return ttb.khatrirao(*([responses_params, W]), reverse=True).reshape(
        (np.prod(responses_dims) * num_samples, rank * covariates_dims[mode]), order="F"
    )


def _setup_linear_system(
    G: np.ndarray,
    responses: np.ndarray,
    mode: int,
    covariates: ttb.tensor,
    covariates_params: list[np.ndarray],
    dims: list[int] | tuple[int],
    num_samples: int,
    rank: int,
    mode_end: int,
):
    if mode_end == 1:
        HH = np.kron(np.matmul(covariates.data, covariates.data.T), G)
        Hy = np.matmul(covariates.data, responses).flatten()
        return HH, Hy
    A = covariates.to_tenmat(rdims=np.array([mode_end, mode]))
    B = ttb.khatrirao(
        *(covariates_params[:mode] + covariates_params[mode + 1 :]), reverse=True
    )
    W = np.matmul(A.double(), B).reshape((num_samples, rank * dims[mode]), order="F")
    HH = np.matmul(W.T, W) * np.kron(G, np.ones((dims[mode], dims[mode])))
    Hy = np.sum(np.kron(responses, np.ones((1, dims[mode]))) * W, axis=0)
    return HH, Hy


def _gram_factors(
    coef: ttb.ktensor | list[np.ndarray],
    begin: int = 0,
    end: int | None = None,
    rank: int | None = None,
):
    if isinstance(coef, ttb.ktensor):
        if rank is None:
            rank = coef.ncomponents
        if end is None:
            end = coef.ndims
        U = coef.factor_matrices[begin:end]
    elif isinstance(coef, list):
        for u in coef:
            if not isinstance(u, np.ndarray):
                raise TypeError("Expected a list of numpy arrays.")
        if rank is None:
            rank = coef[0].shape[1]
        if end is None:
            end = len(coef)
        U = coef[begin:end]
    G = np.ones((rank, rank))
    for u in U:
        G *= np.matmul(u.T, u)
    return G
