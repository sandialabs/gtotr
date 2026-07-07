# gtotr/solvers/cp_ao_backends/gaussian_identity.py
"""CP alternating-optimization backend for Gaussian family with Identity link."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pyttb as ttb

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .base import CPAOBackendBase


class CPAOGaussianIdentityBackend(CPAOBackendBase):
    """CP alternating optimization backend for Gaussian family with Identity link."""

    name = "cp_ao_gaussian_identity"

    def initialize(self, model: Any, *, rank: int, **opts: Any) -> None:
        """Initialize shared objects and controls for Gaussian-Identity CP AO."""
        super().initialize(model, rank=rank, **opts)

        self.sig2 = float(opts.get("sig2", 1.0))
        self.loglik_nonconst: float | None = None
        self.loglik_const: float | None = None
        self.deviance_nonconst: float | None = None
        self.deviance_const: float | None = None

        # constants during solves
        self.covariate_constant_UU: NDArray[np.float64] | None = None
        self.covariate_constant_Uy: NDArray[np.float64] | None = None
        self.response_constant_W: NDArray[np.float64] | None = None

    def compute_covariate_constants(self, coef: ttb.ktensor) -> None:
        """Compute constants used in factor updates associated with covariate modes."""
        self.covariate_constant_UU = np.ones((self.rank, self.rank))
        for X in coef.factor_matrices[self.covariates_ndims :]:
            self.covariate_constant_UU *= X.T @ X
        self.covariate_constant_Uy = self.responses.mttkrp(
            [*coef.factor_matrices[self.covariates_ndims :], 1], self.responses_ndims
        )

    def compute_covariate_factor(self, coef: ttb.ktensor, mode: int) -> np.ndarray:
        """Compute factor associated with covariate mode by solving a linear system."""
        assert (
            self.covariate_constant_UU is not None
            and self.covariate_constant_Uy is not None
        ), "Must call compute_covariate_constants before compute_covariate_factor"
        Vs = coef.factor_matrices[: self.covariates_ndims]

        # find HH and Hy in linear system
        order: Literal["C", "F"]
        if self.covariates_ndims == 1:
            HH = np.kron(
                self.covariates.data @ self.covariates.data.T,
                self.covariate_constant_UU,
            )
            Hy = (self.covariates.data @ self.covariate_constant_Uy).flatten()
            order = "C"
        else:
            W1 = self.covariates.to_tenmat(
                rdims=np.array([self.covariates_ndims, mode])
            )
            W2 = ttb.khatrirao(*(Vs[:mode] + Vs[mode + 1 :]), reverse=True)
            W = np.matmul(W1.double(), W2).reshape(
                (self.num_samples, self.rank * self.covariates.shape[mode]), order="F"
            )
            HH = (W.T @ W) * np.kron(
                self.covariate_constant_UU,
                np.ones((self.covariates.shape[mode], self.covariates.shape[mode])),
            )
            Hy = np.sum(
                np.kron(
                    self.covariate_constant_Uy,
                    np.ones((1, self.covariates.shape[mode])),
                )
                * W,
                axis=0,
            )
            order = "F"

        # solve linear system HHb = Hy
        # b = np.linalg.solve(HH, Hy)
        HH = np.asarray(HH, dtype=float)
        Hy = np.asarray(Hy, dtype=float)
        b = np.asarray(np.linalg.lstsq(HH, Hy, rcond=None)[0], dtype=float)
        V0 = cast("NDArray[np.float64]", b.reshape(Vs[mode].shape, order=order))

        # find loglikelihood non-constant component and deviance
        sse = -2 * Hy @ b + b @ HH @ b
        self.deviance_nonconst = sse
        self.loglik_nonconst = -0.5 * sse / self.sig2

        return V0

    def compute_response_constants(self, coef: ttb.ktensor) -> None:
        """Compute constants used in factor updates associated with response modes."""
        self.response_constant_W = self.covariates.mttkrp(
            [*coef.factor_matrices[: self.covariates_ndims], 1], self.covariates_ndims
        )

    def compute_response_factor(self, coef: ttb.ktensor, mode: int) -> np.ndarray:
        """Compute factor associated with response mode by solving a linear system."""
        assert self.response_constant_W is not None, (
            "Must call compute_response_constants before compute_response_factor"
        )
        Us = coef.factor_matrices[self.covariates_ndims :]
        k = mode - self.covariates_ndims

        # find GG and Gy in linear system
        GG = np.ones((self.rank, self.rank))

        for F in Us[:k] + Us[k + 1 :] + [self.response_constant_W]:
            GG *= F.T @ F
        Gy = self.responses.mttkrp([*Us, self.response_constant_W], k).T

        # solve linear system GGb = Gy
        # U0 = np.linalg.solve(GG, Gy).T
        GG = np.asarray(GG, dtype=float)
        Gy = np.asarray(Gy, dtype=float)
        U0 = np.linalg.lstsq(GG, Gy)[0].T

        # find loglikelihood non-constant component and deviance
        sse = -2 * np.sum(Gy * U0.T) + np.sum((U0.T @ U0) * GG)
        self.deviance_nonconst = sse
        self.loglik_nonconst = -0.5 * sse / self.sig2
        return U0
