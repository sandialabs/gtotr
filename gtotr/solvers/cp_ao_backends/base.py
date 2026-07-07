# gtotr/solvers/cp_ao_backends/base.py
"""Base class for CP alternating-optimization backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pyttb as ttb


class CPAOBackendBase(ABC):
    """Base class for CP alternating-optimization backends."""

    name: str  # e.g., "cp_ao_glm", "cp_ao_gaussian_identity"

    def initialize(self, model: Any, *, rank: int, **opts: Any) -> None:
        """Initialize shared objects and controls for CP AO."""
        self.model = model
        self.responses = model.responses
        self.covariates = model.covariates
        self.family = model.family

        self.rank = int(rank)

        self.responses_shape = self.responses.shape[:-1]
        self.covariates_shape = self.covariates.shape[:-1]
        self.responses_ndims = self.responses.ndims - 1
        self.covariates_ndims = self.covariates.ndims - 1
        self.num_samples = self.responses.shape[-1]

        self.eps = float(opts.get("eps", 0.0))

        if self.responses.shape[-1] != self.covariates.shape[-1]:
            raise ValueError(
                "responses and covariates must share the same sample mode size"
            )

    @abstractmethod
    def compute_covariate_constants(self, coef: ttb.ktensor) -> None:
        """Compute constants used in factor updates associated with covariate modes."""
        ...

    @abstractmethod
    def compute_covariate_factor(self, coef: ttb.ktensor, mode: int) -> np.ndarray:
        """Compute factor associated with covariate mode."""
        ...

    @abstractmethod
    def compute_response_constants(self, coef: ttb.ktensor) -> None:
        """Compute constants used in factor updates associated with response modes."""
        ...

    @abstractmethod
    def compute_response_factor(self, coef: ttb.ktensor, mode: int) -> np.ndarray:
        """Compute factor associated with response mode."""
        ...
