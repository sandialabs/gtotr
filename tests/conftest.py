# tests/conftest.py
from __future__ import annotations

import numpy as np
import pytest
import pyttb as ttb

from gtotr import gtotr_cp
from gtotr.utils import contract_xb_cp

from .helpers import make_nonneg_cp_coef, make_nonneg_tensor, make_tensor


@pytest.fixture
def gtotr_cp_gaussian_identity_model():
    X = make_tensor((2, 3, 5), seed=0)
    Y = make_tensor((4, 5), seed=1)
    return gtotr_cp(responses=Y, covariates=X, family="gaussian", link="identity")


@pytest.fixture
def gtotr_cp_poisson_log_model():
    rng = np.random.default_rng(3)
    X = make_tensor((2, 3, 5), seed=2)
    Y = ttb.tensor(rng.poisson(lam=2.0, size=(4, 5)).astype(float))
    return gtotr_cp(responses=Y, covariates=X, family="poisson", link="log")


@pytest.fixture
def gtotr_cp_poisson_identity_model():
    # Non-negative low-rank truth: positive B, non-negative X => mu >= 0 (identity link).
    rank = 2
    cov_shape = (3, 20)
    resp_shape = (4, 20)
    X = make_nonneg_tensor(cov_shape, seed=5)
    B_true = make_nonneg_cp_coef(cov_shape, resp_shape, rank=rank, seed=6)
    mu = contract_xb_cp(B_true, X, normtype=1).to_tensor()
    rng = np.random.default_rng(7)
    Y = ttb.tensor(rng.poisson(lam=np.clip(mu.data, 1e-9, None)).astype(float))
    model = gtotr_cp(responses=Y, covariates=X, family="poisson", link="identity")
    model._B_true = B_true
    return model
