# tests/conftest.py
from __future__ import annotations

import numpy as np
import pytest
import pyttb as ttb

from gtotr import gtotr_cp

from .helpers import make_tensor


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
