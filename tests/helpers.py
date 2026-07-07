# gtotr/tests/helpers.py
from __future__ import annotations

import numpy as np
import pyttb as ttb


def make_tensor(shape, seed=0):
    rng = np.random.default_rng(seed)
    return ttb.tensor(rng.normal(size=shape))


def make_cp_coef(cov_shape, resp_shape, rank, seed=0):
    rng = np.random.default_rng(seed)
    Q = len(cov_shape) - 1
    P = len(resp_shape) - 1
    factors = []
    for q in range(Q):
        factors.append(rng.normal(size=(cov_shape[q], rank)))
    for p in range(P):
        factors.append(rng.normal(size=(resp_shape[p], rank)))
    return ttb.ktensor(factors).normalize()
