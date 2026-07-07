# gtotr/utils/tensor_ops.py
"""Tensor operations for GToTR models and fit methods."""

from __future__ import annotations

import numpy as np
import pyttb as ttb


def contract_xb_cp(B: ttb.ktensor, X: ttb.tensor, normtype: float = 2) -> ttb.ktensor:
    """
    Contract a CP tensor `B` with a tensor `X`.

    The modes used in the contraction are the first Q modes, which are associated
    with the covariates in a GToTR model.

    Parameters
    ----------
    B : pyttb.ktensor
        CP tensor to contract.

    X : pyttb.tensor
        Tensor to contract with `B`.

    normtype : float, default=2
        Normalization type passed to `pyttb.ktensor.normalize`.

    Returns
    -------
    pyttb.ktensor
        CP representation of the contracted tensor.
    """
    Q = len(X.shape) - 1
    W = X.mttkrp([*B.factor_matrices[:Q], np.array([1])], Q)
    W = W @ np.diag(B.weights)
    return ttb.ktensor([*B.factor_matrices[Q:], W]).normalize(normtype=normtype)
