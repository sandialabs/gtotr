# gtotr/utils/tensor_ops.py
"""Tensor operations for GToTR models and fit methods."""

from __future__ import annotations

import numpy as np
import pyttb as ttb


def contract_xb_cp(
    B: ttb.ktensor,
    X: ttb.sptensor | ttb.tensor,
    normtype: float = 2,
) -> ttb.ktensor:
    """
    Contract a CP tensor ``B`` with a covariate tensor ``X``.

    Parameters
    ----------
    B : pyttb.ktensor
        CP regression coefficient tensor. If ``X`` has ``Q`` covariate modes and one
        sample mode, then the first ``Q`` factor matrices of ``B`` correspond to the
        covariate modes and the remaining factor matrices correspond to response modes.

    X : pyttb.sptensor or pyttb.tensor
        Covariate tensor with sample mode in the last dimension.

    normtype : float, default=2
        Normalization type passed to ``pyttb.ktensor.normalize``.

    Returns
    -------
    pyttb.ktensor
        CP tensor representing ``<X|B>`` over the response modes and sample mode.

    Notes
    -----
    This utility is sparse-compatible when the underlying ``pyttb`` tensor type
    supports the required ``mttkrp`` operation. Sparse tensor support at the model and
    fit-method level is still controlled by the corresponding model class and fit
    method.
    """
    Q = len(X.shape) - 1
    W = X.mttkrp([*B.factor_matrices[:Q], np.array([1])], Q)
    W = W @ np.diag(B.weights)
    return ttb.ktensor([*B.factor_matrices[Q:], W]).normalize(normtype=normtype)
