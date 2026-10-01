# gtotr/utils/likelihood.py
"""Likelihood utilities for tensor-valued responses."""

from __future__ import annotations

import numpy as np
import pyttb as ttb
from scipy.special import gammaln


def _as_dense_array(x: ttb.sptensor | ttb.tensor | np.ndarray) -> np.ndarray:
    """Return a dense ``numpy.ndarray`` for pyttb dense/sparse tensor-like objects."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, ttb.sptensor):
        return x.to_tensor().data
    if isinstance(x, ttb.tensor):
        return x.data
    raise TypeError(f"Expected pyttb.tensor, pyttb.sptensor, or ndarray; got {type(x)!r}")


def _sptensor_vals(y: ttb.sptensor) -> np.ndarray:
    """Return sparse tensor values as a one-dimensional floating array."""
    return np.asarray(y.vals, dtype=float).reshape(-1)


def _sptensor_subs(y: ttb.sptensor) -> np.ndarray:
    """Return sparse tensor subscripts as a two-dimensional integer array."""
    return np.asarray(y.subs, dtype=int)


def poisson_loglike_tensor(
    responses: ttb.sptensor | ttb.tensor,
    mu: ttb.tensor | np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute Poisson log-likelihood for dense or sparse tensor responses.

    Parameters
    ----------
    responses : pyttb.sptensor or pyttb.tensor
        Observed Poisson counts.

    mu : pyttb.tensor or numpy.ndarray
        Mean tensor. This implementation expects ``mu`` to be dense.

    eps : float, default=1e-12
        Lower clipping threshold for ``mu`` to avoid ``log(0)``.

    Returns
    -------
    float
        Poisson log-likelihood
        ``sum_i y_i log(mu_i) - mu_i - log(y_i!)``.

    Notes
    -----
    For sparse responses, zero entries are not stored but still contribute
    ``-mu_i`` to the log-likelihood. The sparse implementation therefore computes

    ``sum_nonzero y_i log(mu_i) - log(y_i!) - sum_all mu_i``

    without materializing the sparse response as a dense array.
    """
    lam = np.clip(_as_dense_array(mu), eps, np.inf)

    if isinstance(responses, ttb.tensor):
        y = responses.data
        return float(np.sum(y * np.log(lam) - lam - gammaln(y + 1.0)))

    if isinstance(responses, ttb.sptensor):
        subs = _sptensor_subs(responses)
        vals = _sptensor_vals(responses)

        if vals.size == 0:
            return float(-np.sum(lam))

        lam_nz = lam[tuple(subs.T)]
        return float(np.sum(vals * np.log(lam_nz) - gammaln(vals + 1.0)) - np.sum(lam))

    raise TypeError(
        "responses must be a pyttb.tensor or pyttb.sptensor; "
        f"got {type(responses)!r}"
    )


def poisson_deviance_tensor(
    responses: ttb.sptensor | ttb.tensor,
    mu: ttb.tensor | np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute Poisson deviance for dense or sparse tensor responses.

    Parameters
    ----------
    responses : pyttb.sptensor or pyttb.tensor
        Observed Poisson counts.

    mu : pyttb.tensor or numpy.ndarray
        Mean tensor. This implementation expects ``mu`` to be dense.

    eps : float, default=1e-12
        Lower clipping threshold for ``mu``.

    Returns
    -------
    float
        Poisson deviance
        ``2 * sum_i [y_i log(y_i / mu_i) - (y_i - mu_i)]``.

    Notes
    -----
    For sparse responses, zero entries contribute ``2 * mu_i``. The sparse
    implementation computes

    ``2 * (sum_nonzero y_i log(y_i / mu_i) - sum_nonzero y_i + sum_all mu_i)``

    without materializing the sparse response as a dense array.
    """
    lam = np.clip(_as_dense_array(mu), eps, np.inf)

    if isinstance(responses, ttb.tensor):
        y = responses.data
        term = np.zeros_like(lam, dtype=float)
        mask = y > 0
        term[mask] = y[mask] * np.log(y[mask] / lam[mask])
        dev_obs = 2.0 * (term - (y - lam))
        return float(np.sum(dev_obs))

    if isinstance(responses, ttb.sptensor):
        subs = _sptensor_subs(responses)
        vals = _sptensor_vals(responses)

        if vals.size == 0:
            return float(2.0 * np.sum(lam))

        lam_nz = lam[tuple(subs.T)]
        nz_term = vals * np.log(vals / lam_nz)
        return float(2.0 * (np.sum(nz_term) - np.sum(vals) + np.sum(lam)))

    raise TypeError(
        "responses must be a pyttb.tensor or pyttb.sptensor; "
        f"got {type(responses)!r}"
    )
