# gtotr/solvers/cp_ao_backends/poisson_identity.py
"""Self-contained multiplicative CP solver for Poisson family with Identity link.

Unlike the Gaussian/Identity path, this solver owns its whole outer/inner loop and
convergence check, so it is a module-level function rather than a
:class:`gtotr.solvers.cp_ao_backends.base.CPAOBackendBase` subclass. It is reached
via the standard fit-method dispatch (see
:class:`gtotr.fitmethods.cp_ao_poisson_identity.CPAOPoissonIdentity`).

Non-negativity of a user-supplied ``init`` and of the covariates ``X`` is validated
by default (``check_inputs=True``); the multiplicative-update convergence guarantee
assumes a non-negative initial ``B`` and non-negative ``X``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyttb as ttb
from numpy_groupies import aggregate as accumarray

if TYPE_CHECKING:
    from collections.abc import Sequence


def _calculate_pi(
    Data: ttb.sptensor | ttb.tensor,
    Model: ttb.ktensor,
    rank: int,
    factorIndex: int,
    ndims: int,
) -> np.ndarray:
    """Calculate the Pi matrix for the Poisson multiplicative update.

    Parameters
    ----------
    Data : pyttb.sptensor or pyttb.tensor
        Response data tensor.
    Model : pyttb.ktensor
        Current Kruskal-form model whose factors define Pi.
    rank : int
        CP rank.
    factorIndex : int
        Index of the factor being updated (excluded from the product).
    ndims : int
        Number of modes of ``Model``.

    Returns
    -------
    numpy.ndarray
        The Pi matrix.
    """
    if isinstance(Data, ttb.sptensor):
        Pi = np.ones((Data.nnz, rank))
        for i in np.setdiff1d(np.arange(ndims), factorIndex).astype(int):
            Pi *= Model.factor_matrices[i][Data.subs[:, i], :]
    else:
        Pi = ttb.khatrirao(
            *(
                Model.factor_matrices[:factorIndex]
                + Model.factor_matrices[factorIndex + 1 :]
            ),
            reverse=True,
        )

    return Pi


def _calculate_phi(
    Data: ttb.sptensor | ttb.tensor,
    Model: ttb.ktensor,
    rank: int,
    factorIndex: int,
    Pi: np.ndarray,
    epsilon: float,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Calculate the Phi matrix for the Poisson multiplicative update.

    Parameters
    ----------
    Data : pyttb.sptensor or pyttb.tensor
        Response data tensor.
    Model : pyttb.ktensor
        Current Kruskal-form model.
    rank : int
        CP rank.
    factorIndex : int
        Index of the factor being updated.
    Pi : numpy.ndarray
        Pi matrix from :func:`_calculate_pi`.
    epsilon : float
        Safeguard against division by zero.

    Returns
    -------
    numpy.ndarray
        The Phi matrix.
    """
    if isinstance(Data, ttb.sptensor):
        Phi = -np.ones((Data.shape[factorIndex], rank))
        xsubs = Data.subs[:, factorIndex]
        v = np.sum(Model.factor_matrices[factorIndex][xsubs, :] * Pi, axis=1)
        num = Data.vals
        den = np.maximum(v, epsilon)[:, None]
        wvals = num / den
        for r in range(rank):
            Yr = accumarray(
                xsubs,
                np.squeeze(wvals * Pi[:, r][:, None]),
                size=Data.shape[factorIndex],
            )
            Phi[:, r] = Yr
    else:
        Xn = Data.to_tenmat(np.array([factorIndex], order=Data.order), copy=False).data
        V = Model.factor_matrices[factorIndex].dot(Pi.transpose())
        num = Xn
        den = np.maximum(V, epsilon)
        W = num / den
        Phi = W.dot(Pi)

    return Phi


def _mttkrp_keeplastmode(
    X: ttb.sptensor | ttb.tensor,
    U: Sequence[np.ndarray],
    mode: int | np.integer,
) -> np.ndarray[Any, Any]:
    """MTTKRP that keeps the last (sample) mode intact.

    Parameters
    ----------
    X : pyttb.sptensor or pyttb.tensor
        Covariate tensor.
    U : sequence of numpy.ndarray
        Factor matrices for the covariate modes.
    mode : int
        Mode being updated (kept, along with the last/sample mode).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(X.shape[mode], n, R)``.
    """
    q = len(X.shape) - 1
    n = X.shape[q]

    if mode == 0:
        R = U[1].shape[1]
    else:
        R = U[0].shape[1]

    dims = [i for i in range(q) if i != mode]
    Wn = np.zeros((X.shape[mode], n, R))

    for r in range(R):
        Z = [U[i][:, r] for i in dims]
        result = X.ttv(Z, dims)
        if isinstance(result, float):
            Wn[:, :, r] = result
        else:
            Wn[:, :, r] = result.double()

    return Wn


def _check_nonneg_init(B: ttb.ktensor) -> None:
    """Raise ValueError if a user-supplied init ktensor has negative entries."""
    if np.any(B.weights < 0):
        raise ValueError(
            "cp_ao_poisson_identity requires a non-negative initial B: the ktensor "
            "weights must all be >= 0. Pass a non-negative init, use init='random', "
            "or set check_inputs=False to bypass this check (voids the non-negativity "
            "guarantee)."
        )
    for k, Fk in enumerate(B.factor_matrices):
        if np.any(Fk < 0):
            raise ValueError(
                "cp_ao_poisson_identity requires a non-negative initial B: "
                f"factor matrix {k} has negative entries. Pass a non-negative init, "
                "use init='random', or set check_inputs=False to bypass this check "
                "(voids the non-negativity guarantee)."
            )


def _check_nonneg_covariates(X: ttb.sptensor | ttb.tensor) -> None:
    """Raise ValueError if the covariate tensor X has negative entries."""
    vals = X.vals if isinstance(X, ttb.sptensor) else X.data
    if np.any(vals < 0):
        raise ValueError(
            "cp_ao_poisson_identity requires non-negative covariates X. Provide "
            "non-negative covariates, or set check_inputs=False to bypass this check "
            "(voids the non-negativity guarantee)."
        )


def cp_ao_poisson_identity_solve(
    model: Any,
    *,
    rank: int,
    init: Literal["random"] | ttb.ktensor = "random",
    maxiters: int = 100,
    tolerance: float = 1e-10,
    maxinneriters: int = 30,
    epsDivZero: float = 1e-10,
    printitn: int = 0,
    trace: bool = True,
    check_inputs: bool = True,
    seed: int = 0,
    **_ignored: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit Poisson + Identity CP regression via multiplicative alternating updates.

    Handles both dense (``pyttb.tensor``) and sparse (``pyttb.sptensor``) responses.

    Parameters
    ----------
    model : GToTR_CP
        The model providing ``responses``, ``covariates``, ``family``, ``contract_xb``,
        and ``_init_params``.
    rank : int
        Rank of the CP factorization.
    init : {"random"} or pyttb.ktensor, default="random"
        Initial regression coefficient. ``"random"`` uses ``model._init_params``
        (``numpy.random.default_rng``). A supplied ktensor must be non-negative when
        ``check_inputs`` is True.
    maxiters : int, default=100
        Maximum number of outer iterations.
    tolerance : float, default=1e-10
        Tolerance on the relative change in log-likelihood between outer iterations.
    maxinneriters : int, default=30
        Maximum inner (multiplicative-update) iterations per factor per outer iteration.
    epsDivZero : float, default=1e-10
        Safeguard against division by zero in the multiplicative updates.
    printitn : int, default=0
        Print progress every ``printitn`` outer iterations; 0 disables printing.
    trace : bool, default=True
        If True, record per-iteration log-likelihood, deviance, and convergence trace.
    check_inputs : bool, default=True
        If True, validate that a user-supplied ``init`` and the covariates ``X`` are
        non-negative before iterating. Set False to skip these checks (e.g. for large or
        sparse inputs), which voids the non-negativity guarantee.
    seed : int, default=0
        Random number generator seed used when ``init="random"``.
    **_ignored : Any
        Extra keyword arguments (e.g. ``normtype``, ``printinneritn``) are ignored so
        mixed-convention callers do not crash.

    Returns
    -------
    tuple[dict, dict]
        ``(params, fit_info)`` where ``params = {"coef": ktensor}`` and ``fit_info``
        has keys ``method, converged, niter, llf, deviance, rank`` (plus ``trace_llf``,
        ``trace_deviance``, ``trace_convergence`` when ``trace=True``).

    See Also
    --------
    gtotr.fitmethods.cp_ao_poisson_identity.CPAOPoissonIdentity : Fit method that
        dispatches to this solver.
    gtotr.models.gtotr_cp.ptotr_cp : Convenience constructor for Poisson + Identity
        models.

    References
    ----------
    Llosa-Vite, C., & Dunlavy, D. M. (2026). *Poisson-response Tensor-on-Tensor
    Regression and Applications.* arXiv:2604.07377 [stat.ME].
    [https://arxiv.org/abs/2604.07377](https://arxiv.org/abs/2604.07377)
    """
    Y = model.responses
    X = model.covariates

    # data-defined dimensions
    p = len(Y.shape) - 1
    q = len(X.shape) - 1
    ms = np.delete(Y.shape, p)

    # covariate non-negativity check (before iterating)
    if check_inputs:
        _check_nonneg_covariates(X)

    if q == 1:
        Xden = np.outer(X.collapse(1).data, np.ones(rank))

    # initial values
    if isinstance(init, str):
        if init != "random":
            raise ValueError(f"Unknown init '{init}'; use 'random' or a pyttb.ktensor.")
        B = model._init_params(rank=rank, seed=seed)
    else:
        if check_inputs:
            _check_nonneg_init(init)
        B = init.copy()

    B.normalize(normtype=1)

    def full_loglike(coef: ttb.ktensor) -> tuple[float, float]:
        # identity link => mu = <X|B>
        mu = model.contract_xb(coef).to_tensor().data
        return (
            float(model.family.loglike(Y.data, mu)),
            float(model.family.deviance(Y.data, mu)),
        )

    ll_prev, dev_prev = full_loglike(B)
    trace_ll: list[float] = []
    trace_dev: list[float] = []
    trace_delta: list[float] = []

    converged = False
    rr = 0

    # outer loop
    for rr in range(maxiters):
        # First the factors V_1 .. V_q (covariate factors)
        for ii in range(q):
            B.redistribute(mode=ii)
            Vt = B.factor_matrices[ii].copy()
            if q > 1:  # Wn avoids redundant operations inside the inner loop
                Wn = _mttkrp_keeplastmode(X, B.factor_matrices[:q], ii)
                Xden = Wn.sum(1)
            for _kk in range(maxinneriters):
                if q == 1:  # W is a MTTKRP, most operations already formed in Wn
                    W = X.ttm(Vt.T, 0).data.T
                else:
                    W = np.einsum("abc,ac->bc", Wn, Vt)
                UW = ttb.ktensor([*B.factor_matrices[q : q + p], W])
                Pit = _calculate_pi(Y, UW, rank, p, p + 1)
                Phit = _calculate_phi(Y, UW, rank, p, Pit, epsDivZero)
                if q == 1:
                    num = X.ttm(Phit.T, 1).data
                else:
                    num = np.einsum("abc,bc->ac", Wn, Phit)
                Vt *= num / np.maximum(Xden, epsDivZero)
            B.factor_matrices[ii] = Vt
            B.normalize(normtype=1, mode=ii)

        if q == 1:  # W is a MTTKRP, most operations already formed in Wn
            W = X.ttm(B.factor_matrices[q - 1].T, 0).data.T
        else:
            W = np.einsum("abc,ac->bc", Wn, B.factor_matrices[q - 1])
        Wsum = W.sum(0)

        # Now the factors U_1 .. U_p (response factors)
        for ii in range(p):
            B.redistribute(mode=ii + q)
            Ut = B.factor_matrices[ii + q].copy()
            Wden = np.outer(np.ones(ms[ii]), Wsum)
            for _kk in range(maxinneriters):
                UW = ttb.ktensor([*B.factor_matrices[q : q + p], W])
                Pit = _calculate_pi(Y, UW, rank, ii, p + 1)
                Phit = _calculate_phi(Y, UW, rank, ii, Pit, epsDivZero)
                Ut *= Phit / np.maximum(Wden, epsDivZero)
                B.factor_matrices[ii + q] = Ut
            B.normalize(normtype=1, mode=ii + q)

        # Convergence, using gtotr's full Poisson loglike (single source of truth)
        ll, dev = full_loglike(B)
        convcrit = abs((ll - ll_prev) / (abs(ll) + 1e-12))
        if trace:
            trace_ll.append(ll)
            trace_dev.append(dev)
            trace_delta.append(convcrit)
        ll_prev, dev_prev = ll, dev

        if printitn and (rr % printitn == 0):
            print(f"\tIter {rr}: loglikelihood = {ll}")

        if convcrit < tolerance:
            converged = True
            if printitn:
                print(
                    "converged after",
                    rr + 1,
                    "iterations with a loglikelihood of",
                    ll,
                )
            break
        elif rr == maxiters - 1 and printitn:
            print(
                "reached maximum iterations of",
                maxiters,
                "with a relative change in loglikelihood of",
                convcrit,
            )

    params = {"coef": B}
    fit_info: dict[str, Any] = {
        "method": "cp_ao_poisson_identity",
        "converged": bool(converged),
        "niter": rr + 1,
        "llf": ll_prev,
        "deviance": dev_prev,
        "rank": rank,
    }
    if trace:
        fit_info["trace_llf"] = trace_ll
        fit_info["trace_deviance"] = trace_dev
        fit_info["trace_convergence"] = trace_delta

    return params, fit_info
