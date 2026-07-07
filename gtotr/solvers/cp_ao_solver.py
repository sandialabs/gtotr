# gtotr/solvers/cp_ao_solver.py
"""Shared CP alternating-optimization solver for GToTR models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pyttb as ttb

if TYPE_CHECKING:
    from .cp_ao_backends.base import CPAOBackendBase


def cp_ao_solver(
    model: Any,
    backend: CPAOBackendBase,
    *,
    rank: int,
    init: Literal["random"] | ttb.ktensor = "random",
    maxiters: int = 1000,
    tolerance: float = 1e-4,
    normtype: float = 2,
    printitn: int = 0,
    printinneritn: bool = False,
    trace: bool = True,
    seed: int = 0,
    **backend_options: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Shared CP alternating-optimization solver.

    Returns
    -------
    params : dict
        {"coef": fitted_ktensor}
    fit_info : dict
        Diagnostics (llf, deviance, converged, niter, traces, etc.)
    """
    Y = model.responses
    X = model.covariates

    P = Y.ndims - 1
    Q = X.ndims - 1

    if Y.shape[-1] != X.shape[-1]:
        raise ValueError(
            "responses and covariates must share the same sample mode size"
        )

    # init coefficient ktensor
    if isinstance(init, ttb.ktensor):
        coef = init.copy()
    elif isinstance(init, str) and init.lower() == "random":
        coef = model._init_params(rank=rank, seed=seed)
    else:
        raise ValueError("init must be 'random' or a pyttb.ktensor")

    coef.normalize(normtype=normtype)
    coef.redistribute(mode=0)
    coef_prev = coef.copy()

    backend.initialize(model, rank=rank, **backend_options)

    def eval_ll_dev(coef_now: ttb.ktensor) -> tuple[float, float]:
        eta = model.contract_xb(coef_now).to_tensor().data
        mu = model.link.inverse(eta)
        llf = float(model.family.loglike(Y.data, mu))
        dev = float(model.family.deviance(Y.data, mu))
        return llf, dev

    ll_prev, dev_prev = eval_ll_dev(coef)
    ll, dev = ll_prev, dev_prev

    trace_ll = [ll_prev]
    trace_dev = [dev_prev]
    trace_delta = []

    converged = False
    failed = False
    it = 1

    while it <= maxiters and not converged:
        # --- update covariate-side factors (V modes) ---
        coef.redistribute(mode=0)
        backend.compute_covariate_constants(coef)

        for mode in range(Q):
            if X.shape[mode] == 1:
                continue

            coef.redistribute(mode=mode)
            factor_prev = coef.factor_matrices[mode].copy()

            factor = backend.compute_covariate_factor(coef, mode)
            coef.factor_matrices[mode] = factor

            ll, dev = eval_ll_dev(coef)
            if ll < ll_prev:
                coef.factor_matrices[mode] = factor_prev
                ll, dev = ll_prev, dev_prev
                continue

            coef.normalize(normtype=normtype, mode=mode)

            if printinneritn:
                print(f"\tIter {it:3d}, V_{mode}: llf={ll:.16f}")

        # --- update response-side factors (U modes) ---
        coef.redistribute(mode=Q)
        backend.compute_response_constants(coef)

        for p in range(P):
            if Y.shape[p] == 1:
                continue

            mode = Q + p
            coef.redistribute(mode=mode)
            factor_prev = coef.factor_matrices[mode].copy()

            factor = backend.compute_response_factor(coef, mode)
            coef.factor_matrices[mode] = factor

            ll, dev = eval_ll_dev(coef)
            if ll < ll_prev:
                coef.factor_matrices[mode] = factor_prev
                ll, dev = ll_prev, dev_prev
                continue

            coef.normalize(normtype=normtype, mode=mode)

            if printinneritn:
                print(f"\tIter {it:3d}, U_{p}: llf={ll:.16f}")

        delta = abs(ll - ll_prev) / (abs(ll_prev) + 1e-12)
        converged = delta < tolerance
        failed = ll < ll_prev  # should not happen due to rollback

        if trace:
            trace_ll.append(ll)
            trace_dev.append(dev)
            trace_delta.append(delta)

        if printitn and (it % printitn == 0):
            print(f"Iter {it:3d}: llf={ll:.16f}, delta={delta:.6e}")

        if failed:
            coef = coef_prev.copy()
            ll, dev = ll_prev, dev_prev
            break

        ll_prev, dev_prev = ll, dev
        coef_prev = coef.copy()
        it += 1

    coef.normalize(normtype=normtype)

    params = {"coef": coef}
    fit_info: dict[str, Any] = {
        "method": backend.name,
        "converged": bool(converged and not failed),
        "niter": it - 1,
        "llf": ll,
        "deviance": dev,
        "rank": rank,
    }

    if hasattr(backend, "glm_method"):
        fit_info["glm_method"] = backend.glm_method
    if hasattr(backend, "glm_method_options"):
        fit_info["glm_method_options"] = dict(backend.glm_method_options)

    if trace:
        fit_info["trace_llf"] = trace_ll
        fit_info["trace_deviance"] = trace_dev
        fit_info["trace_convergence"] = trace_delta

    return params, fit_info
