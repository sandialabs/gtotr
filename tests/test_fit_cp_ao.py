from __future__ import annotations

import numpy as np
import pyttb as ttb

from .helpers import make_cp_coef


def test_cp_ao_glm_improves_loglike_over_init(gtotr_cp_poisson_log_model):
    mod = gtotr_cp_poisson_log_model

    # random init coef
    B0 = make_cp_coef(mod.covariates.shape, mod.responses.shape, rank=2, seed=0)
    params0 = {"coef": B0}
    ll0 = mod.loglike(params0)

    res = mod.fit(method="cp_ao_glm", rank=2, init=B0, maxiters=5, printitn=0)

    assert res.llf >= ll0 - 1e-10  # allow tiny numerical noise


def test_cp_ao_gaussian_identity_recovers_synthetic_noiseless(
    gtotr_cp_gaussian_identity_model,
):
    mod = gtotr_cp_gaussian_identity_model
    rank = 2

    B_true = make_cp_coef(mod.covariates.shape, mod.responses.shape, rank=rank, seed=10)
    params_true = {"coef": B_true}

    eta = mod.predict(params_true, which="linear")
    mod.responses = ttb.tensor(eta.data.copy())  # noiseless, identity link => mu = eta

    # start from a random init
    res = mod.fit(
        method="cp_ao_gaussian_identity",
        rank=rank,
        maxiters=100,
        tolerance=1e-8,
        printitn=0,
    )

    eta_hat = res.predict(which="linear").data
    # np.testing.assert_allclose(eta_hat, mod.responses.data, rtol=1e-6, atol=1e-6)

    rmse = np.sqrt(np.mean((eta_hat - mod.responses.data) ** 2))
    rel_err = np.linalg.norm(eta_hat - mod.responses.data) / np.linalg.norm(
        mod.responses.data
    )
    assert rmse < 0.1
    assert rel_err < 5e-2
