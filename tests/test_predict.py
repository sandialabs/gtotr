from __future__ import annotations

import numpy as np

from .helpers import make_cp_coef


def test_predict_mean_is_inverse_link(gtotr_cp_gaussian_identity_model):
    mod = gtotr_cp_gaussian_identity_model  # fixture or construct inline

    B = make_cp_coef(mod.covariates.shape, mod.responses.shape, rank=2, seed=1)
    params = {"coef": B}

    eta = mod.predict(params, which="linear")
    mu = mod.predict(params, which="mean")

    # For identity link, mu == eta
    np.testing.assert_allclose(mu.data, eta.data, rtol=0, atol=1e-12)


def test_predict_mean_poisson_log_is_exp_eta(gtotr_cp_poisson_log_model):
    mod = gtotr_cp_poisson_log_model
    B = make_cp_coef(mod.covariates.shape, mod.responses.shape, rank=2, seed=2)
    params = {"coef": B}

    eta = mod.predict(params, which="linear").data
    mu = mod.predict(params, which="mean").data

    np.testing.assert_allclose(mu, np.exp(eta), rtol=1e-10, atol=1e-12)
