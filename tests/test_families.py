from __future__ import annotations

import numpy as np
import pyttb as ttb
import statsmodels.api as sm

from .helpers import make_cp_coef


def test_gaussian_identity_loglike_prefers_true_mu():
    fam = sm.families.Gaussian(sm.families.links.Identity())

    y = np.array([0.0, 1.0, 2.0, 3.0])
    mu_true = y.copy()
    mu_bad = y + 10.0

    ll_true = fam.loglike(y, mu_true)
    ll_bad = fam.loglike(y, mu_bad)

    assert ll_true > ll_bad


def test_model_loglike_gaussian_identity_prefers_true_coef(
    gtotr_cp_gaussian_identity_model,
):
    mod = gtotr_cp_gaussian_identity_model

    # Build a coefficient tensor and define responses to exactly match mu (identity)
    B = make_cp_coef(mod.covariates.shape, mod.responses.shape, rank=2, seed=3)
    params_true = {"coef": B}

    mu = mod.predict(params_true, which="mean")
    mod.responses = ttb.tensor(
        mu.data.copy()
    )  # y := mu so this should be best possible

    ll_true = mod.loglike(params_true)

    # Perturb coefficients -> worse fit -> lower loglike
    B2 = make_cp_coef(mod.covariates.shape, mod.responses.shape, rank=2, seed=4)
    params_bad = {"coef": B2}
    ll_bad = mod.loglike(params_bad)

    assert ll_true > ll_bad


def test_poisson_deviance_near_zero_when_mu_equals_y():
    fam = sm.families.Poisson(sm.families.links.Log())

    y = np.array([1.0, 2.0, 3.0, 4.0])
    mu = y.copy()

    dev = fam.deviance(y, mu)
    assert abs(dev) < 1e-12
