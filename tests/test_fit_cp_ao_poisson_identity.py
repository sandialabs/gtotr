from __future__ import annotations

import numpy as np
import pytest
import pyttb as ttb

from gtotr import gtotr_cp, ptotr_cp

from .helpers import make_nonneg_cp_coef


def test_method_registered_and_supported(gtotr_cp_poisson_identity_model):
    mod = gtotr_cp_poisson_identity_model
    assert "cp_ao_poisson_identity" in mod.fit_methods()
    assert mod.family.family_name == "poisson"
    assert mod.family.link.link_name == "identity"


def test_default_method_is_poisson_identity(gtotr_cp_poisson_identity_model):
    mod = gtotr_cp_poisson_identity_model
    assert mod.get_default_method() == "cp_ao_poisson_identity"


def test_default_methods_unchanged_for_other_models(
    gtotr_cp_gaussian_identity_model, gtotr_cp_poisson_log_model
):
    assert (
        gtotr_cp_gaussian_identity_model.get_default_method()
        == "cp_ao_gaussian_identity"
    )
    assert gtotr_cp_poisson_log_model.get_default_method() == "cp_ao_glm"


def test_loglike_improves_over_init(gtotr_cp_poisson_identity_model):
    mod = gtotr_cp_poisson_identity_model
    rank = 2
    B0 = make_nonneg_cp_coef(
        mod.covariates.shape, mod.responses.shape, rank=rank, seed=2
    )
    ll0 = mod.loglike({"coef": B0})

    res = mod.fit(
        method="cp_ao_poisson_identity", rank=rank, init=B0, maxiters=50, printitn=0
    )

    assert res.llf >= ll0 - 1e-8
    for key in ("method", "converged", "niter", "llf", "deviance", "rank"):
        assert key in res.fit_info
    assert res.fit_info["rank"] == rank
    assert res.method == "cp_ao_poisson_identity"


def test_recovers_low_rank_mean(gtotr_cp_poisson_identity_model):
    mod = gtotr_cp_poisson_identity_model
    res = mod.fit(rank=2, maxiters=100, printitn=0)

    mean_hat = res.predict(which="mean").data
    mean_true = mod.contract_xb(mod._B_true).to_tensor().data
    rel_err = np.linalg.norm(mean_hat - mean_true) / np.linalg.norm(mean_true)
    assert rel_err < 0.5  # loose: responses are Poisson counts


def test_trace_keys_present_when_trace_true(gtotr_cp_poisson_identity_model):
    mod = gtotr_cp_poisson_identity_model
    res = mod.fit(rank=2, maxiters=10, printitn=0, trace=True)
    for key in ("trace_llf", "trace_deviance", "trace_convergence"):
        assert key in res.fit_info
        assert len(res.fit_info[key]) >= 1


def test_predict_mean_equals_linear_for_identity(gtotr_cp_poisson_identity_model):
    mod = gtotr_cp_poisson_identity_model
    res = mod.fit(rank=2, maxiters=20, printitn=0)
    mean = res.predict(which="mean")
    linear = res.predict(which="linear")
    assert isinstance(mean, ttb.tensor)
    assert isinstance(linear, ttb.tensor)
    np.testing.assert_allclose(mean.data, linear.data, rtol=1e-10, atol=1e-10)


def test_negative_init_raises_by_default(gtotr_cp_poisson_identity_model):
    mod = gtotr_cp_poisson_identity_model
    B_bad = make_nonneg_cp_coef(
        mod.covariates.shape, mod.responses.shape, rank=2, seed=1
    )
    B_bad.factor_matrices[0][0, 0] = -1.0
    with pytest.raises(ValueError):
        mod.fit(rank=2, init=B_bad, maxiters=3, printitn=0)


def test_negative_init_bypassed_with_check_inputs_false(
    gtotr_cp_poisson_identity_model,
):
    mod = gtotr_cp_poisson_identity_model
    B_bad = make_nonneg_cp_coef(
        mod.covariates.shape, mod.responses.shape, rank=2, seed=1
    )
    B_bad.factor_matrices[0][0, 0] = -1.0
    # Should not raise (guarantee is voided, but the check is bypassed).
    mod.fit(rank=2, init=B_bad, maxiters=3, printitn=0, check_inputs=False)


def test_negative_covariates_raise_by_default():
    rng = np.random.default_rng(0)
    X = ttb.tensor(rng.normal(size=(3, 20)))  # signed covariates
    Y = ttb.tensor(rng.integers(0, 5, size=(4, 20)).astype(float))
    mod = gtotr_cp(responses=Y, covariates=X, family="poisson", link="identity")
    with pytest.raises(ValueError):
        mod.fit(rank=2, maxiters=3, printitn=0)


def test_negative_covariates_bypassed_with_check_inputs_false():
    rng = np.random.default_rng(0)
    X = ttb.tensor(rng.normal(size=(3, 20)))
    Y = ttb.tensor(rng.integers(0, 5, size=(4, 20)).astype(float))
    mod = gtotr_cp(responses=Y, covariates=X, family="poisson", link="identity")
    mod.fit(rank=2, maxiters=3, printitn=0, check_inputs=False)


def test_ptotr_cp_alias_equivalent():
    rng = np.random.default_rng(0)
    X = ttb.tensor(rng.random((3, 20)))
    Y = ttb.tensor(rng.integers(0, 5, size=(4, 20)).astype(float))

    m_alias = ptotr_cp(responses=Y, covariates=X)
    m_full = gtotr_cp(responses=Y, covariates=X, family="poisson", link="identity")

    assert m_alias.family.family_name == m_full.family.family_name == "poisson"
    assert m_alias.family.link.link_name == m_full.family.link.link_name == "identity"
    assert set(m_alias.fit_methods()) == set(m_full.fit_methods())


def test_ptotr_cp_rejects_family_and_link():
    rng = np.random.default_rng(0)
    X = ttb.tensor(rng.random((3, 20)))
    Y = ttb.tensor(rng.integers(0, 5, size=(4, 20)).astype(float))
    with pytest.raises(TypeError):
        ptotr_cp(responses=Y, covariates=X, family="poisson")
    with pytest.raises(TypeError):
        ptotr_cp(responses=Y, covariates=X, link="identity")
