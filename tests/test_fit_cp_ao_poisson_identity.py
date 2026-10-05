from __future__ import annotations

import numpy as np
import pytest
import pyttb as ttb

from gtotr import PToTR_CP, gtotr_cp, ptotr_cp

from .helpers import make_nonneg_cp_coef


def _dense_to_sptensor(data: np.ndarray) -> ttb.sptensor:
    """Convert a dense ndarray to pyttb.sptensor, preserving explicit nonzeros."""
    subs = np.argwhere(data != 0)
    vals = data[tuple(subs.T)].astype(float) if subs.size else np.array([], dtype=float)
    return ttb.sptensor(subs, vals, shape=data.shape)


def _make_sparse_ptotr_problem(
    seed: int = 123,
) -> tuple[
    ttb.tensor,
    ttb.tensor,
    ttb.sptensor,
    ttb.sptensor,
    ttb.ktensor,
]:
    """Create a small non-negative dense/sparse PToTR problem.

    Returns
    -------
    tuple
        ``(Y_dense, X_dense, Y_sparse, X_sparse, B)``
    """
    rng = np.random.default_rng(seed)

    n = 12
    cov_shape = (3, n)
    resp_shape = (4, n)
    rank = 2

    # Sparse, non-negative covariates with at least one nonzero per sample.
    X_data = rng.random(cov_shape)
    X_data *= rng.random(cov_shape) < 0.45
    for j in range(n):
        if not np.any(X_data[:, j]):
            X_data[rng.integers(0, cov_shape[0]), j] = rng.random() + 0.1

    X_dense = ttb.tensor(X_data)
    X_sparse = _dense_to_sptensor(X_data)

    B = make_nonneg_cp_coef(cov_shape, resp_shape, rank=rank, seed=seed + 1)

    # Use a PToTR model for contraction so the same sparse-aware code path is tested
    # below. The dense model is only used to construct a small synthetic response.
    tmp_model = ptotr_cp(
        responses=ttb.tensor(np.ones(resp_shape)),
        covariates=X_dense,
    )
    mu_data = tmp_model.contract_xb(B).to_tensor().data
    lam = np.clip(mu_data, 1e-9, None)

    Y_data = rng.poisson(lam=lam).astype(float)
    Y_data *= rng.random(resp_shape) < 0.55

    # Ensure the response is not identically zero.
    if not np.any(Y_data):
        idx = np.unravel_index(np.argmax(lam), lam.shape)
        Y_data[idx] = 1.0

    Y_dense = ttb.tensor(Y_data)
    Y_sparse = _dense_to_sptensor(Y_data)

    return Y_dense, X_dense, Y_sparse, X_sparse, B


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


@pytest.mark.filterwarnings("ignore:overflow encountered in multiply:RuntimeWarning")
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in multiply:RuntimeWarning"
)
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


@pytest.mark.filterwarnings("ignore:overflow encountered in multiply:RuntimeWarning")
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in multiply:RuntimeWarning"
)
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

    assert isinstance(m_alias, PToTR_CP)
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


def test_ptotr_cp_accepts_sparse_responses_and_covariates():
    _, _, Y_sparse, X_sparse, _ = _make_sparse_ptotr_problem()

    mod = ptotr_cp(responses=Y_sparse, covariates=X_sparse)

    assert isinstance(mod, PToTR_CP)
    assert isinstance(mod.responses, ttb.sptensor)
    assert isinstance(mod.covariates, ttb.sptensor)
    assert mod.family.family_name == "poisson"
    assert mod.family.link.link_name == "identity"
    assert mod.get_default_method() == "cp_ao_poisson_identity"


def test_sparse_ptotr_fit_methods_are_limited_to_sparse_aware_methods():
    _, _, Y_sparse, X_sparse, _ = _make_sparse_ptotr_problem()

    mod = ptotr_cp(responses=Y_sparse, covariates=X_sparse)
    methods = set(mod.fit_methods())

    assert "cp_ao_poisson_identity" in methods
    assert "cp_ao_glm" not in methods
    assert "cp_ao_gaussian_identity" not in methods


def test_sparse_ptotr_loglike_and_deviance_match_dense_inputs():
    Y_dense, X_dense, Y_sparse, X_sparse, B = _make_sparse_ptotr_problem()

    dense_mod = ptotr_cp(responses=Y_dense, covariates=X_dense)
    sparse_mod = ptotr_cp(responses=Y_sparse, covariates=X_sparse)
    params = {"coef": B}

    dense_llf = dense_mod.loglike(params)
    sparse_llf = sparse_mod.loglike(params)
    dense_dev = dense_mod.deviance(params)
    sparse_dev = sparse_mod.deviance(params)

    np.testing.assert_allclose(sparse_llf, dense_llf, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sparse_dev, dense_dev, rtol=1e-10, atol=1e-10)


def test_sparse_ptotr_predict_returns_dense_tensor():
    _, _, Y_sparse, X_sparse, B = _make_sparse_ptotr_problem()

    mod = ptotr_cp(responses=Y_sparse, covariates=X_sparse)
    pred = mod.predict({"coef": B}, which="mean")

    assert isinstance(pred, ttb.tensor)
    assert pred.shape == Y_sparse.shape
    assert np.all(np.isfinite(pred.data))


def test_sparse_ptotr_fit_runs_short_problem():
    _, _, Y_sparse, X_sparse, B0 = _make_sparse_ptotr_problem()

    mod = ptotr_cp(responses=Y_sparse, covariates=X_sparse)
    res = mod.fit(
        rank=2,
        init=B0,
        maxiters=3,
        maxinneriters=2,
        printitn=0,
        trace=True,
    )

    assert res.method == "cp_ao_poisson_identity"
    assert res.fit_info["method"] == "cp_ao_poisson_identity"
    assert res.fit_info["rank"] == 2
    assert res.fit_info["niter"] >= 1
    assert "llf" in res.fit_info
    assert "deviance" in res.fit_info
    assert np.isfinite(res.fit_info["llf"])
    assert np.isfinite(res.fit_info["deviance"])
    assert isinstance(res.predict(which="mean"), ttb.tensor)
