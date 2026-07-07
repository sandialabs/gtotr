from __future__ import annotations

import pytest


def test_cp_ao_glm_default_method_runs(gtotr_cp_poisson_log_model):
    """
    Default cp_ao_glm behavior should work when glm_method is omitted.
    """
    mod = gtotr_cp_poisson_log_model

    res = mod.fit(
        method="cp_ao_glm",
        rank=2,
        maxiters=2,
        tolerance=1e-3,
        printitn=0,
    )

    assert res.method == "cp_ao_glm"
    assert res.fit_info["method"] == "cp_ao_glm"
    assert res.fit_info["glm_method"] == "irls"
    assert "glm_method_options" in res.fit_info
    assert "llf" in res.fit_info


@pytest.mark.parametrize("glm_method", ["irls", "newton"])
def test_cp_ao_glm_valid_methods_run(gtotr_cp_poisson_log_model, glm_method):
    """
    Selected valid inner GLM methods should run through cp_ao_glm.
    """
    mod = gtotr_cp_poisson_log_model

    glm_method_options = {
        "maxiter": 5,
        "tol": 1e-6,
        "disp": 0,
    }

    res = mod.fit(
        method="cp_ao_glm",
        rank=2,
        maxiters=2,
        tolerance=1e-3,
        printitn=0,
        glm_method=glm_method,
        glm_method_options=glm_method_options,
    )

    assert res.method == "cp_ao_glm"
    assert res.fit_info["glm_method"] == glm_method
    assert res.fit_info["glm_method_options"]["maxiter"] == 5
    assert res.fit_info["glm_method_options"]["tol"] == 1e-6
    assert res.fit_info["glm_method_options"]["disp"] == 0
    assert "llf" in res.fit_info


def test_cp_ao_glm_invalid_method_raises(gtotr_cp_poisson_log_model):
    """
    Invalid glm_method names should raise a clean ValueError from the backend.
    """
    mod = gtotr_cp_poisson_log_model

    with pytest.raises(ValueError, match="Unknown glm_method"):
        mod.fit(
            method="cp_ao_glm",
            rank=2,
            maxiters=1,
            printitn=0,
            glm_method="definitely_not_a_real_method",
        )


def test_cp_ao_glm_default_method_options_are_recorded(gtotr_cp_poisson_log_model):
    """
    Default glm_method_options should be present in fit_info when omitted by user.
    """
    mod = gtotr_cp_poisson_log_model

    res = mod.fit(
        method="cp_ao_glm",
        rank=2,
        maxiters=1,
        printitn=0,
    )

    opts = res.fit_info["glm_method_options"]
    assert opts["maxiter"] == 50
    assert opts["tol"] == 1e-8
    assert opts["disp"] == 0
