from __future__ import annotations

import numpy as np

from src.postfit_sqlite import d_pk_invariance_check, effective_parameters_from_flat, run_step01_postfit_sqlite


def test_step01_invariance_helper_preserves_effective_gap_permeability() -> None:
    base_params = {
        "gki": 50.0,
        "pk": 2e-4,
        "d": 3.0,
        "gt": 8.0,
        "gs": 10.0,
        "zth": 0.2,
        "zs": 0.05,
        "K_bath_value_middle": 8.2,
        "eps": 0.01,
        "eps_middle": 0.5,
        "wo": 1500.0,
        "ca": 400.0,
        "gl_a": 0.01,
        "Va_l": -70.0,
        "Va_s": -90.0,
        "switching_function": "sigmoid",
    }
    check = d_pk_invariance_check(base_params)
    assert np.isclose(check.P_gap_eff_a, check.P_gap_eff_b)
    assert np.allclose(check.dzdt_a, check.dzdt_b, rtol=1e-10, atol=1e-10)
    assert np.isclose(check.I_kgap_a, check.I_kgap_b, rtol=1e-10, atol=1e-10)

    effective = effective_parameters_from_flat(base_params, "CONTROL", 100)
    assert set(effective) == {"P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo"}


def test_step01_pipeline_returns_expected_cached_outputs(project_root) -> None:
    results = run_step01_postfit_sqlite(project_root)
    effective_df = results["effective_parameter_summary"]
    rep_df = results["representative_mechanism_summary"]
    assert len(effective_df) == 18
    assert set(rep_df["condition"]) == {"CONTROL", "MFA", "BARIUM"}
