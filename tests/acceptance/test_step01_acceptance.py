from __future__ import annotations

from pathlib import Path

import numpy as np

from src.postfit_sqlite import (
    DEFAULT_REPRESENTATIVE_DBS,
    d_pk_invariance_check,
    representative_mechanism_summary,
    run_step01_postfit_sqlite,
    top_trials_with_effective_parameters,
)


def test_step01_top_trials_export_covers_all_dbs(project_root: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "postfit_sqlite"
    results = run_step01_postfit_sqlite(project_root, top_n=3, output_dir=output_dir)
    top_trials = results["top_trials_all_dbs"]

    assert len(top_trials) == 18 * 3
    assert top_trials["db_name"].nunique() == 18
    for column in ["P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo"]:
        assert column in top_trials.columns
        assert np.isfinite(top_trials[column]).all()


def test_step01_effective_parameter_summary_is_complete(project_root: Path, tmp_path: Path) -> None:
    results = run_step01_postfit_sqlite(project_root, top_n=2, output_dir=tmp_path / "postfit_sqlite")
    summary = results["effective_parameter_summary"]
    assert len(summary) == 18
    assert set(summary.columns) >= {
        "db_name",
        "condition",
        "current_na",
        "objective",
        "P_gap_eff",
        "gamma_t_eff",
        "gamma_s_eff",
        "volume_ratio_wa_wo",
    }


def test_step01_representative_mechanism_summary_contains_hidden_flux_metrics(project_root: Path, tmp_path: Path) -> None:
    results = run_step01_postfit_sqlite(project_root, top_n=2, output_dir=tmp_path / "postfit_sqlite")
    representative = results["representative_mechanism_summary"]

    assert set(representative["db_name"]) == set(DEFAULT_REPRESENTATIVE_DBS)
    for column in [
        "I_Kir_integral",
        "I_kgap_integral",
        "I_leak_integral",
        "gap_to_kir_integral_ratio",
        "dominant_mechanism",
        "proxy_validity_class",
        "proxy_pearson_r",
        "K_o_peak",
    ]:
        assert column in representative.columns
    assert representative["dominant_mechanism"].isin(["Gap", "Kir", "Leak", "Mixed"]).all()
    assert representative["proxy_validity_class"].isin(["strong", "moderate", "weak", "failed"]).all()
    assert np.isfinite(representative["gap_to_kir_integral_ratio"]).all()


def test_step01_invariance_demo_remains_exact() -> None:
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
    check = d_pk_invariance_check(base_params, experiment_type="CONTROL", current_na=100)
    assert np.allclose(check.dzdt_a, check.dzdt_b, rtol=1e-10, atol=1e-10)
    assert np.isclose(check.I_kgap_a, check.I_kgap_b, rtol=1e-10, atol=1e-10)
