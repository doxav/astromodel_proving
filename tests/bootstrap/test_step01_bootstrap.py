from __future__ import annotations

from pathlib import Path

import numpy as np

from src.postfit_sqlite import DEFAULT_REPRESENTATIVE_DBS, d_pk_invariance_check, effective_parameters_from_flat
from src.optuna_sqlite import read_best_trial


def test_step01_representative_dbs_exist(initial_fit_dir: Path) -> None:
    for name in DEFAULT_REPRESENTATIVE_DBS:
        assert (initial_fit_dir / name).exists(), f"Missing representative DB: {name}"


def test_step01_direct_sqlite_best_trial_loading(initial_fit_dir: Path) -> None:
    record = read_best_trial(initial_fit_dir / "MFA_100nA.db")
    assert record.condition == "MFA"
    assert record.current_na == 100
    assert np.isfinite(record.objective)
    assert record.trial_number >= 0
    assert record.params["switching_function"] in {"sigmoid", "tanh", "hill", "soft_threshold"}


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
