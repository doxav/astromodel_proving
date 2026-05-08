from __future__ import annotations

from pathlib import Path

import numpy as np

from src.identifiability import PRIMARY_EFFECTIVE_PARAMETERS, build_effective_parameter_map, structural_invariance_table
from src.optuna_sqlite import read_best_trial


def test_step03_representative_sqlite_inputs_exist(initial_fit_dir: Path) -> None:
    expected = {"CONTROL_75nA.db", "MFA_100nA.db", "BARIUM_100nA.db"}
    assert expected.issubset({p.name for p in initial_fit_dir.glob("*.db")})
    for name in expected:
        record = read_best_trial(initial_fit_dir / name)
        assert record.objective >= 0
        assert {"d", "pk", "gt", "gs", "wo", "gki", "gl_a", "ca"}.issubset(record.params)


def test_step03_effective_parameter_map_declares_structural_screen(initial_fit_dir: Path) -> None:
    record = read_best_trial(initial_fit_dir / "CONTROL_75nA.db")
    mapping = build_effective_parameter_map(record.params)

    effective_rows = mapping[mapping["coordinate_type"] == "effective"]
    assert set(PRIMARY_EFFECTIVE_PARAMETERS).issubset(set(effective_rows["parameter"]))
    assert set(effective_rows["classification"]) == {"primary_interpretable"}

    d_pk = mapping[mapping["parameter"].isin(["d", "pk"])]
    assert set(d_pk["classification"]) == {"effective_combination_member"}
    assert set(d_pk["effective_parameter"]) == {"P_gap_eff"}


def test_step03_d_pk_product_invariance_is_exact_at_rhs_level(initial_fit_dir: Path) -> None:
    record = read_best_trial(initial_fit_dir / "CONTROL_75nA.db")
    invariance = structural_invariance_table(record.params, record.condition, record.current_na)

    assert set(invariance["invariance"]) == {"d_pk_product"}
    assert np.allclose(invariance["abs_P_gap_eff_diff"], 0.0, atol=1e-14)
    assert float(invariance["max_abs_dzdt_diff"].max()) < 1e-10
    assert float(invariance["abs_I_kgap_diff"].max()) < 1e-10
