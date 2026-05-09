from __future__ import annotations

from pathlib import Path

from src.optuna_sqlite import parse_db_name, parse_study_name, read_best_trial
from src.provenance import build_trace_source_summary, parse_atf_filename


def test_step00_required_paths_exist(initial_fit_dir: Path, atf_dir: Path, threshold_csv: Path) -> None:
    assert initial_fit_dir.exists()
    assert atf_dir.exists()
    assert threshold_csv.exists()
    assert len(list(initial_fit_dir.glob("*.db"))) == 18
    assert len(list(atf_dir.glob("*.atf"))) == 37


def test_step00_representative_name_parsers() -> None:
    condition, current_na = parse_db_name("CONTROL_75nA.db")
    assert condition == "CONTROL"
    assert current_na == 75

    spec = parse_study_name("MFA_100nA_centered_COMBINED_20000t_2024-10-13_10-31-24")
    assert spec.condition == "MFA"
    assert spec.current_na == 100
    assert spec.target_mean_mode == "centered"
    assert spec.objective_loss_type == "COMBINED"
    assert spec.n_target_points == 20000

    control = parse_atf_filename("DH_old.atf")
    assert control.region == "DH"
    assert control.condition == "CONTROL"
    barium = parse_atf_filename("VH_OG_MFA_Ba.atf")
    assert barium.region == "VH"
    assert barium.condition == "MFA_BA"


def test_step00_trace_summary_is_machine_readable(initial_fit_dir: Path) -> None:
    summary = build_trace_source_summary(initial_fit_dir)
    assert set(summary["trace_source"]) >= {
        "CONTROL_TRACES.csv",
        "CONTROL_TRACES_old.csv",
        "MFA_TRACES.csv",
        "BARIUM_TRACES.csv",
    }
    present = summary[summary["provenance_status"] == "present"]
    assert (present["rows"] > 10).all()
    assert (present["n_columns"] >= 2).all()
    removed = summary[summary["trace_source"] == "CONTROL_TRACES_old.csv"].iloc[0]
    assert bool(removed["exists"]) is False
    assert removed["provenance_status"] == "removed_not_used"


def test_step00_sqlite_reader_can_load_representative_best_trial(initial_fit_dir: Path) -> None:
    record = read_best_trial(initial_fit_dir / "BARIUM_100nA.db")
    assert record.condition == "BARIUM"
    assert record.current_na == 100
    assert record.objective >= 0
    for key in ["gki", "pk", "d", "gt", "gs", "K_bath_value_middle", "switching_function"]:
        assert key in record.params
