from __future__ import annotations

from pathlib import Path

from src.provenance import EXPECTED_ATF_COUNTS, run_step00_provenance


def test_step00_pipeline_writes_expected_tables(project_root: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "provenance"
    results = run_step00_provenance(project_root, output_dir=output_dir)

    expected_files = {
        "db_study_summary.csv",
        "trace_source_summary.csv",
        "control_trace_verification.csv",
        "atf_region_condition_inventory.csv",
        "atf_region_condition_counts.csv",
        "data_source_contract.csv",
    }
    assert expected_files == {p.name for p in output_dir.glob("*.csv")}

    db_summary = results["db_study_summary"]
    assert len(db_summary) == 18
    assert db_summary.groupby("condition")["current_na"].nunique().to_dict() == {"BARIUM": 6, "CONTROL": 6, "MFA": 6}


def test_step00_atf_inventory_matches_region_condition_design(project_root: Path, tmp_path: Path) -> None:
    results = run_step00_provenance(project_root, output_dir=tmp_path / "provenance")
    inventory = results["atf_region_condition_inventory"]
    counts = results["atf_region_condition_counts"]

    assert len(inventory) == 37
    assert set(inventory["region"]) == {"DH", "VH"}
    assert set(inventory["condition"]) == {"CONTROL", "MFA", "MFA_BA"}

    observed = {(row.region, row.condition): int(row.n_cells) for row in counts.itertuples(index=False)}
    assert observed == EXPECTED_ATF_COUNTS
    assert counts.loc[(counts["region"] == "VH") & (counts["condition"] == "CONTROL"), "small_stratum"].item() is True


def test_step00_provenance_statuses_remain_explicit(project_root: Path, tmp_path: Path) -> None:
    results = run_step00_provenance(project_root, output_dir=tmp_path / "provenance")
    provenance = results["control_trace_verification"]

    assert set(provenance["trace_source"]) == {"CONTROL_TRACES.csv", "MFA_TRACES.csv", "BARIUM_TRACES.csv"}

    chosen = provenance.drop_duplicates("db_name")
    assert chosen.groupby(["condition", "chosen_status"]).size().to_dict() == {
        ("BARIUM", "verified"): 6,
        ("CONTROL", "unresolved"): 6,
        ("MFA", "unresolved"): 1,
        ("MFA", "verified"): 5,
    }

    control_rows = provenance[provenance["condition"] == "CONTROL"]
    assert set(control_rows["trace_source"]) == {"CONTROL_TRACES.csv"}
    assert (control_rows["chosen_trace_source"] == "CONTROL_TRACES.csv").all()
    assert (control_rows["chosen_status"] == "unresolved").all()

    mfa150 = provenance[(provenance["condition"] == "MFA") & (provenance["current_na"] == 150)]
    assert len(mfa150) == 1
    assert mfa150["chosen_status"].item() == "unresolved"
