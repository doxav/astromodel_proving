from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.atf_step02 import run_step02_rebuild_atf_thresholds


@pytest.fixture(scope="module")
def step02_results(project_root: Path, tmp_path_factory: pytest.TempPathFactory):
    outputs_dir = project_root / "outputs" / "features"
    expected = {
        "feature_table_by_sweep.csv",
        "condition_region_sweep_thresholds.csv",
        "feature_reliability_weights.csv",
        "condition_feature_reliability.csv",
        "region_condition_cell_counts.csv",
        "region_effect_summary.csv",
        "redundancy_diagnostics.csv",
        "experimental_kinetic_direction_targets.csv",
        "region_specific_perturbation_direction_targets.csv",
        "experimental_condition_contrast_summary.csv",
        "experimental_region_condition_profile_terms.csv",
    }
    if expected.issubset({p.name for p in outputs_dir.glob("*.csv")}):
        return {
            "feature_table_by_sweep": pd.read_csv(outputs_dir / "feature_table_by_sweep.csv"),
            "region_condition_cell_counts": pd.read_csv(outputs_dir / "region_condition_cell_counts.csv"),
            "redundancy_diagnostics": pd.read_csv(outputs_dir / "redundancy_diagnostics.csv"),
            "condition_feature_reliability": pd.read_csv(outputs_dir / "condition_feature_reliability.csv"),
            "condition_region_sweep_thresholds": pd.read_csv(outputs_dir / "condition_region_sweep_thresholds.csv"),
            "experimental_kinetic_direction_targets": pd.read_csv(outputs_dir / "experimental_kinetic_direction_targets.csv"),
            "region_specific_perturbation_direction_targets": pd.read_csv(outputs_dir / "region_specific_perturbation_direction_targets.csv"),
        }
    output_dir = tmp_path_factory.mktemp("step02_features")
    return run_step02_rebuild_atf_thresholds(project_root, output_dir=output_dir)


def test_step02_pipeline_extracts_all_sweeps(step02_results) -> None:
    feature_df = step02_results["feature_table_by_sweep"]

    assert len(feature_df) == 222
    assert feature_df.groupby("file_id")["sweep"].nunique().eq(6).all()
    assert set(feature_df["region"]) == {"DH", "VH"}
    assert set(feature_df["condition"]) == {"CONTROL", "MFA", "MFA_BA"}


def test_step02_counts_and_redundancy(step02_results) -> None:
    counts = step02_results["region_condition_cell_counts"]
    redundancy = step02_results["redundancy_diagnostics"]

    observed = {(row.region, row.condition): int(row.n_cells) for row in counts.itertuples(index=False)}
    assert observed == {
        ("DH", "CONTROL"): 7,
        ("VH", "CONTROL"): 4,
        ("DH", "MFA"): 6,
        ("VH", "MFA"): 7,
        ("DH", "MFA_BA"): 6,
        ("VH", "MFA_BA"): 7,
    }

    pair = redundancy[
        ((redundancy["feature_a"] == "peak_depolarization_mV") & (redundancy["feature_b"] == "stim_end_depolarization_mV"))
        | ((redundancy["feature_b"] == "peak_depolarization_mV") & (redundancy["feature_a"] == "stim_end_depolarization_mV"))
    ]
    assert len(pair) == 1
    assert bool(pair["redundant_flag"].item()) is True
    assert float(pair["abs_spearman_r"].item()) > 0.95


def test_step02_thresholds_include_reliability_and_condition_reliability(step02_results) -> None:
    thresholds = step02_results["condition_region_sweep_thresholds"]
    condition_reliability = step02_results["condition_feature_reliability"]

    assert {"condition", "region", "sweep", "feature", "median", "iqr", "acceptable_lower", "acceptable_upper", "missing_rate", "reliability_weight", "threshold_scope"}.issubset(thresholds.columns)
    assert "mean_reliability_weight" in condition_reliability.columns
    assert set(condition_reliability["condition"]) == {"CONTROL", "MFA", "MFA_BA"}


def test_step02_writes_experimental_perturbation_targets(step02_results) -> None:
    kinetic = step02_results["experimental_kinetic_direction_targets"]
    regional = step02_results["region_specific_perturbation_direction_targets"]

    assert {"CONTROL_to_MFA", "MFA_to_MFA_BA", "CONTROL_to_MFA_BA"}.issubset(
        set(kinetic["experimental_contrast"])
    )
    assert {
        "experimental_contrast",
        "feature",
        "experimental_direction",
        "estimate",
    }.issubset(kinetic.columns)
    assert "delta_of_delta_DH_minus_VH" in set(regional["scope"])
