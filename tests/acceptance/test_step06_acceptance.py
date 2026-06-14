from __future__ import annotations

import json

import pytest

from src.step06_predictive_validation import Step06Config, load_step06_inputs, run_step06_predictive_validation


def test_step06_writes_required_outputs_and_conservative_labels(project_root):
    out_dir = project_root / "outputs" / "step06_acceptance"
    result = run_step06_predictive_validation(
        project_root,
        Step06Config(max_candidates=1, time_points=40, write_outputs=True),
        output_dir=out_dir,
    )
    required_files = {
        "heldout_current_errors.csv",
        "prediction_intervals.csv",
        "feature_distribution_ppc.csv",
        "perturbation_sweeps.csv",
        "robustness_summary.csv",
        "analysis_summary.json",
        "performance_benchmark.csv",
    }
    assert required_files.issubset({p.name for p in out_dir.iterdir()})
    robustness = result["robustness_summary"]
    assert "step06_screen_claim" in robustness.columns
    assert "final_biological_degeneracy_claim_allowed" in robustness.columns
    assert (~robustness["final_biological_degeneracy_claim_allowed"].astype(bool)).all()
    assert not robustness["claim_scope_after_step06"].str.contains(
        "final biological degeneracy.*allowed", regex=True
    ).any()
    summary = json.loads((out_dir / "analysis_summary.json").read_text())
    assert summary["n_perturbation_rows"] >= 1


def test_step06_ppc_rows_are_region_aware_and_weighted(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=1, time_points=40, write_outputs=False)
    )
    ppc = result["feature_distribution_ppc"]
    required = {
        "region",
        "condition",
        "sweep",
        "feature",
        "empirical_lower",
        "empirical_upper",
        "coverage_fraction",
        "reliability_weight",
        "threshold_fallback",
    }
    assert required.issubset(ppc.columns)
    assert ppc["region"].notna().all()
    assert ppc["condition"].notna().all()
    assert ppc["threshold_fallback"].isin(
        {"region_specific", "region_pooled", "global_fallback", "missing"}
    ).all()


def test_step06_heldout_rows_cover_all_six_ordered_sweeps(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=1, time_points=40, write_outputs=False)
    )
    heldout = result["heldout_current_errors"]
    loo = heldout[heldout["split_strategy"] == "leave_one_current_out"]
    assert set(loo["sweep"]) == {1, 2, 3, 4, 5, 6}
    assert set(loo["current_na"]) == {50, 75, 100, 125, 150, 175}


def test_step06_heldout_is_candidate_level_when_step04_rows_exist(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    heldout = result["heldout_current_errors"]
    loo = heldout[heldout["split_strategy"] == "leave_one_current_out"]
    counts = loo.groupby(
        ["file_id", "region", "condition", "candidate_id"], dropna=False
    )["sweep"].nunique()
    assert counts.ge(6).all()


def test_step06_missing_step05_labels_are_explicit(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    heldout = result["heldout_current_errors"]
    assert "mechanism_label_status" in heldout.columns
    assert heldout["mechanism_label_status"].isin(
        {"step05_label_available", "missing_step05_label"}
    ).all()


def test_step06_top_k_candidate_policy_limits_candidates_per_cell(project_root):
    candidates, _ = load_step06_inputs(
        project_root,
        Step06Config(candidate_policy="top_k_per_cell", candidates_per_cell=2),
    )

    per_cell_counts = candidates.groupby("file_id", dropna=False).size()
    assert per_cell_counts.le(2).all()
    assert per_cell_counts.gt(1).any()


def test_step06_mechanism_diverse_policy_uses_one_candidate_per_cell_mechanism(project_root):
    candidates, _ = load_step06_inputs(
        project_root,
        Step06Config(candidate_policy="mechanism_diverse_per_cell", candidates_per_cell=3),
    )

    per_cell_counts = candidates.groupby("file_id", dropna=False).size()
    per_cell_mechanism_counts = candidates.groupby(["file_id", "mechanism_cluster"], dropna=False).size()
    assert per_cell_counts.le(3).all()
    assert per_cell_mechanism_counts.le(1).all()


def test_step06_candidate_policy_rejects_invalid_k(project_root):
    with pytest.raises(ValueError, match="candidates_per_cell"):
        load_step06_inputs(
            project_root,
            Step06Config(candidate_policy="top_k_per_cell", candidates_per_cell=0),
        )


def test_step06_final_degeneracy_claim_is_never_enabled_by_default(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    robust = result["robustness_summary"]

    assert "final_biological_degeneracy_claim_allowed" in robust.columns
    assert (~robust["final_biological_degeneracy_claim_allowed"].astype(bool)).all()


def test_step06_default_perturbations_cover_all_six_currents(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=1, time_points=30, write_outputs=False)
    )
    perturb = result["perturbation_sweeps"]
    assert set(perturb["current_na"]) == {50, 75, 100, 125, 150, 175}
    robust = result["robustness_summary"]
    assert "biological_description_score" in robust.columns


def test_step06_partial_step04_heldout_is_explicit_not_silent(project_root, tmp_path):
    result = run_step06_predictive_validation(
        project_root,
        Step06Config(
            max_candidates=1,
            time_points=30,
            write_outputs=True,
            require_candidate_level_heldout=True,
        ),
        output_dir=tmp_path,
    )
    heldout = result["heldout_current_errors"]

    assert "split_strategy" in heldout.columns
    assert heldout["prediction_status"].isin(
        {"predictive_pass", "prediction_limited", "fit_only"}
    ).all()
