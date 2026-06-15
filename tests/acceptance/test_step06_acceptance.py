from __future__ import annotations

import json

import pytest

from src.step06_predictive_validation import Step06Config, load_step06_inputs, run_step06_predictive_validation
from src.step05_mechanistic_decomposition import Step05Config, run_step05_mechanistic_decomposition


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


def test_step06_mechanism_score_diverse_policy_uses_continuous_score_selection(project_root):
    candidates, _ = load_step06_inputs(
        project_root,
        Step06Config(candidate_policy="mechanism_score_diverse_per_cell", candidates_per_cell=3),
    )

    per_cell_counts = candidates.groupby("file_id", dropna=False).size()
    assert per_cell_counts.le(3).all()
    assert per_cell_counts.gt(1).any()
    assert "step06_selection_min_mechanism_score_distance" in candidates.columns
    assert "step06_selection_stable_phenotype_novel" in candidates.columns
    assert candidates.groupby("file_id", dropna=False)["step06_selection_rank"].min().eq(1).all()


def test_step06_can_read_effective_diverse_step04_source(project_root):
    source_path = project_root / "outputs" / "cell_fits" / "effective_diverse_cell_ensembles.csv"
    if not source_path.exists():
        pytest.skip("effective-diverse Step 04 artifact is not present")

    candidates, _ = load_step06_inputs(
        project_root,
        Step06Config(
            step04_source_path="outputs/cell_fits/effective_diverse_cell_ensembles.csv",
            candidate_policy="all",
        ),
    )

    assert not candidates.empty
    assert "effective_selection_strategy" in candidates.columns
    assert candidates.groupby("file_id", dropna=False).size().ge(1).all()


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


def test_step06_writes_legacy_biological_perturbation_outputs(project_root, tmp_path):
    run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(
            max_candidates=1,
            time_points=30,
            bootstrap_iterations=0,
            run_legacy_mapping=True,
            legacy_top_n_per_db=1,
            legacy_max_configs=2,
            write_outputs=True,
        ),
        output_dir=tmp_path / "mechanisms",
    )
    result = run_step06_predictive_validation(
        project_root,
        Step06Config(
            max_candidates=1,
            time_points=30,
            run_legacy_biological_perturbations=True,
            biological_max_configs_per_category=1,
            biological_run_pair_sweeps=False,
            write_outputs=True,
        ),
        output_dir=tmp_path / "predictive_validation",
    )
    sweeps = result["biological_parameter_perturbation_sweeps"]
    sigmoid = result["sigmoid_state_change_summary"]
    direction = result["experimental_direction_match_summary"]

    assert not sweeps.empty
    assert {
        "perturbation_context",
        "perturbed_parameter",
        "sigmoid_state_change",
        "direction_Ko_efficiency_score",
    }.issubset(sweeps.columns)
    assert not sigmoid.empty
    assert not direction.empty
    assert {
        "regional_experimental_direction",
        "regional_target_scope",
        "regional_match_status",
        "regional_match_interpretation",
    }.issubset(direction.columns)
    assert not direction["regional_match_status"].eq("not_evaluated_in_first_pass").any()
    assert (
        tmp_path / "legacy_perturbation" / "biological_parameter_perturbation_sweeps.csv"
    ).exists()
