from __future__ import annotations

import numpy as np
import pandas as pd

from src.step06_predictive_validation import (
    Step06Config,
    build_prediction_intervals,
    load_step06_inputs,
    run_step06_predictive_validation,
)


def test_step06_inputs_preserve_identity_and_mechanism_labels(project_root):
    candidates, mechanisms = load_step06_inputs(
        project_root, Step06Config(max_candidates=1, write_outputs=False)
    )
    required = {"file_id", "region", "condition", "candidate_id", "mechanism_cluster"}
    assert required.issubset(candidates.columns)
    assert required.issubset(mechanisms.columns)
    assert candidates["region"].notna().all()
    assert candidates["condition"].notna().all()


def test_step06_prediction_intervals_are_finite_for_successful_features(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=1, time_points=40, write_outputs=False)
    )
    intervals = build_prediction_intervals(
        result["candidate_feature_predictions"],
        Step06Config(max_candidates=1, time_points=40, write_outputs=False),
    )
    assert not intervals.empty
    assert {"region", "condition", "sweep", "feature", "pi_lower", "pi_median", "pi_upper"}.issubset(intervals.columns)
    assert np.isfinite(intervals[["pi_lower", "pi_median", "pi_upper"]].to_numpy()).all()


def test_step06_perturbation_rows_have_status_and_hidden_metrics(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=1, time_points=40, write_outputs=False)
    )
    perturb = result["perturbation_sweeps"]
    assert {"simulation_status", "failure_reason", "K_o_peak", "K_o_recovery_error", "robust_under_perturbation"}.issubset(perturb.columns)
    assert "nominal" in set(perturb["perturbation"])
    assert perturb["simulation_status"].isin({"ok", "failed", "unsupported"}).all()


def test_step06_preserves_required_identity_columns(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    required = {"file_id", "region", "condition", "candidate_id"}
    for key in [
        "heldout_current_errors",
        "candidate_feature_predictions",
        "perturbation_sweeps",
    ]:
        assert required.issubset(result[key].columns), key


def test_step06_no_nan_pass_values_for_evaluated_rows(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )

    heldout = result["heldout_current_errors"]
    evaluated_heldout = heldout[heldout["split_strategy"].eq("leave_one_current_out")]
    assert evaluated_heldout["prediction_pass"].notna().all()

    perturb = result["perturbation_sweeps"]
    evaluated_perturb = perturb[perturb["perturbation_status"].eq("evaluated")]
    assert evaluated_perturb["functional_buffering_pass"].notna().all()


def test_step06_ppc_threshold_fallback_is_auditable(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    ppc = result["feature_distribution_ppc"]

    assert {"threshold_fallback", "reliability_weight"}.issubset(ppc.columns)
    assert ppc["threshold_fallback"].isin(
        {"region_specific", "region_pooled", "global_fallback", "missing"}
    ).all()
    assert pd.to_numeric(ppc["reliability_weight"], errors="coerce").notna().all()
