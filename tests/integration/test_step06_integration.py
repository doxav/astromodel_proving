from __future__ import annotations

from tests._notebook import execute_notebook
from src.step06_predictive_validation import Step06Config, run_step06_predictive_validation


def test_step06_region_condition_summaries_are_coherent(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    summary_groups = set(
        zip(
            result["robustness_summary"]["region"],
            result["robustness_summary"]["condition"],
            result["robustness_summary"]["mechanism_cluster"],
        )
    )
    heldout_groups = set(
        zip(
            result["heldout_current_errors"]["region"],
            result["heldout_current_errors"]["condition"],
            result["heldout_current_errors"]["mechanism_cluster"],
        )
    )
    assert summary_groups.issubset(heldout_groups)


def test_step06_cross_table_coherence(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    intervals = result["prediction_intervals"]
    ppc = result["feature_distribution_ppc"]
    perturb = result["perturbation_sweeps"]
    robust = result["robustness_summary"]

    interval_keys = set(zip(intervals["region"], intervals["condition"], intervals["sweep"], intervals["feature"]))
    ppc_keys = set(zip(ppc["region"], ppc["condition"], ppc["sweep"], ppc["feature"]))
    assert ppc_keys.issubset(interval_keys)
    assert set(robust["mechanism_cluster"].dropna()).issubset(
        set(perturb["mechanism_cluster"].dropna())
    )


def test_step06_prediction_intervals_are_traceable_to_candidates(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=2, time_points=40, write_outputs=False)
    )
    intervals = result["prediction_intervals"]
    assert "source_candidate_ids" in intervals.columns
    assert intervals["source_candidate_ids"].astype(str).str.len().gt(0).all()


def test_step06_duration_perturbations_are_evaluated_not_marked_unsupported(project_root):
    result = run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=1, time_points=40, write_outputs=False)
    )
    perturb = result["perturbation_sweeps"]
    assert "perturbation_status" in perturb.columns
    duration = perturb[perturb["perturbation"].astype(str).str.startswith("stimulus_duration")]
    assert not duration.empty
    assert duration["perturbation_status"].eq("evaluated").all()
    assert duration["simulation_status"].eq("ok").all()
    assert "Vm_feature_pass_fraction" in perturb.columns
    assert "hidden_flux_plausible" in perturb.columns


def test_step06_notebook_executes_and_saves_auditable_copy(project_root):
    executed = execute_notebook(
        project_root / "analysis" / "06_predictive_validation_and_perturbation.ipynb",
        project_root,
    )
    assert executed.exists()
