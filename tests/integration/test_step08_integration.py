from __future__ import annotations

from tests._notebook import execute_notebook
from src.astro_model import VALID_CURRENTS
from src.step08_parameter_plausibility import (
    IDENTITY_COLUMNS,
    Step08Config,
    build_constrained_rerun_comparison,
    build_parameter_range_audit,
    load_step08_inputs,
    merge_candidate_annotations,
    run_step08_parameter_plausibility,
)


def test_step08_config_defaults_to_all_valid_currents():
    """Ensure the default configuration covers the full F-I protocol rather than a single step."""
    config = Step08Config()

    assert len(config.currents_na) == len(VALID_CURRENTS), "Config should map all valid currents."
    assert set(config.currents_na) == set(int(c) for c in VALID_CURRENTS), "Config currents mismatch."


def test_step08_constrained_comparison_runs_multiple_currents(project_root):
    """Ensure that for each candidate, a rescore comparison is made per current sweep."""
    config = Step08Config(
        max_candidates=1,
        constrained_max_candidates=1,
        time_points=10,
        write_outputs=False,
    )

    inputs = load_step08_inputs(project_root, config)
    candidates = merge_candidate_annotations(inputs)
    audit = build_parameter_range_audit(candidates, inputs, config)

    constrained = build_constrained_rerun_comparison(candidates, audit, config)

    n_expected_rows = len(config.currents_na)
    assert len(constrained) == n_expected_rows, "A rerun row must be generated for each current."
    assert set(constrained["current_na"]) == set(config.currents_na), "All currents must be simulated."


def test_step08_candidate_status_coherent_with_parameter_audit(project_root):
    result = run_step08_parameter_plausibility(project_root, Step08Config(max_candidates=2, write_outputs=False))
    audit = result["parameter_range_audit"]
    status = result["interpretability_status"]
    audit_keys = set(zip(*(audit[col] for col in IDENTITY_COLUMNS)))
    status_keys = set(zip(*(status[col] for col in IDENTITY_COLUMNS)))
    assert status_keys.issubset(audit_keys)
    counts = audit.groupby(IDENTITY_COLUMNS).size().rename("expected").reset_index()
    merged = status.merge(counts, on=IDENTITY_COLUMNS, how="left")
    assert (merged["n_parameters_audited"] == merged["expected"]).all()


def test_step08_constrained_comparison_has_provenance_and_status(project_root):
    result = run_step08_parameter_plausibility(project_root, Step08Config(max_candidates=2, write_outputs=False))
    constrained = result["constrained_rerun_comparison"]
    assert {
        "constrained_screen_type",
        "changed_by_constraints",
        "changed_parameters",
        "prediction_persists_under_constraints",
        "mechanism_persists_under_constraints",
        "constrained_claim_status",
    }.issubset(constrained.columns)
    assert constrained["constrained_screen_type"].eq("broad_range_projection_not_full_optimizer").all()
    assert constrained["constrained_claim_status"].isin({
        "claim_persists_under_broad_constraints",
        "claim_downgraded_by_constraint_screen",
    }).all()


def test_step08_notebook_executes_and_saves_auditable_copy(project_root):
    executed = execute_notebook(
        project_root / "analysis" / "08_parameter_plausibility_and_constrained_reruns.ipynb",
        project_root,
    )
    assert executed.exists()
