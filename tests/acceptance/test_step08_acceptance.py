from __future__ import annotations

from src.step08_parameter_plausibility import EFFECTIVE_PARAMETERS, Step08Config, run_step08_parameter_plausibility


def test_step08_writes_required_outputs(project_root):
    out_dir = project_root / "outputs" / "step08_acceptance"
    result = run_step08_parameter_plausibility(
        project_root,
        Step08Config(max_candidates=2, write_outputs=True),
        output_dir=out_dir,
    )
    for filename in [
        "parameter_range_audit.csv",
        "effective_parameter_plausibility.csv",
        "constrained_rerun_comparison.csv",
        "interpretability_status.csv",
        "analysis_summary.json",
    ]:
        assert (out_dir / filename).exists()
    assert result["analysis_summary"]["n_parameter_rows"] == len(result["parameter_range_audit"])


def test_step08_effective_output_contains_only_effective_coordinates(project_root):
    result = run_step08_parameter_plausibility(
        project_root,
        Step08Config(max_candidates=2, write_outputs=False),
    )
    effective = result["effective_parameter_plausibility"]
    assert set(effective["parameter"]).issubset(set(EFFECTIVE_PARAMETERS))
    assert not effective.empty
    assert "coordinate_type" not in effective.columns


def test_step08_final_degeneracy_claims_remain_disabled(project_root):
    result = run_step08_parameter_plausibility(
        project_root,
        Step08Config(max_candidates=2, write_outputs=False),
    )
    status = result["interpretability_status"]
    assert "final_degeneracy_claim_allowed_after_step08" in status.columns
    assert not status["final_degeneracy_claim_allowed_after_step08"].astype(bool).any()
    assert result["analysis_summary"]["final_degeneracy_claim_allowed_after_step08"] is False
