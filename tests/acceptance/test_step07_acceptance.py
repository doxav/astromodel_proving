from __future__ import annotations

import json

from src.step07_assumption_sensitivity import Step07Config, run_step07_assumption_sensitivity


def test_step07_writes_required_outputs(project_root):
    out_dir = project_root / "outputs" / "step07_acceptance"
    result = run_step07_assumption_sensitivity(
        project_root,
        Step07Config(max_candidates=1, time_points=25, write_outputs=True),
        output_dir=out_dir,
    )
    required = {
        "model_comparison.csv",
        "gating_family_comparison.csv",
        "proxy_validity_by_ensemble.csv",
        "compartment_split_sensitivity.csv",
        "claim_scope_table.csv",
        "analysis_summary.json",
    }
    assert required.issubset({p.name for p in out_dir.iterdir()})
    summary = json.loads((out_dir / "analysis_summary.json").read_text())
    assert summary["n_gating_rows"] == len(result["gating_family_comparison"])


def test_step07_model_comparison_contains_same_contract_metrics(project_root):
    result = run_step07_assumption_sensitivity(
        project_root, Step07Config(max_candidates=1, time_points=25, write_outputs=False)
    )
    model = result["model_comparison"]
    required = {
        "model_family",
        "assumption_axis",
        "median_step04_trace_rmse_mV",
        "mean_heldout_pass_fraction",
        "same_split_same_loss_contract",
        "mechanism_stability_fraction",
        "claim_scope",
    }
    assert required.issubset(model.columns)
    assert model["same_split_same_loss_contract"].eq("step07_same_candidates_currents_timegrid_loss_v1").all()


def test_step07_claim_scope_never_enables_final_degeneracy(project_root):
    result = run_step07_assumption_sensitivity(
        project_root, Step07Config(max_candidates=2, time_points=25, write_outputs=False)
    )
    claims = result["claim_scope_table"]
    assert "final_degeneracy_claim_allowed_after_step07" in claims.columns
    assert not claims["final_degeneracy_claim_allowed_after_step07"].astype(bool).any()
