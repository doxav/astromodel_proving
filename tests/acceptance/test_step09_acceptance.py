from __future__ import annotations

import json

from src.step09_reviewer_synthesis import Step09Config, run_step09_reviewer_synthesis


def test_step09_writes_reviewer_synthesis_outputs(project_root):
    out_dir = project_root / "outputs" / "step09_acceptance"
    result = run_step09_reviewer_synthesis(
        project_root,
        Step09Config(write_outputs=True),
        output_dir=out_dir,
    )
    for filename in [
        "reviewer_traceability_table.csv",
        "claim_maturity_table.csv",
        "reviewer_remark_artifact_links.csv",
        "mechanistic_pathway_perturbation_gate.csv",
        "legacy_perturbation_claim_gate.csv",
        "degeneracy_scientific_value_statement.csv",
        "manuscript_asset_manifest.csv",
        "analysis_summary.json",
    ]:
        assert (out_dir / filename).exists()
    traceability = result["reviewer_traceability_table"]
    assert set(traceability["reviewer_id"]) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}
    links = result["reviewer_remark_artifact_links"]
    assert set(links["reviewer_id"]) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}
    assert {"artifact", "notebook", "cell_reference", "impact_rank"}.issubset(links.columns)
    assert list(links.columns[:3]) == [
        "reviewer_id",
        "impact_rank",
        "usefulness_rationale",
    ]
    value = result["degeneracy_scientific_value_statement"]
    assert value["current_status"].iloc[0] in {"allowed", "not_biologically_proven"}
    summary = json.loads((out_dir / "analysis_summary.json").read_text())
    assert summary["n_reviewer_rows"] == 7


def test_step09_final_degeneracy_claim_not_enabled_without_all_layers(project_root):
    result = run_step09_reviewer_synthesis(
        project_root,
        Step09Config(write_outputs=False),
    )
    summary = result["analysis_summary"]
    maturity = result["claim_maturity_table"]
    final_row = maturity[maturity["claim"].eq("final biological degeneracy wording is allowed")]
    assert not summary["final_biological_degeneracy_claim_allowed"]
    assert not final_row.empty
    assert final_row["maturity"].iloc[0] in {"not_allowed_yet", "supported"}
