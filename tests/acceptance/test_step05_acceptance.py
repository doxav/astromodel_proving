from __future__ import annotations

import json

from src.step05_mechanistic_decomposition import (
    Step05Config,
    run_step05_mechanistic_decomposition,
)


def test_step05_writes_required_outputs_and_conservative_claims(project_root):
    out_dir = project_root / "outputs" / "step05_acceptance"
    result = run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(
            max_candidates=2, time_points=60, bootstrap_iterations=2, write_outputs=True
        ),
        output_dir=out_dir,
    )
    required_files = {
        "accepted_fit_mechanisms.csv",
        "mechanism_clusters.csv",
        "representatives.csv",
        "region_mechanism_enrichment.csv",
        "geometry_classification.csv",
        "bootstrap_cluster_stability.csv",
        "claim_scope_table.csv",
        "analysis_summary.json",
    }
    assert required_files.issubset({p.name for p in out_dir.iterdir()})
    clusters = result["mechanism_clusters"]
    assert {"region", "condition", "mechanism_cluster", "cluster_claim_scope"}.issubset(
        clusters.columns
    )
    claims = result["claim_scope_table"]
    assert (
        not claims["allowed_pre_step06_claim"]
        .str.contains("candidate_degenerate_regimes")
        .any()
    )
    assert (
        claims["forbidden_pre_step06_claim"]
        .str.contains("candidate_degenerate_regimes")
        .any()
    )
    summary = json.loads((out_dir / "analysis_summary.json").read_text())
    assert summary["n_successful_sweep_simulations"] >= 6


def test_step05_small_ensemble_is_downgraded_not_reported_as_candidate_regime(
    project_root,
):
    result = run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(
            max_candidates=2,
            time_points=60,
            bootstrap_iterations=2,
            write_outputs=False,
        ),
    )
    clusters = result["mechanism_clusters"]
    claims = result["claim_scope_table"]
    summary = result["analysis_summary"]

    assert clusters["mechanism_cluster"].nunique() == 1
    assert clusters["cluster_evidence_status"].eq("insufficient_evidence").all()
    assert summary["cluster_evidence_status"] == "insufficient_evidence"
    assert (
        claims.loc[
            claims["claim_topic"] == "mechanism_diversity", "allowed_pre_step06_claim"
        ].iloc[0]
        == "insufficient_evidence_for_candidate_mechanism_regimes"
    )
    assert (
        not claims["allowed_pre_step06_claim"]
        .str.contains("candidate_degenerate_regimes")
        .any()
    )


def test_step05_representatives_preserve_step04_acceptance(project_root):
    result = run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(
            max_candidates=3,
            time_points=60,
            bootstrap_iterations=0,
            write_outputs=False,
        ),
    )
    reps = result["representatives"]
    assert not reps.empty
    assert reps["accepted"].astype(bool).all()
    assert not reps["claim_scope"].str.contains("candidate_degenerate_regimes").any()
