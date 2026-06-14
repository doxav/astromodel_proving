"""Step 09 reviewer-response synthesis tables.

This step does not create new biological evidence.  It integrates the prior
step outputs into reviewer-facing traceability, claim-maturity, and manuscript
asset tables so unsupported claims remain visibly downgraded.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_SUBDIR = "reviewer_synthesis"


@dataclass(slots=True)
class Step09Config:
    """Configuration for Step 09 reviewer synthesis."""

    write_outputs: bool = True


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _support_from_bool(value: bool, partial: bool = False) -> str:
    if value:
        return "supported"
    return "partial" if partial else "unresolved"


def build_reviewer_traceability_table(project_root: Path | str) -> pd.DataFrame:
    """Map reviewer critiques R1-R7 to concrete step outputs and statuses."""

    root = Path(project_root).resolve()
    step03 = _read_csv(root / "outputs" / "identifiability" / "profile_summary.csv")
    step05_claims = _read_csv(root / "outputs" / "mechanisms" / "claim_scope_table.csv")
    step06 = _read_csv(root / "outputs" / "predictive_validation" / "robustness_summary.csv")
    step07 = _read_csv(root / "outputs" / "assumption_sensitivity" / "claim_scope_table.csv")
    step08 = _read_csv(root / "outputs" / "parameter_plausibility" / "interpretability_status.csv")
    atf = _read_csv(root / "outputs" / "features" / "feature_table_by_sweep.csv")
    provenance = _read_csv(root / "outputs" / "provenance" / "atf_region_condition_inventory.csv")

    step06_supported = (
        not step06.empty and step06["validation_label"].astype(str).eq("predictive_supported").any()
    )
    assumption_unresolved = (
        step07.empty
        or step07["status"].astype(str).str.contains("needed|insufficient|sensitive|dependent", regex=True).any()
    )
    final_degeneracy_allowed = (
        not step08.empty
        and "final_degeneracy_claim_allowed_after_step08" in step08
        and step08["final_degeneracy_claim_allowed_after_step08"].astype(bool).any()
    )

    rows = [
        {
            "reviewer_id": "R1",
            "critique": "degeneracy versus non-identifiability/sloppiness",
            "primary_outputs": "outputs/identifiability; outputs/mechanisms; outputs/reviewer_synthesis",
            "evidence_status": _support_from_bool(not step03.empty, partial=not step05_claims.empty),
            "claim_boundary": "Degeneracy wording remains disabled unless Step 06 and Step 09 synthesis support it.",
        },
        {
            "reviewer_id": "R2",
            "critique": "experimental variability, noise, and data constraints",
            "primary_outputs": "outputs/provenance; outputs/features",
            "evidence_status": _support_from_bool(
                not atf.empty and atf.get("file_id", pd.Series(dtype=object)).nunique() >= 37
            ),
            "claim_boundary": "Cell/file is the independent unit; no paired animal-level claims are made.",
        },
        {
            "reviewer_id": "R3",
            "critique": "model assumptions and proxy validity",
            "primary_outputs": "outputs/assumption_sensitivity",
            "evidence_status": "partial" if assumption_unresolved else "supported",
            "claim_boundary": "Explicit ECS/proxy variants remain required when proxy rows are limited.",
        },
        {
            "reviewer_id": "R4",
            "critique": "Vm-only fits and physiological parameter plausibility",
            "primary_outputs": "outputs/identifiability; outputs/parameter_plausibility",
            "evidence_status": _support_from_bool(not step08.empty, partial=not step03.empty),
            "claim_boundary": "Raw parameters are interpreted only when range and identifiability guardrails pass.",
        },
        {
            "reviewer_id": "R5",
            "critique": "pathways, mechanisms, and phenotypes",
            "primary_outputs": "outputs/mechanisms; outputs/predictive_validation",
            "evidence_status": "supported" if step06_supported else ("partial" if not step05_claims.empty else "unresolved"),
            "claim_boundary": "Phenotype tags are provisional until prediction and perturbation support mature.",
        },
        {
            "reviewer_id": "R6",
            "critique": "held-out prediction and perturbation robustness",
            "primary_outputs": "outputs/predictive_validation",
            "evidence_status": "supported" if step06_supported else ("partial" if not step06.empty else "unresolved"),
            "claim_boundary": "Prediction-limited or fit-only clusters are not biological degeneracy evidence.",
        },
        {
            "reviewer_id": "R7",
            "critique": "clarity, organization, units, and figure traceability",
            "primary_outputs": "outputs/reviewer_synthesis/manuscript_asset_manifest.csv",
            "evidence_status": _support_from_bool(not provenance.empty, partial=True),
            "claim_boundary": "Figures/tables must cite source step outputs and conservative claim scope.",
        },
    ]
    out = pd.DataFrame(rows)
    out["final_biological_degeneracy_claim_allowed"] = bool(final_degeneracy_allowed and step06_supported and not assumption_unresolved)
    return out


def build_claim_maturity_table(project_root: Path | str) -> pd.DataFrame:
    """Build claim-level maturity statuses from actual step outputs."""

    root = Path(project_root).resolve()
    step05_summary = _read_json(root / "outputs" / "mechanisms" / "analysis_summary.json")
    step06 = _read_csv(root / "outputs" / "predictive_validation" / "robustness_summary.csv")
    step07 = _read_csv(root / "outputs" / "assumption_sensitivity" / "claim_scope_table.csv")
    step08 = _read_csv(root / "outputs" / "parameter_plausibility" / "interpretability_status.csv")
    phenotype_gate = _read_csv(root / "outputs" / "predictive_validation" / "phenotype_robustness_summary.csv")
    stratum_gate = _read_csv(root / "outputs" / "reviewer_synthesis" / "stratum_support_gate.csv")
    assumption_gate = _read_csv(root / "outputs" / "reviewer_synthesis" / "assumption_gate_audit.csv")
    parameter_class = _read_csv(root / "outputs" / "parameter_plausibility" / "parameter_interpretation_class_audit.csv")
    restricted_join = _read_csv(root / "outputs" / "reviewer_synthesis" / "restricted_all_gate_join.csv")

    predictive_supported = (
        not step06.empty and step06["validation_label"].astype(str).eq("predictive_supported").any()
    )
    assumptions_clear = (
        not step07.empty
        and not step07["status"].astype(str).str.contains("needed|insufficient|sensitive|dependent", regex=True).any()
    )
    if not assumption_gate.empty and "gate_pass" in assumption_gate:
        assumptions_clear = bool(assumption_gate["gate_pass"].astype(bool).all())
    phenotype_restricted_support = (
        not phenotype_gate.empty
        and "phenotype_support_gate" in phenotype_gate
        and phenotype_gate["phenotype_support_gate"].astype(str).eq("pass").any()
    )
    stratum_restricted_support = (
        not stratum_gate.empty
        and "support_status" in stratum_gate
        and stratum_gate["support_status"].astype(str).isin(
            {"all_groups_supported", "restricted_supported_groups_only"}
        ).any()
    )
    direct_parameter_rows = (
        int(parameter_class["physiology_claim_allowed_after_semantic_filter"].astype(bool).sum())
        if not parameter_class.empty and "physiology_claim_allowed_after_semantic_filter" in parameter_class
        else 0
    )
    effective_parameter_rows = (
        int(parameter_class["effective_coordinate_claim_allowed_after_semantic_filter"].astype(bool).sum())
        if not parameter_class.empty and "effective_coordinate_claim_allowed_after_semantic_filter" in parameter_class
        else 0
    )
    restricted_final_supported = (
        not restricted_join.empty
        and "restricted_degeneracy_claim_allowed" in restricted_join
        and restricted_join["restricted_degeneracy_claim_allowed"].astype(bool).any()
    )
    mean_bio_score = (
        float(pd.to_numeric(step06["biological_description_score"], errors="coerce").mean(skipna=True))
        if "biological_description_score" in step06
        else np.nan
    )
    mechanism_maturity = str(step05_summary.get("cluster_evidence_status", "unresolved"))
    mechanism_remaining = "requires Step 06 predictive/perturbation support before stronger wording"
    if phenotype_restricted_support and stratum_restricted_support:
        mechanism_maturity = "restricted_model_phenotype_support"
        mechanism_remaining = "biological pathway wording still requires assumption, parameter, and external-validation gates"
    parameter_maturity = "partial" if not step08.empty else "unresolved"
    parameter_remaining = "raw coordinates require both range and identifiability support"
    if not parameter_class.empty:
        if direct_parameter_rows > 0:
            parameter_maturity = "restricted_direct_parameter_support"
            parameter_remaining = "direct parameter wording limited to cited, identifiable semantic classes"
        elif effective_parameter_rows > 0:
            parameter_maturity = "effective_coordinate_only"
            parameter_remaining = "direct physiological parameter wording remains blocked by semantics or identifiability"
        else:
            parameter_maturity = "semantic_and_identifiability_blocked"
            parameter_remaining = "current ranges are guardrails and fitted coordinates are not direct physiological parameters"
    rows = [
        {
            "claim": "accepted cell-specific six-sweep ensembles exist",
            "maturity": "supported",
            "basis": "Step 04 accepted ensemble and held-out-current artifacts",
            "remaining_requirement": "increase accepted/reviewer-facing cell coverage if strata remain sparse",
        },
        {
            "claim": "candidate mechanism regimes are biologically interpretable",
            "maturity": mechanism_maturity,
            "basis": "Step 05 mechanisms plus Step 06 phenotype robustness and Step 09 stratum gates",
            "remaining_requirement": mechanism_remaining,
        },
        {
            "claim": "mechanism or phenotype labels are predictive under perturbation",
            "maturity": "supported" if predictive_supported else "partial_or_unresolved",
            "basis": "Step 06 robustness labels and biological_description_score",
            "remaining_requirement": "broaden support across region, condition, mechanism clusters, and all accepted cells",
        },
        {
            "claim": "model assumptions do not drive the conclusion",
            "maturity": "supported" if assumptions_clear else "quantified_model_dependent_or_unresolved",
            "basis": "Step 07 sensitivity plus explicit Step 09 assumption gate audit",
            "remaining_requirement": "explicit ECS variant or additional data remains required while the proxy gate fails",
        },
        {
            "claim": "accepted parameters are physiologically interpretable",
            "maturity": parameter_maturity,
            "basis": "Step 08 range/identifiability plus semantic interpretation-class audit",
            "remaining_requirement": parameter_remaining,
        },
        {
            "claim": "final biological degeneracy wording is allowed",
            "maturity": "restricted_supported" if restricted_final_supported else "not_allowed_yet",
            "basis": "Integrated Step 03-08 synthesis plus restricted all-gate join",
            "remaining_requirement": "requires mechanism distinction, predictive support, assumption robustness, parameter plausibility, and K_o endpoint support together",
        },
    ]
    out = pd.DataFrame(rows)
    out["mean_biological_description_score"] = mean_bio_score
    return out


def build_manuscript_asset_manifest(project_root: Path | str) -> pd.DataFrame:
    """List reviewer-facing artifacts and their source steps."""

    root = Path(project_root).resolve()
    assets = [
        ("Step 00", "outputs/provenance/atf_region_condition_inventory.csv", "ATF data provenance and region/condition audit"),
        ("Step 02", "outputs/features/condition_region_sweep_thresholds.csv", "Region-aware feature thresholds"),
        ("Step 03", "outputs/identifiability/effective_parameter_map.csv", "Effective-parameter and identifiability guardrails"),
        ("Step 04", "outputs/cell_fits/accepted_cell_ensembles.csv", "Accepted six-sweep cell ensembles"),
        ("Step 04", "outputs/cell_fits/cell_fit_candidates.csv", "Full candidate history for audit"),
        ("Step 05", "outputs/mechanisms/accepted_fit_mechanisms.csv", "Hidden-current flux decomposition"),
        ("Step 05", "outputs/mechanisms/accepted_fit_mechanisms_windowed.csv", "Windowed local/spatial mechanism characterization"),
        ("Step 05", "outputs/mechanisms/buffering_phenotype_tags.csv", "Provisional phenotype tags"),
        ("Step 06", "outputs/predictive_validation/robustness_summary.csv", "Prediction and perturbation robustness"),
        ("Step 07", "outputs/assumption_sensitivity/claim_scope_table.csv", "Assumption-sensitivity claim scope"),
        ("Step 08", "outputs/parameter_plausibility/interpretability_status.csv", "Parameter interpretability guardrails"),
        ("Step 09", "outputs/reviewer_synthesis/reviewer_traceability_table.csv", "R1-R7 response map"),
        ("Selected action 1", "outputs/predictive_validation/phenotype_robustness_summary.csv", "Phenotype robustness by mechanism/stratum"),
        ("Selected action 1", "outputs/reviewer_synthesis/stratum_support_gate.csv", "Region-condition support gates"),
        ("Selected action 1", "outputs/predictive_validation/prediction_limited_failure_modes.csv", "Prediction-limited failure decomposition"),
        ("Selected action 1", "outputs/reviewer_synthesis/assumption_gate_audit.csv", "Assumption pass/fail gate audit"),
        ("Selected action 1", "outputs/assumption_sensitivity/proxy_exclusion_claim_sensitivity.csv", "Proxy-exclusion claim sensitivity"),
        ("Selected action 1", "outputs/parameter_plausibility/parameter_semantics_audit.csv", "Parameter semantic-class audit"),
        ("Selected action 1", "outputs/parameter_plausibility/full_accepted_parameter_audit.csv", "Full accepted ensemble parameter audit"),
        ("Selected action 1", "outputs/parameter_plausibility/parameter_interpretation_class_audit.csv", "Parameter interpretation-class audit"),
        ("Selected action 1", "outputs/parameter_plausibility/constrained_failure_modes.csv", "Constrained-screen failure modes"),
        ("Selected action 1", "outputs/reviewer_synthesis/integrated_degeneracy_gate_matrix.csv", "Integrated degeneracy gate matrix"),
        ("Selected action 1", "outputs/reviewer_synthesis/degeneracy_level_table.csv", "Degeneracy wording level table"),
        ("Selected action 1", "outputs/reviewer_synthesis/restricted_validation_claims.csv", "Restricted validation claim wording"),
        ("Selected action 1", "outputs/reviewer_synthesis/restricted_all_gate_join.csv", "All-gate restricted claim join"),
        ("Selected action 1", "outputs/predictive_validation/K_o_homeostasis_endpoint_audit.csv", "K_o homeostasis endpoint audit"),
        ("Selected action 1", "outputs/reviewer_synthesis/claim_to_artifact_ledger.csv", "Claim-to-artifact wording ledger"),
        ("Selected action 2", "outputs/mechanisms/intercluster_interpolation_acceptance.csv", "Intercluster interpolation feature-contract screen"),
        ("Selected action 2", "outputs/mechanisms/phenotype_threshold_sensitivity.csv", "Phenotype threshold sensitivity"),
        ("Selected action 2", "outputs/assumption_sensitivity/all_current_assumption_sensitivity.csv", "All-current assumption sensitivity"),
        ("Selected action 2", "outputs/parameter_plausibility/cell_specific_identifiability_audit.csv", "Cell-specific accepted-ensemble identifiability audit"),
        ("Selected action 2", "outputs/parameter_plausibility/parameter_ranges_citation_audit.csv", "Parameter range citation/basis audit"),
        ("Selected action meta", "outputs/reviewer_synthesis/selected_action_scientific_value_assessment.csv", "Scientific value assessment for selected-action artifacts"),
        ("Selected action meta", "outputs/reviewer_synthesis/selected_action_results_summary.csv", "Selected-action result summary and upgrade screen"),
        ("Selected action meta", "outputs/reviewer_synthesis/notebook_update_screen_after_selected_actions.csv", "Notebook rerun/comment update screen after selected actions"),
    ]
    rows = []
    for step, rel_path, purpose in assets:
        path = root / rel_path
        rows.append(
            {
                "step": step,
                "artifact": rel_path,
                "purpose": purpose,
                "exists": path.exists(),
                "claim_scope": "reviewer_facing_source_table" if path.exists() else "missing_or_pending",
            }
        )
    return pd.DataFrame(rows)


def run_step09_reviewer_synthesis(
    project_root: Path | str,
    config: Step09Config | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run Step 09 and optionally write synthesis artifacts."""

    root = Path(project_root).resolve()
    cfg = config or Step09Config()
    out_dir = _ensure_dir(
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "outputs" / OUTPUT_SUBDIR
    )
    start = time.perf_counter()
    traceability = build_reviewer_traceability_table(root)
    maturity = build_claim_maturity_table(root)
    if cfg.write_outputs:
        traceability.to_csv(out_dir / "reviewer_traceability_table.csv", index=False)
        maturity.to_csv(out_dir / "claim_maturity_table.csv", index=False)
    manifest = build_manuscript_asset_manifest(root)
    final_allowed = bool(traceability["final_biological_degeneracy_claim_allowed"].all())
    summary = {
        "step_name": "Step 09 - reviewer-response synthesis",
        "config": asdict(cfg),
        "n_reviewer_rows": int(len(traceability)),
        "n_claim_rows": int(len(maturity)),
        "n_manifest_rows": int(len(manifest)),
        "missing_manifest_artifacts": manifest.loc[~manifest["exists"].astype(bool), "artifact"].astype(str).tolist(),
        "final_biological_degeneracy_claim_allowed": final_allowed,
        "headline_claim_scope": "Final degeneracy wording is allowed only if all upstream evidence layers are supported.",
        "elapsed_seconds": time.perf_counter() - start,
    }
    if cfg.write_outputs:
        manifest.to_csv(out_dir / "manuscript_asset_manifest.csv", index=False)
        (out_dir / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return {
        "reviewer_traceability_table": traceability,
        "claim_maturity_table": maturity,
        "manuscript_asset_manifest": manifest,
        "analysis_summary": summary,
    }
