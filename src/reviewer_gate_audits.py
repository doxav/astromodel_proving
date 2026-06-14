"""Reviewer-response gate audits for selected immediate and modest actions.

The builders in this module do not rerun Step 04 optimizers or create new
experimental evidence. They derive conservative gate tables from the current
Step 04-08 outputs so Step 09 can decide which reviewer-facing claims can be
restricted, upgraded, or must remain blocked.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.integrate import ODEintWarning

from .astro_model import VALID_CURRENTS, simulate_with_hidden_outputs
from .atf_features import FEATURE_COLUMNS, extract_features_from_trace
from .parameter_space import EFFECTIVE_COORDINATES
from .phenotype_classifier import classify_buffering_phenotype
from .protocols import protocol_condition, stim_window_seconds
from .step05_mechanistic_decomposition import (
    EFFECTIVE_COLUMNS,
    IDENTITY_COLUMNS,
    reconstruct_flat_params,
)
from .step06_predictive_validation import Step06Config, load_step06_inputs
from .step07_assumption_sensitivity import (
    Step07Config,
    build_compartment_split_sensitivity,
    build_gating_family_comparison,
    build_proxy_validity,
    load_step07_inputs,
)
from .step08_parameter_plausibility import (
    AUDITED_PARAMETERS,
    Step08Config,
    build_parameter_range_audit,
    default_parameter_ranges,
    load_identifiability_status,
)


@dataclass(slots=True)
class ReviewerGateAuditConfig:
    """Configuration for derived reviewer gate audits.

    ``max_candidates`` is intended for tests and quick local checks. The default
    reviewer-facing run uses all current output rows.
    """

    max_candidates: int | None = None
    write_outputs: bool = True
    min_reviewer_facing_cells: int = 3
    min_holdout_pass_fraction: float = 0.30
    min_perturbation_robust_fraction: float = 0.50
    min_ppc_coverage: float = 0.30
    min_homeostasis_pass_fraction: float = 0.50
    max_proxy_limited_fraction: float = 0.25
    max_gating_unstable_fraction: float = 0.25
    max_split_sensitive_fraction: float = 0.25
    all_current_time_points: int = 40
    interpolation_time_points: int = 40
    interpolation_points: int = 5


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_fraction(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(_bool_series(series).mean())


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean(skipna=True)) if values.notna().any() else float("nan")


def _finite_min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.min(skipna=True)) if numeric.notna().any() else float("nan")


def _finite_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max(skipna=True)) if numeric.notna().any() else float("nan")


def _join_unique(values: pd.Series, limit: int = 8) -> str:
    unique = sorted({str(v) for v in values.dropna().astype(str) if str(v)})
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";+{len(unique) - limit}_more"
    return ";".join(unique)


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "predictive": root / "outputs" / "predictive_validation",
        "reviewer": root / "outputs" / "reviewer_synthesis",
        "assumption": root / "outputs" / "assumption_sensitivity",
        "parameter": root / "outputs" / "parameter_plausibility",
        "mechanisms": root / "outputs" / "mechanisms",
    }


def build_selected_action_strategy_comparison() -> pd.DataFrame:
    """Compare implementation strategies considered for the selected actions."""

    rows = [
        {
            "strategy": "current_output_gate_join",
            "actions_covered": "phase_1",
            "description": "Reuse Step04-08 outputs to build explicit support/blocker gates without any new simulations.",
            "scientific_strength": "high_for_traceability_medium_for_new_support",
            "runtime_risk": "low",
            "claim_risk": "low",
            "selected": True,
            "selection_reason": "Best first pass because every result is auditable and cannot overfit new analyses.",
        },
        {
            "strategy": "modest_no_optimizer_sensitivity",
            "actions_covered": "phase_2",
            "description": "Run bounded resimulation or threshold sweeps without Step04 refitting.",
            "scientific_strength": "medium",
            "runtime_risk": "medium",
            "claim_risk": "low_medium",
            "selected": True,
            "selection_reason": "Improves evidence on label stability and assumptions while preserving notebook runtime.",
        },
        {
            "strategy": "full_refit_or_new_structural_model",
            "actions_covered": "future_phase_3",
            "description": "Rerun constrained fits, alternative gating refits, explicit ECS variants, or omitted-process models.",
            "scientific_strength": "highest_if_successful",
            "runtime_risk": "high",
            "claim_risk": "medium_high",
            "selected": False,
            "selection_reason": "Deferred because it can break runtime and should be targeted after current gates identify promising strata.",
        },
    ]
    return pd.DataFrame(rows)


def build_phenotype_robustness_summary(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Summarize Step 06 robustness by model-derived phenotype."""

    perturb = _read_csv(root / "outputs" / "predictive_validation" / "perturbation_sweeps.csv")
    heldout = _read_csv(root / "outputs" / "predictive_validation" / "heldout_current_errors.csv")
    if perturb.empty:
        return pd.DataFrame()

    if config.max_candidates is not None:
        keep = perturb[IDENTITY_COLUMNS].drop_duplicates().head(int(config.max_candidates))
        perturb = perturb.merge(keep, on=IDENTITY_COLUMNS, how="inner")
        heldout = heldout.merge(keep, on=IDENTITY_COLUMNS, how="inner") if not heldout.empty else heldout

    rows: list[dict[str, Any]] = []
    group_cols = ["buffering_phenotype", "mechanism_cluster", "region", "condition"]
    for keys, group in perturb.groupby(group_cols, dropna=False):
        phenotype, cluster, region, condition = keys
        non_nominal = group[
            group["perturbation"].astype(str).ne("nominal")
            & group["perturbation_status"].astype(str).eq("evaluated")
        ]
        h = heldout[
            heldout["mechanism_cluster"].astype(str).eq(str(cluster))
            & heldout["region"].astype(str).eq(str(region))
            & heldout["condition"].astype(str).eq(str(condition))
        ] if not heldout.empty else pd.DataFrame()
        perturbation_robust_fraction = _safe_fraction(non_nominal.get("functional_buffering_pass", pd.Series(dtype=bool)))
        holdout_pass_fraction = _safe_fraction(h.get("prediction_pass", pd.Series(dtype=bool)))
        vm_feature_pass_fraction = _safe_mean(non_nominal, "Vm_feature_pass_fraction")
        hidden_flux_plausible_fraction = _safe_fraction(non_nominal.get("hidden_flux_plausible", pd.Series(dtype=bool)))
        n_cells = int(group["file_id"].nunique()) if "file_id" in group else 0
        supported = bool(
            n_cells >= config.min_reviewer_facing_cells
            and np.isfinite(holdout_pass_fraction)
            and holdout_pass_fraction >= config.min_holdout_pass_fraction
            and np.isfinite(perturbation_robust_fraction)
            and perturbation_robust_fraction >= config.min_perturbation_robust_fraction
            and np.isfinite(vm_feature_pass_fraction)
            and vm_feature_pass_fraction >= config.min_ppc_coverage
        )
        rows.append(
            {
                "buffering_phenotype": phenotype,
                "mechanism_cluster": cluster,
                "region": region,
                "condition": condition,
                "n_candidates": int(group["candidate_id"].nunique()),
                "n_cells": n_cells,
                "n_perturbation_rows": int(len(non_nominal)),
                "holdout_pass_fraction": holdout_pass_fraction,
                "perturbation_robust_fraction": perturbation_robust_fraction,
                "perturbation_vm_feature_pass_fraction": vm_feature_pass_fraction,
                "hidden_flux_plausible_fraction": hidden_flux_plausible_fraction,
                "phenotype_support_gate": "pass" if supported else "fail",
                "claim_scope": (
                    "restricted_model_phenotype_supported_by_current_outputs"
                    if supported
                    else "phenotype_label_remains_provisional_or_prediction_limited"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_stratum_support_gate(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Gate region-condition mechanism claims by Step 04 support and Step 06 labels."""

    quality = _read_csv(root / "outputs" / "cell_fits" / "cell_fit_quality_summary.csv")
    inventory = _read_csv(root / "outputs" / "cell_fits" / "cell_trace_inventory.csv")
    accepted = _read_csv(root / "outputs" / "cell_fits" / "accepted_cell_ensembles.csv")
    robust = _read_csv(root / "outputs" / "predictive_validation" / "robustness_summary.csv")
    strata = inventory[["region", "condition"]].drop_duplicates() if not inventory.empty else robust[["region", "condition"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for _, stratum in strata.sort_values(["region", "condition"]).iterrows():
        region = str(stratum["region"])
        condition = str(stratum["condition"])
        q = quality[quality["region"].astype(str).eq(region) & quality["condition"].astype(str).eq(condition)] if not quality.empty else pd.DataFrame()
        inv = inventory[inventory["region"].astype(str).eq(region) & inventory["condition"].astype(str).eq(condition)] if not inventory.empty else pd.DataFrame()
        acc = accepted[accepted["region"].astype(str).eq(region) & accepted["condition"].astype(str).eq(condition)] if not accepted.empty else pd.DataFrame()
        rs = robust[robust["region"].astype(str).eq(region) & robust["condition"].astype(str).eq(condition)] if not robust.empty else pd.DataFrame()
        n_reviewer = int(_bool_series(q["cell_reviewer_facing"]).sum()) if "cell_reviewer_facing" in q else 0
        n_predictive = int(rs["validation_label"].astype(str).eq("predictive_supported").sum()) if not rs.empty else 0
        n_limited = int(rs["validation_label"].astype(str).ne("predictive_supported").sum()) if not rs.empty else 0
        minimum_cell_gate = n_reviewer >= int(config.min_reviewer_facing_cells)
        any_predictive_gate = n_predictive > 0
        all_group_gate = n_limited == 0 and n_predictive > 0
        if minimum_cell_gate and all_group_gate:
            support_status = "all_groups_supported"
        elif minimum_cell_gate and any_predictive_gate:
            support_status = "restricted_supported_groups_only"
        elif minimum_cell_gate:
            support_status = "coverage_present_prediction_limited"
        else:
            support_status = "insufficient_reviewer_facing_cells"
        rows.append(
            {
                "region": region,
                "condition": condition,
                "n_inventory_cells": int(inv["file_id"].nunique()) if "file_id" in inv else 0,
                "n_accepted_cells": int(acc["file_id"].nunique()) if "file_id" in acc else 0,
                "n_reviewer_facing_cells": n_reviewer,
                "min_reviewer_facing_cells": int(config.min_reviewer_facing_cells),
                "n_step06_groups": int(len(rs)),
                "n_predictive_supported_groups": n_predictive,
                "n_prediction_limited_or_fit_only_groups": n_limited,
                "minimum_cell_gate": minimum_cell_gate,
                "any_predictive_group_gate": any_predictive_gate,
                "all_groups_predictive_gate": all_group_gate,
                "support_status": support_status,
                "allowed_claim_scope": (
                    "region_condition_mechanism_claim_allowed"
                    if support_status == "all_groups_supported"
                    else (
                        "only_predictive_supported_groups_allowed"
                        if support_status == "restricted_supported_groups_only"
                        else "no_region_condition_mechanism_claim"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_limited_failure_modes(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Decompose Step 06 prediction-limited groups into objective failure axes."""

    robust = _read_csv(root / "outputs" / "predictive_validation" / "robustness_summary.csv")
    perturb = _read_csv(root / "outputs" / "predictive_validation" / "perturbation_sweeps.csv")
    ppc = _read_csv(root / "outputs" / "predictive_validation" / "feature_distribution_ppc.csv")
    if robust.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    limited = robust[robust["validation_label"].astype(str).ne("predictive_supported")].copy()
    for _, row in limited.iterrows():
        cluster = str(row["mechanism_cluster"])
        region = str(row["region"])
        condition = str(row["condition"])
        q = perturb[
            perturb["mechanism_cluster"].astype(str).eq(cluster)
            & perturb["region"].astype(str).eq(region)
            & perturb["condition"].astype(str).eq(condition)
            & perturb["perturbation"].astype(str).ne("nominal")
        ] if not perturb.empty else pd.DataFrame()
        p = ppc[
            ppc["region"].astype(str).eq(region)
            & ppc["condition"].astype(str).eq(condition)
        ] if not ppc.empty else pd.DataFrame()
        axes: list[str] = []
        if pd.to_numeric(pd.Series([row.get("holdout_pass_fraction")]), errors="coerce").iloc[0] < config.min_holdout_pass_fraction:
            axes.append("heldout_prediction")
        if pd.to_numeric(pd.Series([row.get("perturbation_robust_fraction")]), errors="coerce").iloc[0] < config.min_perturbation_robust_fraction:
            axes.append("perturbation_robustness")
        if pd.to_numeric(pd.Series([row.get("mean_weighted_ppc_coverage")]), errors="coerce").iloc[0] < config.min_ppc_coverage:
            axes.append("posterior_predictive_coverage")
        if pd.to_numeric(pd.Series([row.get("hidden_flux_plausible_fraction")]), errors="coerce").iloc[0] < 1.0:
            axes.append("hidden_flux_plausibility")
        if not axes:
            axes.append("threshold_or_sparse_group_boundary")
        worst_perturb = ""
        if not q.empty and "functional_buffering_pass" in q:
            by_pert = q.groupby("perturbation", dropna=False)["functional_buffering_pass"].apply(lambda s: _safe_fraction(s)).sort_values()
            worst_perturb = str(by_pert.index[0]) if not by_pert.empty else ""
        worst_feature = ""
        if not p.empty and "weighted_coverage" in p:
            worst = p.assign(weighted_coverage_num=pd.to_numeric(p["weighted_coverage"], errors="coerce")).sort_values("weighted_coverage_num")
            worst_feature = str(worst["feature"].iloc[0]) if not worst.empty else ""
        rows.append(
            {
                "mechanism_cluster": cluster,
                "region": region,
                "condition": condition,
                "validation_label": row.get("validation_label"),
                "n_candidates": int(row.get("n_candidates", 0)),
                "n_cells": int(row.get("n_cells", 0)),
                "holdout_pass_fraction": row.get("holdout_pass_fraction"),
                "mean_weighted_ppc_coverage": row.get("mean_weighted_ppc_coverage"),
                "perturbation_robust_fraction": row.get("perturbation_robust_fraction"),
                "hidden_flux_plausible_fraction": row.get("hidden_flux_plausible_fraction"),
                "failure_axes": ";".join(axes),
                "worst_perturbation": worst_perturb,
                "worst_ppc_feature": worst_feature,
                "remediation_priority": "high" if "perturbation_robustness" in axes or "heldout_prediction" in axes else "medium",
                "claim_action": "keep_prediction_limited_until_failure_axes_are_resolved",
            }
        )
    return pd.DataFrame(rows)


def build_assumption_gate_audit(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Create explicit pass/fail gates for Step 07 assumption axes."""

    gating = _read_csv(root / "outputs" / "assumption_sensitivity" / "gating_family_comparison.csv")
    proxy = _read_csv(root / "outputs" / "assumption_sensitivity" / "proxy_validity_by_ensemble.csv")
    split = _read_csv(root / "outputs" / "assumption_sensitivity" / "compartment_split_sensitivity.csv")
    rows: list[dict[str, Any]] = []
    gating_unstable_fraction = 1.0 - _safe_fraction(gating.get("mechanism_claim_stable", pd.Series(dtype=bool)))
    rows.append(
        {
            "assumption_axis": "gating_form",
            "n_rows": int(len(gating)),
            "metric": "unstable_fraction",
            "metric_value": gating_unstable_fraction,
            "threshold": float(config.max_gating_unstable_fraction),
            "gate_pass": bool(np.isfinite(gating_unstable_fraction) and gating_unstable_fraction <= config.max_gating_unstable_fraction),
            "gate_status": "pass" if np.isfinite(gating_unstable_fraction) and gating_unstable_fraction <= config.max_gating_unstable_fraction else "fail",
            "claim_scope": "same-parameter gating-family screen; refit-level test still pending",
        }
    )
    proxy_limited_fraction = 1.0 - _safe_fraction(proxy.get("proxy_validity_status", pd.Series(dtype=object)).astype(str).eq("proxy_supported") if not proxy.empty else pd.Series(dtype=bool))
    rows.append(
        {
            "assumption_axis": "intracellular_K_as_ECS_proxy",
            "n_rows": int(len(proxy)),
            "metric": "proxy_limited_fraction",
            "metric_value": proxy_limited_fraction,
            "threshold": float(config.max_proxy_limited_fraction),
            "gate_pass": bool(np.isfinite(proxy_limited_fraction) and proxy_limited_fraction <= config.max_proxy_limited_fraction),
            "gate_status": "pass" if np.isfinite(proxy_limited_fraction) and proxy_limited_fraction <= config.max_proxy_limited_fraction else "fail",
            "claim_scope": "proxy exclusion can restrict claims; explicit ECS variant still needed for broad ECS-homeostasis claims",
        }
    )
    split_sensitive_fraction = 1.0 - _safe_fraction(split.get("split_sensitivity_status", pd.Series(dtype=object)).astype(str).eq("split_robust") if not split.empty else pd.Series(dtype=bool))
    rows.append(
        {
            "assumption_axis": "local_syncytial_compartment_split",
            "n_rows": int(len(split)),
            "metric": "split_sensitive_fraction",
            "metric_value": split_sensitive_fraction,
            "threshold": float(config.max_split_sensitive_fraction),
            "gate_pass": bool(np.isfinite(split_sensitive_fraction) and split_sensitive_fraction <= config.max_split_sensitive_fraction),
            "gate_status": "pass" if np.isfinite(split_sensitive_fraction) and split_sensitive_fraction <= config.max_split_sensitive_fraction else "fail",
            "claim_scope": "lumped split appears stable in current screen if gate passes; spatial-scale validation remains separate",
        }
    )
    return pd.DataFrame(rows)


def build_proxy_exclusion_claim_sensitivity(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Assess which Step 06 groups survive after excluding proxy-limited candidates."""

    proxy = _read_csv(root / "outputs" / "assumption_sensitivity" / "proxy_validity_by_ensemble.csv")
    perturb = _read_csv(root / "outputs" / "predictive_validation" / "perturbation_sweeps.csv")
    heldout = _read_csv(root / "outputs" / "predictive_validation" / "heldout_current_errors.csv")
    if proxy.empty or perturb.empty:
        return pd.DataFrame()
    supported_ids = proxy[proxy["proxy_validity_status"].astype(str).eq("proxy_supported")][IDENTITY_COLUMNS].drop_duplicates()
    rows: list[dict[str, Any]] = []
    group_cols = ["mechanism_cluster", "region", "condition"]
    for keys, group in perturb.groupby(group_cols, dropna=False):
        cluster, region, condition = keys
        before_candidates = group[IDENTITY_COLUMNS].drop_duplicates()
        after_group = group.merge(supported_ids, on=IDENTITY_COLUMNS, how="inner")
        h_after = heldout.merge(supported_ids, on=IDENTITY_COLUMNS, how="inner") if not heldout.empty else pd.DataFrame()
        h_after = h_after[
            h_after["mechanism_cluster"].astype(str).eq(str(cluster))
            & h_after["region"].astype(str).eq(str(region))
            & h_after["condition"].astype(str).eq(str(condition))
        ] if not h_after.empty else h_after
        non_nominal_after = after_group[after_group["perturbation"].astype(str).ne("nominal")]
        holdout_after = _safe_fraction(h_after.get("prediction_pass", pd.Series(dtype=bool)))
        robust_after = _safe_fraction(non_nominal_after.get("functional_buffering_pass", pd.Series(dtype=bool)))
        support_after = bool(
            after_group["file_id"].nunique() >= config.min_reviewer_facing_cells
            and np.isfinite(holdout_after)
            and holdout_after >= config.min_holdout_pass_fraction
            and np.isfinite(robust_after)
            and robust_after >= config.min_perturbation_robust_fraction
        )
        rows.append(
            {
                "mechanism_cluster": cluster,
                "region": region,
                "condition": condition,
                "n_candidates_before_proxy_exclusion": int(before_candidates["candidate_id"].nunique()),
                "n_cells_before_proxy_exclusion": int(before_candidates["file_id"].nunique()),
                "n_candidates_after_proxy_exclusion": int(after_group["candidate_id"].nunique()) if not after_group.empty else 0,
                "n_cells_after_proxy_exclusion": int(after_group["file_id"].nunique()) if not after_group.empty else 0,
                "holdout_pass_fraction_after_proxy_exclusion": holdout_after,
                "perturbation_robust_fraction_after_proxy_exclusion": robust_after,
                "proxy_exclusion_support_status": "survives_proxy_exclusion" if support_after else "blocked_or_underpowered_after_proxy_exclusion",
                "claim_scope": "restricted_to_proxy_supported_candidates" if support_after else "proxy_assumption_blocks_or_underpowers_group_claim",
            }
        )
    return pd.DataFrame(rows)


def _semantic_rows(ranges: pd.DataFrame) -> list[dict[str, Any]]:
    semantic_map = {
        "gki": ("raw_physiological_candidate", "requires_literature_citation", "Kir conductance can be biological only with cited units and identifiability support."),
        "eps": ("phenomenological_exchange_guardrail", "not_direct_physiology", "Reduced exchange factor is a model guardrail, not a direct ECS physiology measurement."),
        "gl_a": ("raw_physiological_candidate", "requires_literature_citation", "Leak conductance can be biological only with cited units and identifiability support."),
        "zth": ("phenomenological_gating_coordinate", "not_direct_physiology", "Gating threshold is a reduced-model activation coordinate."),
        "zs": ("phenomenological_gating_coordinate", "not_direct_physiology", "Gating slope/scale is a reduced-model activation coordinate."),
        "P_gap_eff": ("effective_coordinate", "effective_only", "Effective product is interpretable as a model coordinate, not raw anatomy."),
        "gamma_t_eff": ("effective_coordinate", "effective_only", "Effective local transport coordinate is model-level."),
        "gamma_s_eff": ("effective_coordinate", "effective_only", "Effective spatial transport coordinate is model-level."),
        "volume_ratio_wa_wo": ("effective_coordinate", "effective_only", "Effective volume ratio is model-level and needs source basis for physiology."),
    }
    rows: list[dict[str, Any]] = []
    for _, row in ranges.iterrows():
        parameter = str(row["parameter"])
        semantic_class, citation_status, note = semantic_map.get(
            parameter,
            ("unclassified_guardrail", "requires_manual_review", "Parameter requires manual semantic review."),
        )
        rows.append(
            {
                **row.to_dict(),
                "interpretation_class": semantic_class,
                "citation_status": citation_status,
                "direct_physiology_claim_allowed_by_semantics": semantic_class == "raw_physiological_candidate" and citation_status == "citation_present",
                "effective_coordinate_claim_allowed_by_semantics": semantic_class == "effective_coordinate",
                "semantic_guardrail": note,
            }
        )
    return rows


def build_parameter_semantics_audit(root: Path) -> pd.DataFrame:
    """Classify parameter ranges by semantic interpretability."""

    ranges_path = root / "outputs" / "parameter_plausibility" / "parameter_ranges.csv"
    ranges = _read_csv(ranges_path)
    if ranges.empty:
        ranges = default_parameter_ranges()
    return pd.DataFrame(_semantic_rows(ranges))


def build_full_accepted_parameter_audit(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Audit parameter plausibility for the full accepted Step 04 ensemble."""

    candidates, _ = load_step06_inputs(
        root,
        Step06Config(
            max_candidates=config.max_candidates,
            candidate_policy="all",
            write_outputs=False,
        ),
    )
    ranges = _read_csv(root / "outputs" / "parameter_plausibility" / "parameter_ranges.csv")
    if ranges.empty:
        ranges = default_parameter_ranges()
    identifiability = load_identifiability_status(root)
    audit = build_parameter_range_audit(candidates, ranges, identifiability)
    audit["audit_scope"] = "full_accepted_ensemble"
    return audit


def build_parameter_interpretation_class_audit(root: Path, semantics: pd.DataFrame) -> pd.DataFrame:
    """Apply semantic interpretation classes to the Step 08 row-level audit."""

    audit = _read_csv(root / "outputs" / "parameter_plausibility" / "parameter_range_audit.csv")
    if audit.empty:
        return pd.DataFrame()
    out = audit.merge(
        semantics[
            [
                "parameter",
                "interpretation_class",
                "citation_status",
                "direct_physiology_claim_allowed_by_semantics",
                "effective_coordinate_claim_allowed_by_semantics",
                "semantic_guardrail",
            ]
        ],
        on="parameter",
        how="left",
        validate="many_to_one",
    )
    out["physiology_claim_allowed_after_semantic_filter"] = (
        out["physiologically_interpretable"].astype(bool)
        & out["direct_physiology_claim_allowed_by_semantics"].fillna(False).astype(bool)
    )
    out["effective_coordinate_claim_allowed_after_semantic_filter"] = (
        out["physiologically_interpretable"].astype(bool)
        & out["effective_coordinate_claim_allowed_by_semantics"].fillna(False).astype(bool)
    )
    out["claim_class_after_semantic_filter"] = np.select(
        [
            out["physiology_claim_allowed_after_semantic_filter"],
            out["effective_coordinate_claim_allowed_after_semantic_filter"],
            out["interpretation_class"].astype(str).str.contains("phenomenological", na=False),
        ],
        ["direct_physiology_candidate", "effective_coordinate_only", "phenomenological_guardrail_only"],
        default="not_claim_supporting",
    )
    return out


def build_constrained_failure_modes(root: Path) -> pd.DataFrame:
    """Summarize constrained-screen candidate failures from Step 08."""

    constrained = _read_csv(root / "outputs" / "parameter_plausibility" / "constrained_rerun_comparison.csv")
    if constrained.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in constrained.groupby(IDENTITY_COLUMNS + ["mechanism_cluster"], dropna=False):
        *identity, cluster = keys
        prediction_pass = bool(_bool_series(group["prediction_persists_under_constraints"]).all()) if "prediction_persists_under_constraints" in group else False
        mechanism_pass = bool(_bool_series(group["mechanism_persists_under_constraints"]).all()) if "mechanism_persists_under_constraints" in group else False
        failure_axes: list[str] = []
        if not prediction_pass:
            failure_axes.append("prediction_constraint_sensitive")
        if not mechanism_pass:
            failure_axes.append("mechanism_constraint_sensitive")
        if group["simulation_status"].astype(str).ne("ok").any():
            failure_axes.append("simulation_failed")
        if not failure_axes:
            failure_axes.append("no_constraint_failure_detected")
        rows.append(
            {
                **dict(zip(IDENTITY_COLUMNS, identity)),
                "mechanism_cluster": cluster,
                "n_current_rows": int(len(group)),
                "changed_parameters": _join_unique(group.get("changed_parameters", pd.Series(dtype=object))),
                "max_fit_degradation_fraction": _finite_max(group.get("fit_degradation_fraction", pd.Series(dtype=float))),
                "max_mechanism_flux_fraction_delta": _finite_max(group.get("mechanism_flux_fraction_delta_max", pd.Series(dtype=float))),
                "prediction_persists_all_currents": prediction_pass,
                "mechanism_persists_all_currents": mechanism_pass,
                "failure_axes": ";".join(failure_axes),
                "claim_action": "candidate_parameter_claim_retained" if failure_axes == ["no_constraint_failure_detected"] else "candidate_parameter_claim_downgraded_or_excluded",
            }
        )
    return pd.DataFrame(rows)


def build_ko_homeostasis_endpoint_audit(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Audit K_o-specific endpoints separately from Vm feature robustness."""

    perturb = _read_csv(root / "outputs" / "predictive_validation" / "perturbation_sweeps.csv")
    if perturb.empty:
        return pd.DataFrame()
    non_nominal = perturb[perturb["perturbation"].astype(str).ne("nominal")].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["mechanism_cluster", "buffering_phenotype", "region", "condition"]
    for keys, group in non_nominal.groupby(group_cols, dropna=False):
        cluster, phenotype, region, condition = keys
        peak_pass = pd.to_numeric(group["K_o_peak"], errors="coerce").le(15.0)
        final_pass = pd.to_numeric(group["K_o_final"], errors="coerce").abs().le(7.0)
        recovery_pass = pd.to_numeric(group["K_o_recovery_error"], errors="coerce").abs().le(1.5)
        ratio_pass = pd.to_numeric(group["K_o_peak_ratio_to_nominal"], errors="coerce").abs().le(1.75) | pd.to_numeric(group["K_o_peak_ratio_to_nominal"], errors="coerce").isna()
        endpoint_pass = peak_pass & final_pass & recovery_pass & ratio_pass
        pass_fraction = float(endpoint_pass.mean()) if len(endpoint_pass) else float("nan")
        rows.append(
            {
                "mechanism_cluster": cluster,
                "buffering_phenotype": phenotype,
                "region": region,
                "condition": condition,
                "n_candidates": int(group["candidate_id"].nunique()),
                "n_cells": int(group["file_id"].nunique()),
                "n_perturbation_rows": int(len(group)),
                "K_o_endpoint_pass_fraction": pass_fraction,
                "max_K_o_peak": _finite_max(group["K_o_peak"]),
                "max_abs_K_o_final": _finite_max(pd.to_numeric(group["K_o_final"], errors="coerce").abs()),
                "max_abs_K_o_recovery_error": _finite_max(pd.to_numeric(group["K_o_recovery_error"], errors="coerce").abs()),
                "homeostasis_endpoint_gate": "pass" if np.isfinite(pass_fraction) and pass_fraction >= config.min_homeostasis_pass_fraction else "fail",
                "claim_scope": "K_o_endpoint_supported" if np.isfinite(pass_fraction) and pass_fraction >= config.min_homeostasis_pass_fraction else "Vm_or_mechanism_support_only_not_K_o_homeostasis",
            }
        )
    return pd.DataFrame(rows)


def _load_thresholds(root: Path) -> pd.DataFrame:
    primary = root / "outputs" / "features" / "condition_region_sweep_thresholds.csv"
    pooled = root / "outputs" / "features" / "region_pooled_condition_sweep_thresholds.csv"
    frames = [_read_csv(path) for path in (primary, pooled) if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _find_threshold(thresholds: pd.DataFrame, condition: str, region: str, sweep: int, feature: str) -> pd.Series | None:
    if thresholds.empty:
        return None
    for cond, reg in ((condition, region), (condition, "ALL"), ("ALL", "ALL")):
        hit = thresholds[
            thresholds["condition"].astype(str).eq(str(cond))
            & thresholds["region"].astype(str).eq(str(reg))
            & thresholds["sweep"].astype(int).eq(int(sweep))
            & thresholds["feature"].astype(str).eq(str(feature))
        ]
        if not hit.empty:
            return hit.iloc[0]
    return None


def _feature_pass_fraction(root: Path, sim: Mapping[str, Any], *, region: str, condition: str, sweep: int) -> tuple[float, int]:
    thresholds = _load_thresholds(root)
    window = stim_window_seconds(condition)
    time_s = np.asarray(sim["t_ms"], dtype=float) / 1000.0
    features = extract_features_from_trace(
        time_s,
        np.asarray(sim["Vm"], dtype=float),
        onset_s=window[0],
        offset_s=window[1],
    )
    evaluated = 0
    passed = 0
    for feature, value in features.items():
        if feature not in FEATURE_COLUMNS or not np.isfinite(float(value)):
            continue
        threshold = _find_threshold(thresholds, condition, region, sweep, str(feature))
        if threshold is None:
            continue
        evaluated += 1
        passed += int(float(threshold["acceptable_lower"]) <= float(value) <= float(threshold["acceptable_upper"]))
    if evaluated == 0:
        return float("nan"), 0
    return float(passed / evaluated), int(evaluated)


def _interpolate_candidate_row(row_a: pd.Series, row_b: pd.Series, alpha: float) -> dict[str, Any]:
    interpolated = row_a.to_dict()
    for column in AUDITED_PARAMETERS:
        if column not in row_a or column not in row_b:
            continue
        a = float(row_a[column])
        b = float(row_b[column])
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            interpolated[column] = float(10 ** ((1.0 - alpha) * np.log10(a) + alpha * np.log10(b)))
        elif np.isfinite(a) and np.isfinite(b):
            interpolated[column] = float((1.0 - alpha) * a + alpha * b)
    interpolated["candidate_id"] = f"interp_{row_a.get('mechanism_cluster')}_{row_b.get('mechanism_cluster')}_{alpha:.2f}"
    return interpolated


def build_intercluster_interpolation_acceptance(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Screen intercluster interpolation with no optimizer and feature-band scoring."""

    clusters = _read_csv(root / "outputs" / "mechanisms" / "mechanism_clusters.csv")
    if clusters.empty:
        return pd.DataFrame()
    if config.max_candidates is not None:
        clusters = clusters.head(int(config.max_candidates)).copy()
    alphas = np.linspace(0.0, 1.0, int(config.interpolation_points))
    time_grid = np.linspace(0.0, 50_000.0, int(config.interpolation_time_points), dtype=float)
    rows: list[dict[str, Any]] = []
    for (region, condition), stratum in clusters.groupby(["region", "condition"], dropna=False):
        cluster_names = sorted(stratum["mechanism_cluster"].astype(str).unique())
        if len(cluster_names) < 2:
            continue
        medians = (
            stratum.groupby("mechanism_cluster", as_index=False)
            .median(numeric_only=True)
            .merge(
                stratum.groupby("mechanism_cluster", as_index=False)[["file_id", "candidate_id", "region", "condition", "mechanism_cluster"]].first(),
                on="mechanism_cluster",
                how="left",
            )
        )
        for cluster_a, cluster_b in combinations(cluster_names, 2):
            row_a = medians[medians["mechanism_cluster"].astype(str).eq(cluster_a)]
            row_b = medians[medians["mechanism_cluster"].astype(str).eq(cluster_b)]
            if row_a.empty or row_b.empty:
                continue
            for alpha in alphas:
                cand = _interpolate_candidate_row(row_a.iloc[0], row_b.iloc[0], float(alpha))
                cand["region"] = region
                cand["condition"] = condition
                for sweep, current_na in enumerate(VALID_CURRENTS, start=1):
                    try:
                        params = reconstruct_flat_params(cand, current_na=int(current_na), sweep=sweep)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", ODEintWarning)
                            sim = simulate_with_hidden_outputs(
                                params,
                                {
                                    "experiment_type": protocol_condition(str(condition)),
                                    "current_na": int(current_na),
                                    "t_eval_ms": time_grid,
                                },
                            )
                        pass_fraction, n_features = _feature_pass_fraction(
                            root, sim, region=str(region), condition=str(condition), sweep=sweep
                        )
                        accepted_like = bool(np.isfinite(pass_fraction) and pass_fraction >= config.min_holdout_pass_fraction)
                        warning_text = "; ".join(str(item.message) for item in caught)
                        rows.append(
                            {
                                "region": region,
                                "condition": condition,
                                "cluster_a": cluster_a,
                                "cluster_b": cluster_b,
                                "alpha": float(alpha),
                                "sweep": int(sweep),
                                "current_na": int(current_na),
                                "feature_pass_fraction": pass_fraction,
                                "n_features_scored": n_features,
                                "accepted_like_feature_contract": accepted_like,
                                "screen_type": "interpolated_parameter_resimulation_feature_contract_not_step04_refit",
                                "simulation_status": "ok",
                                "simulation_warning": warning_text,
                                "failure_reason": "",
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        rows.append(
                            {
                                "region": region,
                                "condition": condition,
                                "cluster_a": cluster_a,
                                "cluster_b": cluster_b,
                                "alpha": float(alpha),
                                "sweep": int(sweep),
                                "current_na": int(current_na),
                                "feature_pass_fraction": np.nan,
                                "n_features_scored": 0,
                                "accepted_like_feature_contract": False,
                                "screen_type": "interpolated_parameter_resimulation_feature_contract_not_step04_refit",
                                "simulation_status": "failed",
                                "simulation_warning": "",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pair_summary = out.groupby(["region", "condition", "cluster_a", "cluster_b"], dropna=False).agg(
        interpolation_feature_pass_fraction=("accepted_like_feature_contract", lambda s: float(_bool_series(s).mean())),
        min_feature_pass_fraction=("feature_pass_fraction", "min"),
        warning_fraction=("simulation_warning", lambda s: float(s.astype(str).ne("").mean())),
    ).reset_index()
    pair_summary["interpolation_acceptance_status"] = np.where(
        pair_summary["interpolation_feature_pass_fraction"].ge(0.80),
        "continuous_compensation_manifold_feature_supported",
        "possible_separated_modes_or_feature_contract_gap",
    )
    return out.merge(pair_summary, on=["region", "condition", "cluster_a", "cluster_b"], how="left")


def _candidate_label_for_thresholds(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> str:
    activation = float(row.get("dKs_activation_score_mean", np.nan))
    long_fraction = float(row.get("long_range_distribution_fraction_mean", np.nan))
    kir_score = float(row.get("kir_current_score_mean", np.nan))
    voltage_score = float(row.get("voltage_coupling_score_mean", np.nan))
    if activation < thresholds["activation_low"] and voltage_score >= thresholds["voltage_coupling"]:
        return "available_surface_voltage_coupled_but_ionic_recruitment_low"
    if long_fraction >= thresholds["long_range"] and activation >= thresholds["activation_recruited"]:
        return "long_range_recruited_spatial_buffering"
    if kir_score >= thresholds["kir_dominant"] and activation < thresholds["activation_recruited"]:
        return "kir_dominant_local_buffering"
    if activation < thresholds["activation_low"]:
        return "low_recruitment_local_storage"
    return "mixed_local_spatial_buffering"


def build_phenotype_threshold_sensitivity(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Measure phenotype-label stability under plausible threshold grids."""

    tags = _read_csv(root / "outputs" / "mechanisms" / "buffering_phenotype_tags.csv")
    if tags.empty:
        return pd.DataFrame()
    if config.max_candidates is not None:
        tags = tags.head(int(config.max_candidates)).copy()
    grid = list(
        product(
            (0.05, 0.10, 0.15),
            (0.50, 0.60, 0.70),
            (0.20, 0.30, 0.40),
            (0.50, 0.60, 0.70),
            (0.20, 0.30, 0.40),
        )
    )
    rows: list[dict[str, Any]] = []
    for _, row in tags.iterrows():
        labels: list[str] = []
        for activation_low, long_range, activation_recruited, kir_dominant, voltage in grid:
            labels.append(
                _candidate_label_for_thresholds(
                    row,
                    {
                        "activation_low": activation_low,
                        "long_range": long_range,
                        "activation_recruited": activation_recruited,
                        "kir_dominant": kir_dominant,
                        "voltage_coupling": voltage,
                    },
                )
            )
        counts = pd.Series(labels).value_counts()
        baseline = str(row["buffering_phenotype"])
        rows.append(
            {
                **{c: row.get(c) for c in IDENTITY_COLUMNS},
                "baseline_buffering_phenotype": baseline,
                "n_threshold_configs": int(len(labels)),
                "n_unique_sensitivity_labels": int(counts.size),
                "modal_sensitivity_label": str(counts.index[0]),
                "modal_label_fraction": float(counts.iloc[0] / len(labels)),
                "baseline_label_fraction": float(labels.count(baseline) / len(labels)),
                "threshold_sensitivity_status": "stable" if float(counts.iloc[0] / len(labels)) >= 0.80 else "threshold_sensitive",
                "claim_scope": "discrete_phenotype_label_supported" if float(counts.iloc[0] / len(labels)) >= 0.80 and labels.count(baseline) / len(labels) >= 0.50 else "prefer_continuous_scores_or_modal_label",
            }
        )
    return pd.DataFrame(rows)


def build_all_current_assumption_sensitivity(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Extend Step 07 same-parameter assumption checks to all six currents."""

    step07_config = Step07Config(
        max_candidates=config.max_candidates,
        candidate_policy="best_per_cell",
        time_points=config.all_current_time_points,
        currents_na=tuple(int(c) for c in VALID_CURRENTS),
        write_outputs=False,
    )
    candidates = load_step07_inputs(root, step07_config)
    gating = build_gating_family_comparison(candidates, step07_config)
    proxy = build_proxy_validity(candidates, step07_config)
    split = build_compartment_split_sensitivity(candidates, step07_config)
    rows: list[dict[str, Any]] = []
    for keys, group in gating.groupby(["region", "condition", "current_na"], dropna=False):
        region, condition, current_na = keys
        stable_fraction = _safe_fraction(group["mechanism_claim_stable"])
        rows.append(
            {
                "assumption_axis": "gating_form",
                "region": region,
                "condition": condition,
                "current_na": int(current_na),
                "n_rows": int(len(group)),
                "support_fraction": stable_fraction,
                "sensitivity_status": "stable" if np.isfinite(stable_fraction) and stable_fraction >= 1.0 - config.max_gating_unstable_fraction else "current_sensitive",
            }
        )
    for keys, group in proxy.groupby(["region", "condition", "current_na"], dropna=False):
        region, condition, current_na = keys
        support_fraction = _safe_fraction(group["proxy_validity_status"].astype(str).eq("proxy_supported"))
        rows.append(
            {
                "assumption_axis": "intracellular_K_as_ECS_proxy",
                "region": region,
                "condition": condition,
                "current_na": int(current_na),
                "n_rows": int(len(group)),
                "support_fraction": support_fraction,
                "sensitivity_status": "stable" if np.isfinite(support_fraction) and support_fraction >= 1.0 - config.max_proxy_limited_fraction else "current_sensitive",
            }
        )
    for keys, group in split.groupby(["region", "condition", "current_na"], dropna=False):
        region, condition, current_na = keys
        support_fraction = _safe_fraction(group["split_sensitivity_status"].astype(str).eq("split_robust"))
        rows.append(
            {
                "assumption_axis": "local_syncytial_compartment_split",
                "region": region,
                "condition": condition,
                "current_na": int(current_na),
                "n_rows": int(len(group)),
                "support_fraction": support_fraction,
                "sensitivity_status": "stable" if np.isfinite(support_fraction) and support_fraction >= 1.0 - config.max_split_sensitive_fraction else "current_sensitive",
            }
        )
    return pd.DataFrame(rows)


def _log10_span(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric) & (numeric > 0)]
    if numeric.empty:
        return float("nan")
    return float(np.log10(numeric.max()) - np.log10(numeric.min()))


def build_cell_specific_identifiability_audit(root: Path, full_audit: pd.DataFrame) -> pd.DataFrame:
    """Use within-cell accepted-ensemble spread as a cell-specific identifiability screen."""

    if full_audit.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in full_audit.groupby(["file_id", "region", "condition", "parameter"], dropna=False):
        file_id, region, condition, parameter = keys
        span = _log10_span(group["value"])
        within_fraction = _safe_fraction(group["plausibility_status"].astype(str).eq("within_range"))
        if not np.isfinite(span):
            status = "insufficient_numeric_values"
        elif span <= 0.50:
            status = "cell_ensemble_tightly_constrained"
        elif span <= 1.50:
            status = "cell_ensemble_moderately_broad"
        else:
            status = "cell_ensemble_broad_or_sloppy"
        rows.append(
            {
                "file_id": file_id,
                "region": region,
                "condition": condition,
                "parameter": parameter,
                "coordinate_type": str(group["coordinate_type"].iloc[0]),
                "n_accepted_candidate_values": int(len(group)),
                "log10_value_span": span,
                "within_range_fraction": within_fraction,
                "step03_identifiability_status": str(group["identifiability_status"].iloc[0]),
                "cell_specific_identifiability_status": status,
                "claim_scope": "cell_specific_parameter_value_supported" if status == "cell_ensemble_tightly_constrained" and within_fraction == 1.0 else "cell_specific_parameter_claim_downgraded",
            }
        )
    return pd.DataFrame(rows)


def build_parameter_ranges_citation_audit(root: Path, semantics: pd.DataFrame) -> pd.DataFrame:
    """Report which parameter ranges still need citation/basis curation."""

    if semantics.empty:
        return pd.DataFrame()
    out = semantics.copy()
    out["range_basis_is_literature_citation"] = out["range_source"].astype(str).str.contains("literature|citation|doi|pmid", case=False, regex=True)
    out["citation_audit_status"] = np.where(
        out["range_basis_is_literature_citation"].astype(bool),
        "citation_basis_present",
        "citation_or_unit_basis_missing",
    )
    out["claim_action"] = np.where(
        out["citation_audit_status"].eq("citation_basis_present") | out["interpretation_class"].eq("effective_coordinate"),
        "can_be_reported_with_declared_scope",
        "do_not_use_for_direct_physiological_parameter_claim",
    )
    return out


def build_integrated_degeneracy_gate_matrix(root: Path, config: ReviewerGateAuditConfig) -> pd.DataFrame:
    """Join mechanism, prediction, assumption, parameter, and K_o gates."""

    step08 = _read_csv(root / "outputs" / "parameter_plausibility" / "interpretability_status.csv")
    mechanisms = _read_csv(root / "outputs" / "mechanisms" / "mechanism_clusters.csv")
    robust = _read_csv(root / "outputs" / "predictive_validation" / "robustness_summary.csv")
    assumption = _read_csv(root / "outputs" / "reviewer_synthesis" / "assumption_gate_audit.csv")
    ko = _read_csv(root / "outputs" / "predictive_validation" / "K_o_homeostasis_endpoint_audit.csv")
    parameter_classes = _read_csv(root / "outputs" / "parameter_plausibility" / "parameter_interpretation_class_audit.csv")
    if step08.empty:
        return pd.DataFrame()
    cols = IDENTITY_COLUMNS + ["mechanism_cluster", "dominant_mechanism", "parameter_interpretability_status", "parameter_claim_allowed_after_step08", "prediction_persists_under_constraints", "mechanism_persists_under_constraints"]
    base = step08[[c for c in cols if c in step08.columns]].copy()
    mech_cols = IDENTITY_COLUMNS + ["buffering_phenotype", "phenotype_claim_scope"]
    if not mechanisms.empty:
        base = base.merge(mechanisms[[c for c in mech_cols if c in mechanisms.columns]].drop_duplicates(IDENTITY_COLUMNS), on=IDENTITY_COLUMNS, how="left")
    if not robust.empty:
        base = base.merge(
            robust[["mechanism_cluster", "region", "condition", "validation_label", "biological_description_score"]],
            on=["mechanism_cluster", "region", "condition"],
            how="left",
        )
    assumption_gate = bool(not assumption.empty and assumption["gate_pass"].astype(bool).all())
    parameter_semantic = (
        parameter_classes.groupby(IDENTITY_COLUMNS, dropna=False)
        .agg(
            n_direct_physiology_parameters=("physiology_claim_allowed_after_semantic_filter", lambda s: int(_bool_series(s).sum())),
            n_effective_coordinate_parameters=("effective_coordinate_claim_allowed_after_semantic_filter", lambda s: int(_bool_series(s).sum())),
        )
        .reset_index()
        if not parameter_classes.empty
        else pd.DataFrame()
    )
    if not parameter_semantic.empty:
        base = base.merge(parameter_semantic, on=IDENTITY_COLUMNS, how="left")
    if not ko.empty:
        ko_gate = ko[["mechanism_cluster", "buffering_phenotype", "region", "condition", "homeostasis_endpoint_gate"]].drop_duplicates()
        base = base.merge(ko_gate, on=["mechanism_cluster", "buffering_phenotype", "region", "condition"], how="left")
    base["step05_mechanism_gate"] = base["mechanism_cluster"].astype(str).ne("unlabeled")
    base["step06_prediction_gate"] = base["validation_label"].astype(str).eq("predictive_supported")
    base["assumption_all_axis_gate"] = assumption_gate
    base["parameter_semantic_gate"] = base.get("n_direct_physiology_parameters", pd.Series(0, index=base.index)).fillna(0).astype(int).gt(0) | base.get("n_effective_coordinate_parameters", pd.Series(0, index=base.index)).fillna(0).astype(int).gt(0)
    base["parameter_step08_gate"] = base["parameter_claim_allowed_after_step08"].fillna(False).astype(bool)
    base["ko_homeostasis_gate"] = base.get("homeostasis_endpoint_gate", pd.Series("missing", index=base.index)).astype(str).eq("pass")
    base["restricted_degeneracy_claim_allowed"] = (
        base["step05_mechanism_gate"]
        & base["step06_prediction_gate"]
        & base["assumption_all_axis_gate"]
        & base["parameter_step08_gate"]
        & base["parameter_semantic_gate"]
        & base["ko_homeostasis_gate"]
    )
    base["blocking_axes"] = base.apply(_blocking_axes, axis=1)
    return base


def _blocking_axes(row: pd.Series) -> str:
    axes = []
    for column, axis in [
        ("step05_mechanism_gate", "mechanism_label"),
        ("step06_prediction_gate", "prediction_perturbation"),
        ("assumption_all_axis_gate", "assumptions"),
        ("parameter_step08_gate", "parameter_step08"),
        ("parameter_semantic_gate", "parameter_semantics"),
        ("ko_homeostasis_gate", "K_o_homeostasis"),
    ]:
        if not bool(row.get(column, False)):
            axes.append(axis)
    return ";".join(axes) if axes else "none"


def build_degeneracy_level_table(integrated: pd.DataFrame) -> pd.DataFrame:
    """Classify the strongest allowed degeneracy wording for each integrated row."""

    if integrated.empty:
        return pd.DataFrame()
    cols = [
        *IDENTITY_COLUMNS,
        "mechanism_cluster",
        "buffering_phenotype",
        "validation_label",
        "blocking_axes",
        "restricted_degeneracy_claim_allowed",
    ]
    out = integrated[[c for c in cols if c in integrated.columns]].copy()
    out["degeneracy_level"] = np.select(
        [
            out["restricted_degeneracy_claim_allowed"].astype(bool),
            out["validation_label"].astype(str).eq("predictive_supported") & out["blocking_axes"].astype(str).str.contains("assumptions|parameter|K_o", regex=True),
            out["validation_label"].astype(str).eq("predictive_supported"),
            out["mechanism_cluster"].astype(str).ne("unlabeled"),
        ],
        [
            "restricted_validated_degeneracy_candidate",
            "predictive_model_regime_blocked_by_other_gates",
            "predictive_model_regime",
            "candidate_regime_or_compensation_manifold",
        ],
        default="non_identifiability_or_insufficient_evidence",
    )
    out["allowed_wording"] = np.select(
        [
            out["restricted_degeneracy_claim_allowed"].astype(bool),
            out["degeneracy_level"].eq("predictive_model_regime_blocked_by_other_gates"),
            out["degeneracy_level"].eq("candidate_regime_or_compensation_manifold"),
        ],
        [
            "restricted validated degeneracy candidate",
            "predictive model-derived regime with blocked biological-degeneracy wording",
            "candidate model-derived regime or compensation manifold",
        ],
        default="insufficient evidence for degeneracy wording",
    )
    return out


def build_restricted_validation_claims(root: Path, integrated: pd.DataFrame) -> pd.DataFrame:
    """Generate allowed restricted claim rows from integrated gates."""

    robust = _read_csv(root / "outputs" / "predictive_validation" / "robustness_summary.csv")
    if robust.empty:
        return pd.DataFrame()
    grouped = integrated.groupby(["mechanism_cluster", "region", "condition"], dropna=False) if not integrated.empty else []
    allowed_map: dict[tuple[str, str, str], bool] = {}
    blocking_map: dict[tuple[str, str, str], str] = {}
    for keys, group in grouped:
        allowed_map[tuple(map(str, keys))] = bool(group["restricted_degeneracy_claim_allowed"].any())
        blocking_map[tuple(map(str, keys))] = _join_unique(group["blocking_axes"])
    rows: list[dict[str, Any]] = []
    for _, row in robust.iterrows():
        key = (str(row["mechanism_cluster"]), str(row["region"]), str(row["condition"]))
        predictive = str(row["validation_label"]) == "predictive_supported"
        all_gate = allowed_map.get(key, False)
        rows.append(
            {
                "mechanism_cluster": row["mechanism_cluster"],
                "region": row["region"],
                "condition": row["condition"],
                "validation_label": row["validation_label"],
                "restricted_all_gate_pass": all_gate,
                "blocking_axes": blocking_map.get(key, "not_joined"),
                "allowed_claim": (
                    "restricted biological degeneracy candidate"
                    if all_gate
                    else (
                        "predictive model-derived mechanism scenario"
                        if predictive
                        else "prediction-limited mechanism candidate"
                    )
                ),
                "forbidden_claim": "broad biological degeneracy or pathway-level phenotype" if not all_gate else "broad/global degeneracy beyond passing stratum",
            }
        )
    return pd.DataFrame(rows)


def build_restricted_all_gate_join(integrated: pd.DataFrame) -> pd.DataFrame:
    """Return a compact all-gate join for Step 09 maturity decisions."""

    if integrated.empty:
        return pd.DataFrame()
    cols = [
        *IDENTITY_COLUMNS,
        "mechanism_cluster",
        "buffering_phenotype",
        "validation_label",
        "step05_mechanism_gate",
        "step06_prediction_gate",
        "assumption_all_axis_gate",
        "parameter_step08_gate",
        "parameter_semantic_gate",
        "ko_homeostasis_gate",
        "restricted_degeneracy_claim_allowed",
        "blocking_axes",
    ]
    return integrated[[c for c in cols if c in integrated.columns]].copy()


def build_claim_to_artifact_ledger(root: Path) -> pd.DataFrame:
    """Map updated reviewer claims to artifacts and allowed wording."""

    rows = [
        ("candidate mechanism regimes are biologically interpretable", "outputs/predictive_validation/phenotype_robustness_summary.csv", "model-derived phenotype robustness where gates pass", "biological pathway or validated phenotype without external support"),
        ("candidate mechanism regimes are biologically interpretable", "outputs/reviewer_synthesis/stratum_support_gate.csv", "restricted region-condition support with explicit sparse strata", "population-level claim in unsupported strata"),
        ("model assumptions do not drive the conclusion", "outputs/reviewer_synthesis/assumption_gate_audit.csv", "assumption axes pass/fail under configured gates", "assumption-independent conclusion where any gate fails"),
        ("accepted parameters are physiologically interpretable", "outputs/parameter_plausibility/parameter_interpretation_class_audit.csv", "effective-coordinate or cited raw-parameter interpretation only", "direct physiological interpretation for phenomenological or uncited guardrails"),
        ("final biological degeneracy wording is allowed", "outputs/reviewer_synthesis/restricted_all_gate_join.csv", "restricted degeneracy wording only for all-gate passing rows", "global biological degeneracy if any required layer blocks"),
        ("final biological degeneracy wording is allowed", "outputs/predictive_validation/K_o_homeostasis_endpoint_audit.csv", "K_o endpoint support separate from Vm feature support", "K_o homeostasis claim from Vm-only gates"),
    ]
    out_rows: list[dict[str, Any]] = []
    for claim, artifact, allowed, forbidden in rows:
        path = root / artifact
        out_rows.append(
            {
                "claim": claim,
                "artifact": artifact,
                "artifact_exists": path.exists(),
                "allowed_wording": allowed,
                "forbidden_wording": forbidden,
                "claim_status": "available_for_restricted_wording" if path.exists() else "missing_artifact",
            }
        )
    return pd.DataFrame(out_rows)


def build_scientific_value_assessment(root: Path, outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Assess whether each generated artifact is useful enough to retain."""

    rows: list[dict[str, Any]] = []
    for name, frame in outputs.items():
        if name.startswith("_"):
            continue
        status = "retain"
        rationale = "artifact has rows and explicit gates"
        if frame.empty:
            status = "remove_or_regenerate"
            rationale = "artifact is empty and cannot support reviewer response"
        elif name == "restricted_all_gate_join" and not frame.get("restricted_degeneracy_claim_allowed", pd.Series(dtype=bool)).astype(bool).any():
            status = "retain_as_blocker_evidence"
            rationale = "all-gate join objectively shows final degeneracy remains blocked"
        elif name == "assumption_gate_audit" and not frame.get("gate_pass", pd.Series(dtype=bool)).astype(bool).all():
            status = "retain_as_blocker_evidence"
            rationale = "assumption gate failures are scientifically important limitations"
        rows.append(
            {
                "artifact_key": name,
                "n_rows": int(len(frame)),
                "n_columns": int(len(frame.columns)),
                "scientific_value_status": status,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def build_selected_action_results_summary(root: Path, outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a compact run summary for the selected actions."""

    rows: list[dict[str, Any]] = []
    for name, frame in outputs.items():
        if name.startswith("_"):
            continue
        rows.append(
            {
                "artifact_key": name,
                "n_rows": int(len(frame)),
                "n_columns": int(len(frame.columns)),
                "non_empty": bool(not frame.empty),
                "can_upgrade_claim_directly": bool(
                    name == "restricted_all_gate_join"
                    and not frame.empty
                    and frame.get("restricted_degeneracy_claim_allowed", pd.Series(dtype=bool)).astype(bool).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_notebook_update_screen_after_selected_actions(root: Path) -> pd.DataFrame:
    """Screen executed notebooks for rerun or comment-update needs."""

    rows = [
        {
            "notebook": "outputs/executed_notebooks/00_data_provenance_and_trace_audit.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "none_required",
            "remaining_action": "none",
            "reason": "Selected actions reuse downstream Step 04-08 outputs and do not change data provenance.",
        },
        {
            "notebook": "outputs/executed_notebooks/01_feature_extraction_and_qc.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "none_required",
            "remaining_action": "none",
            "reason": "Feature thresholds are consumed by the interpolation screen but were not modified.",
        },
        {
            "notebook": "outputs/executed_notebooks/02_feature_contracts_and_region_aware_thresholds.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "none_required",
            "remaining_action": "none",
            "reason": "Region-aware feature bands are reused unchanged by the new gate audits.",
        },
        {
            "notebook": "outputs/executed_notebooks/03_combined_identifiability_profiles_fim.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "none_required",
            "remaining_action": "none",
            "reason": "Step 03 identifiability evidence is reused unchanged in cell-specific and semantic parameter audits.",
        },
        {
            "notebook": "outputs/executed_notebooks/04_cell_specific_six_sweep_fitting.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "none_required",
            "remaining_action": "none",
            "reason": "No Step 04 refit is performed; full accepted ensemble and candidate history are reused.",
        },
        {
            "notebook": "outputs/executed_notebooks/05_mechanistic_decomposition.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "interpretation_centralized_in_step09",
            "remaining_action": "optional_future_cross_reference_only",
            "reason": "New interpolation and phenotype-threshold audits qualify Step 05 mechanism labels, but core Step 05 outputs are unchanged and Step 09 now reports the derived interpretation.",
        },
        {
            "notebook": "outputs/executed_notebooks/06_predictive_validation_and_perturbation.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "interpretation_centralized_in_step09",
            "remaining_action": "optional_future_cross_reference_only",
            "reason": "New phenotype robustness, prediction-limited failure, and K_o endpoint summaries refine interpretation of existing Step 06 outputs and are reported by Step 09.",
        },
        {
            "notebook": "outputs/executed_notebooks/07_assumption_sensitivity.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "interpretation_centralized_in_step09",
            "remaining_action": "optional_future_cross_reference_only",
            "reason": "All-current and proxy-exclusion audits refine Step 07 interpretation and are reported by Step 09; no optimizer rerun is required.",
        },
        {
            "notebook": "outputs/executed_notebooks/08_parameter_plausibility_and_constrained_reruns.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "interpretation_centralized_in_step09",
            "remaining_action": "optional_future_cross_reference_only",
            "reason": "Full-ensemble, semantic-class, citation, and cell-specific spread audits strengthen Step 08 guardrails and are reported by Step 09 without changing Step 08 core calculations.",
        },
        {
            "notebook": "outputs/executed_notebooks/09_reviewer_response_synthesis.ipynb",
            "rerun_required": False,
            "comment_update_recommended": False,
            "action_taken": "updated_and_rerun",
            "remaining_action": "none_unless_selected_action_gates_change",
            "reason": "Step 09 was updated to incorporate selected-action gates in claim maturity, manifest outputs, and post-execution interpretation.",
        },
    ]
    return pd.DataFrame(rows)


def _write_outputs(root: Path, outputs: Mapping[str, pd.DataFrame], config: ReviewerGateAuditConfig, elapsed: float) -> None:
    paths = _output_paths(root)
    for path in paths.values():
        _ensure_dir(path)
    output_map = {
        "phenotype_robustness_summary": paths["predictive"] / "phenotype_robustness_summary.csv",
        "stratum_support_gate": paths["reviewer"] / "stratum_support_gate.csv",
        "prediction_limited_failure_modes": paths["predictive"] / "prediction_limited_failure_modes.csv",
        "assumption_gate_audit": paths["reviewer"] / "assumption_gate_audit.csv",
        "proxy_exclusion_claim_sensitivity": paths["assumption"] / "proxy_exclusion_claim_sensitivity.csv",
        "parameter_semantics_audit": paths["parameter"] / "parameter_semantics_audit.csv",
        "full_accepted_parameter_audit": paths["parameter"] / "full_accepted_parameter_audit.csv",
        "parameter_interpretation_class_audit": paths["parameter"] / "parameter_interpretation_class_audit.csv",
        "constrained_failure_modes": paths["parameter"] / "constrained_failure_modes.csv",
        "integrated_degeneracy_gate_matrix": paths["reviewer"] / "integrated_degeneracy_gate_matrix.csv",
        "degeneracy_level_table": paths["reviewer"] / "degeneracy_level_table.csv",
        "restricted_validation_claims": paths["reviewer"] / "restricted_validation_claims.csv",
        "restricted_all_gate_join": paths["reviewer"] / "restricted_all_gate_join.csv",
        "K_o_homeostasis_endpoint_audit": paths["predictive"] / "K_o_homeostasis_endpoint_audit.csv",
        "claim_to_artifact_ledger": paths["reviewer"] / "claim_to_artifact_ledger.csv",
        "intercluster_interpolation_acceptance": paths["mechanisms"] / "intercluster_interpolation_acceptance.csv",
        "phenotype_threshold_sensitivity": paths["mechanisms"] / "phenotype_threshold_sensitivity.csv",
        "all_current_assumption_sensitivity": paths["assumption"] / "all_current_assumption_sensitivity.csv",
        "cell_specific_identifiability_audit": paths["parameter"] / "cell_specific_identifiability_audit.csv",
        "parameter_ranges_citation_audit": paths["parameter"] / "parameter_ranges_citation_audit.csv",
        "selected_action_strategy_comparison": paths["reviewer"] / "selected_action_strategy_comparison.csv",
        "selected_action_scientific_value_assessment": paths["reviewer"] / "selected_action_scientific_value_assessment.csv",
        "selected_action_results_summary": paths["reviewer"] / "selected_action_results_summary.csv",
        "notebook_update_screen_after_selected_actions": paths["reviewer"] / "notebook_update_screen_after_selected_actions.csv",
    }
    for key, path in output_map.items():
        frame = outputs.get(key)
        if frame is not None:
            frame.to_csv(path, index=False)
    summary = {
        "step_name": "selected reviewer-response action gate audits",
        "config": asdict(config),
        "n_artifacts": int(len([k for k in output_map if k in outputs])),
        "elapsed_seconds": float(elapsed),
        "claim_scope": "Derived gate audits only; stronger claims require passing integrated gates.",
    }
    (paths["reviewer"] / "selected_action_gate_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def run_reviewer_gate_audits(
    project_root: Path | str,
    config: ReviewerGateAuditConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the selected Phase 1 and Phase 2 gate audits."""

    root = Path(project_root).resolve()
    cfg = config or ReviewerGateAuditConfig()
    start = time.perf_counter()

    outputs: dict[str, pd.DataFrame] = {}
    outputs["selected_action_strategy_comparison"] = build_selected_action_strategy_comparison()
    outputs["phenotype_robustness_summary"] = build_phenotype_robustness_summary(root, cfg)
    outputs["stratum_support_gate"] = build_stratum_support_gate(root, cfg)
    outputs["prediction_limited_failure_modes"] = build_prediction_limited_failure_modes(root, cfg)
    outputs["assumption_gate_audit"] = build_assumption_gate_audit(root, cfg)
    outputs["proxy_exclusion_claim_sensitivity"] = build_proxy_exclusion_claim_sensitivity(root, cfg)
    outputs["parameter_semantics_audit"] = build_parameter_semantics_audit(root)
    outputs["full_accepted_parameter_audit"] = build_full_accepted_parameter_audit(root, cfg)
    outputs["parameter_interpretation_class_audit"] = build_parameter_interpretation_class_audit(root, outputs["parameter_semantics_audit"])
    outputs["constrained_failure_modes"] = build_constrained_failure_modes(root)
    outputs["K_o_homeostasis_endpoint_audit"] = build_ko_homeostasis_endpoint_audit(root, cfg)
    outputs["intercluster_interpolation_acceptance"] = build_intercluster_interpolation_acceptance(root, cfg)
    outputs["phenotype_threshold_sensitivity"] = build_phenotype_threshold_sensitivity(root, cfg)
    outputs["all_current_assumption_sensitivity"] = build_all_current_assumption_sensitivity(root, cfg)
    outputs["cell_specific_identifiability_audit"] = build_cell_specific_identifiability_audit(root, outputs["full_accepted_parameter_audit"])
    outputs["parameter_ranges_citation_audit"] = build_parameter_ranges_citation_audit(root, outputs["parameter_semantics_audit"])
    outputs["notebook_update_screen_after_selected_actions"] = build_notebook_update_screen_after_selected_actions(root)

    if cfg.write_outputs:
        _write_outputs(root, outputs, cfg, time.perf_counter() - start)

    outputs["integrated_degeneracy_gate_matrix"] = build_integrated_degeneracy_gate_matrix(root, cfg)
    outputs["degeneracy_level_table"] = build_degeneracy_level_table(outputs["integrated_degeneracy_gate_matrix"])
    outputs["restricted_validation_claims"] = build_restricted_validation_claims(root, outputs["integrated_degeneracy_gate_matrix"])
    outputs["restricted_all_gate_join"] = build_restricted_all_gate_join(outputs["integrated_degeneracy_gate_matrix"])
    if cfg.write_outputs:
        _write_outputs(root, outputs, cfg, time.perf_counter() - start)
    outputs["claim_to_artifact_ledger"] = build_claim_to_artifact_ledger(root)
    outputs["selected_action_scientific_value_assessment"] = build_scientific_value_assessment(root, outputs)
    outputs["selected_action_results_summary"] = build_selected_action_results_summary(root, outputs)
    if cfg.write_outputs:
        _write_outputs(root, outputs, cfg, time.perf_counter() - start)
    return outputs
