"""Step 08 parameter-plausibility and constrained-rerun screens.

This module audits accepted cell-specific candidates against broad plausibility
ranges and Step 03 identifiability evidence.  It intentionally separates raw
parameter bounds from reviewer-facing interpretation: being inside a range is not
enough when Vm-only profiles or structural confounding make a coordinate weakly
identified.  The constrained screen is a lightweight projection/rescoring pass,
not a replacement Step 04 optimizer.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .astro_model import VALID_CURRENTS, build_paramdict, simulate_with_hidden_outputs
from .mechanisms import compute_flux_summary
from .parameter_space import EFFECTIVE_COORDINATES
from .protocols import protocol_condition, stim_window_seconds
from .step05_mechanistic_decomposition import reconstruct_flat_params
from .step06_predictive_validation import Step06Config, load_step06_inputs

OUTPUT_SUBDIR = "parameter_plausibility"
IDENTITY_COLUMNS = ["file_id", "region", "condition", "candidate_id"]
RAW_PARAMETERS = ("gki", "eps", "gl_a", "zth", "zs")
EFFECTIVE_PARAMETERS = tuple(EFFECTIVE_COORDINATES)
AUDITED_PARAMETERS = RAW_PARAMETERS + EFFECTIVE_PARAMETERS
FINAL_CLAIM_TEXT = (
    "Step 08 supports parameter-interpretability guardrails, but final "
    "biological degeneracy claims remain disabled until Step 09 synthesis."
)
STEP08_CONTRACT_ID = "step08_all_valid_currents_plausibility_flux_rescore_v1"


@dataclass(slots=True)
class Step08Config:
    """Configuration for Step 08.

    ``max_candidates`` is an optional development/test cap.  The default
    reviewer-facing execution keeps all selected cell-level candidates.
    """

    max_candidates: int | None = None
    candidate_policy: str = "best_per_cell"
    constrained_max_candidates: int | None = None
    time_points: int = 40
    t_final_ms: float = 50_000.0
    currents_na: tuple[int, ...] = tuple(int(c) for c in VALID_CURRENTS)
    min_holdout_pass_fraction: float = 0.30
    max_trace_rmse_mV: float = 25.0
    boundary_fraction: float = 0.05
    acceptable_fit_degradation_fraction: float = 0.15
    mechanism_flux_fraction_delta_max: float = 0.20
    changed_fit_degradation_warn_mV: float = 5.0
    parameter_ranges_path: str | None = None
    write_outputs: bool = True


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _time_grid(config: Step08Config) -> np.ndarray:
    return np.linspace(0.0, float(config.t_final_ms), int(config.time_points), dtype=float)


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def default_parameter_ranges() -> pd.DataFrame:
    """Return broad plausibility guardrails for raw and effective coordinates."""

    rows = [
        ("gki", "raw", 1.0, 300.0, "broad_model_guardrail", "Kir conductance kept within broad positive astrocyte-model scale"),
        ("eps", "raw", 1e-5, 0.20, "broad_model_guardrail", "extracellular exchange factor must be positive and sub-unity"),
        ("gl_a", "raw", 1e-4, 50.0, "broad_model_guardrail", "leak conductance kept positive and below dominant Kir scale"),
        ("zth", "raw", 1e-3, 5.0, "broad_model_guardrail", "gating threshold kept positive on modeled potassium-deviation scale"),
        ("zs", "raw", 1e-4, 5.0, "broad_model_guardrail", "gating slope/scale kept positive on modeled potassium-deviation scale"),
        ("P_gap_eff", "effective", 1e-8, 5e-3, "effective_coordinate_guardrail", "interpretable product d × pk for gap coupling"),
        ("gamma_t_eff", "effective", 1e-8, 5e-2, "effective_coordinate_guardrail", "transport effective coordinate normalized by volume/Faraday terms"),
        ("gamma_s_eff", "effective", 1e-8, 5e-2, "effective_coordinate_guardrail", "syncytial effective coordinate normalized by volume/Faraday terms"),
        ("volume_ratio_wa_wo", "effective", 1e-3, 1e3, "effective_coordinate_guardrail", "cell/extracellular volume ratio effective coordinate"),
    ]
    return pd.DataFrame(
        rows,
        columns=["parameter", "coordinate_type", "lower_bound", "upper_bound", "range_source", "range_basis"],
    )


def validate_parameter_ranges(ranges: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a parameter-range table before audit use."""

    required = {
        "parameter",
        "coordinate_type",
        "lower_bound",
        "upper_bound",
        "range_source",
        "range_basis",
    }
    missing = sorted(required - set(ranges.columns))
    if missing:
        raise ValueError(f"Parameter range table is missing required columns: {missing}")
    out = ranges.copy()
    out["lower_bound"] = pd.to_numeric(out["lower_bound"], errors="coerce")
    out["upper_bound"] = pd.to_numeric(out["upper_bound"], errors="coerce")
    invalid = out["lower_bound"].isna() | out["upper_bound"].isna() | (
        out["lower_bound"] >= out["upper_bound"]
    )
    if bool(invalid.any()):
        bad = out.loc[invalid, "parameter"].astype(str).tolist()
        raise ValueError(f"Invalid lower/upper bounds for parameters: {bad}")
    missing_params = sorted(set(AUDITED_PARAMETERS) - set(out["parameter"].astype(str)))
    if missing_params:
        raise ValueError(f"Parameter range table is missing audited parameters: {missing_params}")
    return out.reset_index(drop=True)


def load_or_create_parameter_ranges(
    project_root: Path | str, configured_path: str | None = None
) -> tuple[pd.DataFrame, Path, str]:
    """Load an editable range table or create the default one if absent."""

    root = Path(project_root).resolve()
    path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else root / "outputs" / OUTPUT_SUBDIR / "parameter_ranges.csv"
    )
    if path.exists():
        return validate_parameter_ranges(pd.read_csv(path)), path, "reused_existing_editable_csv"
    ranges = default_parameter_ranges()
    path.parent.mkdir(parents=True, exist_ok=True)
    ranges.to_csv(path, index=False)
    return validate_parameter_ranges(ranges), path, "created_default_editable_csv"


def _profile_to_status(profile_class: str) -> str:
    text = str(profile_class).lower()
    if any(token in text for token in ("flat", "broad", "boundary", "unbounded", "sloppy", "weak")):
        return "weakly_identified"
    if any(token in text for token in ("clear", "ident", "stiff", "well")):
        return "identifiable"
    return "not_profiled"


def load_identifiability_status(project_root: Path | str) -> pd.DataFrame:
    """Load Step 03 identifiability evidence with conservative fallbacks."""

    root = Path(project_root).resolve()
    rows: list[dict[str, Any]] = []

    profile_path = root / "outputs" / "identifiability" / "profile_summary.csv"
    if profile_path.exists():
        profile = pd.read_csv(profile_path)
        for _, row in profile.iterrows():
            parameter = str(row.get("profile_parameter", row.get("parameter", "")))
            if not parameter:
                continue
            rows.append(
                {
                    "parameter": parameter,
                    "identifiability_status": _profile_to_status(row.get("profile_classification", row.get("profile_class", ""))),
                    "identifiability_source": "step03_profile_summary",
                    "identifiability_evidence": str(row.get("profile_classification", row.get("profile_class", "not_profiled"))),
                }
            )

    map_path = root / "outputs" / "identifiability" / "effective_parameter_map.csv"
    if map_path.exists():
        param_map = pd.read_csv(map_path)
        for _, row in param_map.iterrows():
            parameter = str(row.get("parameter", ""))
            if not parameter:
                continue
            classification = str(row.get("identifiability_class", row.get("classification", ""))).lower()
            if parameter in EFFECTIVE_PARAMETERS or str(row.get("coordinate_type", "")).lower() == "effective":
                status = "effective_only"
            elif any(token in classification for token in ("weak", "confound", "structural")):
                status = "weakly_identified"
            elif "direct" in classification or "ident" in classification:
                status = "identifiable"
            else:
                status = "not_profiled"
            rows.append(
                {
                    "parameter": parameter,
                    "identifiability_status": status,
                    "identifiability_source": "step03_effective_parameter_map",
                    "identifiability_evidence": str(row.get("reviewer_interpretation", row.get("classification", ""))),
                }
            )

    if not rows:
        rows = [
            {
                "parameter": p,
                "identifiability_status": "effective_only" if p in EFFECTIVE_PARAMETERS else "not_profiled",
                "identifiability_source": "step08_fallback_no_step03_file",
                "identifiability_evidence": "No Step 03 file available; interpretation downgraded.",
            }
            for p in AUDITED_PARAMETERS
        ]

    df = pd.DataFrame(rows)
    priority = {"weakly_identified": 0, "not_profiled": 1, "effective_only": 2, "identifiable": 3}
    df["_priority"] = df["identifiability_status"].map(priority).fillna(1)
    best = df.sort_values(["parameter", "_priority"]).groupby("parameter", as_index=False).first()
    best = best.drop(columns=["_priority"])

    present = set(best["parameter"])
    missing = [p for p in AUDITED_PARAMETERS if p not in present]
    if missing:
        best = pd.concat(
            [
                best,
                pd.DataFrame(
                    [
                        {
                            "parameter": p,
                            "identifiability_status": "effective_only" if p in EFFECTIVE_PARAMETERS else "not_profiled",
                            "identifiability_source": "step08_default_missing_parameter",
                            "identifiability_evidence": "Parameter absent from Step 03 profile outputs; downgraded unless effective coordinate.",
                        }
                        for p in missing
                    ]
                ),
            ],
            ignore_index=True,
        )
    return best


def load_step08_inputs(project_root: Path | str, config: Step08Config) -> pd.DataFrame:
    """Load accepted candidates with mechanism labels and prediction metrics."""

    candidates, _ = load_step06_inputs(
        project_root,
        Step06Config(
            max_candidates=config.max_candidates,
            candidate_policy=config.candidate_policy,
            min_holdout_pass_fraction=config.min_holdout_pass_fraction,
            max_trace_rmse_mV=config.max_trace_rmse_mV,
            write_outputs=False,
        ),
    )
    required = set(IDENTITY_COLUMNS + list(AUDITED_PARAMETERS) + ["mechanism_cluster", "dominant_mechanism"])
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Step 08 inputs missing required columns: {missing}")
    return candidates.reset_index(drop=True)


def merge_candidate_annotations(inputs: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    """Return a candidate table with Step 08 annotation columns.

    Earlier Step 08 drafts passed a small bundle into this helper.  The current
    pipeline already loads merged candidate annotations, so this function is a
    compatibility and validation layer used by integration tests and notebooks.
    """

    if isinstance(inputs, pd.DataFrame):
        candidates = inputs.copy()
    elif "candidates" in inputs:  # type: ignore[operator]
        candidates = pd.DataFrame(inputs["candidates"]).copy()  # type: ignore[index]
    else:
        candidates = pd.DataFrame(inputs).copy()

    for column, default in (
        ("mechanism_cluster", "unlabeled"),
        ("dominant_mechanism", "unknown"),
        ("holdout_mean_rmse_mV", np.nan),
        ("holdout_mean_pass_fraction", np.nan),
    ):
        if column not in candidates.columns:
            candidates[column] = default
    return candidates.reset_index(drop=True)


def _interpretability_guardrail(plausibility: str, identifiability: str, coordinate_type: str) -> tuple[bool, str]:
    if plausibility != "within_range":
        return False, "not physiologically interpretable because the value is outside the broad plausibility range or missing"
    if identifiability == "identifiable":
        return True, "within broad range and supported by available identifiability evidence"
    if identifiability == "effective_only" and coordinate_type == "effective":
        return True, "within broad range and interpretable only as an effective coordinate"
    if identifiability == "effective_only":
        return False, "raw coordinate is only interpretable through an effective combination"
    if identifiability == "weakly_identified":
        return False, "within broad range but Step 03 indicates weak/sloppy/broad identifiability"
    return False, "within broad range but not profiled; interpretation downgraded"


def _looks_like_range_table(frame: Any) -> bool:
    return isinstance(frame, pd.DataFrame) and {"parameter", "lower_bound", "upper_bound"}.issubset(frame.columns)


def _looks_like_identifiability_table(frame: Any) -> bool:
    return isinstance(frame, pd.DataFrame) and {"parameter", "identifiability_status"}.issubset(frame.columns)


def _default_identifiability_for_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parameter": p,
                "identifiability_status": "effective_only" if p in EFFECTIVE_PARAMETERS else "not_profiled",
                "identifiability_source": "step08_default_no_project_root",
                "identifiability_evidence": "No project-root Step 03 lookup supplied; interpretation downgraded unless effective coordinate.",
            }
            for p in AUDITED_PARAMETERS
        ]
    )


def build_parameter_range_audit(
    candidates: pd.DataFrame,
    ranges: pd.DataFrame | Mapping[str, Any] | Step08Config | None = None,
    identifiability: pd.DataFrame | Step08Config | None = None,
) -> pd.DataFrame:
    """Build candidate × parameter plausibility and identifiability rows.

    The preferred call is ``build_parameter_range_audit(candidates, ranges,
    identifiability)``.  For compatibility with the reviewer-requested
    integration tests, ``ranges`` may also be an input bundle/candidate table and
    ``identifiability`` may be a ``Step08Config``; in that case default ranges
    and conservative identifiability fallbacks are used.
    """

    range_table = ranges if _looks_like_range_table(ranges) else default_parameter_ranges()
    ident_table = identifiability if _looks_like_identifiability_table(identifiability) else _default_identifiability_for_audit()

    range_map = range_table.set_index("parameter").to_dict("index")
    ident_map = ident_table.set_index("parameter").to_dict("index")
    rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        base = {c: cand.get(c) for c in IDENTITY_COLUMNS}
        base.update(
            {
                "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                "dominant_mechanism": cand.get("dominant_mechanism", "unknown"),
                "heldout_mean_rmse_mV": _as_float(cand.get("holdout_mean_rmse_mV", cand.get("mean_trace_rmse_mV"))),
                "holdout_mean_pass_fraction": _as_float(cand.get("holdout_mean_pass_fraction", cand.get("mean_weighted_pass_fraction"))),
            }
        )
        for parameter in AUDITED_PARAMETERS:
            r = range_map[parameter]
            ident = ident_map.get(parameter, {})
            value = _as_float(cand.get(parameter))
            if not np.isfinite(value):
                plausibility = "missing_value"
            elif float(r["lower_bound"]) <= value <= float(r["upper_bound"]):
                plausibility = "within_range"
            else:
                plausibility = "out_of_range"
            below_lower_bound = bool(np.isfinite(value) and value < float(r["lower_bound"]))
            above_upper_bound = bool(np.isfinite(value) and value > float(r["upper_bound"]))
            ident_status = str(ident.get("identifiability_status", "not_profiled"))
            interpretable, guardrail = _interpretability_guardrail(plausibility, ident_status, str(r["coordinate_type"]))
            rows.append(
                {
                    **base,
                    "parameter": parameter,
                    "coordinate_type": r["coordinate_type"],
                    "value": value,
                    "lower_bound": float(r["lower_bound"]),
                    "upper_bound": float(r["upper_bound"]),
                    "below_lower_bound": below_lower_bound,
                    "above_upper_bound": above_upper_bound,
                    "within_lower_bound": bool(np.isfinite(value) and value >= float(r["lower_bound"])),
                    "within_upper_bound": bool(np.isfinite(value) and value <= float(r["upper_bound"])),
                    "bound_violation": (
                        "below_lower_bound"
                        if below_lower_bound
                        else ("above_upper_bound" if above_upper_bound else "none")
                    ),
                    "range_source": r["range_source"],
                    "range_basis": r["range_basis"],
                    "plausibility_status": plausibility,
                    "identifiability_status": ident_status,
                    "identifiability_source": ident.get("identifiability_source", "step08_default"),
                    "identifiability_evidence": ident.get("identifiability_evidence", "not profiled"),
                    "physiologically_interpretable": bool(interpretable),
                    "interpretation_guardrail": guardrail,
                }
            )
    return pd.DataFrame(rows)


def build_effective_parameter_plausibility(audit: pd.DataFrame) -> pd.DataFrame:
    """Return only effective-coordinate plausibility rows."""

    cols = [
        *IDENTITY_COLUMNS,
        "mechanism_cluster",
        "dominant_mechanism",
        "parameter",
        "value",
        "lower_bound",
        "upper_bound",
        "plausibility_status",
        "identifiability_status",
        "physiologically_interpretable",
        "interpretation_guardrail",
    ]
    out = audit[audit["coordinate_type"].eq("effective")].copy()
    return out[cols].reset_index(drop=True)


def _project_candidate_to_ranges(candidate: Mapping[str, Any], ranges: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    range_map = ranges.set_index("parameter").to_dict("index")
    projected: dict[str, float] = {}
    changed: list[str] = []
    for parameter in AUDITED_PARAMETERS:
        value = _as_float(candidate.get(parameter))
        if not np.isfinite(value):
            continue
        lower = float(range_map[parameter]["lower_bound"])
        upper = float(range_map[parameter]["upper_bound"])
        clipped = float(np.clip(value, lower, upper))
        projected[parameter] = clipped
        if not np.isclose(value, clipped, rtol=1e-10, atol=1e-12):
            changed.append(parameter)
    return projected, changed


def _constrained_candidate_from_audit(
    candidate: Mapping[str, Any], parameter_audit: pd.DataFrame
) -> tuple[dict[str, Any], list[str]]:
    adjusted = dict(candidate)
    changed: list[str] = []
    if parameter_audit.empty:
        return adjusted, changed
    mask = np.ones(len(parameter_audit), dtype=bool)
    for col in IDENTITY_COLUMNS:
        if col in parameter_audit.columns:
            mask &= parameter_audit[col].astype(str).to_numpy() == str(candidate.get(col))
    rows = parameter_audit.loc[mask]
    if rows.empty and {"parameter", "lower_bound", "upper_bound"}.issubset(parameter_audit.columns):
        rows = parameter_audit
    for _, row in rows.iterrows():
        parameter = str(row.get("parameter"))
        if parameter not in AUDITED_PARAMETERS:
            continue
        value = _as_float(candidate.get(parameter))
        lower = _as_float(row.get("lower_bound"))
        upper = _as_float(row.get("upper_bound"))
        if not (np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper)):
            continue
        clipped = float(np.clip(value, lower, upper))
        adjusted[parameter] = clipped
        if not np.isclose(value, clipped, rtol=1e-10, atol=1e-12):
            changed.append(parameter)
    return adjusted, changed


def _simulate_vm_and_flux(candidate: Mapping[str, Any], current_na: int, config: Step08Config) -> tuple[np.ndarray, dict[str, Any]]:
    sweep = list(VALID_CURRENTS).index(int(current_na)) + 1 if int(current_na) in VALID_CURRENTS else 1
    params = reconstruct_flat_params(dict(candidate), current_na=int(current_na), sweep=sweep)
    paramdict = build_paramdict(protocol_condition(str(candidate["condition"])), int(current_na), params)
    sim = simulate_with_hidden_outputs(
        paramdict,
        {
            "experiment_type": protocol_condition(str(candidate["condition"])),
            "current_na": int(current_na),
            "sim_time_ms": _time_grid(config),
        },
    )
    window = stim_window_seconds(str(candidate["condition"]))
    flux = compute_flux_summary(sim, stim_window_s=window)
    return np.asarray(sim["Vm"], dtype=float), flux


def _dominant_from_flux(flux: Mapping[str, Any]) -> str:
    fractions = {
        "gap": abs(float(flux.get("gap_fraction", 0.0))),
        "kir": abs(float(flux.get("kir_fraction", 0.0))),
        "leak": abs(float(flux.get("leak_fraction", 0.0))),
    }
    if not any(np.isfinite(v) for v in fractions.values()):
        return "unknown"
    if max(fractions.values()) - min(fractions.values()) < 0.05:
        return "Mixed"
    return max(fractions, key=fractions.get)


def build_constrained_rerun_comparison(candidates: pd.DataFrame, parameter_audit: pd.DataFrame, config: Step08Config) -> pd.DataFrame:
    """Compare unconstrained candidates with broad-range constrained projections.

    One row is produced for each constrained candidate/current pair.  The screen
    re-simulates both the original and clipped candidates, computes flux
    fractions, and only marks mechanism conclusions as persistent if both trace
    degradation and flux-mechanism changes remain small.
    """

    if _looks_like_range_table(parameter_audit) and "plausibility_status" not in parameter_audit.columns:
        parameter_audit = build_parameter_range_audit(candidates, parameter_audit, _default_identifiability_for_audit())

    rows: list[dict[str, Any]] = []
    selected = (
        candidates.copy()
        if config.constrained_max_candidates is None
        else candidates.head(int(config.constrained_max_candidates)).copy()
    )
    for _, cand in selected.iterrows():
        constrained, adjusted_parameters = _constrained_candidate_from_audit(cand.to_dict(), parameter_audit)
        unconstrained_rmse = _as_float(cand.get("holdout_mean_rmse_mV", cand.get("mean_trace_rmse_mV")))
        unconstrained_pass = _as_float(cand.get("holdout_mean_pass_fraction", cand.get("mean_weighted_pass_fraction")))
        base = {
            **{c: cand.get(c) for c in IDENTITY_COLUMNS},
            "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
            "dominant_mechanism": cand.get("dominant_mechanism", "unknown"),
            "mechanism_cluster_unconstrained": cand.get("mechanism_cluster", "unlabeled"),
            "constrained_screen_type": "broad_range_projection_not_full_optimizer",
            "comparison_kind": "step04_unconstrained_candidate_vs_broad_range_projection",
            "unconstrained_source": "step04_accepted_ensemble",
            "changed_by_constraints": bool(adjusted_parameters),
            "changed_parameters": ";".join(adjusted_parameters),
            "adjusted_parameters": ";".join(adjusted_parameters),
            "n_changed_parameters": int(len(adjusted_parameters)),
            "unconstrained_mean_trace_rmse_mV": unconstrained_rmse,
            "unconstrained_holdout_rmse_mV": unconstrained_rmse,
            "unconstrained_holdout_pass_fraction": unconstrained_pass,
        }
        for current_na in config.currents_na:
            current_base = {
                **base,
                "current_na": int(current_na),
                "step08_contract_id": STEP08_CONTRACT_ID,
            }
            try:
                vm_unconstrained, flux_un = _simulate_vm_and_flux(cand.to_dict(), int(current_na), config)
                vm_constrained, flux_con = _simulate_vm_and_flux(constrained, int(current_na), config)
                trace_delta = float(np.sqrt(np.mean((vm_constrained - vm_unconstrained) ** 2)))
                fit_degradation_fraction = trace_delta / max(abs(unconstrained_rmse), 1.0)
                dominant_un = _dominant_from_flux(flux_un)
                dominant_con = _dominant_from_flux(flux_con)
                flux_delta = max(
                    abs(float(flux_con.get("gap_fraction", 0.0)) - float(flux_un.get("gap_fraction", 0.0))),
                    abs(float(flux_con.get("kir_fraction", 0.0)) - float(flux_un.get("kir_fraction", 0.0))),
                    abs(float(flux_con.get("leak_fraction", 0.0)) - float(flux_un.get("leak_fraction", 0.0))),
                )
                mechanism_changed = dominant_un != dominant_con or flux_delta > config.mechanism_flux_fraction_delta_max
                persists = bool(
                    fit_degradation_fraction <= config.acceptable_fit_degradation_fraction
                    and not mechanism_changed
                    and str(base["mechanism_cluster_unconstrained"]) != "unlabeled"
                )
                constrained_pass = max(0.0, unconstrained_pass - fit_degradation_fraction) if np.isfinite(unconstrained_pass) else float("nan")
                claim_status = "claim_persists_under_broad_constraints" if persists else "claim_downgraded_by_constraint_screen"
                rows.append(
                    {
                        **current_base,
                        "simulation_status": "ok",
                        "failure_reason": "",
                        "constrained_trace_delta_rmse_mV": trace_delta,
                        "fit_degradation_fraction": fit_degradation_fraction,
                        "constrained_holdout_rmse_mV": unconstrained_rmse + trace_delta if np.isfinite(unconstrained_rmse) else float("nan"),
                        "delta_holdout_rmse_mV": trace_delta,
                        "constrained_holdout_pass_fraction": constrained_pass,
                        "constrained_holdout_pass_fraction_estimate": constrained_pass,
                        "prediction_persists_under_constraints": bool(np.isfinite(constrained_pass) and constrained_pass >= config.min_holdout_pass_fraction),
                        "dominant_mechanism_unconstrained": dominant_un,
                        "dominant_mechanism_constrained": dominant_con,
                        "mechanism_flux_fraction_delta_max": float(flux_delta),
                        "mechanism_changed_under_constraints": bool(mechanism_changed),
                        "mechanism_cluster_constrained": base["mechanism_cluster_unconstrained"],
                        "mechanism_persists_under_constraints": persists,
                        "mechanism_conclusion_persists": persists,
                        "constrained_claim_status": claim_status,
                        "constraint_action": "not_required" if not adjusted_parameters else "clipped_to_broad_plausibility_range",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - explicit audit rows are required
                rows.append(
                    {
                        **current_base,
                        "simulation_status": "failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "constrained_trace_delta_rmse_mV": np.nan,
                        "fit_degradation_fraction": np.nan,
                        "constrained_holdout_rmse_mV": np.nan,
                        "delta_holdout_rmse_mV": np.nan,
                        "constrained_holdout_pass_fraction": np.nan,
                        "constrained_holdout_pass_fraction_estimate": np.nan,
                        "prediction_persists_under_constraints": False,
                        "dominant_mechanism_unconstrained": "unknown",
                        "dominant_mechanism_constrained": "unknown",
                        "mechanism_flux_fraction_delta_max": np.nan,
                        "mechanism_changed_under_constraints": True,
                        "mechanism_cluster_constrained": "unresolved",
                        "mechanism_persists_under_constraints": False,
                        "mechanism_conclusion_persists": False,
                        "constrained_claim_status": "claim_downgraded_by_constraint_screen",
                        "constraint_action": "simulation_failed",
                    }
                )
    return pd.DataFrame(rows)


def build_interpretability_status(audit: pd.DataFrame, constrained: pd.DataFrame) -> pd.DataFrame:
    """Summarize parameter audit into candidate-level claim status."""

    grouped = audit.groupby(IDENTITY_COLUMNS + ["mechanism_cluster", "dominant_mechanism"], dropna=False).agg(
        n_parameters_audited=("parameter", "size"),
        n_out_of_range=("plausibility_status", lambda s: int((s == "out_of_range").sum())),
        n_weakly_identified=("identifiability_status", lambda s: int((s == "weakly_identified").sum())),
        n_effective_only=("identifiability_status", lambda s: int((s == "effective_only").sum())),
        n_phys_interpretable=("physiologically_interpretable", lambda s: int(pd.Series(s).astype(bool).sum())),
    ).reset_index()
    constrained_summary = (
        constrained.groupby(IDENTITY_COLUMNS, dropna=False)
        .agg(
            prediction_persists_under_constraints=("prediction_persists_under_constraints", "all"),
            mechanism_persists_under_constraints=("mechanism_persists_under_constraints", "all"),
            n_constrained_current_rows=("current_na", "size"),
            max_mechanism_flux_fraction_delta=("mechanism_flux_fraction_delta_max", "max"),
        )
        .reset_index()
    )
    constrained_summary["constrained_claim_status"] = np.where(
        constrained_summary["prediction_persists_under_constraints"].astype(bool)
        & constrained_summary["mechanism_persists_under_constraints"].astype(bool),
        "claim_persists_under_broad_constraints",
        "claim_downgraded_by_constraint_screen",
    )
    merged = grouped.merge(
        constrained_summary,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    merged["prediction_persists_under_constraints"] = merged["prediction_persists_under_constraints"].fillna(False)
    merged["mechanism_persists_under_constraints"] = merged["mechanism_persists_under_constraints"].fillna(False)
    merged["constrained_claim_status"] = merged["constrained_claim_status"].fillna("not_selected_for_constrained_screen")

    def _status(row: pd.Series) -> str:
        if int(row["n_out_of_range"]) > 0:
            return "downgraded_out_of_range_parameters"
        if not bool(row.get("prediction_persists_under_constraints", False)) or not bool(row.get("mechanism_persists_under_constraints", False)):
            return "downgraded_constraint_sensitive"
        if int(row["n_phys_interpretable"]) < int(row["n_parameters_audited"]):
            return "partial_effective_or_weak_identifiability_only"
        return "physiologically_interpretable_candidate"

    merged["parameter_interpretability_status"] = merged.apply(_status, axis=1)
    merged["parameter_claim_allowed_after_step08"] = merged["parameter_interpretability_status"].isin(
        {"physiologically_interpretable_candidate", "partial_effective_or_weak_identifiability_only"}
    )
    merged["final_degeneracy_claim_allowed_after_step08"] = False
    merged["claim_scope_note"] = FINAL_CLAIM_TEXT
    return merged


def run_step08_parameter_plausibility(
    project_root: Path | str,
    config: Step08Config | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the full Step 08 audit and optionally write outputs."""

    root = Path(project_root).resolve()
    config = config or Step08Config()
    out_dir = Path(output_dir).resolve() if output_dir is not None else root / "outputs" / OUTPUT_SUBDIR

    candidates = load_step08_inputs(root, config)
    ranges, ranges_path, ranges_status = load_or_create_parameter_ranges(
        root, configured_path=config.parameter_ranges_path
    )
    identifiability = load_identifiability_status(root)
    audit = build_parameter_range_audit(candidates, ranges, identifiability)
    effective = build_effective_parameter_plausibility(audit)
    constrained = build_constrained_rerun_comparison(candidates, audit, config)
    status = build_interpretability_status(audit, constrained)

    summary = {
        "step": "08_parameter_plausibility_and_constrained_reruns",
        "config": asdict(config),
        "n_candidates": int(candidates[IDENTITY_COLUMNS].drop_duplicates().shape[0]),
        "n_parameter_rows": int(audit.shape[0]),
        "n_effective_rows": int(effective.shape[0]),
        "n_out_of_range_rows": int((audit["plausibility_status"] == "out_of_range").sum()),
        "n_phys_interpretable_rows": int(audit["physiologically_interpretable"].astype(bool).sum()),
        "n_constrained_rerun_rows": int(constrained.shape[0]),
        "currents_na": [int(c) for c in config.currents_na],
        "parameter_ranges_path": str(ranges_path),
        "parameter_ranges_status": ranges_status,
        "final_degeneracy_claim_allowed_after_step08": False,
        "claim_scope": FINAL_CLAIM_TEXT,
    }

    result = {
        "candidates": candidates,
        "parameter_ranges": ranges,
        "identifiability_status": identifiability,
        "parameter_range_audit": audit,
        "effective_parameter_plausibility": effective,
        "constrained_rerun_comparison": constrained,
        "interpretability_status": status,
        "analysis_summary": summary,
    }

    if config.write_outputs:
        _ensure_dir(out_dir)
        audit.to_csv(out_dir / "parameter_range_audit.csv", index=False)
        effective.to_csv(out_dir / "effective_parameter_plausibility.csv", index=False)
        constrained.to_csv(out_dir / "constrained_rerun_comparison.csv", index=False)
        status.to_csv(out_dir / "interpretability_status.csv", index=False)
        ranges.to_csv(out_dir / "parameter_ranges.csv", index=False)
        identifiability.to_csv(out_dir / "identifiability_status.csv", index=False)
        with (out_dir / "analysis_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return result


def compare_step08_runtime_presets(project_root: Path | str, max_candidates: int = 1) -> pd.DataFrame:
    """Benchmark coarse/default Step 08 presets for tuning decisions."""

    rows: list[dict[str, Any]] = []
    for preset, n in [("coarse", max_candidates), ("default", max(1, min(3, max_candidates + 1)))]:
        cfg = Step08Config(max_candidates=n, write_outputs=False)
        start = time.perf_counter()
        result = run_step08_parameter_plausibility(project_root, cfg)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "preset": preset,
                "max_candidates": n,
                "elapsed_seconds": elapsed,
                "n_parameter_rows": int(result["parameter_range_audit"].shape[0]),
                "tuning_recommendation": "use_default_for_notebook" if elapsed < 30 else "use_coarse_for_interactive_runs",
            }
        )
    return pd.DataFrame(rows)


def compare_step08_performance(project_root: Path | str, max_candidates: int = 1) -> pd.DataFrame:
    return compare_step08_runtime_presets(project_root, max_candidates=max_candidates)
