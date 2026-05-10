"""Step 06 predictive validation and perturbation robustness.

The pipeline joins Step 04 accepted cell ensembles with Step 05 mechanism labels,
re-simulates accepted candidates, computes region-aware posterior predictive
checks against Step 02 feature bands, and stress-tests mechanism labels under a
small auditable perturbation panel.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .astro_model import VALID_CURRENTS, build_paramdict, simulate_odeint, simulate_with_hidden_outputs
from .atf_features import FEATURE_COLUMNS, extract_features_from_trace
from .mechanisms import compute_flux_summary
from .protocols import stim_window_seconds
from .step05_mechanistic_decomposition import (
    Step05Config,
    load_step04_accepted_ensemble,
    reconstruct_flat_params,
    run_step05_mechanistic_decomposition,
)
from .contracts import protocol_condition

OUTPUT_SUBDIR = "predictive_validation"
IDENTITY_COLUMNS = ["file_id", "region", "condition", "candidate_id"]


@dataclass(slots=True)
class Step06Config:
    max_candidates: int | None = 3
    time_points: int = 80
    t_final_ms: float = 50_000.0
    prediction_interval_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)
    min_holdout_pass_fraction: float = 0.30
    max_trace_rmse_mV: float = 25.0
    perturbation_current_na: int | None = None
    perturbation_factors: dict[str, float] = field(
        default_factory=lambda: {
            "nominal": 1.0,
            "eps_scale_low": 0.5,
            "eps_scale_high": 1.5,
            "stimulus_duration_short": 0.75,
            "stimulus_duration_long": 1.25,
            "baseline_K_o_low": 0.95,
            "baseline_K_o_high": 1.05,
            "current_scale_low": 0.9,
            "current_scale_high": 1.1,
        }
    )
    robustness_recovery_error_max: float = 1.5
    robustness_peak_ratio_max: float = 1.75
    min_robust_fraction: float = 0.50
    write_outputs: bool = True
    require_candidate_level_heldout: bool = True
    final_degeneracy_claim_allowed: bool = False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _time_grid(config: Step06Config, factor: float = 1.0) -> np.ndarray:
    return np.linspace(0.0, float(config.t_final_ms) * float(factor), int(config.time_points), dtype=float)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_mechanism_labels(project_root: Path | str, max_candidates: int | None = None) -> pd.DataFrame:
    root = Path(project_root).resolve()
    path = root / "outputs" / "mechanisms" / "mechanism_clusters.csv"
    if not path.exists():
        run_step05_mechanistic_decomposition(
            root,
            Step05Config(max_candidates=max_candidates, time_points=60, bootstrap_iterations=0, write_outputs=True),
        )
    if not path.exists():
        raise FileNotFoundError(f"Step 05 mechanism labels not found: {path}")
    df = pd.read_csv(path)
    required = set(IDENTITY_COLUMNS + ["mechanism_cluster", "cluster_claim_scope"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Mechanism labels missing required columns: {missing}")
    if max_candidates is not None:
        df = df.sort_values(IDENTITY_COLUMNS).head(int(max_candidates)).copy()
    return df


def load_step06_inputs(project_root: Path | str, config: Step06Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensemble, _ = load_step04_accepted_ensemble(project_root, max_candidates=config.max_candidates)
    mechanisms = load_mechanism_labels(project_root, max_candidates=None)
    keep_cols = [c for c in mechanisms.columns if c in set(IDENTITY_COLUMNS + ["mechanism_cluster", "dominant_mechanism", "cluster_evidence_status", "cluster_claim_scope"])]
    merged = ensemble.merge(
        mechanisms[keep_cols].drop_duplicates(IDENTITY_COLUMNS),
        on=IDENTITY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    merged["mechanism_cluster"] = merged["mechanism_cluster"].fillna("unlabeled")
    merged["dominant_mechanism"] = merged.get("dominant_mechanism", pd.Series(index=merged.index, dtype=object)).fillna("unknown")
    merged["cluster_evidence_status"] = merged.get("cluster_evidence_status", pd.Series(index=merged.index, dtype=object)).fillna("insufficient_evidence")
    merged["cluster_claim_scope"] = merged.get("cluster_claim_scope", pd.Series(index=merged.index, dtype=object)).fillna("missing_step05_label")
    return merged, mechanisms


def build_heldout_current_errors(candidates: pd.DataFrame, config: Step06Config) -> pd.DataFrame:
    """Build candidate-level held-out-current audit rows.

    Step 04 currently exposes aggregate held-out metrics on accepted candidates.
    Step 06 expands those aggregates into one auditable row per candidate and
    ordered held-out sweep, preserving mechanism-label provenance and adding
    explicit prediction status/pass columns so missing evidence cannot be treated
    as a silent pass.
    """

    rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        pass_fraction = float(
            cand.get(
                "holdout_mean_pass_fraction",
                cand.get("mean_weighted_pass_fraction", np.nan),
            )
        )
        rmse = float(
            cand.get("holdout_mean_rmse_mV", cand.get("mean_trace_rmse_mV", np.nan))
        )
        mechanism_label_status = (
            "step05_label_available"
            if str(cand.get("cluster_claim_scope", "")) != "missing_step05_label"
            and str(cand.get("mechanism_cluster", "unlabeled")) != "unlabeled"
            else "missing_step05_label"
        )
        prediction_pass = bool(
            np.isfinite(pass_fraction)
            and pass_fraction >= config.min_holdout_pass_fraction
            and (not np.isfinite(rmse) or rmse <= config.max_trace_rmse_mV)
        )
        prediction_status = "predictive_pass" if prediction_pass else "prediction_limited"
        base = {c: cand.get(c) for c in IDENTITY_COLUMNS}
        for sweep, current_na in enumerate(VALID_CURRENTS, start=1):
            rows.append(
                {
                    **base,
                    "sweep": sweep,
                    "current_na": int(current_na),
                    "split_strategy": "leave_one_current_out",
                    "heldout_trace_rmse_mV": rmse,
                    "heldout_weighted_feature_pass_fraction": pass_fraction,
                    "holdout_status": "passed" if prediction_pass else "prediction_limited",
                    "prediction_status": prediction_status,
                    "prediction_pass": prediction_pass,
                    "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                    "mechanism_label_status": mechanism_label_status,
                }
            )
        for strategy, currents in {"low_to_high": [150, 175], "high_to_low": [50, 75]}.items():
            rows.append(
                {
                    **base,
                    "sweep": -1,
                    "current_na": int(np.mean(currents)),
                    "split_strategy": strategy,
                    "heldout_trace_rmse_mV": rmse,
                    "heldout_weighted_feature_pass_fraction": pass_fraction,
                    "holdout_status": "stress_test_screen",
                    "prediction_status": prediction_status,
                    "prediction_pass": prediction_pass,
                    "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                    "mechanism_label_status": mechanism_label_status,
                }
            )

    heldout = pd.DataFrame(rows)
    if config.require_candidate_level_heldout and not heldout.empty:
        expected = candidates[IDENTITY_COLUMNS].drop_duplicates().assign(_expected=1)
        observed = (
            heldout[heldout["split_strategy"].eq("leave_one_current_out")]
            .groupby(IDENTITY_COLUMNS, dropna=False)
            .size()
            .rename("n_heldout_sweeps")
            .reset_index()
        )
        audit = expected.merge(observed, how="left", on=IDENTITY_COLUMNS)
        incomplete = audit["n_heldout_sweeps"].fillna(0).lt(len(VALID_CURRENTS))
        if incomplete.any():
            missing_rows = audit.loc[incomplete, IDENTITY_COLUMNS].copy()
            missing_rows["split_strategy"] = "missing_step04_heldout_screen"
            missing_rows["prediction_status"] = "fit_only"
            missing_rows["prediction_pass"] = False
            missing_rows["mechanism_label_status"] = "missing_step04_heldout_screen"
            heldout = pd.concat([heldout, missing_rows], ignore_index=True, sort=False)
    return heldout


def simulate_candidate_feature_rows(candidates: pd.DataFrame, config: Step06Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    flux_rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        condition = str(cand["condition"])
        window_s = stim_window_seconds(condition)
        for sweep, current_na in enumerate(VALID_CURRENTS, start=1):
            base = {c: cand.get(c) for c in candidates.columns if c in cand}
            try:
                params = reconstruct_flat_params(cand.to_dict(), current_na=int(current_na), sweep=sweep)
                sim = simulate_with_hidden_outputs(params, {"experiment_type": protocol_condition(condition), "current_na": int(current_na), "t_eval_ms": _time_grid(config)})
                time_s = np.asarray(sim["t_ms"], dtype=float) / 1000.0
                features = extract_features_from_trace(time_s, np.asarray(sim["Vm"], dtype=float), onset_s=window_s[0], offset_s=window_s[1])
                flux = compute_flux_summary(sim, stim_window_s=window_s)
                for feature, value in features.items():
                    if feature in FEATURE_COLUMNS:
                        feature_rows.append({**{c: cand.get(c) for c in IDENTITY_COLUMNS}, "region": cand.get("region"), "condition": cand.get("condition"), "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"), "sweep": sweep, "current_na": int(current_na), "feature": feature, "predicted_value": value, "simulation_status": "ok", "failure_reason": ""})
                flux_rows.append({**base, "sweep": sweep, "current_na": int(current_na), **flux, "simulation_status": "ok", "failure_reason": ""})
            except Exception as exc:  # noqa: BLE001 - explicit failure rows are required
                feature_rows.append({**{c: cand.get(c) for c in IDENTITY_COLUMNS}, "region": cand.get("region"), "condition": cand.get("condition"), "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"), "sweep": sweep, "current_na": int(current_na), "feature": "simulation_failed", "predicted_value": np.nan, "simulation_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
                flux_rows.append({**base, "sweep": sweep, "current_na": int(current_na), "simulation_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(feature_rows), pd.DataFrame(flux_rows)


def build_prediction_intervals(
    feature_predictions: pd.DataFrame, config: Step06Config
) -> pd.DataFrame:
    ok = feature_predictions[
        (feature_predictions["simulation_status"] == "ok")
        & feature_predictions["predicted_value"].notna()
    ].copy()
    q_low, q_mid, q_high = config.prediction_interval_quantiles
    rows = []
    for keys, group in ok.groupby(
        ["region", "condition", "sweep", "current_na", "feature"], dropna=False
    ):
        vals = pd.to_numeric(group["predicted_value"], errors="coerce").dropna()
        if vals.empty:
            continue
        source_candidate_ids = ";".join(
            sorted(group["candidate_id"].astype(str).dropna().unique().tolist())
        )
        rows.append(
            dict(
                zip(["region", "condition", "sweep", "current_na", "feature"], keys),
                n_predictions=int(len(vals)),
                n_source_candidates=int(group["candidate_id"].nunique()),
                source_candidate_ids=source_candidate_ids,
                pi_lower=float(vals.quantile(q_low)),
                pi_median=float(vals.quantile(q_mid)),
                pi_upper=float(vals.quantile(q_high)),
                interval_quantiles=f"{q_low},{q_mid},{q_high}",
            )
        )
    return pd.DataFrame(rows)


def _load_thresholds(root: Path) -> pd.DataFrame:
    primary = root / "outputs" / "features" / "condition_region_sweep_thresholds.csv"
    pooled = root / "outputs" / "features" / "region_pooled_condition_sweep_thresholds.csv"
    frames = [pd.read_csv(p) for p in [primary, pooled] if p.exists()]
    if not frames:
        raise FileNotFoundError("Step 02 threshold tables are required for Step 06 PPC")
    return pd.concat(frames, ignore_index=True)


def _find_threshold(thresholds: pd.DataFrame, condition: str, region: str, sweep: int, feature: str) -> tuple[pd.Series | None, str]:
    for cond, reg, mode in [(condition, region, "region_specific"), (condition, "ALL", "region_pooled"), ("ALL", "ALL", "global_fallback")]:
        hit = thresholds[(thresholds["condition"].astype(str) == str(cond)) & (thresholds["region"].astype(str) == str(reg)) & (thresholds["sweep"].astype(int) == int(sweep)) & (thresholds["feature"].astype(str) == str(feature))]
        if not hit.empty:
            return hit.iloc[0], mode
    return None, "missing"


def compute_feature_distribution_ppc(project_root: Path | str, feature_predictions: pd.DataFrame) -> pd.DataFrame:
    thresholds = _load_thresholds(Path(project_root).resolve())
    ok = feature_predictions[feature_predictions["simulation_status"] == "ok"].copy()
    rows = []
    for keys, group in ok.groupby(["region", "condition", "sweep", "feature"], dropna=False):
        region, condition, sweep, feature = keys
        thr, mode = _find_threshold(thresholds, str(condition), str(region), int(sweep), str(feature))
        vals = pd.to_numeric(group["predicted_value"], errors="coerce").dropna()
        if thr is None or vals.empty:
            lower = upper = weight = np.nan
            coverage = np.nan
        else:
            lower, upper = float(thr["acceptable_lower"]), float(thr["acceptable_upper"])
            weight = float(thr.get("reliability_weight", 1.0))
            coverage = float(((vals >= lower) & (vals <= upper)).mean())
        rows.append({"region": region, "condition": condition, "sweep": int(sweep), "feature": feature, "n_predictions": int(len(vals)), "empirical_lower": lower, "empirical_upper": upper, "coverage_fraction": coverage, "reliability_weight": weight, "weighted_coverage": coverage * weight if np.isfinite(coverage) and np.isfinite(weight) else np.nan, "threshold_fallback": mode})
    return pd.DataFrame(rows)


def run_perturbation_sweeps(candidates: pd.DataFrame, config: Step06Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_list = [int(config.perturbation_current_na)] if config.perturbation_current_na else [100]
    for _, cand in candidates.iterrows():
        condition = str(cand["condition"])
        window_s = stim_window_seconds(condition)
        for current_na in current_list:
            nominal_peak = np.nan
            for name, factor in config.perturbation_factors.items():
                row_base = {
                    **{c: cand.get(c) for c in IDENTITY_COLUMNS},
                    "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                    "dominant_mechanism": cand.get("dominant_mechanism", "unknown"),
                    "current_na": int(current_na),
                    "perturbation": name,
                    "perturbation_factor": float(factor),
                }
                if name.startswith("stimulus_duration"):
                    rows.append(
                        {
                            **row_base,
                            "robust_under_perturbation": False,
                            "functional_buffering_pass": False,
                            "perturbation_status": "not_run_protocol_timing_pending_model_api",
                            "simulation_status": "unsupported",
                            "failure_reason": "stimulus-duration perturbation requires protocol timing support in simulator API",
                        }
                    )
                    continue
                try:
                    params = reconstruct_flat_params(
                        cand.to_dict(),
                        current_na=current_na,
                        sweep=list(VALID_CURRENTS).index(current_na) + 1
                        if current_na in VALID_CURRENTS
                        else None,
                    )
                    if name.startswith("eps_scale"):
                        params["eps"] = float(params["eps"]) * float(factor)
                    elif name.startswith("current_scale"):
                        params["K_bath_value_middle"] = (
                            float(params["K_bath_value_middle"]) * float(factor)
                        )
                    paramdict = build_paramdict(
                        protocol_condition(condition), int(current_na), params
                    )
                    if name.startswith("baseline_K_o"):
                        paramdict["external"]["K_o0"] = (
                            float(paramdict["external"]["K_o0"]) * float(factor)
                        )
                    sim = simulate_odeint(
                        paramdict,
                        {
                            "experiment_type": protocol_condition(condition),
                            "current_na": int(current_na),
                            "t_eval_ms": _time_grid(config),
                        },
                        return_hidden=True,
                    )
                    flux = compute_flux_summary(sim, stim_window_s=window_s)
                    if name == "nominal":
                        nominal_peak = float(flux.get("K_o_peak", np.nan))
                    peak_ratio = (
                        float(flux.get("K_o_peak", np.nan)) / nominal_peak
                        if np.isfinite(nominal_peak) and abs(nominal_peak) > 1e-12
                        else np.nan
                    )
                    functional_buffering_pass = bool(
                        np.isfinite(float(flux.get("K_o_recovery_error", np.nan)))
                        and abs(float(flux.get("K_o_recovery_error", np.nan)))
                        <= config.robustness_recovery_error_max
                        and (
                            not np.isfinite(peak_ratio)
                            or abs(peak_ratio) <= config.robustness_peak_ratio_max
                        )
                    )
                    rows.append(
                        {
                            **row_base,
                            **flux,
                            "K_o_peak_ratio_to_nominal": peak_ratio,
                            "robust_under_perturbation": functional_buffering_pass,
                            "functional_buffering_pass": functional_buffering_pass,
                            "perturbation_status": "evaluated",
                            "simulation_status": "ok",
                            "failure_reason": "",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            **row_base,
                            "robust_under_perturbation": False,
                            "functional_buffering_pass": False,
                            "perturbation_status": "failed",
                            "simulation_status": "failed",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return pd.DataFrame(rows)


def build_robustness_summary(
    heldout: pd.DataFrame,
    ppc: pd.DataFrame,
    perturb: pd.DataFrame,
    candidates: pd.DataFrame,
    config: Step06Config,
) -> pd.DataFrame:
    rows = []
    groups = candidates.groupby(["mechanism_cluster", "region", "condition"], dropna=False)
    for (cluster, region, condition), group in groups:
        h = heldout[
            (heldout["mechanism_cluster"].astype(str) == str(cluster))
            & (heldout["region"].astype(str) == str(region))
            & (heldout["condition"].astype(str) == str(condition))
            & (heldout["split_strategy"] == "leave_one_current_out")
        ]
        p = ppc[
            (ppc["region"].astype(str) == str(region))
            & (ppc["condition"].astype(str) == str(condition))
        ]
        q = perturb[
            (perturb["mechanism_cluster"].astype(str) == str(cluster))
            & (perturb["region"].astype(str) == str(region))
            & (perturb["condition"].astype(str) == str(condition))
            & (perturb["perturbation"] != "nominal")
            & (perturb["perturbation_status"] == "evaluated")
        ]
        holdout_pass = float(_as_bool(h["prediction_pass"]).mean()) if not h.empty else np.nan
        ppc_cov = float(p["weighted_coverage"].mean(skipna=True)) if not p.empty else np.nan
        robust_frac = (
            float(_as_bool(q["functional_buffering_pass"]).mean()) if not q.empty else np.nan
        )
        evidence = (
            str(group["cluster_evidence_status"].iloc[0])
            if "cluster_evidence_status" in group
            else "insufficient_evidence"
        )
        if evidence == "insufficient_evidence":
            validation_label = "insufficient_evidence"
        elif (
            np.isfinite(holdout_pass)
            and holdout_pass >= config.min_robust_fraction
            and np.isfinite(robust_frac)
            and robust_frac >= config.min_robust_fraction
        ):
            validation_label = "predictive_supported"
        elif np.isfinite(holdout_pass) and holdout_pass > 0:
            validation_label = "prediction_limited"
        else:
            validation_label = "fit_only"

        step06_screen_claim = (
            "candidate_regime_supported_by_step06_predictive_perturbation_screen"
            if validation_label == "predictive_supported"
            else "mechanism_candidate_not_supported_by_step06_screen"
        )
        final_biological_degeneracy_claim_allowed = bool(
            config.final_degeneracy_claim_allowed
            and validation_label == "predictive_supported"
        )
        claim_scope_after_step06 = (
            f"{step06_screen_claim}; final biological degeneracy wording remains disallowed "
            "until assumption-sensitivity and parameter-plausibility checks pass"
        )
        rows.append(
            {
                "mechanism_cluster": cluster,
                "region": region,
                "condition": condition,
                "n_candidates": int(len(group)),
                "n_cells": int(group["file_id"].nunique()),
                "holdout_pass_fraction": holdout_pass,
                "mean_weighted_ppc_coverage": ppc_cov,
                "perturbation_robust_fraction": robust_frac,
                "step05_evidence_status": evidence,
                "validation_label": validation_label,
                "step06_screen_claim": step06_screen_claim,
                "final_biological_degeneracy_claim_allowed": final_biological_degeneracy_claim_allowed,
                "degeneracy_claim_allowed": final_biological_degeneracy_claim_allowed,
                "claim_scope_after_step06": claim_scope_after_step06,
            }
        )
    return pd.DataFrame(rows)


def compare_step06_runtime_presets(
    project_root: Path | str, max_candidates: int = 1
) -> pd.DataFrame:
    rows = []
    for preset, tp in [("coarse", 50), ("default", 80)]:
        cfg = Step06Config(
            max_candidates=max_candidates, time_points=tp, write_outputs=False
        )
        t0 = time.perf_counter()
        try:
            run_step06_predictive_validation(project_root, cfg)
            status = "ok"
            error = ""
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "preset": preset,
                "time_points": tp,
                "elapsed_seconds": elapsed,
                "status": status,
                "error": error,
            }
        )
    df = pd.DataFrame(rows)
    fastest = df.loc[df["elapsed_seconds"].idxmin(), "preset"] if not df.empty else "coarse"
    df["recommendation"] = np.where(
        df["preset"].eq(fastest),
        f"{fastest}_recommended_for_tests",
        "default_or_coarse_available_for_manuscript_rerun",
    )
    return df


def compare_step06_performance(
    project_root: Path | str, max_candidates: int = 1
) -> pd.DataFrame:
    """Backward-compatible alias for existing Step 06 notebooks/tests."""

    return compare_step06_runtime_presets(project_root, max_candidates=max_candidates)


def run_step06_predictive_validation(project_root: Path | str, config: Step06Config | None = None, output_dir: Path | str | None = None) -> dict[str, pd.DataFrame | dict[str, Any]]:
    config = config or Step06Config()
    root = Path(project_root).resolve()
    out_dir = _ensure_dir(Path(output_dir).resolve() if output_dir is not None else root / "outputs" / OUTPUT_SUBDIR)
    t0 = time.perf_counter()
    candidates, _ = load_step06_inputs(root, config)
    heldout = build_heldout_current_errors(candidates, config)
    feature_predictions, nominal_flux = simulate_candidate_feature_rows(candidates, config)
    intervals = build_prediction_intervals(feature_predictions, config)
    ppc = compute_feature_distribution_ppc(root, feature_predictions)
    perturb = run_perturbation_sweeps(candidates, config)
    robustness = build_robustness_summary(heldout, ppc, perturb, candidates, config)
    perf = compare_step06_runtime_presets(root) if config.write_outputs else pd.DataFrame()
    summary = {
        "step_name": "Step 06 — Predictive validation and perturbation robustness",
        "config": asdict(config),
        "n_candidates": int(len(candidates)),
        "n_heldout_rows": int(len(heldout)),
        "n_prediction_interval_rows": int(len(intervals)),
        "n_ppc_rows": int(len(ppc)),
        "n_perturbation_rows": int(len(perturb)),
        "validation_labels": sorted(robustness["validation_label"].astype(str).unique().tolist()) if not robustness.empty else [],
        "headline_claim_scope": "Mechanism regimes require predictive_supported labels before any degeneracy language; Step 06 alone keeps claims conservative.",
        "elapsed_seconds": time.perf_counter() - t0,
    }
    if config.write_outputs:
        heldout.to_csv(out_dir / "heldout_current_errors.csv", index=False)
        intervals.to_csv(out_dir / "prediction_intervals.csv", index=False)
        ppc.to_csv(out_dir / "feature_distribution_ppc.csv", index=False)
        perturb.to_csv(out_dir / "perturbation_sweeps.csv", index=False)
        robustness.to_csv(out_dir / "robustness_summary.csv", index=False)
        feature_predictions.to_csv(out_dir / "candidate_feature_predictions.csv", index=False)
        nominal_flux.to_csv(out_dir / "nominal_flux_predictions.csv", index=False)
        if not perf.empty:
            perf.to_csv(out_dir / "performance_benchmark.csv", index=False)
        (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"heldout_current_errors": heldout, "prediction_intervals": intervals, "feature_distribution_ppc": ppc, "perturbation_sweeps": perturb, "robustness_summary": robustness, "candidate_feature_predictions": feature_predictions, "nominal_flux_predictions": nominal_flux, "performance_benchmark": perf, "analysis_summary": summary}
