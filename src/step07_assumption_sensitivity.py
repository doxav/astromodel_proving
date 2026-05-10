"""Step 07 assumption-sensitivity analyses.

This module evaluates reviewer-facing model-assumption checks after the Step 04-06
candidate pipeline is available.  The implementation is intentionally scored and
contract-based rather than a new optimizer: every gating family, proxy check, and
compartment-split sensitivity row uses the same accepted candidates, currents,
time grid, and conservative claim rules.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .astro_model import VALID_CURRENTS, build_paramdict, simulate_with_hidden_outputs
from .protocols import protocol_condition
from .step05_mechanistic_decomposition import reconstruct_flat_params
from .step06_predictive_validation import Step06Config, load_step06_inputs, run_step06_predictive_validation

OUTPUT_SUBDIR = "assumption_sensitivity"
GATING_FAMILIES: tuple[str, ...] = (
    "sigmoid",
    "tanh",
    "hill",
    "soft_threshold",
    "hard_threshold",
    "double_sigmoid",
)
IDENTITY_COLUMNS = ["file_id", "region", "condition", "candidate_id"]


@dataclass(slots=True)
class Step07Config:
    max_candidates: int | None = 2
    time_points: int = 50
    t_final_ms: float = 50_000.0
    gating_families: tuple[str, ...] = GATING_FAMILIES
    currents_na: tuple[int, ...] = (100,)
    proxy_corr_min: float = 0.50
    proxy_rmse_max: float = 1.25
    gating_divergence_rmse_max_mV: float = 8.0
    heldout_pass_min: float = 0.30
    write_outputs: bool = True


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _time_grid(config: Step07Config) -> np.ndarray:
    return np.linspace(0.0, float(config.t_final_ms), int(config.time_points), dtype=float)


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    if method == "spearman":
        corr, _ = spearmanr(x, y)
        return float(corr)
    return float(np.corrcoef(x, y)[0, 1])


def _scaled_rmse(proxy: np.ndarray, target: np.ndarray) -> float:
    proxy = np.asarray(proxy, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(proxy) & np.isfinite(target)
    if mask.sum() < 3:
        return float("nan")
    x = proxy[mask]
    y = target[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = (slope * x + intercept) - y
    scale = float(np.nanstd(y)) or 1.0
    return float(np.sqrt(np.mean(resid**2)) / scale)


def _lag_samples(proxy: np.ndarray, target: np.ndarray, max_lag: int = 8) -> int:
    proxy = np.asarray(proxy, dtype=float) - float(np.nanmean(proxy))
    target = np.asarray(target, dtype=float) - float(np.nanmean(target))
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = proxy[-lag:], target[:lag]
        elif lag > 0:
            x, y = proxy[:-lag], target[lag:]
        else:
            x, y = proxy, target
        score = abs(_safe_corr(x, y)) if len(x) >= 3 else np.nan
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def _simulate_candidate(cand: Mapping[str, Any], family: str, current_na: int, config: Step07Config) -> dict[str, Any]:
    sweep = list(VALID_CURRENTS).index(int(current_na)) + 1 if int(current_na) in VALID_CURRENTS else 1
    params = reconstruct_flat_params(dict(cand), current_na=int(current_na), sweep=sweep)
    params["switching_function"] = family
    if family == "hill":
        params.setdefault("hill_coefficient", 2.0)
        params.setdefault("K_d", max(abs(float(params.get("zth", 0.2))), 0.2))
    paramdict = build_paramdict(protocol_condition(str(cand["condition"])), int(current_na), params)
    return simulate_with_hidden_outputs(
        paramdict,
        {"experiment_type": protocol_condition(str(cand["condition"])), "current_na": int(current_na), "sim_time_ms": _time_grid(config)},
    )


def load_step07_inputs(project_root: Path | str, config: Step07Config) -> pd.DataFrame:
    candidates, _ = load_step06_inputs(project_root, Step06Config(max_candidates=config.max_candidates, time_points=config.time_points, write_outputs=False))
    required = set(IDENTITY_COLUMNS + ["region", "condition"])
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Step 07 inputs missing required columns: {missing}")
    return candidates


def build_gating_family_comparison(candidates: pd.DataFrame, config: Step07Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        for current_na in config.currents_na:
            try:
                ref = _simulate_candidate(cand, "sigmoid", int(current_na), config)
                ref_vm = np.asarray(ref["Vm"], dtype=float)
                ref_status = "ok"
                ref_error = ""
            except Exception as exc:  # noqa: BLE001
                ref_vm = np.asarray([], dtype=float)
                ref_status = "failed"
                ref_error = f"{type(exc).__name__}: {exc}"
            for family in config.gating_families:
                base = {c: cand.get(c) for c in IDENTITY_COLUMNS}
                base.update(
                    {
                        "region": cand.get("region"),
                        "condition": cand.get("condition"),
                        "current_na": int(current_na),
                        "sweep": list(VALID_CURRENTS).index(int(current_na)) + 1 if int(current_na) in VALID_CURRENTS else np.nan,
                        "gating_family": family,
                        "identical_contract_id": "step07_same_candidates_currents_timegrid_loss_v1",
                        "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                        "dominant_mechanism": cand.get("dominant_mechanism", "unknown"),
                        "holdout_pass_fraction": float(cand.get("holdout_mean_pass_fraction", cand.get("mean_weighted_pass_fraction", np.nan))),
                        "mean_trace_rmse_mV_step04": float(cand.get("holdout_mean_rmse_mV", cand.get("mean_trace_rmse_mV", np.nan))),
                    }
                )
                if ref_status != "ok":
                    rows.append({**base, "simulation_status": "failed", "failure_reason": ref_error, "trace_rmse_vs_sigmoid_mV": np.nan, "family_supported_under_same_contract": False, "mechanism_claim_stable": False})
                    continue
                try:
                    sim = ref if family == "sigmoid" else _simulate_candidate(cand, family, int(current_na), config)
                    vm = np.asarray(sim["Vm"], dtype=float)
                    rmse = float(np.sqrt(np.mean((vm - ref_vm) ** 2)))
                    supported = bool(rmse <= config.gating_divergence_rmse_max_mV and base["holdout_pass_fraction"] >= config.heldout_pass_min)
                    rows.append({**base, "simulation_status": "ok", "failure_reason": "", "trace_rmse_vs_sigmoid_mV": rmse, "family_supported_under_same_contract": supported, "mechanism_claim_stable": supported and str(base["mechanism_cluster"]) != "unlabeled"})
                except Exception as exc:  # noqa: BLE001
                    rows.append({**base, "simulation_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}", "trace_rmse_vs_sigmoid_mV": np.nan, "family_supported_under_same_contract": False, "mechanism_claim_stable": False})
    return pd.DataFrame(rows)


def build_model_comparison(gating: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in gating.groupby("gating_family", dropna=False):
        ok = group[group["simulation_status"].eq("ok")]
        rows.append(
            {
                "model_family": str(family),
                "assumption_axis": "gating_form",
                "n_candidate_current_rows": int(len(group)),
                "n_successful_simulations": int(len(ok)),
                "mean_trace_rmse_vs_sigmoid_mV": float(ok["trace_rmse_vs_sigmoid_mV"].mean()) if not ok.empty else np.nan,
                "median_step04_trace_rmse_mV": float(pd.to_numeric(group["mean_trace_rmse_mV_step04"], errors="coerce").median()),
                "mean_heldout_pass_fraction": float(pd.to_numeric(group["holdout_pass_fraction"], errors="coerce").mean()),
                "same_split_same_loss_contract": "step07_same_candidates_currents_timegrid_loss_v1",
                "mechanism_stability_fraction": float(group["mechanism_claim_stable"].mean()) if len(group) else np.nan,
                "claim_scope": "robust_across_configured_gating_families" if bool(group["mechanism_claim_stable"].all()) and len(group) else "model_dependent_or_insufficient_evidence",
            }
        )
    return pd.DataFrame(rows)


def build_proxy_validity(candidates: pd.DataFrame, config: Step07Config) -> pd.DataFrame:
    rows = []
    for _, cand in candidates.iterrows():
        for current_na in config.currents_na:
            try:
                sim = _simulate_candidate(cand, "sigmoid", int(current_na), config)
                proxy = np.asarray(sim["states"][:, 1], dtype=float)
                ko = np.asarray(sim["derived"]["K_o"], dtype=float)
                rows.append(
                    {
                        **{c: cand.get(c) for c in IDENTITY_COLUMNS},
                        "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                        "region": cand.get("region"),
                        "condition": cand.get("condition"),
                        "current_na": int(current_na),
                        "sweep": list(VALID_CURRENTS).index(int(current_na)) + 1 if int(current_na) in VALID_CURRENTS else np.nan,
                        "proxy_signal": "delta_K_a_t",
                        "target_signal": "K_o",
                        "pearson_r": _safe_corr(proxy, ko),
                        "spearman_r": _safe_corr(proxy, ko, method="spearman"),
                        "scaled_rmse": _scaled_rmse(proxy, ko),
                        "best_lag_samples": _lag_samples(proxy, ko),
                        "proxy_validity_status": "proxy_supported" if np.isfinite(_safe_corr(proxy, ko)) and abs(_safe_corr(proxy, ko)) >= config.proxy_corr_min and np.isfinite(_scaled_rmse(proxy, ko)) and _scaled_rmse(proxy, ko) <= config.proxy_rmse_max else "proxy_limited",
                        "explicit_ecs_variant_required": not (np.isfinite(_safe_corr(proxy, ko)) and abs(_safe_corr(proxy, ko)) >= config.proxy_corr_min and np.isfinite(_scaled_rmse(proxy, ko)) and _scaled_rmse(proxy, ko) <= config.proxy_rmse_max),
                        "simulation_status": "ok",
                        "failure_reason": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({**{c: cand.get(c) for c in IDENTITY_COLUMNS}, "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"), "region": cand.get("region"), "condition": cand.get("condition"), "current_na": int(current_na), "sweep": np.nan, "proxy_signal": "delta_K_a_t", "target_signal": "K_o", "pearson_r": np.nan, "spearman_r": np.nan, "scaled_rmse": np.nan, "best_lag_samples": np.nan, "proxy_validity_status": "simulation_failed", "explicit_ecs_variant_required": True, "simulation_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def build_compartment_split_sensitivity(candidates: pd.DataFrame, config: Step07Config) -> pd.DataFrame:
    rows = []
    for _, cand in candidates.iterrows():
        for current_na in config.currents_na:
            try:
                sim = _simulate_candidate(cand, "sigmoid", int(current_na), config)
                two_state = np.asarray(sim["states"][:, 1], dtype=float)
                one_state = np.asarray(sim["states"][:, 1] + sim["states"][:, 2], dtype=float)
                ko = np.asarray(sim["derived"]["K_o"], dtype=float)
                two_corr = abs(_safe_corr(two_state, ko))
                one_corr = abs(_safe_corr(one_state, ko))
                rows.append(
                    {
                        **{c: cand.get(c) for c in IDENTITY_COLUMNS},
                        "region": cand.get("region"),
                        "condition": cand.get("condition"),
                        "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                        "current_na": int(current_na),
                        "sweep": list(VALID_CURRENTS).index(int(current_na)) + 1 if int(current_na) in VALID_CURRENTS else np.nan,
                        "two_state_proxy_abs_corr": two_corr,
                        "one_state_proxy_abs_corr": one_corr,
                        "corr_delta_one_minus_two": float(one_corr - two_corr),
                        "split_sensitivity_status": "split_robust" if abs(one_corr - two_corr) <= 0.25 else "split_sensitive",
                        "mechanism_structure_persists": bool(abs(one_corr - two_corr) <= 0.25 and str(cand.get("mechanism_cluster", "unlabeled")) != "unlabeled"),
                        "simulation_status": "ok",
                        "failure_reason": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({**{c: cand.get(c) for c in IDENTITY_COLUMNS}, "region": cand.get("region"), "condition": cand.get("condition"), "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"), "current_na": int(current_na), "sweep": np.nan, "two_state_proxy_abs_corr": np.nan, "one_state_proxy_abs_corr": np.nan, "corr_delta_one_minus_two": np.nan, "split_sensitivity_status": "simulation_failed", "mechanism_structure_persists": False, "simulation_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def build_assumption_claim_scope(model_comparison: pd.DataFrame, proxy: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    all_gating_robust = bool(model_comparison["claim_scope"].eq("robust_across_configured_gating_families").all()) if not model_comparison.empty else False
    proxy_supported = bool(proxy["proxy_validity_status"].eq("proxy_supported").all()) if not proxy.empty else False
    split_robust = bool(split["split_sensitivity_status"].eq("split_robust").all()) if not split.empty else False
    return pd.DataFrame(
        [
            {"assumption_axis": "gating_form", "status": "robust" if all_gating_robust else "model_dependent_or_insufficient_evidence", "final_degeneracy_claim_allowed_after_step07": False},
            {"assumption_axis": "intracellular_K_as_ECS_proxy", "status": "proxy_supported" if proxy_supported else "explicit_ecs_variant_or_extra_data_needed", "final_degeneracy_claim_allowed_after_step07": False},
            {"assumption_axis": "local_syncytial_compartment_split", "status": "split_robust" if split_robust else "split_sensitive_or_insufficient_evidence", "final_degeneracy_claim_allowed_after_step07": False},
        ]
    )


def run_step07_assumption_sensitivity(project_root: Path | str, config: Step07Config | None = None, output_dir: Path | str | None = None) -> dict[str, Any]:
    config = config or Step07Config()
    root = Path(project_root).resolve()
    out_dir = _ensure_dir(Path(output_dir).resolve() if output_dir is not None else root / "outputs" / OUTPUT_SUBDIR)
    t0 = time.perf_counter()
    candidates = load_step07_inputs(root, config)
    gating = build_gating_family_comparison(candidates, config)
    model = build_model_comparison(gating)
    proxy = build_proxy_validity(candidates, config)
    split = build_compartment_split_sensitivity(candidates, config)
    claims = build_assumption_claim_scope(model, proxy, split)
    summary = {
        "step_name": "Step 07 — Assumption sensitivity: gating, proxy, and compartment split",
        "config": asdict(config),
        "n_candidates": int(len(candidates)),
        "n_gating_rows": int(len(gating)),
        "n_proxy_rows": int(len(proxy)),
        "n_compartment_rows": int(len(split)),
        "gating_families": list(config.gating_families),
        "claim_scope": "Step 07 can mark assumptions robust or model-dependent, but final biological degeneracy claims remain disallowed until later plausibility/statistical checks pass.",
        "elapsed_seconds": time.perf_counter() - t0,
    }
    if config.write_outputs:
        model.to_csv(out_dir / "model_comparison.csv", index=False)
        gating.to_csv(out_dir / "gating_family_comparison.csv", index=False)
        proxy.to_csv(out_dir / "proxy_validity_by_ensemble.csv", index=False)
        split.to_csv(out_dir / "compartment_split_sensitivity.csv", index=False)
        claims.to_csv(out_dir / "claim_scope_table.csv", index=False)
        (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"model_comparison": model, "gating_family_comparison": gating, "proxy_validity_by_ensemble": proxy, "compartment_split_sensitivity": split, "claim_scope_table": claims, "analysis_summary": summary}


def compare_step07_runtime_presets(project_root: Path | str, max_candidates: int = 1) -> pd.DataFrame:
    rows = []
    for preset, time_points, families in [
        ("coarse", 30, ("sigmoid", "tanh", "hard_threshold")),
        ("default", 50, GATING_FAMILIES),
    ]:
        cfg = Step07Config(max_candidates=max_candidates, time_points=time_points, gating_families=families, write_outputs=False)
        t0 = time.perf_counter()
        try:
            run_step07_assumption_sensitivity(project_root, cfg)
            status, error = "ok", ""
        except Exception as exc:  # noqa: BLE001
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        rows.append({"preset": preset, "time_points": time_points, "n_gating_families": len(families), "elapsed_seconds": time.perf_counter() - t0, "status": status, "error": error})
    df = pd.DataFrame(rows)
    fastest = df.loc[df["elapsed_seconds"].idxmin(), "preset"] if not df.empty else "coarse"
    df["recommendation"] = np.where(df["preset"].eq(fastest), f"{fastest}_recommended_for_tests", "default_or_coarse_available_for_manuscript_rerun")
    return df


def compare_step07_performance(project_root: Path | str, max_candidates: int = 1) -> pd.DataFrame:
    return compare_step07_runtime_presets(project_root, max_candidates=max_candidates)
