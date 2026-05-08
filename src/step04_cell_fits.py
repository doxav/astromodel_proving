from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .astro_model import DEFAULT_Z0, simulate_odeint
from .atf_io import load_all_cells, load_cell_protocol, canonical_file_id
from .atf_features import FEATURE_COLUMNS, build_feature_table, compute_feature_reliability, build_threshold_table, extract_features_from_trace

TRACE_SCALE_MV_DEFAULT = 10.0
TRACE_RMSE_ACCEPT_DEFAULT = 18.0
FEATURE_PASS_ACCEPT_DEFAULT = 0.30
HELDOUT_TRACE_RMSE_ACCEPT_DEFAULT = 20.0
HELDOUT_PASS_ACCEPT_DEFAULT = 0.30
HELDOUT_MIN_PASS_COUNT_DEFAULT = 3
ACCEPTED_TOP_K_PER_CELL_DEFAULT = 3
N_FIT_POINTS_DEFAULT = 40
N_STARTS_DEFAULT = 3
MAX_NFEV_ALL6_DEFAULT = 60
MAX_NFEV_HOLDOUT_DEFAULT = 40

BASE_CONDITION_DEFAULTS: dict[str, dict[str, Any]] = {
    "CONTROL": dict(ca=400.0, gl_a=0.01, Va_l=-70.0, Va_s=-90.0, switching_function="sigmoid", w_a=2000.0, eps_middle=1.0, wo_middle=1.0, g_k_a=0.0),
    "MFA": dict(ca=400.0, gl_a=0.01, Va_l=-70.0, Va_s=-90.0, switching_function="tanh", w_a=2000.0, eps_middle=1.0, wo_middle=1.0, g_k_a=0.0),
    "MFA_BA": dict(ca=400.0, gl_a=0.01, Va_l=-70.0, Va_s=-90.0, switching_function="tanh", w_a=2000.0, eps_middle=1.0, wo_middle=1.0, g_k_a=0.0),
}

EFFECTIVE_KEYS = ("P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo")
OPTIMIZED_KEYS = ("P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo", "gki", "eps", "gl_a", "zth", "zs")


@dataclass(frozen=True)
class SweepTrace:
    file_id: str
    region: str
    condition: str
    sweep: int
    current_na: int
    time_ms_fit: np.ndarray
    vm_fit: np.ndarray
    time_s_full: np.ndarray
    vm_full: np.ndarray


@dataclass
class Step04Config:
    project_root: Path
    output_dir: Optional[Path] = None
    max_cells: Optional[int] = None
    selected_file_ids: Optional[List[str]] = None
    n_fit_points: int = N_FIT_POINTS_DEFAULT
    n_starts: int = N_STARTS_DEFAULT
    max_nfev_all6: int = MAX_NFEV_ALL6_DEFAULT
    max_nfev_holdout: int = MAX_NFEV_HOLDOUT_DEFAULT
    trace_scale_mV: float = TRACE_SCALE_MV_DEFAULT
    trace_rmse_accept: float = TRACE_RMSE_ACCEPT_DEFAULT
    feature_pass_accept: float = FEATURE_PASS_ACCEPT_DEFAULT
    heldout_trace_rmse_accept: float = HELDOUT_TRACE_RMSE_ACCEPT_DEFAULT
    heldout_pass_accept: float = HELDOUT_PASS_ACCEPT_DEFAULT
    heldout_min_pass_count: int = HELDOUT_MIN_PASS_COUNT_DEFAULT
    accepted_top_k_per_cell: int = ACCEPTED_TOP_K_PER_CELL_DEFAULT

    def resolve(self) -> "Step04Config":
        if self.output_dir is None:
            self.output_dir = self.project_root / "outputs" / "cell_fits"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


def _project_paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "project_root": root,
        "atf_dir": root / "data" / "2_K+ Pumps Data",
        "features_dir": root / "outputs" / "features",
        "provenance_dir": root / "outputs" / "provenance",
        "postfit_dir": root / "outputs" / "postfit_sqlite",
        "cell_fit_dir": root / "outputs" / "cell_fits",
    }


def load_step02_outputs_or_run(project_root: Path, reuse_existing: bool = True) -> dict[str, pd.DataFrame]:
    paths = _project_paths(project_root)
    feature_csv = paths["features_dir"] / "feature_table_by_sweep.csv"
    threshold_csv = paths["features_dir"] / "condition_region_sweep_thresholds.csv"
    counts_csv = paths["features_dir"] / "region_condition_cell_counts.csv"
    reliability_csv = paths["features_dir"] / "feature_reliability_weights.csv"
    if reuse_existing and feature_csv.exists() and threshold_csv.exists() and counts_csv.exists():
        out = {
            "feature_table_by_sweep": pd.read_csv(feature_csv),
            "condition_region_sweep_thresholds": pd.read_csv(threshold_csv),
            "region_condition_cell_counts": pd.read_csv(counts_csv),
        }
        if reliability_csv.exists():
            out["feature_reliability_weights"] = pd.read_csv(reliability_csv)
        return out
    cells = load_all_cells(paths["atf_dir"])
    feature_df = build_feature_table(cells)
    # normalize expected columns from existing Step 02 outputs
    if "plateau_reached" not in feature_df.columns:
        feature_df["plateau_reached"] = False
    if "has_undershoot" not in feature_df.columns:
        feature_df["has_undershoot"] = feature_df.get("undershoot_magnitude_mV", 0).fillna(0) > 0
    counts_df = feature_df[["file_id", "region", "condition"]].drop_duplicates().groupby(["region", "condition"]).size().rename("n_cells").reset_index()
    expected = {("DH","CONTROL"):7,("VH","CONTROL"):4,("DH","MFA"):6,("VH","MFA"):7,("DH","MFA_BA"):6,("VH","MFA_BA"):7}
    rows=[]
    for (region,condition), expected_n in expected.items():
        n = int(counts_df[(counts_df.region==region)&(counts_df.condition==condition)]["n_cells"].iloc[0]) if not counts_df[(counts_df.region==region)&(counts_df.condition==condition)].empty else 0
        rows.append({"region":region,"condition":condition,"n_cells":n,"expected_n_cells":expected_n,"matches_expected":n==expected_n,"small_stratum":n<5})
    counts_df = pd.DataFrame(rows).sort_values(["region","condition"]).reset_index(drop=True)
    reliability_df = compute_feature_reliability(feature_df)
    thresholds_df = build_threshold_table(feature_df, reliability_df)
    paths["features_dir"].mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(feature_csv, index=False)
    thresholds_df.to_csv(threshold_csv, index=False)
    counts_df.to_csv(counts_csv, index=False)
    reliability_df.to_csv(reliability_csv, index=False)
    return {"feature_table_by_sweep": feature_df, "condition_region_sweep_thresholds": thresholds_df, "region_condition_cell_counts": counts_df, "feature_reliability_weights": reliability_df}


def _safe_downsample_trace(time_s: np.ndarray, vm: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(time_s) <= n_points:
        return time_s * 1000.0, vm
    target_time_s = np.linspace(float(time_s[0]), float(time_s[-1]), int(n_points), dtype=float)
    target_vm = np.interp(target_time_s, time_s, vm)
    return target_time_s * 1000.0, target_vm


def build_cell_trace_inventory(atf_dir: str | Path, n_fit_points: int = N_FIT_POINTS_DEFAULT, file_ids: Optional[Sequence[str]] = None) -> dict[str, dict[int, SweepTrace]]:
    inventory: dict[str, dict[int, SweepTrace]] = {}
    wanted = set(file_ids) if file_ids is not None else None
    paths = sorted(Path(atf_dir).glob("*.atf"))
    for path in paths:
        fid = canonical_file_id(Path(path))
        if wanted is not None and fid not in wanted:
            continue
        cell = load_cell_protocol(Path(path))
        out: dict[int, SweepTrace] = {}
        for sweep in cell.sweeps:
            time_ms_fit, vm_fit = _safe_downsample_trace(np.asarray(sweep.time_s, dtype=float), np.asarray(sweep.vm_mV, dtype=float), n_fit_points)
            out[int(sweep.sweep)] = SweepTrace(
                file_id=cell.file_id,
                region=cell.region,
                condition=cell.condition,
                sweep=int(sweep.sweep),
                current_na=int(sweep.current_na),
                time_ms_fit=time_ms_fit,
                vm_fit=vm_fit,
                time_s_full=np.asarray(sweep.time_s, dtype=float),
                vm_full=np.asarray(sweep.vm_mV, dtype=float),
            )
        inventory[cell.file_id] = out
    return inventory


def heldout_splits(n_sweeps: int = 6) -> list[tuple[list[int], int]]:
    sweeps = list(range(1, n_sweeps + 1))
    return [([s for s in sweeps if s != heldout], heldout) for heldout in sweeps]


def _baseline_subtract(trace: np.ndarray, time_ms: np.ndarray, onset_s: float) -> np.ndarray:
    t_s = np.asarray(time_ms, dtype=float) / 1000.0
    mask = (t_s >= max(0.0, onset_s - 5.0)) & (t_s <= max(0.5, onset_s - 1.0))
    if mask.any():
        baseline = float(np.nanmedian(trace[mask]))
    else:
        baseline = float(np.nanmedian(trace[: max(5, min(50, len(trace)))]))
    return np.asarray(trace, dtype=float) - baseline


def _threshold_row(thresholds_df: pd.DataFrame, region: str, condition: str, sweep: int, feature: str) -> pd.Series:
    row = thresholds_df[(thresholds_df["region"] == region) & (thresholds_df["condition"] == condition) & (thresholds_df["sweep"] == sweep) & (thresholds_df["feature"] == feature)]
    if row.empty:
        raise KeyError((region, condition, sweep, feature))
    return row.iloc[0]


def _feature_contract_score(sim_features: Mapping[str, Any], empirical_row: Mapping[str, Any], thresholds_df: pd.DataFrame, region: str, condition: str, sweep: int) -> dict[str, Any]:
    total_weight = 0.0
    soft_score_sum = 0.0
    weighted_distance = 0.0
    feature_passes: dict[str, bool] = {}
    for feature in FEATURE_COLUMNS:
        th = _threshold_row(thresholds_df, region, condition, sweep, feature)
        weight = float(th.get("reliability_weight", 1.0))
        if weight <= 0:
            continue
        total_weight += weight
        val = sim_features.get(feature, np.nan)
        lower = float(th["acceptable_lower"])
        upper = float(th["acceptable_upper"])
        iqr = float(th["iqr"]) if np.isfinite(th["iqr"]) and abs(float(th["iqr"])) > 1e-12 else max(abs(float(th["median"])) * 0.25, 0.5)
        passed = bool(np.isfinite(val) and val >= lower and val <= upper)
        feature_passes[f"pass_{feature}"] = passed
        if np.isfinite(val):
            distance = 0.0
            if val < lower:
                distance = lower - val
            elif val > upper:
                distance = val - upper
            soft_score = max(0.0, 1.0 - distance / max(2.0 * abs(iqr), 1e-9))
            soft_score_sum += weight * soft_score
            weighted_distance += weight * (1.0 - soft_score)
        else:
            weighted_distance += weight
    weighted_pass_fraction = soft_score_sum / max(total_weight, 1e-12)
    feature_loss = weighted_distance / max(total_weight, 1e-12)
    plateau_match = float(bool(sim_features.get("plateau_reached", False)) == bool(empirical_row.get("plateau_reached", False)))
    undershoot_match = float(bool(sim_features.get("has_undershoot", False)) == bool(empirical_row.get("has_undershoot", False)))
    binary_penalty = 0.5 * ((1.0 - plateau_match) + (1.0 - undershoot_match))
    return {
        "weighted_pass_fraction": float(weighted_pass_fraction),
        "feature_loss": float(feature_loss),
        "binary_penalty": float(binary_penalty),
        **feature_passes,
    }


def _feature_residuals(sim_features: Mapping[str, Any], empirical_row: Mapping[str, Any], thresholds_df: pd.DataFrame, region: str, condition: str, sweep: int) -> np.ndarray:
    residuals: list[float] = []
    for feature in FEATURE_COLUMNS:
        th = _threshold_row(thresholds_df, region, condition, sweep, feature)
        weight = float(th.get("reliability_weight", 1.0))
        if weight <= 0:
            continue
        emp = empirical_row.get(feature, np.nan)
        sim = sim_features.get(feature, np.nan)
        width = float(th.get("iqr", np.nan))
        if not np.isfinite(width) or abs(width) < 1e-12:
            width = abs(float(th.get("acceptable_upper", 1.0)) - float(th.get("acceptable_lower", 0.0))) / 2.0
        if not np.isfinite(width) or abs(width) < 1e-12:
            width = max(abs(float(th.get("median", 1.0))) * 0.25, 0.5)
        if np.isfinite(emp) and np.isfinite(sim):
            residuals.append(math.sqrt(weight) * (float(sim) - float(emp)) / max(abs(width), 1e-9))
        else:
            residuals.append(math.sqrt(weight) * 3.0)
    plateau_emp = bool(empirical_row.get("plateau_reached", False))
    plateau_sim = bool(sim_features.get("plateau_reached", False))
    residuals.append(0.5 if plateau_emp != plateau_sim else 0.0)
    undershoot_emp = bool(empirical_row.get("has_undershoot", False))
    undershoot_sim = bool(sim_features.get("has_undershoot", False))
    residuals.append(0.5 if undershoot_emp != undershoot_sim else 0.0)
    return np.asarray(residuals, dtype=float)


def _default_onset_s(condition: str) -> float:
    return 11.173 if str(condition).upper() == "CONTROL" else 21.140


def _effective_from_flat(params: Mapping[str, Any]) -> dict[str, float]:
    w_a = float(params.get("w_a", 2000.0))
    sig_a = 1600.0
    F = 96485.0
    d = float(params.get("d", 1.0))
    pk = float(params.get("pk", 0.0))
    return {
        "P_gap_eff": d * pk,
        "gamma_t_eff": float(params.get("gt", 0.0)) * sig_a / (w_a * F),
        "gamma_s_eff": float(params.get("gs", 0.0)) * sig_a / (w_a * F),
        "volume_ratio_wa_wo": w_a / max(float(params.get("wo", 1500.0)), 1e-12),
    }


def _flat_from_effective(condition: str, eff: Mapping[str, float], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = dict(BASE_CONDITION_DEFAULTS[condition])
    w_a = float(base["w_a"])
    sig_a = 1600.0
    F = 96485.0
    out = dict(base)
    out["d"] = 1.0
    out["pk"] = float(eff["P_gap_eff"])
    out["gt"] = float(eff["gamma_t_eff"]) * w_a * F / sig_a
    out["gs"] = float(eff["gamma_s_eff"]) * w_a * F / sig_a
    out["wo"] = w_a / max(float(eff["volume_ratio_wa_wo"]), 1e-12)
    if extra:
        out.update(extra)
    return out


def _default_effective_by_condition(project_root: Path, condition: str) -> tuple[dict[str, float], str]:
    paths = _project_paths(project_root)
    eff_csv = paths["postfit_dir"] / "effective_parameter_summary.csv"
    prov_csv = paths["provenance_dir"] / "control_trace_verification.csv"
    if eff_csv.exists() and prov_csv.exists():
        eff = pd.read_csv(eff_csv)
        prov = pd.read_csv(prov_csv)
        verified = prov[prov.get("chosen_status", prov.get("status", "")) == "verified"]["db_name"].tolist()
        eff = eff[eff["db_name"].isin(verified)].copy()
        seed_condition = "BARIUM" if condition == "MFA_BA" else condition
        subset = eff[eff["condition"] == seed_condition]
        if not subset.empty:
            med = subset[list(EFFECTIVE_KEYS)].median(numeric_only=True).to_dict()
            return {k: float(med[k]) for k in EFFECTIVE_KEYS}, "legacy_db_verified_median"
    flat = dict(BASE_CONDITION_DEFAULTS[condition])
    flat.update({"d": 1.0, "pk": 1e-4, "gt": 4.0, "gs": 8.0, "wo": 1500.0, "gki": 45.0, "eps": 0.01, "zth": 0.2, "zs": 0.05})
    return _effective_from_flat(flat), "generic_default"


def _x_to_named(x: np.ndarray) -> dict[str, float]:
    return {
        "P_gap_eff": float(np.exp(x[0])),
        "gamma_t_eff": float(np.exp(x[1])),
        "gamma_s_eff": float(np.exp(x[2])),
        "volume_ratio_wa_wo": float(np.exp(x[3])),
        "gki": float(np.exp(x[4])),
        "eps": float(np.exp(x[5])),
        "gl_a": float(np.exp(x[6])),
        "zth": float(x[7]),
        "zs": float(np.exp(x[8])),
    }


def _named_to_x(named: Mapping[str, float]) -> np.ndarray:
    return np.asarray([
        np.log(max(float(named["P_gap_eff"]), 1e-12)),
        np.log(max(float(named["gamma_t_eff"]), 1e-12)),
        np.log(max(float(named["gamma_s_eff"]), 1e-12)),
        np.log(max(float(named["volume_ratio_wa_wo"]), 1e-12)),
        np.log(max(float(named["gki"]), 1e-12)),
        np.log(max(float(named["eps"]), 1e-12)),
        np.log(max(float(named["gl_a"]), 1e-12)),
        float(named["zth"]),
        np.log(max(float(named["zs"]), 1e-12)),
    ], dtype=float)


def _x_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([
        np.log(1e-8), np.log(1e-8), np.log(1e-8), np.log(0.05), np.log(0.5), np.log(1e-5), np.log(1e-5), 0.0, np.log(1e-3)
    ], dtype=float)
    upper = np.asarray([
        np.log(1e-2), np.log(1e-2), np.log(1e-1), np.log(10.0), np.log(500.0), np.log(0.1), np.log(20.0), 150.0, np.log(50.0)
    ], dtype=float)
    return lower, upper


def _start_vectors(project_root: Path, condition: str, n_starts: int) -> list[tuple[np.ndarray, str, str]]:
    eff0, seed_source = _default_effective_by_condition(project_root, condition)
    base_named = {
        **eff0,
        "gki": 45.0 if condition == "CONTROL" else 30.0,
        "eps": 0.01,
        "gl_a": 0.01,
        "zth": 0.2,
        "zs": 0.05,
    }
    starts = []
    scales = [
        {"P_gap_eff": 1.0, "gamma_t_eff": 1.0, "gamma_s_eff": 1.0, "volume_ratio_wa_wo": 1.0, "gki": 1.0, "eps": 1.0, "gl_a": 1.0, "zth": 1.0, "zs": 1.0},
        {"P_gap_eff": 0.6, "gamma_t_eff": 1.2, "gamma_s_eff": 0.8, "volume_ratio_wa_wo": 1.1, "gki": 0.8, "eps": 1.2, "gl_a": 1.0, "zth": 1.0, "zs": 1.5},
        {"P_gap_eff": 1.4, "gamma_t_eff": 0.8, "gamma_s_eff": 1.4, "volume_ratio_wa_wo": 0.9, "gki": 1.2, "eps": 0.8, "gl_a": 1.0, "zth": 1.0, "zs": 0.7},
        {"P_gap_eff": 0.8, "gamma_t_eff": 0.8, "gamma_s_eff": 1.2, "volume_ratio_wa_wo": 1.3, "gki": 1.1, "eps": 1.5, "gl_a": 0.8, "zth": 1.2, "zs": 1.0},
    ]
    for idx, scale in enumerate(scales[: max(1, n_starts)], start=1):
        named = dict(base_named)
        for k, f in scale.items():
            if k == "zth":
                named[k] = float(named[k]) * float(f)
            else:
                named[k] = max(float(named[k]) * float(f), 1e-10)
        starts.append((_named_to_x(named), seed_source, f"start_{idx:02d}"))
    return starts


def _params_from_x(condition: str, x: np.ndarray, seed_source: str, start_label: str) -> dict[str, Any]:
    named = _x_to_named(x)
    eff = {k: named[k] for k in EFFECTIVE_KEYS}
    flat = _flat_from_effective(condition, eff, extra={
        "gki": named["gki"],
        "eps": named["eps"],
        "gl_a": named["gl_a"],
        "zth": named["zth"],
        "zs": named["zs"],
        "seed_source": seed_source,
        "start_label": start_label,
    })
    return flat


def _simulate_sweep(params: Mapping[str, Any], sweep_trace: SweepTrace) -> tuple[np.ndarray, dict[str, Any], float]:
    protocol = {"experiment_type": sweep_trace.condition, "current_na": sweep_trace.current_na, "t_eval_ms": sweep_trace.time_ms_fit}
    sim = simulate_odeint(params, protocol, z0=DEFAULT_Z0, t_eval_ms=sweep_trace.time_ms_fit, return_hidden=False)
    sim_vm = np.asarray(sim["Vm"], dtype=float)
    sim_features = extract_features_from_trace(sweep_trace.time_ms_fit / 1000.0, sim_vm)
    onset_s = float(sim_features.get("stim_onset_s", _default_onset_s(sweep_trace.condition)))
    return sim_vm, sim_features, onset_s


def _residual_vector(x: np.ndarray, condition: str, sweeps_to_fit: Sequence[int], trace_inventory: Mapping[str, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, trace_scale_mV: float) -> np.ndarray:
    params = _params_from_x(condition, x, seed_source="runtime", start_label="runtime")
    residuals: list[np.ndarray] = []
    penalty = 0.0
    for sweep_idx in sweeps_to_fit:
        sweep_trace = trace_inventory[sweep_idx]
        try:
            sim_vm, sim_features, onset_s = _simulate_sweep(params, sweep_trace)
            exp_bs = _baseline_subtract(sweep_trace.vm_fit, sweep_trace.time_ms_fit, onset_s)
            sim_bs = _baseline_subtract(sim_vm, sweep_trace.time_ms_fit, onset_s)
            trace_resid = (sim_bs - exp_bs) / max(trace_scale_mV, 1e-9)
            feature_resid = _feature_residuals(sim_features, empirical_rows[sweep_idx], thresholds_df, sweep_trace.region, sweep_trace.condition, sweep_idx)
        except Exception:
            trace_resid = np.full_like(sweep_trace.vm_fit, 10.0, dtype=float)
            feature_resid = np.full(len(FEATURE_COLUMNS) + 2, 5.0, dtype=float)
            penalty += 20.0
        residuals.append(np.asarray(trace_resid, dtype=float))
        residuals.append(np.asarray(feature_resid, dtype=float))
    if penalty > 0:
        residuals.append(np.asarray([penalty], dtype=float))
    return np.concatenate(residuals).astype(float)


def _score_candidate_metrics(params: Mapping[str, Any], trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sweep_idx, sweep_trace in sorted(trace_inventory.items()):
        empirical = empirical_rows[sweep_idx]
        try:
            sim_vm, sim_features, onset_s = _simulate_sweep(params, sweep_trace)
            exp_bs = _baseline_subtract(sweep_trace.vm_fit, sweep_trace.time_ms_fit, onset_s)
            sim_bs = _baseline_subtract(sim_vm, sweep_trace.time_ms_fit, onset_s)
            trace_rmse = float(np.sqrt(np.mean((exp_bs - sim_bs) ** 2)))
            contract = _feature_contract_score(sim_features, empirical, thresholds_df, sweep_trace.region, sweep_trace.condition, sweep_idx)
            row = {
                "simulation_health": "ok",
                "trace_rmse_mV": trace_rmse,
                **contract,
                **{k: sim_features.get(k, np.nan) for k in FEATURE_COLUMNS},
                "plateau_reached_sim": bool(sim_features.get("plateau_reached", False)),
                "has_undershoot_sim": bool(sim_features.get("has_undershoot", False)),
            }
        except Exception as exc:
            row = {
                "simulation_health": f"failed:{type(exc).__name__}",
                "trace_rmse_mV": np.nan,
                "weighted_pass_fraction": 0.0,
                "feature_loss": 1.0,
                "binary_penalty": 1.0,
                **{f"pass_{f}": False for f in FEATURE_COLUMNS},
                **{k: np.nan for k in FEATURE_COLUMNS},
                "plateau_reached_sim": False,
                "has_undershoot_sim": False,
            }
        row.update({"sweep": int(sweep_idx), "current_na": int(sweep_trace.current_na), "region": sweep_trace.region, "condition": sweep_trace.condition})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("sweep").reset_index(drop=True)
    n_failures = int((df["simulation_health"] != "ok").sum())
    agg = {
        "n_sweeps_scored": int(len(df)),
        "n_failures": n_failures,
        "mean_trace_rmse_mV": float(df["trace_rmse_mV"].mean(skipna=True)) if df["trace_rmse_mV"].notna().any() else np.nan,
        "mean_weighted_pass_fraction": float(df["weighted_pass_fraction"].mean()) if not df.empty else 0.0,
        "mean_feature_loss": float(df["feature_loss"].mean()) if not df.empty else 1.0,
        "mean_binary_penalty": float(df["binary_penalty"].mean()) if not df.empty else 1.0,
    }
    return df, agg


def acceptance_contract_table(cfg: Step04Config | None = None) -> pd.DataFrame:
    cfg = cfg or Step04Config(project_root=Path("."))
    rows = [
        {"criterion": "trace_rmse_mean_mV", "scope": "all6", "operator": "<=", "value": cfg.trace_rmse_accept, "role": "accepted_by_trace"},
        {"criterion": "weighted_pass_fraction_mean", "scope": "all6", "operator": ">=", "value": cfg.feature_pass_accept, "role": "accepted_by_feature_contract"},
        {"criterion": "heldout_trace_rmse_mV", "scope": "leave_one_out", "operator": "<=", "value": cfg.heldout_trace_rmse_accept, "role": "heldout_screen"},
        {"criterion": "heldout_weighted_pass_fraction", "scope": "leave_one_out", "operator": ">=", "value": cfg.heldout_pass_accept, "role": "heldout_screen"},
        {"criterion": "ensemble_rank", "scope": "all6", "operator": "<=", "value": cfg.accepted_top_k_per_cell, "role": "accepted_all6_topk"},
        {"criterion": "holdout_pass_count", "scope": "cell", "operator": ">=", "value": cfg.heldout_min_pass_count, "role": "reviewer_facing_cell"},
    ]
    return pd.DataFrame(rows)


def _fit_cell_all6(project_root: Path, file_id: str, trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, cfg: Step04Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition = next(iter(trace_inventory.values())).condition
    lower, upper = _x_bounds()
    candidate_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    for start_idx, (x0, seed_source, start_label) in enumerate(_start_vectors(project_root, condition, cfg.n_starts), start=1):
        result = least_squares(
            _residual_vector,
            x0=x0,
            bounds=(lower, upper),
            args=(condition, list(sorted(trace_inventory)), trace_inventory, empirical_rows, thresholds_df, cfg.trace_scale_mV),
            max_nfev=cfg.max_nfev_all6,
            method="trf",
        )
        params = _params_from_x(condition, result.x, seed_source=seed_source, start_label=start_label)
        sweep_df, agg = _score_candidate_metrics(params, trace_inventory, empirical_rows, thresholds_df)
        candidate_id = f"{file_id}__cand_{start_idx:02d}"
        sweep_df.insert(0, "candidate_id", candidate_id)
        sweep_df.insert(0, "file_id", file_id)
        sweep_df.insert(0, "fit_scope", "all6")
        sweep_rows.append(sweep_df)
        eff = _effective_from_flat(params)
        row = {
            "file_id": file_id,
            "region": next(iter(trace_inventory.values())).region,
            "condition": condition,
            "candidate_id": candidate_id,
            "fit_scope": "all6",
            "seed_source": seed_source,
            "provenance_status": "verified_seed" if seed_source == "legacy_db_verified_median" else "generic_seed",
            "start_label": start_label,
            "optimizer_status": int(result.status),
            "optimizer_success": bool(result.success),
            "optimizer_cost": float(result.cost),
            "optimizer_nfev": int(result.nfev),
            "gki": float(params["gki"]),
            "eps": float(params["eps"]),
            "gl_a": float(params["gl_a"]),
            "zth": float(params["zth"]),
            "zs": float(params["zs"]),
            **eff,
            **agg,
        }
        row["accepted_by_trace"] = bool(np.isfinite(row["mean_trace_rmse_mV"]) and row["mean_trace_rmse_mV"] <= cfg.trace_rmse_accept)
        row["accepted_by_feature_contract"] = bool(row["mean_weighted_pass_fraction"] >= cfg.feature_pass_accept)
        row["eligible_by_contract"] = bool(row["accepted_by_trace"] and row["accepted_by_feature_contract"] and row["n_failures"] == 0)
        candidate_rows.append(row)
    cand_df = pd.DataFrame(candidate_rows).sort_values(["eligible_by_contract", "mean_weighted_pass_fraction", "mean_trace_rmse_mV", "optimizer_cost"], ascending=[False, False, True, True]).reset_index(drop=True)
    if not cand_df.empty:
        cand_df["ensemble_rank"] = np.arange(1, len(cand_df) + 1)
        cand_df["accepted_all6"] = cand_df["eligible_by_contract"] & (cand_df["ensemble_rank"] <= cfg.accepted_top_k_per_cell)
    sweep_df = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    return cand_df, sweep_df


def _fit_holdout(file_id: str, trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, best_candidate: Mapping[str, Any], cfg: Step04Config) -> pd.DataFrame:
    condition = next(iter(trace_inventory.values())).condition
    named = {
        "P_gap_eff": float(best_candidate["P_gap_eff"]),
        "gamma_t_eff": float(best_candidate["gamma_t_eff"]),
        "gamma_s_eff": float(best_candidate["gamma_s_eff"]),
        "volume_ratio_wa_wo": float(best_candidate["volume_ratio_wa_wo"]),
        "gki": float(best_candidate["gki"]),
        "eps": float(best_candidate["eps"]),
        "gl_a": float(best_candidate["gl_a"]),
        "zth": float(best_candidate["zth"]),
        "zs": float(best_candidate["zs"]),
    }
    x_start = _named_to_x(named)
    lower, upper = _x_bounds()
    rows: list[dict[str, Any]] = []
    for train_sweeps, heldout in heldout_splits(6):
        res = least_squares(
            _residual_vector,
            x0=x_start,
            bounds=(lower, upper),
            args=(condition, train_sweeps, trace_inventory, empirical_rows, thresholds_df, cfg.trace_scale_mV),
            max_nfev=cfg.max_nfev_holdout,
            method="trf",
        )
        params = _params_from_x(condition, res.x, seed_source=str(best_candidate.get("seed_source", "runtime")), start_label=f"heldout_{heldout}")
        full_sweep_df, _ = _score_candidate_metrics(params, trace_inventory, empirical_rows, thresholds_df)
        held = full_sweep_df[full_sweep_df["sweep"] == heldout].iloc[0].to_dict()
        held_pass = bool(np.isfinite(held["trace_rmse_mV"]) and held["trace_rmse_mV"] <= cfg.heldout_trace_rmse_accept and held["weighted_pass_fraction"] >= cfg.heldout_pass_accept and str(held["simulation_health"]) == "ok")
        rows.append({
            "file_id": file_id,
            "region": next(iter(trace_inventory.values())).region,
            "condition": condition,
            "fit_scope": "leave_one_out",
            "heldout_sweep": int(heldout),
            "train_sweeps": ",".join(map(str, train_sweeps)),
            "candidate_id": str(best_candidate["candidate_id"]),
            "optimizer_status": int(res.status),
            "optimizer_success": bool(res.success),
            "optimizer_cost": float(res.cost),
            "optimizer_nfev": int(res.nfev),
            "heldout_trace_rmse_mV": float(held["trace_rmse_mV"]),
            "heldout_weighted_pass_fraction": float(held["weighted_pass_fraction"]),
            "simulation_health": held["simulation_health"],
            "heldout_pass": held_pass,
        })
    return pd.DataFrame(rows)


def reconstruct_candidate_params(candidate_row: Mapping[str, Any]) -> dict[str, Any]:
    condition = str(candidate_row["condition"])
    eff = {k: float(candidate_row[k]) for k in EFFECTIVE_KEYS}
    extra = {k: float(candidate_row[k]) for k in ["gki", "eps", "gl_a", "zth", "zs"]}
    return _flat_from_effective(condition, eff, extra=extra)


def build_candidate_overlay_frame(candidate_row: Mapping[str, Any], trace_inventory: Mapping[str, Mapping[int, SweepTrace]]) -> pd.DataFrame:
    file_id = str(candidate_row["file_id"])
    sweeps = trace_inventory[file_id]
    params = reconstruct_candidate_params(candidate_row)
    rows = []
    for sweep_idx, sweep_trace in sorted(sweeps.items()):
        try:
            sim = simulate_odeint(params, {"experiment_type": sweep_trace.condition, "current_na": sweep_trace.current_na, "t_eval_ms": sweep_trace.time_ms_fit}, z0=DEFAULT_Z0, t_eval_ms=sweep_trace.time_ms_fit, return_hidden=False)
            sim_vm = np.asarray(sim["Vm"], dtype=float)
            sim_health = "ok"
        except Exception:
            sim_vm = np.full_like(sweep_trace.vm_fit, np.nan, dtype=float)
            sim_health = "failed"
        for t_ms, obs_vm, pred_vm in zip(sweep_trace.time_ms_fit, sweep_trace.vm_fit, sim_vm):
            rows.append({
                "file_id": file_id,
                "region": sweep_trace.region,
                "condition": sweep_trace.condition,
                "candidate_id": candidate_row.get("candidate_id", "unknown"),
                "sweep": int(sweep_idx),
                "current_na": int(sweep_trace.current_na),
                "time_ms": float(t_ms),
                "vm_observed_mV": float(obs_vm),
                "vm_predicted_mV": float(pred_vm) if np.isfinite(pred_vm) else np.nan,
                "simulation_health": sim_health,
            })
    return pd.DataFrame(rows)


def build_best_candidate_overlay_table(candidates_df: pd.DataFrame, trace_inventory: Mapping[str, Mapping[int, SweepTrace]], accepted_only: bool = False) -> pd.DataFrame:
    source = candidates_df.copy()
    if accepted_only and "accepted_all6" in source.columns:
        source = source[source["accepted_all6"]].copy()
    if source.empty:
        return pd.DataFrame()
    rows = []
    for file_id, group in source.groupby("file_id", sort=True):
        group = group.sort_values(["accepted_all6", "mean_weighted_pass_fraction", "mean_trace_rmse_mV"], ascending=[False, False, True])
        rows.append(build_candidate_overlay_frame(group.iloc[0].to_dict(), trace_inventory))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_cell_candidates(project_root: str | Path, file_id: str, trace_inventory_all: Mapping[str, Mapping[int, SweepTrace]], feature_df: pd.DataFrame, thresholds_df: pd.DataFrame, cfg: Step04Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sweeps = trace_inventory_all[file_id]
    empirical_rows = {
        int(sweep_idx): feature_df[(feature_df["file_id"] == file_id) & (feature_df["sweep"] == sweep_idx)].iloc[0].to_dict()
        for sweep_idx in sorted(sweeps)
    }
    cand_df, sweep_df = _fit_cell_all6(Path(project_root), file_id, sweeps, empirical_rows, thresholds_df, cfg)
    held_df = pd.DataFrame()
    if not cand_df.empty:
        best = cand_df.iloc[0].to_dict()
        held_df = _fit_holdout(file_id, sweeps, empirical_rows, thresholds_df, best, cfg)
    holdout_pass_count = int(held_df["heldout_pass"].sum()) if not held_df.empty else 0
    summary = {
        "file_id": file_id,
        "region": next(iter(sweeps.values())).region,
        "condition": next(iter(sweeps.values())).condition,
        "n_candidates": int(len(cand_df)),
        "n_accepted_candidates": int(cand_df["accepted_all6"].sum()) if not cand_df.empty else 0,
        "best_candidate_id": str(cand_df.iloc[0]["candidate_id"]) if not cand_df.empty else None,
        "best_trace_rmse_mV": float(cand_df.iloc[0]["mean_trace_rmse_mV"]) if not cand_df.empty else np.nan,
        "best_weighted_pass_fraction": float(cand_df.iloc[0]["mean_weighted_pass_fraction"]) if not cand_df.empty else np.nan,
        "holdout_pass_count": holdout_pass_count,
        "holdout_mean_rmse_mV": float(held_df["heldout_trace_rmse_mV"].mean()) if not held_df.empty else np.nan,
        "holdout_mean_pass_fraction": float(held_df["heldout_weighted_pass_fraction"].mean()) if not held_df.empty else np.nan,
        "cell_reviewer_facing": bool((not cand_df.empty) and bool(cand_df.iloc[0]["accepted_all6"]) and holdout_pass_count >= cfg.heldout_min_pass_count),
    }
    return cand_df, sweep_df, {"heldout": held_df, "summary": summary}


def run_step04_cell_specific_six_sweep_fitting(project_root: str | Path, output_dir: str | Path | None = None, max_cells: int | None = None, n_fit_points: int = N_FIT_POINTS_DEFAULT, selected_file_ids: Optional[Sequence[str]] = None, n_starts: int = N_STARTS_DEFAULT, max_nfev_all6: int = MAX_NFEV_ALL6_DEFAULT, max_nfev_holdout: int = MAX_NFEV_HOLDOUT_DEFAULT, trace_rmse_accept: float = TRACE_RMSE_ACCEPT_DEFAULT, feature_pass_accept: float = FEATURE_PASS_ACCEPT_DEFAULT, heldout_trace_rmse_accept: float = HELDOUT_TRACE_RMSE_ACCEPT_DEFAULT, heldout_pass_accept: float = HELDOUT_PASS_ACCEPT_DEFAULT, heldout_min_pass_count: int = HELDOUT_MIN_PASS_COUNT_DEFAULT, accepted_top_k_per_cell: int = ACCEPTED_TOP_K_PER_CELL_DEFAULT, reuse_step02_outputs: bool = True) -> dict[str, pd.DataFrame]:
    cfg = Step04Config(project_root=Path(project_root).resolve(), output_dir=(Path(output_dir) if output_dir is not None else None), max_cells=max_cells, selected_file_ids=list(selected_file_ids) if selected_file_ids else None, n_fit_points=n_fit_points, n_starts=n_starts, max_nfev_all6=max_nfev_all6, max_nfev_holdout=max_nfev_holdout, trace_rmse_accept=trace_rmse_accept, feature_pass_accept=feature_pass_accept, heldout_trace_rmse_accept=heldout_trace_rmse_accept, heldout_pass_accept=heldout_pass_accept, heldout_min_pass_count=heldout_min_pass_count, accepted_top_k_per_cell=accepted_top_k_per_cell).resolve()
    paths = _project_paths(cfg.project_root)
    step02 = load_step02_outputs_or_run(cfg.project_root, reuse_existing=reuse_step02_outputs)
    feature_df = step02["feature_table_by_sweep"]
    thresholds_df = step02["condition_region_sweep_thresholds"]
    cell_counts_df = step02["region_condition_cell_counts"]

    if cfg.selected_file_ids is not None:
        file_ids = list(cfg.selected_file_ids)
        if cfg.max_cells is not None:
            file_ids = file_ids[: int(cfg.max_cells)]
    else:
        file_meta = feature_df[["file_id", "condition", "region"]].drop_duplicates().sort_values(["condition", "region", "file_id"])
        file_ids = file_meta["file_id"].tolist()
        if cfg.max_cells is not None:
            file_ids = file_ids[: int(cfg.max_cells)]

    trace_inventory = build_cell_trace_inventory(paths["atf_dir"], n_fit_points=cfg.n_fit_points, file_ids=file_ids)
    candidate_tables: list[pd.DataFrame] = []
    sweep_tables: list[pd.DataFrame] = []
    heldout_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for file_id in sorted(trace_inventory):
        cand_df, sweep_df, extra = evaluate_cell_candidates(cfg.project_root, file_id, trace_inventory, feature_df, thresholds_df, cfg)
        candidate_tables.append(cand_df)
        sweep_tables.append(sweep_df)
        heldout_tables.append(extra["heldout"])
        summaries.append(extra["summary"])

    candidates_df = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    sweep_metrics_df = pd.concat(sweep_tables, ignore_index=True) if sweep_tables else pd.DataFrame()
    heldout_df = pd.concat(heldout_tables, ignore_index=True) if heldout_tables else pd.DataFrame()
    summary_df = pd.DataFrame(summaries).sort_values(["condition", "region", "file_id"]).reset_index(drop=True) if summaries else pd.DataFrame()
    accepted_df = candidates_df[candidates_df["accepted_all6"]].copy().reset_index(drop=True) if (not candidates_df.empty and "accepted_all6" in candidates_df.columns) else pd.DataFrame()
    contract_df = acceptance_contract_table(cfg)
    inventory_df = pd.DataFrame([
        {"file_id": fid, "region": next(iter(sweeps.values())).region, "condition": next(iter(sweeps.values())).condition, "n_sweeps": len(sweeps)}
        for fid, sweeps in trace_inventory.items()
    ]).sort_values(["condition", "region", "file_id"]).reset_index(drop=True) if trace_inventory else pd.DataFrame()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(cfg.output_dir / "cell_fit_candidates.csv", index=False)
    accepted_df.to_csv(cfg.output_dir / "accepted_cell_ensembles.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "cell_fit_quality_summary.csv", index=False)
    heldout_df.to_csv(cfg.output_dir / "heldout_current_screen.csv", index=False)
    contract_df.to_csv(cfg.output_dir / "acceptance_contract.csv", index=False)
    sweep_metrics_df.to_csv(cfg.output_dir / "candidate_sweep_metrics.csv", index=False)
    inventory_df.to_csv(cfg.output_dir / "cell_trace_inventory.csv", index=False)

    analysis_summary = {
        "step_name": "Step 04 cell-specific six-sweep fitting and accepted ensemble construction",
        "implementation_mode": "shared-cell least-squares fitting with multi-start and leave-one-sweep-out validation",
        "n_cells": int(len(inventory_df)),
        "n_candidates": int(len(candidates_df)),
        "n_accepted_candidates": int(len(accepted_df)),
        "n_reviewer_facing_cells": int(summary_df["cell_reviewer_facing"].sum()) if not summary_df.empty else 0,
        "selected_file_ids": sorted(trace_inventory.keys()),
        "n_fit_points": int(cfg.n_fit_points),
        "n_starts": int(cfg.n_starts),
        "max_nfev_all6": int(cfg.max_nfev_all6),
        "max_nfev_holdout": int(cfg.max_nfev_holdout),
        "trace_rmse_accept": float(cfg.trace_rmse_accept),
        "feature_pass_accept": float(cfg.feature_pass_accept),
        "heldout_trace_rmse_accept": float(cfg.heldout_trace_rmse_accept),
        "heldout_pass_accept": float(cfg.heldout_pass_accept),
        "heldout_min_pass_count": int(cfg.heldout_min_pass_count),
        "accepted_top_k_per_cell": int(cfg.accepted_top_k_per_cell),
        "uses_step02_thresholds": True,
        "uses_region_specific_acceptance": True,
        "model_alignment": "src.astro_model.model matches the expected reviewer-facing ODE form with numerical safeguards only",
    }
    (cfg.output_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")
    return {
        "cell_fit_candidates": candidates_df,
        "accepted_cell_ensembles": accepted_df,
        "cell_fit_quality_summary": summary_df,
        "heldout_current_screen": heldout_df,
        "acceptance_contract": contract_df,
        "candidate_sweep_metrics": sweep_metrics_df,
        "feature_table_by_sweep": feature_df,
        "region_condition_cell_counts": cell_counts_df,
        "cell_trace_inventory": inventory_df,
    }
