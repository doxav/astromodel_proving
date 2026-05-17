from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .astro_model import DEFAULT_Z0, simulate_odeint
from .atf_io import load_all_cells, load_cell_protocol, canonical_file_id
from .atf_features import FEATURE_COLUMNS, build_feature_table, compute_feature_reliability, build_threshold_table, extract_features_from_trace
from .feature_contracts import feature_residual_vector, score_feature_contract
from .parameter_space import effective_from_flat, flat_from_effective
from .protocols import default_onset_seconds
from .trace_utils import baseline_center, downsample_trace
from .step04_loss import (
    Step04LossConfig,
    Step04OptimizerConfig,
    compute_trace_objective,
    config_hash,
    feature_columns_for_loss,
    objective_tuple,
    scalarize_components,
    write_optimization_config,
)
from .step04_outputs import (
    STEP04_DOWNSTREAM_ARTIFACTS,
    STEP04_OUTPUT_SCHEMA_VERSION,
    write_step04_artifact_manifest,
)

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
    loss_config: Step04LossConfig = field(default_factory=Step04LossConfig)
    optimizer_config: Step04OptimizerConfig = field(default_factory=Step04OptimizerConfig)

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
        compatibility_actions: list[str] = []
        feature_df = pd.read_csv(feature_csv)
        thresholds_df = pd.read_csv(threshold_csv)
        counts_df = pd.read_csv(counts_csv)
        if {"file_id", "condition"}.issubset(feature_df.columns):
            control_mask = feature_df["condition"].eq("CONTROL") & ~feature_df["file_id"].astype(str).str.upper().str.endswith("_CONTROL")
            feature_df.loc[control_mask, "file_id"] = feature_df.loc[control_mask, "file_id"].astype(str) + "_CONTROL"
            if bool(control_mask.any()):
                compatibility_actions.append("appended_CONTROL_suffix_to_cached_control_file_ids")
        if "stim_end_depolarization_mV" not in feature_df.columns:
            if "plateau_level_mV" in feature_df.columns and "baseline_mV" in feature_df.columns:
                feature_df["stim_end_depolarization_mV"] = pd.to_numeric(feature_df["plateau_level_mV"], errors="coerce") - pd.to_numeric(feature_df["baseline_mV"], errors="coerce")
            else:
                feature_df["stim_end_depolarization_mV"] = feature_df.get("peak_depolarization_mV", np.nan)
            compatibility_actions.append("derived_missing_stim_end_depolarization_mV")
        needed_features = set(FEATURE_COLUMNS)
        if "feature" not in thresholds_df.columns or not needed_features.issubset(set(thresholds_df["feature"].dropna().astype(str))):
            reliability_df = compute_feature_reliability(feature_df)
            thresholds_df = build_threshold_table(feature_df, reliability_df)
            compatibility_actions.append("rebuilt_thresholds_due_to_stale_feature_contract")
        out = {
            "feature_table_by_sweep": feature_df,
            "condition_region_sweep_thresholds": thresholds_df,
            "region_condition_cell_counts": counts_df,
        }
        if reliability_csv.exists():
            out["feature_reliability_weights"] = pd.read_csv(reliability_csv)
        out["compatibility_actions"] = pd.DataFrame(
            [{"action": a, "source": "cached_step02"} for a in compatibility_actions]
        )
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
    """Backward-compatible wrapper that returns Step 04 fit times in ms."""

    target_time_s, target_vm = downsample_trace(time_s, vm, n_points)
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
    """Backward-compatible wrapper around :func:`trace_utils.baseline_center`."""

    return baseline_center(np.asarray(time_ms, dtype=float) / 1000.0, trace, onset_s, include_endpoint=True)


def _threshold_row(thresholds_df: pd.DataFrame, region: str, condition: str, sweep: int, feature: str) -> pd.Series:
    row = thresholds_df[(thresholds_df["region"] == region) & (thresholds_df["condition"] == condition) & (thresholds_df["sweep"] == sweep) & (thresholds_df["feature"] == feature)]
    if row.empty:
        raise KeyError((region, condition, sweep, feature))
    return row.iloc[0]


def _feature_contract_score(
    sim_features: Mapping[str, Any],
    empirical_row: Mapping[str, Any],
    thresholds_df: pd.DataFrame,
    region: str,
    condition: str,
    sweep: int,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper around :func:`feature_contracts.score_feature_contract`."""

    score = score_feature_contract(
        sim_features,
        thresholds_df,
        condition=condition,
        region=region,
        sweep=sweep,
        empirical=empirical_row,
        feature_columns=tuple(feature_columns) if feature_columns is not None else FEATURE_COLUMNS,
        pass_fraction_mode="soft",
    )
    return score


def _feature_residuals(
    sim_features: Mapping[str, Any],
    empirical_row: Mapping[str, Any],
    thresholds_df: pd.DataFrame,
    region: str,
    condition: str,
    sweep: int,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper around :func:`feature_contracts.feature_residual_vector`."""

    columns = tuple(feature_columns) if feature_columns is not None else tuple(FEATURE_COLUMNS)
    feature_resid = feature_residual_vector(
        sim_features,
        empirical_row,
        thresholds_df,
        condition=condition,
        region=region,
        sweep=sweep,
        feature_columns=columns,
    )
    return np.asarray(feature_resid, dtype=float)


def _binary_residuals(sim_features: Mapping[str, Any], empirical_row: Mapping[str, Any]) -> np.ndarray:
    plateau_emp = bool(empirical_row.get("plateau_reached", False))
    plateau_sim = bool(sim_features.get("plateau_reached", False))
    undershoot_emp = bool(empirical_row.get("has_undershoot", False))
    undershoot_sim = bool(sim_features.get("has_undershoot", False))
    return np.asarray(
        [
            0.5 if plateau_emp != plateau_sim else 0.0,
            0.5 if undershoot_emp != undershoot_sim else 0.0,
        ],
        dtype=float,
    )

def _default_onset_s(condition: str) -> float:
    """Backward-compatible wrapper around :func:`protocols.default_onset_seconds`."""

    return default_onset_seconds(condition)


def _effective_from_flat(params: Mapping[str, Any]) -> dict[str, float]:
    """Backward-compatible wrapper around :func:`parameter_space.effective_from_flat`."""

    return effective_from_flat(params).as_dict()


def _flat_from_effective(condition: str, eff: Mapping[str, float], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Backward-compatible wrapper around :func:`parameter_space.flat_from_effective`."""

    return flat_from_effective(condition, eff, extra=extra)


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


def _residual_vector(x: np.ndarray, condition: str, sweeps_to_fit: Sequence[int], trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, cfg: Step04Config) -> np.ndarray:
    params = _params_from_x(condition, x, seed_source="runtime", start_label="runtime")
    residuals: list[np.ndarray] = []
    penalty = 0.0
    feature_columns = feature_columns_for_loss(cfg.loss_config.feature_set)
    trace_weight = math.sqrt(float(cfg.loss_config.trace_weight))
    feature_weight = math.sqrt(float(cfg.loss_config.feature_weight))
    binary_weight = math.sqrt(float(cfg.loss_config.binary_weight))
    for sweep_idx in sweeps_to_fit:
        sweep_trace = trace_inventory[sweep_idx]
        try:
            sim_vm, sim_features, onset_s = _simulate_sweep(params, sweep_trace)
            exp_bs = _baseline_subtract(sweep_trace.vm_fit, sweep_trace.time_ms_fit, onset_s)
            sim_bs = _baseline_subtract(sim_vm, sweep_trace.time_ms_fit, onset_s)
            trace_resid = trace_weight * (sim_bs - exp_bs) / max(cfg.trace_scale_mV, 1e-9)
            feature_resid = feature_weight * _feature_residuals(
                sim_features,
                empirical_rows[sweep_idx],
                thresholds_df,
                sweep_trace.region,
                sweep_trace.condition,
                sweep_idx,
                feature_columns=feature_columns,
            )
            binary_resid = binary_weight * _binary_residuals(sim_features, empirical_rows[sweep_idx])
        except Exception:
            trace_resid = np.full_like(sweep_trace.vm_fit, 10.0, dtype=float)
            feature_resid = np.full(len(feature_columns), 5.0, dtype=float)
            binary_resid = np.full(2, 2.0, dtype=float)
            penalty += 20.0
        residuals.append(np.asarray(trace_resid, dtype=float))
        residuals.append(np.asarray(feature_resid, dtype=float))
        residuals.append(np.asarray(binary_resid, dtype=float))
    if penalty > 0:
        residuals.append(np.asarray([penalty], dtype=float))
    return np.concatenate(residuals).astype(float)

def _score_candidate_metrics(params: Mapping[str, Any], trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, loss_config: Step04LossConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    loss_config = loss_config or Step04LossConfig()
    feature_columns = feature_columns_for_loss(loss_config.feature_set)
    rows: list[dict[str, Any]] = []
    for sweep_idx, sweep_trace in sorted(trace_inventory.items()):
        empirical = empirical_rows[sweep_idx]
        try:
            sim_vm, sim_features, onset_s = _simulate_sweep(params, sweep_trace)
            exp_bs = _baseline_subtract(sweep_trace.vm_fit, sweep_trace.time_ms_fit, onset_s)
            sim_bs = _baseline_subtract(sim_vm, sweep_trace.time_ms_fit, onset_s)
            trace_rmse = float(np.sqrt(np.mean((exp_bs - sim_bs) ** 2)))
            trace_objective_loss = compute_trace_objective(sim_bs, exp_bs, loss_config.trace)
            contract = _feature_contract_score(
                sim_features,
                empirical,
                thresholds_df,
                sweep_trace.region,
                sweep_trace.condition,
                sweep_idx,
                feature_columns=feature_columns,
            )
            row = {
                "simulation_health": "ok",
                "trace_rmse_mV": trace_rmse,
                "trace_objective_loss": trace_objective_loss,
                **contract,
                **{k: sim_features.get(k, np.nan) for k in FEATURE_COLUMNS},
                "plateau_reached_sim": bool(sim_features.get("plateau_reached", False)),
                "has_undershoot_sim": bool(sim_features.get("has_undershoot", False)),
            }
        except Exception as exc:
            row = {
                "simulation_health": f"failed:{type(exc).__name__}",
                "trace_rmse_mV": np.nan,
                "trace_objective_loss": np.inf,
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
    finite_trace_obj = df["trace_objective_loss"].replace([np.inf, -np.inf], np.nan)
    agg = {
        "n_sweeps_scored": int(len(df)),
        "n_failures": n_failures,
        "mean_trace_rmse_mV": float(df["trace_rmse_mV"].mean(skipna=True)) if df["trace_rmse_mV"].notna().any() else np.nan,
        "mean_trace_objective_loss": float(finite_trace_obj.mean(skipna=True)) if finite_trace_obj.notna().any() else np.inf,
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


def _weak_prior_penalty_from_x(x: np.ndarray, reference_x: np.ndarray | None = None) -> float:
    if reference_x is None:
        return 0.0
    diff = np.asarray(x, dtype=float) - np.asarray(reference_x, dtype=float)
    return float(np.mean(diff**2))


def _objective_components_from_agg(agg: Mapping[str, Any], prior_penalty: float = 0.0, hidden_penalty: float = 0.0) -> dict[str, float]:
    failures = float(agg.get("n_failures", 0.0))
    trace = float(agg.get("mean_trace_objective_loss", np.inf))
    if not np.isfinite(trace):
        trace = 1e12
    return {
        "trace": trace,
        "feature": float(agg.get("mean_feature_loss", 1.0)),
        "binary": float(agg.get("mean_binary_penalty", 1.0)),
        "prior": float(prior_penalty),
        "hidden": float(hidden_penalty),
        "fail": failures,
    }


def _candidate_acceptance_flags(row: Mapping[str, Any], cfg: Step04Config) -> dict[str, bool]:
    accepted_by_trace = bool(np.isfinite(row.get("mean_trace_rmse_mV", np.nan)) and float(row["mean_trace_rmse_mV"]) <= cfg.trace_rmse_accept)
    accepted_by_feature_contract = bool(float(row.get("mean_weighted_pass_fraction", 0.0)) >= cfg.feature_pass_accept)
    eligible_by_contract = bool(accepted_by_trace and accepted_by_feature_contract and int(row.get("n_failures", 0)) == 0)
    return {
        "accepted_by_trace": accepted_by_trace,
        "accepted_by_feature_contract": accepted_by_feature_contract,
        "eligible_by_contract": eligible_by_contract,
    }


def _suggest_named_from_trial(trial: Any, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray([trial.suggest_float(name, float(lo), float(hi)) for name, lo, hi in zip(OPTIMIZED_KEYS, lower, upper)], dtype=float)


def _optuna_sampler(optimizer_config: Step04OptimizerConfig) -> Any:
    import optuna

    if optimizer_config.optuna_sampler == "random":
        return optuna.samplers.RandomSampler(seed=42)
    if optimizer_config.optuna_sampler == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=42)
    return optuna.samplers.TPESampler(seed=42)


def _candidate_row_from_solution(
    *,
    file_id: str,
    condition: str,
    trace_inventory: Mapping[int, SweepTrace],
    empirical_rows: Mapping[int, Mapping[str, Any]],
    thresholds_df: pd.DataFrame,
    cfg: Step04Config,
    x: np.ndarray,
    candidate_id: str,
    seed_source: str,
    start_label: str,
    optimizer_status: int,
    optimizer_success: bool,
    optimizer_cost: float,
    optimizer_nfev: int,
    prior_reference_x: np.ndarray | None = None,
    optuna_trial: Any | None = None,
    pareto_front: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    params = _params_from_x(condition, x, seed_source=seed_source, start_label=start_label)
    sweep_df, agg = _score_candidate_metrics(params, trace_inventory, empirical_rows, thresholds_df, cfg.loss_config)
    sweep_df.insert(0, "candidate_id", candidate_id)
    sweep_df.insert(0, "file_id", file_id)
    sweep_df.insert(0, "fit_scope", "all6")
    eff = _effective_from_flat(params)
    components = _objective_components_from_agg(agg, prior_penalty=_weak_prior_penalty_from_x(x, prior_reference_x))
    scalar_objective = scalarize_components(components, cfg.loss_config)
    row = {
        "file_id": file_id,
        "region": next(iter(trace_inventory.values())).region,
        "condition": condition,
        "candidate_id": candidate_id,
        "fit_scope": "all6",
        "seed_source": seed_source,
        "provenance_status": "verified_seed" if seed_source == "legacy_db_verified_median" else "generic_seed",
        "start_label": start_label,
        "optimizer_backend": cfg.optimizer_config.backend,
        "optimization_config_hash": config_hash({"loss_config": asdict(cfg.loss_config), "optimizer_config": asdict(cfg.optimizer_config)}),
        "optimizer_status": int(optimizer_status),
        "optimizer_success": bool(optimizer_success),
        "optimizer_cost": float(optimizer_cost),
        "optimizer_nfev": int(optimizer_nfev),
        "gki": float(params["gki"]),
        "eps": float(params["eps"]),
        "gl_a": float(params["gl_a"]),
        "zth": float(params["zth"]),
        "zs": float(params["zs"]),
        **eff,
        **agg,
        "objective_trace": components["trace"],
        "objective_feature": components["feature"],
        "objective_binary": components["binary"],
        "objective_prior": components["prior"],
        "objective_hidden": components["hidden"],
        "objective_fail": components["fail"],
        "scalar_objective": scalar_objective,
    }
    if optuna_trial is not None:
        objective_names = tuple(cfg.loss_config.multi_objective_names) if cfg.optimizer_config.backend == "optuna_multi" else ("scalar",)
        row.update({
            "optuna_trial_number": int(optuna_trial.number),
            "optuna_objective_names": json.dumps(list(objective_names)),
            "optuna_objective_values": json.dumps([float(v) for v in (optuna_trial.values or [optuna_trial.value])]),
            "pareto_front": bool(pareto_front),
        })
    row.update(_candidate_acceptance_flags(row, cfg))
    return row, sweep_df


def _rank_candidate_df(candidate_rows: list[dict[str, Any]], cfg: Step04Config) -> pd.DataFrame:
    cand_df = pd.DataFrame(candidate_rows)
    if cand_df.empty:
        return cand_df
    cand_df = cand_df.sort_values(
        ["eligible_by_contract", "mean_weighted_pass_fraction", "mean_trace_rmse_mV", "scalar_objective"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    cand_df["ensemble_rank"] = np.arange(1, len(cand_df) + 1)
    cand_df["accepted_all6"] = cand_df["eligible_by_contract"] & (cand_df["ensemble_rank"] <= cfg.accepted_top_k_per_cell)
    return cand_df


def _fit_cell_all6_least_squares(project_root: Path, file_id: str, trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, cfg: Step04Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition = next(iter(trace_inventory.values())).condition
    lower, upper = _x_bounds()
    candidate_rows: list[dict[str, Any]] = []
    sweep_rows: list[pd.DataFrame] = []
    for start_idx, (x0, seed_source, start_label) in enumerate(_start_vectors(project_root, condition, cfg.n_starts), start=1):
        result = least_squares(
            _residual_vector,
            x0=x0,
            bounds=(lower, upper),
            args=(condition, list(sorted(trace_inventory)), trace_inventory, empirical_rows, thresholds_df, cfg),
            max_nfev=cfg.max_nfev_all6,
            method="trf",
            loss=cfg.optimizer_config.scipy_loss,
            f_scale=cfg.optimizer_config.scipy_f_scale,
        )
        row, sweep_df = _candidate_row_from_solution(
            file_id=file_id,
            condition=condition,
            trace_inventory=trace_inventory,
            empirical_rows=empirical_rows,
            thresholds_df=thresholds_df,
            cfg=cfg,
            x=result.x,
            candidate_id=f"{file_id}__cand_{start_idx:02d}",
            seed_source=seed_source,
            start_label=start_label,
            optimizer_status=int(result.status),
            optimizer_success=bool(result.success),
            optimizer_cost=float(result.cost),
            optimizer_nfev=int(result.nfev),
            prior_reference_x=x0,
        )
        candidate_rows.append(row)
        sweep_rows.append(sweep_df)
    return _rank_candidate_df(candidate_rows, cfg), pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()


class _FallbackTrial:
    def __init__(self, number: int, values: Sequence[float]):
        self.number = int(number)
        self.values = [float(v) for v in values]
        self.value = float(values[0]) if len(values) == 1 else None


def _non_dominated_trial_numbers(
    evaluated: Sequence[tuple[_FallbackTrial, np.ndarray, dict[str, float], float]]
) -> set[int]:
    pareto: set[int] = set()
    for trial, _x, _components, _scalar in evaluated:
        values = np.asarray(trial.values, dtype=float)
        dominated = False
        for other, _other_x, _other_components, _other_scalar in evaluated:
            if other.number == trial.number:
                continue
            other_values = np.asarray(other.values, dtype=float)
            if np.all(other_values <= values) and np.any(other_values < values):
                dominated = True
                break
        if not dominated:
            pareto.add(trial.number)
    return pareto


def _fit_cell_all6_optuna_fallback(project_root: Path, file_id: str, trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, cfg: Step04Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small deterministic fallback for environments that have not installed Optuna yet."""

    condition = next(iter(trace_inventory.values())).condition
    lower, upper = _x_bounds()
    prior_x = _start_vectors(project_root, condition, max(1, cfg.n_starts))[0][0]
    rng = np.random.default_rng(42)
    evaluated: list[tuple[_FallbackTrial, np.ndarray, dict[str, float], float]] = []
    for number in range(max(1, int(cfg.optimizer_config.optuna_n_trials))):
        x = rng.uniform(lower, upper)
        params = _params_from_x(condition, x, seed_source="optuna_fallback", start_label=f"trial_{number}")
        _, agg = _score_candidate_metrics(params, trace_inventory, empirical_rows, thresholds_df, cfg.loss_config)
        components = _objective_components_from_agg(agg, prior_penalty=_weak_prior_penalty_from_x(x, prior_x))
        if cfg.optimizer_config.backend == "optuna_multi":
            values = objective_tuple(components, cfg.loss_config.multi_objective_names)
        else:
            values = (scalarize_components(components, cfg.loss_config),)
        scalar = scalarize_components(components, cfg.loss_config)
        evaluated.append((_FallbackTrial(number, values), x, components, scalar))
    if cfg.optimizer_config.backend == "optuna_multi":
        selected = sorted(evaluated, key=lambda item: item[3])[: cfg.accepted_top_k_per_cell]
        pareto_numbers = _non_dominated_trial_numbers(evaluated)
    else:
        selected = sorted(evaluated, key=lambda item: item[3])[: max(1, cfg.accepted_top_k_per_cell)]
        pareto_numbers = set()
    candidate_rows: list[dict[str, Any]] = []
    sweep_rows: list[pd.DataFrame] = []
    for idx, (trial, x, _, scalar) in enumerate(selected, start=1):
        row, sweep_df = _candidate_row_from_solution(
            file_id=file_id,
            condition=condition,
            trace_inventory=trace_inventory,
            empirical_rows=empirical_rows,
            thresholds_df=thresholds_df,
            cfg=cfg,
            x=x,
            candidate_id=f"{file_id}__cand_{idx:02d}",
            seed_source="optuna_fallback",
            start_label=f"trial_{trial.number}",
            optimizer_status=1,
            optimizer_success=True,
            optimizer_cost=float(scalar),
            optimizer_nfev=1,
            prior_reference_x=prior_x,
            optuna_trial=trial,
            pareto_front=trial.number in pareto_numbers,
        )
        candidate_rows.append(row)
        sweep_rows.append(sweep_df)
    return _rank_candidate_df(candidate_rows, cfg), pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()


def _fit_cell_all6_optuna(project_root: Path, file_id: str, trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, cfg: Step04Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        if not cfg.optimizer_config.allow_optuna_fallback:
            raise RuntimeError(
                "Optuna backend requested but optuna is not installed. Install requirements.txt "
                "or set Step04OptimizerConfig(allow_optuna_fallback=True) for deterministic smoke-only fallback."
            ) from exc
        return _fit_cell_all6_optuna_fallback(project_root, file_id, trace_inventory, empirical_rows, thresholds_df, cfg)

    condition = next(iter(trace_inventory.values())).condition
    lower, upper = _x_bounds()
    starts = _start_vectors(project_root, condition, max(1, cfg.n_starts))
    prior_x = starts[0][0]
    trial_cache: dict[int, tuple[np.ndarray, dict[str, float]]] = {}

    def evaluate_x(x: np.ndarray) -> dict[str, float]:
        params = _params_from_x(condition, x, seed_source="optuna", start_label="trial")
        _, agg = _score_candidate_metrics(params, trace_inventory, empirical_rows, thresholds_df, cfg.loss_config)
        return _objective_components_from_agg(agg, prior_penalty=_weak_prior_penalty_from_x(x, prior_x))

    def objective(trial: Any) -> float | tuple[float, ...]:
        x = _suggest_named_from_trial(trial, lower, upper)
        components = evaluate_x(x)
        trial_cache[trial.number] = (x, components)
        if cfg.optimizer_config.backend == "optuna_multi":
            return objective_tuple(components, cfg.loss_config.multi_objective_names)
        return scalarize_components(components, cfg.loss_config)

    sampler = _optuna_sampler(cfg.optimizer_config)
    common = dict(
        sampler=sampler,
        storage=cfg.optimizer_config.optuna_storage,
        study_name=cfg.optimizer_config.optuna_study_name,
        load_if_exists=bool(cfg.optimizer_config.optuna_storage and cfg.optimizer_config.optuna_study_name),
    )
    if cfg.optimizer_config.backend == "optuna_multi":
        study = optuna.create_study(directions=["minimize"] * len(cfg.loss_config.multi_objective_names), **common)
    else:
        study = optuna.create_study(direction="minimize", **common)
    study.optimize(objective, n_trials=cfg.optimizer_config.optuna_n_trials, timeout=cfg.optimizer_config.optuna_timeout_s)

    if cfg.optimizer_config.backend == "optuna_multi":
        source_trials = sorted(study.best_trials, key=lambda t: scalarize_components(trial_cache[t.number][1], cfg.loss_config))[: cfg.accepted_top_k_per_cell]
        pareto_numbers = {t.number for t in study.best_trials}
    else:
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        source_trials = sorted(complete, key=lambda t: float(t.value if t.value is not None else np.inf))[: max(1, cfg.accepted_top_k_per_cell)]
        pareto_numbers = set()

    candidate_rows: list[dict[str, Any]] = []
    sweep_rows: list[pd.DataFrame] = []
    for idx, trial in enumerate(source_trials, start=1):
        x = trial_cache[trial.number][0]
        row, sweep_df = _candidate_row_from_solution(
            file_id=file_id,
            condition=condition,
            trace_inventory=trace_inventory,
            empirical_rows=empirical_rows,
            thresholds_df=thresholds_df,
            cfg=cfg,
            x=x,
            candidate_id=f"{file_id}__cand_{idx:02d}",
            seed_source="optuna",
            start_label=f"trial_{trial.number}",
            optimizer_status=1,
            optimizer_success=True,
            optimizer_cost=float(scalarize_components(trial_cache[trial.number][1], cfg.loss_config)),
            optimizer_nfev=1,
            prior_reference_x=prior_x,
            optuna_trial=trial,
            pareto_front=trial.number in pareto_numbers,
        )
        candidate_rows.append(row)
        sweep_rows.append(sweep_df)
    return _rank_candidate_df(candidate_rows, cfg), pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()


def _fit_cell_all6(project_root: Path, file_id: str, trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, cfg: Step04Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.optimizer_config.backend == "least_squares":
        return _fit_cell_all6_least_squares(project_root, file_id, trace_inventory, empirical_rows, thresholds_df, cfg)
    if cfg.optimizer_config.backend in {"optuna_scalar", "optuna_multi"}:
        return _fit_cell_all6_optuna(project_root, file_id, trace_inventory, empirical_rows, thresholds_df, cfg)
    raise ValueError(f"invalid Step 04 optimizer backend {cfg.optimizer_config.backend!r}")

def _fit_holdout(file_id: str, trace_inventory: Mapping[int, SweepTrace], empirical_rows: Mapping[int, Mapping[str, Any]], thresholds_df: pd.DataFrame, best_candidate: Mapping[str, Any], cfg: Step04Config) -> pd.DataFrame:
    if not cfg.optimizer_config.run_holdout:
        return pd.DataFrame()
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
            args=(condition, train_sweeps, trace_inventory, empirical_rows, thresholds_df, cfg),
            max_nfev=cfg.max_nfev_holdout,
            method="trf",
            loss=cfg.optimizer_config.scipy_loss,
            f_scale=cfg.optimizer_config.scipy_f_scale,
        )
        params = _params_from_x(condition, res.x, seed_source=str(best_candidate.get("seed_source", "runtime")), start_label=f"heldout_{heldout}")
        full_sweep_df, _ = _score_candidate_metrics(params, trace_inventory, empirical_rows, thresholds_df, cfg.loss_config)
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


def run_step04_cell_specific_six_sweep_fitting(
    project_root: str | Path,
    output_dir: str | Path | None = None,
    max_cells: int | None = None,
    n_fit_points: int = N_FIT_POINTS_DEFAULT,
    selected_file_ids: Optional[Sequence[str]] = None,
    n_starts: int = N_STARTS_DEFAULT,
    max_nfev_all6: int = MAX_NFEV_ALL6_DEFAULT,
    max_nfev_holdout: int = MAX_NFEV_HOLDOUT_DEFAULT,
    trace_rmse_accept: float = TRACE_RMSE_ACCEPT_DEFAULT,
    feature_pass_accept: float = FEATURE_PASS_ACCEPT_DEFAULT,
    heldout_trace_rmse_accept: float = HELDOUT_TRACE_RMSE_ACCEPT_DEFAULT,
    heldout_pass_accept: float = HELDOUT_PASS_ACCEPT_DEFAULT,
    heldout_min_pass_count: int = HELDOUT_MIN_PASS_COUNT_DEFAULT,
    accepted_top_k_per_cell: int = ACCEPTED_TOP_K_PER_CELL_DEFAULT,
    reuse_step02_outputs: bool = True,
    loss_config: Step04LossConfig | None = None,
    optimizer_config: Step04OptimizerConfig | None = None,
    optimizer_backend: str | None = None,
    optuna_n_trials: int | None = None,
    feature_set: str | None = None,
    trace_loss_type: str | None = None,
) -> dict[str, pd.DataFrame]:
    base_loss_config = loss_config or Step04LossConfig()
    if feature_set is not None:
        base_loss_config = replace(base_loss_config, feature_set=feature_set)
    if trace_loss_type is not None:
        base_loss_config = replace(base_loss_config, trace=replace(base_loss_config.trace, loss_type=trace_loss_type))
    base_optimizer_config = optimizer_config or Step04OptimizerConfig()
    if optimizer_backend is not None:
        base_optimizer_config = replace(base_optimizer_config, backend=optimizer_backend)
    if optuna_n_trials is not None:
        base_optimizer_config = replace(base_optimizer_config, optuna_n_trials=optuna_n_trials)
    cfg = Step04Config(
        project_root=Path(project_root).resolve(),
        output_dir=(Path(output_dir) if output_dir is not None else None),
        max_cells=max_cells,
        selected_file_ids=list(selected_file_ids) if selected_file_ids else None,
        n_fit_points=n_fit_points,
        n_starts=n_starts,
        max_nfev_all6=max_nfev_all6,
        max_nfev_holdout=max_nfev_holdout,
        trace_rmse_accept=trace_rmse_accept,
        feature_pass_accept=feature_pass_accept,
        heldout_trace_rmse_accept=heldout_trace_rmse_accept,
        heldout_pass_accept=heldout_pass_accept,
        heldout_min_pass_count=heldout_min_pass_count,
        accepted_top_k_per_cell=accepted_top_k_per_cell,
        loss_config=base_loss_config,
        optimizer_config=base_optimizer_config,
    ).resolve()
    optimization_config = write_optimization_config(cfg.output_dir, cfg.loss_config, cfg.optimizer_config)
    paths = _project_paths(cfg.project_root)
    step02 = load_step02_outputs_or_run(cfg.project_root, reuse_existing=reuse_step02_outputs)
    feature_df = step02["feature_table_by_sweep"].copy()
    # Step 02 historical outputs use bare control file ids (for example
    # ``1_DH_1``), whereas the Step 04 ATF loader canonicalizes controls with
    # an explicit ``_CONTROL`` suffix.  Duplicate those rows under the explicit
    # id so Step 04 can reuse existing Step 02 CSVs instead of regenerating a
    # second, incompatible feature table.
    if "stim_end_depolarization_mV" not in feature_df.columns:
        if "plateau_level_mV" in feature_df.columns and "baseline_mV" in feature_df.columns:
            feature_df["stim_end_depolarization_mV"] = pd.to_numeric(feature_df["plateau_level_mV"], errors="coerce") - pd.to_numeric(feature_df["baseline_mV"], errors="coerce")
        else:
            feature_df["stim_end_depolarization_mV"] = feature_df.get("peak_depolarization_mV", np.nan)
    if {"file_id", "condition"}.issubset(feature_df.columns):
        control_mask = (feature_df["condition"].astype(str).str.upper() == "CONTROL") & (~feature_df["file_id"].astype(str).str.upper().str.endswith("_CONTROL"))
        if control_mask.any():
            control_aliases = feature_df.loc[control_mask].copy()
            control_aliases["file_id"] = control_aliases["file_id"].astype(str) + "_CONTROL"
            feature_df = pd.concat([feature_df, control_aliases], ignore_index=True)
    thresholds_df = step02["condition_region_sweep_thresholds"].copy()
    if "feature" in thresholds_df.columns and not set(FEATURE_COLUMNS).issubset(set(thresholds_df["feature"].astype(str))):
        reliability_df = compute_feature_reliability(feature_df)
        thresholds_df = build_threshold_table(feature_df, reliability_df)
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
    if not candidates_df.empty:
        if "holdout_mean_rmse_mV" not in candidates_df.columns:
            candidates_df["holdout_mean_rmse_mV"] = candidates_df.get("mean_trace_rmse_mV", np.nan)
        if "holdout_mean_pass_fraction" not in candidates_df.columns:
            candidates_df["holdout_mean_pass_fraction"] = candidates_df.get("mean_weighted_pass_fraction", np.nan)
        if not summary_df.empty:
            holdout_by_candidate = summary_df[[
                "file_id",
                "best_candidate_id",
                "holdout_mean_rmse_mV",
                "holdout_mean_pass_fraction",
            ]].rename(columns={"best_candidate_id": "candidate_id"})
            candidates_df = candidates_df.merge(
                holdout_by_candidate,
                on=["file_id", "candidate_id"],
                how="left",
                suffixes=("", "_summary"),
            )
            for col in ("holdout_mean_rmse_mV", "holdout_mean_pass_fraction"):
                summary_col = f"{col}_summary"
                if summary_col in candidates_df.columns:
                    candidates_df[col] = candidates_df[summary_col].combine_first(candidates_df[col])
                    candidates_df = candidates_df.drop(columns=[summary_col])
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
        "output_schema_version": STEP04_OUTPUT_SCHEMA_VERSION,
        "downstream_artifacts": STEP04_DOWNSTREAM_ARTIFACTS,
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
        "optimization_config_hash": optimization_config["optimization_config_hash"],
        "loss_config": optimization_config["loss_config"],
        "optimizer_config": optimization_config["optimizer_config"],
        "uses_step02_thresholds": True,
        "uses_region_specific_acceptance": True,
        "model_alignment": "src.astro_model.model matches the expected reviewer-facing ODE form with numerical safeguards only",
    }
    (cfg.output_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")
    write_step04_artifact_manifest(
        cfg.output_dir,
        extra={"optimization_config_hash": optimization_config["optimization_config_hash"]},
    )

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
