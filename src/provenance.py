"""Step 00 provenance and historical objective reproducibility audit."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .astro_model import DEFAULT_Z0, normalize_flat_params, normalize_trace_for_target_mode, simulate_odeint
from .optuna_sqlite import StudySpec, parse_db_name, parse_study_name, read_best_trial, read_db_study_summary

SIM_DT_MS = 0.1
EXPERIMENT_CUTS: dict[str, tuple[int, float]] = {
    "MFA": (30, 0.2),
    "BARIUM": (30, 0.2),
    "CONTROL": (0, 0.1),
}
CURRENT_COLUMN_MAP: dict[int, int] = {50: 1, 75: 2, 100: 3, 125: 4, 150: 5, 175: 6}
EXPECTED_ATF_COUNTS: dict[tuple[str, str], int] = {
    ("DH", "CONTROL"): 7,
    ("VH", "CONTROL"): 4,
    ("DH", "MFA"): 6,
    ("VH", "MFA"): 7,
    ("DH", "MFA_BA"): 6,
    ("VH", "MFA_BA"): 7,
}
ATF_REGION_RE = re.compile(r"(^|_)(DH|VH)(_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class AtfInventoryRecord:
    file_id: str
    region: str
    condition: str
    source_path: str


@dataclass(frozen=True)
class LegacyTrace:
    source_name: str
    source_path: Path
    condition: str
    current_na: int
    time_ms: np.ndarray
    trace: np.ndarray
    target_mean_mode: str


class ProvenanceError(RuntimeError):
    """Raised when provenance parsing fails explicitly."""


def project_paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "project_root": root,
        "initial_fit_dir": root / "data" / "1_Initial_xp_fit",
        "atf_dir": root / "data" / "2_K+ Pumps Data",
        "threshold_csv": root / "data" / "threshold_for_good_enough_fits.csv",
        "outputs_dir": root / "outputs" / "provenance",
    }


def _historical_compute_loss(z: Sequence[float], target: Sequence[float], loss_type: str) -> float:
    z_arr = np.asarray(z, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if z_arr.shape != target_arr.shape:
        raise ValueError(f"Loss inputs must have the same shape, got {z_arr.shape} vs {target_arr.shape}")
    loss_type = loss_type.upper()
    if loss_type == "L2":
        return float(np.sum((z_arr - target_arr) ** 2))
    if loss_type == "L1":
        return float(np.sum(np.abs(z_arr - target_arr)))
    if loss_type == "HUBER":
        delta = 1.0
        abs_diff = np.abs(z_arr - target_arr)
        is_small_error = abs_diff <= delta
        squared_loss = 0.5 * (abs_diff ** 2)
        linear_loss = delta * (abs_diff - 0.5 * delta)
        return float(np.sum(np.where(is_small_error, squared_loss, linear_loss)))
    if loss_type == "LOG_COSH":
        return float(np.sum(np.log(np.cosh(z_arr - target_arr))))
    if loss_type == "COMBINED":
        l2_loss = np.sum((z_arr - target_arr) ** 2)
        z_grad = np.gradient(z_arr)
        target_grad = np.gradient(target_arr)
        gradient_loss = 20.0 * np.sum(np.abs(z_grad - target_grad))
        return float(l2_loss + gradient_loss)
    raise ValueError(f"Unknown historical loss type: {loss_type}")


def _downsample_array_median_fast(values: Sequence[float], num_samples: int) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if len(values_arr) < num_samples:
        raise ValueError(f"Cannot downsample array of length {len(values_arr)} to {num_samples} samples")
    points_per_segment = len(values_arr) // num_samples
    if points_per_segment < 1:
        raise ValueError("points_per_segment must be at least 1")
    trimmed_length = points_per_segment * num_samples
    trimmed = values_arr[:trimmed_length].reshape(num_samples, points_per_segment)
    return np.median(trimmed, axis=1)


def parse_atf_filename(path: str | Path) -> AtfInventoryRecord:
    file_path = Path(path)
    stem = file_path.stem
    region_match = ATF_REGION_RE.search(stem)
    if not region_match:
        raise ProvenanceError(f"Could not infer DH/VH region from ATF filename: {file_path.name}")
    region = region_match.group(2).upper()
    upper_stem = stem.upper()
    if "MFA" in upper_stem and "BA" in upper_stem:
        condition = "MFA_BA"
    elif "MFA" in upper_stem:
        condition = "MFA"
    else:
        condition = "CONTROL"
    return AtfInventoryRecord(file_id=stem, region=region, condition=condition, source_path=str(file_path))


def inventory_atf_files(atf_dir: str | Path) -> pd.DataFrame:
    atf_dir = Path(atf_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(atf_dir.glob("*.atf")):
        record = parse_atf_filename(path)
        rows.append({
            "file_id": record.file_id,
            "region": record.region,
            "condition": record.condition,
            "source_path": record.source_path,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ProvenanceError(f"No ATF files found in {atf_dir}")
    if not set(df["region"]).issubset({"DH", "VH"}):
        raise ProvenanceError("Unexpected region label detected in ATF inventory")
    return df.sort_values(["condition", "region", "file_id"]).reset_index(drop=True)


def atf_region_condition_counts(atf_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        atf_df.groupby(["region", "condition"], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["region", "condition"])
        .reset_index(drop=True)
    )
    expected_rows: list[dict[str, Any]] = []
    observed_map = {(row.region, row.condition): int(row.n_cells) for row in counts.itertuples(index=False)}
    for (region, condition), expected in EXPECTED_ATF_COUNTS.items():
        observed = observed_map.get((region, condition), 0)
        expected_rows.append(
            {
                "region": region,
                "condition": condition,
                "n_cells": observed,
                "expected_n_cells": expected,
                "matches_expected": bool(observed == expected),
                "small_stratum": bool(observed < 5),
            }
        )
    return pd.DataFrame(expected_rows).sort_values(["region", "condition"]).reset_index(drop=True)


def build_trace_source_summary(initial_fit_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(initial_fit_dir).glob("*_TRACES*.csv")):
        uses_header = path.name.endswith("_old.csv")
        df = pd.read_csv(path, header=0 if uses_header else None)
        values = df.to_numpy(dtype=float)
        time_ms = values[:, 0]
        dt_ms = float(np.nanmedian(np.diff(time_ms))) if len(time_ms) > 1 else np.nan
        if path.name.startswith("CONTROL"):
            condition = "CONTROL"
        elif path.name.startswith("MFA"):
            condition = "MFA"
        elif path.name.startswith("BARIUM"):
            condition = "BARIUM"
        else:
            condition = "UNKNOWN"
        rows.append(
            {
                "trace_source": path.name,
                "condition": condition,
                "rows": int(values.shape[0]),
                "n_columns": int(values.shape[1]),
                "time_start_ms": float(time_ms[0]),
                "time_end_ms": float(time_ms[-1]),
                "dt_ms": dt_ms,
                "uses_header": uses_header,
                "file_size_bytes": int(path.stat().st_size),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "trace_source"]).reset_index(drop=True)


def load_legacy_trace(trace_source_path: str | Path, condition: str, current_na: int, target_mean_mode: str) -> LegacyTrace:
    path = Path(trace_source_path)
    if not path.exists():
        raise FileNotFoundError(path)
    uses_header = path.name.endswith("_old.csv")
    df = pd.read_csv(path, header=0 if uses_header else None)
    data = df.to_numpy(dtype=float)
    if current_na not in CURRENT_COLUMN_MAP:
        raise ValueError(f"Unsupported current {current_na}")
    cut_after_index, cut_before_ratio = EXPERIMENT_CUTS[condition]
    index_to_cut = data.shape[0] - cut_after_index if cut_after_index > 0 else data.shape[0]
    data_cut = data[:index_to_cut]
    if len(data_cut) <= 1:
        raise ProvenanceError(f"Trace source {path.name} became empty after historical preprocessing")
    data_cut = data_cut[1:]
    time_ms = data_cut[:, 0]
    trace = data_cut[:, CURRENT_COLUMN_MAP[current_na]].copy()
    stable_index = int(round(len(time_ms) * cut_before_ratio))
    if stable_index >= len(time_ms):
        raise ProvenanceError(f"Stable index beyond trace length for {path.name}")
    time_ms = time_ms[stable_index:]
    trace = trace[stable_index:]
    if condition == "MFA" and current_na == 100:
        outlier_mask = (time_ms >= 22649) & (time_ms <= 22650)
        if np.any(outlier_mask):
            outlier_start_index = np.where(outlier_mask)[0][0] - 1
            if outlier_start_index >= 0:
                trace[outlier_mask] = trace[outlier_start_index]
    trace = normalize_trace_for_target_mode(trace, target_mean_mode)
    return LegacyTrace(
        source_name=path.name,
        source_path=path,
        condition=condition,
        current_na=current_na,
        time_ms=time_ms,
        trace=trace,
        target_mean_mode=target_mean_mode,
    )


def recompute_best_trial_objective(
    db_path: str | Path,
    trace_source_path: str | Path,
    relative_tolerance: float = 1e-2,
) -> dict[str, Any]:
    best_trial = read_best_trial(db_path)
    spec = parse_study_name(best_trial.study_name)
    legacy_trace = load_legacy_trace(trace_source_path, spec.condition, spec.current_na, spec.target_mean_mode)
    last_trace_time_ms = float(legacy_trace.time_ms[-1])
    sim_time_ms = np.linspace(0.0, last_trace_time_ms, int(last_trace_time_ms / SIM_DT_MS))
    stable_index_simulation = int(round(len(sim_time_ms) * EXPERIMENT_CUTS[spec.condition][1]))
    sim = simulate_odeint(
        best_trial.params,
        {"experiment_type": spec.condition, "current_na": spec.current_na, "t_eval_ms": sim_time_ms},
        z0=DEFAULT_Z0,
        return_hidden=False,
    )
    vm_stable = np.asarray(sim["Vm"], dtype=float)[stable_index_simulation:]
    vm_downsampled = _downsample_array_median_fast(vm_stable, len(legacy_trace.trace))
    vm_downsampled_norm = normalize_trace_for_target_mode(vm_downsampled, spec.target_mean_mode)
    objective_recomputed = _historical_compute_loss(vm_downsampled_norm, legacy_trace.trace, spec.objective_loss_type)
    relative_error = float(abs(objective_recomputed - best_trial.objective) / (abs(best_trial.objective) + 1e-12))
    status = "verified" if relative_error <= float(relative_tolerance) else "unresolved"
    return {
        "db_name": best_trial.db_name,
        "study_name": best_trial.study_name,
        "condition": spec.condition,
        "current_na": spec.current_na,
        "trace_source": legacy_trace.source_name,
        "target_mean_mode": spec.target_mean_mode,
        "objective_loss_type": spec.objective_loss_type,
        "n_target_points": spec.n_target_points,
        "best_trial_number": best_trial.trial_number,
        "stored_objective": best_trial.objective,
        "recomputed_objective": objective_recomputed,
        "relative_objective_error": relative_error,
        "status": status,
    }


def audit_db_trace_provenance(initial_fit_dir: str | Path, relative_tolerance: float = 1e-2) -> pd.DataFrame:
    initial_fit_dir = Path(initial_fit_dir)
    rows: list[dict[str, Any]] = []
    for db_path in sorted(initial_fit_dir.glob("*.db")):
        condition, _current_na = parse_db_name(db_path)
        candidate_paths: list[Path] = []
        if condition == "CONTROL":
            for name in ["CONTROL_TRACES.csv", "CONTROL_TRACES_old.csv"]:
                candidate_paths.append(initial_fit_dir / name)
        else:
            candidate_paths.append(initial_fit_dir / f"{condition}_TRACES.csv")
        for candidate in candidate_paths:
            if not candidate.exists():
                best_trial = read_best_trial(db_path)
                rows.append(
                    {
                        "db_name": db_path.name,
                        "study_name": best_trial.study_name,
                        "condition": best_trial.condition,
                        "current_na": best_trial.current_na,
                        "trace_source": candidate.name,
                        "target_mean_mode": parse_study_name(best_trial.study_name).target_mean_mode,
                        "objective_loss_type": parse_study_name(best_trial.study_name).objective_loss_type,
                        "n_target_points": parse_study_name(best_trial.study_name).n_target_points,
                        "best_trial_number": best_trial.trial_number,
                        "stored_objective": best_trial.objective,
                        "recomputed_objective": np.nan,
                        "relative_objective_error": np.nan,
                        "status": "missing_source",
                    }
                )
                continue
            rows.append(recompute_best_trial_objective(db_path, candidate, relative_tolerance=relative_tolerance))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ProvenanceError(f"No DB files found in {initial_fit_dir}")
    df = df.sort_values(["condition", "current_na", "relative_objective_error", "trace_source"], na_position="last").reset_index(drop=True)
    chosen = (
        df.sort_values(["db_name", "relative_objective_error", "trace_source"], na_position="last")
        .groupby("db_name", as_index=False)
        .first()[["db_name", "trace_source", "relative_objective_error", "status"]]
        .rename(
            columns={
                "trace_source": "chosen_trace_source",
                "relative_objective_error": "chosen_relative_objective_error",
                "status": "chosen_status",
            }
        )
    )
    df = df.merge(chosen, on="db_name", how="left")
    return df


def build_db_study_summary(initial_fit_dir: str | Path) -> pd.DataFrame:
    rows = [read_db_study_summary(path) for path in sorted(Path(initial_fit_dir).glob("*.db"))]
    df = pd.DataFrame(rows)
    return df.sort_values(["condition", "current_na"]).reset_index(drop=True)


def run_step00_provenance(project_root: str | Path, relative_tolerance: float = 1e-2, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    paths = project_paths(project_root)
    outputs_dir = Path(output_dir) if output_dir is not None else paths["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    db_summary = build_db_study_summary(paths["initial_fit_dir"])
    trace_summary = build_trace_source_summary(paths["initial_fit_dir"])
    provenance = audit_db_trace_provenance(paths["initial_fit_dir"], relative_tolerance=relative_tolerance)
    atf_inventory = inventory_atf_files(paths["atf_dir"])
    atf_counts = atf_region_condition_counts(atf_inventory)

    db_summary.to_csv(outputs_dir / "db_study_summary.csv", index=False)
    trace_summary.to_csv(outputs_dir / "trace_source_summary.csv", index=False)
    provenance.to_csv(outputs_dir / "control_trace_verification.csv", index=False)
    atf_inventory.to_csv(outputs_dir / "atf_region_condition_inventory.csv", index=False)
    atf_counts.to_csv(outputs_dir / "atf_region_condition_counts.csv", index=False)

    return {
        "db_study_summary": db_summary,
        "trace_source_summary": trace_summary,
        "control_trace_verification": provenance,
        "atf_region_condition_inventory": atf_inventory,
        "atf_region_condition_counts": atf_counts,
    }
