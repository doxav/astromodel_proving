from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .atf_io import CellProtocol, SweepTrace

FEATURE_COLUMNS = [
    "peak_depolarization_mV",
    "stim_end_depolarization_mV",
    "rise_slope_mV_per_s",
    "rise_tau_s",
    "plateau_slope_mV_per_s",
    "decay_slope_mV_per_s",
    "decay_tau_s",
    "undershoot_magnitude_mV",
    "return_slope_mV_per_s",
]


def infer_stim_window(time_s: np.ndarray, current_nA: np.ndarray) -> tuple[float, float]:
    baseline_mask = time_s < min(5.0, float(time_s.max()) * 0.2)
    baseline = float(np.median(current_nA[baseline_mask]))
    noise = float(np.std(current_nA[baseline_mask]))
    threshold = max(0.2, 5.0 * noise)
    active = np.abs(current_nA - baseline) > threshold
    if not np.any(active):
        # fall back to condition-level defaults seen in the dataset
        if time_s.max() > 60:
            return 11.15, 31.12
        return float(time_s[len(time_s)//3]), float(time_s[len(time_s)//3*2])
    idx = np.flatnonzero(active)
    return float(time_s[idx[0]]), float(time_s[idx[-1]])


def _first_crossing(time_s: np.ndarray, signal: np.ndarray, level: float) -> float | None:
    idx = np.flatnonzero(signal >= level)
    if len(idx) == 0:
        return None
    return float(time_s[idx[0]])


def _last_window_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    if len(time_s) < 2:
        return float("nan")
    coeff = np.polyfit(time_s, values, 1)
    return float(coeff[0])


def extract_features_from_trace(time_s: np.ndarray, vm_mV: np.ndarray, current_nA: np.ndarray | None = None) -> dict[str, float | bool]:
    time_s = np.asarray(time_s, dtype=float)
    vm_mV = np.asarray(vm_mV, dtype=float)
    if current_nA is None:
        current_nA = np.zeros_like(time_s)
    current_nA = np.asarray(current_nA, dtype=float)
    stim_onset_s, stim_offset_s = infer_stim_window(time_s, current_nA)

    baseline_mask = (time_s >= max(0.0, stim_onset_s - 5.0)) & (time_s <= max(0.0, stim_onset_s - 1.0))
    if not np.any(baseline_mask):
        baseline_mask = time_s < stim_onset_s
    baseline = float(np.mean(vm_mV[baseline_mask]))

    stim_mask = (time_s >= stim_onset_s) & (time_s <= stim_offset_s)
    post_mask = time_s > stim_offset_s
    if not np.any(stim_mask):
        stim_mask = slice(None)
    stim_time = time_s[stim_mask]
    stim_vm = vm_mV[stim_mask]

    peak_idx = int(np.argmax(stim_vm))
    peak_t = float(stim_time[peak_idx])
    peak_mV = float(stim_vm[peak_idx])
    peak_depol = peak_mV - baseline

    stim_end_mask = (time_s >= stim_offset_s - 1.0) & (time_s <= stim_offset_s)
    if not np.any(stim_end_mask):
        stim_end_mask = stim_mask
    stim_end_level = float(np.mean(vm_mV[stim_end_mask]))
    stim_end_depol = stim_end_level - baseline

    plateau_window = (time_s >= max(stim_onset_s, stim_offset_s - 3.0)) & (time_s <= stim_offset_s)
    plateau_reached = bool(stim_end_depol >= 0.6 * max(peak_depol, 1e-9))
    plateau_slope = _last_window_slope(time_s[plateau_window], vm_mV[plateau_window]) if np.any(plateau_window) else float("nan")

    y = stim_vm - baseline
    t20 = _first_crossing(stim_time, y, 0.2 * peak_depol)
    t63 = _first_crossing(stim_time, y, 0.63 * peak_depol)
    t80 = _first_crossing(stim_time, y, 0.8 * peak_depol)
    if t20 is not None and t80 is not None and t80 > t20:
        rise_slope = (0.8 * peak_depol - 0.2 * peak_depol) / (t80 - t20)
    else:
        rise_slope = float("nan")
    rise_tau = (t63 - stim_onset_s) if t63 is not None else float("nan")

    if np.any(post_mask):
        post_time = time_s[post_mask]
        post_vm = vm_mV[post_mask]
        target_80 = baseline + 0.8 * (stim_end_level - baseline)
        target_63 = baseline + 0.63 * (stim_end_level - baseline)
        target_20 = baseline + 0.2 * (stim_end_level - baseline)
        t80_decay = _first_crossing(post_time, target_80 - post_vm, 0.0)
        t63_decay = _first_crossing(post_time, target_63 - post_vm, 0.0)
        t20_decay = _first_crossing(post_time, target_20 - post_vm, 0.0)
        decay_window = post_time <= min(post_time[0] + 10.0, post_time[-1])
        decay_slope = _last_window_slope(post_time[decay_window], post_vm[decay_window]) * -1.0 if np.any(decay_window) else float("nan")
        decay_tau = (t63_decay - stim_offset_s) if t63_decay is not None else float("nan")

        undershoot_min = float(np.min(post_vm))
        undershoot_mag = max(0.0, baseline - undershoot_min)
        has_undershoot = bool(undershoot_mag > 1e-6)
        final_window = time_s >= max(time_s.max() - 5.0, stim_offset_s)
        return_slope = _last_window_slope(time_s[final_window], vm_mV[final_window]) if np.any(final_window) else float("nan")
    else:
        decay_slope = float("nan")
        decay_tau = float("nan")
        undershoot_mag = float("nan")
        has_undershoot = False
        return_slope = float("nan")

    return {
        "stim_onset_s": stim_onset_s,
        "stim_offset_s": stim_offset_s,
        "baseline_mV": baseline,
        "peak_t_s": peak_t,
        "peak_depolarization_mV": peak_depol,
        "stim_end_depolarization_mV": stim_end_depol,
        "plateau_reached": plateau_reached,
        "plateau_slope_mV_per_s": plateau_slope,
        "rise_slope_mV_per_s": rise_slope,
        "rise_tau_s": rise_tau,
        "has_undershoot": has_undershoot,
        "undershoot_magnitude_mV": undershoot_mag,
        "decay_slope_mV_per_s": decay_slope,
        "decay_tau_s": decay_tau,
        "return_slope_mV_per_s": return_slope,
    }


def build_feature_table(cells: list[CellProtocol]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell in cells:
        for sweep in cell.sweeps:
            feat = extract_features_from_trace(sweep.time_s, sweep.vm_mV, sweep.current_nA)
            rows.append(
                {
                    "file_id": cell.file_id,
                    "file": cell.file_name,
                    "region": cell.region,
                    "condition": cell.condition,
                    "sweep": sweep.sweep,
                    **feat,
                }
            )
    return pd.DataFrame(rows)


def compute_feature_reliability(feature_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = FEATURE_COLUMNS
    for (region, condition), sub in feature_table.groupby(["region", "condition"]):
        n_rows = len(sub)
        n_cells = sub["file_id"].nunique()
        corr = sub[feature_cols].corr(method="spearman").abs().fillna(0.0)
        for feature in feature_cols:
            n_non_missing = int(sub[feature].notna().sum())
            missing_rate = 1.0 - (n_non_missing / max(n_rows, 1))
            redundancy = float(corr.loc[feature].drop(labels=[feature], errors="ignore").max())
            redundant = feature in {"peak_depolarization_mV", "stim_end_depolarization_mV"}
            small_stratum = bool(n_cells < 5)
            weight = (1.0 - missing_rate)
            if redundant:
                weight *= 0.5
            if small_stratum:
                weight *= 0.8
            rows.append(
                {
                    "region": region,
                    "condition": condition,
                    "feature": feature,
                    "n_rows": n_rows,
                    "n_cells": n_cells,
                    "n_non_missing": n_non_missing,
                    "missing_rate": missing_rate,
                    "completeness": 1.0 - missing_rate,
                    "max_abs_spearman": redundancy,
                    "redundant_flag": redundant,
                    "small_stratum": small_stratum,
                    "reliability_weight": weight,
                }
            )
    return pd.DataFrame(rows)


def build_threshold_table(feature_table: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, region, sweep), sub in feature_table.groupby(["condition", "region", "sweep"]):
        for feature in FEATURE_COLUMNS:
            s = sub[feature].dropna()
            if len(s) == 0:
                continue
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            rel = reliability[(reliability["condition"] == condition) & (reliability["region"] == region) & (reliability["feature"] == feature)]
            weight = float(rel["reliability_weight"].iloc[0]) if not rel.empty else 1.0
            rows.append(
                {
                    "condition": condition,
                    "region": region,
                    "sweep": int(sweep),
                    "feature": feature,
                    "n_total_rows": int(len(sub)),
                    "n_non_missing": int(s.notna().sum()),
                    "missing_rate": 1.0 - (len(s) / max(len(sub), 1)),
                    "median": float(s.median()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "acceptable_lower": q1 - 1.5 * iqr,
                    "acceptable_upper": q3 + 1.5 * iqr,
                    "threshold_scope": "region_specific",
                    "reliability_weight": weight,
                }
            )
    return pd.DataFrame(rows)
