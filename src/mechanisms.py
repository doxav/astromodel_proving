from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _window_mask(t_ms: Sequence[float], window_s: tuple[float, float] | None = None) -> np.ndarray:
    t_s = np.asarray(t_ms, dtype=float) / 1000.0
    if window_s is None:
        return np.isfinite(t_s)
    start_s, end_s = window_s
    return (t_s >= float(start_s)) & (t_s <= float(end_s))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def compute_flux_summary(sim: Mapping[str, Any], stim_window_s: tuple[float, float] | None = None) -> dict[str, Any]:
    t_ms = np.asarray(sim["t_ms"], dtype=float)
    mask = _window_mask(t_ms, stim_window_s)
    t_s = t_ms[mask] / 1000.0

    def _get_current(name: str) -> np.ndarray:
        return np.asarray(sim["currents"][name], dtype=float)[mask]

    i_kir = np.abs(_get_current("I_Kir"))
    i_gap = np.abs(_get_current("I_kgap"))
    i_leak = np.abs(_get_current("I_leak"))
    i_k_a = np.abs(_get_current("I_k_a")) if "I_k_a" in sim["currents"] else np.zeros_like(i_kir)

    def _integral(values: np.ndarray) -> float:
        if len(values) < 2:
            return float(np.nansum(values))
        return float(np.trapezoid(values, t_s))

    kir_integral = _integral(i_kir)
    gap_integral = _integral(i_gap)
    leak_integral = _integral(i_leak)
    k_a_integral = _integral(i_k_a)
    total = max(kir_integral + gap_integral + leak_integral, 1e-12)
    gap_fraction = gap_integral / total
    kir_fraction = kir_integral / total
    leak_fraction = leak_integral / total
    ratio = gap_integral / max(kir_integral, 1e-12)

    if ratio >= 2.0:
        gap_kir_classification = "gap_dominant"
    elif ratio <= 0.5:
        gap_kir_classification = "kir_dominant"
    else:
        gap_kir_classification = "mixed"

    if gap_fraction >= 0.6:
        dominant = "Gap"
    elif kir_fraction >= 0.6:
        dominant = "Kir"
    elif leak_fraction >= 0.6:
        dominant = "Leak"
    else:
        dominant = "Mixed"

    k_o = np.asarray(sim["derived"]["K_o"], dtype=float)
    full_t_s = t_ms / 1000.0
    baseline_end = stim_window_s[0] if stim_window_s else min(5.0, float(full_t_s[-1]))
    final_start = stim_window_s[1] if stim_window_s else max(0.0, float(full_t_s[-1]) - 5.0)
    baseline_mask = full_t_s < baseline_end
    final_mask = full_t_s > final_start
    k_o_baseline = float(np.nanmedian(k_o[baseline_mask])) if baseline_mask.any() else float(np.nanmedian(k_o))
    k_o_peak = float(np.nanmax(k_o))
    k_o_final = float(np.nanmedian(k_o[final_mask])) if final_mask.any() else float(k_o[-1])
    k_o_delta_peak = float(k_o_peak - k_o_baseline)
    k_o_recovery_error = float(abs(k_o_final - k_o_baseline))

    return {
        "I_Kir_integral": kir_integral,
        "I_kgap_integral": gap_integral,
        "I_leak_integral": leak_integral,
        "I_k_a_integral": k_a_integral,
        "I_Kir_peak_abs": float(np.nanmax(i_kir)) if i_kir.size else np.nan,
        "I_kgap_peak_abs": float(np.nanmax(i_gap)) if i_gap.size else np.nan,
        "gap_to_kir_integral_ratio": ratio,
        "gap_kir_classification": gap_kir_classification,
        "gap_fraction": gap_fraction,
        "kir_fraction": kir_fraction,
        "leak_fraction": leak_fraction,
        "K_o_baseline": k_o_baseline,
        "K_o_peak": k_o_peak,
        "K_o_final": k_o_final,
        "K_o_delta_peak": k_o_delta_peak,
        "K_o_recovery_error": k_o_recovery_error,
        "dominant_mechanism": dominant,
    }


def compute_proxy_validity(sim: Mapping[str, Any], window_s: tuple[float, float] | None = None) -> dict[str, Any]:
    t_ms = np.asarray(sim["t_ms"], dtype=float)
    mask = _window_mask(t_ms, window_s)
    x = np.asarray(sim["derived"]["DK_a"], dtype=float)[mask]
    y = np.asarray(sim["derived"]["K_o"], dtype=float)[mask]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return {
            "proxy": "DK_a_t",
            "target": "K_o",
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "rmse_after_scaling": np.nan,
            "lag_s_at_max_corr": np.nan,
            "validity_class": "failed",
        }
    x = x[valid]
    y = y[valid]
    x_center = x - np.nanmean(x)
    y_center = y - np.nanmean(y)
    denom = np.nanstd(x_center)
    if not np.isfinite(denom) or denom == 0:
        scaled = np.repeat(np.nanmean(y), len(y))
    else:
        beta = np.nanstd(y_center) / denom
        scaled = np.nanmean(y) + beta * x_center
    rmse = float(np.sqrt(np.nanmean((scaled - y) ** 2)))
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
    spearman = _safe_spearman(x, y)

    if np.isfinite(pearson) and np.isfinite(spearman) and pearson >= 0.9 and spearman >= 0.9:
        validity = "strong"
    elif np.isfinite(pearson) and pearson >= 0.3:
        validity = "weak"
    else:
        validity = "failed"

    return {
        "proxy": "DK_a_t",
        "target": "K_o",
        "pearson_r": pearson,
        "spearman_r": spearman,
        "rmse_after_scaling": rmse,
        "lag_s_at_max_corr": 0.0,
        "validity_class": validity,
    }
