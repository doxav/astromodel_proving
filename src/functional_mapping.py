"""Shared function-mapping helpers for K_o efficiency and sigmoid states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


EPS = 1e-12
EF_UNDEFINED_STATUS = "undefined_flat_or_missing"
EF_DEFINED_STATUS = "defined"
EF_THRESHOLD_INTERPRETATION = (
    "descriptive_legacy_baseline_median_no_experimental_observable"
)


@dataclass(frozen=True)
class KineticFeatureConfig:
    """Configuration for hidden K_o kinetic feature extraction."""

    baseline_window_s: float = 5.0
    final_window_s: float = 5.0
    min_points_per_segment: int = 3
    near_zero_rate: float = 1e-12


def _finite_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least three points")
    return arr


def _interp_crossing_time(
    t_s: np.ndarray,
    y: np.ndarray,
    threshold: float,
    start_idx: int,
    end_idx: int,
    direction: str,
) -> float:
    """Return the first threshold crossing time in a segment."""

    if end_idx <= start_idx:
        return float("nan")
    seg = y[start_idx : end_idx + 1]
    times = t_s[start_idx : end_idx + 1]
    if direction == "up":
        hits = np.where(seg >= threshold)[0]
    elif direction == "down":
        hits = np.where(seg <= threshold)[0]
    else:
        raise ValueError("direction must be 'up' or 'down'")
    if len(hits) == 0:
        return float("nan")
    hit = int(hits[0])
    if hit == 0:
        return float(times[hit])
    y0 = float(seg[hit - 1])
    y1 = float(seg[hit])
    t0 = float(times[hit - 1])
    t1 = float(times[hit])
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < EPS:
        return t1
    frac = (float(threshold) - y0) / (y1 - y0)
    return float(t0 + np.clip(frac, 0.0, 1.0) * (t1 - t0))


def _window_median(
    t_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float
) -> float:
    mask = (t_s >= float(start_s)) & (t_s <= float(end_s))
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.nanmedian(values[mask]))


def extract_ko_kinetic_features(
    t_s: Any,
    ko_mM: Any,
    *,
    onset_s: float,
    offset_s: float,
    config: KineticFeatureConfig | None = None,
) -> dict[str, float | str]:
    """Extract K_o rise/decay features and continuous EF score.

    The EF score is the K_o rise rate divided by the absolute K_o decay rate.
    Class labels are intentionally assigned later from frozen descriptive
    baseline medians, not from biological thresholds.
    """

    cfg = config or KineticFeatureConfig()
    t = _finite_array(t_s, "t_s")
    ko = _finite_array(ko_mM, "ko_mM")
    if t.shape != ko.shape:
        raise ValueError("t_s and ko_mM must have the same length")
    if not np.all(np.isfinite(t)):
        raise ValueError("t_s must contain finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_s must be strictly increasing")
    onset = float(onset_s)
    offset = float(offset_s)
    if not (np.isfinite(onset) and np.isfinite(offset) and offset > onset):
        raise ValueError("onset_s and offset_s must be finite with offset_s > onset_s")

    baseline = _window_median(t, ko, max(float(t[0]), onset - cfg.baseline_window_s), onset)
    final = _window_median(t, ko, max(offset, float(t[-1]) - cfg.final_window_s), float(t[-1]))
    stim_mask = (t >= onset) & (t <= offset)
    post_mask = t >= offset
    if int(stim_mask.sum()) < cfg.min_points_per_segment:
        raise ValueError("stimulation segment contains too few points")
    if int(post_mask.sum()) < cfg.min_points_per_segment:
        raise ValueError("post-stimulation segment contains too few points")

    peak_idx_candidates = np.where(stim_mask)[0]
    peak_idx = int(peak_idx_candidates[np.nanargmax(ko[stim_mask])])
    peak = float(ko[peak_idx])
    ko_peak_delta = peak - baseline if np.isfinite(baseline) else float("nan")

    rise_rate = float("nan")
    t20 = float("nan")
    t80 = float("nan")
    if np.isfinite(ko_peak_delta) and ko_peak_delta > cfg.near_zero_rate:
        y20 = baseline + 0.2 * ko_peak_delta
        y80 = baseline + 0.8 * ko_peak_delta
        onset_idx = int(np.searchsorted(t, onset, side="left"))
        t20 = _interp_crossing_time(t, ko, y20, onset_idx, peak_idx, "up")
        t80 = _interp_crossing_time(t, ko, y80, onset_idx, peak_idx, "up")
        if np.isfinite(t20) and np.isfinite(t80) and t80 > t20:
            rise_rate = float(0.6 * ko_peak_delta / (t80 - t20))

    decay_rate = float("nan")
    td80 = float("nan")
    td20 = float("nan")
    if np.isfinite(final) and np.isfinite(peak) and peak - final > cfg.near_zero_rate:
        decay_drop = peak - final
        y80d = peak - 0.2 * decay_drop
        y20d = peak - 0.8 * decay_drop
        final_idx = len(t) - 1
        td80 = _interp_crossing_time(t, ko, y80d, peak_idx, final_idx, "down")
        td20 = _interp_crossing_time(t, ko, y20d, peak_idx, final_idx, "down")
        if np.isfinite(td80) and np.isfinite(td20) and td20 > td80:
            decay_rate = float(0.6 * decay_drop / (td20 - td80))

    status = EF_DEFINED_STATUS
    score = float("nan")
    ratio = float("nan")
    if not (
        np.isfinite(rise_rate)
        and np.isfinite(decay_rate)
        and rise_rate > cfg.near_zero_rate
        and decay_rate > cfg.near_zero_rate
    ):
        status = EF_UNDEFINED_STATUS
    else:
        ratio = float(rise_rate / decay_rate)
        score = ratio

    return {
        "Ko_baseline_mM": float(baseline),
        "Ko_peak_mM": float(peak),
        "Ko_final_mM": float(final),
        "Ko_peak_delta_mM": float(ko_peak_delta),
        "Ko_recovery_error_mM": float(final - baseline)
        if np.isfinite(final) and np.isfinite(baseline)
        else float("nan"),
        "Ko_rise_t20_s": float(t20),
        "Ko_rise_t80_s": float(t80),
        "Ko_decay_t80_s": float(td80),
        "Ko_decay_t20_s": float(td20),
        "Ko_rise_rate_mM_per_s": float(rise_rate),
        "Ko_decay_rate_abs_mM_per_s": float(decay_rate),
        "Ko_rise_over_decay_rate": float(ratio),
        "Ko_efficiency_score": float(score),
        "Ko_efficiency_status": status,
    }


def classify_efficiency_quadrant(
    rise_rate: float,
    decay_rate: float,
    *,
    rise_cutoff: float,
    decay_cutoff: float,
) -> dict[str, str]:
    """Classify K_o kinetics into descriptive fast/slow rise/decay quadrants."""

    values = [rise_rate, decay_rate, rise_cutoff, decay_cutoff]
    if not all(np.isfinite(float(v)) for v in values):
        return {
            "Ko_rise_speed_class": "undefined",
            "Ko_decay_speed_class": "undefined",
            "Ko_efficiency_quadrant": "undefined",
        }
    rise_class = "fast_rise" if float(rise_rate) >= float(rise_cutoff) else "slow_rise"
    decay_class = "fast_decay" if float(decay_rate) >= float(decay_cutoff) else "slow_decay"
    return {
        "Ko_rise_speed_class": rise_class,
        "Ko_decay_speed_class": decay_class,
        "Ko_efficiency_quadrant": f"{rise_class}_{decay_class}",
    }


def compute_efficiency_threshold_table(
    baseline_features: pd.DataFrame,
    *,
    min_rows_per_stratum: int = 10,
) -> pd.DataFrame:
    """Compute frozen descriptive median cutoffs for EF quadrants."""

    required = {"source_scope", "current_na", "Ko_rise_rate_mM_per_s", "Ko_decay_rate_abs_mM_per_s"}
    missing = sorted(required - set(baseline_features.columns))
    if missing:
        raise ValueError(f"baseline_features is missing required columns: {missing}")
    df = baseline_features.copy()
    df["Ko_rise_rate_mM_per_s"] = pd.to_numeric(df["Ko_rise_rate_mM_per_s"], errors="coerce")
    df["Ko_decay_rate_abs_mM_per_s"] = pd.to_numeric(df["Ko_decay_rate_abs_mM_per_s"], errors="coerce")
    valid = df.dropna(subset=["Ko_rise_rate_mM_per_s", "Ko_decay_rate_abs_mM_per_s"])
    global_rise = float(valid["Ko_rise_rate_mM_per_s"].median()) if not valid.empty else float("nan")
    global_decay = float(valid["Ko_decay_rate_abs_mM_per_s"].median()) if not valid.empty else float("nan")
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["source_scope", "current_na"], dropna=False):
        source_scope, current_na = keys
        local = group.dropna(subset=["Ko_rise_rate_mM_per_s", "Ko_decay_rate_abs_mM_per_s"])
        if len(local) >= int(min_rows_per_stratum):
            rise_cutoff = float(local["Ko_rise_rate_mM_per_s"].median())
            decay_cutoff = float(local["Ko_decay_rate_abs_mM_per_s"].median())
            threshold_source = "source_scope_current_na_legacy_median"
        else:
            rise_cutoff = global_rise
            decay_cutoff = global_decay
            threshold_source = "global_legacy_median_fallback"
        rows.append(
            {
                "source_scope": source_scope,
                "current_na": current_na,
                "n_rows_for_threshold": int(len(local)),
                "Ko_rise_fast_slow_cutoff": rise_cutoff,
                "Ko_decay_fast_slow_cutoff": decay_cutoff,
                "Ko_efficiency_threshold_source": threshold_source,
                "Ko_efficiency_threshold_interpretation": EF_THRESHOLD_INTERPRETATION,
            }
        )
    return pd.DataFrame(rows)


def apply_efficiency_quadrants(
    features: pd.DataFrame, threshold_table: pd.DataFrame
) -> pd.DataFrame:
    """Attach descriptive EF quadrant labels using frozen baseline cutoffs."""

    if features.empty:
        return features.copy()
    out = features.copy()
    merged = out.merge(
        threshold_table,
        on=["source_scope", "current_na"],
        how="left",
        validate="many_to_one",
    )
    rows = [
        classify_efficiency_quadrant(
            row.Ko_rise_rate_mM_per_s,
            row.Ko_decay_rate_abs_mM_per_s,
            rise_cutoff=row.Ko_rise_fast_slow_cutoff,
            decay_cutoff=row.Ko_decay_fast_slow_cutoff,
        )
        for row in merged.itertuples(index=False)
    ]
    labels = pd.DataFrame(rows)
    return pd.concat([merged.reset_index(drop=True), labels], axis=1)


def state_10_90(value: float, low: float = 0.10, high: float = 0.90) -> str:
    """Classify a bounded activation value into low, partial, or open state."""

    try:
        x = float(value)
    except (TypeError, ValueError):
        return "undefined"
    if not np.isfinite(x):
        return "undefined"
    if x <= low:
        return "closed_low"
    if x >= high:
        return "open_high"
    return "partial_mid"


def _gate_from_simulation(sim: Mapping[str, Any]) -> np.ndarray:
    currents = sim.get("currents", {})
    derived = sim.get("derived", {})
    th_s = np.asarray(currents.get("Th_s"), dtype=float)
    dk_a = np.asarray(derived.get("DK_a"), dtype=float)
    if th_s.ndim != 1 or dk_a.ndim != 1 or th_s.shape != dk_a.shape:
        raise ValueError("simulation must contain one-dimensional Th_s and DK_a arrays")
    return np.clip(np.abs(th_s) / np.maximum(np.abs(dk_a), EPS), 0.0, 1.0)


def extract_sigmoid_state_features(
    sim: Mapping[str, Any],
    *,
    stim_window_s: tuple[float, float],
) -> dict[str, float | str]:
    """Extract sigmoid/gating state features from a hidden-output simulation."""

    t_s = _finite_array(np.asarray(sim["t_ms"], dtype=float) / 1000.0, "t_s")
    gate = _gate_from_simulation(sim)
    if gate.shape != t_s.shape:
        raise ValueError("gate and time arrays must have the same length")
    start_s, end_s = float(stim_window_s[0]), float(stim_window_s[1])
    stim_mask = (t_s >= start_s) & (t_s <= end_s)
    if int(stim_mask.sum()) == 0:
        stim_idx = int(np.searchsorted(t_s, end_s, side="left"))
        stim_idx = max(0, min(stim_idx, len(t_s) - 1))
    else:
        stim_idx = int(np.where(stim_mask)[0][-1])
    stim_peak = float(np.nanmax(gate[stim_mask])) if int(stim_mask.sum()) else float("nan")
    stim_end_value = float(gate[stim_idx])
    sim_end_value = float(gate[-1])
    stim_end_state = state_10_90(stim_end_value)
    sim_end_state = state_10_90(sim_end_value)
    stim_peak_state = state_10_90(stim_peak)
    if stim_peak_state in {"open_high", "partial_mid"} and sim_end_state == "closed_low":
        temporal_class = "recruited_during_load_then_closed_by_end"
    elif stim_end_state == "closed_low" and sim_end_state in {"partial_mid", "open_high"}:
        temporal_class = "delayed_ionic_recruitment_after_load"
    elif stim_end_state == "open_high" and sim_end_state == "open_high":
        temporal_class = "early_sustained_open_recruitment"
    elif sim_end_state == "open_high":
        temporal_class = "fully_open_at_sim_end"
    elif sim_end_state == "closed_low":
        temporal_class = "fully_closed_at_sim_end"
    elif stim_end_state == "closed_low" and sim_end_state == "closed_low":
        temporal_class = "persistently_low_range_closed"
    elif "undefined" in {stim_end_state, sim_end_state}:
        temporal_class = "undefined"
    else:
        temporal_class = "intermediate_or_mixed_temporal_recruitment"
    return {
        "sigmoid_value_at_stim_end": stim_end_value,
        "sigmoid_value_at_sim_end": sim_end_value,
        "sigmoid_peak_during_stim": stim_peak,
        "sigmoid_state_at_stim_end_10_90": stim_end_state,
        "sigmoid_state_at_sim_end_10_90": sim_end_state,
        "sigmoid_peak_state_during_stim_10_90": stim_peak_state,
        "temporal_recruitment_class": temporal_class,
    }


def classify_sigmoid_state_change(
    baseline_stim_state: str,
    baseline_end_state: str,
    perturbed_stim_state: str,
    perturbed_end_state: str,
    perturbed_temporal_class: str = "",
) -> str:
    """Return a compact label for baseline-to-perturbed sigmoid state change."""

    if (
        "undefined" in {str(baseline_stim_state), str(baseline_end_state), str(perturbed_stim_state), str(perturbed_end_state)}
        or not str(perturbed_end_state)
    ):
        return "undefined_or_failed"
    if str(perturbed_temporal_class) == "recruited_during_load_then_closed_by_end":
        return "opened_during_stim_then_closed_by_end"
    if str(perturbed_temporal_class) == "delayed_ionic_recruitment_after_load":
        return "delayed_opening_after_load"
    compact = {
        "closed_low": "closed",
        "partial_mid": "partial",
        "open_high": "open",
    }
    before = compact.get(str(baseline_end_state), "undefined")
    after = compact.get(str(perturbed_end_state), "undefined")
    if "undefined" in {before, after}:
        return "undefined_or_failed"
    if before == after:
        return f"unchanged_{before}"
    return f"{before}_to_{after}"


def direction_label(delta: float, *, tolerance: float = 1e-9) -> str:
    """Classify a signed numeric delta as increase/decrease/no-change."""

    try:
        x = float(delta)
    except (TypeError, ValueError):
        return "undefined"
    if not np.isfinite(x):
        return "undefined"
    tol = abs(float(tolerance))
    if x > tol:
        return "increase"
    if x < -tol:
        return "decrease"
    return "no_change"


def compare_direction_to_target(simulated: str, experimental: str) -> str:
    """Compare simulated and experimental direction labels."""

    sim = str(simulated)
    exp = str(experimental)
    if exp in {"undefined", "nan", ""} or sim in {"undefined", "nan", ""}:
        return "undefined"
    if exp in {"no_clear_change", "no_change"}:
        return "no_clear_experimental_change"
    if sim in {"no_clear_change", "no_change"}:
        return "simulation_no_change"
    return "match" if sim == exp else "opposite"
