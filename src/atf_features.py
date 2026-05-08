"""ATF parsing, preprocessing, and feature extraction for step 02.

The implementation is adapted from the original exploratory notebooks but made
local-file-only, testable, and explicit about region/condition labels.
"""

from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:  # optional acceleration; step 02 keeps numpy as the default engine
    from numba import njit
except Exception:  # pragma: no cover - optional dependency path
    njit = None

EXPECTED_ATF_COUNTS: dict[tuple[str, str], int] = {
    ("DH", "CONTROL"): 7,
    ("VH", "CONTROL"): 4,
    ("DH", "MFA"): 6,
    ("VH", "MFA"): 7,
    ("DH", "MFA_BA"): 6,
    ("VH", "MFA_BA"): 7,
}

MANUAL_TRANSIENT_ARTIFACT_HINTS: dict[str, dict[int, dict[str, Any]]] = {
    "DH_6_MFA_Ba.atf": {5: {"windows_s": [(0.0, 10.0)], "ipatch_abs_thr_mV": 0.8}},
    "VH_5_MFA_Ba.atf": {2: {"windows_s": [(3.0, 8.0)], "ipatch_abs_thr_mV": 0.8}},
    "VH_6_MFA.atf": {1: {"windows_s": [(35.0, 40.0)], "ipatch_abs_thr_mV": 0.6}},
    "VH_6_MFA_Ba.atf": {3: {"windows_s": [(32.0, 35.0)], "ipatch_abs_thr_mV": 0.8}},
}

PRIMARY_CONTINUOUS_FEATURES: list[str] = [
    "peak_depolarization_mV",
    "stim_end_depolarization_mV",
    "rise_slope_mV_per_s",
    "rise_tau_s",
    "decay_slope_mV_per_s",
    "decay_tau_s",
]
CONDITIONAL_CONTINUOUS_FEATURES: list[str] = [
    "plateau_level_mV",
    "plateau_slope_mV_per_s",
    "undershoot_magnitude_mV",
    "return_slope_mV_per_s",
]
BINARY_FEATURES: list[str] = ["plateau_reached", "has_undershoot"]
ALL_FEATURES: list[str] = PRIMARY_CONTINUOUS_FEATURES + CONDITIONAL_CONTINUOUS_FEATURES + BINARY_FEATURES

FEATURE_FAMILY: dict[str, str] = {
    **{name: "primary_continuous" for name in PRIMARY_CONTINUOUS_FEATURES},
    **{name: "conditional_continuous" for name in CONDITIONAL_CONTINUOUS_FEATURES},
    **{name: "binary" for name in BINARY_FEATURES},
}

FEATURE_PRIORITY: dict[str, int] = {
    # lower value => keep more weight when redundant
    "peak_depolarization_mV": 0,
    "stim_end_depolarization_mV": 1,
    "rise_slope_mV_per_s": 0,
    "rise_tau_s": 0,
    "plateau_level_mV": 0,
    "plateau_slope_mV_per_s": 0,
    "decay_slope_mV_per_s": 0,
    "decay_tau_s": 0,
    "undershoot_magnitude_mV": 0,
    "return_slope_mV_per_s": 0,
    "plateau_reached": 0,
    "has_undershoot": 0,
}

ATF_REGION_RE = re.compile(r"(^|_)(DH|VH)(_|$)", re.IGNORECASE)


class AtfParseError(RuntimeError):
    """Raised when an ATF file cannot be parsed consistently."""


@dataclass(frozen=True)
class ExperimentalFactors:
    file_id: str
    region: str
    condition: str
    group_label: str
    cell_label: str
    source_path: str


if njit is not None:  # pragma: no cover - exercised indirectly in benchmark tests
    @njit(cache=True)
    def _moving_average_nan_numba_core(x: np.ndarray, window_pts: int) -> np.ndarray:
        n = max(3, int(window_pts))
        if n % 2 == 0:
            n += 1
        half = n // 2
        out = np.empty_like(x)
        m = x.shape[0]
        for i in range(m):
            start = 0 if i - half < 0 else i - half
            stop = m if i + half + 1 > m else i + half + 1
            acc = 0.0
            cnt = 0
            for j in range(start, stop):
                val = x[j]
                if not math.isnan(val):
                    acc += val
                    cnt += 1
            out[i] = np.nan if cnt == 0 else acc / cnt
        return out
else:
    _moving_average_nan_numba_core = None


def make_unique(columns: Sequence[str]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    out: list[str] = []
    for name in columns:
        counts[name] += 1
        if counts[name] == 1:
            out.append(name)
        else:
            out.append(f"{name}__{counts[name]}")
    return out


def discover_atf_files(atf_root: str | Path) -> list[Path]:
    root = Path(atf_root)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("*.atf"))
    if not files:
        raise FileNotFoundError(f"No .atf files found under {root}")
    return files


def parse_experimental_factors(path_or_name: str | Path) -> ExperimentalFactors:
    path = Path(path_or_name)
    stem = path.stem
    upper = stem.upper()

    if "MFA_BA" in upper or "MFA-BA" in upper or ("MFA" in upper and "BA" in upper):
        condition = "MFA_BA"
    elif "MFA" in upper:
        condition = "MFA"
    else:
        condition = "CONTROL"

    region_match = ATF_REGION_RE.search(upper)
    region = region_match.group(2).upper() if region_match else "UNKNOWN"
    if region == "UNKNOWN" and re.match(r"^T\d", upper):
        region = "DH"

    cell_label = re.sub(r"(?i)_MFA_BA|_MFA-BA|_MFA", "", stem)
    cell_label = re.sub(r"__+", "_", cell_label).strip("_")

    return ExperimentalFactors(
        file_id=stem,
        region=region,
        condition=condition,
        group_label=f"{region}_{condition}",
        cell_label=cell_label,
        source_path=str(path.resolve()),
    )


def parse_atf(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", errors="replace", newline="") as f:
        version = f.readline().rstrip("\n\r")
        line2 = f.readline().rstrip("\n\r")
        try:
            n_header_lines, declared_n_cols = map(int, line2.split("\t"))
        except Exception as exc:  # pragma: no cover - malformed file path
            raise AtfParseError(f"Could not parse ATF header counts in {path.name}: {line2}") from exc
        header_lines = [f.readline().rstrip("\n\r") for _ in range(n_header_lines)]
        columns_line = f.readline().rstrip("\n\r")
        columns = next(csv.reader([columns_line], delimiter="\t", quotechar='"'))
        unique_columns = make_unique(columns)
        data = pd.read_csv(f, sep="\t", header=None, names=unique_columns)

    meta: OrderedDict[str, Any] = OrderedDict()
    for line in header_lines:
        row = next(csv.reader([line], delimiter="\t", quotechar='"'))
        if not row:
            continue
        if len(row) == 1:
            match = re.match(r"([^=]+)=(.*)", row[0])
            if match:
                meta[match.group(1)] = match.group(2)
            else:
                meta[row[0]] = None
        else:
            key = row[0][:-1] if row[0].endswith("=") else row[0]
            meta[key] = row[1:]

    return {
        "path": path,
        "version": version,
        "n_header_lines": n_header_lines,
        "declared_n_cols": declared_n_cols,
        "meta": meta,
        "columns": columns,
        "unique_columns": unique_columns,
        "data": data,
    }


def get_sweep_map(parsed: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    data_cols = [c for c in parsed["unique_columns"] if c != "Time (s)"]
    signals = parsed["meta"].get("Signals", [])
    if isinstance(signals, str):
        signals = [s for s in signals.split(",") if s]

    sweep_map: defaultdict[int, dict[str, str]] = defaultdict(dict)
    if len(signals) == len(data_cols):
        for col_name, signal_name in zip(data_cols, signals):
            match = re.search(r"Trace #(\d+)", col_name)
            sweep_num = int(match.group(1)) if match else len(sweep_map) + 1
            sweep_map[sweep_num][signal_name] = col_name
    else:
        for i, col_name in enumerate(data_cols, start=1):
            match = re.search(r"Trace #(\d+)", col_name)
            sweep_num = int(match.group(1)) if match else i
            sweep_map[sweep_num]["signal"] = col_name
    return {k: sweep_map[k] for k in sorted(sweep_map)}


def moving_average(x: np.ndarray, window_pts: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = max(3, int(window_pts))
    if n % 2 == 0:
        n += 1
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(x, kernel, mode="same")


def moving_average_nan(x: np.ndarray, window_pts: int, engine: str = "numpy") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = max(3, int(window_pts))
    if n % 2 == 0:
        n += 1
    if engine == "numba" and _moving_average_nan_numba_core is not None:
        return _moving_average_nan_numba_core(x, n)
    valid = np.isfinite(x).astype(float)
    x0 = np.nan_to_num(x, nan=0.0)
    num = np.convolve(x0, np.ones(n), mode="same")
    den = np.convolve(valid, np.ones(n), mode="same")
    out = num / np.where(den == 0, np.nan, den)
    return out


def average_signal(data: pd.DataFrame, sweep_map: Mapping[int, Mapping[str, str]], signal_name: str) -> Optional[np.ndarray]:
    cols = [cols[signal_name] for cols in sweep_map.values() if signal_name in cols]
    if not cols:
        return None
    return data[cols].mean(axis=1).to_numpy(dtype=float)


def detect_step_edges(t: np.ndarray, x: np.ndarray, response_only: bool = False) -> tuple[float, float]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    x_smooth = moving_average(x, max(5, int(round(0.05 / dt))))
    dx = np.diff(x_smooth)
    duration = float(t[-1] - t[0])
    if response_only:
        onset_mask = (t[1:] >= t[0] + 0.10 * duration) & (t[1:] <= t[0] + 0.60 * duration)
    else:
        onset_mask = (t[1:] >= t[0] + 0.02 * duration) & (t[1:] <= t[0] + 0.70 * duration)
    onset_candidates = np.where(onset_mask)[0]
    onset_idx = onset_candidates[np.argmax(dx[onset_mask])] if len(onset_candidates) else int(np.argmax(dx))

    min_gap = max(2.0, 0.05 * duration)
    offset_mask = (t[1:] >= t[onset_idx + 1] + min_gap) & (t[1:] <= t[-1] - min_gap)
    offset_candidates = np.where(offset_mask)[0]
    if len(offset_candidates):
        offset_idx = offset_candidates[np.argmin(dx[offset_mask])]
    else:
        rel = int(np.argmin(dx[onset_idx + 1 :]))
        offset_idx = onset_idx + 1 + rel
    return float(t[onset_idx + 1]), float(t[offset_idx + 1])


def pick_step_source(parsed: Mapping[str, Any], sweep_map: Mapping[int, Mapping[str, str]]) -> tuple[str, float, float]:
    data = parsed["data"]
    t = data["Time (s)"].to_numpy(dtype=float)
    for signal_name, response_only in (("IP_curr", False), ("IP_volt", False), ("Ipatch_R", True)):
        x = average_signal(data, sweep_map, signal_name)
        if x is not None:
            onset_s, offset_s = detect_step_edges(t, x, response_only=response_only)
            return signal_name, onset_s, offset_s
    first_sweep = next(iter(sweep_map.values()))
    first_signal_name = next(iter(first_sweep.keys()))
    x = average_signal(data, sweep_map, first_signal_name)
    if x is None:
        raise AtfParseError(f"Could not infer any sweep signal in {parsed['path']}")
    onset_s, offset_s = detect_step_edges(t, x, response_only=True)
    return first_signal_name, onset_s, offset_s


def merge_intervals(intervals: Sequence[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [list(sorted(intervals)[0])]
    for a, b in sorted(intervals)[1:]:
        if a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged]


def time_intervals_to_index_intervals(t: np.ndarray, intervals_s: Sequence[tuple[float, float]], pad_s: float = 0.0) -> list[tuple[int, int]]:
    t = np.asarray(t, dtype=float)
    out: list[tuple[int, int]] = []
    for a_s, b_s in intervals_s:
        a_s = max(float(t[0]), float(a_s) - float(pad_s))
        b_s = min(float(t[-1]), float(b_s) + float(pad_s))
        a = max(0, int(np.searchsorted(t, a_s, side="left")))
        b = min(len(t) - 1, int(np.searchsorted(t, b_s, side="right")) - 1)
        if a <= b:
            out.append((a, b))
    return merge_intervals(out, gap=1)


def correct_brief_jump_artifacts(
    v: np.ndarray,
    diff_nmad: float = 12.0,
    abs_jump_mV: float = 0.5,
    max_artifact_pts: int = 8,
    pad_pts: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, list[tuple[int, int, float]]]:
    v = np.asarray(v, dtype=float)
    dv = np.diff(v)
    robust_sigma = 1.4826 * np.median(np.abs(dv - np.median(dv)))
    jump_thr = max(abs_jump_mV, diff_nmad * robust_sigma)
    jump_idx = np.where(np.abs(dv) > jump_thr)[0]
    candidate_intervals = [(max(0, i - pad_pts), min(len(v) - 1, i + 1 + pad_pts)) for i in jump_idx]
    candidate_intervals = merge_intervals(candidate_intervals, gap=1)

    out = v.copy()
    corrected_mask = np.zeros(len(v), dtype=bool)
    corrected_segments: list[tuple[int, int, float]] = []
    for a, b in candidate_intervals:
        seg_len = b - a + 1
        if seg_len > max_artifact_pts or a == 0 or b >= len(v) - 1:
            continue
        left = a - 1
        right = b + 1
        interp = np.interp(np.arange(a, b + 1), [left, right], [out[left], out[right]])
        max_dev = float(np.max(np.abs(out[a : b + 1] - interp)))
        if max_dev >= abs_jump_mV:
            out[a : b + 1] = interp
            corrected_mask[a : b + 1] = True
            corrected_segments.append((a, b, max_dev))
    return out, corrected_mask, float(jump_thr), corrected_segments


def correct_isolated_outliers(
    v: np.ndarray,
    abs_threshold_mV: float = 1.5,
    nmad: float = 12.0,
    max_run_pts: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.asarray(v, dtype=float)
    med = moving_average(v, 5)
    resid = v - med
    robust_sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    thr = max(abs_threshold_mV, nmad * robust_sigma)
    bad = np.abs(resid) > thr
    runs = merge_intervals([(i, i) for i in np.where(bad)[0]], gap=1)
    out = v.copy()
    mask = np.zeros(len(v), dtype=bool)
    for a, b in runs:
        if (b - a + 1) > max_run_pts or a == 0 or b >= len(v) - 1:
            continue
        out[a : b + 1] = np.interp(np.arange(a, b + 1), [a - 1, b + 1], [out[a - 1], out[b + 1]])
        mask[a : b + 1] = True
    return out, mask, float(thr)


def auto_transient_artifact_intervals(
    t: np.ndarray,
    v_ipatch: np.ndarray,
    onset_s: Optional[float] = None,
    offset_s: Optional[float] = None,
    v_aux: Optional[np.ndarray] = None,
    short_s: float = 0.015,
    ref_s: float = 0.75,
    abs_thr_mV: float = 0.8,
    nmad: float = 8.0,
    aux_ref_s: float = 0.4,
    aux_abs_thr: float = 1.0,
    aux_nmad: float = 8.0,
    exclude_boundary_s: float = 0.5,
    onset_guard_s: float = 1.5,
    edge_guard_s: float = 0.5,
    offset_guard_s: float = 0.7,
    merge_gap_s: float = 0.05,
    expand_s: float = 0.05,
    max_window_s: float = 5.0,
    min_window_s: float = 0.02,
) -> list[tuple[int, int]]:
    t = np.asarray(t, dtype=float)
    v = np.asarray(v_ipatch, dtype=float)
    dt = float(np.median(np.diff(t)))
    n = len(v)
    valid = np.ones(n, dtype=bool)
    valid &= ~((t >= t[0]) & (t <= t[0] + exclude_boundary_s))
    valid &= ~((t >= t[-1] - exclude_boundary_s) & (t <= t[-1]))
    if onset_s is not None:
        valid &= ~((t >= onset_s - edge_guard_s) & (t <= onset_s + onset_guard_s))
    if offset_s is not None:
        valid &= ~((t >= offset_s - edge_guard_s) & (t <= offset_s + offset_guard_s))

    x = moving_average(v, max(5, int(round(short_s / dt))))
    ref = moving_average(x, max(11, int(round(ref_s / dt))))
    resid = x - ref
    sigma = 1.4826 * np.median(np.abs(resid[valid] - np.median(resid[valid]))) if valid.any() else 0.0
    thr = max(abs_thr_mV, nmad * sigma)
    cand = valid & (np.abs(resid) >= thr)

    if v_aux is not None:
        xa = moving_average(np.asarray(v_aux, dtype=float), max(5, int(round(short_s / dt))))
        refa = moving_average(xa, max(11, int(round(aux_ref_s / dt))))
        resida = xa - refa
        sigmaa = 1.4826 * np.median(np.abs(resida[valid] - np.median(resida[valid]))) if valid.any() else 0.0
        thra = max(aux_abs_thr, aux_nmad * sigmaa)
        cand |= valid & (np.abs(resida) >= thra)

    gap_pts = max(1, int(round(merge_gap_s / dt)))
    expand_pts = int(round(expand_s / dt))
    max_pts = max(2, int(round(max_window_s / dt)))
    min_pts = max(1, int(round(min_window_s / dt)))

    runs = merge_intervals([(i, i) for i in np.where(cand)[0]], gap=gap_pts)
    refined: list[tuple[int, int]] = []
    for a, b in runs:
        a = max(0, a - expand_pts)
        b = min(n - 1, b + expand_pts)
        seg_len = b - a + 1
        if max_pts >= seg_len >= min_pts:
            refined.append((a, b))
    return merge_intervals(refined, gap=gap_pts)


def interpolate_over_intervals(
    v: np.ndarray,
    intervals: Sequence[tuple[int, int]],
    dt: float,
    anchor_window_s: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    out = v.copy()
    repaired_mask = np.zeros(len(v), dtype=bool)
    anchor_pts = max(3, int(round(anchor_window_s / dt)))
    for a, b in intervals:
        if a <= 0 or b >= len(v) - 1:
            continue
        left0 = max(0, a - anchor_pts)
        left1 = a
        right0 = b + 1
        right1 = min(len(v), b + 1 + anchor_pts)
        if left1 - left0 < 1 or right1 - right0 < 1:
            continue
        y_left = float(np.median(out[left0:left1]))
        y_right = float(np.median(out[right0:right1]))
        out[a : b + 1] = np.interp(np.arange(a, b + 1), [a - 1, b + 1], [y_left, y_right])
        repaired_mask[a : b + 1] = True
    return out, repaired_mask


def preprocess_ipatch_trace(
    v: np.ndarray,
    t: np.ndarray,
    dt: float,
    onset_s: Optional[float] = None,
    offset_s: Optional[float] = None,
    ipvolt: Optional[np.ndarray] = None,
    artifact_hint_config: Optional[Mapping[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[tuple[int, int]], dict[str, np.ndarray]]:
    v = np.asarray(v, dtype=float)
    artifact_hint_config = dict(artifact_hint_config or {})
    jump_corrected, jump_mask, jump_thr, jump_segments = correct_brief_jump_artifacts(
        v,
        diff_nmad=12.0,
        abs_jump_mV=2.0,
        max_artifact_pts=max(3, int(round(0.008 / dt))),
        pad_pts=1,
    )
    despiked, iso_mask, iso_thr = correct_isolated_outliers(
        jump_corrected,
        abs_threshold_mV=1.5,
        nmad=12.0,
        max_run_pts=max(1, int(round(0.002 / dt))),
    )
    repaired = despiked.copy()
    auto_intervals_idx = auto_transient_artifact_intervals(
        t=t,
        v_ipatch=despiked,
        onset_s=onset_s,
        offset_s=offset_s,
        v_aux=ipvolt,
        short_s=float(artifact_hint_config.get("auto_short_s", 0.015)),
        ref_s=float(artifact_hint_config.get("auto_ref_s", 0.75)),
        abs_thr_mV=float(artifact_hint_config.get("ipatch_abs_thr_mV", artifact_hint_config.get("auto_abs_thr_mV", 0.8))),
        nmad=float(artifact_hint_config.get("auto_nmad", 8.0)),
        aux_ref_s=float(artifact_hint_config.get("auto_aux_ref_s", 0.4)),
        aux_abs_thr=float(artifact_hint_config.get("auto_aux_abs_thr", 1.0)),
        aux_nmad=float(artifact_hint_config.get("auto_aux_nmad", 8.0)),
        exclude_boundary_s=float(artifact_hint_config.get("auto_exclude_boundary_s", 0.5)),
        onset_guard_s=float(artifact_hint_config.get("auto_onset_guard_s", 1.5)),
        edge_guard_s=float(artifact_hint_config.get("auto_edge_guard_s", 0.5)),
        offset_guard_s=float(artifact_hint_config.get("auto_offset_guard_s", 0.7)),
        merge_gap_s=float(artifact_hint_config.get("auto_merge_gap_s", 0.05)),
        expand_s=float(artifact_hint_config.get("auto_expand_s", 0.05)),
        max_window_s=float(artifact_hint_config.get("auto_max_window_s", 5.0)),
        min_window_s=float(artifact_hint_config.get("auto_min_window_s", 0.02)),
    )
    repaired, auto_mask = interpolate_over_intervals(
        repaired,
        auto_intervals_idx,
        dt,
        anchor_window_s=float(artifact_hint_config.get("anchor_window_s", 0.03)),
    )
    manual_intervals_s = list(artifact_hint_config.get("windows_s", []))
    manual_idx = time_intervals_to_index_intervals(t, manual_intervals_s, pad_s=float(artifact_hint_config.get("manual_pad_s", 0.0)))
    if manual_idx:
        repaired, manual_mask = interpolate_over_intervals(
            repaired,
            manual_idx,
            dt,
            anchor_window_s=float(artifact_hint_config.get("anchor_window_s", 0.03)),
        )
    else:
        manual_mask = np.zeros(len(v), dtype=bool)
    corrected_mask = jump_mask | iso_mask | auto_mask | manual_mask
    qc = {
        "n_corrected_points": int(corrected_mask.sum()),
        "fraction_corrected": float(corrected_mask.mean()),
        "jump_threshold_mV": float(jump_thr),
        "isolated_threshold_mV": float(iso_thr),
        "n_jump_segments": int(len(jump_segments)),
        "n_auto_intervals": int(len(auto_intervals_idx)),
        "n_auto_repaired_points": int(auto_mask.sum()),
        "n_manual_intervals": int(len(manual_idx)),
        "n_manual_repaired_points": int(manual_mask.sum()),
    }
    all_intervals_idx = merge_intervals(auto_intervals_idx + manual_idx, gap=1)
    return repaired, corrected_mask, qc, all_intervals_idx, {
        "jump": jump_mask,
        "isolated": iso_mask,
        "auto": auto_mask,
        "manual": manual_mask,
    }


def preprocess_parsed(
    parsed: Mapping[str, Any],
    artifact_hint_map: Optional[Mapping[str, Mapping[int, Mapping[str, Any]]]] = None,
) -> dict[str, Any]:
    out = deepcopy(parsed)
    data = out["data"].copy(deep=True)
    t = data["Time (s)"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))
    sweep_map = get_sweep_map(out)
    _, onset_s, offset_s = pick_step_source(out, sweep_map)
    qc_rows: list[dict[str, Any]] = []
    artifact_masks_by_sweep: dict[int, np.ndarray] = {}
    artifact_component_masks: dict[int, dict[str, np.ndarray]] = {}
    artifact_intervals_s_by_sweep: dict[int, list[tuple[float, float]]] = {}
    artifact_hint_map = artifact_hint_map or {}
    file_hints = artifact_hint_map.get(out["path"].name, {})

    for sweep_num, cols in sweep_map.items():
        if "Ipatch_R" not in cols:
            continue
        v_raw = data[cols["Ipatch_R"]].to_numpy(dtype=float)
        ipvolt = data[cols["IP_volt"]].to_numpy(dtype=float) if "IP_volt" in cols else None
        hint_cfg = file_hints.get(int(sweep_num), {})
        v_clean, corrected_mask, qc, intervals_idx, comp_masks = preprocess_ipatch_trace(
            v_raw,
            t,
            dt,
            onset_s,
            offset_s,
            ipvolt,
            hint_cfg,
        )
        data[cols["Ipatch_R"]] = v_clean
        artifact_masks_by_sweep[int(sweep_num)] = corrected_mask
        artifact_component_masks[int(sweep_num)] = {k: np.asarray(v, dtype=bool) for k, v in comp_masks.items()}
        artifact_intervals_s_by_sweep[int(sweep_num)] = [(float(t[a]), float(t[b])) for a, b in intervals_idx]
        qc_rows.append({
            "file_id": out["path"].stem,
            "file": out["path"].name,
            "sweep": int(sweep_num),
            "step_onset_s": float(onset_s),
            "step_offset_s": float(offset_s),
            **qc,
        })

    out = dict(out)
    out["data"] = data
    out["preprocess_qc"] = pd.DataFrame(qc_rows)
    out["artifact_masks_by_sweep"] = artifact_masks_by_sweep
    out["artifact_component_masks_by_sweep"] = artifact_component_masks
    out["artifact_intervals_s_by_sweep"] = artifact_intervals_s_by_sweep
    return out


def get_sweep_artifact_mask(parsed: Mapping[str, Any], sweep_num: int, n_points: int) -> np.ndarray:
    masks = parsed.get("artifact_masks_by_sweep", {})
    mask = masks.get(int(sweep_num))
    if mask is None:
        return np.zeros(int(n_points), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != int(n_points):
        return np.zeros(int(n_points), dtype=bool)
    return mask.copy()


def robust_median(x: Sequence[float], default: float = np.nan) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(default)
    return float(np.median(values))


def line_slope(t: Sequence[float], y: Sequence[float]) -> float:
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_arr = t_arr[finite]
    y_arr = y_arr[finite]
    if len(t_arr) < 2 or np.allclose(t_arr, t_arr[0]):
        return np.nan
    return float(np.polyfit(t_arr, y_arr, 1)[0])


def slope_over_window(t: Sequence[float], y: Sequence[float]) -> float:
    return line_slope(t, y)


def first_crossing_time_interp(
    t: Sequence[float],
    x: Sequence[float],
    threshold: float,
    start_idx: int,
    end_idx: Optional[int] = None,
    direction: str = "up",
) -> float:
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    if end_idx is None:
        end_idx = len(x_arr) - 1
    if end_idx <= start_idx:
        return np.nan
    xx = np.asarray(x_arr[start_idx : end_idx + 1], dtype=float)
    tt = np.asarray(t_arr[start_idx : end_idx + 1], dtype=float)
    finite = np.isfinite(xx)
    xx = xx[finite]
    tt = tt[finite]
    if len(xx) < 2:
        return np.nan
    cross = np.where(xx >= threshold)[0] if direction == "up" else np.where(xx <= threshold)[0]
    if len(cross) == 0:
        return np.nan
    k = int(cross[0])
    if k == 0:
        return float(tt[0])
    x0, x1 = float(xx[k - 1]), float(xx[k])
    t0, t1 = float(tt[k - 1]), float(tt[k])
    if x1 == x0:
        return float(t1)
    frac = (threshold - x0) / (x1 - x0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(t0 + frac * (t1 - t0))


def closest_sample_on_side(
    t: Sequence[float],
    x: Sequence[float],
    threshold: float,
    start_idx: int,
    end_idx: Optional[int] = None,
    side: str = "right",
) -> tuple[float, float, int]:
    t_arr = np.asarray(t, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if end_idx is None:
        end_idx = len(x_arr) - 1
    if end_idx <= start_idx:
        return np.nan, np.nan, -1
    idx = np.arange(start_idx, end_idx + 1)
    xx = x_arr[start_idx : end_idx + 1]
    tt = t_arr[start_idx : end_idx + 1]
    finite = np.isfinite(xx)
    if not finite.any():
        return np.nan, np.nan, -1
    idx = idx[finite]
    xx = xx[finite]
    tt = tt[finite]
    if side == "right":
        cand = np.where(xx >= threshold)[0]
        if len(cand) == 0:
            return np.nan, np.nan, -1
        k = int(cand[np.argmin(xx[cand] - threshold)])
    else:
        cand = np.where(xx <= threshold)[0]
        if len(cand) == 0:
            return np.nan, np.nan, -1
        k = int(cand[np.argmin(threshold - xx[cand])])
    return float(tt[k]), float(xx[k]), int(idx[k])


def first_crossing_time_interp_with_fallback(
    t: Sequence[float],
    x: Sequence[float],
    threshold: float,
    start_idx: int,
    end_idx: Optional[int] = None,
    direction: str = "up",
) -> float:
    tcross = first_crossing_time_interp(t, x, threshold, start_idx, end_idx, direction)
    if np.isfinite(tcross):
        return float(tcross)
    side = "right" if direction == "up" else "left"
    tfallback, _, _ = closest_sample_on_side(t, x, threshold, start_idx, end_idx, side=side)
    return float(tfallback) if np.isfinite(tfallback) else np.nan


def detect_plateau_window(
    t: np.ndarray,
    v_smooth: np.ndarray,
    baseline: float,
    onset_s: float,
    offset_s: float,
    artifact_mask: np.ndarray,
    edge_guard_s: float = 0.05,
    late_window_s: float = 3.0,
    frac_change_thr: float = 0.02,
    peak_buffer_s: float = 1.0,
) -> dict[str, float | bool]:
    stim_duration = offset_s - onset_s
    late_window_s = min(late_window_s, max(1.0, 0.25 * stim_duration))
    late_mask = (t >= offset_s - late_window_s) & (t <= offset_s - edge_guard_s) & (~artifact_mask)
    late_slope = slope_over_window(t[late_mask], v_smooth[late_mask]) if late_mask.sum() > 5 else np.nan
    stim_mask = (t >= onset_s + edge_guard_s) & (t <= offset_s - edge_guard_s) & (~artifact_mask)
    if stim_mask.sum() < 10 or not np.isfinite(late_slope):
        return {
            "plateau_reached": False,
            "late_slope": np.nan,
            "plateau_level": np.nan,
            "plateau_start": np.nan,
            "plateau_end": np.nan,
        }
    stim_v = v_smooth[stim_mask]
    stim_t = t[stim_mask]
    peak_idx = int(np.nanargmax(stim_v))
    peak_t = float(stim_t[peak_idx])
    end_level = robust_median(v_smooth[(t >= offset_s - 1.0) & (t <= offset_s - edge_guard_s) & (~artifact_mask)])
    dep = max(1e-9, end_level - baseline)
    abs_change = abs(late_slope) * late_window_s
    plateau_reached = (peak_t <= offset_s - peak_buffer_s) and (abs_change <= max(0.2, frac_change_thr * abs(dep)))
    if plateau_reached:
        return {
            "plateau_reached": True,
            "late_slope": late_slope,
            "plateau_level": end_level,
            "plateau_start": float(offset_s - late_window_s),
            "plateau_end": float(offset_s - edge_guard_s),
        }
    return {
        "plateau_reached": False,
        "late_slope": late_slope,
        "plateau_level": np.nan,
        "plateau_start": np.nan,
        "plateau_end": np.nan,
    }


def extract_features(
    parsed: Mapping[str, Any],
    kinetics_smoothing_s: float = 0.025,
    baseline_window_s: float = 5.0,
    baseline_guard_s: float = 1.0,
    edge_guard_s: float = 0.0,
    stim_end_window_s: float = 1.0,
    plateau_late_window_s: float = 3.0,
    peak_buffer_s: float = 1.0,
    plateau_frac_change_thr: float = 0.02,
    post_search_guard_s: float = 0.0,
    final_window_s: float = 0.5,
    smoothing_engine: str = "numpy",
) -> pd.DataFrame:
    data = parsed["data"]
    t = data["Time (s)"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))
    sweep_map = get_sweep_map(parsed)
    exp = parse_experimental_factors(parsed["path"])
    step_source, onset_s, offset_s = pick_step_source(parsed, sweep_map)

    rows: list[dict[str, Any]] = []
    for sweep_num, cols in sweep_map.items():
        if "Ipatch_R" in cols:
            v_clean = data[cols["Ipatch_R"]].to_numpy(dtype=float)
        else:
            v_clean = data[next(iter(cols.values()))].to_numpy(dtype=float)

        artifact_mask = get_sweep_artifact_mask(parsed, sweep_num, len(t))
        v_kin = moving_average_nan(v_clean, max(5, int(round(kinetics_smoothing_s / dt))), engine=smoothing_engine)

        base_mask = (
            (t >= max(t[0], onset_s - baseline_window_s))
            & (t < onset_s - baseline_guard_s)
            & (~artifact_mask)
        )
        baseline = robust_median(v_clean[base_mask])
        if not np.isfinite(baseline):
            continue
        baseline_window_start_s = float(np.nanmin(t[base_mask])) if base_mask.any() else np.nan
        baseline_window_end_s = float(np.nanmax(t[base_mask])) if base_mask.any() else np.nan

        stim_mask = (
            (t >= onset_s + edge_guard_s)
            & (t <= offset_s - edge_guard_s)
            & (~artifact_mask)
        )
        if stim_mask.sum() < 10:
            continue
        stim_t = t[stim_mask]
        stim_v = v_kin[stim_mask]
        if not np.isfinite(np.nanmax(stim_v)):
            continue

        stim_end_mask = (
            (t >= max(onset_s + edge_guard_s, offset_s - stim_end_window_s))
            & (t <= offset_s)
            & (~artifact_mask)
        )
        stim_end_level = robust_median(v_clean[stim_end_mask])
        stim_end_dep = stim_end_level - baseline if np.isfinite(stim_end_level) else np.nan
        stim_end_window_start_s = float(np.nanmin(t[stim_end_mask])) if stim_end_mask.any() else np.nan
        stim_end_window_end_s = float(np.nanmax(t[stim_end_mask])) if stim_end_mask.any() else np.nan
        stim_end_window_center_s = float(np.nanmedian(t[stim_end_mask])) if stim_end_mask.any() else np.nan

        plateau_info = detect_plateau_window(
            t,
            v_kin,
            baseline,
            onset_s,
            offset_s,
            artifact_mask,
            edge_guard_s=edge_guard_s,
            late_window_s=plateau_late_window_s,
            peak_buffer_s=peak_buffer_s,
            frac_change_thr=plateau_frac_change_thr,
        )
        late_slope = plateau_info["late_slope"]
        plateau_reached = bool(plateau_info["plateau_reached"])
        plateau_level = plateau_info["plateau_level"]
        plateau_window_start_s = float(plateau_info.get("plateau_start", np.nan)) if np.isfinite(plateau_info.get("plateau_start", np.nan)) else np.nan
        plateau_window_end_s = float(plateau_info.get("plateau_end", np.nan)) if np.isfinite(plateau_info.get("plateau_end", np.nan)) else np.nan
        plateau_window_center_s = float(0.5 * (plateau_window_start_s + plateau_window_end_s)) if np.isfinite(plateau_window_start_s) and np.isfinite(plateau_window_end_s) else np.nan

        peak_rel_idx = int(np.nanargmax(stim_v))
        peak_t = float(stim_t[peak_rel_idx])
        peak_v = float(stim_v[peak_rel_idx])
        peak_dep = float(peak_v - baseline)
        onset_idx = int(np.searchsorted(t, onset_s + edge_guard_s, side="left"))
        search_end_idx = int(np.searchsorted(t, offset_s - edge_guard_s, side="right")) - 1
        search_end_idx = max(onset_idx + 1, min(len(t) - 1, search_end_idx))

        thr20 = baseline + 0.2 * peak_dep
        thr63 = baseline + 0.632 * peak_dep
        thr80 = baseline + 0.8 * peak_dep
        t20 = t63 = t80 = np.nan
        rise_slope = np.nan
        rise_tau = np.nan
        if np.isfinite(peak_dep) and peak_dep > 0:
            t20 = first_crossing_time_interp_with_fallback(t, v_kin, thr20, onset_idx, search_end_idx, "up")
            t63 = first_crossing_time_interp_with_fallback(t, v_kin, thr63, onset_idx, search_end_idx, "up")
            t80 = first_crossing_time_interp_with_fallback(t, v_kin, thr80, onset_idx, search_end_idx, "up")
            if np.isfinite(t20) and np.isfinite(t80) and t80 > t20:
                rise_slope = (0.6 * peak_dep) / (t80 - t20)
            if np.isfinite(t63):
                rise_tau = t63 - onset_s

        post_mask = (t >= offset_s + post_search_guard_s) & (t <= t[-1]) & (~artifact_mask)
        if post_mask.sum() < 10:
            continue
        post_t = t[post_mask]
        post_v = v_kin[post_mask]
        min_rel_idx = int(np.nanargmin(post_v))
        min_t = float(post_t[min_rel_idx])
        min_v = float(post_v[min_rel_idx])
        undershoot_signed = float(min_v - baseline)
        has_undershoot = bool(undershoot_signed < 0)
        undershoot_mag = float(max(0.0, baseline - min_v))
        post_min_above_baseline_mV = float(max(0.0, min_v - baseline))

        decay_start_level = stim_end_level
        drop = decay_start_level - min_v if np.isfinite(decay_start_level) else np.nan
        decay_slope = np.nan
        decay_tau = np.nan
        td80 = td63 = td20 = np.nan
        thr80d = thr63d = thr20d = np.nan
        if np.isfinite(drop) and drop > 0:
            offset_idx = int(np.searchsorted(t, offset_s + post_search_guard_s, side="left"))
            min_idx = int(np.searchsorted(t, min_t, side="right")) - 1
            min_idx = max(offset_idx + 1, min(len(t) - 1, min_idx))
            thr80d = decay_start_level - 0.2 * drop
            thr63d = decay_start_level - 0.632 * drop
            thr20d = decay_start_level - 0.8 * drop
            td80 = first_crossing_time_interp_with_fallback(t, v_kin, thr80d, offset_idx, min_idx, "down")
            td63 = first_crossing_time_interp_with_fallback(t, v_kin, thr63d, offset_idx, min_idx, "down")
            td20 = first_crossing_time_interp_with_fallback(t, v_kin, thr20d, offset_idx, min_idx, "down")
            if np.isfinite(td80) and np.isfinite(td20) and td20 > td80:
                decay_slope = (0.6 * drop) / (td20 - td80)
            if np.isfinite(td63):
                decay_tau = td63 - offset_s

        final_mask = (t >= t[-1] - final_window_s) & (t <= t[-1] - edge_guard_s) & (~artifact_mask)
        final_v = robust_median(v_clean[final_mask])
        final_t = float(np.nanmedian(t[final_mask])) if final_mask.sum() else float(t[-1])
        final_window_start_s = float(np.nanmin(t[final_mask])) if final_mask.any() else np.nan
        final_window_end_s = float(np.nanmax(t[final_mask])) if final_mask.any() else np.nan

        return_slope = np.nan
        if (
            undershoot_signed < 0
            and final_v > min_v
            and np.isfinite(final_v)
            and np.isfinite(min_v)
            and np.isfinite(final_t)
            and final_t > min_t
        ):
            return_slope = (final_v - min_v) / (final_t - min_t)

        rows.append(
            {
                "file_id": exp.file_id,
                "file": parsed["path"].name,
                "region": exp.region,
                "condition": exp.condition,
                "group_label": exp.group_label,
                "cell_label": exp.cell_label,
                "sweep": int(sweep_num),
                "step_source": step_source,
                "stim_onset_s": onset_s,
                "stim_offset_s": offset_s,
                "baseline_mV": baseline,
                "baseline_window_start_s": baseline_window_start_s,
                "baseline_window_end_s": baseline_window_end_s,
                "peak_t_s": peak_t,
                "peak_mV": peak_v,
                "peak_depolarization_mV": peak_dep,
                "stim_end_level_mV": stim_end_level,
                "stim_end_depolarization_mV": stim_end_dep,
                "stim_end_window_start_s": stim_end_window_start_s,
                "stim_end_window_end_s": stim_end_window_end_s,
                "stim_end_window_center_s": stim_end_window_center_s,
                "plateau_reached": plateau_reached,
                "plateau_level_mV": plateau_level,
                "plateau_slope_mV_per_s": late_slope,
                "plateau_window_start_s": plateau_window_start_s,
                "plateau_window_end_s": plateau_window_end_s,
                "plateau_window_center_s": plateau_window_center_s,
                "rise_t20_s": t20,
                "rise_t63_s": t63,
                "rise_t80_s": t80,
                "rise_y20_mV": thr20 if np.isfinite(peak_dep) and peak_dep > 0 else np.nan,
                "rise_y63_mV": thr63 if np.isfinite(peak_dep) and peak_dep > 0 else np.nan,
                "rise_y80_mV": thr80 if np.isfinite(peak_dep) and peak_dep > 0 else np.nan,
                "rise_slope_mV_per_s": rise_slope,
                "rise_tau_s": rise_tau,
                "undershoot_min_t_s": min_t,
                "undershoot_min_mV": min_v,
                "undershoot_signed_mV": undershoot_signed,
                "has_undershoot": has_undershoot,
                "undershoot_magnitude_mV": undershoot_mag,
                "post_min_above_baseline_mV": post_min_above_baseline_mV,
                "decay_start_level_mV": decay_start_level,
                "decay_t80_s": td80,
                "decay_t63_s": td63,
                "decay_t20_s": td20,
                "decay_y80_mV": thr80d,
                "decay_y63_mV": thr63d,
                "decay_y20_mV": thr20d,
                "decay_slope_mV_per_s": decay_slope,
                "decay_tau_s": decay_tau,
                "final_t_s": final_t,
                "final_mV": final_v,
                "final_window_start_s": final_window_start_s,
                "final_window_end_s": final_window_end_s,
                "return_slope_mV_per_s": return_slope,
                "n_artifact_points_total": int(artifact_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_atf_inventory(atf_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in discover_atf_files(atf_root):
        factors = parse_experimental_factors(path)
        if factors.region not in {"DH", "VH"}:
            raise AtfParseError(f"Unknown region label in {path.name}")
        rows.append(
            {
                "file_id": factors.file_id,
                "file": path.name,
                "region": factors.region,
                "condition": factors.condition,
                "group_label": factors.group_label,
                "cell_label": factors.cell_label,
                "source_path": factors.source_path,
            }
        )
    df = pd.DataFrame(rows).sort_values(["condition", "region", "file_id"]).reset_index(drop=True)
    if len(df) != 37:
        raise AtfParseError(f"Expected 37 ATF files, found {len(df)}")
    return df


def count_region_condition_cells(inventory_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed = (
        inventory_df.groupby(["region", "condition"], dropna=False).size().rename("n_cells").to_dict()
    )
    for (region, condition), expected_n in EXPECTED_ATF_COUNTS.items():
        n_cells = int(observed.get((region, condition), 0))
        rows.append(
            {
                "region": region,
                "condition": condition,
                "n_cells": n_cells,
                "expected_n_cells": int(expected_n),
                "matches_expected": bool(n_cells == expected_n),
                "small_stratum": bool(n_cells < 5),
            }
        )
    return pd.DataFrame(rows).sort_values(["region", "condition"]).reset_index(drop=True)




def load_preprocessed_atf_files(
    atf_root: str | Path,
    artifact_hint_map: Optional[Mapping[str, Mapping[int, Mapping[str, Any]]]] = None,
) -> list[dict[str, Any]]:
    atf_files = discover_atf_files(atf_root)
    parsed_raw_all = [parse_atf(p) for p in atf_files]
    return [preprocess_parsed(parsed, artifact_hint_map=artifact_hint_map or MANUAL_TRANSIENT_ARTIFACT_HINTS) for parsed in parsed_raw_all]


def extract_feature_tables(
    atf_root: str | Path,
    artifact_hint_map: Optional[Mapping[str, Mapping[int, Mapping[str, Any]]]] = None,
    smoothing_engine: str = "numpy",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parsed_clean_all = load_preprocessed_atf_files(atf_root, artifact_hint_map=artifact_hint_map or MANUAL_TRANSIENT_ARTIFACT_HINTS)

    feature_frames = [extract_features(parsed, smoothing_engine=smoothing_engine) for parsed in parsed_clean_all]
    preprocess_qc_frames = [parsed["preprocess_qc"] for parsed in parsed_clean_all if "preprocess_qc" in parsed]
    feature_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    preprocess_qc_df = pd.concat(preprocess_qc_frames, ignore_index=True) if preprocess_qc_frames else pd.DataFrame()
    inventory_df = build_atf_inventory(atf_root)
    counts_df = count_region_condition_cells(inventory_df)

    if feature_df.empty:
        raise AtfParseError("Feature extraction returned no rows")
    return feature_df, preprocess_qc_df, counts_df, {
        "n_files": int(feature_df["file_id"].nunique()),
        "n_rows": int(len(feature_df)),
        "n_sweeps_per_file_min": int(feature_df.groupby("file_id")["sweep"].nunique().min()),
        "n_sweeps_per_file_max": int(feature_df.groupby("file_id")["sweep"].nunique().max()),
        "n_plateau_reached": int(feature_df["plateau_reached"].sum()),
        "n_has_undershoot": int(feature_df["has_undershoot"].sum()),
        "mean_artifact_fraction": float(preprocess_qc_df["fraction_corrected"].mean()) if not preprocess_qc_df.empty else 0.0,
    }
