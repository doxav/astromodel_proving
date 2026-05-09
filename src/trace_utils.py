"""Numerical helpers for voltage traces."""

from __future__ import annotations

import numpy as np


def baseline_center(
    time_s: np.ndarray,
    vm_mV: np.ndarray,
    onset_s: float,
    *,
    include_endpoint: bool = False,
) -> np.ndarray:
    """Subtract the median pre-onset baseline from a voltage trace."""

    t = np.asarray(time_s, dtype=float)
    v = np.asarray(vm_mV, dtype=float)
    start = max(float(t[0]), float(onset_s) - 5.0)
    end = max(float(t[0]), float(onset_s) - 1.0)
    if include_endpoint:
        mask = (t >= start) & (t <= end)
    else:
        mask = (t >= start) & (t < end)
    if not np.any(mask):
        mask = t < float(onset_s)
    if not np.any(mask):
        mask = np.arange(len(v)) < max(1, min(50, len(v)))
    return v - float(np.nanmedian(v[mask]))


def downsample_trace(time_s: np.ndarray, vm_mV: np.ndarray, n_points: int, *, preserve_short: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a trace onto a common grid, preserving short traces."""

    t = np.asarray(time_s, dtype=float)
    v = np.asarray(vm_mV, dtype=float)
    if int(n_points) <= 0:
        raise ValueError("n_points must be positive")
    if preserve_short and len(t) <= int(n_points):
        return t.copy(), v.copy()
    grid = np.linspace(float(t[0]), float(t[-1]), int(n_points), dtype=float)
    return grid, np.interp(grid, t, v)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error."""

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError(f"shape mismatch: {aa.shape} != {bb.shape}")
    finite = np.isfinite(aa) & np.isfinite(bb)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((aa[finite] - bb[finite]) ** 2)))


def nrmse(a: np.ndarray, b: np.ndarray, denominator: float | None = None) -> float:
    """Normalized RMSE using an explicit or range-derived denominator."""

    denom = float(denominator) if denominator is not None else float(np.nanmax(b) - np.nanmin(b))
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        denom = max(float(np.nanmax(np.abs(b))), 1.0)
    value = rmse(a, b)
    return float(value / denom)
