"""Shared protocol timing/window helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import canonical_condition, protocol_condition


def stim_window_seconds(condition: str) -> tuple[float, float]:
    """Return stimulus start/end in seconds for CONTROL vs MFA/BARIUM protocols."""

    return (11.173, 31.173) if protocol_condition(condition) == "CONTROL" else (21.140, 41.140)


def default_onset_seconds(condition: str) -> float:
    """Return the default feature onset in seconds for a condition."""

    return stim_window_seconds(condition)[0]


def representative_context(condition: str, current_na: int, n_timepoints: int) -> dict[str, Any]:
    """Build a standard lightweight simulation context for mechanism summaries."""

    _start, end = stim_window_seconds(condition)
    return {
        "experiment_type": protocol_condition(condition),
        "condition": canonical_condition(condition),
        "current_na": int(current_na),
        "sim_time_ms": np.linspace(0.0, (end + 5.0) * 1000.0, int(n_timepoints), dtype=float),
    }
