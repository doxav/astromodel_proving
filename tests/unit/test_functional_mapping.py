from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.functional_mapping import (
    apply_efficiency_quadrants,
    classify_efficiency_quadrant,
    classify_sigmoid_state_change,
    compute_efficiency_threshold_table,
    direction_label,
    extract_ko_kinetic_features,
)


def test_extract_ko_kinetic_features_finite_score() -> None:
    t = np.linspace(0.0, 20.0, 201)
    ko = np.where(t <= 10.0, 4.8 + 0.2 * t, 6.8 - 0.15 * (t - 10.0))

    features = extract_ko_kinetic_features(t, ko, onset_s=1.0, offset_s=10.0)

    assert features["Ko_efficiency_status"] == "defined"
    assert np.isfinite(float(features["Ko_efficiency_score"]))
    assert float(features["Ko_rise_rate_mM_per_s"]) > 0
    assert float(features["Ko_decay_rate_abs_mM_per_s"]) > 0


def test_extract_ko_kinetic_features_flat_trace_is_undefined() -> None:
    t = np.linspace(0.0, 20.0, 201)
    ko = np.full_like(t, 4.8)

    features = extract_ko_kinetic_features(t, ko, onset_s=1.0, offset_s=10.0)

    assert features["Ko_efficiency_status"] == "undefined_flat_or_missing"
    assert not np.isfinite(float(features["Ko_efficiency_score"]))


def test_extract_ko_kinetic_features_rejects_bad_lengths() -> None:
    with pytest.raises(ValueError, match="same length|at least three points"):
        extract_ko_kinetic_features([0.0, 1.0, 2.0], [4.8, 5.0], onset_s=0.5, offset_s=1.5)


@pytest.mark.parametrize(
    ("rise", "decay", "expected"),
    [
        (2.0, 2.0, "fast_rise_fast_decay"),
        (0.5, 2.0, "slow_rise_fast_decay"),
        (0.5, 0.5, "slow_rise_slow_decay"),
        (2.0, 0.5, "fast_rise_slow_decay"),
    ],
)
def test_classify_efficiency_quadrants(rise: float, decay: float, expected: str) -> None:
    out = classify_efficiency_quadrant(rise, decay, rise_cutoff=1.0, decay_cutoff=1.0)
    assert out["Ko_efficiency_quadrant"] == expected


def test_efficiency_thresholds_are_frozen_from_baseline() -> None:
    baseline = pd.DataFrame(
        {
            "source_scope": ["legacy"] * 4,
            "current_na": [100] * 4,
            "Ko_rise_rate_mM_per_s": [1.0, 2.0, 3.0, 4.0],
            "Ko_decay_rate_abs_mM_per_s": [1.0, 1.0, 5.0, 5.0],
        }
    )
    thresholds = compute_efficiency_threshold_table(baseline, min_rows_per_stratum=2)
    perturbed = pd.DataFrame(
        {
            "source_scope": ["legacy"],
            "current_na": [100],
            "Ko_rise_rate_mM_per_s": [4.0],
            "Ko_decay_rate_abs_mM_per_s": [1.0],
        }
    )

    labeled = apply_efficiency_quadrants(perturbed, thresholds)

    assert labeled["Ko_efficiency_quadrant"].iloc[0] == "fast_rise_slow_decay"
    assert labeled["Ko_efficiency_threshold_interpretation"].iloc[0].startswith("descriptive_")


def test_sigmoid_state_change_labels() -> None:
    assert (
        classify_sigmoid_state_change(
            "closed_low",
            "closed_low",
            "open_high",
            "closed_low",
            "recruited_during_load_then_closed_by_end",
        )
        == "opened_during_stim_then_closed_by_end"
    )
    assert (
        classify_sigmoid_state_change(
            "open_high", "open_high", "closed_low", "closed_low", "fully_closed_at_sim_end"
        )
        == "open_to_closed"
    )


def test_direction_label() -> None:
    assert direction_label(0.1) == "increase"
    assert direction_label(-0.1) == "decrease"
    assert direction_label(0.0) == "no_change"
    assert direction_label(np.nan) == "undefined"
