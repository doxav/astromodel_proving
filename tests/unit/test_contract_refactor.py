from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import postfit_sqlite
from src.contracts import canonical_condition, protocol_condition
from src.feature_contracts import ThresholdScope, build_threshold_table, compute_reliability_weights, score_feature_contract
from src.parameter_space import EffectiveParams, coordinate_value, effective_from_flat, flat_from_effective, set_coordinate
from src.protocols import default_onset_seconds, representative_context, stim_window_seconds
from src.trace_utils import baseline_center, downsample_trace, nrmse, rmse


def representative_flat() -> dict[str, float]:
    return {
        "d": 2.0,
        "pk": 3e-5,
        "gt": 7.0,
        "gs": 8.0,
        "wo": 1250.0,
        "w_a": 2000.0,
        "gki": 42.0,
        "eps": 0.02,
    }


def test_effective_parameter_values_match_legacy_step01_wrapper() -> None:
    params = representative_flat()
    for condition in ["CONTROL", "MFA", "BARIUM", "MFA_BA"]:
        expected = postfit_sqlite.effective_parameters_from_flat(params, condition, 100)
        observed = effective_from_flat(params, condition=condition, current_na=100).as_dict()
        assert observed == expected


def test_condition_canonicalization_and_protocol_contract() -> None:
    assert canonical_condition("mfa+ba") == "MFA_BA"
    assert canonical_condition("barium") == "MFA_BA"
    assert protocol_condition("MFA_BA") == "BARIUM"
    assert protocol_condition("control") == "CONTROL"


def test_flat_effective_round_trip_preserves_effective_coordinates() -> None:
    eff = EffectiveParams(P_gap_eff=6e-5, gamma_t_eff=5.804e-5, gamma_s_eff=6.633e-5, volume_ratio_wa_wo=1.6)
    flat = flat_from_effective("MFA_BA", eff, extra={"gki": 42.0})
    observed = effective_from_flat(flat, condition="MFA_BA", current_na=100)
    assert np.isclose(observed.P_gap_eff, eff.P_gap_eff)
    assert np.isclose(observed.gamma_t_eff, eff.gamma_t_eff)
    assert np.isclose(observed.gamma_s_eff, eff.gamma_s_eff)
    assert np.isclose(observed.volume_ratio_wa_wo, eff.volume_ratio_wa_wo)
    assert flat["gki"] == 42.0


def test_protocol_windows_and_representative_context_are_canonical() -> None:
    assert stim_window_seconds("CONTROL") == (11.173, 31.173)
    assert stim_window_seconds("MFA") == (21.140, 41.140)
    assert stim_window_seconds("MFA_BA") == (21.140, 41.140)
    assert default_onset_seconds("CONTROL") == 11.173
    ctx = representative_context("MFA_BA", current_na=150, n_timepoints=12)
    assert ctx["experiment_type"] == "BARIUM"
    assert ctx["current_na"] == 150
    assert len(ctx["sim_time_ms"]) == 12
    assert np.isclose(ctx["sim_time_ms"][-1], 46140.0)


def test_trace_utils_baseline_downsample_and_errors() -> None:
    t = np.linspace(0.0, 9.0, 10)
    v = np.array([5, 5, 5, 5, 5, 7, 9, 7, 5, 5], dtype=float)
    centered = baseline_center(t, v, onset_s=5.0)
    assert np.isclose(np.median(centered[t < 5.0]), 0.0)
    t_down, v_down = downsample_trace(t, v, n_points=5)
    assert len(t_down) == len(v_down) == 5
    assert np.isclose(rmse(np.array([1.0, 3.0]), np.array([1.0, 1.0])), np.sqrt(2.0))
    assert np.isclose(nrmse(np.array([1.0, 3.0]), np.array([1.0, 1.0]), denominator=2.0), np.sqrt(2.0) / 2.0)


def test_feature_contract_fixture_scores_expected_pass_fraction() -> None:
    feature_df = pd.DataFrame(
        [
            {"file_id": "A", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 1.0, "stim_end_depolarization_mV": 0.8},
            {"file_id": "B", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 2.0, "stim_end_depolarization_mV": 1.0},
            {"file_id": "C", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 3.0, "stim_end_depolarization_mV": 1.2},
        ]
    )
    features = ["peak_depolarization_mV", "stim_end_depolarization_mV"]
    reliability = compute_reliability_weights(feature_df, features)
    thresholds = build_threshold_table(feature_df, reliability, ThresholdScope("region_specific"), feature_columns=features)
    score = score_feature_contract(
        {"peak_depolarization_mV": 2.0, "stim_end_depolarization_mV": 10.0},
        thresholds,
        condition="CONTROL",
        region="DH",
        sweep=1,
        feature_columns=features,
    )
    assert score["pass_peak_depolarization_mV"] is True
    assert score["pass_stim_end_depolarization_mV"] is False
    assert 0.0 < score["weighted_pass_fraction"] < 1.0


def test_feature_contract_scores_existing_csv_regression(project_root) -> None:
    feature_df = pd.read_csv(project_root / "outputs" / "features" / "feature_table_by_sweep.csv")
    thresholds = pd.read_csv(project_root / "outputs" / "features" / "condition_region_sweep_thresholds.csv")
    row = feature_df[(feature_df["condition"] == "CONTROL") & (feature_df["region"] == "DH") & (feature_df["sweep"] == 1)].iloc[0]
    score = score_feature_contract(
        row.to_dict(),
        thresholds,
        condition=str(row["condition"]),
        region=str(row["region"]),
        sweep=int(row["sweep"]),
        feature_columns=[str(x) for x in thresholds["feature"].dropna().unique()],
    )
    assert len(thresholds) > 0
    assert {"condition", "region", "sweep", "feature", "acceptable_lower", "acceptable_upper"}.issubset(thresholds.columns)
    assert score["weighted_pass_fraction"] == 1.0
    assert score["weighted_feature_penalty"] == 0.0


def test_representative_context_includes_canonical_data_condition() -> None:
    ctx = representative_context("BARIUM", current_na=100, n_timepoints=3)

    assert ctx["experiment_type"] == "BARIUM"
    assert ctx["condition"] == "MFA_BA"
    assert ctx["current_na"] == 100


def test_set_coordinate_rejects_nonpositive_effective_coordinates() -> None:
    params = representative_flat()

    with pytest.raises(ValueError):
        set_coordinate(params, "P_gap_eff", 0.0)

    with pytest.raises(ValueError):
        set_coordinate(params, "volume_ratio_wa_wo", -1.0)


def test_flat_from_effective_rejects_nonpositive_volume_ratio() -> None:
    eff = EffectiveParams(
        P_gap_eff=1e-5,
        gamma_t_eff=1e-5,
        gamma_s_eff=1e-5,
        volume_ratio_wa_wo=0.0,
    )

    with pytest.raises(ValueError):
        flat_from_effective("CONTROL", eff)


def test_downsample_trace_rejects_nonpositive_point_count() -> None:
    with pytest.raises(ValueError):
        downsample_trace(np.array([0.0, 1.0]), np.array([1.0, 2.0]), 0)


def test_rmse_uses_pairwise_finite_values() -> None:
    observed = rmse(
        np.array([1.0, np.nan, 5.0, np.inf]),
        np.array([2.0, 3.0, 1.0, 4.0]),
    )

    assert observed == pytest.approx(np.sqrt(((1.0 - 2.0) ** 2 + (5.0 - 1.0) ** 2) / 2.0))


def test_shared_feature_score_can_include_binary_penalty() -> None:
    feature_df = pd.DataFrame(
        [
            {
                "file_id": "A",
                "condition": "CONTROL",
                "region": "DH",
                "sweep": 1,
                "peak_depolarization_mV": 1.0,
                "stim_end_depolarization_mV": 0.8,
            },
            {
                "file_id": "B",
                "condition": "CONTROL",
                "region": "DH",
                "sweep": 1,
                "peak_depolarization_mV": 2.0,
                "stim_end_depolarization_mV": 1.0,
            },
            {
                "file_id": "C",
                "condition": "CONTROL",
                "region": "DH",
                "sweep": 1,
                "peak_depolarization_mV": 3.0,
                "stim_end_depolarization_mV": 1.2,
            },
        ]
    )
    features = ["peak_depolarization_mV", "stim_end_depolarization_mV"]
    reliability = compute_reliability_weights(feature_df, features)
    thresholds = build_threshold_table(
        feature_df,
        reliability,
        ThresholdScope("region_specific"),
        feature_columns=features,
    )

    score = score_feature_contract(
        {
            "peak_depolarization_mV": 2.0,
            "stim_end_depolarization_mV": 1.0,
            "plateau_reached": True,
            "has_undershoot": False,
        },
        thresholds,
        condition="CONTROL",
        region="DH",
        sweep=1,
        empirical={
            "plateau_reached": False,
            "has_undershoot": False,
        },
        feature_columns=features,
    )

    assert score["weighted_pass_fraction"] == pytest.approx(1.0)
    assert score["binary_penalty"] == pytest.approx(0.5)


def test_feature_contract_requires_explicit_feature_columns_when_called_standalone() -> None:
    feature_df = pd.DataFrame(
        [
            {"file_id": "A", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 1.0},
            {"file_id": "B", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 2.0},
        ]
    )

    with pytest.raises(ValueError):
        compute_reliability_weights(feature_df)


def test_effective_parameter_golden_formula_is_independent_of_wrapper() -> None:
    params = {
        "d": 2.0,
        "pk": 3.0e-5,
        "gt": 7.0,
        "gs": 8.0,
        "wo": 1250.0,
        "w_a": 2000.0,
    }

    observed = effective_from_flat(params, condition="CONTROL", current_na=100)

    assert observed.P_gap_eff == pytest.approx(6.0e-5)
    assert observed.gamma_t_eff == pytest.approx(7.0 * 1600.0 / (2000.0 * 96485.0))
    assert observed.gamma_s_eff == pytest.approx(8.0 * 1600.0 / (2000.0 * 96485.0))
    assert observed.volume_ratio_wa_wo == pytest.approx(2000.0 / 1250.0)


def test_set_coordinate_preserves_relative_gamma_scaling() -> None:
    params = representative_flat()
    base = effective_from_flat(params)
    updated = set_coordinate(params, "gamma_t_eff", 2.0 * base.gamma_t_eff)

    assert effective_from_flat(updated).gamma_t_eff == pytest.approx(2.0 * base.gamma_t_eff)
    assert updated["gt"] == pytest.approx(2.0 * params["gt"])


def test_multisweep_trace_nrmse_matches_formula() -> None:
    obs = np.array([0.0, 1.0, 2.0])
    sim = np.array([0.0, 2.0, 4.0])
    expected = float(np.sqrt(np.mean((sim - obs) ** 2)) / max(np.max(np.abs(obs)), 1.0))

    assert nrmse(sim, obs, denominator=max(float(np.nanmax(np.abs(obs))), 1.0)) == pytest.approx(expected)


def test_step04_all6_feature_score_keeps_soft_pass_fraction() -> None:
    feature_df = pd.DataFrame(
        [
            {"file_id": "A", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 1.0},
            {"file_id": "B", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 2.0},
            {"file_id": "C", "condition": "CONTROL", "region": "DH", "sweep": 1, "peak_depolarization_mV": 3.0},
        ]
    )
    features = ["peak_depolarization_mV"]
    reliability = compute_reliability_weights(feature_df, features)
    thresholds = build_threshold_table(feature_df, reliability, ThresholdScope("region_specific"), feature_columns=features)

    hard = score_feature_contract(
        {"peak_depolarization_mV": 10.0},
        thresholds,
        condition="CONTROL",
        region="DH",
        sweep=1,
        feature_columns=features,
        pass_fraction_mode="hard",
    )
    soft = score_feature_contract(
        {"peak_depolarization_mV": 10.0},
        thresholds,
        condition="CONTROL",
        region="DH",
        sweep=1,
        feature_columns=features,
        pass_fraction_mode="soft",
    )

    assert hard["weighted_pass_fraction"] == 0.0
    assert 0.0 <= soft["weighted_pass_fraction"] <= 1.0
    assert soft["feature_loss"] == pytest.approx(1.0 - soft["weighted_pass_fraction"])


def test_step04_all6_baseline_endpoint_matches_legacy_mask() -> None:
    t = np.arange(0.0, 6.0, 1.0)
    v = np.array([0.0, 0.0, 0.0, 100.0, 10.0, 10.0])
    onset_s = 4.0

    # Legacy all-six mask included onset_s - 1.0, so the 100.0 point is included.
    centered = baseline_center(t, v, onset_s, include_endpoint=True)

    assert centered.tolist() == pytest.approx((v - np.median([0.0, 0.0, 0.0, 100.0])).tolist())


def test_step04_multisweep_downsample_always_returns_requested_points() -> None:
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 1.0, 0.0])

    grid, values = downsample_trace(t, v, 5, preserve_short=False)

    assert len(grid) == 5
    assert len(values) == 5
    assert grid.tolist() == pytest.approx(np.linspace(0.0, 2.0, 5).tolist())


def test_step04_all6_downsample_preserves_short_trace() -> None:
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 1.0, 0.0])

    grid, values = downsample_trace(t, v, 5, preserve_short=True)

    assert grid.tolist() == pytest.approx(t.tolist())
    assert values.tolist() == pytest.approx(v.tolist())
