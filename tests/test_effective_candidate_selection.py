from __future__ import annotations

import pandas as pd
import pytest

from src.effective_candidate_selection import (
    select_effective_diverse_candidates,
    summarize_effective_diverse_selection,
)


def _candidate_rows() -> pd.DataFrame:
    rows = [
        {
            "file_id": "cell_a",
            "region": "DH",
            "condition": "MFA",
            "candidate_id": "a_best",
            "accepted_all6": True,
            "ensemble_rank": 1,
            "mean_weighted_pass_fraction": 0.50,
            "mean_trace_rmse_mV": 4.0,
            "P_gap_eff": 1e-6,
            "gamma_t_eff": 1e-5,
            "gamma_s_eff": 1e-5,
            "volume_ratio_wa_wo": 1.0,
        },
        {
            "file_id": "cell_a",
            "region": "DH",
            "condition": "MFA",
            "candidate_id": "a_duplicate",
            "accepted_all6": True,
            "ensemble_rank": 2,
            "mean_weighted_pass_fraction": 0.49,
            "mean_trace_rmse_mV": 4.2,
            "P_gap_eff": 1.1e-6,
            "gamma_t_eff": 1e-5,
            "gamma_s_eff": 1e-5,
            "volume_ratio_wa_wo": 1.0,
        },
        {
            "file_id": "cell_a",
            "region": "DH",
            "condition": "MFA",
            "candidate_id": "a_far",
            "accepted_all6": True,
            "ensemble_rank": 3,
            "mean_weighted_pass_fraction": 0.48,
            "mean_trace_rmse_mV": 4.3,
            "P_gap_eff": 1e-4,
            "gamma_t_eff": 1e-5,
            "gamma_s_eff": 1e-5,
            "volume_ratio_wa_wo": 1.0,
        },
        {
            "file_id": "cell_b",
            "region": "VH",
            "condition": "MFA_BA",
            "candidate_id": "b_implausible",
            "accepted_all6": True,
            "ensemble_rank": 1,
            "mean_weighted_pass_fraction": 0.60,
            "mean_trace_rmse_mV": 3.0,
            "P_gap_eff": 1e-2,
            "gamma_t_eff": 1e-5,
            "gamma_s_eff": 1e-5,
            "volume_ratio_wa_wo": 1.0,
        },
        {
            "file_id": "cell_b",
            "region": "VH",
            "condition": "MFA_BA",
            "candidate_id": "b_plausible",
            "accepted_all6": True,
            "ensemble_rank": 2,
            "mean_weighted_pass_fraction": 0.55,
            "mean_trace_rmse_mV": 3.5,
            "P_gap_eff": 1e-5,
            "gamma_t_eff": 1e-5,
            "gamma_s_eff": 1e-5,
            "volume_ratio_wa_wo": 1.0,
        },
    ]
    return pd.DataFrame(rows)


def test_quality_filtered_maximin_skips_effective_duplicates_and_implausible_candidates() -> None:
    selected = select_effective_diverse_candidates(
        _candidate_rows(),
        candidates_per_cell=2,
        strategy="quality_filtered_effective_maximin",
        distance_threshold=0.5,
    )

    by_cell = selected.groupby("file_id")["candidate_id"].apply(list).to_dict()

    assert by_cell["cell_a"] == ["a_best", "a_far"]
    assert by_cell["cell_b"] == ["b_plausible"]
    assert selected["effective_diverse_selected"].all()
    assert selected["effective_plausible"].all()


def test_effective_selection_summary_reports_cell_spacing() -> None:
    selected = select_effective_diverse_candidates(
        _candidate_rows(),
        candidates_per_cell=2,
        strategy="effective_maximin_best_seed",
    )

    summary = summarize_effective_diverse_selection(selected)
    cell_a = summary.loc[summary["file_id"].eq("cell_a")].iloc[0]

    assert cell_a["n_selected"] == 2
    assert cell_a["effective_cluster_count"] == 2
    assert cell_a["min_pairwise_effective_log_distance"] >= 1.0
    assert bool(cell_a["rank1_retained"])


def test_effective_selector_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy must be one of"):
        select_effective_diverse_candidates(
            _candidate_rows(),
            strategy="not_a_strategy",
        )
