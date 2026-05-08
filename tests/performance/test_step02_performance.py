from __future__ import annotations

import numpy as np


def test_step02_benchmark_returns_tuning_decision(step02_results_with_benchmark) -> None:
    benchmark = step02_results_with_benchmark["performance_benchmark"]
    assert {"preprocess", "feature_extraction", "decision"}.issubset(set(benchmark["stage"]))

    decision_row = benchmark[benchmark["stage"] == "decision"].iloc[0]
    assert decision_row["numba_decision"] in {"keep_numpy_default", "use_numba_default"}
    assert isinstance(decision_row["note"], str) and len(decision_row["note"]) > 10

    preprocess_row = benchmark[benchmark["stage"] == "preprocess"].iloc[0]
    numpy_row = benchmark[(benchmark["stage"] == "feature_extraction") & (benchmark["engine"] == "numpy")].iloc[0]
    assert float(preprocess_row["elapsed_seconds"]) < 60.0
    assert float(numpy_row["elapsed_seconds"]) < 10.0


def test_step02_performance_table_supports_numba_decision_logic(step02_results_with_benchmark) -> None:
    benchmark = step02_results_with_benchmark["performance_benchmark"]
    decision_row = benchmark[benchmark["stage"] == "decision"].iloc[0]
    assert np.isfinite(float(decision_row["elapsed_seconds"]))

    numba_rows = benchmark[(benchmark["stage"] == "feature_extraction") & (benchmark["engine"] == "numba")]
    if len(numba_rows):
        assert np.isfinite(float(decision_row["compute_speedup_numba_vs_numpy"]))
        assert np.isfinite(float(decision_row["estimated_total_gain_seconds"]))
