from __future__ import annotations

import time

from src.step06_predictive_validation import (
    Step06Config,
    compare_step06_runtime_presets,
    run_step06_predictive_validation,
)


def test_step06_small_run_is_practical(project_root):
    t0 = time.perf_counter()
    run_step06_predictive_validation(
        project_root, Step06Config(max_candidates=1, time_points=30, write_outputs=False)
    )
    assert time.perf_counter() - t0 < 45.0


def test_step06_performance_comparison_records_tuning_decision(project_root):
    perf = compare_step06_runtime_presets(project_root, max_candidates=1)
    assert {"coarse", "default"} == set(perf["preset"])
    assert (perf["elapsed_seconds"] >= 0).all()
    assert perf["recommendation"].str.contains("default|coarse", regex=True).all()
