from __future__ import annotations

import time

from src.step07_assumption_sensitivity import (
    Step07Config,
    compare_step07_runtime_presets,
    run_step07_assumption_sensitivity,
)


def test_step07_small_run_is_practical(project_root):
    t0 = time.perf_counter()
    run_step07_assumption_sensitivity(
        project_root,
        Step07Config(max_candidates=1, time_points=20, gating_families=("sigmoid", "tanh"), write_outputs=False),
    )
    assert time.perf_counter() - t0 < 45.0


def test_step07_performance_comparison_records_tuning_decision(project_root):
    perf = compare_step07_runtime_presets(project_root, max_candidates=1)
    assert {"coarse", "default"} == set(perf["preset"])
    assert (perf["elapsed_seconds"] >= 0).all()
    assert perf["recommendation"].str.contains("default|coarse", regex=True).all()
