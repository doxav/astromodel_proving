from __future__ import annotations

import time

from src.step05_mechanistic_decomposition import Step05Config, compare_simulation_grid_performance, run_step05_mechanistic_decomposition


def test_step05_small_run_is_practical(project_root):
    t0 = time.perf_counter()
    run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(max_candidates=1, time_points=60, bootstrap_iterations=0, write_outputs=False),
    )
    assert time.perf_counter() - t0 < 45.0


def test_step05_grid_performance_comparison_records_tuning_decision(project_root):
    perf = compare_simulation_grid_performance(project_root, max_candidates=1)
    assert {"coarse", "default"} == set(perf["preset"])
    assert (perf["elapsed_seconds"] >= 0).all()
    assert perf["recommendation"].str.contains("default|coarse", regex=True).all()
