from __future__ import annotations

from src.step08_parameter_plausibility import Step08Config, compare_step08_runtime_presets, run_step08_parameter_plausibility


def test_step08_small_run_is_practical(project_root):
    result = run_step08_parameter_plausibility(project_root, Step08Config(max_candidates=1, write_outputs=False))
    assert len(result["parameter_range_audit"]) > 0


def test_step08_performance_comparison_records_tuning_decision(project_root):
    perf = compare_step08_runtime_presets(project_root, max_candidates=1)
    assert {"coarse", "default"}.issubset(set(perf["preset"]))
    assert (perf["elapsed_seconds"] >= 0).all()
    assert perf["tuning_recommendation"].str.len().gt(0).all()
