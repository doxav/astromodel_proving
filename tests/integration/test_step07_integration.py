from __future__ import annotations

from tests._notebook import execute_notebook
from src.step07_assumption_sensitivity import Step07Config, run_step07_assumption_sensitivity


def test_step07_cross_table_coherence(project_root):
    result = run_step07_assumption_sensitivity(
        project_root, Step07Config(max_candidates=2, time_points=25, write_outputs=False)
    )
    gating = result["gating_family_comparison"]
    proxy = result["proxy_validity_by_ensemble"]
    split = result["compartment_split_sensitivity"]
    gating_keys = set(zip(gating["file_id"], gating["region"], gating["condition"], gating["candidate_id"]))
    proxy_keys = set(zip(proxy["file_id"], proxy["region"], proxy["condition"], proxy["candidate_id"]))
    split_keys = set(zip(split["file_id"], split["region"], split["condition"], split["candidate_id"]))
    assert proxy_keys.issubset(gating_keys)
    assert split_keys.issubset(gating_keys)


def test_step07_assumption_statuses_are_conservative(project_root):
    result = run_step07_assumption_sensitivity(
        project_root, Step07Config(max_candidates=1, time_points=25, write_outputs=False)
    )
    proxy = result["proxy_validity_by_ensemble"]
    limited = proxy[proxy["proxy_validity_status"].ne("proxy_supported")]
    if not limited.empty:
        assert limited["explicit_ecs_variant_required"].astype(bool).all()


def test_step07_notebook_executes_and_saves_auditable_copy(project_root):
    executed = execute_notebook(project_root / "analysis" / "07_assumption_sensitivity.ipynb", project_root)
    assert executed.exists()
