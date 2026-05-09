from __future__ import annotations

from tests._notebook import execute_notebook
from src.step05_mechanistic_decomposition import Step05Config, run_step05_mechanistic_decomposition


def test_step05_enrichment_and_geometry_are_coherent(project_root):
    result = run_step05_mechanistic_decomposition(
        project_root,
        Step05Config(max_candidates=3, time_points=60, bootstrap_iterations=2, write_outputs=False),
    )
    clusters = set(result["mechanism_clusters"]["mechanism_cluster"])
    enrichment_clusters = set(result["region_mechanism_enrichment"]["mechanism_cluster"])
    assert enrichment_clusters.issubset(clusters)
    assert result["geometry_classification"]["validation_status"].eq("pending_step06").all()


def test_step05_notebook_executes_and_saves_auditable_copy(project_root):
    executed = execute_notebook(project_root / "analysis" / "05_mechanistic_decomposition.ipynb", project_root)
    assert executed.exists()
