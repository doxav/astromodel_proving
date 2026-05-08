from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests._notebook import execute_notebook


def test_step02_notebook_executes_and_writes_outputs(project_root: Path) -> None:
    notebook_path = project_root / "analysis" / "02_rebuild_atf_thresholds.ipynb"
    executed_path = project_root / "outputs" / "executed_notebooks" / notebook_path.name
    if not executed_path.exists():
        executed_path = execute_notebook(notebook_path, project_root)
    assert executed_path.exists()

    outputs_dir = project_root / "outputs" / "features"
    expected = {
        "feature_table_by_sweep.csv",
        "condition_region_sweep_thresholds.csv",
        "feature_reliability_weights.csv",
        "condition_feature_reliability.csv",
        "region_condition_cell_counts.csv",
        "region_effect_summary.csv",
        "redundancy_diagnostics.csv",
    }
    assert expected.issubset({p.name for p in outputs_dir.glob("*.csv")})

    feature_df = pd.read_csv(outputs_dir / "feature_table_by_sweep.csv")
    assert len(feature_df) == 222
