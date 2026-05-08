from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests._notebook import execute_notebook


def test_step00_notebook_executes_and_writes_outputs(project_root: Path) -> None:
    notebook_path = project_root / "analysis" / "00_data_provenance_audit.ipynb"
    executed_path = execute_notebook(notebook_path, project_root)
    assert executed_path.exists()

    outputs_dir = project_root / "outputs" / "provenance"
    expected = {
        "db_study_summary.csv",
        "trace_source_summary.csv",
        "control_trace_verification.csv",
        "atf_region_condition_inventory.csv",
        "atf_region_condition_counts.csv",
    }
    assert expected.issubset({p.name for p in outputs_dir.glob("*.csv")})

    db_summary = pd.read_csv(outputs_dir / "db_study_summary.csv")
    counts = pd.read_csv(outputs_dir / "atf_region_condition_counts.csv")
    assert len(db_summary) == 18
    assert len(counts) == 6
