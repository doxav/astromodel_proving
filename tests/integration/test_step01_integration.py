from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests._notebook import execute_notebook


def test_step01_notebook_executes_and_writes_outputs(project_root: Path) -> None:
    notebook_path = project_root / "analysis" / "01_postfit_sqlite_pipeline.ipynb"
    executed_path = project_root / "outputs" / "executed_notebooks" / notebook_path.name
    if not executed_path.exists():
        executed_path = execute_notebook(notebook_path, project_root)
    assert executed_path.exists()

    outputs_dir = project_root / "outputs" / "postfit_sqlite"
    expected = {
        "top_trials_all_dbs.csv",
        "effective_parameter_summary.csv",
        "representative_mechanism_summary.csv",
    }
    assert expected.issubset({p.name for p in outputs_dir.glob("*.csv")})

    rep = pd.read_csv(outputs_dir / "representative_mechanism_summary.csv")
    assert len(rep) == 3
    assert set(rep["condition"]) == {"CONTROL", "MFA", "BARIUM"}
