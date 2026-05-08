from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests._notebook import execute_notebook


def test_step03_notebook_executes_and_writes_outputs(project_root: Path) -> None:
    notebook_path = project_root / "analysis" / "03_combined_identifiability_profiles_fim.ipynb"
    executed_path = project_root / "outputs" / "executed_notebooks" / notebook_path.name
    if not executed_path.exists():
        executed_path = execute_notebook(notebook_path, project_root)
    assert executed_path.exists()

    outputs_dir = project_root / "outputs" / "identifiability"
    expected = {
        "effective_parameter_map.csv",
        "structural_invariance_diagnostics.csv",
        "invariance_diagnostics.csv",
        "profile_likelihoods.csv",
        "profile_summary.csv",
        "fim_spectrum.csv",
        "fim_mode_loadings.csv",
        "fim_diagnostics.csv",
        "interpretation_notes.csv",
        "analysis_summary.csv",
    }
    assert expected.issubset({p.name for p in outputs_dir.glob("*.csv")})

    summary = pd.read_csv(outputs_dir / "analysis_summary.csv")
    assert "not a full symbolic STRIKE-GOLDD proof" in summary["claim_boundary"].iloc[0]
