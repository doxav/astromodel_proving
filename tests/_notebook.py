from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(notebook_path: Path, project_root: Path) -> Path:
    """Execute a notebook in place and return its compatibility alias path."""

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    env = os.environ.copy()
    env["ASTROMODEL_PROJECT_ROOT"] = str(project_root)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
        env=env,
        allow_errors=False,
    )
    client.execute()

    with notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return project_root / "outputs" / "executed_notebooks" / notebook_path.name
