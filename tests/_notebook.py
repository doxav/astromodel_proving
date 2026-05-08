from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(notebook_path: Path, project_root: Path, timeout: int = 1200) -> Path:
    notebook_path = notebook_path.resolve()
    project_root = project_root.resolve()
    old_cwd = Path.cwd()
    old_env = os.environ.get("ASTROMODEL_PROJECT_ROOT")
    try:
        os.chdir(project_root)
        os.environ["ASTROMODEL_PROJECT_ROOT"] = str(project_root)
        nb = nbformat.read(notebook_path, as_version=4)
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False)
        client.execute()
        executed_dir = project_root / "outputs" / "executed_notebooks"
        executed_dir.mkdir(parents=True, exist_ok=True)
        executed_path = executed_dir / notebook_path.name
        nbformat.write(nb, executed_path)
        return executed_path
    finally:
        os.chdir(old_cwd)
        if old_env is None:
            os.environ.pop("ASTROMODEL_PROJECT_ROOT", None)
        else:
            os.environ["ASTROMODEL_PROJECT_ROOT"] = old_env
