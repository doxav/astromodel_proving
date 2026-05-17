from __future__ import annotations

import json
import zipfile

from src.step04_outputs import (
    STEP04_DOWNSTREAM_ARTIFACTS,
    save_step04_run_snapshot,
)


def test_save_step04_run_snapshot_writes_params_and_archive(tmp_path, monkeypatch):
    out = tmp_path / "step04_out"
    out.mkdir()
    for filename in STEP04_DOWNSTREAM_ARTIFACTS.values():
        (out / filename).write_text("placeholder\n", encoding="utf-8")

    notebook = tmp_path / "04_cell_specific_six_sweep_fitting.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    backup = tmp_path / "drive_backup"
    monkeypatch.setenv("ASTROMODEL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ASTROMODEL_STEP04_OUTPUT_DIR", str(out))
    monkeypatch.setenv("ASTROMODEL_STEP04_OPTIMIZER_BACKEND", "least_squares")
    monkeypatch.setenv("ASTROMODEL_STEP04_MAX_CELLS", "1")

    snapshot = save_step04_run_snapshot(
        out,
        backup_dir=backup,
        label="smoke",
        compress=True,
        include_paths=[notebook],
        extra={"purpose": "unit-test"},
    )

    snapshot_path = out / "step04_run_snapshot.json"
    assert snapshot_path.exists()
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["missing_artifacts"] == []
    assert payload["step04_env"]["ASTROMODEL_STEP04_OPTIMIZER_BACKEND"] == "least_squares"
    assert payload["extra"]["purpose"] == "unit-test"

    archive = snapshot["archive_path"]
    assert archive.endswith(".zip")
    assert zipfile.is_zipfile(archive)
    with zipfile.ZipFile(archive) as zf:
        assert "step04_run_snapshot.json" in zf.namelist()
        assert "analysis_summary.json" in zf.namelist()

    backup_snapshot = list(backup.glob("smoke_*/step04_run_snapshot.json"))
    assert backup_snapshot
    copied_notebook = list(backup.glob("smoke_*/04_cell_specific_six_sweep_fitting.ipynb"))
    assert copied_notebook


def test_save_step04_run_snapshot_reports_missing_artifacts(tmp_path):
    out = tmp_path / "partial_step04_out"
    out.mkdir()
    (out / "analysis_summary.json").write_text("{}", encoding="utf-8")

    snapshot = save_step04_run_snapshot(out, compress=False)

    assert "archive_path" not in snapshot
    assert snapshot["missing_artifacts"]
    assert "cell_fit_candidates.csv" in snapshot["missing_artifacts"]
