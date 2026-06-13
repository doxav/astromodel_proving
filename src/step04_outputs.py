from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


STEP04_OUTPUT_SCHEMA_VERSION = "step04_cell_fits_v2"
STEP04_ENV_PREFIX = "ASTROMODEL_STEP04_"

STEP04_DOWNSTREAM_ARTIFACTS: dict[str, str] = {
    "candidates": "cell_fit_candidates.csv",
    "accepted_ensembles": "accepted_cell_ensembles.csv",
    "quality_summary": "cell_fit_quality_summary.csv",
    "heldout_screen": "heldout_current_screen.csv",
    "acceptance_contract": "acceptance_contract.csv",
    "sweep_metrics": "candidate_sweep_metrics.csv",
    "trace_inventory": "cell_trace_inventory.csv",
    "sqlite_database": "step04_cell_fits.sqlite",
    "analysis_summary": "analysis_summary.json",
    "optimization_config": "optimization_config.json",
}


def step04_downstream_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {key: root / filename for key, filename in STEP04_DOWNSTREAM_ARTIFACTS.items()}


def missing_step04_downstream_artifacts(output_dir: str | Path) -> list[str]:
    return [
        path.name
        for path in step04_downstream_paths(output_dir).values()
        if not path.exists()
    ]


def write_step04_artifact_manifest(
    output_dir: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    payload: dict[str, Any] = {
        "step": "step04",
        "output_schema_version": STEP04_OUTPUT_SCHEMA_VERSION,
        "output_dir": str(root),
        "downstream_artifacts": STEP04_DOWNSTREAM_ARTIFACTS,
        "missing_artifacts": missing_step04_downstream_artifacts(root),
    }
    if extra:
        payload.update(dict(extra))
    (root / "step04_artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def collect_step04_environment(
    *,
    extra: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the Step 04 runtime environment needed to reproduce a run."""
    source = dict(os.environ if env is None else env)
    payload: dict[str, Any] = {
        "step04_env": {
            key: source[key]
            for key in sorted(source)
            if key.startswith(STEP04_ENV_PREFIX)
        },
        "astromodel_project_root": source.get("ASTROMODEL_PROJECT_ROOT"),
        "astromodel_data_dir": source.get("ASTROMODEL_DATA_DIR"),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def save_step04_run_snapshot(
    output_dir: str | Path,
    *,
    backup_dir: str | Path | None = None,
    label: str | None = None,
    compress: bool = True,
    include_paths: Sequence[str | Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a reproducible Step 04 snapshot and optionally archive it.

    Intended Colab use:
    - set output_dir to a Google Drive path when possible;
    - set backup_dir to a Google Drive backup folder;
    - keep compress=True for full runs.

    The snapshot captures:
    - Step 04 env vars and project/data roots;
    - required downstream artifact status;
    - optimization/config manifests already written by Step 04;
    - optional extra paths, for example the notebook file.
    """
    root = Path(output_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Step 04 output_dir does not exist: {root}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = (label or os.environ.get("ASTROMODEL_STEP04_RUN_LABEL") or root.name).replace(" ", "_")
    snapshot_name = f"{safe_label}_{timestamp}"

    backup_root: Path | None = None
    if backup_dir:
        backup_root = Path(backup_dir).expanduser().resolve()
    elif os.environ.get("ASTROMODEL_STEP04_BACKUP_DIR"):
        backup_root = Path(os.environ["ASTROMODEL_STEP04_BACKUP_DIR"]).expanduser().resolve()

    runtime_payload = collect_step04_environment(extra=extra)
    manifest_payload = {
        "step": "step04",
        "snapshot_name": snapshot_name,
        "snapshot_created_at_local": timestamp,
        "output_dir": str(root),
        "output_schema_version": STEP04_OUTPUT_SCHEMA_VERSION,
        "downstream_artifacts": STEP04_DOWNSTREAM_ARTIFACTS,
        "missing_artifacts": missing_step04_downstream_artifacts(root),
        **runtime_payload,
    }

    snapshot_json = root / "step04_run_snapshot.json"
    snapshot_json.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    copied_paths: list[str] = []
    if backup_root:
        snapshot_dir = backup_root / snapshot_name
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_if_exists(snapshot_json, snapshot_dir / snapshot_json.name)
        if copied:
            copied_paths.append(copied)
        for path_like in include_paths:
            src = Path(path_like).expanduser().resolve()
            copied = _copy_if_exists(src, snapshot_dir / src.name)
            if copied:
                copied_paths.append(copied)
        manifest_payload["backup_dir"] = str(snapshot_dir)
        manifest_payload["copied_metadata_paths"] = copied_paths

    archive_path: str | None = None
    if compress:
        archive_base = (backup_root / snapshot_name) if backup_root else (root.parent / snapshot_name)
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=root)
        manifest_payload["archive_path"] = archive_path

    # Rewrite after adding archive/backup metadata.
    snapshot_json.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if backup_root:
        _copy_if_exists(snapshot_json, backup_root / snapshot_name / snapshot_json.name)

    return manifest_payload
