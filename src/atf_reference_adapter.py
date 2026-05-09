from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import io
from contextlib import redirect_stdout

import nbformat
import pandas as pd


REFERENCE_MATCHERS = [
    "from pathlib import Path",
    "DATA_SOURCE =",
    "def running_in_colab",
    "def moving_average(",
    "def correct_brief_jump_artifacts",
    "def moving_average_nan",
    "def extract_features(",
]


def default_reference_notebook(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    return Path(project_root) / "analysis" / "astro_atf_analysis_improved_sectioned.ipynb"


def _clean_cell5_source(source: str) -> str:
    lines = source.splitlines()
    cleaned: list[str] = []
    skip_exact = {
        "DATA_DIR = resolve_data_dir()",
        "OUTPUT_DIR = DATA_DIR / \"astro_feature_outputs_expanded\"",
        "RAW_INSPECTION_DIR = OUTPUT_DIR / \"raw_inspection\"",
        "PREPROC_REVIEW_DIR = OUTPUT_DIR / \"preprocessing_review\"",
        "print(\"Using data folder:\", DATA_DIR)",
        "print(\"Output folder:\", OUTPUT_DIR)",
    }
    skip_next_indented = False
    for line in lines:
        if line in skip_exact:
            continue
        if line.startswith("for folder in [OUTPUT_DIR, RAW_INSPECTION_DIR, PREPROC_REVIEW_DIR]:"):
            skip_next_indented = True
            continue
        if skip_next_indented and line.startswith("    "):
            continue
        skip_next_indented = False
        cleaned.append(line)
    return "\n".join(cleaned)


def _select_reference_sources(nb) -> list[str]:
    selected: list[str] = []
    for marker in REFERENCE_MATCHERS:
        match_source = None
        for cell in nb.cells:
            if cell.cell_type != "code" or marker not in cell.source:
                continue
            if marker == "from pathlib import Path" and "import numpy as np" not in cell.source:
                continue
            match_source = cell.source
            break
        if match_source is None:
            raise RuntimeError(f"Could not find reference notebook cell for marker: {marker}")
        selected.append(_clean_cell5_source(match_source) if marker == "def running_in_colab" else match_source)
    return selected


@lru_cache(maxsize=4)
def load_reference_namespace(reference_notebook: str | Path) -> Dict[str, Any]:
    path = Path(reference_notebook)
    nb = nbformat.read(path, as_version=4)
    ns: Dict[str, Any] = {}
    for source in _select_reference_sources(nb):
        exec(source, ns)
    return ns


def extract_reference_feature_table(
    atf_dir: str | Path,
    reference_notebook: str | Path | None = None,
) -> pd.DataFrame:
    atf_dir = Path(atf_dir)
    if reference_notebook is None:
        reference_notebook = default_reference_notebook(atf_dir.parents[1] if len(atf_dir.parents) > 1 else None)
    ns = load_reference_namespace(str(reference_notebook))
    files = ns["discover_atf_files"](atf_dir)
    rows: list[pd.DataFrame] = []
    for path in files:
        with redirect_stdout(io.StringIO()):
            parsed = ns["parse_atf"](path)
            parsed = ns["preprocess_parsed"](parsed)
            feature_df = ns["extract_features"](parsed)
        if feature_df is not None and not feature_df.empty:
            rows.append(feature_df)
    if not rows:
        raise RuntimeError(f"No ATF features extracted from {atf_dir}")
    return pd.concat(rows, ignore_index=True)
