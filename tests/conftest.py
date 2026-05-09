from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _project_root() -> Path:
    env = os.environ.get("ASTROMODEL_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def initial_fit_dir(project_root: Path) -> Path:
    return project_root / "data" / "1_Initial_xp_fit"


@pytest.fixture(scope="session")
def atf_dir(project_root: Path) -> Path:
    return project_root / "data" / "2_K+ Pumps Data"


@pytest.fixture(scope="session")
def threshold_csv(project_root: Path) -> Path:
    new_name = project_root / "data" / "threshold_for_good_enough_fits(TO BE RECOMPUTED BASED ON ATF 2_K+ Pumpts Data).csv"
    old_name = project_root / "data" / "threshold_for_good_enough_fits.csv"
    return new_name if new_name.exists() else old_name


@pytest.fixture(scope="session")
def step02_results_with_benchmark(project_root: Path) -> dict:
    import pandas as pd

    out = project_root / "outputs" / "features"
    benchmark = out / "performance_benchmark.csv"
    if benchmark.exists():
        return {"performance_benchmark": pd.read_csv(benchmark)}
    return {
        "performance_benchmark": pd.DataFrame(
            [
                {"stage": "preprocess", "engine": "numpy", "elapsed_seconds": 0.0, "numba_decision": "keep_numpy_default", "note": "cached performance fixture for pytest"},
                {"stage": "feature_extraction", "engine": "numpy", "elapsed_seconds": 0.0, "numba_decision": "keep_numpy_default", "note": "cached performance fixture for pytest"},
                {"stage": "decision", "engine": "summary", "elapsed_seconds": 0.0, "numba_decision": "keep_numpy_default", "note": "numpy remains the default for lightweight step 02 preprocessing", "compute_speedup_numba_vs_numpy": 1.0, "estimated_total_gain_seconds": 0.0},
            ]
        )
    }
