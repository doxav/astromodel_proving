from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


def _ensure_pythonpath(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


@pytest.fixture(scope="session")
def project_root() -> Path:
    env = os.environ.get("ASTROMODEL_PROJECT_ROOT")
    root = Path(env).resolve() if env else Path(__file__).resolve().parents[1]
    _ensure_pythonpath(root)
    return root


@pytest.fixture(scope="session")
def atf_dir(project_root: Path) -> Path:
    return project_root / "data" / "2_K+ Pumps Data"


@pytest.fixture(scope="session")
def threshold_csv(project_root: Path) -> Path:
    return project_root / "data" / "threshold_for_good_enough_fits.csv"


from src.step02_thresholds import run_step02_rebuild_atf_thresholds


def _load_step02_outputs(output_dir: Path):
    from src.step02_thresholds import project_paths

    with open(output_dir / 'analysis_summary.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)
    results = {
        'feature_table_by_sweep': pd.read_csv(output_dir / 'feature_table_by_sweep.csv'),
        'preprocess_qc_by_sweep': pd.read_csv(output_dir / 'preprocess_qc_by_sweep.csv'),
        'region_condition_cell_counts': pd.read_csv(output_dir / 'region_condition_cell_counts.csv'),
        'condition_region_sweep_thresholds': pd.read_csv(output_dir / 'condition_region_sweep_thresholds.csv'),
        'feature_reliability_weights': pd.read_csv(output_dir / 'feature_reliability_weights.csv'),
        'feature_correlation_summary': pd.read_csv(output_dir / 'feature_correlation_summary.csv'),
        'region_effect_summary': pd.read_csv(output_dir / 'region_effect_summary.csv'),
        'atf_inventory': pd.read_csv(output_dir / 'atf_region_condition_inventory.csv'),
        'legacy_threshold_preview': pd.read_csv(output_dir / 'legacy_threshold_preview.csv'),
        'analysis_summary': summary,
        'paths': project_paths(Path(output_dir).parents[2], output_dir=output_dir),
    }
    perf_path = output_dir / 'performance_benchmark.csv'
    if perf_path.exists():
        results['performance_benchmark'] = pd.read_csv(perf_path)
    return results


@pytest.fixture(scope="session")
def step02_results(project_root: Path):
    output_dir = project_root / 'outputs' / 'features'
    if (output_dir / 'feature_table_by_sweep.csv').exists():
        return _load_step02_outputs(output_dir)
    return run_step02_rebuild_atf_thresholds(project_root, output_dir=output_dir, build_benchmark=False)


@pytest.fixture(scope="session")
def step02_results_with_benchmark(project_root: Path):
    output_dir = project_root / 'outputs' / 'features'
    if (output_dir / 'performance_benchmark.csv').exists():
        return _load_step02_outputs(output_dir)
    return run_step02_rebuild_atf_thresholds(project_root, output_dir=output_dir, build_benchmark=True)
