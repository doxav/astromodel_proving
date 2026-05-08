from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_step02_notebook_executes_and_writes_outputs(project_root: Path, tmp_path: Path) -> None:
    notebook_path = project_root / 'analysis' / '02_rebuild_atf_thresholds.ipynb'
    assert notebook_path.exists()

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    executor = ExecutePreprocessor(timeout=1200, kernel_name='python3')
    executor.preprocess(nb, {'metadata': {'path': str(project_root)}})

    executed_path = tmp_path / '02_rebuild_atf_thresholds.executed.ipynb'
    with open(executed_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    output_dir = project_root / 'outputs' / 'features'
    assert (output_dir / 'feature_table_by_sweep.csv').exists()
    assert (output_dir / 'condition_region_sweep_thresholds.csv').exists()
    assert (output_dir / 'feature_reliability_weights.csv').exists()
    assert (output_dir / 'region_effect_summary.csv').exists()
    assert (output_dir / 'performance_benchmark.csv').exists()
