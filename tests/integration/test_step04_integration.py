from __future__ import annotations
import json

def test_step04_summary_file_is_coherent(project_root):
    summary_path = project_root / 'outputs' / 'cell_fits_step04_model_aligned_demo' / 'analysis_summary.json'
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary['step_name'].startswith('Step 04')
    assert 'model_alignment' in summary
