from __future__ import annotations
from src.step04_cell_fits import run_step04_cell_specific_six_sweep_fitting

def test_representative_control_cell_becomes_reviewer_facing(project_root):
    res = run_step04_cell_specific_six_sweep_fitting(project_root, output_dir=project_root / 'outputs' / 'step04_acceptance_control', selected_file_ids=['1_DH_1_CONTROL'], max_cells=1, n_fit_points=10, n_starts=1, max_nfev_all6=2, max_nfev_holdout=1)
    summary = res['cell_fit_quality_summary'].iloc[0]
    assert bool(summary['cell_reviewer_facing'])
    assert int(summary['holdout_pass_count']) >= 3
