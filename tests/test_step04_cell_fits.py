from __future__ import annotations
from src.step04_cell_fits import build_cell_trace_inventory, run_step04_cell_specific_six_sweep_fitting

def test_single_control_cell_fit_writes_expected_outputs(project_root):
    inv = build_cell_trace_inventory(project_root / 'data' / '2_K+ Pumps Data', n_fit_points=10, file_ids=['1_DH_1_CONTROL'])
    assert list(inv) == ['1_DH_1_CONTROL']
    assert len(inv['1_DH_1_CONTROL']) == 6
    out = project_root / 'outputs' / 'step04_test_single'
    res = run_step04_cell_specific_six_sweep_fitting(project_root, output_dir=out, selected_file_ids=['1_DH_1_CONTROL'], max_cells=1, n_fit_points=10, n_starts=1, max_nfev_all6=1, max_nfev_holdout=1)
    assert (out / 'cell_fit_candidates.csv').exists()
    assert (out / 'accepted_cell_ensembles.csv').exists()
    assert len(res['heldout_current_screen']) == 6
