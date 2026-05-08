from __future__ import annotations
import time
from src.step04_cell_fits import run_step04_cell_specific_six_sweep_fitting

def test_single_cell_runtime_is_practical(project_root):
    t0 = time.perf_counter()
    run_step04_cell_specific_six_sweep_fitting(project_root, output_dir=project_root / 'outputs' / 'step04_perf', selected_file_ids=['1_DH_1_CONTROL'], max_cells=1, n_fit_points=8, n_starts=1, max_nfev_all6=1, max_nfev_holdout=1)
    elapsed = time.perf_counter() - t0
    assert elapsed < 90.0
