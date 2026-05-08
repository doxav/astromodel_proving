# Step 04 validation report

Notebook executed copy:
- `outputs/executed_notebooks/04_cell_specific_multisweep_fitting.ipynb`

Runtime configuration used for the currently saved notebook outputs and machine-readable artifacts:
- `ASTROMODEL_STEP04_MAX_CELLS=6`
- `ASTROMODEL_STEP04_N_CANDIDATES=2`
- `ASTROMODEL_STEP04_RANDOM_SEED=13`
- `ASTROMODEL_STEP04_CELL_SELECTION_MODE=group_balanced`

Pytest validation executed file-by-file:
- `pytest tests/unit/test_step04_unit.py -q` → passed
- `pytest tests/acceptance/test_step04_acceptance.py -q` → passed
- `pytest tests/bootstrap/test_step04_bootstrap.py -q` → passed
- `pytest tests/integration/test_step04_integration.py -q` → passed
- `pytest tests/performance/test_step04_performance.py -q` → passed

Notes:
- The notebook integration test can still use the same notebook with a smaller runtime override when needed.
- The committed executed notebook and the current Step 04 CSV/JSON artifacts use the six-cell group-balanced configuration listed above.
