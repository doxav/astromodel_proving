from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.identifiability import Step03Config, compute_fim_tables_with_diagnostics
from src.optuna_sqlite import read_best_trial


def test_step03_fim_step_size_tuning_table(project_root: Path, initial_fit_dir: Path) -> None:
    record = read_best_trial(initial_fit_dir / "CONTROL_75nA.db")
    rows = []
    for step in [0.02, 0.05]:
        config = Step03Config(n_timepoints=70, observable_stride=2, fim_log_step=step)
        started = perf_counter()
        spectrum, _loadings, diagnostics = compute_fim_tables_with_diagnostics(record, config)
        elapsed = perf_counter() - started
        rows.append(
            {
                "fim_log_step": step,
                "elapsed_seconds": elapsed,
                "n_modes": len(spectrum),
                "finite_eigenvalues": bool(np.isfinite(spectrum["eigenvalue"]).all()),
                "condition_number_estimate": float(spectrum["condition_number_estimate"].iloc[0]),
                "near_zero_mode_count": int(diagnostics["near_zero_mode_count"].iloc[0]),
                "n_stiff_modes": int((spectrum["mode_class"] == "stiff").sum()),
            }
        )
    tuning = pd.DataFrame(rows)
    output_dir = project_root / "outputs" / "identifiability"
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(output_dir / "fim_step_size_tuning.csv", index=False)

    assert len(tuning) == 2
    assert tuning["finite_eigenvalues"].all()
    assert tuning["elapsed_seconds"].max() < 20.0
    assert (tuning["n_modes"] >= 5).all()
