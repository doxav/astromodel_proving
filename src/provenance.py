from __future__ import annotations

from pathlib import Path
import pandas as pd


def run_step00_provenance(project_root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(project_root).resolve()
    path = root / 'outputs' / 'provenance' / 'control_trace_verification.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing Step 00 output: {path}')
    df = pd.read_csv(path)
    return {'control_trace_verification': df}
