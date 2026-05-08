from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import numpy as np
import pandas as pd

CURRENTS_NA = [50, 75, 100, 125, 150, 175]


@dataclass(frozen=True)
class SweepTrace:
    file_id: str
    file_name: str
    region: str
    condition: str
    sweep: int
    current_na: int
    time_s: np.ndarray
    vm_mV: np.ndarray
    current_nA: np.ndarray


@dataclass(frozen=True)
class CellProtocol:
    file_id: str
    file_name: str
    region: str
    condition: str
    sweeps: tuple[SweepTrace, ...]


def infer_region(filename: str) -> str:
    name = filename.upper()
    if name.startswith("DH") or "_DH_" in name or name.startswith("DH_"):
        return "DH"
    if name.startswith("VH") or "_VH_" in name or name.startswith("VH_"):
        return "VH"
    raise ValueError(f"Cannot infer region from {filename!r}")


def infer_condition(filename: str) -> str:
    name = filename.upper()
    if "_MFA_BA" in name or "_MFA_BA." in name or "_MFA_BA_" in name or "_MFA_BA" in name:
        return "MFA_BA"
    if "_MFA_BA" not in name and ("_MFA" in name or name.endswith("MFA.ATF") or name.startswith("MFA_")):
        return "MFA"
    return "CONTROL"


def canonical_file_id(path: Path) -> str:
    stem = path.stem
    region = infer_region(stem)
    condition = infer_condition(stem)
    if condition == "CONTROL" and not stem.upper().endswith("_CONTROL"):
        return f"{stem}_{condition}"
    return stem.replace("Ba", "BA")


def _header_line_index(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith('"Time (s)"'):
                return i
    raise ValueError(f"Could not locate ATF data header in {path}")


def read_atf_dataframe(path: Path) -> pd.DataFrame:
    hdr = _header_line_index(path)
    df = pd.read_csv(path, sep="\t", skiprows=hdr)
    # Normalize potential duplicated column names emitted by pandas.
    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    return df


def iter_sweeps_from_atf(path: Path) -> Iterator[SweepTrace]:
    path = Path(path)
    region = infer_region(path.name)
    condition = infer_condition(path.name)
    file_id = canonical_file_id(path)
    df = read_atf_dataframe(path)
    time_s = df.iloc[:, 0].astype(float).to_numpy()
    n_signal_cols = df.shape[1] - 1
    if n_signal_cols % 6 != 0:
        raise ValueError(f"Unexpected ATF column count for {path}: {df.shape[1]}")
    signals_per_sweep = n_signal_cols // 6
    for sweep_idx, current_na in enumerate(CURRENTS_NA, start=1):
        vm_col = 1 + signals_per_sweep * (sweep_idx - 1)
        current_col = vm_col + 2 if signals_per_sweep >= 3 else None
        current = df.iloc[:, current_col].astype(float).to_numpy() if current_col is not None else np.zeros_like(time_s)
        yield SweepTrace(
            file_id=file_id,
            file_name=path.name,
            region=region,
            condition=condition,
            sweep=sweep_idx,
            current_na=current_na,
            time_s=time_s.copy(),
            vm_mV=df.iloc[:, vm_col].astype(float).to_numpy(),
            current_nA=current,
        )


def load_cell_protocol(path: Path) -> CellProtocol:
    sweeps = tuple(iter_sweeps_from_atf(path))
    if len(sweeps) != 6:
        raise ValueError(f"Expected 6 sweeps in {path}, found {len(sweeps)}")
    first = sweeps[0]
    return CellProtocol(
        file_id=first.file_id,
        file_name=first.file_name,
        region=first.region,
        condition=first.condition,
        sweeps=sweeps,
    )


def load_all_cells(atf_dir: Path) -> list[CellProtocol]:
    paths = sorted(Path(atf_dir).glob("*.atf"))
    return [load_cell_protocol(path) for path in paths]
