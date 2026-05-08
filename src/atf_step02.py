from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .legacy_runtime import load_filtered_runtime

SWEEP_CURRENTS_NA: Tuple[int, ...] = (50, 75, 100, 125, 150, 175)
FEATURE_COLUMNS: Tuple[str, ...] = (
    "peak_depolarization_mV",
    "rise_slope_mV_per_s",
    "rise_tau_s",
    "plateau_slope_mV_per_s",
    "decay_slope_mV_per_s",
    "decay_tau_s",
    "undershoot_magnitude_mV",
    "return_slope_mV_per_s",
)


@dataclass(frozen=True)
class AtfCellContract:
    file_id: str
    file_name: str
    region: str
    condition: str


def parse_atf_contract(file_name: str) -> AtfCellContract:
    stem = file_name.replace(".atf", "")
    if "DH" in stem:
        region = "DH"
    elif "VH" in stem:
        region = "VH"
    else:
        raise ValueError(f"Unknown region in ATF filename: {file_name}")

    upper = stem.upper()
    if "MFA_BA" in upper or "MFA_BA" in upper or "MFA_BA" in upper:
        condition = "MFA_BA"
    elif "MFA_BA" in upper or "MFA_BA" in upper:
        condition = "MFA_BA"
    elif "MFA_BA" in stem or "MFA_Ba" in stem:
        condition = "MFA_BA"
    elif "MFA" in upper:
        condition = "MFA"
    else:
        condition = "CONTROL"
    return AtfCellContract(file_id=stem, file_name=file_name, region=region, condition=condition)


def parse_ipatch_r_columns(atf_path: Path) -> List[int]:
    lines = atf_path.read_text(encoding="utf-8", errors="replace").splitlines()
    signal_line_index = next(i for i, line in enumerate(lines) if line.startswith('"Signals="'))
    tokens = [token.strip().strip('"') for token in lines[signal_line_index].split("	")]
    signal_names = tokens[1:]
    columns = [idx + 1 for idx, name in enumerate(signal_names) if name == "Ipatch_R"]
    if len(columns) != 6:
        raise ValueError(f"Expected 6 Ipatch_R columns in {atf_path.name}, found {len(columns)}")
    return columns


def load_cell_sweeps(atf_path: Path) -> List[Dict[str, object]]:
    rt = load_filtered_runtime()
    data = rt["load_atf_file_numeric"](atf_path)
    contract = parse_atf_contract(atf_path.name)
    columns = parse_ipatch_r_columns(atf_path)
    sweeps: List[Dict[str, object]] = []
    for sweep_index, (current_na, column_index) in enumerate(zip(SWEEP_CURRENTS_NA, columns), start=1):
        sweeps.append(
            {
                "file_id": contract.file_id,
                "file": contract.file_name,
                "region": contract.region,
                "condition": contract.condition,
                "sweep": sweep_index,
                "current_na": current_na,
                "time_ms": data[:, 0].copy(),
                "time_s": data[:, 0].copy() / 1000.0,
                "voltage_mV": data[:, column_index].copy(),
                "signal_column_index": column_index,
            }
        )
    return sweeps


def build_feature_table(atf_dir: Path) -> pd.DataFrame:
    rt = load_filtered_runtime()
    rows: List[Dict[str, object]] = []
    for atf_path in sorted(atf_dir.glob("*.atf")):
        for sweep in load_cell_sweeps(atf_path):
            features = rt["extract_features_from_trace"](sweep["time_s"], sweep["voltage_mV"])
            row = {
                "file_id": sweep["file_id"],
                "file": sweep["file"],
                "region": sweep["region"],
                "condition": sweep["condition"],
                "sweep": sweep["sweep"],
                "current_na": sweep["current_na"],
            }
            row.update(features)
            peak_dep = float(features.get("peak_depolarization_mV", np.nan))
            baseline = float(features.get("baseline_mV", np.nan)) if np.isfinite(features.get("baseline_mV", np.nan)) else np.nan
            plateau_level = float(features.get("plateau_level_mV", np.nan)) if np.isfinite(features.get("plateau_level_mV", np.nan)) else np.nan
            if np.isfinite(plateau_level) and np.isfinite(baseline) and np.isfinite(peak_dep):
                row["plateau_reached"] = bool((plateau_level - baseline) >= 0.7 * peak_dep)
            else:
                row["plateau_reached"] = False
            row["has_undershoot"] = bool(features.get("undershoot_magnitude_mV", np.nan) > 0)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["region", "condition", "file_id", "sweep"]).reset_index(drop=True)


def build_region_condition_cell_counts(feature_table: pd.DataFrame) -> pd.DataFrame:
    expected = {
        ("DH", "CONTROL"): 7,
        ("VH", "CONTROL"): 4,
        ("DH", "MFA"): 6,
        ("VH", "MFA"): 7,
        ("DH", "MFA_BA"): 6,
        ("VH", "MFA_BA"): 7,
    }
    rows = []
    grouped = feature_table.groupby(["region", "condition"])["file_id"].nunique().reset_index(name="n_cells")
    for _, row in grouped.iterrows():
        exp = expected.get((row["region"], row["condition"]), np.nan)
        rows.append(
            {
                "region": row["region"],
                "condition": row["condition"],
                "n_cells": int(row["n_cells"]),
                "expected_n_cells": exp,
                "matches_expected": bool(row["n_cells"] == exp) if pd.notna(exp) else False,
                "small_stratum": bool(row["n_cells"] <= 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["region", "condition"]).reset_index(drop=True)


def build_feature_reliability_weights(feature_table: pd.DataFrame) -> pd.DataFrame:
    counts = build_region_condition_cell_counts(feature_table).set_index(["region", "condition"])
    rows: List[Dict[str, object]] = []
    for (region, condition), group in feature_table.groupby(["region", "condition"]):
        corr = group[list(FEATURE_COLUMNS)].corr(method="spearman").abs().fillna(0.0)
        for feature in FEATURE_COLUMNS:
            series = pd.to_numeric(group[feature], errors="coerce")
            n_non_missing = int(series.notna().sum())
            missing_rate = 1.0 - float(n_non_missing / max(len(group), 1))
            max_corr = 0.0
            if feature in corr.index:
                max_corr = float(corr.loc[feature].drop(feature).max())
            redundant = max_corr >= 0.98
            small_stratum = bool(counts.loc[(region, condition), "small_stratum"])
            reliability = (1.0 - missing_rate) * (0.5 if redundant else 1.0) * (0.8 if small_stratum else 1.0)
            rows.append(
                {
                    "region": region,
                    "condition": condition,
                    "feature": feature,
                    "n_rows": int(len(group)),
                    "n_cells": int(group["file_id"].nunique()),
                    "n_non_missing": n_non_missing,
                    "missing_rate": missing_rate,
                    "completeness": 1.0 - missing_rate,
                    "max_abs_spearman": max_corr,
                    "redundant_flag": redundant,
                    "small_stratum": small_stratum,
                    "reliability_weight": float(reliability),
                }
            )
    return pd.DataFrame(rows).sort_values(["region", "condition", "feature"]).reset_index(drop=True)


def build_condition_region_sweep_thresholds(
    feature_table: pd.DataFrame,
    reliability_weights: pd.DataFrame,
) -> pd.DataFrame:
    reliability_lookup = reliability_weights.set_index(["region", "condition", "feature"])['reliability_weight'].to_dict()
    rows: List[Dict[str, object]] = []
    for (condition, region, sweep), group in feature_table.groupby(["condition", "region", "sweep"]):
        for feature in FEATURE_COLUMNS:
            series = pd.to_numeric(group[feature], errors="coerce")
            valid = series.dropna()
            q1 = float(valid.quantile(0.25)) if len(valid) else np.nan
            q3 = float(valid.quantile(0.75)) if len(valid) else np.nan
            iqr = q3 - q1 if pd.notna(q1) and pd.notna(q3) else np.nan
            rows.append(
                {
                    "condition": condition,
                    "region": region,
                    "sweep": int(sweep),
                    "feature": feature,
                    "n_total_rows": int(len(group)),
                    "n_non_missing": int(valid.shape[0]),
                    "missing_rate": 1.0 - float(valid.shape[0] / max(len(group), 1)),
                    "median": float(valid.median()) if len(valid) else np.nan,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "acceptable_lower": float(q1 - 1.5 * iqr) if pd.notna(iqr) else np.nan,
                    "acceptable_upper": float(q3 + 1.5 * iqr) if pd.notna(iqr) else np.nan,
                    "threshold_scope": "region_specific",
                    "reliability_weight": float(reliability_lookup[(region, condition, feature)]),
                }
            )
    return pd.DataFrame(rows).sort_values(["condition", "region", "sweep", "feature"]).reset_index(drop=True)
