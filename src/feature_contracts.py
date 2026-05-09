"""Shared Step 02/Step 04 feature-threshold contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdScope:
    name: Literal["region_specific", "region_pooled", "global_pooled", "leave_one_cell_out"]
    exclude_file_id: str | None = None


def _feature_columns(feature_columns: Sequence[str] | None = None) -> list[str]:
    if feature_columns is None:
        raise ValueError("feature_columns must be provided")
    return list(feature_columns)


def compute_reliability_weights(feature_df: pd.DataFrame, feature_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Compute coverage/redundancy-aware reliability weights by condition/region."""

    features = _feature_columns(feature_columns)
    rows: list[dict[str, Any]] = []
    group_cols = ["region", "condition"]
    for (region, condition), group in feature_df.groupby(group_cols, dropna=False):
        numeric = group[[f for f in features if f in group.columns]].apply(pd.to_numeric, errors="coerce")
        corr = numeric.corr(method="spearman").abs() if not numeric.empty else pd.DataFrame()
        n_cells = int(group["file_id"].nunique()) if "file_id" in group.columns else int(len(group))
        small_stratum = bool(n_cells < 5)
        for feature in features:
            vals = pd.to_numeric(group[feature], errors="coerce") if feature in group.columns else pd.Series(dtype=float)
            n_rows = int(len(group))
            n_non_missing = int(vals.notna().sum())
            missing_rate = float(1.0 - n_non_missing / max(n_rows, 1))
            if feature in corr.index:
                others = corr.loc[feature].drop(labels=[feature], errors="ignore")
                max_corr = float(others.max()) if len(others) else 0.0
            else:
                max_corr = 0.0
            redundant = bool(max_corr > 0.98 and feature in {"peak_depolarization_mV", "stim_end_depolarization_mV"})
            weight = 1.0 - missing_rate
            if redundant:
                weight *= 0.5
            if small_stratum:
                weight *= 0.8
            rows.append(
                {
                    "region": region,
                    "condition": condition,
                    "feature": feature,
                    "n_rows": n_rows,
                    "n_cells": n_cells,
                    "n_non_missing": n_non_missing,
                    "missing_rate": missing_rate,
                    "completeness": 1.0 - missing_rate,
                    "max_abs_spearman": max_corr,
                    "redundant_flag": redundant,
                    "small_stratum": small_stratum,
                    "reliability_weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def build_threshold_table(
    feature_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    scope: ThresholdScope,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build an interval threshold table for a requested pooling scope."""

    features = _feature_columns(feature_columns)
    df = feature_df.copy()
    threshold_scope = scope.name
    if scope.name == "leave_one_cell_out":
        if scope.exclude_file_id is None:
            raise ValueError("leave_one_cell_out scope requires exclude_file_id")
        df = df[df["file_id"].astype(str) != str(scope.exclude_file_id)].copy()
        threshold_scope = "leave_one_cell_out_region_specific"
        group_cols = ["condition", "region", "sweep"]
    elif scope.name == "region_specific":
        group_cols = ["condition", "region", "sweep"]
    elif scope.name == "region_pooled":
        df["region"] = "ALL"
        group_cols = ["condition", "region", "sweep"]
    elif scope.name == "global_pooled":
        df["condition"] = "ALL"
        df["region"] = "ALL"
        group_cols = ["condition", "region", "sweep"]
    else:
        raise ValueError(f"Unsupported threshold scope: {scope.name}")

    rows: list[dict[str, Any]] = []
    for (condition, region, sweep), group in df.groupby(group_cols, dropna=False):
        for feature in features:
            vals = pd.to_numeric(group[feature], errors="coerce").dropna() if feature in group.columns else pd.Series(dtype=float)
            if vals.empty:
                continue
            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = float(q3 - q1)
            rel = reliability_df[(reliability_df["feature"].astype(str) == feature)]
            if region != "ALL" and "region" in rel.columns:
                rel = rel[rel["region"].astype(str) == str(region)]
            if condition != "ALL" and "condition" in rel.columns:
                rel = rel[rel["condition"].astype(str) == str(condition)]
            weight = float(rel["reliability_weight"].mean()) if not rel.empty else 1.0
            rows.append(
                {
                    "condition": condition,
                    "region": region,
                    "sweep": int(sweep),
                    "feature": feature,
                    "n_total_rows": int(group["file_id"].nunique()) if "file_id" in group.columns else int(len(group)),
                    "n_non_missing": int(vals.shape[0]),
                    "missing_rate": float(1.0 - vals.shape[0] / max(len(group), 1)),
                    "median": float(vals.median()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "acceptable_lower": q1 - 1.5 * iqr,
                    "acceptable_upper": q3 + 1.5 * iqr,
                    "threshold_scope": threshold_scope,
                    "reliability_weight": weight,
                }
            )
    return pd.DataFrame(rows).sort_values(["condition", "region", "sweep", "feature"]).reset_index(drop=True)


def _threshold_row(thresholds: pd.DataFrame, condition: str, region: str, sweep: int, feature: str) -> pd.Series | None:
    candidates = [
        (condition, region),
        (condition, "ALL"),
        ("ALL", "ALL"),
    ]
    for cond, reg in candidates:
        row = thresholds[
            (thresholds["condition"].astype(str) == str(cond))
            & (thresholds["region"].astype(str) == str(reg))
            & (thresholds["sweep"].astype(int) == int(sweep))
            & (thresholds["feature"].astype(str) == str(feature))
        ]
        if not row.empty:
            return row.iloc[0]
    return None


def score_feature_contract(
    predicted: Mapping[str, Any],
    thresholds: pd.DataFrame,
    *,
    condition: str,
    region: str,
    sweep: int,
    empirical: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] | None = None,
    pass_fraction_mode: Literal["hard", "soft"] = "hard",
) -> dict[str, Any]:
    """Score predicted features against interval thresholds."""

    total_weight = 0.0
    total_pass = 0.0
    penalties: list[float] = []
    out: dict[str, Any] = {}
    for feature in _feature_columns(feature_columns):
        row = _threshold_row(thresholds, condition, region, sweep, feature)
        if row is None:
            continue
        weight = float(row.get("reliability_weight", 1.0))
        value = float(predicted.get(feature, np.nan))
        low = float(row["acceptable_lower"])
        high = float(row["acceptable_upper"])
        within = bool(np.isfinite(value) and low <= value <= high)

        interval_width = max(high - low, 1e-6)
        iqr = float(row.get("iqr", np.nan))
        if not np.isfinite(iqr) or abs(iqr) < 1e-12:
            median = float(row.get("median", 1.0))
            iqr = max(abs(median) * 0.25, 0.5)

        if not np.isfinite(value):
            distance = 1.0
        elif within:
            distance = 0.0
        elif value < low:
            distance = float((low - value) / interval_width)
        else:
            distance = float((value - high) / interval_width)

        if pass_fraction_mode == "soft":
            if not np.isfinite(value):
                pass_credit = 0.0
            else:
                absolute_distance = 0.0 if within else min(abs(value - low), abs(value - high))
                pass_credit = max(0.0, 1.0 - absolute_distance / max(2.0 * abs(iqr), 1e-9))
        else:
            pass_credit = float(within)

        out[f"pass_{feature}"] = within
        total_weight += weight
        total_pass += weight * pass_credit
        penalties.append(weight * (1.0 - pass_credit) if pass_fraction_mode == "soft" else weight * distance)
    out["weighted_pass_fraction"] = float(total_pass / total_weight) if total_weight > 0 else 0.0
    out["weighted_feature_penalty"] = float(sum(penalties) / total_weight) if total_weight > 0 else 1.0
    out["feature_loss"] = out["weighted_feature_penalty"]
    if empirical is not None:
        plateau_match = bool(predicted.get("plateau_reached", False)) == bool(empirical.get("plateau_reached", False))
        undershoot_match = bool(predicted.get("has_undershoot", False)) == bool(empirical.get("has_undershoot", False))
        out["binary_penalty"] = float(0.5 * ((not plateau_match) + (not undershoot_match)))
    return out


def feature_residual_vector(
    predicted: Mapping[str, Any],
    empirical: Mapping[str, Any],
    thresholds: pd.DataFrame,
    *,
    condition: str,
    region: str,
    sweep: int,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Return normalized residuals for feature-aware least-squares fitting."""

    residuals: list[float] = []
    for feature in _feature_columns(feature_columns):
        row = _threshold_row(thresholds, condition, region, sweep, feature)
        if row is None:
            continue
        weight = float(row.get("reliability_weight", 1.0))
        width = float(row.get("iqr", np.nan))
        if not np.isfinite(width) or abs(width) < 1e-12:
            width = max(abs(float(row.get("acceptable_upper", 1.0)) - float(row.get("acceptable_lower", 0.0))) / 2.0, 1e-6)
        pred = predicted.get(feature, np.nan)
        emp = empirical.get(feature, np.nan)
        if np.isfinite(pred) and np.isfinite(emp):
            residuals.append(float(np.sqrt(weight) * (float(pred) - float(emp)) / max(abs(width), 1e-9)))
        else:
            residuals.append(float(np.sqrt(weight) * 3.0))
    return np.asarray(residuals, dtype=float)
