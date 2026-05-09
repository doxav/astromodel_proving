from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from functools import lru_cache

import numpy as np
import pandas as pd

from .atf_reference_adapter import default_reference_notebook, extract_reference_feature_table
from .provenance import EXPECTED_ATF_COUNTS, parse_atf_filename

PRIMARY_FEATURES = [
    "peak_depolarization_mV",
    "stim_end_depolarization_mV",
    "rise_slope_mV_per_s",
    "rise_tau_s",
    "plateau_slope_mV_per_s",
    "decay_slope_mV_per_s",
    "decay_tau_s",
    "undershoot_magnitude_mV",
    "return_slope_mV_per_s",
]

CONDITION_MAP = {
    "control": "CONTROL",
    "mfa": "MFA",
    "mfa_ba": "MFA_BA",
    "mfa+ba": "MFA_BA",
    "barium": "MFA_BA",
}


def _project_paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "project_root": root,
        "atf_dir": root / "data" / "2_K+ Pumps Data",
        "outputs_dir": root / "outputs" / "features",
        "reference_notebook": root / "analysis" / "astro_atf_analysis_improved_sectioned.ipynb",
    }


def canonicalize_condition(value: Any) -> str:
    text = str(value).strip().replace("-", "_").lower()
    if text not in CONDITION_MAP:
        raise ValueError(f"Unexpected condition label: {value!r}")
    return CONDITION_MAP[text]


@lru_cache(maxsize=4)
def _cached_feature_table(atf_dir_str: str, reference_notebook_str: str) -> pd.DataFrame:
    atf_dir = Path(atf_dir_str)
    reference_notebook = Path(reference_notebook_str)
    df = extract_reference_feature_table(atf_dir, reference_notebook=reference_notebook).copy()
    df["file"] = df["file"].astype(str)
    df["file_id"] = df["file"].map(lambda x: Path(x).stem)
    df["region"] = df["region"].astype(str).str.upper()
    df["condition"] = df["condition"].map(canonicalize_condition)
    df["sweep"] = df["sweep"].astype(int)
    contract = df["file_id"].map(lambda x: parse_atf_filename(f"{x}.atf"))
    df["contract_region"] = contract.map(lambda x: x.region)
    df["contract_condition"] = contract.map(lambda x: x.condition)
    if not (df["region"] == df["contract_region"]).all():
        raise ValueError("Reference notebook region parsing disagrees with filename contract")
    if not (df["condition"] == df["contract_condition"]).all():
        raise ValueError("Reference notebook condition parsing disagrees with filename contract")
    ordered_cols = ["file_id", "file", "region", "condition", "sweep"] + [c for c in df.columns if c not in {"file_id", "file", "region", "condition", "sweep"}]
    return df[ordered_cols].sort_values(["condition", "region", "file_id", "sweep"]).reset_index(drop=True)


def build_feature_table_by_sweep(atf_dir: str | Path, reference_notebook: str | Path | None = None) -> pd.DataFrame:
    reference_notebook = Path(reference_notebook) if reference_notebook is not None else default_reference_notebook(Path(atf_dir).parents[1] if len(Path(atf_dir).parents) > 1 else None)
    return _cached_feature_table(str(Path(atf_dir).resolve()), str(reference_notebook.resolve())).copy()


def region_condition_cell_counts(feature_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        feature_df.groupby(["region", "condition"], dropna=False)["file_id"]
        .nunique()
        .rename("n_cells")
        .reset_index()
        .sort_values(["region", "condition"])
        .reset_index(drop=True)
    )
    observed_map = {(r.region, r.condition): int(r.n_cells) for r in counts.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for (region, condition), expected in EXPECTED_ATF_COUNTS.items():
        observed = observed_map.get((region, condition), 0)
        rows.append(
            {
                "region": region,
                "condition": condition,
                "n_cells": observed,
                "expected_n_cells": expected,
                "matches_expected": bool(observed == expected),
                "small_stratum": bool(observed < 5),
            }
        )
    return pd.DataFrame(rows).sort_values(["region", "condition"]).reset_index(drop=True)


def _pairwise_spearman(feature_df: pd.DataFrame, feature_a: str, feature_b: str) -> float:
    sub = feature_df[[feature_a, feature_b]].dropna()
    if len(sub) < 3:
        return np.nan
    return float(sub[feature_a].corr(sub[feature_b], method="spearman"))


def compute_redundancy_diagnostics(feature_df: pd.DataFrame, features: Iterable[str] = PRIMARY_FEATURES) -> pd.DataFrame:
    features = list(features)
    rows: list[dict[str, Any]] = []
    for i, fa in enumerate(features):
        for fb in features[i + 1:]:
            sub = feature_df[[fa, fb]].dropna()
            if len(sub) >= 3:
                pearson = float(sub[fa].corr(sub[fb], method="pearson"))
                spearman = float(sub[fa].corr(sub[fb], method="spearman"))
            else:
                pearson = np.nan
                spearman = np.nan
            rows.append(
                {
                    "feature_a": fa,
                    "feature_b": fb,
                    "n_complete_pairs": int(len(sub)),
                    "pearson_r": pearson,
                    "spearman_r": spearman,
                    "abs_spearman_r": abs(spearman) if np.isfinite(spearman) else np.nan,
                    "redundant_flag": bool(np.isfinite(spearman) and abs(spearman) >= 0.95),
                }
            )
    return pd.DataFrame(rows).sort_values(["redundant_flag", "abs_spearman_r"], ascending=[False, False]).reset_index(drop=True)


def compute_feature_reliability_weights(
    feature_df: pd.DataFrame,
    cell_counts: pd.DataFrame,
    redundancy: pd.DataFrame,
    features: Iterable[str] = PRIMARY_FEATURES,
) -> pd.DataFrame:
    features = list(features)
    redundancy_lookup: dict[str, float] = {}
    for feature in features:
        mask = (redundancy["feature_a"] == feature) | (redundancy["feature_b"] == feature)
        redundancy_lookup[feature] = float(redundancy.loc[mask, "abs_spearman_r"].max()) if mask.any() else np.nan

    count_lookup = {(r.region, r.condition): bool(r.small_stratum) for r in cell_counts.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    grouped = feature_df.groupby(["region", "condition"], dropna=False)
    for (region, condition), group in grouped:
        n_rows = len(group)
        n_cells = int(group["file_id"].nunique())
        small_stratum = count_lookup.get((region, condition), False)
        for feature in features:
            n_non_missing = int(group[feature].notna().sum())
            missing_rate = float(1.0 - n_non_missing / max(n_rows, 1))
            completeness = float(1.0 - missing_rate)
            max_abs_spearman = redundancy_lookup.get(feature, np.nan)
            redundancy_penalty = 0.5 if (np.isfinite(max_abs_spearman) and max_abs_spearman >= 0.95) else 1.0
            small_stratum_penalty = 0.8 if small_stratum else 1.0
            reliability_weight = float(completeness * redundancy_penalty * small_stratum_penalty)
            rows.append(
                {
                    "region": region,
                    "condition": condition,
                    "feature": feature,
                    "n_rows": n_rows,
                    "n_cells": n_cells,
                    "n_non_missing": n_non_missing,
                    "missing_rate": missing_rate,
                    "completeness": completeness,
                    "max_abs_spearman": max_abs_spearman,
                    "redundant_flag": bool(np.isfinite(max_abs_spearman) and max_abs_spearman >= 0.95),
                    "small_stratum": bool(small_stratum),
                    "reliability_weight": reliability_weight,
                }
            )
    return pd.DataFrame(rows).sort_values(["condition", "region", "feature"]).reset_index(drop=True)


def compute_condition_feature_reliability(reliability_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        reliability_df.groupby(["condition", "feature"], dropna=False)
        .agg(
            n_regions=("region", "nunique"),
            mean_missing_rate=("missing_rate", "mean"),
            mean_completeness=("completeness", "mean"),
            mean_reliability_weight=("reliability_weight", "mean"),
            any_small_stratum=("small_stratum", "max"),
        )
        .reset_index()
        .sort_values(["condition", "feature"])
        .reset_index(drop=True)
    )
    return grouped


def _threshold_rows_from_group(
    group: pd.DataFrame,
    group_cols: list[str],
    threshold_scope: str,
    reliability_df: pd.DataFrame,
    features: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_total = len(group)
    for feature in features:
        series = pd.to_numeric(group[feature], errors="coerce")
        valid = series.dropna()
        q1 = float(valid.quantile(0.25)) if not valid.empty else np.nan
        median = float(valid.median()) if not valid.empty else np.nan
        q3 = float(valid.quantile(0.75)) if not valid.empty else np.nan
        iqr = float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else np.nan
        lower = float(q1 - 1.5 * iqr) if np.isfinite(iqr) else np.nan
        upper = float(q3 + 1.5 * iqr) if np.isfinite(iqr) else np.nan
        out = {col: group.iloc[0][col] for col in group_cols}
        out.update(
            {
                "feature": feature,
                "n_total_rows": n_total,
                "n_non_missing": int(valid.shape[0]),
                "missing_rate": float(1.0 - valid.shape[0] / max(n_total, 1)),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "acceptable_lower": lower,
                "acceptable_upper": upper,
                "threshold_scope": threshold_scope,
            }
        )
        if threshold_scope == "region_specific":
            match = reliability_df[
                (reliability_df["condition"] == out["condition"])
                & (reliability_df["region"] == out["region"])
                & (reliability_df["feature"] == feature)
            ]
        elif threshold_scope == "region_pooled":
            match = reliability_df[
                (reliability_df["condition"] == out["condition"])
                & (reliability_df["feature"] == feature)
            ]
        else:
            match = reliability_df[reliability_df["feature"] == feature]
        out["reliability_weight"] = float(match["reliability_weight"].mean()) if not match.empty else np.nan
        rows.append(out)
    return rows


def compute_thresholds(
    feature_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    features: Iterable[str] = PRIMARY_FEATURES,
    threshold_scope: str = "region_specific",
) -> pd.DataFrame:
    features = list(features)
    if threshold_scope == "region_specific":
        group_cols = ["condition", "region", "sweep"]
        grouped = feature_df.groupby(group_cols, dropna=False)
    elif threshold_scope == "region_pooled":
        pooled = feature_df.copy()
        pooled["region"] = "ALL"
        group_cols = ["condition", "region", "sweep"]
        grouped = pooled.groupby(group_cols, dropna=False)
    elif threshold_scope == "global_pooled":
        pooled = feature_df.copy()
        pooled["condition"] = "ALL"
        pooled["region"] = "ALL"
        group_cols = ["condition", "region", "sweep"]
        grouped = pooled.groupby(group_cols, dropna=False)
    else:
        raise ValueError(f"Unsupported threshold_scope={threshold_scope!r}")

    rows: list[dict[str, Any]] = []
    for _, group in grouped:
        rows.extend(_threshold_rows_from_group(group, group_cols, threshold_scope, reliability_df, features))
    return pd.DataFrame(rows).sort_values(group_cols + ["feature"]).reset_index(drop=True)


def _bootstrap_difference(x_vh: np.ndarray, x_dh: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vh = rng.choice(x_vh, size=len(x_vh), replace=True)
        dh = rng.choice(x_dh, size=len(x_dh), replace=True)
        boot[i] = np.nanmedian(vh) - np.nanmedian(dh)
    return float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))


def compute_region_effect_summary(feature_df: pd.DataFrame, features: Iterable[str] = PRIMARY_FEATURES) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in sorted(feature_df["condition"].unique()):
        for sweep in sorted(feature_df["sweep"].unique()):
            sub = feature_df[(feature_df["condition"] == condition) & (feature_df["sweep"] == sweep)]
            for feature in features:
                dh = pd.to_numeric(sub.loc[sub["region"] == "DH", feature], errors="coerce").dropna().to_numpy(dtype=float)
                vh = pd.to_numeric(sub.loc[sub["region"] == "VH", feature], errors="coerce").dropna().to_numpy(dtype=float)
                if len(dh) and len(vh):
                    median_dh = float(np.nanmedian(dh))
                    median_vh = float(np.nanmedian(vh))
                    diff = float(median_vh - median_dh)
                    ci_low, ci_high = _bootstrap_difference(vh, dh, n_boot=500)
                else:
                    median_dh = np.nan if len(dh) == 0 else float(np.nanmedian(dh))
                    median_vh = np.nan if len(vh) == 0 else float(np.nanmedian(vh))
                    diff = np.nan
                    ci_low = np.nan
                    ci_high = np.nan
                rows.append(
                    {
                        "condition": condition,
                        "sweep": int(sweep),
                        "feature": feature,
                        "n_dh": int(len(dh)),
                        "n_vh": int(len(vh)),
                        "median_dh": median_dh,
                        "median_vh": median_vh,
                        "vh_minus_dh_median": diff,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "small_stratum": bool(min(len(dh), len(vh)) < 5),
                    }
                )
    return pd.DataFrame(rows).sort_values(["condition", "feature", "sweep"]).reset_index(drop=True)


def run_step02_rebuild_atf_thresholds(project_root: str | Path, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    paths = _project_paths(project_root)
    outputs_dir = Path(output_dir) if output_dir is not None else paths["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    feature_df = build_feature_table_by_sweep(paths["atf_dir"], reference_notebook=paths["reference_notebook"])
    cell_counts = region_condition_cell_counts(feature_df)
    redundancy = compute_redundancy_diagnostics(feature_df)
    reliability = compute_feature_reliability_weights(feature_df, cell_counts, redundancy)
    condition_reliability = compute_condition_feature_reliability(reliability)
    thresholds = compute_thresholds(feature_df, reliability, threshold_scope="region_specific")
    region_pooled = compute_thresholds(feature_df, reliability, threshold_scope="region_pooled")
    global_pooled = compute_thresholds(feature_df, reliability, threshold_scope="global_pooled")
    region_effects = compute_region_effect_summary(feature_df)

    feature_df.to_csv(outputs_dir / "feature_table_by_sweep.csv", index=False)
    cell_counts.to_csv(outputs_dir / "region_condition_cell_counts.csv", index=False)
    redundancy.to_csv(outputs_dir / "redundancy_diagnostics.csv", index=False)
    reliability.to_csv(outputs_dir / "feature_reliability_weights.csv", index=False)
    condition_reliability.to_csv(outputs_dir / "condition_feature_reliability.csv", index=False)
    thresholds.to_csv(outputs_dir / "condition_region_sweep_thresholds.csv", index=False)
    region_pooled.to_csv(outputs_dir / "region_pooled_condition_sweep_thresholds.csv", index=False)
    global_pooled.to_csv(outputs_dir / "global_pooled_sweep_thresholds.csv", index=False)
    region_effects.to_csv(outputs_dir / "region_effect_summary.csv", index=False)

    return {
        "feature_table_by_sweep": feature_df,
        "region_condition_cell_counts": cell_counts,
        "redundancy_diagnostics": redundancy,
        "feature_reliability_weights": reliability,
        "condition_feature_reliability": condition_reliability,
        "condition_region_sweep_thresholds": thresholds,
        "region_pooled_condition_sweep_thresholds": region_pooled,
        "global_pooled_sweep_thresholds": global_pooled,
        "region_effect_summary": region_effects,
    }
