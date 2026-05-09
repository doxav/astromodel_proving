"""Step 02 orchestration: ATF feature thresholds, reliability, region effects, and benchmarking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .atf_features import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CONDITIONAL_CONTINUOUS_FEATURES,
    FEATURE_FAMILY,
    FEATURE_PRIORITY,
    MANUAL_TRANSIENT_ARTIFACT_HINTS,
    PRIMARY_CONTINUOUS_FEATURES,
    build_atf_inventory,
    count_region_condition_cells,
    extract_feature_tables,
    extract_features,
    load_preprocessed_atf_files,
)

ALLOWED_THRESHOLD_SCOPES = ("region_specific", "region_pooled", "global_pooled")
CORRELATION_REDUNDANCY_THRESHOLD = 0.98
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class Step02Paths:
    project_root: Path
    atf_dir: Path
    threshold_csv: Path
    output_dir: Path


def project_paths(project_root: str | Path, output_dir: str | Path | None = None) -> Step02Paths:
    root = Path(project_root).resolve()
    atf_dir = root / "data" / "2_K+ Pumps Data"
    threshold_csv = root / "data" / "threshold_for_good_enough_fits.csv"
    out = Path(output_dir).resolve() if output_dir is not None else root / "outputs" / "features"
    out.mkdir(parents=True, exist_ok=True)
    return Step02Paths(project_root=root, atf_dir=atf_dir, threshold_csv=threshold_csv, output_dir=out)


def _feature_family(feature_name: str) -> str:
    return FEATURE_FAMILY.get(feature_name, "other")


def _describe_continuous(x: pd.Series) -> dict[str, float]:
    x = x.dropna().astype(float)
    n = int(len(x))
    if n == 0:
        return {
            "n_nonmissing": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else np.nan
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = float(q3 - q1)
    sem = std / np.sqrt(n) if n > 1 and np.isfinite(std) else np.nan
    ci95_low = mean - 1.96 * sem if np.isfinite(sem) else mean
    ci95_high = mean + 1.96 * sem if np.isfinite(sem) else mean
    return {
        "n_nonmissing": n,
        "mean": mean,
        "median": float(x.median()),
        "std": std,
        "min": float(x.min()),
        "q1": q1,
        "q3": q3,
        "max": float(x.max()),
        "iqr": iqr,
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
    }


def _describe_binary(x: pd.Series) -> dict[str, float]:
    x = x.dropna()
    n = int(len(x))
    if n == 0:
        return {
            "n_nonmissing": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_positive": 0,
            "n_zero": 0,
        }
    x_num = x.astype(float)
    p = float(x_num.mean())
    se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n))
    return {
        "n_nonmissing": n,
        "mean": p,
        "median": float(x_num.median()),
        "std": float(x_num.std(ddof=1)) if n > 1 else np.nan,
        "min": float(x_num.min()),
        "q1": float(x_num.quantile(0.25)),
        "q3": float(x_num.quantile(0.75)),
        "max": float(x_num.max()),
        "iqr": float(x_num.quantile(0.75) - x_num.quantile(0.25)),
        "ci95_low": float(max(0.0, p - 1.96 * se)),
        "ci95_high": float(min(1.0, p + 1.96 * se)),
        "n_positive": int((x_num > 0.5).sum()),
        "n_zero": int((x_num <= 0.5).sum()),
    }


def build_threshold_table(
    feature_df: pd.DataFrame,
    threshold_scope: str,
) -> pd.DataFrame:
    if threshold_scope not in ALLOWED_THRESHOLD_SCOPES:
        raise ValueError(f"Unsupported threshold_scope: {threshold_scope}")

    if threshold_scope == "region_specific":
        group_cols = ["region", "condition", "sweep"]
    elif threshold_scope == "region_pooled":
        group_cols = ["condition", "sweep"]
    else:
        group_cols = ["sweep"]

    rows: list[dict[str, Any]] = []
    for keys, group in feature_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {col: val for col, val in zip(group_cols, keys)}
        n_cells = int(group["file_id"].nunique())
        small_stratum = bool((threshold_scope == "region_specific" and n_cells < 5) or (threshold_scope == "region_pooled" and n_cells < 8))
        for feature_name in ALL_FEATURES:
            family = _feature_family(feature_name)
            series = group[feature_name] if feature_name in group.columns else pd.Series(dtype=float)
            if family == "binary":
                desc = _describe_binary(series)
                acceptable_low_q1 = 0.0
                acceptable_high_q3 = 1.0
                acceptable_low_ci95 = desc["ci95_low"]
                acceptable_high_ci95 = desc["ci95_high"]
            else:
                desc = _describe_continuous(series)
                iqr = desc["iqr"]
                acceptable_low_q1 = desc["q1"] - 1.5 * iqr if np.isfinite(iqr) else np.nan
                acceptable_high_q3 = desc["q3"] + 1.5 * iqr if np.isfinite(iqr) else np.nan
                acceptable_low_ci95 = desc["ci95_low"]
                acceptable_high_ci95 = desc["ci95_high"]
            n_total = n_cells
            missing_rate = float(max(0.0, 1.0 - desc["n_nonmissing"] / n_total)) if n_total > 0 else np.nan
            rows.append(
                {
                    **{"region": key_map.get("region", "ALL"), "condition": key_map.get("condition", "ALL"), "sweep": int(key_map["sweep"])},
                    "feature": feature_name,
                    "family": family,
                    "threshold_scope": threshold_scope,
                    "n_cells": n_cells,
                    "small_stratum": small_stratum,
                    **desc,
                    "missing_rate": missing_rate,
                    "acceptable_low_q1": acceptable_low_q1,
                    "acceptable_high_q3": acceptable_high_q3,
                    "acceptable_low_ci95": acceptable_low_ci95,
                    "acceptable_high_ci95": acceptable_high_ci95,
                }
            )
    out = pd.DataFrame(rows).sort_values(["threshold_scope", "condition", "region", "sweep", "feature"]).reset_index(drop=True)
    return out


def compute_feature_correlation_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    continuous = [f for f in PRIMARY_CONTINUOUS_FEATURES + CONDITIONAL_CONTINUOUS_FEATURES if f in feature_df.columns]
    rows: list[dict[str, Any]] = []
    for i, fa in enumerate(continuous):
        for fb in continuous[i + 1 :]:
            pair = feature_df[[fa, fb]].dropna()
            n = len(pair)
            if n < 3:
                rho = np.nan
            else:
                rho = float(pair.corr(method="spearman").iloc[0, 1])
            redundant = bool(np.isfinite(rho) and abs(rho) >= CORRELATION_REDUNDANCY_THRESHOLD)
            rows.append(
                {
                    "feature_a": fa,
                    "feature_b": fb,
                    "spearman_r": rho,
                    "n_rows": int(n),
                    "is_redundant": redundant,
                }
            )
    return pd.DataFrame(rows).sort_values(["is_redundant", "spearman_r"], ascending=[False, False]).reset_index(drop=True)


def compute_redundancy_penalties(correlation_df: pd.DataFrame) -> dict[str, float]:
    penalties = {feature: 1.0 for feature in ALL_FEATURES}
    for row in correlation_df.itertuples(index=False):
        if not bool(row.is_redundant):
            continue
        fa = str(row.feature_a)
        fb = str(row.feature_b)
        keep, down = (fa, fb) if FEATURE_PRIORITY.get(fa, 99) <= FEATURE_PRIORITY.get(fb, 99) else (fb, fa)
        penalties[keep] = min(penalties.get(keep, 1.0), 1.0)
        penalties[down] = min(penalties.get(down, 1.0), 0.5)
    return penalties


def compute_feature_reliability_weights(
    threshold_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> pd.DataFrame:
    penalties = compute_redundancy_penalties(correlation_df)
    out = threshold_df.copy()
    out["coverage_weight"] = (1.0 - out["missing_rate"]).clip(lower=0.0, upper=1.0)
    out["redundancy_penalty"] = out["feature"].map(lambda x: penalties.get(str(x), 1.0)).astype(float)
    out["reliability_weight"] = (out["coverage_weight"] * out["redundancy_penalty"]).astype(float)
    out["is_redundant_feature"] = out["redundancy_penalty"] < 1.0
    out["recommended_for_primary_loss"] = (~out["small_stratum"] | (out["threshold_scope"] != "region_specific")) & (out["reliability_weight"] > 0.25)
    return out


def compute_region_effect_summary(
    feature_df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    n_boot: int = 200,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    features = list(features or (PRIMARY_CONTINUOUS_FEATURES + CONDITIONAL_CONTINUOUS_FEATURES + BINARY_FEATURES))
    rows: list[dict[str, Any]] = []
    for condition in sorted(feature_df["condition"].unique()):
        for sweep in sorted(feature_df["sweep"].unique()):
            subset = feature_df[(feature_df["condition"] == condition) & (feature_df["sweep"] == sweep)]
            dh = subset[subset["region"] == "DH"]
            vh = subset[subset["region"] == "VH"]
            n_dh = int(dh["file_id"].nunique())
            n_vh = int(vh["file_id"].nunique())
            small_stratum = bool(n_dh < 5 or n_vh < 5)
            for feature_name in features:
                if feature_name not in subset.columns:
                    continue
                x_dh = dh[feature_name].dropna().astype(float).to_numpy()
                x_vh = vh[feature_name].dropna().astype(float).to_numpy()
                if len(x_dh) == 0 or len(x_vh) == 0:
                    diff = ci_low = ci_high = np.nan
                else:
                    diff = float(np.median(x_dh) - np.median(x_vh))
                    boot = np.empty(n_boot, dtype=float)
                    for i in range(n_boot):
                        dh_s = rng.choice(x_dh, size=len(x_dh), replace=True)
                        vh_s = rng.choice(x_vh, size=len(x_vh), replace=True)
                        boot[i] = np.median(dh_s) - np.median(vh_s)
                    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                rows.append(
                    {
                        "condition": condition,
                        "sweep": int(sweep),
                        "feature": feature_name,
                        "family": _feature_family(feature_name),
                        "n_cells_DH": n_dh,
                        "n_cells_VH": n_vh,
                        "small_stratum": small_stratum,
                        "dh_median": float(np.median(x_dh)) if len(x_dh) else np.nan,
                        "vh_median": float(np.median(x_vh)) if len(x_vh) else np.nan,
                        "dh_mean": float(np.mean(x_dh)) if len(x_dh) else np.nan,
                        "vh_mean": float(np.mean(x_vh)) if len(x_vh) else np.nan,
                        "dh_minus_vh_median": diff,
                        "dh_minus_vh_ci95_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                        "dh_minus_vh_ci95_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["feature", "condition", "sweep"]).reset_index(drop=True)


def benchmark_step02_pipeline(project_root: str | Path) -> pd.DataFrame:
    paths = project_paths(project_root)

    prep_start = time.perf_counter()
    parsed_clean_all = load_preprocessed_atf_files(paths.atf_dir, artifact_hint_map=MANUAL_TRANSIENT_ARTIFACT_HINTS)
    preprocess_elapsed = time.perf_counter() - prep_start
    benchmark_subset = list(parsed_clean_all[:8]) + list(parsed_clean_all[-2:]) if len(parsed_clean_all) > 10 else list(parsed_clean_all)

    rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    feature_frames_numpy = [extract_features(parsed, smoothing_engine="numpy") for parsed in benchmark_subset]
    extract_numpy_elapsed = time.perf_counter() - t0
    n_rows = int(sum(len(df) for df in feature_frames_numpy))
    rows.append({
        "stage": "preprocess",
        "engine": "shared",
        "elapsed_seconds": float(preprocess_elapsed),
        "n_rows": n_rows,
        "note": f"parse_atf + robust artifact preprocessing for {len(parsed_clean_all)} files; extraction benchmark uses {len(benchmark_subset)} files",
    })
    rows.append({
        "stage": "feature_extraction",
        "engine": "numpy",
        "elapsed_seconds": float(extract_numpy_elapsed),
        "n_rows": n_rows,
        "note": "extract_features over cached preprocessed traces",
    })

    extract_numba_elapsed = np.nan
    try:
        _ = extract_features(benchmark_subset[0], smoothing_engine="numba")
        t1 = time.perf_counter()
        feature_frames_numba = [extract_features(parsed, smoothing_engine="numba") for parsed in benchmark_subset]
        extract_numba_elapsed = time.perf_counter() - t1
        n_rows_numba = int(sum(len(df) for df in feature_frames_numba))
        rows.append({
            "stage": "feature_extraction",
            "engine": "numba",
            "elapsed_seconds": float(extract_numba_elapsed),
            "n_rows": n_rows_numba,
            "note": "extract_features over cached preprocessed traces (warm numba)",
        })
    except Exception:
        pass

    total_numpy = preprocess_elapsed + extract_numpy_elapsed * (len(parsed_clean_all) / max(len(benchmark_subset), 1))
    total_numba = preprocess_elapsed + extract_numba_elapsed * (len(parsed_clean_all) / max(len(benchmark_subset), 1)) if np.isfinite(extract_numba_elapsed) else np.nan
    speedup_compute = extract_numpy_elapsed / extract_numba_elapsed if np.isfinite(extract_numba_elapsed) and extract_numba_elapsed > 0 else np.nan
    total_gain = total_numpy - total_numba if np.isfinite(total_numba) else np.nan

    decision = "keep_numpy_default"
    rationale = "step02 is dominated by ATF parsing and artifact preprocessing; numba does not change the primary bottleneck"
    if np.isfinite(speedup_compute) and np.isfinite(total_gain) and speedup_compute >= 1.25 and total_gain >= 1.0:
        decision = "use_numba_default"
        rationale = "numba materially reduces end-to-end runtime even after the preprocessing bottleneck"

    rows.append({
        "stage": "decision",
        "engine": decision.replace("_default", ""),
        "elapsed_seconds": float(total_numpy),
        "n_rows": n_rows,
        "compute_speedup_numba_vs_numpy": float(speedup_compute) if np.isfinite(speedup_compute) else np.nan,
        "estimated_total_gain_seconds": float(total_gain) if np.isfinite(total_gain) else np.nan,
        "numba_decision": decision,
        "note": rationale,
    })

    return pd.DataFrame(rows)


def build_step02_outputs(
    feature_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    legacy_threshold_df: Optional[pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    thresholds_region = build_threshold_table(feature_df, threshold_scope="region_specific")
    thresholds_region_pooled = build_threshold_table(feature_df, threshold_scope="region_pooled")
    thresholds_global = build_threshold_table(feature_df, threshold_scope="global_pooled")
    thresholds_all = pd.concat([thresholds_region, thresholds_region_pooled, thresholds_global], ignore_index=True)
    corr_df = compute_feature_correlation_summary(feature_df)
    reliability_df = compute_feature_reliability_weights(thresholds_all, corr_df)
    region_effect_df = compute_region_effect_summary(feature_df)

    outputs = {
        "feature_table_by_sweep": feature_df,
        "region_condition_cell_counts": counts_df,
        "condition_region_sweep_thresholds": thresholds_all,
        "feature_reliability_weights": reliability_df,
        "feature_correlation_summary": corr_df,
        "region_effect_summary": region_effect_df,
    }
    if legacy_threshold_df is not None:
        outputs["legacy_threshold_preview"] = legacy_threshold_df
    return outputs




def load_step02_outputs(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    with open(output_dir / "analysis_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    results: dict[str, Any] = {
        "feature_table_by_sweep": pd.read_csv(output_dir / "feature_table_by_sweep.csv"),
        "preprocess_qc_by_sweep": pd.read_csv(output_dir / "preprocess_qc_by_sweep.csv"),
        "region_condition_cell_counts": pd.read_csv(output_dir / "region_condition_cell_counts.csv"),
        "condition_region_sweep_thresholds": pd.read_csv(output_dir / "condition_region_sweep_thresholds.csv"),
        "feature_reliability_weights": pd.read_csv(output_dir / "feature_reliability_weights.csv"),
        "feature_correlation_summary": pd.read_csv(output_dir / "feature_correlation_summary.csv"),
        "region_effect_summary": pd.read_csv(output_dir / "region_effect_summary.csv"),
        "atf_inventory": pd.read_csv(output_dir / "atf_region_condition_inventory.csv"),
        "legacy_threshold_preview": pd.read_csv(output_dir / "legacy_threshold_preview.csv"),
        "analysis_summary": summary,
    }
    perf_path = output_dir / "performance_benchmark.csv"
    if perf_path.exists():
        results["performance_benchmark"] = pd.read_csv(perf_path)
    results["paths"] = Step02Paths(
        project_root=output_dir.parents[1],
        atf_dir=output_dir.parents[1] / "data" / "2_K+ Pumps Data",
        threshold_csv=output_dir.parents[1] / "data" / "threshold_for_good_enough_fits.csv",
        output_dir=output_dir,
    )
    return results


def run_step02_rebuild_atf_thresholds(
    project_root: str | Path,
    output_dir: str | Path | None = None,
    build_benchmark: bool = True,
) -> dict[str, Any]:
    paths = project_paths(project_root, output_dir=output_dir)
    legacy_threshold_df = pd.read_csv(paths.threshold_csv) if paths.threshold_csv.exists() else None

    inventory_df = build_atf_inventory(paths.atf_dir)
    counts_df = count_region_condition_cells(inventory_df)
    feature_df, preprocess_qc_df, _, summary = extract_feature_tables(
        paths.atf_dir,
        artifact_hint_map=MANUAL_TRANSIENT_ARTIFACT_HINTS,
        smoothing_engine="numpy",
    )

    outputs = build_step02_outputs(feature_df, counts_df, legacy_threshold_df)
    outputs["atf_inventory"] = inventory_df
    outputs["preprocess_qc_by_sweep"] = preprocess_qc_df
    if build_benchmark:
        outputs["performance_benchmark"] = benchmark_step02_pipeline(project_root)

    file_map = {
        "atf_inventory": "atf_region_condition_inventory.csv",
        "feature_table_by_sweep": "feature_table_by_sweep.csv",
        "preprocess_qc_by_sweep": "preprocess_qc_by_sweep.csv",
        "region_condition_cell_counts": "region_condition_cell_counts.csv",
        "condition_region_sweep_thresholds": "condition_region_sweep_thresholds.csv",
        "feature_reliability_weights": "feature_reliability_weights.csv",
        "feature_correlation_summary": "feature_correlation_summary.csv",
        "region_effect_summary": "region_effect_summary.csv",
        "performance_benchmark": "performance_benchmark.csv",
        "legacy_threshold_preview": "legacy_threshold_preview.csv",
    }
    for key, file_name in file_map.items():
        if key in outputs:
            outputs[key].to_csv(paths.output_dir / file_name, index=False)

    summary_payload = {
        **summary,
        "n_region_specific_threshold_rows": int((outputs["condition_region_sweep_thresholds"]["threshold_scope"] == "region_specific").sum()),
        "n_region_pooled_threshold_rows": int((outputs["condition_region_sweep_thresholds"]["threshold_scope"] == "region_pooled").sum()),
        "n_global_pooled_threshold_rows": int((outputs["condition_region_sweep_thresholds"]["threshold_scope"] == "global_pooled").sum()),
        "redundant_pairs": int(outputs["feature_correlation_summary"]["is_redundant"].sum()),
        "mean_reliability_weight": float(outputs["feature_reliability_weights"]["reliability_weight"].mean()),
    }
    with open(paths.output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    outputs["analysis_summary"] = summary_payload
    outputs["paths"] = paths
    return outputs
