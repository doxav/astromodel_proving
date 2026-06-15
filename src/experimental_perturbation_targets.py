"""Experimental condition and regional perturbation targets from ATF features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .functional_mapping import direction_label

DEFAULT_TARGET_FEATURES = [
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

CONDITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("CONTROL", "MFA"),
    ("MFA", "MFA_BA"),
    ("CONTROL", "MFA_BA"),
)


def bh_adjust_preserve_nan(
    df: pd.DataFrame,
    *,
    p_col: str = "pvalue",
    q_col: str = "qvalue_bh",
    reject_col: str = "reject_bh_0p05",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values while preserving NaN p-values."""

    out = df.copy()
    if p_col not in out.columns:
        out[p_col] = np.nan
    out[q_col] = np.nan
    out[reject_col] = False
    mask = pd.to_numeric(out[p_col], errors="coerce").notna()
    pvals = pd.to_numeric(out.loc[mask, p_col], errors="coerce").to_numpy(dtype=float)
    if len(pvals) == 0:
        return out
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = float(len(ranked))
    adjusted = np.empty(len(ranked), dtype=float)
    running = 1.0
    for idx in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[idx] * n / float(idx + 1))
        adjusted[idx] = running
    qvals = np.empty_like(adjusted)
    qvals[order] = np.clip(adjusted, 0.0, 1.0)
    out.loc[mask, q_col] = qvals
    out.loc[mask, reject_col] = qvals <= float(alpha)
    return out


def _feature_columns(feature_df: pd.DataFrame, features: Iterable[str] | None) -> list[str]:
    selected = list(DEFAULT_TARGET_FEATURES if features is None else features)
    return [feature for feature in selected if feature in feature_df.columns]


def _mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(x))
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(x))
    if n < 2:
        return mean, mean, mean, n
    sem = float(stats.sem(x, nan_policy="omit"))
    if not np.isfinite(sem):
        return mean, np.nan, np.nan, n
    low, high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return mean, float(low), float(high), n


def _welch_pvalue(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    yy = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    if len(xx) < 2 or len(yy) < 2:
        return float("nan")
    _stat, pvalue = stats.ttest_ind(xx, yy, equal_var=False, nan_policy="omit")
    return float(pvalue) if np.isfinite(pvalue) else float("nan")


def build_condition_contrast_summary(
    feature_df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute condition direction targets pooled across regions and by region."""

    required = {"condition", "region", "sweep", "file_id"}
    missing = sorted(required - set(feature_df.columns))
    if missing:
        raise ValueError(f"feature_df is missing required columns: {missing}")
    feature_cols = _feature_columns(feature_df, features)
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("region_blind", feature_df.assign(region_scope="ALL"))]
    scopes.extend(
        (f"region_{region}", group.assign(region_scope=region))
        for region, group in feature_df.groupby("region", dropna=False)
    )
    for scope, data in scopes:
        for sweep, sweep_df in data.groupby("sweep", dropna=False):
            for cond1, cond2 in CONDITION_PAIRS:
                left = sweep_df[sweep_df["condition"] == cond1]
                right = sweep_df[sweep_df["condition"] == cond2]
                if left.empty or right.empty:
                    continue
                region_scope = str(sweep_df["region_scope"].iloc[0])
                for feature in feature_cols:
                    mean_1, ci1_low, ci1_high, n1 = _mean_ci(left[feature])
                    mean_2, ci2_low, ci2_high, n2 = _mean_ci(right[feature])
                    estimate = (
                        float(mean_2 - mean_1)
                        if np.isfinite(mean_1) and np.isfinite(mean_2)
                        else np.nan
                    )
                    rows.append(
                        {
                            "scope": scope,
                            "region": region_scope,
                            "sweep": int(sweep),
                            "feature": feature,
                            "experimental_contrast": f"{cond1}_to_{cond2}",
                            "cond_1": cond1,
                            "cond_2": cond2,
                            "mean_cond_1": mean_1,
                            "mean_cond_2": mean_2,
                            "ci_low_cond_1": ci1_low,
                            "ci_high_cond_1": ci1_high,
                            "ci_low_cond_2": ci2_low,
                            "ci_high_cond_2": ci2_high,
                            "estimate": estimate,
                            "pvalue": _welch_pvalue(left[feature], right[feature]),
                            "n_cond_1": n1,
                            "n_cond_2": n2,
                            "n_files": int(
                                pd.concat([left["file_id"], right["file_id"]])
                                .dropna()
                                .nunique()
                            ),
                            "experimental_direction_raw": direction_label(estimate),
                        }
                    )
    out = pd.DataFrame(rows)
    out = bh_adjust_preserve_nan(out)
    out["experimental_direction"] = np.where(
        out["qvalue_bh"].notna() & (out["qvalue_bh"].astype(float) > 0.05),
        "no_clear_change",
        out["experimental_direction_raw"],
    )
    return out.sort_values(["scope", "experimental_contrast", "feature", "sweep"]).reset_index(drop=True)


def summarize_direction_targets(condition_contrasts: pd.DataFrame) -> pd.DataFrame:
    """Collapse contrast rows into one direction target per feature/contrast/scope."""

    rows: list[dict[str, object]] = []
    for keys, group in condition_contrasts.groupby(
        ["scope", "region", "experimental_contrast", "feature"], dropna=False
    ):
        scope, region, contrast, feature = keys
        estimates = pd.to_numeric(group["estimate"], errors="coerce")
        estimate = float(estimates.median(skipna=True)) if estimates.notna().any() else np.nan
        direction_counts = group["experimental_direction"].astype(str).value_counts()
        direction = (
            str(direction_counts.index[0])
            if not direction_counts.empty
            else "undefined"
        )
        if "increase" in direction_counts and "decrease" in direction_counts:
            direction = "no_clear_change"
        if {"n_cond_1", "n_cond_2"}.issubset(group.columns):
            n_obs = int(group[["n_cond_1", "n_cond_2"]].sum(axis=1).sum())
        else:
            count_cols = [col for col in group.columns if str(col).startswith("n_")]
            n_obs = int(group[count_cols].sum(axis=1).sum()) if count_cols else int(len(group))
        rows.append(
            {
                "scope": scope,
                "region": region,
                "experimental_contrast": contrast,
                "feature": feature,
                "estimate": estimate,
                "experimental_direction": direction,
                "n_sweeps": int(group["sweep"].nunique()),
                "n_obs": n_obs,
                "min_qvalue_bh": float(pd.to_numeric(group["qvalue_bh"], errors="coerce").min(skipna=True))
                if group["qvalue_bh"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "experimental_contrast", "feature"]).reset_index(drop=True)


def build_matched_sweep_delta_of_delta(
    feature_df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute DH-minus-VH condition-effect differences at matched sweeps."""

    feature_cols = _feature_columns(feature_df, features)
    rows: list[dict[str, object]] = []
    for sweep, sweep_df in feature_df.groupby("sweep", dropna=False):
        for cond1, cond2 in CONDITION_PAIRS:
            for feature in feature_cols:
                means = (
                    sweep_df[sweep_df["condition"].isin([cond1, cond2])]
                    .groupby(["region", "condition"], dropna=False)[feature]
                    .mean()
                    .unstack("condition")
                )
                counts = (
                    sweep_df[sweep_df["condition"].isin([cond1, cond2])]
                    .groupby(["region", "condition"], dropna=False)[feature]
                    .count()
                    .unstack("condition")
                )
                if not {"DH", "VH"}.issubset(set(means.index)) or not {cond1, cond2}.issubset(set(means.columns)):
                    continue
                dh_delta = float(means.loc["DH", cond2] - means.loc["DH", cond1])
                vh_delta = float(means.loc["VH", cond2] - means.loc["VH", cond1])
                estimate = float(dh_delta - vh_delta)
                rows.append(
                    {
                        "feature": feature,
                        "sweep": int(sweep),
                        "contrast_label": f"diffdiff__{cond1}_to_{cond2}",
                        "experimental_contrast": f"{cond1}_to_{cond2}",
                        "cond_1": cond1,
                        "cond_2": cond2,
                        "n_DH_cond1": int(counts.loc["DH", cond1]),
                        "n_DH_cond2": int(counts.loc["DH", cond2]),
                        "n_VH_cond1": int(counts.loc["VH", cond1]),
                        "n_VH_cond2": int(counts.loc["VH", cond2]),
                        "diff_DH_cond2_minus_cond1": dh_delta,
                        "diff_VH_cond2_minus_cond1": vh_delta,
                        "delta_of_delta_DH_minus_VH": estimate,
                        "estimate_DH_minus_VH": estimate,
                        "experimental_direction": direction_label(estimate),
                    }
                )
    out = pd.DataFrame(rows)
    out["pvalue"] = np.nan
    return bh_adjust_preserve_nan(out)


def _slope_by_group(
    feature_df: pd.DataFrame,
    group_cols: list[str],
    *,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    feature_cols = _feature_columns(feature_df, features)
    rows: list[dict[str, object]] = []
    means = feature_df.groupby(group_cols + ["sweep"], dropna=False)[feature_cols].mean().reset_index()
    for keys, group in means.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x = pd.to_numeric(group["sweep"], errors="coerce").to_numpy(dtype=float)
        for feature in feature_cols:
            y = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            estimate = float("nan")
            if int(mask.sum()) >= 2:
                estimate = float(np.polyfit(x[mask], y[mask], deg=1)[0])
            rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "feature": feature,
                    "estimate": estimate,
                    "n_sweeps": int(mask.sum()),
                    "pvalue": np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_numeric_sweep_targets(
    feature_df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build numeric sweep-slope target tables matching the updated ATF layer."""

    combo = _slope_by_group(feature_df, ["region", "condition"], features=features)
    combo["label"] = (
        "slope__"
        + combo["region"].astype(str)
        + "__"
        + combo["condition"].astype(str)
    )
    delta_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    for feature in combo["feature"].dropna().unique():
        sub = combo[combo["feature"] == feature]
        for region in ["DH", "VH"]:
            reg = sub[sub["region"] == region]
            for cond1, cond2 in CONDITION_PAIRS:
                s1 = reg.loc[reg["condition"] == cond1, "estimate"]
                s2 = reg.loc[reg["condition"] == cond2, "estimate"]
                if s1.empty or s2.empty:
                    continue
                delta_rows.append(
                    {
                        "feature": feature,
                        "label": f"delta_slope__{region}__{cond1}_to_{cond2}",
                        "region": region,
                        "cond_1": cond1,
                        "cond_2": cond2,
                        "estimate": float(s2.iloc[0] - s1.iloc[0]),
                        "pvalue": np.nan,
                    }
                )
        deltas = pd.DataFrame(delta_rows)
        for cond1, cond2 in CONDITION_PAIRS:
            dh = deltas[
                (deltas["feature"] == feature)
                & (deltas["region"] == "DH")
                & (deltas["cond_1"] == cond1)
                & (deltas["cond_2"] == cond2)
            ]
            vh = deltas[
                (deltas["feature"] == feature)
                & (deltas["region"] == "VH")
                & (deltas["cond_1"] == cond1)
                & (deltas["cond_2"] == cond2)
            ]
            if dh.empty or vh.empty:
                continue
            diff_rows.append(
                {
                    "feature": feature,
                    "label": f"delta_of_delta_slope__DH_minus_VH__{cond1}_to_{cond2}",
                    "cond_1": cond1,
                    "cond_2": cond2,
                    "estimate": float(dh["estimate"].iloc[0] - vh["estimate"].iloc[0]),
                    "experimental_direction": direction_label(float(dh["estimate"].iloc[0] - vh["estimate"].iloc[0])),
                    "pvalue": np.nan,
                }
            )
    deltas_df = bh_adjust_preserve_nan(pd.DataFrame(delta_rows))
    diff_df = bh_adjust_preserve_nan(pd.DataFrame(diff_rows))

    blind_slopes = _slope_by_group(feature_df, ["condition"], features=features)
    blind_slopes["label"] = "region_blind_slope__" + blind_slopes["condition"].astype(str)
    blind_delta_rows: list[dict[str, object]] = []
    for feature in blind_slopes["feature"].dropna().unique():
        sub = blind_slopes[blind_slopes["feature"] == feature]
        for cond1, cond2 in CONDITION_PAIRS:
            s1 = sub.loc[sub["condition"] == cond1, "estimate"]
            s2 = sub.loc[sub["condition"] == cond2, "estimate"]
            if s1.empty or s2.empty:
                continue
            estimate = float(s2.iloc[0] - s1.iloc[0])
            blind_delta_rows.append(
                {
                    "feature": feature,
                    "label": f"region_blind_delta_slope__{cond1}_to_{cond2}",
                    "cond_1": cond1,
                    "cond_2": cond2,
                    "estimate": estimate,
                    "experimental_direction": direction_label(estimate),
                    "pvalue": np.nan,
                }
            )
    return {
        "numeric_sweep_combo_slopes": bh_adjust_preserve_nan(combo),
        "numeric_sweep_condition_slope_deltas_within_region": deltas_df,
        "numeric_sweep_delta_of_delta_between_regions": diff_df,
        "region_blind_condition_slopes": bh_adjust_preserve_nan(blind_slopes),
        "region_blind_condition_slope_deltas": bh_adjust_preserve_nan(pd.DataFrame(blind_delta_rows)),
    }


def build_delta_feature_profiles(
    feature_df: pd.DataFrame,
    *,
    pooled_region: bool,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build multi-feature condition-delta profiles by region or pooled region."""

    feature_cols = _feature_columns(feature_df, features)
    group_cols = ["condition", "sweep"] if pooled_region else ["region", "condition", "sweep"]
    means = feature_df.groupby(group_cols, dropna=False)[feature_cols].mean().reset_index()
    rows: list[dict[str, object]] = []
    region_values = ["ALL"] if pooled_region else sorted(feature_df["region"].dropna().astype(str).unique())
    for region in region_values:
        source = means if pooled_region else means[means["region"].astype(str) == region]
        for sweep in sorted(source["sweep"].dropna().unique()):
            sweep_df = source[source["sweep"] == sweep]
            for cond1, cond2 in CONDITION_PAIRS:
                left = sweep_df[sweep_df["condition"] == cond1]
                right = sweep_df[sweep_df["condition"] == cond2]
                if left.empty or right.empty:
                    continue
                row: dict[str, object] = {
                    "scope": "region_blind" if pooled_region else region,
                    "region": region,
                    "sweep": int(sweep),
                    "contrast": f"{cond1}_to_{cond2}",
                    "cond_1": cond1,
                    "cond_2": cond2,
                }
                for feature in feature_cols:
                    row[feature] = float(right[feature].iloc[0] - left[feature].iloc[0])
                rows.append(row)
    return pd.DataFrame(rows)


def build_region_condition_profile_terms(
    condition_contrasts: pd.DataFrame,
    delta_of_delta: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact cell-profile region-by-condition term rows."""

    contrast_rows = []
    for _, row in delta_of_delta.iterrows():
        contrast_rows.append(
            {
                "family": "primary_continuous",
                "feature": row["feature"],
                "term": "C(region):C(condition)",
                "scope": "matched_sweep_delta_of_delta",
                "sweep": row["sweep"],
                "experimental_contrast": row["experimental_contrast"],
                "estimate": row["delta_of_delta_DH_minus_VH"],
                "pvalue": row.get("pvalue", np.nan),
                "qvalue_bh": row.get("qvalue_bh", np.nan),
                "reject_bh_0p05": row.get("reject_bh_0p05", False),
                "experimental_direction": row.get("experimental_direction", "undefined"),
            }
        )
    pooled = condition_contrasts[condition_contrasts["scope"].eq("region_blind")].copy()
    for _, row in pooled.iterrows():
        contrast_rows.append(
            {
                "family": "primary_continuous",
                "feature": row["feature"],
                "term": "C(condition)",
                "scope": "region_blind_condition_contrast",
                "sweep": row["sweep"],
                "experimental_contrast": row["experimental_contrast"],
                "estimate": row["estimate"],
                "pvalue": row.get("pvalue", np.nan),
                "qvalue_bh": row.get("qvalue_bh", np.nan),
                "reject_bh_0p05": row.get("reject_bh_0p05", False),
                "experimental_direction": row.get("experimental_direction", "undefined"),
            }
        )
    return pd.DataFrame(contrast_rows)


def build_region_perturbation_summary(
    condition_contrasts: pd.DataFrame,
    numeric_delta_of_delta: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected regional perturbation differences by feature."""

    regional = condition_contrasts[condition_contrasts["scope"].isin(["region_DH", "region_VH"])].copy()
    rows: list[dict[str, object]] = []
    for feature in sorted(regional["feature"].dropna().unique()):
        sub = regional[regional["feature"] == feature]
        row: dict[str, object] = {"feature": feature}
        for condition in ["CONTROL", "MFA", "MFA_BA"]:
            condition_rows = sub[sub["cond_2"] == condition]
            row[f"{condition}_mean_abs_gap_proxy"] = float(
                pd.to_numeric(condition_rows["estimate"], errors="coerce").abs().mean(skipna=True)
            ) if not condition_rows.empty else np.nan
        for contrast in ["CONTROL_to_MFA", "CONTROL_to_MFA_BA"]:
            dd = numeric_delta_of_delta[
                (numeric_delta_of_delta["feature"] == feature)
                & ((numeric_delta_of_delta["cond_1"] + "_to_" + numeric_delta_of_delta["cond_2"]) == contrast)
            ]
            row[f"delta_of_delta_slope_{contrast}_estimate"] = (
                float(dd["estimate"].iloc[0]) if not dd.empty else np.nan
            )
            row[f"delta_of_delta_slope_{contrast}_q"] = (
                float(dd["qvalue_bh"].iloc[0]) if not dd.empty and pd.notna(dd["qvalue_bh"].iloc[0]) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_experimental_perturbation_targets(
    feature_df: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    features: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build and optionally write all experimental perturbation-target tables."""

    condition_contrasts = build_condition_contrast_summary(feature_df, features=features)
    direction_targets = summarize_direction_targets(condition_contrasts)
    kinetic_targets = direction_targets[direction_targets["scope"].eq("region_blind")].copy()
    delta_of_delta = build_matched_sweep_delta_of_delta(feature_df, features=features)
    regional_targets = summarize_direction_targets(
        delta_of_delta.rename(columns={"delta_of_delta_DH_minus_VH": "estimate"}).assign(
            scope="delta_of_delta_DH_minus_VH",
            region="DH_minus_VH",
        )
    )
    numeric = build_numeric_sweep_targets(feature_df, features=features)
    delta_profiles_by_region = build_delta_feature_profiles(feature_df, pooled_region=False, features=features)
    delta_profiles_region_blind = build_delta_feature_profiles(feature_df, pooled_region=True, features=features)
    profile_terms = build_region_condition_profile_terms(condition_contrasts, delta_of_delta)
    region_summary = build_region_perturbation_summary(
        condition_contrasts,
        numeric["numeric_sweep_delta_of_delta_between_regions"],
    )
    tables = {
        "experimental_kinetic_direction_targets": kinetic_targets,
        "region_specific_perturbation_direction_targets": regional_targets,
        "experimental_condition_contrast_summary": condition_contrasts,
        "experimental_region_condition_profile_terms": profile_terms,
        "matched_sweep_delta_of_delta": delta_of_delta,
        **numeric,
        "delta_feature_profiles_by_region": delta_profiles_by_region,
        "delta_feature_profiles_region_blind": delta_profiles_region_blind,
        "region_perturbation_summary_selected_features": region_summary,
    }
    if output_dir is not None:
        out_dir = Path(output_dir)
        second = out_dir / "experimental_second_layer"
        out_dir.mkdir(parents=True, exist_ok=True)
        second.mkdir(parents=True, exist_ok=True)
        top_level = {
            "experimental_kinetic_direction_targets",
            "region_specific_perturbation_direction_targets",
            "experimental_condition_contrast_summary",
            "experimental_region_condition_profile_terms",
        }
        for name, table in tables.items():
            target = out_dir / f"{name}.csv" if name in top_level else second / f"{name}.csv"
            table.to_csv(target, index=False)
    return tables
