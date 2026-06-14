from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

EFFECTIVE_COLUMNS: tuple[str, ...] = (
    "P_gap_eff",
    "gamma_t_eff",
    "gamma_s_eff",
    "volume_ratio_wa_wo",
)

DEFAULT_EFFECTIVE_RANGES: Mapping[str, tuple[float, float]] = {
    "P_gap_eff": (1e-8, 5e-3),
    "gamma_t_eff": (1e-8, 5e-2),
    "gamma_s_eff": (1e-8, 5e-2),
    "volume_ratio_wa_wo": (1e-3, 1e3),
}

EFFECTIVE_SELECTION_STRATEGIES: tuple[str, ...] = (
    "quality_top_k",
    "effective_maximin_best_seed",
    "quality_filtered_effective_maximin",
    "effective_cluster_medoids",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _validate_selector_inputs(
    candidates: pd.DataFrame,
    *,
    candidates_per_cell: int,
    strategy: str,
    distance_threshold: float,
    quality_pool_fraction: float,
    min_quality_pool_size: int,
) -> None:
    missing = sorted(set(("file_id", *EFFECTIVE_COLUMNS)) - set(candidates.columns))
    if missing:
        raise ValueError(f"effective-diverse selection is missing required columns: {missing}")
    if int(candidates_per_cell) < 1:
        raise ValueError("candidates_per_cell must be >= 1")
    if strategy not in EFFECTIVE_SELECTION_STRATEGIES:
        raise ValueError(f"strategy must be one of {EFFECTIVE_SELECTION_STRATEGIES}")
    if not np.isfinite(float(distance_threshold)) or float(distance_threshold) < 0:
        raise ValueError("distance_threshold must be finite and non-negative")
    if not (0 < float(quality_pool_fraction) <= 1):
        raise ValueError("quality_pool_fraction must be in (0, 1]")
    if int(min_quality_pool_size) < 1:
        raise ValueError("min_quality_pool_size must be >= 1")


def _with_effective_plausibility(
    candidates: pd.DataFrame,
    effective_ranges: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    out = candidates.copy()
    flags: list[str] = []
    for column in EFFECTIVE_COLUMNS:
        lo, hi = effective_ranges[column]
        flag = f"{column}_effective_plausible"
        out[flag] = pd.to_numeric(out[column], errors="coerce").between(lo, hi, inclusive="both")
        flags.append(flag)
    out["effective_plausible"] = out[flags].all(axis=1)
    return out


def _rank_by_quality(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    sort_cols = [
        "holdout_mean_pass_fraction",
        "mean_weighted_pass_fraction",
        "holdout_mean_rmse_mV",
        "mean_trace_rmse_mV",
        "ensemble_rank",
        "scalar_objective",
    ]
    for column in sort_cols:
        if column not in out.columns:
            out[column] = np.nan
    out = out.sort_values(
        sort_cols,
        ascending=[False, False, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    out["effective_selection_quality_order"] = np.arange(1, len(out) + 1)
    if len(out) == 1:
        out["effective_selection_quality_score"] = 1.0
    else:
        out["effective_selection_quality_score"] = 1.0 - (
            (out["effective_selection_quality_order"] - 1) / (len(out) - 1)
        )
    return out


def _log_effective_values(group: pd.DataFrame) -> np.ndarray:
    values = group.loc[:, EFFECTIVE_COLUMNS].to_numpy(dtype=float)
    return np.log10(np.clip(values, 1e-300, None))


def _min_distance_to_selected(values: np.ndarray, selected: Iterable[int], index: int) -> float:
    selected_list = list(selected)
    if not selected_list:
        return float("inf")
    return float(min(np.linalg.norm(values[index] - values[other]) for other in selected_list))


def _select_quality_top_k(group: pd.DataFrame, k: int) -> pd.DataFrame:
    return _rank_by_quality(group).head(k)


def _select_maximin(group: pd.DataFrame, k: int) -> pd.DataFrame:
    ranked = _rank_by_quality(group)
    values = _log_effective_values(ranked)
    selected = [0]
    while len(selected) < min(k, len(ranked)):
        remaining = [i for i in range(len(ranked)) if i not in selected]
        scored = [
            (
                _min_distance_to_selected(values, selected, i),
                float(ranked.loc[i, "effective_selection_quality_score"]),
                i,
            )
            for i in remaining
        ]
        selected.append(max(scored)[2])
    return ranked.iloc[selected].copy()


def _select_quality_filtered_maximin(
    group: pd.DataFrame,
    k: int,
    *,
    quality_pool_fraction: float,
    min_quality_pool_size: int,
) -> pd.DataFrame:
    ranked = _rank_by_quality(group)
    if len(ranked) <= k:
        pool = ranked
    else:
        keep_n = max(k, min(len(ranked), max(min_quality_pool_size, int(np.ceil(len(ranked) * quality_pool_fraction)))))
        pool = ranked.head(keep_n).reset_index(drop=True)
    values = _log_effective_values(pool)
    selected = [0]
    while len(selected) < min(k, len(pool)):
        remaining = [i for i in range(len(pool)) if i not in selected]
        scored = [
            (
                _min_distance_to_selected(values, selected, i),
                float(pool.loc[i, "effective_selection_quality_score"]),
                i,
            )
            for i in remaining
        ]
        selected.append(max(scored)[2])
    return pool.iloc[selected].copy()


def _select_cluster_medoids(group: pd.DataFrame, k: int, *, distance_threshold: float) -> pd.DataFrame:
    ranked = _rank_by_quality(group)
    values = _log_effective_values(ranked)
    selected: list[int] = []
    centers: list[np.ndarray] = []
    for index in range(len(ranked)):
        if len(selected) >= k:
            break
        if all(float(np.linalg.norm(values[index] - center)) >= distance_threshold for center in centers):
            selected.append(index)
            centers.append(values[index])
    while len(selected) < min(k, len(ranked)):
        remaining = [i for i in range(len(ranked)) if i not in selected]
        scored = [
            (
                _min_distance_to_selected(values, selected, i),
                float(ranked.loc[i, "effective_selection_quality_score"]),
                i,
            )
            for i in remaining
        ]
        selected.append(max(scored)[2])
    return ranked.iloc[selected].copy()


def _annotate_selection(
    selected: pd.DataFrame,
    *,
    strategy: str,
    distance_threshold: float,
    target_k: int,
) -> pd.DataFrame:
    out = selected.copy().reset_index(drop=True)
    values = _log_effective_values(out) if not out.empty else np.empty((0, len(EFFECTIVE_COLUMNS)))
    distances: list[float] = []
    chosen: list[int] = []
    for index in range(len(out)):
        distance = _min_distance_to_selected(values, chosen, index)
        distances.append(np.nan if not np.isfinite(distance) else distance)
        chosen.append(index)
    out["effective_diverse_selected"] = True
    out["effective_selection_strategy"] = strategy
    out["effective_selection_rank"] = np.arange(1, len(out) + 1)
    out["effective_selection_target_k"] = int(target_k)
    out["effective_selection_distance_threshold"] = float(distance_threshold)
    out["effective_selection_min_distance_to_previous"] = distances
    return out


def select_effective_diverse_candidates(
    candidates: pd.DataFrame,
    *,
    candidates_per_cell: int = 3,
    strategy: str = "quality_filtered_effective_maximin",
    distance_threshold: float = 0.5,
    quality_pool_fraction: float = 0.5,
    min_quality_pool_size: int = 6,
    require_accepted_all6: bool = True,
    require_effective_plausible: bool = True,
    effective_ranges: Mapping[str, tuple[float, float]] = DEFAULT_EFFECTIVE_RANGES,
) -> pd.DataFrame:
    """Select per-cell accepted candidates that are diverse in effective-parameter space."""

    _validate_selector_inputs(
        candidates,
        candidates_per_cell=candidates_per_cell,
        strategy=strategy,
        distance_threshold=distance_threshold,
        quality_pool_fraction=quality_pool_fraction,
        min_quality_pool_size=min_quality_pool_size,
    )
    source = _with_effective_plausibility(candidates, effective_ranges)
    if require_accepted_all6 and "accepted_all6" in source.columns:
        source = source[_as_bool(source["accepted_all6"])].copy()
    if require_effective_plausible:
        source = source[source["effective_plausible"]].copy()
    if source.empty:
        return source.assign(
            effective_diverse_selected=pd.Series(dtype=bool),
            effective_selection_strategy=pd.Series(dtype=object),
            effective_selection_rank=pd.Series(dtype=int),
            effective_selection_target_k=pd.Series(dtype=int),
            effective_selection_distance_threshold=pd.Series(dtype=float),
            effective_selection_min_distance_to_previous=pd.Series(dtype=float),
        )

    selectors = {
        "quality_top_k": lambda group: _select_quality_top_k(group, int(candidates_per_cell)),
        "effective_maximin_best_seed": lambda group: _select_maximin(group, int(candidates_per_cell)),
        "quality_filtered_effective_maximin": lambda group: _select_quality_filtered_maximin(
            group,
            int(candidates_per_cell),
            quality_pool_fraction=float(quality_pool_fraction),
            min_quality_pool_size=int(min_quality_pool_size),
        ),
        "effective_cluster_medoids": lambda group: _select_cluster_medoids(
            group,
            int(candidates_per_cell),
            distance_threshold=float(distance_threshold),
        ),
    }
    parts = [
        _annotate_selection(
            selectors[strategy](group),
            strategy=strategy,
            distance_threshold=float(distance_threshold),
            target_k=int(candidates_per_cell),
        )
        for _, group in source.groupby("file_id", sort=True, dropna=False)
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _min_pairwise_distance(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(
        min(
            np.linalg.norm(values[i] - values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
        )
    )


def _greedy_cluster_count(values: np.ndarray, threshold: float) -> int:
    centers: list[np.ndarray] = []
    for row in values:
        if all(float(np.linalg.norm(row - center)) >= threshold for center in centers):
            centers.append(row)
    return len(centers)


def summarize_effective_diverse_selection(
    selected: pd.DataFrame,
    *,
    distance_threshold: float = 0.5,
) -> pd.DataFrame:
    """Summarize effective-diverse selected candidates by cell."""

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "file_id",
                "region",
                "condition",
                "effective_selection_strategy",
                "n_selected",
                "min_pairwise_effective_log_distance",
                "effective_cluster_count",
            ]
        )
    rows: list[dict[str, object]] = []
    for keys, group in selected.groupby(["file_id", "region", "condition"], sort=True, dropna=False):
        values = _log_effective_values(group)
        rows.append(
            {
                "file_id": keys[0],
                "region": keys[1],
                "condition": keys[2],
                "effective_selection_strategy": str(group["effective_selection_strategy"].iloc[0]),
                "n_selected": int(len(group)),
                "min_pairwise_effective_log_distance": _min_pairwise_distance(values),
                "effective_cluster_count": _greedy_cluster_count(values, float(distance_threshold)),
                "mean_trace_rmse_mV": float(pd.to_numeric(group.get("mean_trace_rmse_mV"), errors="coerce").mean()),
                "mean_weighted_pass_fraction": float(
                    pd.to_numeric(group.get("mean_weighted_pass_fraction"), errors="coerce").mean()
                ),
                "rank1_retained": bool((pd.to_numeric(group.get("ensemble_rank"), errors="coerce") == 1).any()),
            }
        )
    return pd.DataFrame(rows)
