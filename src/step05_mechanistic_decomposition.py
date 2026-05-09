"""Step 05 mechanistic decomposition for accepted cell ensembles.

This module implements the reviewer-response Step 05 pipeline.  It adapts the
same local model and ATF-derived conventions used by the reference ATF notebook
(`analysis/astro_atf_analysis_improved_sectioned.ipynb`) and by Step 04, but it
keeps the claims deliberately conservative: mechanism clusters are candidate
regimes pending Step 06 predictive/perturbation validation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from .astro_model import (
    CURRENT_DICT_K_BATH_VALUES,
    VALID_CURRENTS,
    simulate_with_hidden_outputs,
)
from .contracts import protocol_condition
from .mechanisms import compute_flux_summary, compute_proxy_validity
from .protocols import stim_window_seconds

OUTPUT_SUBDIR = "mechanisms"
EFFECTIVE_COLUMNS = ["P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo"]
IDENTITY_COLUMNS = ["file_id", "region", "condition", "candidate_id"]
MECHANISM_FEATURES = [
    "log10_P_gap_eff",
    "log10_gamma_t_eff",
    "log10_gamma_s_eff",
    "log10_volume_ratio_wa_wo",
    "gap_fraction_mean",
    "kir_fraction_mean",
    "leak_fraction_mean",
    "log10_gap_to_kir_integral_ratio_mean",
    "K_o_recovery_error_mean",
]


@dataclass(slots=True)
class Step05Config:
    """Configuration for Step 05.

    The default grid is intentionally coarse enough for notebook/test execution
    while retaining all six canonical current sweeps.
    """

    max_candidates: int | None = None
    time_points: int = 180
    t_final_ms: float = 50_000.0
    n_clusters: int = 3
    bootstrap_iterations: int = 25
    random_seed: int = 13
    max_representatives: int = 3
    interpolation_points: int = 5
    small_stratum_min_cells: int = 3
    min_candidates_for_multicluster: int = 6
    min_cells_for_multicluster: int = 3
    min_cells_per_cluster: int = 2
    proxy_failure_downgrade_fraction: float = 0.5
    write_outputs: bool = True


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_step04_accepted_ensemble(
    project_root: Path | str,
    source_path: Path | str | None = None,
    max_candidates: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load Step 04 accepted/reviewer-facing candidates with provenance.

    The model-aligned Step 04 output is preferred.  The newer multisweep runner
    output is supported as a fallback so Step 05 remains coherent with current
    repository artifacts.
    """

    root = Path(project_root).resolve()
    candidates = [
        root / "outputs" / "cell_fits" / "accepted_cell_ensembles.csv",
        root
        / "outputs"
        / "step04_cell_specific_multisweep"
        / "accepted_candidates.csv",
    ]
    path = (
        Path(source_path).resolve()
        if source_path is not None
        else next((p for p in candidates if p.exists()), candidates[0])
    )
    if not path.exists():
        raise FileNotFoundError(f"Step 04 accepted ensemble not found: {path}")

    df = pd.read_csv(path)
    if "gki" in df.columns and "g_kir" not in df.columns:
        df["g_kir"] = df["gki"]
    if "g_kir" in df.columns and "gki" not in df.columns:
        df["gki"] = df["g_kir"]
    if "accepted" not in df.columns:
        if "cell_reviewer_facing" in df.columns:
            df["accepted"] = _as_bool(df["cell_reviewer_facing"])
        elif "accepted_all6" in df.columns:
            df["accepted"] = _as_bool(df["accepted_all6"])
        else:
            df["accepted"] = True
    else:
        df["accepted"] = _as_bool(df["accepted"])

    missing = sorted(
        set(
            IDENTITY_COLUMNS + EFFECTIVE_COLUMNS + ["g_kir", "gl_a", "zth", "zs", "eps"]
        )
        - set(df.columns)
    )
    if missing:
        raise ValueError(f"Step 04 ensemble is missing required columns: {missing}")
    df = df[df["accepted"]].copy()
    df["step04_source_path"] = str(
        path.relative_to(root) if path.is_relative_to(root) else path
    )
    df["candidate_id"] = df["candidate_id"].astype(str)
    if "switching_function" not in df.columns:
        df["switching_function"] = "sigmoid"
    else:
        df["switching_function"] = df["switching_function"].fillna("sigmoid")
    if "k_bath_gain" not in df.columns:
        df["k_bath_gain"] = 1.0
    if max_candidates is not None:
        df = df.sort_values(IDENTITY_COLUMNS).head(int(max_candidates)).copy()
    if df.empty:
        raise ValueError("No accepted Step 04 candidates are available for Step 05.")
    return df.reset_index(drop=True), path


def _finite_float(row: Mapping[str, Any], key: str) -> float | None:
    if key not in row:
        return None
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def kbath_middle_for_current(
    row: Mapping[str, Any], current_na: int, sweep: int | None = None
) -> tuple[float, str]:
    """Return the middle K bath value to use for one current/sweep.

    Default behavior is the historical current-specific bath drive. A scalar
    ``K_bath_value_middle`` is deliberately not applied to all six sweeps unless
    Step 04 explicitly labeled it as a scalar/single-current override.
    """

    current_key = str(int(current_na))
    if current_key not in CURRENT_DICT_K_BATH_VALUES:
        raise ValueError(f"Unknown current_na={current_na!r}")
    historical = float(CURRENT_DICT_K_BATH_VALUES[current_key][1])

    if sweep is not None:
        for key in (
            f"K_bath_value_middle_sweep_{int(sweep)}",
            f"K_bath_middle_sweep_{int(sweep)}",
        ):
            value = _finite_float(row, key)
            if value is not None:
                return value, key

    for key in (
        f"K_bath_value_middle_{current_key}",
        f"K_bath_middle_{current_key}",
        f"K_bath_middle_{current_key}nA",
    ):
        value = _finite_float(row, key)
        if value is not None:
            return value, key

    intercept = _finite_float(row, "affine_kbath_intercept")
    slope = _finite_float(row, "affine_kbath_slope")
    if intercept is not None and slope is not None:
        return float(intercept + slope * int(current_na)), "affine_kbath_cell_nuisance"

    gain = _finite_float(row, "k_bath_gain")
    if gain is not None:
        return historical * gain, "historical_current_kbath_scaled_by_gain"

    protocol_mode = str(row.get("protocol_mode", "fixed_historical_kbath")).lower()
    scalar = _finite_float(row, "K_bath_value_middle")
    if scalar is not None and protocol_mode in {
        "single_current_kbath_override",
        "scalar_kbath_override",
        "per_candidate_kbath_override",
    }:
        return scalar, "scalar_candidate_kbath_override"

    return historical, "historical_current_kbath"


def reconstruct_flat_params(
    row: Mapping[str, Any], current_na: int = 100, sweep: int | None = None
) -> dict[str, Any]:
    """Reconstruct simulator-ready flat parameters from Step 04 effective coordinates."""

    w_a = float(row.get("w_a", 2000.0))
    sig_a = 1600.0
    faraday = 96485.0
    volume_ratio = max(float(row["volume_ratio_wa_wo"]), 1e-12)
    condition = str(row.get("condition", "CONTROL")).upper()
    protocol_key = protocol_condition(condition)
    k_bath_middle, k_bath_mode = kbath_middle_for_current(
        row, current_na=current_na, sweep=sweep
    )
    params = {
        "d": 1.0,
        "pk": max(float(row["P_gap_eff"]), 1e-18),
        "gt": max(float(row["gamma_t_eff"]), 1e-18) * w_a * faraday / sig_a,
        "gs": max(float(row["gamma_s_eff"]), 1e-18) * w_a * faraday / sig_a,
        "wo": w_a / volume_ratio,
        "w_a": w_a,
        "gki": float(row.get("gki", row.get("g_kir", 1.0))),
        "gl_a": float(row.get("gl_a", 0.01)),
        "zth": float(row.get("zth", 0.2)),
        "zs": float(row.get("zs", 0.05)),
        "eps": float(row.get("eps", 1e-3)),
        "switching_function": str(row.get("switching_function", "sigmoid")),
        "Va_l": float(row.get("Va_l", -70.0)),
        "Va_s": float(row.get("Va_s", -90.0)),
        "ca": float(row.get("ca", 400.0)),
        "K_bath_value_middle": float(k_bath_middle),
        "_K_bath_value_middle_used": float(k_bath_middle),
        "_K_bath_override_mode": k_bath_mode,
        "_protocol_condition_hint": protocol_key,
    }
    return params


def _time_grid(config: Step05Config) -> np.ndarray:
    return np.linspace(
        0.0, float(config.t_final_ms), int(config.time_points), dtype=float
    )


def simulate_candidate_sweeps(
    candidate: Mapping[str, Any], config: Step05Config
) -> pd.DataFrame:
    """Simulate all six sweeps for one accepted candidate and return flux rows."""

    time_ms = _time_grid(config)
    rows: list[dict[str, Any]] = []
    condition = str(candidate["condition"])
    window_s = stim_window_seconds(condition)
    protocol_key = protocol_condition(condition)
    for sweep, current_na in enumerate(VALID_CURRENTS, start=1):
        params = reconstruct_flat_params(
            candidate, current_na=int(current_na), sweep=sweep
        )
        base = {col: candidate.get(col) for col in candidate.keys() if col in candidate}
        base.update(
            {
                "sweep": sweep,
                "current_na": int(current_na),
                "K_bath_middle_used": float(params["_K_bath_value_middle_used"]),
                "K_bath_override_mode": str(params["_K_bath_override_mode"]),
                "stim_window_start_s": float(window_s[0]),
                "stim_window_end_s": float(window_s[1]),
            }
        )
        try:
            sim = simulate_with_hidden_outputs(
                params,
                {
                    "experiment_type": protocol_key,
                    "current_na": int(current_na),
                    "t_eval_ms": time_ms,
                },
            )
            flux = compute_flux_summary(sim, stim_window_s=window_s)
            proxy = compute_proxy_validity(sim, window_s=window_s)
            row = {
                **base,
                **flux,
                "proxy_pearson_r": proxy["pearson_r"],
                "proxy_spearman_r": proxy["spearman_r"],
                "proxy_rmse_after_scaling": proxy["rmse_after_scaling"],
                "proxy_validity_class": proxy["validity_class"],
                "simulation_status": "ok",
                "failure_reason": "",
            }
        except Exception as exc:  # noqa: BLE001 - failures must be explicit rows
            row = {
                **base,
                "simulation_status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "proxy_validity_class": "failed",
            }
        rows.append(row)
    return pd.DataFrame(rows)


def build_candidate_mechanism_table(fit_mechanisms: pd.DataFrame) -> pd.DataFrame:
    ok = fit_mechanisms[fit_mechanisms["simulation_status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(
            "No successful Step 05 hidden-output simulations were available for clustering."
        )
    agg = ok.groupby(IDENTITY_COLUMNS, as_index=False).agg(
        accepted=("accepted", "first"),
        n_sweeps_simulated=("sweep", "nunique"),
        I_Kir_integral_mean=("I_Kir_integral", "mean"),
        I_kgap_integral_mean=("I_kgap_integral", "mean"),
        I_leak_integral_mean=("I_leak_integral", "mean"),
        I_Kir_peak_abs_mean=("I_Kir_peak_abs", "mean"),
        I_kgap_peak_abs_mean=("I_kgap_peak_abs", "mean"),
        gap_to_kir_integral_ratio_mean=("gap_to_kir_integral_ratio", "mean"),
        gap_fraction_mean=("gap_fraction", "mean"),
        kir_fraction_mean=("kir_fraction", "mean"),
        leak_fraction_mean=("leak_fraction", "mean"),
        K_o_peak_mean=("K_o_peak", "mean"),
        K_o_final_mean=("K_o_final", "mean"),
        K_o_recovery_error_mean=("K_o_recovery_error", "mean"),
        proxy_pearson_r_mean=("proxy_pearson_r", "mean"),
        proxy_spearman_r_mean=("proxy_spearman_r", "mean"),
        proxy_validity_failed_fraction=(
            "proxy_validity_class",
            lambda s: float((s.astype(str) == "failed").mean()),
        ),
        proxy_validity_strong_fraction=(
            "proxy_validity_class",
            lambda s: float((s.astype(str) == "strong").mean()),
        ),
        K_bath_middle_min=("K_bath_middle_used", "min"),
        K_bath_middle_max=("K_bath_middle_used", "max"),
        K_bath_middle_n_unique=("K_bath_middle_used", "nunique"),
        K_bath_override_modes=(
            "K_bath_override_mode",
            lambda s: ";".join(sorted(set(map(str, s)))),
        ),
    )
    first_cols = [
        c
        for c in EFFECTIVE_COLUMNS
        + [
            "g_kir",
            "gki",
            "gl_a",
            "zth",
            "zs",
            "eps",
            "k_bath_gain",
            "switching_function",
        ]
        if c in ok.columns
    ]
    first = ok.groupby(IDENTITY_COLUMNS, as_index=False)[first_cols].first()
    out = agg.merge(first, on=IDENTITY_COLUMNS, how="left")
    for col in EFFECTIVE_COLUMNS:
        out[f"log10_{col}"] = np.log10(np.clip(out[col].astype(float), 1e-18, None))
    out["log10_gap_to_kir_integral_ratio_mean"] = np.log10(
        np.clip(out["gap_to_kir_integral_ratio_mean"].astype(float), 1e-18, None)
    )
    out["dominant_mechanism"] = np.select(
        [
            out["gap_fraction_mean"] >= 0.6,
            out["kir_fraction_mean"] >= 0.6,
            out["leak_fraction_mean"] >= 0.6,
        ],
        ["Gap", "Kir", "Leak"],
        default="Mixed",
    )
    return out


def _standardize(
    df: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True)
    x = x.fillna(med).fillna(0.0).to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return (x - mean) / scale, mean, scale


def assign_mechanism_clusters(
    candidate_table: pd.DataFrame, config: Step05Config
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = candidate_table.copy()
    x, mean, scale = _standardize(df, MECHANISM_FEATURES)
    n_candidates = int(len(df))
    n_cells = int(df["file_id"].nunique()) if "file_id" in df.columns else n_candidates
    cluster_status = "candidate_regime_screen"
    cluster_reason = "sufficient_candidates_and_cells_for_screen"

    if n_candidates < int(config.min_candidates_for_multicluster) or n_cells < int(
        config.min_cells_for_multicluster
    ):
        k = 1
        cluster_status = "insufficient_evidence"
        cluster_reason = (
            f"needs >= {config.min_candidates_for_multicluster} candidates and "
            f">= {config.min_cells_for_multicluster} independent cells; observed "
            f"{n_candidates} candidates and {n_cells} cells"
        )
    else:
        k = min(max(1, int(config.n_clusters)), len(df), n_cells)

    if k == 1:
        labels = np.zeros(len(df), dtype=int)
        centers = x.mean(axis=0, keepdims=True)
    else:
        centers, labels = kmeans2(x, k, minit="points", seed=int(config.random_seed))
    df["mechanism_cluster"] = [f"M{int(label) + 1}" for label in labels]

    if k > 1:
        cell_counts = df.groupby("mechanism_cluster")["file_id"].nunique()
        if (cell_counts < int(config.min_cells_per_cluster)).any():
            k = 1
            labels = np.zeros(len(df), dtype=int)
            centers = x.mean(axis=0, keepdims=True)
            df["mechanism_cluster"] = "M1"
            cluster_status = "insufficient_evidence"
            cluster_reason = f"at least one cluster has fewer than {config.min_cells_per_cluster} independent cells"

    df["mechanism_cluster_count"] = int(k)
    df["candidate_count_for_clustering"] = n_candidates
    df["independent_cell_count"] = n_cells
    df["cluster_evidence_status"] = cluster_status
    df["cluster_evidence_reason"] = cluster_reason
    return df, centers, mean, scale


def _nearest_center_labels(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1)


def _pairwise_coassignment_score(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    total = 0
    agree = 0
    for i, j in combinations(range(len(a)), 2):
        total += 1
        agree += int((a[i] == a[j]) == (b[i] == b[j]))
    return float(agree / total) if total else np.nan


def bootstrap_cluster_stability(
    clustered: pd.DataFrame,
    centers: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    config: Step05Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.random_seed))
    cells = clustered["file_id"].drop_duplicates().to_numpy()
    rows: list[dict[str, Any]] = []
    evidence_status = str(
        clustered.get(
            "cluster_evidence_status", pd.Series(["candidate_regime_screen"])
        ).iloc[0]
    )
    evidence_reason = str(
        clustered.get("cluster_evidence_reason", pd.Series([""])).iloc[0]
    )
    if (
        evidence_status != "candidate_regime_screen"
        or clustered["mechanism_cluster"].nunique() < 2
        or config.bootstrap_iterations <= 0
    ):
        return pd.DataFrame(
            [
                {
                    "bootstrap_iteration": 0,
                    "n_cells": len(cells),
                    "n_candidates": len(clustered),
                    "coassignment_stability": np.nan,
                    "stability_status": "insufficient_evidence",
                    "stability_reason": evidence_reason
                    or "not enough clusters/cells for bootstrap stability",
                }
            ]
        )
    for i in range(int(config.bootstrap_iterations)):
        sampled = rng.choice(cells, size=len(cells), replace=True)
        boot = pd.concat(
            [clustered[clustered["file_id"] == cell] for cell in sampled],
            ignore_index=True,
        )
        x = (boot[MECHANISM_FEATURES].astype(float).to_numpy() - mean) / scale
        boot_labels = _nearest_center_labels(x, centers)
        # Compare original center labels for the sampled rows to reassigned center labels.
        orig = (
            boot["mechanism_cluster"]
            .str.replace("M", "", regex=False)
            .astype(int)
            .to_numpy()
            - 1
        )
        score = _pairwise_coassignment_score(orig, boot_labels)
        rows.append(
            {
                "bootstrap_iteration": i + 1,
                "n_cells": int(len(np.unique(sampled))),
                "n_candidates": int(len(boot)),
                "coassignment_stability": score,
                "stability_status": "stable" if score >= 0.75 else "unstable",
                "stability_reason": "cell_level_resampling_against_original_centers",
            }
        )
    return pd.DataFrame(rows)


def summarize_region_enrichment(
    clustered: pd.DataFrame, small_stratum_min_cells: int = 3
) -> pd.DataFrame:
    counts = clustered.groupby(
        ["region", "condition", "mechanism_cluster"], as_index=False
    ).agg(
        n_candidates=("candidate_id", "nunique"),
        n_cells=("file_id", "nunique"),
    )
    totals = counts.groupby(["region", "condition"], as_index=False).agg(
        total_candidates=("n_candidates", "sum"), total_cells=("n_cells", "sum")
    )
    out = counts.merge(totals, on=["region", "condition"], how="left")
    out["candidate_fraction"] = out["n_candidates"] / out["total_candidates"].replace(
        0, np.nan
    )
    out["small_stratum_flag"] = out["total_cells"] < int(small_stratum_min_cells)
    out["interpretation_scope"] = (
        "population_level_cell_association_not_phenotype_claim"
    )
    return out


def select_representatives(
    clustered: pd.DataFrame, config: Step05Config
) -> pd.DataFrame:
    df = clustered[clustered["accepted"].astype(bool)].copy()
    if df.empty:
        return pd.DataFrame(
            columns=list(clustered.columns)
            + ["representative_rank", "selection_reason", "claim_scope"]
        )
    x, _, _ = _standardize(df, MECHANISM_FEATURES)
    selected = [
        int(
            np.argmin(
                df.get(
                    "K_o_recovery_error_mean", pd.Series(np.zeros(len(df)))
                ).to_numpy(dtype=float)
            )
        )
    ]
    while len(selected) < min(int(config.max_representatives), len(df)):
        remaining = [i for i in range(len(df)) if i not in selected]
        dists = []
        for idx in remaining:
            min_dist = min(float(np.linalg.norm(x[idx] - x[j])) for j in selected)
            dists.append((min_dist, idx))
        selected.append(max(dists)[1])
    reps = df.iloc[selected].copy().reset_index(drop=True)
    reps["representative_rank"] = np.arange(1, len(reps) + 1)
    reps["selection_reason"] = "maximin_mechanism_distance_preserving_step04_acceptance"
    if "cluster_claim_scope" in reps.columns:
        reps["claim_scope"] = reps["cluster_claim_scope"]
    else:
        reps["claim_scope"] = "candidate_mechanism_regime_pending_step06_validation"
    return reps


def classify_geometry(clustered: pd.DataFrame, config: Step05Config) -> pd.DataFrame:
    clusters = sorted(clustered["mechanism_cluster"].unique())
    evidence_status = str(
        clustered.get(
            "cluster_evidence_status", pd.Series(["candidate_regime_screen"])
        ).iloc[0]
    )
    if len(clusters) < 2 or evidence_status != "candidate_regime_screen":
        return pd.DataFrame(
            [
                {
                    "cluster_a": clusters[0] if clusters else "none",
                    "cluster_b": "none",
                    "n_interpolation_points": 0,
                    "interpolation_status": "not_run",
                    "geometry_screen_type": "not_run_insufficient_cluster_evidence",
                    "geometry_classification": "insufficient_evidence",
                    "validation_status": "pending_step06",
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    centers = clustered.groupby("mechanism_cluster")[
        EFFECTIVE_COLUMNS
        + ["gap_fraction_mean", "kir_fraction_mean", "leak_fraction_mean"]
    ].median(numeric_only=True)
    functional_low = float(
        clustered.get("K_o_recovery_error_mean", pd.Series([0.0])).quantile(0.05)
    )
    functional_high = float(
        clustered.get("K_o_recovery_error_mean", pd.Series([0.0])).quantile(0.95)
    )
    for a, b in combinations(clusters, 2):
        ca = centers.loc[a]
        cb = centers.loc[b]
        accepted_like = True
        for alpha in np.linspace(0.0, 1.0, int(config.interpolation_points)):
            _interp = (1 - alpha) * np.log10(
                np.clip(ca[EFFECTIVE_COLUMNS].astype(float), 1e-18, None)
            ) + alpha * np.log10(
                np.clip(cb[EFFECTIVE_COLUMNS].astype(float), 1e-18, None)
            )
            # Conservative functional proxy: interpolate observed recovery errors between cluster medians.
            err_a = float(
                clustered.loc[
                    clustered["mechanism_cluster"] == a, "K_o_recovery_error_mean"
                ].median()
            )
            err_b = float(
                clustered.loc[
                    clustered["mechanism_cluster"] == b, "K_o_recovery_error_mean"
                ].median()
            )
            err = (1 - alpha) * err_a + alpha * err_b
            accepted_like &= bool(
                functional_low - 1e-12 <= err <= functional_high + 1e-12
            )
        rows.append(
            {
                "cluster_a": a,
                "cluster_b": b,
                "n_interpolation_points": int(config.interpolation_points),
                "interpolation_status": "effective_space_screened",
                "geometry_screen_type": "proxy_recovery_error_screen_not_simulated",
                "geometry_classification": (
                    "compensation_manifold"
                    if accepted_like
                    else "separated_modes_pending_validation"
                ),
                "validation_status": "pending_step06",
            }
        )
    return pd.DataFrame(rows)


def build_claim_scope_table(
    clustered: pd.DataFrame, geometry: pd.DataFrame
) -> pd.DataFrame:
    has_separated = (
        geometry["geometry_classification"]
        .astype(str)
        .str.contains("separated_modes")
        .any()
    )
    evidence_status = (
        str(
            clustered.get(
                "cluster_evidence_status", pd.Series(["candidate_regime_screen"])
            ).iloc[0]
        )
        if not clustered.empty
        else "insufficient_evidence"
    )
    proxy_status = (
        str(clustered.get("proxy_validity_status", pd.Series(["unknown"])).iloc[0])
        if not clustered.empty
        else "unknown"
    )
    if evidence_status != "candidate_regime_screen":
        headline = "insufficient_evidence_for_candidate_mechanism_regimes"
    elif proxy_status == "proxy_unreliable":
        headline = "mechanism_flux_decomposition_available_proxy_unreliable_pending_assumption_sensitivity"
    elif has_separated:
        headline = "candidate_mechanism_regimes_pending_validation"
    else:
        headline = (
            "accepted_compensation_or_mixed_mechanism_structure_pending_validation"
        )
    return pd.DataFrame(
        [
            {
                "claim_topic": "mechanism_diversity",
                "allowed_pre_step06_claim": headline,
                "forbidden_pre_step06_claim": "candidate_degenerate_regimes",
                "required_next_step": "Step 06 predictive and perturbation validation",
                "n_clusters": (
                    int(clustered["mechanism_cluster"].nunique())
                    if not clustered.empty
                    else 0
                ),
                "n_independent_cells": (
                    int(clustered["file_id"].nunique()) if not clustered.empty else 0
                ),
                "n_candidates": int(len(clustered)),
                "cluster_evidence_status": evidence_status,
                "proxy_validity_status": proxy_status,
            },
            {
                "claim_topic": "region_enrichment",
                "allowed_pre_step06_claim": "population_level_cell_association_not_phenotype_claim",
                "forbidden_pre_step06_claim": "paired_pharmacology_or_animal_level_phenotype",
                "required_next_step": "larger stratified validation and Step 06 robustness checks",
                "n_clusters": (
                    int(clustered["mechanism_cluster"].nunique())
                    if not clustered.empty
                    else 0
                ),
                "n_independent_cells": (
                    int(clustered["file_id"].nunique()) if not clustered.empty else 0
                ),
                "n_candidates": int(len(clustered)),
                "cluster_evidence_status": evidence_status,
                "proxy_validity_status": proxy_status,
            },
        ]
    )


def compare_simulation_grid_performance(
    project_root: Path | str, max_candidates: int = 1
) -> pd.DataFrame:
    rows = []
    for label, points in [("coarse", 90), ("default", 180)]:
        cfg = Step05Config(
            max_candidates=max_candidates,
            time_points=points,
            bootstrap_iterations=0,
            write_outputs=False,
        )
        t0 = time.perf_counter()
        result = run_step05_mechanistic_decomposition(project_root, cfg)
        rows.append(
            {
                "preset": label,
                "time_points": points,
                "elapsed_seconds": time.perf_counter() - t0,
                "n_candidate_sweeps": len(result["accepted_fit_mechanisms"]),
                "n_clusters": result["mechanism_clusters"][
                    "mechanism_cluster"
                ].nunique(),
                "recommendation": (
                    "use_default_for_reviewer_outputs"
                    if label == "default"
                    else "use_coarse_for_debug_only"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_step05_mechanistic_decomposition(
    project_root: Path | str,
    config: Step05Config | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    config = config or Step05Config()
    root = Path(project_root).resolve()
    out_dir = _ensure_dir(
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "outputs" / OUTPUT_SUBDIR
    )
    t0 = time.perf_counter()

    ensemble, source_path = load_step04_accepted_ensemble(
        root, max_candidates=config.max_candidates
    )
    sweep_tables = [
        simulate_candidate_sweeps(row.to_dict(), config)
        for _, row in ensemble.iterrows()
    ]
    fit_mechanisms = pd.concat(sweep_tables, ignore_index=True)
    candidate_table = build_candidate_mechanism_table(fit_mechanisms)
    candidate_table["proxy_validity_status"] = np.where(
        candidate_table["proxy_validity_failed_fraction"].astype(float)
        >= float(config.proxy_failure_downgrade_fraction),
        "proxy_unreliable",
        "proxy_not_failed_majority",
    )
    clustered, centers, mean, scale = assign_mechanism_clusters(candidate_table, config)
    stability = bootstrap_cluster_stability(clustered, centers, mean, scale, config)
    mean_stability = (
        float(stability["coassignment_stability"].mean(skipna=True))
        if "coassignment_stability" in stability
        else np.nan
    )
    clustered["bootstrap_mean_coassignment_stability"] = mean_stability
    clustered["cluster_claim_scope"] = np.select(
        [
            clustered["cluster_evidence_status"].astype(str)
            != "candidate_regime_screen",
            clustered["proxy_validity_status"].astype(str) == "proxy_unreliable",
            np.isfinite(mean_stability) & (mean_stability >= 0.75),
        ],
        [
            "insufficient_evidence_for_candidate_mechanism_regime_claim",
            "mechanism_flux_decomposition_available_proxy_unreliable_pending_assumption_sensitivity",
            "candidate_mechanism_regime_pending_step06_validation",
        ],
        default="insufficient_stability_for_separated_mode_claim",
    )
    representatives = select_representatives(clustered, config)
    enrichment = summarize_region_enrichment(clustered, config.small_stratum_min_cells)
    geometry = classify_geometry(clustered, config)
    claim_scope = build_claim_scope_table(clustered, geometry)
    perf = (
        compare_simulation_grid_performance(root, max_candidates=min(1, len(ensemble)))
        if config.write_outputs
        else pd.DataFrame()
    )

    summary = {
        "step_name": "Step 05 — Mechanistic decomposition of accepted cell ensembles",
        "input_source": str(
            source_path.relative_to(root)
            if source_path.is_relative_to(root)
            else source_path
        ),
        "config": asdict(config),
        "n_input_candidates": int(len(ensemble)),
        "n_candidate_sweep_rows": int(len(fit_mechanisms)),
        "n_successful_sweep_simulations": int(
            (fit_mechanisms["simulation_status"] == "ok").sum()
        ),
        "n_mechanism_clusters": int(clustered["mechanism_cluster"].nunique()),
        "n_independent_cells": (
            int(clustered["file_id"].nunique()) if not clustered.empty else 0
        ),
        "cluster_evidence_status": (
            str(clustered["cluster_evidence_status"].iloc[0])
            if not clustered.empty
            else "insufficient_evidence"
        ),
        "cluster_evidence_reason": (
            str(clustered["cluster_evidence_reason"].iloc[0])
            if not clustered.empty
            else "no clusters"
        ),
        "headline_claim_scope": str(claim_scope.iloc[0]["allowed_pre_step06_claim"]),
        "elapsed_seconds": time.perf_counter() - t0,
    }

    if config.write_outputs:
        fit_mechanisms.to_csv(out_dir / "accepted_fit_mechanisms.csv", index=False)
        clustered.to_csv(out_dir / "mechanism_clusters.csv", index=False)
        representatives.to_csv(out_dir / "representatives.csv", index=False)
        enrichment.to_csv(out_dir / "region_mechanism_enrichment.csv", index=False)
        geometry.to_csv(out_dir / "geometry_classification.csv", index=False)
        stability.to_csv(out_dir / "bootstrap_cluster_stability.csv", index=False)
        claim_scope.to_csv(out_dir / "claim_scope_table.csv", index=False)
        if not perf.empty:
            perf.to_csv(out_dir / "performance_benchmark.csv", index=False)
        (out_dir / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return {
        "accepted_fit_mechanisms": fit_mechanisms,
        "mechanism_clusters": clustered,
        "representatives": representatives,
        "region_mechanism_enrichment": enrichment,
        "geometry_classification": geometry,
        "bootstrap_cluster_stability": stability,
        "claim_scope_table": claim_scope,
        "performance_benchmark": perf,
        "analysis_summary": summary,
    }
