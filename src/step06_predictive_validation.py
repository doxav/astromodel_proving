"""Step 06 predictive validation and perturbation robustness.

The pipeline joins Step 04 accepted cell ensembles with Step 05 mechanism labels,
re-simulates accepted candidates, computes region-aware posterior predictive
checks against Step 02 feature bands, and stress-tests mechanism labels under a
small auditable perturbation panel.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .astro_model import VALID_CURRENTS, build_paramdict, simulate_odeint, simulate_with_hidden_outputs
from .atf_features import FEATURE_COLUMNS, extract_features_from_trace
from .mechanisms import compute_flux_summary
from .protocols import stim_window_seconds
from .step05_mechanistic_decomposition import (
    Step05Config,
    load_step04_accepted_ensemble,
    reconstruct_flat_params,
    run_step05_mechanistic_decomposition,
)
from .contracts import protocol_condition

OUTPUT_SUBDIR = "predictive_validation"
IDENTITY_COLUMNS = ["file_id", "region", "condition", "candidate_id"]
EFFECTIVE_DIVERSITY_COLUMNS = [
    "P_gap_eff",
    "gamma_t_eff",
    "gamma_s_eff",
    "volume_ratio_wa_wo",
]
MECHANISM_SCORE_COLUMNS = [
    "dKs_activation_score_mean",
    "long_range_distribution_fraction_mean",
    "voltage_coupling_score_mean",
    "kir_current_score_mean",
    "log10_recruited_surface_score",
]
STABLE_PHENOTYPE_LABELS = {
    "available_surface_voltage_coupled_but_ionic_recruitment_low",
    "mixed_local_spatial_buffering",
    "recruited_surface_gap_assisted_buffering",
}


@dataclass(slots=True)
class Step06Config:
    max_candidates: int | None = None
    step04_source_path: str | None = None
    candidate_policy: str = "best_per_cell"
    candidates_per_cell: int = 3
    time_points: int = 80
    t_final_ms: float = 50_000.0
    prediction_interval_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)
    min_holdout_pass_fraction: float = 0.30
    max_trace_rmse_mV: float = 25.0
    perturbation_current_na: int | None = None
    perturbation_factors: dict[str, float] = field(
        default_factory=lambda: {
            "nominal": 1.0,
            "eps_scale_low": 0.5,
            "eps_scale_high": 2.0,
            "stimulus_duration_short": 0.75,
            "stimulus_duration_long": 1.25,
            "baseline_K_o_low": 0.90,
            "baseline_K_o_high": 1.10,
            "current_scale_low": 0.75,
            "current_scale_high": 1.25,
        }
    )
    robustness_recovery_error_max: float = 1.5
    robustness_peak_ratio_max: float = 1.75
    robustness_K_o_peak_max_mM: float = 15.0
    robustness_K_o_final_max_mM: float = 7.0
    min_perturbation_feature_pass_fraction: float = 0.25
    min_robust_fraction: float = 0.50
    write_outputs: bool = True
    require_candidate_level_heldout: bool = True
    final_degeneracy_claim_allowed: bool = False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _time_grid(config: Step06Config, factor: float = 1.0) -> np.ndarray:
    return np.linspace(0.0, float(config.t_final_ms) * float(factor), int(config.time_points), dtype=float)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_mechanism_labels(project_root: Path | str, max_candidates: int | None = None) -> pd.DataFrame:
    root = Path(project_root).resolve()
    path = root / "outputs" / "mechanisms" / "mechanism_clusters.csv"
    if not path.exists():
        run_step05_mechanistic_decomposition(
            root,
            Step05Config(max_candidates=max_candidates, time_points=60, bootstrap_iterations=0, write_outputs=True),
        )
    if not path.exists():
        raise FileNotFoundError(f"Step 05 mechanism labels not found: {path}")
    df = pd.read_csv(path)
    required = set(IDENTITY_COLUMNS + ["mechanism_cluster", "cluster_claim_scope"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Mechanism labels missing required columns: {missing}")
    if max_candidates is not None:
        df = df.sort_values(IDENTITY_COLUMNS).head(int(max_candidates)).copy()
    return df


def _candidate_sort(ensemble: pd.DataFrame) -> tuple[list[str], list[bool]]:
    """Return the stable candidate ranking used by Step 06 scope policies."""

    sort_cols = [
        c
        for c in [
            "file_id",
            "cell_reviewer_facing",
            "holdout_mean_pass_fraction",
            "mean_weighted_pass_fraction",
            "holdout_mean_rmse_mV",
            "mean_trace_rmse_mV",
            "ensemble_rank",
        ]
        if c in ensemble.columns
    ]
    ascending = [True, False, False, False, True, True, True][: len(sort_cols)]
    return sort_cols, ascending


def _rank_candidates_by_quality(ensemble: pd.DataFrame) -> pd.DataFrame:
    """Return candidates in stable quality order with a normalized quality score."""

    sort_cols, ascending = _candidate_sort(ensemble)
    ranked = ensemble.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked["step06_selection_quality_order"] = np.arange(1, len(ranked) + 1)
    if len(ranked) == 1:
        ranked["step06_selection_quality_score"] = 1.0
    else:
        ranked["step06_selection_quality_score"] = 1.0 - (
            (ranked["step06_selection_quality_order"] - 1) / (len(ranked) - 1)
        )
    return ranked


def _with_mechanism_score_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add continuous Step 05 mechanism-score features used by diverse selection."""

    out = candidates.copy()
    gamma_s = pd.to_numeric(out.get("gamma_s_eff"), errors="coerce")
    activation = pd.to_numeric(out.get("dKs_activation_score_mean"), errors="coerce")
    recruited_surface = gamma_s * activation
    out["log10_recruited_surface_score"] = np.log10(
        np.clip(recruited_surface.to_numpy(dtype=float), 1e-12, None)
    )
    return out


def _mechanism_score_matrix(candidates: pd.DataFrame) -> np.ndarray:
    """Build a standardized matrix from log effective and continuous mechanism scores."""

    source = _with_mechanism_score_features(candidates)
    missing = sorted(set(EFFECTIVE_DIVERSITY_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"mechanism-score diverse selection is missing effective columns: {missing}")
    feature_parts: list[pd.Series] = []
    for column in EFFECTIVE_DIVERSITY_COLUMNS:
        values = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
        feature_parts.append(pd.Series(np.log10(np.clip(values, 1e-300, None)), index=source.index))
    for column in MECHANISM_SCORE_COLUMNS:
        values = pd.to_numeric(source.get(column), errors="coerce")
        feature_parts.append(values if isinstance(values, pd.Series) else pd.Series(np.nan, index=source.index))
    matrix = pd.concat(feature_parts, axis=1).replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    return (matrix - mean) / scale


def _min_distance_to_selected(values: np.ndarray, selected: list[int], index: int) -> float:
    """Return the minimum Euclidean distance from one candidate to selected candidates."""

    if not selected:
        return float("inf")
    return float(min(np.linalg.norm(values[index] - values[other]) for other in selected))


def _select_mechanism_score_diverse_candidates(ranked: pd.DataFrame, candidates_per_cell: int) -> pd.DataFrame:
    """Select quality-seeded candidates that are diverse in effective and mechanism-score space."""

    k = int(candidates_per_cell)
    selected: list[int] = [0]
    while len(selected) < min(k, len(ranked)):
        values = _mechanism_score_matrix(ranked)
        selected_labels = set(
            ranked.iloc[selected]["buffering_phenotype"].astype(str).dropna()
        )
        remaining = [index for index in range(len(ranked)) if index not in selected]
        scored = []
        for index in remaining:
            label = str(ranked.loc[index, "buffering_phenotype"]) if "buffering_phenotype" in ranked.columns else "unlabeled"
            stable_novel_label = int(label in STABLE_PHENOTYPE_LABELS and label not in selected_labels)
            scored.append(
                (
                    _min_distance_to_selected(values, selected, index),
                    stable_novel_label,
                    float(ranked.loc[index, "step06_selection_quality_score"]),
                    -index,
                    index,
                )
            )
        selected.append(max(scored)[-1])
    out = ranked.iloc[selected].copy().reset_index(drop=True)
    values = _mechanism_score_matrix(out)
    previous: list[int] = []
    distances: list[float] = []
    novel_flags: list[bool] = []
    labels_seen: set[str] = set()
    for index, row in out.iterrows():
        distance = _min_distance_to_selected(values, previous, int(index))
        distances.append(np.nan if not np.isfinite(distance) else distance)
        label = str(row.get("buffering_phenotype", "unlabeled"))
        novel_flags.append(label in STABLE_PHENOTYPE_LABELS and label not in labels_seen)
        labels_seen.add(label)
        previous.append(int(index))
    out["step06_selection_policy"] = "mechanism_score_diverse_per_cell"
    out["step06_selection_rank"] = np.arange(1, len(out) + 1)
    out["step06_selection_min_mechanism_score_distance"] = distances
    out["step06_selection_stable_phenotype_novel"] = novel_flags
    return out


def _select_step06_candidate_scope(ensemble: pd.DataFrame, config: Step06Config) -> pd.DataFrame:
    """Apply the configured candidate-scope sensitivity policy."""

    policy = str(config.candidate_policy)
    valid_policies = {"all", "best_per_cell", "top_k_per_cell", "mechanism_diverse_per_cell", "mechanism_score_diverse_per_cell"}
    if policy not in valid_policies:
        raise ValueError(f"candidate_policy must be one of {sorted(valid_policies)}")
    if ensemble.empty or policy == "all":
        return ensemble.copy()
    candidates_per_cell = int(config.candidates_per_cell)
    if candidates_per_cell < 1:
        raise ValueError("candidates_per_cell must be >= 1")
    k = 1 if policy == "best_per_cell" else candidates_per_cell
    ranked = _rank_candidates_by_quality(ensemble)
    if policy in {"best_per_cell", "top_k_per_cell"}:
        return ranked.groupby("file_id", as_index=False, dropna=False).head(k).reset_index(drop=True)
    if policy == "mechanism_score_diverse_per_cell":
        selected = [
            _select_mechanism_score_diverse_candidates(group.reset_index(drop=True), k)
            for _, group in ranked.groupby("file_id", sort=True, dropna=False)
        ]
        return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if "mechanism_cluster" not in ranked.columns:
        ranked = ranked.assign(mechanism_cluster="unlabeled")
    diverse = (
        ranked.groupby(["file_id", "mechanism_cluster"], as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    sort_cols, ascending = _candidate_sort(diverse)
    return (
        diverse.sort_values(sort_cols, ascending=ascending)
        .groupby("file_id", as_index=False, dropna=False)
        .head(k)
        .reset_index(drop=True)
    )


def load_step06_inputs(project_root: Path | str, config: Step06Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensemble, _ = load_step04_accepted_ensemble(
        project_root,
        source_path=config.step04_source_path,
        max_candidates=None,
    )
    mechanisms = load_mechanism_labels(project_root, max_candidates=None)
    keep_cols = [
        c
        for c in mechanisms.columns
        if c
        in set(
            IDENTITY_COLUMNS
            + [
                "mechanism_cluster",
                "dominant_mechanism",
                "cluster_evidence_status",
                "cluster_claim_scope",
                "buffering_phenotype",
                "phenotype_claim_scope",
                "phenotype_specificity_score",
                "dKs_activation_score_mean",
                "long_range_distribution_fraction_mean",
                "voltage_coupling_score_mean",
                "kir_current_score_mean",
            ]
        )
    ]
    merged = ensemble.merge(
        mechanisms[keep_cols].drop_duplicates(IDENTITY_COLUMNS),
        on=IDENTITY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    merged["mechanism_cluster"] = merged["mechanism_cluster"].fillna("unlabeled")
    merged["dominant_mechanism"] = merged.get("dominant_mechanism", pd.Series(index=merged.index, dtype=object)).fillna("unknown")
    merged["cluster_evidence_status"] = merged.get("cluster_evidence_status", pd.Series(index=merged.index, dtype=object)).fillna("insufficient_evidence")
    merged["cluster_claim_scope"] = merged.get("cluster_claim_scope", pd.Series(index=merged.index, dtype=object)).fillna("missing_step05_label")
    merged["buffering_phenotype"] = merged.get("buffering_phenotype", pd.Series(index=merged.index, dtype=object)).fillna("unlabeled")
    merged["phenotype_claim_scope"] = merged.get("phenotype_claim_scope", pd.Series(index=merged.index, dtype=object)).fillna("missing_step05_phenotype")
    if "phenotype_specificity_score" not in merged.columns:
        merged["phenotype_specificity_score"] = np.nan
    for column in MECHANISM_SCORE_COLUMNS:
        if column == "log10_recruited_surface_score":
            continue
        if column not in merged.columns:
            merged[column] = np.nan
    merged = _select_step06_candidate_scope(merged, config)
    if config.max_candidates is not None:
        merged = merged.sort_values(IDENTITY_COLUMNS).head(int(config.max_candidates)).copy()
    return merged, mechanisms


def build_heldout_current_errors(candidates: pd.DataFrame, config: Step06Config) -> pd.DataFrame:
    """Build candidate-level held-out-current audit rows.

    Step 04 currently exposes aggregate held-out metrics on accepted candidates.
    Step 06 expands those aggregates into one auditable row per candidate and
    ordered held-out sweep, preserving mechanism-label provenance and adding
    explicit prediction status/pass columns so missing evidence cannot be treated
    as a silent pass.
    """

    rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        pass_fraction = float(
            cand.get(
                "holdout_mean_pass_fraction",
                cand.get("mean_weighted_pass_fraction", np.nan),
            )
        )
        rmse = float(
            cand.get("holdout_mean_rmse_mV", cand.get("mean_trace_rmse_mV", np.nan))
        )
        mechanism_label_status = (
            "step05_label_available"
            if str(cand.get("cluster_claim_scope", "")) != "missing_step05_label"
            and str(cand.get("mechanism_cluster", "unlabeled")) != "unlabeled"
            else "missing_step05_label"
        )
        prediction_pass = bool(
            np.isfinite(pass_fraction)
            and pass_fraction >= config.min_holdout_pass_fraction
            and (not np.isfinite(rmse) or rmse <= config.max_trace_rmse_mV)
        )
        prediction_status = "predictive_pass" if prediction_pass else "prediction_limited"
        base = {c: cand.get(c) for c in IDENTITY_COLUMNS}
        for sweep, current_na in enumerate(VALID_CURRENTS, start=1):
            rows.append(
                {
                    **base,
                    "sweep": sweep,
                    "current_na": int(current_na),
                    "split_strategy": "leave_one_current_out",
                    "heldout_trace_rmse_mV": rmse,
                    "heldout_weighted_feature_pass_fraction": pass_fraction,
                    "holdout_status": "passed" if prediction_pass else "prediction_limited",
                    "prediction_status": prediction_status,
                    "prediction_pass": prediction_pass,
                    "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                    "mechanism_label_status": mechanism_label_status,
                }
            )
        for strategy, currents in {"low_to_high": [150, 175], "high_to_low": [50, 75]}.items():
            rows.append(
                {
                    **base,
                    "sweep": -1,
                    "current_na": int(np.mean(currents)),
                    "split_strategy": strategy,
                    "heldout_trace_rmse_mV": rmse,
                    "heldout_weighted_feature_pass_fraction": pass_fraction,
                    "holdout_status": "stress_test_screen",
                    "prediction_status": prediction_status,
                    "prediction_pass": prediction_pass,
                    "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                    "mechanism_label_status": mechanism_label_status,
                }
            )

    heldout = pd.DataFrame(rows)
    if config.require_candidate_level_heldout and not heldout.empty:
        expected = candidates[IDENTITY_COLUMNS].drop_duplicates().assign(_expected=1)
        observed = (
            heldout[heldout["split_strategy"].eq("leave_one_current_out")]
            .groupby(IDENTITY_COLUMNS, dropna=False)
            .size()
            .rename("n_heldout_sweeps")
            .reset_index()
        )
        audit = expected.merge(observed, how="left", on=IDENTITY_COLUMNS)
        incomplete = audit["n_heldout_sweeps"].fillna(0).lt(len(VALID_CURRENTS))
        if incomplete.any():
            missing_rows = audit.loc[incomplete, IDENTITY_COLUMNS].copy()
            missing_rows["split_strategy"] = "missing_step04_heldout_screen"
            missing_rows["prediction_status"] = "fit_only"
            missing_rows["prediction_pass"] = False
            missing_rows["mechanism_label_status"] = "missing_step04_heldout_screen"
            heldout = pd.concat([heldout, missing_rows], ignore_index=True, sort=False)
    return heldout


def simulate_candidate_feature_rows(candidates: pd.DataFrame, config: Step06Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    flux_rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        condition = str(cand["condition"])
        window_s = stim_window_seconds(condition)
        for sweep, current_na in enumerate(VALID_CURRENTS, start=1):
            base = {c: cand.get(c) for c in candidates.columns if c in cand}
            try:
                params = reconstruct_flat_params(cand.to_dict(), current_na=int(current_na), sweep=sweep)
                sim = simulate_with_hidden_outputs(params, {"experiment_type": protocol_condition(condition), "current_na": int(current_na), "t_eval_ms": _time_grid(config)})
                time_s = np.asarray(sim["t_ms"], dtype=float) / 1000.0
                features = extract_features_from_trace(time_s, np.asarray(sim["Vm"], dtype=float), onset_s=window_s[0], offset_s=window_s[1])
                flux = compute_flux_summary(sim, stim_window_s=window_s)
                for feature, value in features.items():
                    if feature in FEATURE_COLUMNS:
                        feature_rows.append({**{c: cand.get(c) for c in IDENTITY_COLUMNS}, "region": cand.get("region"), "condition": cand.get("condition"), "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"), "sweep": sweep, "current_na": int(current_na), "feature": feature, "predicted_value": value, "simulation_status": "ok", "failure_reason": ""})
                flux_rows.append({**base, "sweep": sweep, "current_na": int(current_na), **flux, "simulation_status": "ok", "failure_reason": ""})
            except Exception as exc:  # noqa: BLE001 - explicit failure rows are required
                feature_rows.append({**{c: cand.get(c) for c in IDENTITY_COLUMNS}, "region": cand.get("region"), "condition": cand.get("condition"), "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"), "sweep": sweep, "current_na": int(current_na), "feature": "simulation_failed", "predicted_value": np.nan, "simulation_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
                flux_rows.append({**base, "sweep": sweep, "current_na": int(current_na), "simulation_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(feature_rows), pd.DataFrame(flux_rows)


def build_prediction_intervals(
    feature_predictions: pd.DataFrame, config: Step06Config
) -> pd.DataFrame:
    ok = feature_predictions[
        (feature_predictions["simulation_status"] == "ok")
        & feature_predictions["predicted_value"].notna()
    ].copy()
    q_low, q_mid, q_high = config.prediction_interval_quantiles
    rows = []
    for keys, group in ok.groupby(
        ["region", "condition", "sweep", "current_na", "feature"], dropna=False
    ):
        vals = pd.to_numeric(group["predicted_value"], errors="coerce").dropna()
        if vals.empty:
            continue
        source_candidate_ids = ";".join(
            sorted(group["candidate_id"].astype(str).dropna().unique().tolist())
        )
        rows.append(
            dict(
                zip(["region", "condition", "sweep", "current_na", "feature"], keys),
                n_predictions=int(len(vals)),
                n_source_candidates=int(group["candidate_id"].nunique()),
                source_candidate_ids=source_candidate_ids,
                pi_lower=float(vals.quantile(q_low)),
                pi_median=float(vals.quantile(q_mid)),
                pi_upper=float(vals.quantile(q_high)),
                interval_quantiles=f"{q_low},{q_mid},{q_high}",
            )
        )
    return pd.DataFrame(rows)


def _load_thresholds(root: Path) -> pd.DataFrame:
    primary = root / "outputs" / "features" / "condition_region_sweep_thresholds.csv"
    pooled = root / "outputs" / "features" / "region_pooled_condition_sweep_thresholds.csv"
    frames = [pd.read_csv(p) for p in [primary, pooled] if p.exists()]
    if not frames:
        raise FileNotFoundError("Step 02 threshold tables are required for Step 06 PPC")
    return pd.concat(frames, ignore_index=True)


def _find_threshold(thresholds: pd.DataFrame, condition: str, region: str, sweep: int, feature: str) -> tuple[pd.Series | None, str]:
    for cond, reg, mode in [(condition, region, "region_specific"), (condition, "ALL", "region_pooled"), ("ALL", "ALL", "global_fallback")]:
        hit = thresholds[(thresholds["condition"].astype(str) == str(cond)) & (thresholds["region"].astype(str) == str(reg)) & (thresholds["sweep"].astype(int) == int(sweep)) & (thresholds["feature"].astype(str) == str(feature))]
        if not hit.empty:
            return hit.iloc[0], mode
    return None, "missing"


def compute_feature_distribution_ppc(project_root: Path | str, feature_predictions: pd.DataFrame) -> pd.DataFrame:
    thresholds = _load_thresholds(Path(project_root).resolve())
    ok = feature_predictions[feature_predictions["simulation_status"] == "ok"].copy()
    rows = []
    for keys, group in ok.groupby(["region", "condition", "sweep", "feature"], dropna=False):
        region, condition, sweep, feature = keys
        thr, mode = _find_threshold(thresholds, str(condition), str(region), int(sweep), str(feature))
        vals = pd.to_numeric(group["predicted_value"], errors="coerce").dropna()
        if thr is None or vals.empty:
            lower = upper = weight = np.nan
            coverage = np.nan
        else:
            lower, upper = float(thr["acceptable_lower"]), float(thr["acceptable_upper"])
            weight = float(thr.get("reliability_weight", 1.0))
            coverage = float(((vals >= lower) & (vals <= upper)).mean())
        rows.append({"region": region, "condition": condition, "sweep": int(sweep), "feature": feature, "n_predictions": int(len(vals)), "empirical_lower": lower, "empirical_upper": upper, "coverage_fraction": coverage, "reliability_weight": weight, "weighted_coverage": coverage * weight if np.isfinite(coverage) and np.isfinite(weight) else np.nan, "threshold_fallback": mode})
    return pd.DataFrame(rows)


def _feature_pass_fraction_for_sim(
    thresholds: pd.DataFrame,
    sim: Mapping[str, Any],
    *,
    region: str,
    condition: str,
    sweep: int,
    onset_s: float,
    offset_s: float,
) -> tuple[float, int, str]:
    """Score one perturbed Vm trace against Step 02 feature bands."""

    time_s = np.asarray(sim["t_ms"], dtype=float) / 1000.0
    features = extract_features_from_trace(
        time_s,
        np.asarray(sim["Vm"], dtype=float),
        onset_s=float(onset_s),
        offset_s=float(offset_s),
    )
    evaluated = 0
    passed = 0
    fallbacks: list[str] = []
    for feature, value in features.items():
        if feature not in FEATURE_COLUMNS or not np.isfinite(float(value)):
            continue
        threshold, fallback = _find_threshold(
            thresholds, condition=str(condition), region=str(region), sweep=int(sweep), feature=str(feature)
        )
        fallbacks.append(fallback)
        if threshold is None:
            continue
        evaluated += 1
        lower = float(threshold["acceptable_lower"])
        upper = float(threshold["acceptable_upper"])
        passed += int(lower <= float(value) <= upper)
    if evaluated == 0:
        return float("nan"), 0, "missing"
    fallback_status = (
        "region_specific"
        if "region_specific" in fallbacks
        else ("region_pooled" if "region_pooled" in fallbacks else "global_fallback")
    )
    return float(passed / evaluated), int(evaluated), fallback_status


def _duration_adjusted_paramdict(
    condition: str,
    current_na: int,
    params: Mapping[str, Any],
    factor: float,
) -> tuple[dict[str, Any], tuple[float, float]]:
    """Return a paramdict with the middle K bath interval duration scaled."""

    paramdict = build_paramdict(protocol_condition(condition), int(current_na), params)
    times = np.asarray(paramdict["external"]["K_bath"]["time"], dtype=float).copy()
    if len(times) < 3:
        raise ValueError("K bath protocol must contain baseline, stimulus, and recovery times")
    onset_ms = float(times[1])
    original_offset_ms = float(times[2])
    new_offset_ms = onset_ms + max(1.0, (original_offset_ms - onset_ms) * float(factor))
    times[2] = new_offset_ms
    paramdict["external"]["K_bath"]["time"] = times
    return paramdict, (onset_ms / 1000.0, new_offset_ms / 1000.0)


def _hidden_flux_plausible(flux: Mapping[str, Any]) -> bool:
    """Check that hidden-current fractions are finite and bounded."""

    fractions = [
        float(flux.get("gap_fraction", np.nan)),
        float(flux.get("kir_fraction", np.nan)),
        float(flux.get("leak_fraction", np.nan)),
    ]
    if not all(np.isfinite(f) for f in fractions):
        return False
    return all(-1e-9 <= f <= 1.0 + 1e-9 for f in fractions)


def run_perturbation_sweeps(
    candidates: pd.DataFrame, config: Step06Config, project_root: Path | str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(project_root).resolve()
    thresholds = _load_thresholds(root)
    current_list = (
        [int(config.perturbation_current_na)]
        if config.perturbation_current_na
        else [int(c) for c in VALID_CURRENTS]
    )
    for _, cand in candidates.iterrows():
        condition = str(cand["condition"])
        window_s = stim_window_seconds(condition)
        for current_na in current_list:
            sweep = (
                list(VALID_CURRENTS).index(current_na) + 1
                if current_na in VALID_CURRENTS
                else 1
            )
            nominal_peak = np.nan
            for name, factor in config.perturbation_factors.items():
                row_base = {
                    **{c: cand.get(c) for c in IDENTITY_COLUMNS},
                    "mechanism_cluster": cand.get("mechanism_cluster", "unlabeled"),
                    "dominant_mechanism": cand.get("dominant_mechanism", "unknown"),
                    "buffering_phenotype": cand.get("buffering_phenotype", "unlabeled"),
                    "current_na": int(current_na),
                    "sweep": int(sweep),
                    "perturbation": name,
                    "perturbation_factor": float(factor),
                }
                try:
                    params = reconstruct_flat_params(
                        cand.to_dict(),
                        current_na=current_na,
                        sweep=sweep,
                    )
                    if name.startswith("eps_scale"):
                        params["eps"] = float(params["eps"]) * float(factor)
                    elif name.startswith("current_scale"):
                        params["K_bath_value_middle"] = (
                            float(params["K_bath_value_middle"]) * float(factor)
                        )
                    if name.startswith("stimulus_duration"):
                        paramdict, active_window_s = _duration_adjusted_paramdict(
                            condition, int(current_na), params, float(factor)
                        )
                        t_grid = _time_grid(config)
                    else:
                        paramdict = build_paramdict(
                            protocol_condition(condition), int(current_na), params
                        )
                        active_window_s = window_s
                        t_grid = _time_grid(config)
                    if name.startswith("baseline_K_o"):
                        paramdict["external"]["K_o0"] = (
                            float(paramdict["external"]["K_o0"]) * float(factor)
                        )
                    sim = simulate_odeint(
                        paramdict,
                        {
                            "experiment_type": protocol_condition(condition),
                            "current_na": int(current_na),
                            "t_eval_ms": t_grid,
                        },
                        return_hidden=True,
                    )
                    flux = compute_flux_summary(sim, stim_window_s=active_window_s)
                    if name == "nominal":
                        nominal_peak = float(flux.get("K_o_peak", np.nan))
                    peak_ratio = (
                        float(flux.get("K_o_peak", np.nan)) / nominal_peak
                        if np.isfinite(nominal_peak) and abs(nominal_peak) > 1e-12
                        else np.nan
                    )
                    vm_feature_pass, n_vm_features, feature_fallback = _feature_pass_fraction_for_sim(
                        thresholds,
                        sim,
                        region=str(cand.get("region")),
                        condition=condition,
                        sweep=sweep,
                        onset_s=active_window_s[0],
                        offset_s=active_window_s[1],
                    )
                    hidden_flux_plausible = _hidden_flux_plausible(flux)
                    functional_buffering_pass = bool(
                        np.isfinite(float(flux.get("K_o_recovery_error", np.nan)))
                        and abs(float(flux.get("K_o_recovery_error", np.nan)))
                        <= config.robustness_recovery_error_max
                        and np.isfinite(float(flux.get("K_o_peak", np.nan)))
                        and float(flux.get("K_o_peak", np.nan))
                        <= config.robustness_K_o_peak_max_mM
                        and np.isfinite(float(flux.get("K_o_final", np.nan)))
                        and abs(float(flux.get("K_o_final", np.nan)))
                        <= config.robustness_K_o_final_max_mM
                        and (
                            not np.isfinite(peak_ratio)
                            or abs(peak_ratio) <= config.robustness_peak_ratio_max
                        )
                        and (
                            not np.isfinite(vm_feature_pass)
                            or vm_feature_pass
                            >= config.min_perturbation_feature_pass_fraction
                        )
                        and hidden_flux_plausible
                    )
                    rows.append(
                        {
                            **row_base,
                            **flux,
                            "K_o_peak_ratio_to_nominal": peak_ratio,
                            "Vm_feature_pass_fraction": vm_feature_pass,
                            "n_Vm_features_scored": n_vm_features,
                            "Vm_feature_threshold_fallback": feature_fallback,
                            "hidden_flux_plausible": hidden_flux_plausible,
                            "active_stim_window_start_s": float(active_window_s[0]),
                            "active_stim_window_end_s": float(active_window_s[1]),
                            "robust_under_perturbation": functional_buffering_pass,
                            "functional_buffering_pass": functional_buffering_pass,
                            "perturbation_status": "evaluated",
                            "simulation_status": "ok",
                            "failure_reason": "",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            **row_base,
                            "robust_under_perturbation": False,
                            "functional_buffering_pass": False,
                            "Vm_feature_pass_fraction": np.nan,
                            "n_Vm_features_scored": 0,
                            "Vm_feature_threshold_fallback": "missing",
                            "hidden_flux_plausible": False,
                            "active_stim_window_start_s": float(window_s[0]),
                            "active_stim_window_end_s": float(window_s[1]),
                            "perturbation_status": "failed",
                            "simulation_status": "failed",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return pd.DataFrame(rows)


def build_robustness_summary(
    heldout: pd.DataFrame,
    ppc: pd.DataFrame,
    perturb: pd.DataFrame,
    candidates: pd.DataFrame,
    config: Step06Config,
) -> pd.DataFrame:
    rows = []
    groups = candidates.groupby(["mechanism_cluster", "region", "condition"], dropna=False)
    for (cluster, region, condition), group in groups:
        h = heldout[
            (heldout["mechanism_cluster"].astype(str) == str(cluster))
            & (heldout["region"].astype(str) == str(region))
            & (heldout["condition"].astype(str) == str(condition))
            & (heldout["split_strategy"] == "leave_one_current_out")
        ]
        p = ppc[
            (ppc["region"].astype(str) == str(region))
            & (ppc["condition"].astype(str) == str(condition))
        ]
        q = perturb[
            (perturb["mechanism_cluster"].astype(str) == str(cluster))
            & (perturb["region"].astype(str) == str(region))
            & (perturb["condition"].astype(str) == str(condition))
            & (perturb["perturbation"] != "nominal")
            & (perturb["perturbation_status"] == "evaluated")
        ]
        holdout_pass = float(_as_bool(h["prediction_pass"]).mean()) if not h.empty else np.nan
        ppc_cov = float(p["weighted_coverage"].mean(skipna=True)) if not p.empty else np.nan
        robust_frac = (
            float(_as_bool(q["functional_buffering_pass"]).mean()) if not q.empty else np.nan
        )
        perturb_feature_pass = (
            float(pd.to_numeric(q["Vm_feature_pass_fraction"], errors="coerce").mean(skipna=True))
            if "Vm_feature_pass_fraction" in q and not q.empty
            else np.nan
        )
        hidden_flux_plausible_fraction = (
            float(_as_bool(q["hidden_flux_plausible"]).mean())
            if "hidden_flux_plausible" in q and not q.empty
            else np.nan
        )
        evidence = (
            str(group["cluster_evidence_status"].iloc[0])
            if "cluster_evidence_status" in group
            else "insufficient_evidence"
        )
        if evidence == "insufficient_evidence":
            validation_label = "insufficient_evidence"
        elif (
            np.isfinite(holdout_pass)
            and holdout_pass >= config.min_robust_fraction
            and np.isfinite(robust_frac)
            and robust_frac >= config.min_robust_fraction
        ):
            validation_label = "predictive_supported"
        elif np.isfinite(holdout_pass) and holdout_pass > 0:
            validation_label = "prediction_limited"
        else:
            validation_label = "fit_only"

        step06_screen_claim = (
            "candidate_regime_supported_by_step06_predictive_perturbation_screen"
            if validation_label == "predictive_supported"
            else "mechanism_candidate_not_supported_by_step06_screen"
        )
        final_biological_degeneracy_claim_allowed = bool(
            config.final_degeneracy_claim_allowed
            and validation_label == "predictive_supported"
        )
        score_parts = [
            holdout_pass if np.isfinite(holdout_pass) else np.nan,
            ppc_cov if np.isfinite(ppc_cov) else np.nan,
            robust_frac if np.isfinite(robust_frac) else np.nan,
            perturb_feature_pass if np.isfinite(perturb_feature_pass) else np.nan,
            hidden_flux_plausible_fraction
            if np.isfinite(hidden_flux_plausible_fraction)
            else np.nan,
        ]
        finite_parts = [float(x) for x in score_parts if np.isfinite(x)]
        biological_description_score = (
            float(np.clip(np.mean(finite_parts), 0.0, 1.0))
            if finite_parts
            else np.nan
        )
        phenotype_labels = (
            ";".join(sorted(group["buffering_phenotype"].astype(str).dropna().unique()))
            if "buffering_phenotype" in group
            else "unlabeled"
        )
        claim_scope_after_step06 = (
            f"{step06_screen_claim}; final degeneracy wording remains prohibited "
            "until assumption-sensitivity and parameter-plausibility checks pass"
        )
        rows.append(
            {
                "mechanism_cluster": cluster,
                "region": region,
                "condition": condition,
                "n_candidates": int(len(group)),
                "n_cells": int(group["file_id"].nunique()),
                "holdout_pass_fraction": holdout_pass,
                "mean_weighted_ppc_coverage": ppc_cov,
                "perturbation_robust_fraction": robust_frac,
                "perturbation_vm_feature_pass_fraction": perturb_feature_pass,
                "hidden_flux_plausible_fraction": hidden_flux_plausible_fraction,
                "biological_description_score": biological_description_score,
                "buffering_phenotype_labels": phenotype_labels,
                "step05_evidence_status": evidence,
                "validation_label": validation_label,
                "step06_screen_claim": step06_screen_claim,
                "final_biological_degeneracy_claim_allowed": final_biological_degeneracy_claim_allowed,
                "degeneracy_claim_allowed": final_biological_degeneracy_claim_allowed,
                "claim_scope_after_step06": claim_scope_after_step06,
            }
        )
    return pd.DataFrame(rows)


def compare_step06_runtime_presets(
    project_root: Path | str, max_candidates: int = 1
) -> pd.DataFrame:
    rows = []
    for preset, tp in [("coarse", 50), ("default", 80)]:
        cfg = Step06Config(
            max_candidates=max_candidates, time_points=tp, write_outputs=False
        )
        t0 = time.perf_counter()
        try:
            run_step06_predictive_validation(project_root, cfg)
            status = "ok"
            error = ""
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "preset": preset,
                "time_points": tp,
                "elapsed_seconds": elapsed,
                "status": status,
                "error": error,
            }
        )
    df = pd.DataFrame(rows)
    fastest = df.loc[df["elapsed_seconds"].idxmin(), "preset"] if not df.empty else "coarse"
    df["recommendation"] = np.where(
        df["preset"].eq(fastest),
        f"{fastest}_recommended_for_tests",
        "default_or_coarse_available_for_manuscript_rerun",
    )
    return df


def compare_step06_performance(
    project_root: Path | str, max_candidates: int = 1
) -> pd.DataFrame:
    """Backward-compatible alias for existing Step 06 notebooks/tests."""

    return compare_step06_runtime_presets(project_root, max_candidates=max_candidates)


def run_step06_predictive_validation(project_root: Path | str, config: Step06Config | None = None, output_dir: Path | str | None = None) -> dict[str, pd.DataFrame | dict[str, Any]]:
    config = config or Step06Config()
    root = Path(project_root).resolve()
    out_dir = _ensure_dir(Path(output_dir).resolve() if output_dir is not None else root / "outputs" / OUTPUT_SUBDIR)
    t0 = time.perf_counter()
    candidates, _ = load_step06_inputs(root, config)
    heldout = build_heldout_current_errors(candidates, config)
    feature_predictions, nominal_flux = simulate_candidate_feature_rows(candidates, config)
    intervals = build_prediction_intervals(feature_predictions, config)
    ppc = compute_feature_distribution_ppc(root, feature_predictions)
    perturb = run_perturbation_sweeps(candidates, config, root)
    robustness = build_robustness_summary(heldout, ppc, perturb, candidates, config)
    perf = compare_step06_runtime_presets(root) if config.write_outputs else pd.DataFrame()
    summary = {
        "step_name": "Step 06 — Predictive validation and perturbation robustness",
        "config": asdict(config),
        "n_candidates": int(len(candidates)),
        "n_heldout_rows": int(len(heldout)),
        "n_prediction_interval_rows": int(len(intervals)),
        "n_ppc_rows": int(len(ppc)),
        "n_perturbation_rows": int(len(perturb)),
        "validation_labels": sorted(robustness["validation_label"].astype(str).unique().tolist()) if not robustness.empty else [],
        "headline_claim_scope": "Mechanism regimes require predictive_supported labels before any degeneracy language; Step 06 alone keeps claims conservative.",
        "elapsed_seconds": time.perf_counter() - t0,
    }
    if config.write_outputs:
        heldout.to_csv(out_dir / "heldout_current_errors.csv", index=False)
        intervals.to_csv(out_dir / "prediction_intervals.csv", index=False)
        ppc.to_csv(out_dir / "feature_distribution_ppc.csv", index=False)
        perturb.to_csv(out_dir / "perturbation_sweeps.csv", index=False)
        robustness.to_csv(out_dir / "robustness_summary.csv", index=False)
        feature_predictions.to_csv(out_dir / "candidate_feature_predictions.csv", index=False)
        nominal_flux.to_csv(out_dir / "nominal_flux_predictions.csv", index=False)
        if not perf.empty:
            perf.to_csv(out_dir / "performance_benchmark.csv", index=False)
        (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"heldout_current_errors": heldout, "prediction_intervals": intervals, "feature_distribution_ppc": ppc, "perturbation_sweeps": perturb, "robustness_summary": robustness, "candidate_feature_predictions": feature_predictions, "nominal_flux_predictions": nominal_flux, "performance_benchmark": perf, "analysis_summary": summary}
