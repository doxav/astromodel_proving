"""Biological MFA/MFA+BA-like perturbations on legacy mechanism categories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .astro_model import simulate_with_hidden_outputs
from .atf_features import FEATURE_COLUMNS, extract_features_from_trace
from .contracts import canonical_condition, protocol_condition
from .functional_mapping import (
    apply_efficiency_quadrants,
    classify_sigmoid_state_change,
    compare_direction_to_target,
    direction_label,
    extract_ko_kinetic_features,
    extract_sigmoid_state_features,
)
from .parameter_space import coordinate_value, set_coordinate
from .postfit_sqlite import FILTER_BASELINE_FOLD_GRID, run_step01_postfit_sqlite
from .protocols import stim_window_seconds
from .step05_mechanistic_decomposition import Step05Config, run_step05_mechanistic_decomposition


ONE_D_PARAMETERS: tuple[str, ...] = ("P_gap_eff", "gamma_s_eff", "zth", "zs", "gki")
PAIR_SWEEPS: tuple[tuple[str, str], ...] = (
    ("P_gap_eff", "gamma_s_eff"),
    ("P_gap_eff", "zth"),
    ("gamma_s_eff", "zth"),
    ("zth", "zs"),
    ("gki", "P_gap_eff"),
)
TARGET_FEATURES: tuple[str, ...] = (
    "rise_slope_mV_per_s",
    "rise_tau_s",
    "decay_slope_mV_per_s",
    "decay_tau_s",
    "return_slope_mV_per_s",
    "stim_end_depolarization_mV",
    "peak_depolarization_mV",
    "undershoot_magnitude_mV",
)
REGIONAL_ALIGNMENT_INTERPRETATION = (
    "legacy_category_direction_compared_to_DH_minus_VH_delta_target_not_region_assigned"
)


@dataclass(slots=True)
class BiologicalPerturbationConfig:
    """Configuration for first-pass legacy biological perturbation sweeps."""

    time_points: int = 80
    t_final_ms: float = 50_000.0
    max_configs_per_category: int | None = 2
    pair_grid_factors: tuple[float, ...] = (0.75, 1.0, 1.25)
    run_pair_sweeps: bool = True
    write_outputs: bool = True


def _time_grid(config: BiologicalPerturbationConfig) -> np.ndarray:
    return np.linspace(0.0, float(config.t_final_ms), int(config.time_points), dtype=float)


def _ensure_legacy_inputs(root: Path) -> None:
    if not (root / "outputs" / "postfit_sqlite" / "legacy_configuration_library.csv").exists():
        run_step01_postfit_sqlite(root)
    if not (root / "outputs" / "legacy_mechanisms" / "legacy_fit_mechanisms.csv").exists():
        run_step05_mechanistic_decomposition(
            root,
            Step05Config(
                time_points=80,
                bootstrap_iterations=0,
                run_legacy_mapping=True,
                write_outputs=True,
            ),
        )


def _legacy_params(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "gki",
        "pk",
        "d",
        "gs",
        "gt",
        "zth",
        "zs",
        "eps",
        "gl_a",
        "wo",
        "w_a",
        "ca",
        "Va_l",
        "Va_s",
        "switching_function",
        "K_bath_value_middle",
        "eps_middle",
        "wo_middle",
    ]
    params = {key: row[key] for key in keys if key in row and pd.notna(row[key])}
    params.setdefault("switching_function", "sigmoid")
    return params


def _context_for_parameter(condition: str, parameter: str) -> str:
    cond = canonical_condition(condition)
    if cond == "CONTROL" and parameter != "gki":
        return "MFA_like_from_control_legacy"
    if cond == "MFA" and parameter != "gki":
        return "MFA_like_from_mfa_legacy"
    if cond == "MFA" and parameter == "gki":
        return "MFA_BA_from_MFA_legacy"
    if cond == "CONTROL" and parameter == "gki":
        return "MFA_BA_stacked_on_control_legacy"
    return "MFA_BA_from_MFA_legacy"


def _experimental_contrast_for_context(context: str) -> str:
    if context == "MFA_BA_from_MFA_legacy":
        return "MFA_to_MFA_BA"
    if context == "MFA_BA_stacked_on_control_legacy":
        return "CONTROL_to_MFA_BA"
    return "CONTROL_to_MFA"


def _select_category_representatives(
    categories: pd.DataFrame,
    config: BiologicalPerturbationConfig,
) -> pd.DataFrame:
    group_cols = [
        "condition",
        "current_na",
        "sigmoid_state_at_sim_end_10_90",
        "sigmoid_state_at_stim_end_10_90",
        "temporal_recruitment_class",
        "gj_ionic_state_10_90",
    ]
    available = [col for col in group_cols if col in categories.columns]
    ranked = categories.sort_values(["condition", "current_na", "rank_in_db", "objective"]).copy()
    if config.max_configs_per_category is None:
        selected = ranked
    else:
        selected = (
            ranked.groupby(available, dropna=False, as_index=False)
            .head(int(config.max_configs_per_category))
            .reset_index(drop=True)
        )
    selected["category_id"] = (
        selected[available].astype(str).agg("|".join, axis=1)
        if available
        else "all_legacy"
    )
    selected["biological_perturbation_selection_policy"] = (
        "top_ranked_per_sigmoid_temporal_category"
    )
    selected["max_configs_per_category"] = (
        -1 if config.max_configs_per_category is None else int(config.max_configs_per_category)
    )
    return selected


def _simulate_features(
    params: Mapping[str, Any],
    *,
    condition: str,
    current_na: int,
    config: BiologicalPerturbationConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    window_s = stim_window_seconds(condition)
    sim = simulate_with_hidden_outputs(
        params,
        {
            "experiment_type": protocol_condition(condition),
            "current_na": int(current_na),
            "t_eval_ms": _time_grid(config),
        },
    )
    time_s = np.asarray(sim["t_ms"], dtype=float) / 1000.0
    vm_features = extract_features_from_trace(
        time_s,
        np.asarray(sim["Vm"], dtype=float),
        onset_s=window_s[0],
        offset_s=window_s[1],
    )
    ko_features = extract_ko_kinetic_features(
        time_s,
        np.asarray(sim["derived"]["K_o"], dtype=float),
        onset_s=window_s[0],
        offset_s=window_s[1],
    )
    sigmoid = extract_sigmoid_state_features(sim, stim_window_s=window_s)
    return vm_features, ko_features, sigmoid


def _perturb_coordinate(
    params: Mapping[str, Any],
    *,
    condition: str,
    current_na: int,
    parameter: str,
    factor: float,
) -> tuple[dict[str, Any], float, float]:
    baseline = coordinate_value(params, parameter, condition=condition, current_na=current_na)
    perturbed_value = max(float(baseline) * float(factor), 1e-12)
    return set_coordinate(params, parameter, perturbed_value), float(baseline), float(perturbed_value)


def _base_identity(row: Mapping[str, Any], context: str) -> dict[str, Any]:
    return {
        "source_scope": str(row.get("source_scope", "legacy_single_current_optuna")),
        "perturbation_context": context,
        "baseline_condition": canonical_condition(str(row["condition"])),
        "simulated_protocol_condition": protocol_condition(str(row["condition"])),
        "baseline_candidate_id": str(row["candidate_id"]),
        "db_name": str(row["db_name"]),
        "trial_number": int(row["trial_number"]),
        "current_na": int(row["current_na"]),
        "category_id": str(row.get("category_id", "uncategorized")),
        "factor_source": "filter_baseline_fold_grid",
        "legacy_selection_rule": str(row.get("legacy_selection_rule", "top_n_by_objective")),
        "legacy_configuration_status": str(
            row.get("legacy_configuration_status", "legacy_top300_optuna_trial")
        ),
        "biological_perturbation_selection_policy": str(
            row.get("biological_perturbation_selection_policy", "")
        ),
        "max_configs_per_category": int(row.get("max_configs_per_category", -1)),
        "baseline_sigmoid_state_at_stim_end_10_90": str(
            row.get("sigmoid_state_at_stim_end_10_90", "undefined")
        ),
        "baseline_sigmoid_state_at_sim_end_10_90": str(
            row.get("sigmoid_state_at_sim_end_10_90", "undefined")
        ),
        "baseline_temporal_recruitment_class": str(
            row.get("temporal_recruitment_class", "undefined")
        ),
        "baseline_Ko_efficiency_score": row.get("Ko_efficiency_score", np.nan),
        "baseline_Ko_efficiency_quadrant": row.get("Ko_efficiency_quadrant", "undefined"),
    }


def _build_perturbation_record(
    row: Mapping[str, Any],
    *,
    perturbed_params: Mapping[str, Any],
    context: str,
    parameter: str,
    factor: float,
    baseline_value: float,
    perturbed_value: float,
    config: BiologicalPerturbationConfig,
    parameter_2: str | None = None,
    factor_2: float | None = None,
    baseline_value_2: float | None = None,
    perturbed_value_2: float | None = None,
) -> dict[str, Any]:
    base = _base_identity(row, context)
    condition = canonical_condition(str(row["condition"]))
    try:
        vm, ko, sigmoid = _simulate_features(
            perturbed_params,
            condition=condition,
            current_na=int(row["current_na"]),
            config=config,
        )
        record: dict[str, Any] = {
            **base,
            "perturbed_parameter": parameter,
            "perturbation_factor": float(factor),
            "baseline_value": float(baseline_value),
            "perturbed_value": float(perturbed_value),
            "perturbed_parameter_1": parameter,
            "factor_1": float(factor),
            "baseline_value_1": float(baseline_value),
            "perturbed_value_1": float(perturbed_value),
            "perturbed_parameter_2": parameter_2 or "",
            "factor_2": np.nan if factor_2 is None else float(factor_2),
            "baseline_value_2": np.nan if baseline_value_2 is None else float(baseline_value_2),
            "perturbed_value_2": np.nan if perturbed_value_2 is None else float(perturbed_value_2),
            "simulation_status": "ok",
            "failure_reason": "",
            **{f"perturbed_{key}": value for key, value in vm.items() if key in FEATURE_COLUMNS},
            **{f"perturbed_{key}": value for key, value in ko.items()},
            **{f"perturbed_{key}": value for key, value in sigmoid.items()},
        }
        for feature in TARGET_FEATURES:
            baseline_feature = row.get(f"baseline_{feature}", np.nan)
            perturbed_feature = record.get(f"perturbed_{feature}", np.nan)
            record[f"baseline_{feature}"] = baseline_feature
            record[f"delta_{feature}"] = (
                float(perturbed_feature) - float(baseline_feature)
                if np.isfinite(float(perturbed_feature)) and np.isfinite(float(baseline_feature))
                else np.nan
            )
            record[f"direction_{feature}"] = direction_label(record[f"delta_{feature}"])
        record["Ko_efficiency_score"] = record.get("perturbed_Ko_efficiency_score", np.nan)
        record["delta_Ko_efficiency_score"] = (
            float(record["Ko_efficiency_score"]) - float(row.get("Ko_efficiency_score", np.nan))
            if np.isfinite(float(record.get("Ko_efficiency_score", np.nan)))
            and np.isfinite(float(row.get("Ko_efficiency_score", np.nan)))
            else np.nan
        )
        record["direction_Ko_efficiency_score"] = direction_label(record["delta_Ko_efficiency_score"])
        record["sigmoid_state_change"] = classify_sigmoid_state_change(
            str(row.get("sigmoid_state_at_stim_end_10_90", "undefined")),
            str(row.get("sigmoid_state_at_sim_end_10_90", "undefined")),
            str(record.get("perturbed_sigmoid_state_at_stim_end_10_90", "undefined")),
            str(record.get("perturbed_sigmoid_state_at_sim_end_10_90", "undefined")),
            str(record.get("perturbed_temporal_recruitment_class", "undefined")),
        )
        record["sigmoid_phase_transition"] = (
            not str(record["sigmoid_state_change"]).startswith("unchanged")
            and str(record["sigmoid_state_change"]) != "undefined_or_failed"
        )
        return record
    except Exception as exc:  # noqa: BLE001 - explicit failure rows are required
        return {
            **base,
            "perturbed_parameter": parameter,
            "perturbation_factor": float(factor),
            "baseline_value": float(baseline_value),
            "perturbed_value": float(perturbed_value),
            "perturbed_parameter_1": parameter,
            "factor_1": float(factor),
            "perturbed_parameter_2": parameter_2 or "",
            "factor_2": np.nan if factor_2 is None else float(factor_2),
            "simulation_status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "sigmoid_state_change": "undefined_or_failed",
            "sigmoid_phase_transition": False,
        }


def _baseline_feature_rows(
    selected: pd.DataFrame,
    config: BiologicalPerturbationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        params = _legacy_params(row)
        base = row.to_dict()
        try:
            vm, _ko, _sigmoid = _simulate_features(
                params,
                condition=canonical_condition(str(row["condition"])),
                current_na=int(row["current_na"]),
                config=config,
            )
            for feature in TARGET_FEATURES:
                base[f"baseline_{feature}"] = vm.get(feature, np.nan)
            base["baseline_simulation_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            for feature in TARGET_FEATURES:
                base[f"baseline_{feature}"] = np.nan
            base["baseline_simulation_status"] = "failed"
            base["baseline_failure_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(base)
    return pd.DataFrame(rows)


def _one_dimensional_sweeps(
    selected: pd.DataFrame,
    config: BiologicalPerturbationConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        condition = canonical_condition(str(row["condition"]))
        params = _legacy_params(row)
        for parameter in ONE_D_PARAMETERS:
            if parameter == "gki" and condition == "MFA_BA":
                continue
            if parameter != "gki" and condition == "MFA_BA":
                continue
            context = _context_for_parameter(condition, parameter)
            for factor in FILTER_BASELINE_FOLD_GRID[parameter]:
                perturbed, baseline_value, perturbed_value = _perturb_coordinate(
                    params,
                    condition=condition,
                    current_na=int(row["current_na"]),
                    parameter=parameter,
                    factor=float(factor),
                )
                records.append(
                    _build_perturbation_record(
                        row,
                        perturbed_params=perturbed,
                        context=context,
                        parameter=parameter,
                        factor=float(factor),
                        baseline_value=baseline_value,
                        perturbed_value=perturbed_value,
                        config=config,
                    )
                )
    return pd.DataFrame(records)


def _pair_sweeps(
    selected: pd.DataFrame,
    config: BiologicalPerturbationConfig,
) -> pd.DataFrame:
    if not config.run_pair_sweeps:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        condition = canonical_condition(str(row["condition"]))
        if condition == "MFA_BA":
            continue
        params = _legacy_params(row)
        for parameter_1, parameter_2 in PAIR_SWEEPS:
            if "gki" in {parameter_1, parameter_2} and condition != "MFA":
                context = "MFA_BA_stacked_on_control_legacy"
            elif "gki" in {parameter_1, parameter_2}:
                context = "MFA_BA_from_MFA_legacy"
            else:
                context = _context_for_parameter(condition, parameter_1)
            for factor_1 in config.pair_grid_factors:
                for factor_2 in config.pair_grid_factors:
                    p1, base_1, value_1 = _perturb_coordinate(
                        params,
                        condition=condition,
                        current_na=int(row["current_na"]),
                        parameter=parameter_1,
                        factor=float(factor_1),
                    )
                    p2, base_2, value_2 = _perturb_coordinate(
                        p1,
                        condition=condition,
                        current_na=int(row["current_na"]),
                        parameter=parameter_2,
                        factor=float(factor_2),
                    )
                    records.append(
                        _build_perturbation_record(
                            row,
                            perturbed_params=p2,
                            context=context,
                            parameter=parameter_1,
                            factor=float(factor_1),
                            baseline_value=base_1,
                            perturbed_value=value_1,
                            config=config,
                            parameter_2=parameter_2,
                            factor_2=float(factor_2),
                            baseline_value_2=base_2,
                            perturbed_value_2=value_2,
                        )
                    )
    return pd.DataFrame(records)


def _target_direction(
    target_table: pd.DataFrame,
    *,
    contrast: str,
    feature: str,
    scope: str | None = None,
) -> tuple[str, str]:
    """Return the target direction and target scope for a contrast-feature row."""

    if target_table.empty:
        return "undefined", ""
    subset = target_table[
        (target_table["experimental_contrast"].astype(str) == str(contrast))
        & (target_table["feature"].astype(str) == str(feature))
    ].copy()
    if scope is not None and "scope" in subset.columns:
        subset = subset[subset["scope"].astype(str) == scope]
    if subset.empty:
        return "undefined", ""
    row = subset.iloc[0]
    target_scope = str(row.get("scope", scope or ""))
    return str(row.get("experimental_direction", "undefined")), target_scope


def _direction_summary(
    sweeps: pd.DataFrame,
    target_table: pd.DataFrame,
    regional_target_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compare simulated perturbation directions with global and regional ATF targets."""

    rows: list[dict[str, Any]] = []
    if not target_table.empty and "scope" in target_table.columns:
        target_lookup = target_table[target_table["scope"].astype(str).eq("region_blind")].copy()
    else:
        target_lookup = target_table.copy()
    if not regional_target_table.empty and "scope" in regional_target_table.columns:
        regional_lookup = regional_target_table[
            regional_target_table["scope"].astype(str).eq("delta_of_delta_DH_minus_VH")
        ].copy()
    else:
        regional_lookup = regional_target_table.copy()
    for _, row in sweeps.iterrows():
        contrast = _experimental_contrast_for_context(str(row["perturbation_context"]))
        for feature in TARGET_FEATURES:
            sim_direction = str(row.get(f"direction_{feature}", "undefined"))
            exp_direction, target_scope = _target_direction(
                target_lookup,
                contrast=contrast,
                feature=feature,
            )
            regional_direction, regional_scope = _target_direction(
                regional_lookup,
                contrast=contrast,
                feature=feature,
                scope="delta_of_delta_DH_minus_VH",
            )
            rows.append(
                {
                    "perturbation_context": row["perturbation_context"],
                    "baseline_candidate_id": row["baseline_candidate_id"],
                    "category_id": row["category_id"],
                    "perturbed_parameter": row["perturbed_parameter"],
                    "perturbation_factor": row["perturbation_factor"],
                    "experimental_contrast": contrast,
                    "feature": feature,
                    "simulated_direction": sim_direction,
                    "experimental_direction": exp_direction,
                    "experimental_target_scope": target_scope,
                    "direction_match_status": compare_direction_to_target(
                        sim_direction, exp_direction
                    ),
                    "regional_experimental_direction": regional_direction,
                    "regional_target_scope": regional_scope,
                    "regional_match_status": compare_direction_to_target(
                        sim_direction, regional_direction
                    ),
                    "regional_match_interpretation": (
                        REGIONAL_ALIGNMENT_INTERPRETATION
                        if regional_scope
                        else "regional_target_missing"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sigmoid_summary(sweeps: pd.DataFrame) -> pd.DataFrame:
    if sweeps.empty:
        return pd.DataFrame()
    return (
        sweeps.groupby(
            [
                "perturbation_context",
                "category_id",
                "perturbed_parameter",
                "sigmoid_state_change",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_rows=("baseline_candidate_id", "size"),
            n_candidates=("baseline_candidate_id", "nunique"),
            phase_transition_fraction=("sigmoid_phase_transition", "mean"),
        )
        .sort_values(["perturbation_context", "category_id", "perturbed_parameter"])
        .reset_index(drop=True)
    )


def _phase_portrait_points(pair_sweeps: pd.DataFrame) -> pd.DataFrame:
    if pair_sweeps.empty:
        return pd.DataFrame()
    cols = [
        "perturbation_context",
        "baseline_candidate_id",
        "category_id",
        "perturbed_parameter_1",
        "factor_1",
        "perturbed_parameter_2",
        "factor_2",
        "perturbed_sigmoid_state_at_sim_end_10_90",
        "sigmoid_state_change",
        "sigmoid_phase_transition",
        "Ko_efficiency_score",
        "delta_Ko_efficiency_score",
        "direction_Ko_efficiency_score",
    ]
    return pair_sweeps[[c for c in cols if c in pair_sweeps.columns]].copy()


def run_legacy_biological_perturbations(
    project_root: str | Path,
    config: BiologicalPerturbationConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Run first-pass MFA/MFA+BA-like perturbations on legacy categories."""

    root = Path(project_root).resolve()
    cfg = config or BiologicalPerturbationConfig()
    out_dir = Path(output_dir).resolve() if output_dir is not None else root / "outputs" / "legacy_perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_legacy_inputs(root)

    categories = pd.read_csv(root / "outputs" / "legacy_mechanisms" / "legacy_mechanism_categories.csv")
    fit = pd.read_csv(root / "outputs" / "legacy_mechanisms" / "legacy_fit_mechanisms.csv")
    thresholds = pd.read_csv(root / "outputs" / "legacy_mechanisms" / "legacy_efficiency_thresholds.csv")
    merged = categories.merge(
        fit.drop_duplicates("candidate_id"),
        on="candidate_id",
        how="left",
        suffixes=("", "_fit"),
    )
    selected = _select_category_representatives(merged, cfg)
    selected = _baseline_feature_rows(selected, cfg)
    one_d = _one_dimensional_sweeps(selected, cfg)
    pairs = _pair_sweeps(selected, cfg)
    combined_for_eff = pd.concat([one_d, pairs], ignore_index=True, sort=False)
    if not combined_for_eff.empty and not thresholds.empty:
        eff_source = pd.DataFrame(
            {
                "source_scope": combined_for_eff["source_scope"],
                "current_na": combined_for_eff["current_na"],
                "Ko_rise_rate_mM_per_s": combined_for_eff.get("perturbed_Ko_rise_rate_mM_per_s"),
                "Ko_decay_rate_abs_mM_per_s": combined_for_eff.get("perturbed_Ko_decay_rate_abs_mM_per_s"),
                "Ko_efficiency_score": combined_for_eff.get("perturbed_Ko_efficiency_score"),
                "Ko_efficiency_status": combined_for_eff.get("perturbed_Ko_efficiency_status"),
            }
        )
        labeled = apply_efficiency_quadrants(eff_source, thresholds)
        for column in [
            "Ko_rise_speed_class",
            "Ko_decay_speed_class",
            "Ko_efficiency_quadrant",
            "Ko_rise_fast_slow_cutoff",
            "Ko_decay_fast_slow_cutoff",
            "Ko_efficiency_threshold_source",
            "Ko_efficiency_threshold_interpretation",
        ]:
            if column in labeled.columns:
                combined_for_eff[column] = labeled[column]
    one_d = combined_for_eff[combined_for_eff["perturbed_parameter_2"].fillna("").eq("")].copy()
    pairs = combined_for_eff[~combined_for_eff["perturbed_parameter_2"].fillna("").eq("")].copy()
    targets_path = root / "outputs" / "features" / "experimental_kinetic_direction_targets.csv"
    regional_targets_path = root / "outputs" / "features" / "region_specific_perturbation_direction_targets.csv"
    targets = pd.read_csv(targets_path) if targets_path.exists() else pd.DataFrame()
    regional_targets = pd.read_csv(regional_targets_path) if regional_targets_path.exists() else pd.DataFrame()
    direction_summary = (
        _direction_summary(one_d, targets, regional_targets)
        if not one_d.empty and (not targets.empty or not regional_targets.empty)
        else pd.DataFrame()
    )
    sigmoid_summary = _sigmoid_summary(combined_for_eff)
    phase_points = _phase_portrait_points(pairs)
    factor_table = pd.read_csv(root / "outputs" / "postfit_sqlite" / "legacy_condition_parameter_ratios.csv")
    analysis_summary = {
        "step_name": "Legacy biological MFA/MFA+BA perturbation layer",
        "config": asdict(cfg),
        "n_baseline_categories": int(categories["category_id"].nunique()) if "category_id" in categories else int(len(categories)),
        "n_selected_baselines": int(len(selected)),
        "n_one_dimensional_rows": int(len(one_d)),
        "n_pair_sweep_rows": int(len(pairs)),
        "selection_policy": "top_ranked_per_sigmoid_temporal_category",
    }
    if cfg.write_outputs:
        factor_table.to_csv(out_dir / "biological_perturbation_factor_table.csv", index=False)
        one_d.to_csv(out_dir / "biological_parameter_perturbation_sweeps.csv", index=False)
        direction_summary.to_csv(out_dir / "biological_parameter_direction_summary.csv", index=False)
        pairs.to_csv(out_dir / "biological_parameter_pair_sweeps.csv", index=False)
        sigmoid_summary.to_csv(out_dir / "sigmoid_state_change_summary.csv", index=False)
        direction_summary.to_csv(out_dir / "experimental_direction_match_summary.csv", index=False)
        phase_points.to_csv(out_dir / "phase_portrait_points.csv", index=False)
        (out_dir / "analysis_summary.json").write_text(
            json.dumps(analysis_summary, indent=2), encoding="utf-8"
        )
    return {
        "biological_perturbation_factor_table": factor_table,
        "biological_parameter_perturbation_sweeps": one_d,
        "biological_parameter_direction_summary": direction_summary,
        "biological_parameter_pair_sweeps": pairs,
        "sigmoid_state_change_summary": sigmoid_summary,
        "experimental_direction_match_summary": direction_summary,
        "phase_portrait_points": phase_points,
        "analysis_summary": analysis_summary,
    }
