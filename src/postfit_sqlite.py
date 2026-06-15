"""Step 01 SQLite post-fit pipeline and hidden-mechanism summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .astro_model import build_paramdict, compute_rhs_and_currents, normalize_flat_params, simulate_with_hidden_outputs
from .contracts import canonical_condition
from .parameter_space import effective_from_flat
from .protocols import representative_context as _shared_representative_context, stim_window_seconds
from .mechanisms import compute_flux_summary, compute_proxy_validity
from .optuna_sqlite import TrialRecord, read_best_trial, read_top_trials

DEFAULT_REPRESENTATIVE_DBS = ["CONTROL_75nA.db", "MFA_100nA.db", "BARIUM_100nA.db"]
LEGACY_TOP_N_REQUESTED = 300
FILTER_BASELINE_FOLD_GRID: dict[str, tuple[float, ...]] = {
    "gki": (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    "P_gap_eff": (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    "gamma_s_eff": (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    "zth": (0.5, 0.75, 1.0, 1.25, 1.5),
    "zs": (0.5, 0.75, 1.0, 1.25, 1.5),
}


@dataclass(frozen=True)
class InvarianceCheck:
    state: np.ndarray
    t_ms: float
    params_a: dict[str, Any]
    params_b: dict[str, Any]
    dzdt_a: np.ndarray
    dzdt_b: np.ndarray
    I_kgap_a: float
    I_kgap_b: float
    P_gap_eff_a: float
    P_gap_eff_b: float


def effective_parameters_from_flat(flat_params: Mapping[str, Any], experiment_type: str, current_na: int) -> dict[str, float]:
    """Backward-compatible wrapper around :func:`parameter_space.effective_from_flat`."""

    return effective_from_flat(flat_params, condition=experiment_type, current_na=current_na).as_dict()


def d_pk_invariance_check(
    base_params: Mapping[str, Any],
    experiment_type: str = "CONTROL",
    current_na: int = 100,
    scale_factor: float = 2.0,
    state: Sequence[float] | None = None,
    t_ms: float = 12_000.0,
) -> InvarianceCheck:
    params_a = normalize_flat_params(base_params)
    params_b = normalize_flat_params(dict(base_params))
    params_b["d"] = float(params_a["d"]) * float(scale_factor)
    params_b["pk"] = float(params_a["pk"]) / float(scale_factor)
    state_arr = np.asarray(state if state is not None else [-84.0, 0.15, 0.02, 0.5], dtype=float)
    out_a = compute_rhs_and_currents(state_arr, t_ms, build_paramdict(experiment_type, current_na, params_a), return_currents=True)
    out_b = compute_rhs_and_currents(state_arr, t_ms, build_paramdict(experiment_type, current_na, params_b), return_currents=True)
    return InvarianceCheck(
        state=state_arr,
        t_ms=float(t_ms),
        params_a=params_a,
        params_b=params_b,
        dzdt_a=np.asarray(out_a["dzdt"], dtype=float),
        dzdt_b=np.asarray(out_b["dzdt"], dtype=float),
        I_kgap_a=float(out_a["currents"]["I_kgap"]),
        I_kgap_b=float(out_b["currents"]["I_kgap"]),
        P_gap_eff_a=float(out_a["effective_params"]["P_gap_eff"]),
        P_gap_eff_b=float(out_b["effective_params"]["P_gap_eff"]),
    )


def _stim_window_seconds(condition: str) -> tuple[float, float]:
    """Backward-compatible wrapper around :func:`protocols.stim_window_seconds`."""

    return stim_window_seconds(condition)


def _representative_context(condition: str, current_na: int, n_timepoints: int = 600) -> dict[str, Any]:
    """Backward-compatible wrapper around :func:`protocols.representative_context`."""

    return _shared_representative_context(condition, current_na, n_timepoints=n_timepoints)


def summarize_representative_trial(record: TrialRecord, n_timepoints: int = 600) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _representative_context(record.condition, record.current_na, n_timepoints=n_timepoints)
    sim = simulate_with_hidden_outputs(record.params, context, solver="odeint")
    flux = compute_flux_summary(sim, stim_window_s=_stim_window_seconds(record.condition))
    proxy = compute_proxy_validity(sim, window_s=_stim_window_seconds(record.condition))
    effective = effective_parameters_from_flat(record.params, record.condition, record.current_na)
    row = {
        "db_name": record.db_name,
        "study_name": record.study_name,
        "condition": record.condition,
        "current_na": record.current_na,
        "trial_id": record.trial_id,
        "trial_number": record.trial_number,
        "objective": record.objective,
        **effective,
        **flux,
        **{f"proxy_{k}": v for k, v in proxy.items()},
    }
    return row, sim


def top_trials_with_effective_parameters(initial_fit_dir: str | Path, top_n: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for db_path in sorted(Path(initial_fit_dir).glob("*.db")):
        for record in read_top_trials(db_path, top_n=top_n):
            rows.append(
                {
                    "db_name": record.db_name,
                    "study_name": record.study_name,
                    "condition": record.condition,
                    "current_na": record.current_na,
                    "trial_id": record.trial_id,
                    "trial_number": record.trial_number,
                    "objective": record.objective,
                    **record.params,
                    **effective_parameters_from_flat(record.params, record.condition, record.current_na),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["condition", "current_na", "objective", "trial_number"]).reset_index(drop=True)
    return df


def build_legacy_configuration_library(
    initial_fit_dir: str | Path,
    *,
    top_n: int = LEGACY_TOP_N_REQUESTED,
    provenance_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the first-pass legacy top-N Optuna configuration library."""

    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    initial_dir = Path(initial_fit_dir)
    provenance = pd.DataFrame()
    if provenance_path is not None and Path(provenance_path).exists():
        provenance = pd.read_csv(provenance_path)
    provenance_cols = [
        "db_name",
        "status",
        "chosen_status",
        "trace_dataset_kind",
        "chosen_trace_source",
    ]
    if not provenance.empty:
        provenance = provenance[[c for c in provenance_cols if c in provenance.columns]].drop_duplicates("db_name")
        provenance = provenance.rename(
            columns={
                "status": "provenance_status",
                "chosen_status": "chosen_provenance_status",
            }
        )

    rows: list[dict[str, Any]] = []
    for db_path in sorted(initial_dir.glob("*.db")):
        records = read_top_trials(db_path, top_n=int(top_n))
        available = len(records)
        for rank, record in enumerate(records, start=1):
            condition = canonical_condition(record.condition)
            rows.append(
                {
                    "source_scope": "legacy_single_current_optuna",
                    "legacy_configuration_status": "legacy_top300_optuna_trial",
                    "legacy_acceptance_rule": "not_thresholded_top_n_first_pass",
                    "legacy_selection_rule": "top_n_by_objective",
                    "legacy_top_n_requested": int(top_n),
                    "legacy_top_n_available": int(available),
                    "rank_in_db": int(rank),
                    "db_name": record.db_name,
                    "study_name": record.study_name,
                    "condition": condition,
                    "legacy_protocol_condition": record.condition,
                    "current_na": int(record.current_na),
                    "trial_id": int(record.trial_id),
                    "trial_number": int(record.trial_number),
                    "objective": float(record.objective),
                    **record.params,
                    **effective_parameters_from_flat(
                        record.params, record.condition, record.current_na
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if not provenance.empty:
        out = out.merge(provenance, on="db_name", how="left", validate="many_to_one")
    if "provenance_status" not in out.columns:
        out["provenance_status"] = "not_available"
    else:
        out["provenance_status"] = out["provenance_status"].fillna("not_available")
    return out.sort_values(["condition", "current_na", "rank_in_db"]).reset_index(drop=True)


def legacy_configuration_status_by_db(legacy_library: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-pass legacy top-N status by DB."""

    if legacy_library.empty:
        return pd.DataFrame()
    grouped = legacy_library.groupby(
        ["db_name", "condition", "legacy_protocol_condition", "current_na"],
        as_index=False,
        dropna=False,
    ).agg(
        n_configurations=("trial_number", "nunique"),
        legacy_top_n_requested=("legacy_top_n_requested", "first"),
        legacy_top_n_available=("legacy_top_n_available", "first"),
        best_objective=("objective", "min"),
        worst_included_objective=("objective", "max"),
        provenance_status=("provenance_status", "first"),
    )
    grouped["legacy_configuration_status"] = "legacy_top300_optuna_trial"
    grouped["legacy_acceptance_rule"] = "not_thresholded_top_n_first_pass"
    return grouped.sort_values(["condition", "current_na"]).reset_index(drop=True)


def legacy_condition_parameter_ratios(legacy_library: pd.DataFrame) -> pd.DataFrame:
    """Build candidate condition-ratio factors and fold-grid perturbation rows."""

    parameters = ["P_gap_eff", "gamma_s_eff", "zth", "zs", "gki"]
    condition_pairs = [
        ("CONTROL", "MFA", "MFA_like_from_control_legacy"),
        ("MFA", "MFA_BA", "MFA_BA_from_MFA_legacy"),
        ("CONTROL", "MFA_BA", "MFA_BA_stacked_on_control_legacy"),
    ]
    rows: list[dict[str, Any]] = []
    if not legacy_library.empty:
        med = (
            legacy_library.groupby(["condition", "current_na"], as_index=False)[parameters]
            .median(numeric_only=True)
        )
        for cond1, cond2, context in condition_pairs:
            for current_na in sorted(med["current_na"].dropna().unique()):
                left = med[(med["condition"] == cond1) & (med["current_na"] == current_na)]
                right = med[(med["condition"] == cond2) & (med["current_na"] == current_na)]
                for parameter in parameters:
                    factor = np.nan
                    status = "missing_condition_pair"
                    if not left.empty and not right.empty:
                        denom = float(left[parameter].iloc[0])
                        numer = float(right[parameter].iloc[0])
                        if np.isfinite(denom) and abs(denom) > 1e-30 and np.isfinite(numer):
                            factor = float(numer / denom)
                            status = "candidate_ratio_available"
                    rows.append(
                        {
                            "condition_pair": f"{cond1}_to_{cond2}",
                            "perturbation_context": context,
                            "current_na": int(current_na),
                            "parameter": parameter,
                            "factor": factor,
                            "factor_source": "legacy_top_trial_ratio",
                            "factor_status": status,
                        }
                    )
    for context in [
        "MFA_like_from_control_legacy",
        "MFA_like_from_mfa_legacy",
        "MFA_BA_from_MFA_legacy",
        "MFA_BA_stacked_on_control_legacy",
    ]:
        for parameter, folds in FILTER_BASELINE_FOLD_GRID.items():
            for fold in folds:
                rows.append(
                    {
                        "condition_pair": "fold_grid",
                        "perturbation_context": context,
                        "current_na": -1,
                        "parameter": parameter,
                        "factor": float(fold),
                        "factor_source": "filter_baseline_fold_grid",
                        "factor_status": "first_pass_grid_factor",
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["factor_source", "perturbation_context", "current_na", "parameter", "factor"]
    ).reset_index(drop=True)


def representative_mechanism_summary(initial_fit_dir: str | Path, representative_dbs: Sequence[str] | None = None, n_timepoints: int = 600) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    initial_fit_dir = Path(initial_fit_dir)
    if representative_dbs is None:
        representative_dbs = DEFAULT_REPRESENTATIVE_DBS
    rows: list[dict[str, Any]] = []
    simulations: dict[str, dict[str, Any]] = {}
    for db_name in representative_dbs:
        record = read_best_trial(initial_fit_dir / db_name)
        row, sim = summarize_representative_trial(record, n_timepoints=n_timepoints)
        rows.append(row)
        simulations[db_name] = sim
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["condition", "current_na"]).reset_index(drop=True)
    return df, simulations


def effective_parameter_summary(initial_fit_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for db_path in sorted(Path(initial_fit_dir).glob("*.db")):
        record = read_best_trial(db_path)
        rows.append(
            {
                "db_name": record.db_name,
                "study_name": record.study_name,
                "condition": record.condition,
                "current_na": record.current_na,
                "trial_id": record.trial_id,
                "trial_number": record.trial_number,
                "objective": record.objective,
                **effective_parameters_from_flat(record.params, record.condition, record.current_na),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["condition", "current_na"]).reset_index(drop=True)


def _project_paths(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "project_root": root,
        "initial_fit_dir": root / "data" / "1_Initial_xp_fit",
        "outputs_dir": root / "outputs" / "postfit_sqlite",
        "cache_dir": root / "data" / "legacy_summary_cache" / "postfit_sqlite",
    }


def _raw_assets_available(initial_fit_dir: Path) -> bool:
    return len(list(initial_fit_dir.glob("*.db"))) == 18


def _load_cached_table(cache_dir: Path, filename: str) -> pd.DataFrame:
    path = cache_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_cached_postfit_tables(project_root: str | Path) -> dict[str, pd.DataFrame]:
    cache_dir = _project_paths(project_root)["cache_dir"]
    return {
        "top_trials_all_dbs": _load_cached_table(cache_dir, "top_trials_all_dbs.csv"),
        "effective_parameter_summary": _load_cached_table(cache_dir, "effective_parameter_summary.csv"),
        "representative_mechanism_summary": _load_cached_table(cache_dir, "representative_mechanism_summary.csv"),
    }


def run_step01_postfit_sqlite(
    project_root: str | Path,
    top_n: int = 5,
    legacy_top_n: int = LEGACY_TOP_N_REQUESTED,
    representative_dbs: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    paths = _project_paths(project_root)
    outputs_dir = Path(output_dir) if output_dir is not None else paths["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if _raw_assets_available(paths["initial_fit_dir"]):
        top_trials_df = top_trials_with_effective_parameters(paths["initial_fit_dir"], top_n=top_n)
        legacy_library = build_legacy_configuration_library(
            paths["initial_fit_dir"],
            top_n=legacy_top_n,
            provenance_path=paths["project_root"]
            / "outputs"
            / "provenance"
            / "control_trace_verification.csv",
        )
        effective_df = effective_parameter_summary(paths["initial_fit_dir"])
        representative_df, simulations = representative_mechanism_summary(paths["initial_fit_dir"], representative_dbs=representative_dbs)
    else:
        cached = load_cached_postfit_tables(project_root)
        top_trials_df = cached["top_trials_all_dbs"].copy()
        legacy_library = top_trials_df.copy()
        legacy_library["source_scope"] = "legacy_single_current_optuna"
        legacy_library["legacy_configuration_status"] = "legacy_top300_optuna_trial"
        legacy_library["legacy_acceptance_rule"] = "not_thresholded_top_n_first_pass"
        legacy_library["legacy_selection_rule"] = "top_n_by_objective"
        legacy_library["legacy_top_n_requested"] = int(legacy_top_n)
        legacy_library["legacy_top_n_available"] = legacy_library.groupby("db_name")["trial_number"].transform("nunique")
        legacy_library["rank_in_db"] = legacy_library.groupby("db_name").cumcount() + 1
        legacy_library["provenance_status"] = "cached_without_raw_provenance"
        effective_df = cached["effective_parameter_summary"].copy()
        representative_df = cached["representative_mechanism_summary"].copy()
        simulations = {}
    status_by_db = legacy_configuration_status_by_db(legacy_library)
    condition_ratios = legacy_condition_parameter_ratios(legacy_library)

    top_trials_df.to_csv(outputs_dir / "top_trials_all_dbs.csv", index=False)
    effective_df.to_csv(outputs_dir / "effective_parameter_summary.csv", index=False)
    representative_df.to_csv(outputs_dir / "representative_mechanism_summary.csv", index=False)
    legacy_library.to_csv(outputs_dir / "legacy_configuration_library.csv", index=False)
    status_by_db.to_csv(outputs_dir / "legacy_configuration_status_by_db.csv", index=False)
    condition_ratios.to_csv(outputs_dir / "legacy_condition_parameter_ratios.csv", index=False)

    return {
        "top_trials_all_dbs": top_trials_df,
        "legacy_configuration_library": legacy_library,
        "legacy_configuration_status_by_db": status_by_db,
        "legacy_condition_parameter_ratios": condition_ratios,
        "effective_parameter_summary": effective_df,
        "representative_mechanism_summary": representative_df,
        "representative_simulations": simulations,
    }
