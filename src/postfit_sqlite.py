"""Step 01 SQLite post-fit pipeline and hidden-mechanism summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .astro_model import build_paramdict, compute_rhs_and_currents, normalize_flat_params, simulate_with_hidden_outputs
from .mechanisms import compute_flux_summary, compute_proxy_validity
from .optuna_sqlite import TrialRecord, read_best_trial, read_top_trials

DEFAULT_REPRESENTATIVE_DBS = ["CONTROL_75nA.db", "MFA_100nA.db", "BARIUM_100nA.db"]


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
    paramdict = build_paramdict(experiment_type, current_na, flat_params)
    astro = paramdict["Astrocyte"]
    external = paramdict["external"]
    w_a = float(astro["w_a"])
    sig_a = float(astro["Sig_a"])
    F = float(astro["F"])
    return {
        "P_gap_eff": float(astro["d_gap"] * astro["P_k"]),
        "gamma_t_eff": float(astro["gama_t"] * sig_a / (w_a * F)),
        "gamma_s_eff": float(astro["gama_s"] * sig_a / (w_a * F)),
        "volume_ratio_wa_wo": float(w_a / float(external["w_o"])),
    }


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
    if condition.upper() == "CONTROL":
        return (11.173, 31.173)
    return (21.140, 41.140)


def _representative_context(condition: str, current_na: int, n_timepoints: int = 600) -> dict[str, Any]:
    start, end = _stim_window_seconds(condition)
    t_final_ms = (end + 5.0) * 1000.0
    return {
        "experiment_type": condition,
        "current_na": current_na,
        "sim_time_ms": np.linspace(0.0, t_final_ms, int(n_timepoints), dtype=float),
    }


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
        df = df.sort_values(["condition", "current_na", "objective", "trial_number"], ascending=[True, True, True, True]).reset_index(drop=True)
    return df


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


def run_step01_postfit_sqlite(
    project_root: str | Path,
    top_n: int = 5,
    representative_dbs: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    initial_fit_dir = project_root / "data" / "1_Initial_xp_fit"
    outputs_dir = Path(output_dir) if output_dir is not None else project_root / "outputs" / "postfit_sqlite"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    top_trials_df = top_trials_with_effective_parameters(initial_fit_dir, top_n=top_n)
    effective_df = effective_parameter_summary(initial_fit_dir)
    representative_df, simulations = representative_mechanism_summary(initial_fit_dir, representative_dbs=representative_dbs)

    top_trials_df.to_csv(outputs_dir / "top_trials_all_dbs.csv", index=False)
    effective_df.to_csv(outputs_dir / "effective_parameter_summary.csv", index=False)
    representative_df.to_csv(outputs_dir / "representative_mechanism_summary.csv", index=False)

    return {
        "top_trials_all_dbs": top_trials_df,
        "effective_parameter_summary": effective_df,
        "representative_mechanism_summary": representative_df,
        "representative_simulations": simulations,
    }
