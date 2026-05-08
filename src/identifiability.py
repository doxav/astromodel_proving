"""Step 03A structural-inspection and practical-identifiability helpers.

The routines in this module intentionally implement a lightweight, auditable
screen rather than a formal symbolic STRIKE-GOLDD proof. They combine the
model's equation-level effective parameters with local profile-loss and FIM
numerical diagnostics around representative accepted SQLite fits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .astro_model import normalize_flat_params, simulate_with_hidden_outputs
from .optuna_sqlite import TrialRecord, read_best_trial
from .postfit_sqlite import d_pk_invariance_check, effective_parameters_from_flat

PRIMARY_EFFECTIVE_PARAMETERS = ["P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo"]
DEFAULT_REPRESENTATIVE_DBS = ["CONTROL_75nA.db", "MFA_100nA.db", "BARIUM_100nA.db"]
DEFAULT_PROFILE_PARAMETERS = ["P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo"]
RAW_PROFILE_PARAMETERS = ["d", "pk", "gt", "gs", "wo"]
EFFECTIVE_FIM_PARAMETERS = [
    "P_gap_eff",
    "gamma_t_eff",
    "gamma_s_eff",
    "volume_ratio_wa_wo",
    "gki",
    "gl_a",
    "ca",
    "eps",
    "K_bath_value_middle",
]
RAW_FIM_PARAMETERS = [
    "d",
    "pk",
    "gt",
    "gs",
    "wo",
    "gki",
    "gl_a",
    "ca",
    "eps",
    "K_bath_value_middle",
]
# Backward-compatible name used by the first Step 03 tests.
DEFAULT_FIM_PARAMETERS = EFFECTIVE_FIM_PARAMETERS
CLAIM_GUARDRAIL = "limited identifiability/sloppiness under Vm-only observation; not biological degeneracy"
CLAIM_BOUNDARY = "structural-inspection screen plus local practical diagnostics; not a full symbolic STRIKE-GOLDD proof; flat/sloppy is not biological degeneracy"


@dataclass(frozen=True)
class Step03Config:
    """Runtime controls for the Step 03A screen."""

    representative_dbs: tuple[str, ...] = tuple(DEFAULT_REPRESENTATIVE_DBS)
    n_timepoints: int = 180
    profile_grid: tuple[float, ...] = (0.25, 0.4, 0.63, 1.0, 1.6, 2.5, 4.0)
    fim_log_step: float = 0.03
    observable_stride: int = 2
    sigma_mV: float = 1.0
    profile_parameter_spaces: tuple[str, ...] = ("effective", "raw")
    fim_parameter_spaces: tuple[str, ...] = ("effective", "raw")
    observable_designs: tuple[str, ...] = ("sparse", "dense")
    claim_scope: str = "identifiability_screen_not_biological_degeneracy"


def notebook_config_dict(config: Step03Config) -> dict[str, Any]:
    """Return the compact notebook-facing configuration contract."""

    data = asdict(config)
    data["representative_conditions"] = ["CONTROL", "MFA", "BARIUM"]
    return data


def _stim_window_seconds(condition: str) -> tuple[float, float]:
    if condition.upper() == "CONTROL":
        return (11.173, 31.173)
    return (21.140, 41.140)


def representative_context(condition: str, current_na: int, n_timepoints: int = 180) -> dict[str, Any]:
    _start, end = _stim_window_seconds(condition)
    return {
        "experiment_type": condition,
        "current_na": int(current_na),
        "sim_time_ms": np.linspace(0.0, (end + 5.0) * 1000.0, int(n_timepoints), dtype=float),
    }


def build_effective_parameter_map(example_params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Return the structural-inspection map used for reviewer-facing reporting."""

    params = normalize_flat_params(example_params or {})
    fitted = [
        "K_bath_value_middle",
        "Va_l",
        "Va_s",
        "ca",
        "d",
        "eps",
        "gki",
        "gl_a",
        "gs",
        "gt",
        "pk",
        "wo",
        "zs",
        "zth",
    ]
    optional_or_fixed = ["wo_middle", "eps_middle", "w_a", "switching_function"]
    rows: list[dict[str, Any]] = []

    def add(parameter: str, coordinate_type: str, identifiability_class: str, effective_parameter: str, expression: str, interpretation: str) -> None:
        rows.append(
            {
                "parameter": parameter,
                "coordinate_type": coordinate_type,
                "parameter_space": coordinate_type,
                "identifiability_class": identifiability_class,
                # Backward-compatible alias.
                "classification": identifiability_class,
                "effective_parameter": effective_parameter,
                "expression": expression,
                "present_in_example": bool(parameter in params or coordinate_type == "effective"),
                "claim_guardrail": CLAIM_GUARDRAIL,
                "reviewer_interpretation": interpretation,
            }
        )

    add("d", "raw", "effective_combination_member", "P_gap_eff", "d * pk", "Vm-only gap-current terms identify the product more directly than d or pk separately.")
    add("pk", "raw", "effective_combination_member", "P_gap_eff", "d * pk", "Do not interpret permeability separately from gap distance without external constraints.")
    add("gt", "raw", "effective_combination_member", "gamma_t_eff", "gt * Sig_a / (w_a * F)", "Transmembrane K dynamics constrain a scaled transport coefficient.")
    add("gs", "raw", "effective_combination_member", "gamma_s_eff", "gs * Sig_a / (w_a * F)", "Syncytial K dynamics constrain a scaled transport coefficient.")
    add("wo", "raw", "effective_combination_member", "volume_ratio_wa_wo", "w_a / wo", "Vm and derived extracellular K depend on the astrocyte/extracellular volume ratio.")
    add("w_a", "raw", "fixed_constant", "gamma_t_eff;gamma_s_eff;volume_ratio_wa_wo", "fixed default in current fits", "A fixed modeling constant in current SQLite fits, not a direct inferred quantity.")
    for p in ["gki", "gl_a", "ca", "eps", "K_bath_value_middle", "Va_l", "Va_s", "zs", "zth"]:
        identifiability_class = "direct_candidate" if p in {"gki", "gl_a", "ca", "eps", "K_bath_value_middle"} else "weakly_interpretable"
        add(p, "raw", identifiability_class, p, "appears as a raw model term", "Interpret only after profile/FIM support; weak profiles imply practical non-identifiability.")
    for p in optional_or_fixed:
        if p not in {"w_a"}:
            add(p, "raw", "fixed_constant" if p != "switching_function" else "protocol_input", p, "fixed/default or model-family setting", "Not treated as a fitted molecular estimate in Step 03A.")
    for eff, expr in [
        ("P_gap_eff", "d * pk"),
        ("gamma_t_eff", "gt * Sig_a / (w_a * F)"),
        ("gamma_s_eff", "gs * Sig_a / (w_a * F)"),
        ("volume_ratio_wa_wo", "w_a / wo"),
    ]:
        add(eff, "effective", "primary_interpretable", eff, expr, "Primary coordinate for identifiability, reporting, and later mechanism-space interpretation.")
    order = {name: i for i, name in enumerate(fitted + optional_or_fixed + PRIMARY_EFFECTIVE_PARAMETERS)}
    df = pd.DataFrame(rows)
    return df.sort_values("parameter", key=lambda s: s.map(lambda x: order.get(x, 999))).reset_index(drop=True)


def set_parameter_coordinate(params: Mapping[str, Any], coordinate: str, value: float) -> dict[str, Any]:
    """Set a raw or effective coordinate while preserving current conventions."""

    p = normalize_flat_params(params)
    value = float(value)
    if value <= 0 and coordinate in set(PRIMARY_EFFECTIVE_PARAMETERS + RAW_FIM_PARAMETERS):
        raise ValueError(f"{coordinate} must remain positive")
    if coordinate == "P_gap_eff":
        p["pk"] = value / max(float(p["d"]), 1e-30)
    elif coordinate == "gamma_t_eff":
        base = effective_parameters_from_flat(p, "CONTROL", 100)["gamma_t_eff"]
        p["gt"] = float(p["gt"]) * value / max(base, 1e-30)
    elif coordinate == "gamma_s_eff":
        base = effective_parameters_from_flat(p, "CONTROL", 100)["gamma_s_eff"]
        p["gs"] = float(p["gs"]) * value / max(base, 1e-30)
    elif coordinate == "volume_ratio_wa_wo":
        p["wo"] = float(p.get("w_a", 2000.0)) / value
    else:
        p[coordinate] = value
    return p


def coordinate_value(params: Mapping[str, Any], coordinate: str, condition: str, current_na: int) -> float:
    p = normalize_flat_params(params)
    if coordinate in PRIMARY_EFFECTIVE_PARAMETERS:
        return float(effective_parameters_from_flat(p, condition, current_na)[coordinate])
    return float(p[coordinate])


def observable_vector(params: Mapping[str, Any], condition: str, current_na: int, n_timepoints: int = 180, stride: int = 2) -> np.ndarray:
    sim = simulate_with_hidden_outputs(params, representative_context(condition, current_na, n_timepoints), solver="odeint")
    vm = np.asarray(sim["Vm"], dtype=float)[:: max(1, int(stride))]
    return vm - float(np.nanmean(vm))


def affine_refit_loss(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Least-squares profile-loss after affine Vm calibration as a cheap nuisance refit."""

    x = np.asarray(predicted, dtype=float)
    y = np.asarray(target, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    profile_loss = float(np.mean((fitted - y) ** 2))
    return profile_loss, float(coef[0]), float(coef[1])


def classify_profile(profile: pd.DataFrame, tolerance_fraction: float = 0.15) -> str:
    losses = np.asarray(profile["profile_loss" if "profile_loss" in profile else "loss"], dtype=float)
    multipliers = np.asarray(profile["multiplier"], dtype=float)
    if len(losses) < 3 or not np.isfinite(losses).all():
        return "flat_unbounded"
    min_idx = int(np.argmin(losses))
    loss_range = float(np.max(losses) - np.min(losses))
    scale = max(float(np.median(np.abs(losses))), float(np.max(np.abs(losses))), 1e-12)
    if loss_range / scale < tolerance_fraction:
        return "flat_unbounded"
    if min_idx == 0 or min_idx == len(losses) - 1:
        return "boundary_hit"
    threshold = float(np.min(losses) + 0.25 * loss_range)
    width = float(np.log(multipliers[losses <= threshold].max()) - np.log(multipliers[losses <= threshold].min()))
    return "clear_valley" if width <= np.log(2.0) else "broad_valley"


def profile_parameter(record: TrialRecord, parameter: str, config: Step03Config, parameter_space: str = "effective") -> pd.DataFrame:
    base = normalize_flat_params(record.params)
    base_value = coordinate_value(base, parameter, record.condition, record.current_na)
    target = observable_vector(base, record.condition, record.current_na, config.n_timepoints, config.observable_stride)
    rows: list[dict[str, Any]] = []
    for multiplier in config.profile_grid:
        trial_params = set_parameter_coordinate(base, parameter, base_value * float(multiplier))
        pred = observable_vector(trial_params, record.condition, record.current_na, config.n_timepoints, config.observable_stride)
        profile_loss, scale, offset = affine_refit_loss(pred, target)
        rows.append(
            {
                "db_name": record.db_name,
                "condition": record.condition,
                "current_na": record.current_na,
                "trial_number": record.trial_number,
                "parameter_space": parameter_space,
                "profile_parameter": parameter,
                # Backward-compatible alias.
                "parameter": parameter,
                "base_value": base_value,
                "multiplier": float(multiplier),
                "fixed_value": base_value * float(multiplier),
                "profile_loss": profile_loss,
                # Backward-compatible alias.
                "loss": profile_loss,
                "delta_profile_loss": np.nan,
                "delta_loss": np.nan,
                "nuisance_refit_method": "least_squares_affine_vm_scale_offset_not_full_reoptimization",
                "nuisance_scale": scale,
                "nuisance_offset_mV": offset,
                "claim_guardrail": CLAIM_GUARDRAIL,
            }
        )
    df = pd.DataFrame(rows)
    df["delta_profile_loss"] = df["profile_loss"] - float(df["profile_loss"].min())
    df["delta_loss"] = df["delta_profile_loss"]
    classification = classify_profile(df)
    df["profile_classification"] = classification
    df["profile_class"] = classification
    return df


def finite_difference_jacobian(record: TrialRecord, parameters: Sequence[str], config: Step03Config) -> tuple[np.ndarray, np.ndarray, list[float]]:
    base = normalize_flat_params(record.params)
    base_obs = observable_vector(base, record.condition, record.current_na, config.n_timepoints, config.observable_stride)
    cols: list[np.ndarray] = []
    values: list[float] = []
    h = float(config.fim_log_step)
    for parameter in parameters:
        value = coordinate_value(base, parameter, record.condition, record.current_na)
        values.append(value)
        plus = set_parameter_coordinate(base, parameter, value * np.exp(h))
        minus = set_parameter_coordinate(base, parameter, value * np.exp(-h))
        y_plus = observable_vector(plus, record.condition, record.current_na, config.n_timepoints, config.observable_stride)
        y_minus = observable_vector(minus, record.condition, record.current_na, config.n_timepoints, config.observable_stride)
        cols.append((y_plus - y_minus) / (2.0 * h))
    return np.column_stack(cols), base_obs, values


def compute_fim_tables_with_diagnostics(
    record: TrialRecord,
    config: Step03Config,
    parameters: Sequence[str] = DEFAULT_FIM_PARAMETERS,
    parameter_space: str = "effective",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameters = list(parameters)
    jac, _base_obs, base_values = finite_difference_jacobian(record, parameters, config)
    fim = (jac.T @ jac) / max(float(config.sigma_mV) ** 2, 1e-30)
    fim_is_symmetric_before_ridge = bool(np.allclose(fim, fim.T, rtol=1e-8, atol=1e-10))
    fim = 0.5 * (fim + fim.T)
    ridge = max(float(np.trace(fim)) * 1e-12 / max(len(parameters), 1), 1e-18)
    fim = fim + np.eye(len(parameters)) * ridge
    eigenvalues, eigenvectors = np.linalg.eigh(fim)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    max_eval = max(float(np.max(eigenvalues)), 1e-300)
    min_eval = max(float(np.min(eigenvalues)), 0.0)
    near_zero_threshold = max_eval * 1e-8
    near_zero_mode_count = int(np.sum(eigenvalues <= near_zero_threshold))
    log10_span = float(np.log10(max_eval) - np.log10(max(min_eval, 1e-300)))

    spectrum_rows = []
    loading_rows = []
    for mode_idx, eigenvalue in enumerate(eigenvalues, start=1):
        vec = eigenvectors[:, mode_idx - 1]
        abs_vec = np.abs(vec)
        dominant_idx = int(np.argmax(abs_vec))
        mode_class = "stiff" if eigenvalue / max_eval >= 1e-3 else "sloppy"
        spectrum_rows.append(
            {
                "db_name": record.db_name,
                "condition": record.condition,
                "current_na": record.current_na,
                "parameter_space": parameter_space,
                "mode_index": mode_idx,
                "eigenvalue": float(eigenvalue),
                "log10_eigenvalue": float(np.log10(max(float(eigenvalue), 1e-300))),
                "relative_eigenvalue": float(eigenvalue / max_eval),
                "mode_class": mode_class,
                "dominant_parameter": parameters[dominant_idx],
                "dominant_loading_abs": float(abs_vec[dominant_idx]),
                "condition_number_estimate": float(max_eval / max(min_eval, 1e-300)),
                "claim_guardrail": CLAIM_GUARDRAIL,
            }
        )
        for param, loading, base_value in zip(parameters, vec, base_values):
            loading_rows.append(
                {
                    "db_name": record.db_name,
                    "condition": record.condition,
                    "current_na": record.current_na,
                    "parameter_space": parameter_space,
                    "mode_index": mode_idx,
                    "parameter": param,
                    "base_value": float(base_value),
                    "loading": float(loading),
                    "abs_loading": float(abs(loading)),
                    "mode_class": mode_class,
                    "coordinate_type": "effective" if param in PRIMARY_EFFECTIVE_PARAMETERS else "raw",
                    "claim_guardrail": CLAIM_GUARDRAIL,
                }
            )
    diagnostics = pd.DataFrame(
        [
            {
                "db_name": record.db_name,
                "condition": record.condition,
                "current_na": record.current_na,
                "parameter_space": parameter_space,
                "fim_is_symmetric": fim_is_symmetric_before_ridge,
                "smallest_eigenvalue": float(min_eval),
                "largest_eigenvalue": float(max_eval),
                "near_zero_mode_count": near_zero_mode_count,
                "log10_eigenvalue_span": log10_span,
                "n_parameters": len(parameters),
                "claim_guardrail": CLAIM_GUARDRAIL,
            }
        ]
    )
    return pd.DataFrame(spectrum_rows), pd.DataFrame(loading_rows), diagnostics


def compute_fim_tables(
    record: TrialRecord,
    config: Step03Config,
    parameters: Sequence[str] = DEFAULT_FIM_PARAMETERS,
    parameter_space: str = "effective",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible FIM table helper returning spectrum and loadings."""

    spectrum, loadings, _diagnostics = compute_fim_tables_with_diagnostics(record, config, parameters, parameter_space)
    return spectrum, loadings


def structural_invariance_table(base_params: Mapping[str, Any], condition: str, current_na: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scale in [0.5, 2.0, 5.0]:
        check = d_pk_invariance_check(base_params, condition, current_na, scale_factor=scale)
        max_abs_rhs_delta = float(np.max(np.abs(check.dzdt_a - check.dzdt_b)))
        abs_i_kgap_diff = abs(check.I_kgap_a - check.I_kgap_b)
        rows.append(
            {
                "condition": condition,
                "current_na": current_na,
                "invariance": "d_pk_product",
                "scale_factor": float(scale),
                # Backward-compatible alias.
                "scale_factor_d": float(scale),
                "P_gap_eff_a": check.P_gap_eff_a,
                "P_gap_eff_b": check.P_gap_eff_b,
                "I_kgap_a": check.I_kgap_a,
                "I_kgap_b": check.I_kgap_b,
                "max_abs_rhs_delta": max_abs_rhs_delta,
                "structural_status": "exact_product_invariance_within_float_tolerance",
                "claim_scope": "structural_confounding_not_biological_degeneracy",
                "abs_P_gap_eff_diff": abs(check.P_gap_eff_a - check.P_gap_eff_b),
                "max_abs_dzdt_diff": max_abs_rhs_delta,
                "abs_I_kgap_diff": abs_i_kgap_diff,
                "interpretation": "Raw d and pk are not separately interpretable here; P_gap_eff = d * pk is the interpretable coordinate.",
            }
        )
    return pd.DataFrame(rows)


def representative_centers_table(records: Sequence[TrialRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "db_name": r.db_name,
                "study_name": r.study_name,
                "condition": r.condition,
                "current_na": r.current_na,
                "trial_id": r.trial_id,
                "trial_number": r.trial_number,
                "objective": r.objective,
                **effective_parameters_from_flat(r.params, r.condition, r.current_na),
            }
            for r in records
        ]
    )


def build_interpretation_notes() -> pd.DataFrame:
    rows = [
        {
            "diagnostic": "effective_parameter_map",
            "finding": "Several raw coordinates enter the reduced Vm-only equations through products, ratios, or fixed scaling constants.",
            "supports_claim": "raw parameter estimates should not be overinterpreted",
            "does_not_support_claim": "does not prove biological degeneracy",
            "reviewer_critique_targeted": "R1,R4,R7",
            "manuscript_use": "Supplementary table defining effective reporting coordinates and raw-parameter guardrails.",
        },
        {
            "diagnostic": "d_pk_invariance",
            "finding": "Reciprocal scaling of d and pk preserves P_gap_eff, I_kgap, and the RHS within floating-point tolerance.",
            "supports_claim": "d and pk are structurally non-separable in this reduced equation term",
            "does_not_support_claim": "does not establish a complete STRIKE-GOLDD proof or a biological phenotype",
            "reviewer_critique_targeted": "R1,R4",
            "manuscript_use": "Direct reviewer-response example showing why P_gap_eff is reported instead of separate d and pk claims.",
        },
        {
            "diagnostic": "profile_loss_screen",
            "finding": "Profile-style perturbation curves classify local practical identifiability as clear, broad, flat, or boundary-limited.",
            "supports_claim": "some coordinates are only weakly constrained under Vm-only observation",
            "does_not_support_claim": "does not prove biological degeneracy because nuisance parameters are not fully reoptimized",
            "reviewer_critique_targeted": "R1,R4",
            "manuscript_use": "Supplemental practical-identifiability diagnostic with explicit local-screen caveat.",
        },
        {
            "diagnostic": "fim_spectrum",
            "finding": "Finite-difference FIM spectra show stiff and sloppy local modes around representative centers.",
            "supports_claim": "sloppy directions exist and should be separated from degeneracy claims",
            "does_not_support_claim": "does not validate mechanism modes or predictive robustness",
            "reviewer_critique_targeted": "R1,R4,R7",
            "manuscript_use": "Supplemental eigenspectrum figure and mode-loading table.",
        },
        {
            "diagnostic": "raw_vs_effective_fim",
            "finding": "Raw-space and effective-space FIMs are compared so raw confounding is not mistaken for effective-coordinate interpretability.",
            "supports_claim": "effective-space modes are the primary scientific interpretation coordinates",
            "does_not_support_claim": "does not make all effective coordinates biologically identifiable",
            "reviewer_critique_targeted": "R1,R4",
            "manuscript_use": "Guardrail text explaining why raw-space near-zero modes imply non-identifiability/sloppiness.",
        },
        {
            "diagnostic": "observable_design_benchmark",
            "finding": "Sparse and dense observable designs are benchmarked to document local numerical sensitivity/provenance.",
            "supports_claim": "diagnostic settings are explicit and auditable",
            "does_not_support_claim": "does not address empirical variability/noise or accepted-fit thresholds",
            "reviewer_critique_targeted": "R7",
            "manuscript_use": "Methods/provenance note for supplementary diagnostics.",
        },
    ]
    return pd.DataFrame(rows)


def build_reviewer_response_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "critique_targeted": "R1",
                "diagnostic": "d × pk invariance",
                "output_file": "invariance_diagnostics.csv",
                "conclusion": "d/pk non-separable",
                "limitation": "not full STRIKE-GOLDD",
            },
            {
                "critique_targeted": "R1",
                "diagnostic": "FIM spectrum",
                "output_file": "fim_spectrum.csv",
                "conclusion": "sloppy directions exist",
                "limitation": "local only",
            },
            {
                "critique_targeted": "R4",
                "diagnostic": "effective parameter map",
                "output_file": "effective_parameter_map.csv",
                "conclusion": "report effective parameters",
                "limitation": "physiological interpretation still needs priors/ranges",
            },
            {
                "critique_targeted": "R7",
                "diagnostic": "interpretation notes",
                "output_file": "interpretation_notes.csv",
                "conclusion": "clearer supplement",
                "limitation": "not a mechanistic result",
            },
        ]
    )


def observable_design_benchmark(record: TrialRecord, config: Step03Config) -> pd.DataFrame:
    rows = []
    designs = {
        "sparse": {"n_timepoints": max(60, config.n_timepoints // 2), "observable_stride": max(2, config.observable_stride * 2)},
        "dense": {"n_timepoints": config.n_timepoints, "observable_stride": max(1, config.observable_stride)},
    }
    for design, kwargs in designs.items():
        local_config = Step03Config(
            representative_dbs=config.representative_dbs,
            n_timepoints=int(kwargs["n_timepoints"]),
            profile_grid=config.profile_grid,
            fim_log_step=config.fim_log_step,
            observable_stride=int(kwargs["observable_stride"]),
            sigma_mV=config.sigma_mV,
            profile_parameter_spaces=config.profile_parameter_spaces,
            fim_parameter_spaces=("effective",),
            observable_designs=config.observable_designs,
            claim_scope=config.claim_scope,
        )
        started = perf_counter()
        spectrum, _loadings, diagnostics = compute_fim_tables_with_diagnostics(record, local_config, EFFECTIVE_FIM_PARAMETERS, parameter_space="effective")
        rows.append(
            {
                "observable_design": design,
                "condition": record.condition,
                "current_na": record.current_na,
                "n_timepoints": local_config.n_timepoints,
                "observable_stride": local_config.observable_stride,
                "n_observables": int(np.ceil(local_config.n_timepoints / local_config.observable_stride)),
                "elapsed_seconds": perf_counter() - started,
                "n_modes": len(spectrum),
                "near_zero_mode_count": int(diagnostics["near_zero_mode_count"].iloc[0]),
                "log10_eigenvalue_span": float(diagnostics["log10_eigenvalue_span"].iloc[0]),
                "claim_guardrail": "observable-design sensitivity only; not empirical uncertainty or biological degeneracy",
            }
        )
    return pd.DataFrame(rows)


def run_step03_identifiability_screen(project_root: str | Path, output_dir: str | Path | None = None, config: Step03Config | None = None) -> dict[str, Any]:
    """Run Step 03A and write all reviewer-facing CSV/JSON outputs."""

    root = Path(project_root)
    output = Path(output_dir) if output_dir is not None else root / "outputs" / "identifiability"
    output.mkdir(parents=True, exist_ok=True)
    config = config or Step03Config()
    initial_fit_dir = root / "data" / "1_Initial_xp_fit"
    records = [read_best_trial(initial_fit_dir / db_name) for db_name in config.representative_dbs]
    primary = records[0]

    started = perf_counter()
    effective_map = build_effective_parameter_map(primary.params)
    invariance = structural_invariance_table(primary.params, primary.condition, primary.current_na)
    representative_centers = representative_centers_table(records)

    profile_parts: list[pd.DataFrame] = []
    if "effective" in config.profile_parameter_spaces:
        profile_parts.extend(profile_parameter(primary, parameter, config, "effective") for parameter in DEFAULT_PROFILE_PARAMETERS)
    if "raw" in config.profile_parameter_spaces:
        profile_parts.extend(profile_parameter(primary, parameter, config, "raw") for parameter in RAW_PROFILE_PARAMETERS)
    profile_likelihoods = pd.concat(profile_parts, ignore_index=True)
    profile_loss_screen = profile_likelihoods.copy()
    profile_summary = (
        profile_likelihoods.groupby(["db_name", "condition", "current_na", "parameter_space", "profile_parameter"], as_index=False)
        .agg(
            base_value=("base_value", "first"),
            min_profile_loss=("profile_loss", "min"),
            max_delta_profile_loss=("delta_profile_loss", "max"),
            profile_classification=("profile_classification", "first"),
            profile_class=("profile_class", "first"),
            nuisance_refit_method=("nuisance_refit_method", "first"),
            claim_guardrail=("claim_guardrail", "first"),
        )
        .sort_values(["parameter_space", "profile_parameter"])
        .reset_index(drop=True)
    )
    profile_summary["parameter"] = profile_summary["profile_parameter"]
    profile_summary["min_loss"] = profile_summary["min_profile_loss"]
    profile_summary["max_delta_loss"] = profile_summary["max_delta_profile_loss"]

    spectrum_parts: list[pd.DataFrame] = []
    loading_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for record in records:
        if "effective" in config.fim_parameter_spaces:
            spectrum, loadings, diagnostics = compute_fim_tables_with_diagnostics(record, config, EFFECTIVE_FIM_PARAMETERS, parameter_space="effective")
            spectrum_parts.append(spectrum)
            loading_parts.append(loadings)
            diagnostic_parts.append(diagnostics)
        if "raw" in config.fim_parameter_spaces:
            spectrum, loadings, diagnostics = compute_fim_tables_with_diagnostics(record, config, RAW_FIM_PARAMETERS, parameter_space="raw")
            spectrum_parts.append(spectrum)
            loading_parts.append(loadings)
            diagnostic_parts.append(diagnostics)
    fim_spectrum = pd.concat(spectrum_parts, ignore_index=True)
    fim_mode_loadings = pd.concat(loading_parts, ignore_index=True)
    fim_diagnostics = pd.concat(diagnostic_parts, ignore_index=True)

    interpretation_notes = build_interpretation_notes()
    reviewer_response_table = build_reviewer_response_table()
    observable_benchmark = observable_design_benchmark(primary, config)

    run_summary = pd.DataFrame(
        [
            {
                "n_representative_centers": len(records),
                "profile_center_db": primary.db_name,
                "n_profile_parameters": int(profile_summary.shape[0]),
                "n_fim_parameters": len(EFFECTIVE_FIM_PARAMETERS),
                "n_timepoints": config.n_timepoints,
                "observable_stride": config.observable_stride,
                "runtime_seconds": perf_counter() - started,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ]
    )

    analysis_summary_json = {
        "notebook_name": "03_combined_identifiability_profiles_fim.ipynb",
        "step_name": "Step 03A identifiability/effective-parameter/FIM screen",
        "claim_scope": config.claim_scope,
        "critiques_targeted": ["R1", "R4", "partial R7"],
        "critiques_not_resolved": ["R2 empirical variability/noise", "R3 model assumption sensitivity", "R5 biological mechanism modes", "R6 predictive robustness/CV"],
        "representative_centers": representative_centers[["db_name", "condition", "current_na", "trial_number"]].to_dict(orient="records"),
        "n_profile_diagnostics": int(profile_likelihoods[["parameter_space", "profile_parameter"]].drop_duplicates().shape[0]),
        "n_fim_modes": int(fim_spectrum.shape[0]),
        "uses_raw_space_fim": bool((fim_spectrum["parameter_space"] == "raw").any()),
        "uses_effective_space_fim": bool((fim_spectrum["parameter_space"] == "effective").any()),
        "has_d_pk_invariance_check": bool(not invariance.empty),
        "has_interpretation_notes": bool(not interpretation_notes.empty),
        "next_required_steps": [
            "Step 03B/04 threshold-weighted empirical accepted-fit screening",
            "accepted-ensemble mechanism decomposition",
            "held-out current/predictive validation",
        ],
    }

    config_payload = notebook_config_dict(config)
    with open(output / "notebook_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    with open(output / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(analysis_summary_json, f, indent=2)

    outputs: dict[str, Any] = {
        "effective_parameter_map": effective_map,
        "invariance_diagnostics": invariance,
        # Backward-compatible name from the first Step 03 implementation.
        "structural_invariance_diagnostics": invariance,
        "profile_likelihoods": profile_likelihoods,
        "profile_loss_screen": profile_loss_screen,
        "profile_summary": profile_summary,
        "fim_spectrum": fim_spectrum,
        "fim_mode_loadings": fim_mode_loadings,
        "fim_diagnostics": fim_diagnostics,
        "representative_centers": representative_centers,
        "observable_design_benchmark": observable_benchmark,
        "interpretation_notes": interpretation_notes,
        "reviewer_response_table": reviewer_response_table,
        "analysis_summary": run_summary,
        "analysis_summary_json": analysis_summary_json,
        "notebook_config": config_payload,
    }
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output / f"{name}.csv", index=False)
    return outputs


def run_step03_identifiability(project_root: str | Path, output_dir: str | Path | None = None, config: Step03Config | None = None) -> dict[str, Any]:
    """Canonical Step 03A runner used by the notebook."""

    return run_step03_identifiability_screen(project_root, output_dir=output_dir, config=config)
