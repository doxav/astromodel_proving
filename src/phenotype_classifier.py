"""Mechanistic phenotype helpers for Step 05 accepted ensembles.

The helpers in this module port the useful measure vocabulary from
``analysis/unified_astrocyte_K_buffering_characterization_EXECUTED_SMOKE.ipynb``
onto the repository's canonical simulator outputs.  They do not reuse the
legacy notebook's Optuna DB loader or duplicate its ODE implementation.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

EPS = 1e-12
WINDOWS: tuple[str, ...] = ("M0", "M_rise", "M_decay", "M_tot")
DEFAULT_DOMINANCE_MARGIN = 1.20

MEASURE_REGISTRY_ROWS: tuple[dict[str, str], ...] = (
    {
        "measure_family": "state reconstruction",
        "measure": "Vm, K_o, DK_a, DK_t, K_s, currents",
        "status": "primary",
        "rationale": "Hidden-state reconstruction is retained for mechanism interpretation and simulation sanity checks.",
    },
    {
        "measure_family": "signed flux budget",
        "measure": "local load, spatial export, bath source/sink",
        "status": "primary",
        "rationale": "Signed local/spatial budgets distinguish local uptake from long-range redistribution.",
    },
    {
        "measure_family": "mechanistic mode",
        "measure": "D_F and D_I_elec with finite dominance margin",
        "status": "primary",
        "rationale": "Continuous local/spatial axes are primary; labels are derived summaries.",
    },
    {
        "measure_family": "recruitment",
        "measure": "dKs_activation_score",
        "status": "primary",
        "rationale": "Current-weighted recruitment is more mechanistic than raw mean gate value.",
    },
    {
        "measure_family": "effective gap mapping",
        "measure": "P_gap_eff = d * pk",
        "status": "primary",
        "rationale": "Raw d and pk remain structurally confounded; the product is the reviewer-facing coordinate.",
    },
    {
        "measure_family": "available surface",
        "measure": "alpha2 = gamma_s_eff",
        "status": "primary",
        "rationale": "Interpreted as available spatial transfer surface-to-volume capacity in the reduced model.",
    },
    {
        "measure_family": "recruited surface",
        "measure": "alpha2 * dKs_activation_score",
        "status": "primary",
        "rationale": "Combines available capacity with dynamic recruitment; not an anatomical cell-count estimate.",
    },
    {
        "measure_family": "isopotentiality proxy",
        "measure": "1 / (1 + r_model)",
        "status": "secondary",
        "rationale": "Reduced-model analogue because the ODE has no explicit neighboring-cell voltage state.",
    },
    {
        "measure_family": "pump proxy",
        "measure": "K_o recovery and undershoot summaries",
        "status": "secondary",
        "rationale": "Observable recovery descriptors, not literal Na/K ATPase flux.",
    },
    {
        "measure_family": "deprecated",
        "measure": "raw d as anatomical syncytium size",
        "status": "deprecated",
        "rationale": "Superseded by P_gap_eff, available surface, and recruited-surface proxies.",
    },
)


def measure_registry_table() -> pd.DataFrame:
    """Return the Step 05 measure registry as a machine-readable table."""

    return pd.DataFrame(MEASURE_REGISTRY_ROWS)


def _integral(t_s: np.ndarray, values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    if clean.size < 2:
        return float(np.nansum(clean))
    return float(np.trapezoid(clean, np.asarray(t_s, dtype=float)))


def _pos(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _neg(values: np.ndarray) -> np.ndarray:
    return np.maximum(-np.asarray(values, dtype=float), 0.0)


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def state_10_90(value: float, low: float = 0.10, high: float = 0.90) -> str:
    """Classify a bounded activation value into low, middle, or high state."""

    x = _finite(value)
    if not np.isfinite(x):
        return "undefined"
    if x <= low:
        return "closed_low"
    if x >= high:
        return "open_high"
    return "partial_mid"


def classify_signed_flux_mode(
    d_f_log10: float,
    d_i_log10: float,
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> str:
    """Classify one window in the signed local/spatial flux plane."""

    d_f = _finite(d_f_log10)
    d_i = _finite(d_i_log10)
    if not np.isfinite(d_f) or not np.isfinite(d_i):
        return "UNDEFINED"
    theta = math.log10(float(dominance_margin)) if dominance_margin > 1.0 else 0.0
    flux_local = d_f > theta
    flux_spatial = d_f < -theta
    current_local = d_i > theta
    current_spatial = d_i < -theta
    if flux_local and current_local:
        return "STRICTLY_LOCAL"
    if flux_spatial and current_spatial:
        return "STRICTLY_SPATIAL"
    if flux_local and current_spatial:
        return "MIXED_LOCAL"
    if flux_spatial and current_local:
        return "MIXED_SPATIAL"
    return "BALANCED_OR_WEAK"


def window_bounds(
    stim_window_s: tuple[float, float],
    t_final_s: float,
    prestim_s: float = 5.0,
) -> dict[str, tuple[float, float]]:
    """Build the four standard Step 05 mechanism windows."""

    start_s, end_s = float(stim_window_s[0]), float(stim_window_s[1])
    if not (np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s):
        raise ValueError("stim_window_s must contain finite increasing times")
    final_s = max(float(t_final_s), end_s)
    return {
        "M0": (max(0.0, start_s - float(prestim_s)), start_s),
        "M_rise": (start_s, end_s),
        "M_decay": (end_s, final_s),
        "M_tot": (0.0, final_s),
    }


def _first_time_fraction(
    t_s: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    fraction: float,
) -> float:
    if int(mask.sum()) < 2:
        return float("nan")
    y = np.asarray(values, dtype=float)
    baseline = float(np.nanmedian(y[~mask])) if int((~mask).sum()) else 0.0
    amplitude = np.abs(y - baseline)
    peak = float(np.nanmax(amplitude[mask]))
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    indices = np.where(mask & (amplitude >= float(fraction) * peak))[0]
    return float(t_s[indices[0]]) if len(indices) else float("nan")


def _safe_gradient(values: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    if len(values) < 2 or len(t_s) < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(np.asarray(values, dtype=float), np.asarray(t_s, dtype=float))


def _activation_from_hidden(
    dk_a: np.ndarray,
    th_s: np.ndarray,
    i_kgap: np.ndarray,
    gamma_s_eff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = -np.asarray(th_s, dtype=float) * float(gamma_s_eff) * np.asarray(i_kgap, dtype=float)
    if_open = -np.asarray(dk_a, dtype=float) * float(gamma_s_eff) * np.asarray(i_kgap, dtype=float)
    gate = np.divide(
        np.abs(th_s),
        np.maximum(np.abs(dk_a), EPS),
        out=np.zeros_like(np.asarray(th_s, dtype=float)),
        where=np.isfinite(dk_a),
    )
    return actual, if_open, np.clip(gate, 0.0, 1.0)


def compute_windowed_buffering_scores(
    sim: Mapping[str, Any],
    stim_window_s: tuple[float, float],
    metadata: Mapping[str, Any],
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> pd.DataFrame:
    """Compute windowed phenotype scores from one hidden-output simulation."""

    t_ms = np.asarray(sim["t_ms"], dtype=float)
    t_s = t_ms / 1000.0
    if t_s.size < 3:
        raise ValueError("At least three simulation time points are required")

    states = np.asarray(sim["states"], dtype=float)
    currents = sim["currents"]
    derived = sim["derived"]
    effective = sim.get("effective_params", {})
    paramdict = sim.get("params", {})
    astro = paramdict.get("Astrocyte", {}) if isinstance(paramdict, Mapping) else {}

    va = states[:, 0]
    dk_t = states[:, 1]
    k_s = states[:, 2]
    kg = states[:, 3]
    dk_a = np.asarray(derived["DK_a"], dtype=float)
    k_o = np.asarray(derived["K_o"], dtype=float)
    e_k_a = np.asarray(derived["E_k_a"], dtype=float)
    i_kir = np.asarray(currents["I_Kir"], dtype=float)
    i_k = np.asarray(currents.get("I_k_a", np.zeros_like(i_kir)), dtype=float)
    i_gap = np.asarray(currents["I_kgap"], dtype=float)
    i_leak = np.asarray(currents["I_leak"], dtype=float)
    th_s = np.asarray(currents.get("Th_s", np.zeros_like(i_kir)), dtype=float)

    gamma_t_eff = _finite(effective.get("gamma_t_eff", metadata.get("gamma_t_eff")))
    gamma_s_eff = _finite(effective.get("gamma_s_eff", metadata.get("gamma_s_eff")))
    p_gap_eff = _finite(effective.get("P_gap_eff", metadata.get("P_gap_eff")))
    va_s = _finite(astro.get("Va_s", metadata.get("Va_s", -90.0)), -90.0)

    d_kt = -float(gamma_t_eff) * (i_kir + i_k)
    d_ks, d_ks_if_open, gate = _activation_from_hidden(dk_a, th_s, i_gap, gamma_s_eff)
    d_kg = _safe_gradient(kg, t_s)
    d_ko = _safe_gradient(k_o, t_s)

    bounds = window_bounds(stim_window_s, float(t_s[-1]))
    rows: list[dict[str, Any]] = []
    for window, (start_s, end_s) in bounds.items():
        mask = (t_s >= start_s) & (t_s <= end_s)
        if int(mask.sum()) < 3:
            continue
        tw = t_s[mask]
        local_uptake = _integral(tw, _pos(d_kt[mask]))
        local_release = _integral(tw, _neg(d_kt[mask]))
        spatial_export = _integral(tw, _neg(d_ks[mask]))
        spatial_import = _integral(tw, _pos(d_ks[mask]))
        bath_source = _integral(tw, _pos(d_kg[mask]))
        bath_sink = _integral(tw, _neg(d_kg[mask]))
        dks_actual_abs = _integral(tw, np.abs(d_ks[mask]))
        dks_open_abs = _integral(tw, np.abs(d_ks_if_open[mask]))
        dks_activation = float(np.clip(dks_actual_abs / (dks_open_abs + EPS), 0.0, 1.0))

        i_kir_abs = _integral(tw, np.abs(i_kir[mask]))
        i_gap_abs = _integral(tw, np.abs(i_gap[mask]))
        i_leak_abs = _integral(tw, np.abs(i_leak[mask]))
        i_local_abs = _integral(tw, np.abs(i_kir[mask]) + np.abs(i_k[mask]))
        i_total_abs = i_local_abs + i_gap_abs + i_leak_abs + EPS
        d_f = math.log10((local_uptake + EPS) / (spatial_export + EPS))
        d_i = math.log10((i_local_abs + EPS) / (i_gap_abs + EPS))
        gate_window = gate[mask]
        k_o_window = k_o[mask]
        va_window = va[mask]
        eka_window = e_k_a[mask]
        r_inst = np.abs((va_window - va_s) / (eka_window - va_s + EPS))
        r_median = float(np.nanmedian(r_inst)) if np.isfinite(r_inst).any() else float("nan")
        alpha2 = gamma_s_eff

        rows.append(
            {
                **dict(metadata),
                "window": window,
                "window_start_s": float(start_s),
                "window_end_s": float(end_s),
                "local_uptake_integral": local_uptake,
                "local_release_integral": local_release,
                "spatial_export_integral": spatial_export,
                "spatial_import_integral": spatial_import,
                "bath_source_integral": bath_source,
                "bath_sink_integral": bath_sink,
                "dKs_actual_abs_integral": dks_actual_abs,
                "dKs_available_if_open_abs_integral": dks_open_abs,
                "dKs_activation_score": dks_activation,
                "sigmoid_activation_mean": float(np.nanmean(gate_window)),
                "sigmoid_fraction_gt_0p5": float(np.nanmean(gate_window > 0.5)),
                "sigmoid_fraction_gt_0p9": float(np.nanmean(gate_window > 0.9)),
                "sigmoid_gate_start_value": float(gate_window[0]),
                "sigmoid_gate_end_value": float(gate_window[-1]),
                "sigmoid_gate_peak_value": float(np.nanmax(gate_window)),
                "sigmoid_gate_end_state_10_90": state_10_90(float(gate_window[-1])),
                "alpha2_available_surface_proxy": alpha2,
                "recruited_surface_alpha2_x_A_dKs": float(alpha2 * dks_activation),
                "P_gap_eff": p_gap_eff,
                "available_surface_conductance_capacity_alpha2_P_gap_eff": float(alpha2 * p_gap_eff),
                "r_ionic_contribution_median": r_median,
                "isopotentiality_score_from_r": float(1.0 / (1.0 + r_median)) if np.isfinite(r_median) else float("nan"),
                "low_range_local_fraction": float(local_uptake / (local_uptake + spatial_export + EPS)),
                "long_range_distribution_fraction": float(spatial_export / (local_uptake + spatial_export + EPS)),
                "D_F_log10_local_load_over_spatial_export": d_f,
                "D_I_elec_log10_local_current_over_gap_current": d_i,
                "mechanistic_mode_signed_flux": classify_signed_flux_mode(d_f, d_i, dominance_margin),
                "mode_confidence_log10": float(min(abs(d_f), abs(d_i))),
                "voltage_coupling_score": float(i_gap_abs / i_total_abs),
                "kir_current_score": float(i_kir_abs / i_total_abs),
                "leak_current_score": float(i_leak_abs / i_total_abs),
                "source_balance_index": float((bath_source - bath_sink) / (bath_source + bath_sink + EPS)),
                "spatial_directionality_index": float((spatial_export - spatial_import) / (spatial_export + spatial_import + EPS)),
                "local_directionality_index": float((local_uptake - local_release) / (local_uptake + local_release + EPS)),
                "K_o_min": float(np.nanmin(k_o_window)),
                "K_o_peak": float(np.nanmax(k_o_window)),
                "K_o_auc_above_baseline": _integral(tw, _pos(k_o_window - 4.8)),
                "K_o_recovery_proxy": _integral(tw, _pos(-d_ko[mask])),
                "K_s_abs_auc": _integral(tw, np.abs(k_s[mask])),
                "DK_t_abs_auc": _integral(tw, np.abs(dk_t[mask])),
                "t10_sigmoid_activation_s": _first_time_fraction(t_s, gate, mask, 0.1),
                "t90_sigmoid_activation_s": _first_time_fraction(t_s, gate, mask, 0.9),
                "t10_dKs_abs_s": _first_time_fraction(t_s, np.abs(d_ks), mask, 0.1),
                "t10_Igap_abs_s": _first_time_fraction(t_s, np.abs(i_gap), mask, 0.1),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["lag_dKs_vs_Igap_s"] = out["t10_dKs_abs_s"] - out["t10_Igap_abs_s"]
    out["buffering_phenotype"] = out.apply(classify_buffering_phenotype, axis=1)
    out["phenotype_claim_scope"] = "provisional_accepted_ensemble_tag_pending_step06_validation"
    return out


def classify_buffering_phenotype(row: Mapping[str, Any]) -> str:
    """Return a conservative biological-description tag for one window row."""

    activation = _finite(row.get("dKs_activation_score"))
    long_fraction = _finite(row.get("long_range_distribution_fraction"))
    kir_score = _finite(row.get("kir_current_score"))
    voltage_score = _finite(row.get("voltage_coupling_score"))
    surface = _finite(row.get("alpha2_available_surface_proxy"))
    recruited = _finite(row.get("recruited_surface_alpha2_x_A_dKs"))
    if activation < 0.10 and voltage_score >= 0.40 and surface > 0:
        return "available_surface_voltage_coupled_but_ionic_recruitment_low"
    if long_fraction >= 0.60 and activation >= 0.30:
        return "long_range_recruited_spatial_buffering"
    if kir_score >= 0.60 and activation < 0.30:
        return "kir_dominant_local_buffering"
    if recruited >= 0.50 * max(surface, EPS) and voltage_score >= 0.30:
        return "recruited_surface_gap_assisted_buffering"
    if activation < 0.10:
        return "low_recruitment_local_storage"
    return "mixed_local_spatial_buffering"


def summarize_phenotypes(windowed_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate windowed phenotype scores to candidate-level tags."""

    if windowed_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    key_cols = ["file_id", "region", "condition", "candidate_id"]
    for keys, group in windowed_scores.groupby(key_cols, dropna=False):
        m_tot = group[group["window"].eq("M_tot")]
        source = m_tot if not m_tot.empty else group
        phenotype_counts = source["buffering_phenotype"].value_counts(dropna=False)
        mode_counts = source["mechanistic_mode_signed_flux"].value_counts(dropna=False)
        rows.append(
            {
                **dict(zip(key_cols, keys)),
                "buffering_phenotype": str(phenotype_counts.index[0]),
                "phenotype_window_scope": "M_tot" if not m_tot.empty else "all_available_windows",
                "phenotype_specificity_score": float(phenotype_counts.iloc[0] / max(int(phenotype_counts.sum()), 1)),
                "dominant_signed_flux_mode": str(mode_counts.index[0]),
                "dKs_activation_score_mean": float(source["dKs_activation_score"].mean()),
                "long_range_distribution_fraction_mean": float(source["long_range_distribution_fraction"].mean()),
                "voltage_coupling_score_mean": float(source["voltage_coupling_score"].mean()),
                "kir_current_score_mean": float(source["kir_current_score"].mean()),
                "phenotype_claim_scope": "provisional_accepted_ensemble_tag_pending_step06_validation",
            }
        )
    return pd.DataFrame(rows)


def build_mode_vectors(windowed_scores: pd.DataFrame) -> pd.DataFrame:
    """Return one row per candidate/sweep with M0/M_rise/M_decay/M_tot modes."""

    if windowed_scores.empty:
        return pd.DataFrame()
    idx_cols = ["file_id", "region", "condition", "candidate_id", "sweep", "current_na"]
    mode = windowed_scores.pivot_table(
        index=idx_cols,
        columns="window",
        values="mechanistic_mode_signed_flux",
        aggfunc="first",
    ).reset_index()
    mode.columns = [
        f"mode_{column}" if column in WINDOWS else str(column) for column in mode.columns
    ]
    phenotype = windowed_scores[windowed_scores["window"].eq("M_tot")][
        idx_cols + ["buffering_phenotype", "phenotype_claim_scope"]
    ].drop_duplicates(idx_cols)
    return mode.merge(phenotype, on=idx_cols, how="left")
