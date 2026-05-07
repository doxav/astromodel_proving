"""Mechanistic summaries and representative selection.

Source -> target merge rationale
--------------------------------
The existing notebooks can fit/simulate Vm traces, but reviewer-facing claims need
mechanistic variables: Kir/gap/leak fluxes, effective parameter combinations, and
proxy validity.  Astrosim provided useful patterns for selecting representatives
with similar output and different structure, but its original criteria were
resting-voltage/Z-parameter based.  This module replaces those criteria with
accepted-fit and astrocyte-flux criteria.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _window_mask(t_ms: Sequence[float], window_s: Optional[tuple[float, float]]) -> np.ndarray:
    t_s = np.asarray(t_ms, dtype=float) / 1000.0
    if window_s is None:
        return np.ones_like(t_s, dtype=bool)
    lo, hi = window_s
    return (t_s >= float(lo)) & (t_s <= float(hi))


def _get_array(sim: Mapping[str, Any], key: str) -> np.ndarray:
    """Fetch an array from common simulation-output locations.

    Besides explicit ``currents`` and ``derived`` dictionaries, the canonical
    simulation output stores state variables in ``states`` with columns
    ``[Va, DK_a_t, K_a_s, Kg]``.  This fallback lets mechanism functions use
    biologically meaningful names without duplicating state columns.
    """

    if key in sim:
        return np.asarray(sim[key], dtype=float)
    for container in ("currents", "derived", "features", "effective_params"):
        if container in sim and isinstance(sim[container], Mapping) and key in sim[container]:
            return np.asarray(sim[container][key], dtype=float)
    if "hidden" in sim and isinstance(sim["hidden"], Mapping):
        for container in ("currents", "derived"):
            h = sim["hidden"].get(container, {})
            if isinstance(h, Mapping) and key in h:
                return np.asarray(h[key], dtype=float)

    state_cols = {"Va": 0, "Vm": 0, "DK_a_t": 1, "K_a_t": 1, "K_a_s": 2, "Kg": 3}
    if key in state_cols and "states" in sim:
        states = np.asarray(sim["states"], dtype=float)
        if states.ndim == 2 and states.shape[1] > state_cols[key]:
            return states[:, state_cols[key]]

    raise KeyError(f"Could not find {key!r} in simulation output")


def compute_gap_kir_ratio(
    I_kgap: Sequence[float],
    I_Kir: Sequence[float],
    mode: str = "integral",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute a gap/Kir dominance ratio.

    Parameters
    ----------
    I_kgap, I_Kir:
        Time courses of gap-junction and Kir currents.
    mode:
        ``integral`` uses sum absolute current, ``mean_abs`` uses mean absolute
        current, and ``peak_abs`` uses maximum absolute current.
    eps:
        Stabilizer for divisions.

    Returns
    -------
    dict
        ``ratio`` plus a coarse classification: ``gap_dominant``,
        ``kir_dominant``, or ``mixed``.
    """

    gap = np.asarray(I_kgap, dtype=float)
    kir = np.asarray(I_Kir, dtype=float)
    if gap.shape != kir.shape:
        raise ValueError(f"I_kgap and I_Kir must have same shape, got {gap.shape} vs {kir.shape}")

    if mode == "integral":
        num = float(np.nansum(np.abs(gap)))
        den = float(np.nansum(np.abs(kir)))
    elif mode == "mean_abs":
        num = float(np.nanmean(np.abs(gap)))
        den = float(np.nanmean(np.abs(kir)))
    elif mode == "peak_abs":
        num = float(np.nanmax(np.abs(gap)))
        den = float(np.nanmax(np.abs(kir)))
    else:
        raise ValueError("mode must be 'integral', 'mean_abs', or 'peak_abs'")

    ratio = num / (den + eps)
    if ratio >= 2.0:
        cls = "gap_dominant"
    elif ratio <= 0.5:
        cls = "kir_dominant"
    else:
        cls = "mixed"
    return {"ratio": float(ratio), "mode": mode, "classification": cls, "gap_metric": num, "kir_metric": den}


def compute_flux_summary(
    sim: Mapping[str, Any],
    stim_window_s: Optional[tuple[float, float]] = None,
    baseline_window_s: Optional[tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Summarize hidden astrocyte currents and K-homeostasis variables.

    Input
    -----
    ``sim`` should be the output of
    ``astro_model.simulate_with_hidden_outputs(..., return_hidden=True)`` or an
    equivalent dict containing ``t_ms``, ``currents`` and ``derived``.

    Output
    ------
    A flat dictionary suitable for joining onto an accepted-fit table.  These
    columns are the recommended mechanism-space inputs for reviewer-facing
    representative selection and flux-partition plots.
    """

    t_ms = np.asarray(sim["t_ms"], dtype=float)
    mask = _window_mask(t_ms, stim_window_s)
    if not np.any(mask):
        raise ValueError("stim_window_s selects no time points")

    I_Kir = _get_array(sim, "I_Kir")[mask]
    I_kgap = _get_array(sim, "I_kgap")[mask]
    I_leak = _get_array(sim, "I_leak")[mask]
    I_k_a = _get_array(sim, "I_k_a")[mask]
    K_o = _get_array(sim, "K_o")
    K_o_window = K_o[mask]

    dt_s = np.nanmedian(np.diff(t_ms[mask])) / 1000.0 if np.sum(mask) > 1 else 1.0
    kir_int = float(np.nansum(np.abs(I_Kir)) * dt_s)
    gap_int = float(np.nansum(np.abs(I_kgap)) * dt_s)
    leak_int = float(np.nansum(np.abs(I_leak)) * dt_s)
    ika_int = float(np.nansum(np.abs(I_k_a)) * dt_s)
    total = kir_int + gap_int + leak_int + 1e-12

    ratio = compute_gap_kir_ratio(I_kgap, I_Kir, mode="integral")
    if gap_int / total > 0.60:
        dominant = "Gap"
    elif kir_int / total > 0.60:
        dominant = "Kir"
    elif leak_int / total > 0.60:
        dominant = "Leak"
    else:
        dominant = "Mixed"

    baseline_mask = _window_mask(t_ms, baseline_window_s)
    if baseline_window_s is None or not np.any(baseline_mask):
        K_o_baseline = float(K_o[0])
    else:
        K_o_baseline = float(np.nanmedian(K_o[baseline_mask]))

    summary: Dict[str, Any] = {
        "I_Kir_integral": kir_int,
        "I_kgap_integral": gap_int,
        "I_leak_integral": leak_int,
        "I_k_a_integral": ika_int,
        "I_Kir_peak_abs": float(np.nanmax(np.abs(I_Kir))),
        "I_kgap_peak_abs": float(np.nanmax(np.abs(I_kgap))),
        "gap_to_kir_integral_ratio": float(ratio["ratio"]),
        "gap_kir_classification": ratio["classification"],
        "gap_fraction": float(gap_int / total),
        "kir_fraction": float(kir_int / total),
        "leak_fraction": float(leak_int / total),
        "K_o_baseline": K_o_baseline,
        "K_o_peak": float(np.nanmax(K_o_window)),
        "K_o_final": float(K_o[-1]),
        "K_o_delta_peak": float(np.nanmax(K_o_window) - K_o_baseline),
        "K_o_recovery_error": float(abs(K_o[-1] - K_o_baseline)),
        "dominant_mechanism": dominant,
    }

    for key, value in sim.get("effective_params", {}).items():
        summary[key] = value
    return summary


def compute_proxy_validity(
    sim: Mapping[str, Any],
    proxy: str = "DK_a_t",
    target: str = "K_o",
    window_s: Optional[tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Quantify whether an intracellular K variable is a valid proxy for ECS K.

    This directly addresses the criticism that intracellular K accumulation may
    not reliably stand in for extracellular K dynamics.
    """

    t_ms = np.asarray(sim["t_ms"], dtype=float)
    mask = _window_mask(t_ms, window_s)
    x = _get_array(sim, proxy)[mask]
    y = _get_array(sim, target)[mask]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {
            "proxy": proxy,
            "target": target,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "rmse_after_scaling": np.nan,
            "lag_s_at_max_corr": np.nan,
            "validity_class": "failed",
        }

    pearson = float(np.corrcoef(x, y)[0, 1])
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    spearman = float(np.corrcoef(xr, yr)[0, 1])

    # Linear scaling x -> y for shape agreement.
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    rmse = float(np.sqrt(np.nanmean((y - yhat) ** 2)))

    # Coarse lag estimate by cross-correlation on z-scored arrays.
    xz = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    yz = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
    corr = np.correlate(xz, yz, mode="full")
    lag_idx = int(np.argmax(corr) - (len(xz) - 1))
    dt_s = np.nanmedian(np.diff(t_ms[mask])) / 1000.0 if np.sum(mask) > 1 else np.nan
    lag_s = float(lag_idx * dt_s) if np.isfinite(dt_s) else np.nan

    abs_r = abs(pearson)
    if abs_r >= 0.85:
        cls = "strong"
    elif abs_r >= 0.60:
        cls = "moderate"
    elif abs_r >= 0.35:
        cls = "weak"
    else:
        cls = "failed"

    return {
        "proxy": proxy,
        "target": target,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "rmse_after_scaling": rmse,
        "lag_s_at_max_corr": lag_s,
        "validity_class": cls,
    }


def _standardize_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Column {c!r} missing from fits dataframe")
        s = pd.to_numeric(df[c], errors="coerce")
        scale = s.quantile(0.75) - s.quantile(0.25)
        if not np.isfinite(scale) or scale == 0:
            scale = s.std()
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out[c] = (s - s.median()) / scale
    return out


def _filter_functionally_similar(group: pd.DataFrame, function_cols: Sequence[str], function_tol: Optional[Mapping[str, float]]) -> pd.DataFrame:
    if not function_cols:
        return group.copy()
    if function_tol is None:
        # Default: keep all accepted fits.  The caller can provide explicit
        # tolerances or pre-filtered accepted ensembles.
        return group.copy()
    mask = np.ones(len(group), dtype=bool)
    for c in function_cols:
        center = pd.to_numeric(group[c], errors="coerce").median()
        tol = float(function_tol.get(c, np.inf))
        mask &= (pd.to_numeric(group[c], errors="coerce") - center).abs().to_numpy() <= tol
    candidates = group.loc[mask].copy()
    return candidates if len(candidates) else group.copy()


def select_mechanistically_diverse_representatives(
    fits: pd.DataFrame,
    function_cols: Sequence[str],
    mechanism_cols: Sequence[str],
    n: int = 4,
    function_tol: Optional[Mapping[str, float]] = None,
    group_cols: Optional[Sequence[str]] = None,
    strategy: str = "maximin",
) -> pd.DataFrame:
    """Select same-function but mechanistically diverse representatives.

    Source
    ------
    Adapted from astrosim ``select_degenerate_representatives``.  The astrosim
    function selected simulations with similar final/resting voltage and diverse
    ``Z_th/Z_s`` values.  This target version should be run on accepted fits and
    uses reviewer-relevant function and mechanism columns.

    Parameters
    ----------
    fits:
        Accepted-fit table with one row per trial/simulation.
    function_cols:
        Columns defining preserved function, for example ``Vm_rmse``,
        ``peak_depolarization_mV``, ``K_o_peak`` or ``decay_tau_s``.
    mechanism_cols:
        Columns defining mechanism diversity, for example ``P_gap_eff``,
        ``gamma_t_eff``, ``gap_to_kir_integral_ratio`` and ``proxy_pearson_r``.
    n:
        Number of representatives per group.
    function_tol:
        Optional tolerances around group medians for ``function_cols``.
    group_cols:
        Optional grouping columns such as ``condition`` and ``current_na``.
    strategy:
        ``maximin`` greedily maximizes distance to already selected reps.

    Returns
    -------
    pandas.DataFrame
        Selected rows with ``representative_rank`` and diagnostic distances.
    """

    if fits.empty:
        return fits.copy()
    if n <= 0:
        raise ValueError("n must be positive")
    if strategy != "maximin":
        raise ValueError("Only strategy='maximin' is initialized")

    groups: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        groups = fits.groupby(list(group_cols), dropna=False)
    else:
        groups = [("all", fits)]

    selected_groups = []
    for group_key, group in groups:
        candidates = _filter_functionally_similar(group, function_cols, function_tol)
        candidates = candidates.dropna(subset=list(mechanism_cols), how="any").copy()
        if candidates.empty:
            continue
        mech_z = _standardize_frame(candidates, mechanism_cols).to_numpy(dtype=float)

        # First representative: closest to functional median if function columns
        # exist, otherwise the first/best row in the provided order.
        if function_cols:
            func_z = _standardize_frame(candidates, function_cols).to_numpy(dtype=float)
            first_pos = int(np.argmin(np.linalg.norm(func_z, axis=1)))
        else:
            first_pos = 0

        chosen_positions = [first_pos]
        while len(chosen_positions) < min(n, len(candidates)):
            remaining = [i for i in range(len(candidates)) if i not in chosen_positions]
            d_to_selected = []
            for i in remaining:
                distances = [float(np.linalg.norm(mech_z[i] - mech_z[j])) for j in chosen_positions]
                d_to_selected.append(min(distances))
            chosen_positions.append(remaining[int(np.argmax(d_to_selected))])

        chosen = candidates.iloc[chosen_positions].copy()
        chosen["representative_rank"] = np.arange(1, len(chosen) + 1)
        chosen["representative_group"] = str(group_key)

        # Distance diagnostics.
        chosen_mech = _standardize_frame(candidates, mechanism_cols).iloc[chosen_positions].to_numpy(dtype=float)
        chosen["mechanism_distance_to_group_median"] = np.linalg.norm(chosen_mech, axis=1)
        if function_cols:
            chosen_func = _standardize_frame(candidates, function_cols).iloc[chosen_positions].to_numpy(dtype=float)
            chosen["function_distance_to_group_median"] = np.linalg.norm(chosen_func, axis=1)
        else:
            chosen["function_distance_to_group_median"] = np.nan
        selected_groups.append(chosen)

    if not selected_groups:
        return pd.DataFrame(columns=list(fits.columns) + ["representative_rank", "representative_group"])
    return pd.concat(selected_groups, ignore_index=True)
