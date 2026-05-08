"""Astrocyte model core and simulation helpers.

Source -> target merge rationale
--------------------------------
This module initializes the target functions proposed for the refactor:

1. ``compute_rhs_and_currents``
   Sources:
   - Existing code: ``Filtered_basline_sweep (1).ipynb::model`` and
     ``AFT_uncut_end_BARIUM_Opt_v2 (1).ipynb::model``.
   - Astrosim: ``features_and_running.py::compute_currents_and_derivatives``.

   What is kept from the existing code:
   - the 4-state astrocyte model ``[Va, DK_a_t, K_a_s, Kg]``;
   - scheduled ``K_bath`` protocols;
   - ``epsilon_middle`` / ``w_o_middle`` handling;
   - ``sigmoid`` / ``tanh`` / ``hill`` switching choices;
   - Optuna-compatible flat parameter names.

   What is kept from astrosim:
   - the cleaner pattern of computing derivatives and currents in one reusable
     model-core function.

2. ``simulate_odeint``
   Source: existing ``simulate_trial_aligned`` logic, but decoupled from the
   Optuna/ATF notebook so the solver can be reused for FIM, profile likelihood,
   accepted-fit ensembles, and perturbation tests.

3. ``simulate_rk4_numba``
   Source: astrosim ``full_model`` pattern.  The implementation below is a
   pure-Python RK4 backend by default.  It is intentionally written with simple
   arrays so it can later be replaced by a true Numba implementation after
   validating numerical equivalence with ``simulate_odeint``.

4. ``simulate_with_hidden_outputs``
   Source merge: existing trace-alignment workflow + astrosim
   ``run_model_with_currents`` pattern.  Unlike astrosim, the hidden outputs are
   astrocytic currents/derived states needed for reviewer-facing mechanistic
   analyses: ``I_Kir``, ``I_kgap``, leak, ``K_o``, and effective parameters.

The goal is to stop treating the ODE as a black-box Vm generator.  This module
exposes identifiable/effective parameter combinations and hidden fluxes so that
"degenerate" modes can be mapped to mechanisms rather than only to different
parameter vectors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is expected in the analysis environment, but keep import failure explicit.
    from scipy.integrate import odeint
except Exception as exc:  # pragma: no cover - only hit if scipy is unavailable.
    odeint = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Constants copied from the existing fitting workflow.
# -----------------------------------------------------------------------------

DEFAULT_Z0 = np.array([-89.0, 0.0, 0.0, 0.0], dtype=float)

CURRENT_DICT_COLUMNS: Dict[str, int] = {"50": 1, "75": 2, "100": 3, "125": 4, "150": 5, "175": 6}

CURRENT_DICT_K_BATH_VALUES: Dict[str, list[float]] = {
    "50": [4.8, 6.4, 4.8],
    "75": [4.8, 7.23, 4.8],
    "100": [4.8, 8.2, 4.8],
    "125": [4.8, 9.5, 4.8],
    "150": [4.8, 10.1, 4.8],
    "175": [4.8, 10.5, 4.8],
}

EXPERIMENT_K_BATH_TIME: Dict[str, list[float]] = {
    "MFA": [0.0, 21140.0, 41140.0],
    "BARIUM": [0.0, 21140.0, 41140.0],
    "CONTROL": [0.0, 11173.0, 31173.0],
    # Compatibility alias used in some discussions.
    "MFA_BA": [0.0, 21140.0, 41140.0],
}

# Default full simulation duration when no experimental time vector is provided.
DEFAULT_T_FINAL_MS = 50_000.0
DEFAULT_DT_MS = 0.1


@dataclass(frozen=True)
class ExperimentContext:
    """Minimal context for aligning a model simulation to an experimental trace.

    This is deliberately compatible with the context object previously created in
    ``Filtered_basline_sweep`` while also accepting plain dictionaries in the
    public API.
    """

    experiment_type: str
    current_na: int
    target_mean_mode: str = "centered"
    exp_times_ms_stable: Optional[np.ndarray] = None
    exp_trace_stable: Optional[np.ndarray] = None
    sim_time_ms: Optional[np.ndarray] = None
    stable_index_simulation: int = 0
    feature_onset_s: Optional[float] = None
    feature_offset_s: Optional[float] = None


def _as_context(context: ExperimentContext | Mapping[str, Any]) -> ExperimentContext:
    """Convert a dict-like context into :class:`ExperimentContext`."""

    if isinstance(context, ExperimentContext):
        return context
    return ExperimentContext(
        experiment_type=str(context.get("experiment_type", context.get("condition", "CONTROL"))),
        current_na=int(context.get("current_na", context.get("current", 100))),
        target_mean_mode=str(context.get("target_mean_mode", "centered")),
        exp_times_ms_stable=(None if context.get("exp_times_ms_stable") is None else np.asarray(context["exp_times_ms_stable"], dtype=float)),
        exp_trace_stable=(None if context.get("exp_trace_stable") is None else np.asarray(context["exp_trace_stable"], dtype=float)),
        sim_time_ms=(None if context.get("sim_time_ms") is None else np.asarray(context["sim_time_ms"], dtype=float)),
        stable_index_simulation=int(context.get("stable_index_simulation", 0)),
        feature_onset_s=context.get("feature_onset_s"),
        feature_offset_s=context.get("feature_offset_s"),
    )


def normalize_flat_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize flat parameter dictionaries from old notebooks/Optuna DBs.

    Parameters
    ----------
    params:
        Flat Optuna-style parameters, for example ``gki``, ``pk``, ``d``,
        ``gt``, ``gs``, ``zth``, ``zs``.

    Returns
    -------
    dict
        Copy with common spelling aliases normalized and default values filled.
    """

    p = dict(params)
    aliases = {
        "Zth": "zth",
        "Z_th": "zth",
        "Zs": "zs",
        "Z_s": "zs",
        "w_o_middle": "wo_middle",
    }
    for old, new in aliases.items():
        if old in p and new not in p:
            p[new] = p[old]
        if old in p and old != new:
            p.pop(old, None)

    p.setdefault("wo_middle", 1.0)
    p.setdefault("eps_middle", 1.0)
    p.setdefault("w_a", 2000.0)
    p.setdefault("switching_function", "sigmoid")
    return p


def build_paramdict(
    experiment_type: str,
    current_na: int,
    flat_params: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build the nested parameter dictionary used by the astrocyte ODE.

    Source
    ------
    Adapted from ``Filtered_basline_sweep (1).ipynb::build_paramdict``.

    Notes
    -----
    This keeps the old flat parameter names so the function can be used directly
    with Optuna DB rows.  It also includes effective combinations used later for
    interpretability, but those are computed in ``compute_rhs_and_currents``.
    """

    experiment_type = str(experiment_type).upper()
    if experiment_type == "MFA+BA":
        experiment_type = "MFA_BA"
    protocol_key = "BARIUM" if experiment_type == "MFA_BA" else experiment_type
    if protocol_key not in EXPERIMENT_K_BATH_TIME:
        raise ValueError(f"Unknown experiment_type={experiment_type!r}; expected one of {sorted(EXPERIMENT_K_BATH_TIME)}")

    p = normalize_flat_params(flat_params)
    current_key = str(int(current_na))
    if current_key not in CURRENT_DICT_K_BATH_VALUES:
        raise ValueError(f"Unknown current_na={current_na!r}; expected one of {sorted(CURRENT_DICT_K_BATH_VALUES)}")

    k_bath_values = np.asarray(CURRENT_DICT_K_BATH_VALUES[current_key], dtype=float).copy()
    k_bath_values[1] = float(p.get("K_bath_value_middle", k_bath_values[1]))

    astro: Dict[str, Any] = {
        "Cm_a": float(p.get("ca", 400.0)),
        "g_kir": float(p.get("gki", 1.0)),
        "g_k_a": float(p.get("g_k_a", 0.0)),
        "w_a": float(p.get("w_a", 2000.0)),
        "P_k": float(p.get("pk", 3e-5)),
        "A": 1.0,
        "gl_a": float(p.get("gl_a", 0.01)),
        "Va_l": float(p.get("Va_l", -70.0)),
        "Va_s": float(p.get("Va_s", -90.0)),
        "d_gap": float(p.get("d", 1.0)),
        "Va_0": -89.0,
        "Sig_a": 1600.0,
        "K_a0": 135.0,
        "F": 96485.0,
        "R": 8.314,
        "T": 298.0,
        "gama_t": float(p.get("gt", 6.0)),
        "gama_s": float(p.get("gs", 6.5)),
        "Z_s": float(p.get("zs", 0.05)) if p.get("zs", None) is not None else None,
        "Z_th": float(p.get("zth", 0.2)) if p.get("zth", None) is not None else None,
        "switching_function": str(p.get("switching_function", "sigmoid")),
    }
    if astro["switching_function"] == "hill":
        astro["hill_coefficient"] = float(p.get("hill_coefficient", 2.0))
        astro["K_d"] = float(p.get("K_d", 1.0))

    return {
        "Astrocyte": astro,
        "external": {
            "K_o0": 4.8,
            "w_o": float(p.get("wo", 1500.0)),
            "w_o_middle": float(p.get("wo_middle", 1.0)),
            "epsilon": float(p.get("eps", 1e-3)),
            "epsilon_middle": float(p.get("eps_middle", 1.0)),
            "K_bath": {
                "time": np.asarray(EXPERIMENT_K_BATH_TIME[protocol_key], dtype=float),
                "value": k_bath_values,
            },
        },
    }


def _is_paramdict(params: Mapping[str, Any]) -> bool:
    return "Astrocyte" in params and "external" in params


def _ensure_paramdict(params: Mapping[str, Any], protocol: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Accept either a nested paramdict or flat Optuna parameters."""

    if _is_paramdict(params):
        return deepcopy(params)  # type: ignore[arg-type]
    if protocol is None:
        raise ValueError("Flat parameters require a protocol with experiment_type and current_na")
    return build_paramdict(str(protocol.get("experiment_type", protocol.get("condition", "CONTROL"))), int(protocol.get("current_na", 100)), params)


def _lookup_kbath(external: Mapping[str, Any], t_ms: float) -> tuple[float, int, float, float]:
    """Return ``(K_bath, interval_index, epsilon_effective, w_o_effective)``."""

    k_bath = external["K_bath"]
    times = np.asarray(k_bath["time"], dtype=float)
    values = np.asarray(k_bath["value"], dtype=float)
    idx = int(np.searchsorted(times, t_ms, side="right") - 1)
    idx = min(max(idx, 0), len(values) - 1)

    epsilon = float(external["epsilon"])
    w_o = float(external["w_o"])
    if idx == 1:
        epsilon *= float(external.get("epsilon_middle", 1.0))
        w_o *= float(external.get("w_o_middle", 1.0))
    return float(values[idx]), idx, epsilon, w_o


def _safe_exp(x: float) -> float:
    """Exponent with clipping to avoid overflow in parameter sweeps."""

    return float(np.exp(np.clip(x, -700.0, 700.0)))


def _switching_gate(DK_a: float, DK_a_t: float, astro: Mapping[str, Any]) -> float:
    """Spatial-buffering activation term ``Th_s``.

    Source: existing ``Filtered.model``.  Astrosim only carried the sigmoid form;
    this keeps the alternative forms needed for model-comparison/sensitivity.
    """

    switching_function = str(astro.get("switching_function", "sigmoid")).lower()
    Z_th = astro.get("Z_th", 0.2)
    Z_s = astro.get("Z_s", 0.05)
    Z_th = 0.2 if Z_th is None else float(Z_th)
    Z_s = 0.05 if Z_s is None else float(Z_s)

    if switching_function == "sigmoid":
        return float(DK_a / (1.0 + _safe_exp((Z_th - DK_a_t) * Z_s)))
    if switching_function == "tanh":
        return float(DK_a * (0.5 * (1.0 + np.tanh((DK_a_t - Z_th) * Z_s))))
    if switching_function == "hill":
        n = float(astro.get("hill_coefficient", 2.0))
        K_d = max(float(astro.get("K_d", 1.0)), 1e-12)
        x = max(float(DK_a_t), 0.0)
        return float(DK_a * (x**n / (K_d**n + x**n)))
    if switching_function in {"soft_threshold", "soft-threshold", "linear_threshold"}:
        # Minimal additional option for assumption-sensitivity tests.
        activation = max(0.0, DK_a_t - Z_th) / (max(1e-12, abs(Z_s)) + max(0.0, DK_a_t - Z_th))
        return float(DK_a * activation)
    raise ValueError(f"Unknown switching_function={switching_function!r}")


def compute_rhs_and_currents(
    z: Sequence[float],
    t_ms: float,
    params: Mapping[str, Mapping[str, Any]],
    return_currents: bool = True,
) -> Dict[str, Any]:
    """Compute astrocyte ODE derivatives plus hidden currents and derived states.

    Parameters
    ----------
    z:
        State vector ``[Va, DK_a_t, K_a_s, Kg]``.
    t_ms:
        Time in milliseconds.  This matches the existing notebooks, where
        ``K_bath`` schedule times are also in ms.
    params:
        Nested parameter dictionary produced by :func:`build_paramdict`.
    return_currents:
        If ``True``, include currents, derived states, and effective parameters.

    Returns
    -------
    dict
        Always contains ``dzdt``.  If ``return_currents`` is true, also contains
        ``currents``, ``states_derived``, and ``effective_params``.

    Important effective parameters
    ------------------------------
    ``P_gap_eff = d_gap * P_k`` is exposed explicitly because ``d_gap`` and
    ``P_k`` are structurally confounded in this model.  Reporting this effective
    quantity is safer than interpreting ``d`` and ``P_k`` separately.
    """

    astro = params["Astrocyte"]
    external = params["external"]

    Cm_a = float(astro["Cm_a"])
    g_kir = float(astro["g_kir"])
    g_k_a = float(astro.get("g_k_a", 0.0))
    gl_a = float(astro["gl_a"])
    w_a = float(astro["w_a"])
    K_a0 = float(astro["K_a0"])
    Sig_a = float(astro["Sig_a"])
    gama_t = float(astro["gama_t"])
    gama_s = float(astro["gama_s"])
    Va_s = float(astro["Va_s"])
    Va_l = float(astro["Va_l"])
    P_k = float(astro["P_k"])
    d_gap = float(astro["d_gap"])
    F = float(astro["F"])

    K_o0 = float(external["K_o0"])
    K_bath, k_bath_interval_index, epsilon_eff, w_o_eff = _lookup_kbath(external, float(t_ms))

    Va, DK_a_t, K_a_s, Kg = [float(x) for x in z]

    DK_a = DK_a_t + K_a_s
    K_a = K_a0 + DK_a
    DK_o_a = -(w_a / w_o_eff) * DK_a_t
    K_o = K_o0 + DK_o_a + Kg
    K_ratio = max(K_o / max(K_a, 1e-12), 1e-12)

    E_k_a = 25.7 * np.log(K_ratio)
    I_k_a = g_k_a * (Va - E_k_a)

    # Kir current; use abs(K_o) as in the existing notebook for numerical stability.
    I_Kir = g_kir * np.sqrt(max(abs(K_o), 1e-12)) * (Va - E_k_a) / (1.0 + _safe_exp((Va - E_k_a) / 19.2))

    PH_a = 0.04 * (Va - Va_s)
    P_gap_eff = d_gap * P_k
    exp_neg_PH_a = _safe_exp(-PH_a)
    denominator = -1.0 + exp_neg_PH_a
    if abs(denominator) < 1e-12:
        denominator = 1e-12 if denominator >= 0 else -1e-12
    I_kgap = P_gap_eff * F * PH_a * (1.0 / denominator) * ((K_a * exp_neg_PH_a) - K_a0)
    I_l_a = gl_a * (Va - Va_l)
    Th_s = _switching_gate(DK_a, DK_a_t, astro)

    dVa = (-1.0 / Cm_a) * (I_Kir + I_k_a + I_l_a + I_kgap)
    gamma_t_eff = gama_t * Sig_a / (w_a * F)
    gamma_s_eff = gama_s * Sig_a / (w_a * F)
    dDK_a_t = -gamma_t_eff * (I_Kir + I_k_a)
    dK_a_s = -Th_s * gamma_s_eff * I_kgap
    dKg = epsilon_eff * (K_bath - K_o)

    result: Dict[str, Any] = {"dzdt": np.asarray([dVa, dDK_a_t, dK_a_s, dKg], dtype=float)}
    if return_currents:
        result.update(
            {
                "currents": {
                    "I_Kir": float(I_Kir),
                    "I_k_a": float(I_k_a),
                    "I_kgap": float(I_kgap),
                    "I_leak": float(I_l_a),
                    "Th_s": float(Th_s),
                },
                "states_derived": {
                    "K_o": float(K_o),
                    "K_a": float(K_a),
                    "DK_a": float(DK_a),
                    "E_k_a": float(E_k_a),
                    "K_bath": float(K_bath),
                    "K_bath_interval_index": int(k_bath_interval_index),
                    "w_o_effective": float(w_o_eff),
                    "epsilon_effective": float(epsilon_eff),
                },
                "effective_params": {
                    "P_gap_eff": float(P_gap_eff),
                    "gamma_t_eff": float(gamma_t_eff),
                    "gamma_s_eff": float(gamma_s_eff),
                    "volume_ratio_wa_wo": float(w_a / w_o_eff),
                },
            }
        )
    return result


def _rhs_for_odeint(z: Sequence[float], t_ms: float, params: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    return compute_rhs_and_currents(z, t_ms, params, return_currents=False)["dzdt"]


def normalize_trace_for_target_mode(trace: Sequence[float], mode: str = "centered") -> np.ndarray:
    """Normalize a trace using the modes used in the old fitting notebooks."""

    y = np.asarray(trace, dtype=float)
    mode = (mode or "default").lower()
    if mode in {"centered", "centered_l2", "centered_combined"}:
        return y - float(np.nanmean(y))
    if mode == "centered_scaled":
        centered = y - float(np.nanmean(y))
        scale = float(np.nanstd(centered))
        return centered / (scale if scale > 0 else 1.0)
    return y


def compute_loss(pred: Sequence[float], target: Sequence[float], loss_type: str = "COMBINED") -> float:
    """Compute a simple objective used for post-fit diagnostics.

    This is not intended to replace the original optimization objective in every
    detail.  It is a stable, explicit version for recomputation and sanity checks.
    """

    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    if p.shape != t.shape:
        raise ValueError(f"pred and target must have same shape, got {p.shape} vs {t.shape}")
    diff = p - t
    lt = loss_type.upper()
    if lt == "L1":
        return float(np.nanmean(np.abs(diff)))
    if lt == "L2":
        return float(np.sqrt(np.nanmean(diff**2)))
    if lt == "HUBER":
        delta = 1.0
        a = np.abs(diff)
        return float(np.nanmean(np.where(a <= delta, 0.5 * diff**2, delta * (a - 0.5 * delta))))
    # COMBINED: robust-ish trace loss used only for comparisons.
    return float(np.sqrt(np.nanmean(diff**2)) + 0.1 * np.nanmean(np.abs(diff)))


def compute_basic_vm_features(
    t_s: Sequence[float],
    vm: Sequence[float],
    onset_s: Optional[float] = None,
    offset_s: Optional[float] = None,
) -> Dict[str, float]:
    """Compute lightweight Vm features for post-fit screening.

    The ATF notebook remains the canonical source for publication-grade feature
    extraction.  This helper is intentionally minimal so simulations can return a
    stable feature dictionary without importing the notebook.
    """

    t = np.asarray(t_s, dtype=float)
    y = np.asarray(vm, dtype=float)
    if len(t) != len(y) or len(t) < 3:
        return {"feature_error": 1.0}

    onset_s = float(onset_s) if onset_s is not None else float(t[int(0.2 * len(t))])
    offset_s = float(offset_s) if offset_s is not None else float(t[int(0.6 * len(t))])

    baseline_mask = t < onset_s
    stim_mask = (t >= onset_s) & (t <= offset_s)
    post_mask = t > offset_s
    baseline = float(np.nanmedian(y[baseline_mask])) if np.any(baseline_mask) else float(y[0])
    yy = y - baseline

    peak = float(np.nanmax(yy[stim_mask])) if np.any(stim_mask) else float(np.nanmax(yy))
    trough_post = float(np.nanmin(yy[post_mask])) if np.any(post_mask) else float(np.nanmin(yy))
    stim_end = float(yy[np.argmin(np.abs(t - offset_s))])
    final_value = float(yy[-1])

    # Coarse slopes over robust windows.
    def _slope(mask: np.ndarray) -> float:
        if np.sum(mask) < 3:
            return np.nan
        x = t[mask]
        z = yy[mask]
        try:
            return float(np.polyfit(x, z, 1)[0])
        except Exception:
            return np.nan

    rise_mask = (t >= onset_s) & (t <= min(offset_s, onset_s + 0.25 * max(offset_s - onset_s, 1e-9)))
    decay_mask = post_mask & (t <= offset_s + 0.25 * max(t[-1] - offset_s, 1e-9))

    return {
        "baseline_mV": baseline,
        "peak_depolarization_mV": peak,
        "stim_end_depolarization_mV": stim_end,
        "undershoot_magnitude_mV": max(0.0, -trough_post),
        "final_value_mV": final_value,
        "rise_slope_mV_per_s": _slope(rise_mask),
        "decay_slope_mV_per_s": _slope(decay_mask),
        "has_undershoot": float(trough_post < -1e-9),
        "stim_onset_s": onset_s,
        "stim_offset_s": offset_s,
    }


def _collect_hidden_outputs(
    t_ms: np.ndarray,
    states: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, np.ndarray] | Dict[str, float]]:
    """Evaluate currents/derived states at every simulated time point."""

    current_keys = ["I_Kir", "I_k_a", "I_kgap", "I_leak", "Th_s"]
    derived_keys = ["K_o", "K_a", "DK_a", "E_k_a", "K_bath", "w_o_effective", "epsilon_effective"]
    effective_keys = ["P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo"]

    currents = {k: np.empty(len(t_ms), dtype=float) for k in current_keys}
    derived = {k: np.empty(len(t_ms), dtype=float) for k in derived_keys}
    effective_params: Dict[str, float] = {}

    for i, (tt, zz) in enumerate(zip(t_ms, states)):
        out = compute_rhs_and_currents(zz, float(tt), params, return_currents=True)
        for k in current_keys:
            currents[k][i] = out["currents"][k]
        for k in derived_keys:
            derived[k][i] = out["states_derived"][k]
        if not effective_params:
            effective_params = {k: float(out["effective_params"][k]) for k in effective_keys}

    return {"currents": currents, "derived": derived, "effective_params": effective_params}


def simulate_odeint(
    params: Mapping[str, Any],
    protocol: Mapping[str, Any],
    z0: Optional[Sequence[float]] = None,
    t_eval_ms: Optional[Sequence[float]] = None,
    return_hidden: bool = False,
) -> Dict[str, Any]:
    """Simulate the 4-state astrocyte model with ``scipy.integrate.odeint``.

    Parameters
    ----------
    params:
        Either a nested parameter dictionary from :func:`build_paramdict` or a
        flat Optuna-style parameter row.
    protocol:
        Dictionary with at least ``experiment_type`` and ``current_na`` if
        ``params`` is flat.  May also include ``t_eval_ms`` or ``t_final_ms``.
    z0:
        Initial state.  Defaults to ``[-89, 0, 0, 0]``.
    t_eval_ms:
        Explicit time grid in ms.  If absent, a grid is created from
        ``protocol['t_final_ms']`` and ``protocol['dt_ms']``.
    return_hidden:
        If true, evaluate currents/derived states at every time point.

    Returns
    -------
    dict
        ``t_ms``, ``states``, ``Vm``, ``params``, ``protocol`` and optionally
        ``hidden``/``currents``/``derived``/``effective_params``.
    """

    if odeint is None:  # pragma: no cover
        raise ImportError("scipy.integrate.odeint is required") from _SCIPY_IMPORT_ERROR

    paramdict = _ensure_paramdict(params, protocol)
    z0_arr = np.asarray(DEFAULT_Z0 if z0 is None else z0, dtype=float)
    if t_eval_ms is None:
        if "t_eval_ms" in protocol and protocol["t_eval_ms"] is not None:
            t_eval_ms = protocol["t_eval_ms"]
        else:
            t_final = float(protocol.get("t_final_ms", DEFAULT_T_FINAL_MS))
            dt = float(protocol.get("dt_ms", DEFAULT_DT_MS))
            t_eval_ms = np.arange(0.0, t_final + dt, dt, dtype=float)
    t_arr = np.asarray(t_eval_ms, dtype=float)

    states = odeint(_rhs_for_odeint, z0_arr, t_arr, args=(paramdict,))
    if not np.isfinite(states).all():
        raise ValueError("Non-finite values detected in ODE solution")

    out: Dict[str, Any] = {
        "t_ms": t_arr,
        "states": states,
        "Vm": states[:, 0].copy(),
        "params": paramdict,
        "protocol": dict(protocol),
        "solver": "odeint",
    }
    if return_hidden:
        hidden = _collect_hidden_outputs(t_arr, states, paramdict)
        out.update(hidden)
        out["hidden"] = {"currents": out["currents"], "derived": out["derived"]}
    return out


def simulate_rk4_numba(
    params: Mapping[str, Any],
    protocol: Mapping[str, Any],
    z0: Optional[Sequence[float]] = None,
    t_final_ms: Optional[float] = None,
    dt_ms: Optional[float] = None,
    return_hidden: bool = False,
) -> Dict[str, Any]:
    """Fixed-step RK4 backend initialized from astrosim's ``full_model`` idea.

    Despite the name, this initial version is pure Python/Numpy.  It is kept
    numerically transparent so it can be validated against ``simulate_odeint``.
    A future optimization can replace the loop with a Numba-jitted function using
    the same input/output contract.
    """

    paramdict = _ensure_paramdict(params, protocol)
    z0_arr = np.asarray(DEFAULT_Z0 if z0 is None else z0, dtype=float)
    dt = float(dt_ms if dt_ms is not None else protocol.get("dt_ms", DEFAULT_DT_MS))
    t_final = float(t_final_ms if t_final_ms is not None else protocol.get("t_final_ms", DEFAULT_T_FINAL_MS))
    if dt <= 0:
        raise ValueError("dt_ms must be positive")
    t_arr = np.arange(0.0, t_final + dt, dt, dtype=float)
    states = np.empty((len(t_arr), len(z0_arr)), dtype=float)
    states[0] = z0_arr

    for i in range(1, len(t_arr)):
        t0 = t_arr[i - 1]
        z = states[i - 1]
        k1 = _rhs_for_odeint(z, t0, paramdict)
        k2 = _rhs_for_odeint(z + 0.5 * dt * k1, t0 + 0.5 * dt, paramdict)
        k3 = _rhs_for_odeint(z + 0.5 * dt * k2, t0 + 0.5 * dt, paramdict)
        k4 = _rhs_for_odeint(z + dt * k3, t0 + dt, paramdict)
        states[i] = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(states[i]).all():
            raise ValueError(f"Non-finite RK4 state at step {i}, t={t_arr[i]}")

    out: Dict[str, Any] = {
        "t_ms": t_arr,
        "states": states,
        "Vm": states[:, 0].copy(),
        "params": paramdict,
        "protocol": dict(protocol),
        "solver": "rk4_python_initial",
        "dt_ms": dt,
    }
    if return_hidden:
        hidden = _collect_hidden_outputs(t_arr, states, paramdict)
        out.update(hidden)
        out["hidden"] = {"currents": out["currents"], "derived": out["derived"]}
    return out


def _downsample_median(y: np.ndarray, n: int) -> np.ndarray:
    """Simple median downsampling used for objective recomputation."""

    if n <= 0:
        raise ValueError("n must be positive")
    if len(y) == n:
        return y.copy()
    edges = np.linspace(0, len(y), n + 1).astype(int)
    out = np.empty(n, dtype=float)
    for i in range(n):
        segment = y[edges[i] : max(edges[i + 1], edges[i] + 1)]
        out[i] = float(np.nanmedian(segment)) if len(segment) else np.nan
    return out


def simulate_with_hidden_outputs(
    full_params: Mapping[str, Any],
    context: ExperimentContext | Mapping[str, Any],
    objective_loss_type: str = "COMBINED",
    solver: str = "odeint",
    cache: bool = False,
) -> Dict[str, Any]:
    """Simulate, align to experiment if available, and return hidden outputs.

    Source merge
    ------------
    - Existing ``simulate_trial_aligned``: protocol alignment, interpolation,
      target normalization, and objective recomputation.
    - Astrosim ``run_model_with_currents``: pattern of simulation plus current
      time-course extraction.  This implementation returns astrocyte currents,
      not the neuronal currents used in astrosim.

    Parameters
    ----------
    full_params:
        Flat Optuna-style parameters or a nested paramdict.
    context:
        :class:`ExperimentContext` or dict with at least ``experiment_type`` and
        ``current_na``.  If experimental arrays are provided, the output includes
        ``Vm_interp``, ``Vm_downsampled``, ``objective_recomputed``, and
        ``features``.
    objective_loss_type:
        ``COMBINED``, ``L2``, ``L1``, or ``HUBER`` for post-fit recomputation.
    solver:
        ``"odeint"`` or ``"rk4"``.
    cache:
        Placeholder for later disk caching.  Present to keep the target API
        compatible with the astrosim pattern.

    Returns
    -------
    dict
        Simulation output with hidden currents, derived states, effective
        parameters, and optional alignment/objective fields.
    """

    if cache:
        # Disk caching is intentionally not implemented in the initialization to
        # avoid silently reusing stale simulations.  A later version should hash
        # params+protocol+solver+dt and store compressed NPZ files.
        pass

    ctx = _as_context(context)
    protocol: Dict[str, Any] = {
        "experiment_type": ctx.experiment_type,
        "current_na": ctx.current_na,
        "target_mean_mode": ctx.target_mean_mode,
    }
    if ctx.sim_time_ms is not None:
        protocol["t_eval_ms"] = ctx.sim_time_ms
    elif ctx.exp_times_ms_stable is not None:
        # Simulate over the observed window when possible.
        protocol["t_eval_ms"] = ctx.exp_times_ms_stable
    else:
        protocol["t_final_ms"] = DEFAULT_T_FINAL_MS
        protocol["dt_ms"] = DEFAULT_DT_MS

    if solver.lower() in {"odeint", "scipy"}:
        sim = simulate_odeint(full_params, protocol, return_hidden=True)
    elif solver.lower() in {"rk4", "rk4_numba", "numba"}:
        sim = simulate_rk4_numba(full_params, protocol, return_hidden=True)
    else:
        raise ValueError("solver must be 'odeint' or 'rk4'")

    t_ms = sim["t_ms"]
    vm_raw = sim["Vm"]
    stable_idx = int(ctx.stable_index_simulation)
    stable_idx = min(max(stable_idx, 0), len(vm_raw) - 1)
    vm_stable_raw = vm_raw[stable_idx:]
    t_stable_ms = t_ms[stable_idx:]
    sim["Vm_stable_raw"] = vm_stable_raw
    sim["t_stable_ms"] = t_stable_ms

    if ctx.exp_times_ms_stable is not None:
        exp_t = np.asarray(ctx.exp_times_ms_stable, dtype=float)
        vm_interp = np.interp(exp_t, t_stable_ms, vm_stable_raw)
        vm_interp_norm = normalize_trace_for_target_mode(vm_interp, ctx.target_mean_mode)
        sim["Vm_interp"] = vm_interp_norm
        sim["features"] = compute_basic_vm_features(
            exp_t / 1000.0,
            vm_interp_norm,
            onset_s=ctx.feature_onset_s,
            offset_s=ctx.feature_offset_s,
        )

        if ctx.exp_trace_stable is not None:
            exp_trace = np.asarray(ctx.exp_trace_stable, dtype=float)
            vm_downsampled = _downsample_median(vm_stable_raw, len(exp_trace))
            vm_downsampled_norm = normalize_trace_for_target_mode(vm_downsampled, ctx.target_mean_mode)
            sim["Vm_downsampled"] = vm_downsampled_norm
            sim["objective_recomputed"] = compute_loss(vm_downsampled_norm, exp_trace, loss_type=objective_loss_type)
    return sim

########## FILE END: src/astro_model.py ##########
