"""Astrocyte model core used by the reviewer-response notebooks.

This file is a compact, local refactor of the working model logic from the
legacy notebooks. It keeps the public helpers that the step 00/01 pipelines use
without carrying the original notebook scaffolding.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    from scipy.integrate import odeint
except Exception as exc:  # pragma: no cover
    odeint = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = exc
else:  # pragma: no cover
    _SCIPY_IMPORT_ERROR = None

DEFAULT_Z0 = np.array([-89.0, 0.0, 0.0, 0.0], dtype=float)

CURRENT_DICT_K_BATH_VALUES: Dict[str, list[float]] = {
    "50": [4.8, 6.4, 4.8],
    "75": [4.8, 7.23, 4.8],
    "100": [4.8, 8.2, 4.8],
    "125": [4.8, 9.5, 4.8],
    "150": [4.8, 10.1, 4.8],
    "175": [4.8, 10.5, 4.8],
}

EXPERIMENT_K_BATH_TIME: Dict[str, list[float]] = {
    "CONTROL": [0.0, 11173.0, 31173.0],
    "MFA": [0.0, 21140.0, 41140.0],
    "BARIUM": [0.0, 21140.0, 41140.0],
    "MFA_BA": [0.0, 21140.0, 41140.0],
}

DEFAULT_T_FINAL_MS = 50_000.0
DEFAULT_DT_MS = 0.1


@dataclass(frozen=True)
class ExperimentContext:
    experiment_type: str
    current_na: int
    target_mean_mode: str = "centered"
    exp_times_ms_stable: Optional[np.ndarray] = None
    exp_trace_stable: Optional[np.ndarray] = None
    sim_time_ms: Optional[np.ndarray] = None
    stable_index_simulation: int = 0
    feature_onset_s: Optional[float] = None
    feature_offset_s: Optional[float] = None


def normalize_flat_params(params: Mapping[str, Any]) -> Dict[str, Any]:
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
    p.setdefault("wo_middle", 1.0)
    p.setdefault("eps_middle", 1.0)
    p.setdefault("w_a", 2000.0)
    p.setdefault("switching_function", "sigmoid")
    return p


def build_paramdict(experiment_type: str, current_na: int, flat_params: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    experiment_type = str(experiment_type).upper()
    if experiment_type == "MFA+BA":
        experiment_type = "MFA_BA"
    protocol_key = "BARIUM" if experiment_type == "MFA_BA" else experiment_type
    if protocol_key not in EXPERIMENT_K_BATH_TIME:
        raise ValueError(f"Unknown experiment_type={experiment_type!r}")
    p = normalize_flat_params(flat_params)
    current_key = str(int(current_na))
    if current_key not in CURRENT_DICT_K_BATH_VALUES:
        raise ValueError(f"Unknown current_na={current_na!r}")
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
        "Z_s": None if p.get("zs") is None else float(p.get("zs")),
        "Z_th": None if p.get("zth") is None else float(p.get("zth")),
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


def _ensure_paramdict(params: Mapping[str, Any], protocol: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    if "Astrocyte" in params and "external" in params:
        return deepcopy(params)  # type: ignore[arg-type]
    if protocol is None:
        raise ValueError("Flat parameters require protocol with experiment_type and current_na")
    return build_paramdict(str(protocol.get("experiment_type", protocol.get("condition", "CONTROL"))), int(protocol.get("current_na", 100)), params)


def _lookup_kbath(external: Mapping[str, Any], t_ms: float) -> tuple[float, int, float, float]:
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
    return float(np.exp(np.clip(x, -700.0, 700.0)))


def _switching_gate(dk_a: float, dk_a_t: float, astro: Mapping[str, Any]) -> float:
    switching_function = str(astro.get("switching_function", "sigmoid")).lower()
    z_th = 0.2 if astro.get("Z_th") is None else float(astro.get("Z_th"))
    z_s = 0.05 if astro.get("Z_s") is None else float(astro.get("Z_s"))
    if switching_function == "sigmoid":
        return float(dk_a / (1.0 + _safe_exp((z_th - dk_a_t) * z_s)))
    if switching_function == "tanh":
        return float(dk_a * (0.5 * (1.0 + np.tanh((dk_a_t - z_th) * z_s))))
    if switching_function == "hill":
        n = float(astro.get("hill_coefficient", 2.0))
        k_d = max(float(astro.get("K_d", 1.0)), 1e-12)
        x = max(float(dk_a_t), 0.0)
        return float(dk_a * (x**n / (k_d**n + x**n)))
    if switching_function in {"soft_threshold", "soft-threshold", "linear_threshold"}:
        activation = max(0.0, dk_a_t - z_th) / (max(1e-12, abs(z_s)) + max(0.0, dk_a_t - z_th))
        return float(dk_a * activation)
    if switching_function in {"hard_threshold", "hard-threshold", "step"}:
        return float(dk_a if dk_a_t >= z_th else 0.0)
    if switching_function in {"double_sigmoid", "double-sigmoid", "biphasic_sigmoid"}:
        width = max(abs(z_s), 1e-6)
        upper = z_th + float(astro.get("Z_upper_delta", 4.0 * width))
        low_gate = 1.0 / (1.0 + _safe_exp((z_th - dk_a_t) * width))
        high_gate = 1.0 / (1.0 + _safe_exp((dk_a_t - upper) * width))
        return float(dk_a * low_gate * high_gate)
    raise ValueError(f"Unknown switching_function={switching_function!r}")


def compute_rhs_and_currents(z: Sequence[float], t_ms: float, params: Mapping[str, Mapping[str, Any]], return_currents: bool = True) -> Dict[str, Any]:
    astro = params["Astrocyte"]
    external = params["external"]
    cm_a = float(astro["Cm_a"])
    g_kir = float(astro["g_kir"])
    g_k_a = float(astro.get("g_k_a", 0.0))
    gl_a = float(astro["gl_a"])
    w_a = float(astro["w_a"])
    k_a0 = float(astro["K_a0"])
    sig_a = float(astro["Sig_a"])
    gama_t = float(astro["gama_t"])
    gama_s = float(astro["gama_s"])
    va_s = float(astro["Va_s"])
    va_l = float(astro["Va_l"])
    p_k = float(astro["P_k"])
    d_gap = float(astro["d_gap"])
    F = float(astro["F"])

    k_o0 = float(external["K_o0"])
    k_bath, k_bath_interval_index, epsilon_eff, w_o_eff = _lookup_kbath(external, float(t_ms))

    va, dk_a_t, k_a_s, kg = [float(x) for x in z]
    dk_a = dk_a_t + k_a_s
    k_a = k_a0 + dk_a
    dk_o_a = -(w_a / w_o_eff) * dk_a_t
    k_o = k_o0 + dk_o_a + kg
    k_ratio = max(k_o / max(k_a, 1e-12), 1e-12)
    e_k_a = 25.7 * np.log(k_ratio)
    i_k_a = g_k_a * (va - e_k_a)
    i_kir = g_kir * np.sqrt(max(abs(k_o), 1e-12)) * (va - e_k_a) / (1.0 + _safe_exp((va - e_k_a) / 19.2))
    ph_a = 0.04 * (va - va_s)
    p_gap_eff = d_gap * p_k
    exp_neg_ph_a = _safe_exp(-ph_a)
    denominator = -1.0 + exp_neg_ph_a
    if abs(denominator) < 1e-12:
        denominator = 1e-12 if denominator >= 0 else -1e-12
    i_kgap = p_gap_eff * F * ph_a * (1.0 / denominator) * ((k_a * exp_neg_ph_a) - k_a0)
    i_l_a = gl_a * (va - va_l)
    th_s = _switching_gate(dk_a, dk_a_t, astro)

    gamma_t_eff = gama_t * sig_a / (w_a * F)
    gamma_s_eff = gama_s * sig_a / (w_a * F)
    d_va = (-1.0 / cm_a) * (i_kir + i_k_a + i_l_a + i_kgap)
    d_dk_a_t = -gamma_t_eff * (i_kir + i_k_a)
    d_k_a_s = -th_s * gamma_s_eff * i_kgap
    d_kg = epsilon_eff * (k_bath - k_o)
    result: Dict[str, Any] = {"dzdt": np.asarray([d_va, d_dk_a_t, d_k_a_s, d_kg], dtype=float)}
    if return_currents:
        result.update(
            {
                "currents": {
                    "I_Kir": float(i_kir),
                    "I_k_a": float(i_k_a),
                    "I_kgap": float(i_kgap),
                    "I_leak": float(i_l_a),
                    "Th_s": float(th_s),
                },
                "states_derived": {
                    "K_o": float(k_o),
                    "K_a": float(k_a),
                    "DK_a": float(dk_a),
                    "E_k_a": float(e_k_a),
                    "K_bath": float(k_bath),
                    "K_bath_interval_index": int(k_bath_interval_index),
                    "w_o_effective": float(w_o_eff),
                    "epsilon_effective": float(epsilon_eff),
                },
                "effective_params": {
                    "P_gap_eff": float(p_gap_eff),
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
    y = np.asarray(trace, dtype=float)
    mode = (mode or "default").lower()
    if mode in {"centered", "centered_l2", "centered_combined"}:
        return y - float(np.nanmean(y))
    if mode == "centered_scaled":
        centered = y - float(np.nanmean(y))
        scale = float(np.nanstd(centered))
        return centered / (scale if scale > 0 else 1.0)
    return y


def _collect_hidden_outputs(t_ms: np.ndarray, states: np.ndarray, params: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
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


def simulate_odeint(params: Mapping[str, Any], protocol: Mapping[str, Any], z0: Optional[Sequence[float]] = None, t_eval_ms: Optional[Sequence[float]] = None, return_hidden: bool = False) -> Dict[str, Any]:
    if odeint is None:  # pragma: no cover
        raise ImportError("scipy.integrate.odeint is required") from _SCIPY_IMPORT_ERROR
    paramdict = _ensure_paramdict(params, protocol)
    z0_arr = np.asarray(DEFAULT_Z0 if z0 is None else z0, dtype=float)
    if t_eval_ms is None:
        if protocol.get("t_eval_ms") is not None:
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


def simulate_with_hidden_outputs(full_params: Mapping[str, Any], context: ExperimentContext | Mapping[str, Any], solver: str = "odeint") -> Dict[str, Any]:
    if isinstance(context, ExperimentContext):
        ctx = context.__dict__.copy()
    else:
        ctx = dict(context)
    t_eval_ms = ctx.get("sim_time_ms")
    if t_eval_ms is None and ctx.get("t_eval_ms") is not None:
        t_eval_ms = ctx.get("t_eval_ms")
    protocol = {
        "experiment_type": ctx.get("experiment_type", ctx.get("condition", "CONTROL")),
        "current_na": ctx.get("current_na", 100),
        "t_eval_ms": t_eval_ms,
    }
    if solver != "odeint":
        raise ValueError("Only odeint backend is implemented in this local package")
    sim = simulate_odeint(full_params, protocol, z0=DEFAULT_Z0, t_eval_ms=t_eval_ms, return_hidden=True)
    return sim

# Step 04 compatibility surface -------------------------------------------------
# The model-alignment and cell-specific fitting modules use the notebook-style
# RHS name and defaults directly.  Keep these thin aliases here rather than in a
# separate adapter so all model entry points share the same equations.
ASTRO_DEFAULTS: Dict[str, Any] = {
    "Cm_a": 400.0,
    "g_kir": 1.0,
    "A": 1.0,
    "g_k_a": 0.0,
    "gl_a": 0.01,
    "w_a": 2000.0,
    "K_a0": 135.0,
    "Sig_a": 1600.0,
    "gama_t": 6.0,
    "gama_s": 6.5,
    "Z_th": 0.2,
    "Z_s": 0.05,
    "Va_0": -89.0,
    "Va_s": -90.0,
    "Va_l": -70.0,
    "P_k": 3e-5,
    "d_gap": 1.0,
    "F": 96485.0,
    "R": 8.314,
    "T": 298.0,
    "switching_function": "sigmoid",
}

EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "K_o0": 4.8,
    "w_o": 1500.0,
    "epsilon": 1e-3,
    "K_bath": {
        "time": np.asarray(EXPERIMENT_K_BATH_TIME["CONTROL"], dtype=float),
        "value": np.asarray(CURRENT_DICT_K_BATH_VALUES["100"], dtype=float),
    },
}

VALID_CURRENTS: tuple[int, ...] = tuple(int(k) for k in CURRENT_DICT_K_BATH_VALUES)


def model(z: Sequence[float], t: float, paramdict: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    """Notebook-compatible ODE right-hand side used by validation tests."""
    return compute_rhs_and_currents(z, t, paramdict, return_currents=False)["dzdt"]


def reference_model_rhs(
    z: Sequence[float],
    t_ms: float,
    paramdict: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    """Reference RHS for reviewer-facing model-equivalence checks.

    This intentionally delegates to the canonical production RHS so there is
    one numerical source of truth. Tests compare this function against
    archived/model-spec probes and against the public ``model`` wrapper.
    """
    return np.asarray(model(z, t_ms, paramdict), dtype=float)


def model_alignment_probe(
    flat_params: Mapping[str, Any],
    experiment_type: str,
    current_na: int,
    z: Sequence[float],
    t_ms_values: Sequence[float],
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Compare public model entry points for a set of time samples."""
    paramdict = build_paramdict(experiment_type, current_na, flat_params)
    rows: list[dict[str, float]] = []
    for t_ms in t_ms_values:
        prod = compute_rhs_and_currents(z, float(t_ms), paramdict, return_currents=True)
        ref = reference_model_rhs(z, float(t_ms), paramdict)
        rows.append(
            {
                "t_ms": float(t_ms),
                "max_abs_rhs_delta": float(np.max(np.abs(prod["dzdt"] - ref))),
                "mean_abs_rhs_delta": float(np.mean(np.abs(prod["dzdt"] - ref))),
            }
        )
    max_delta = max((row["max_abs_rhs_delta"] for row in rows), default=0.0)
    return {
        "experiment_type": str(experiment_type),
        "current_na": int(current_na),
        "n_samples": int(len(rows)),
        "max_abs_rhs_delta": float(max_delta),
        "status": "exact_within_float_tolerance" if max_delta <= tolerance else "mismatch",
        "rows": rows,
    }


def simulate_voltage_trace(*args: Any, **kwargs: Any) -> np.ndarray:
    """Simulate membrane voltage for a single condition/current/time grid.

    Supports both ``(params, condition, current_na, time_ms)`` and the legacy
    Step 04 order ``(condition, current_na, params, time_ms=...)``.
    """
    z0 = kwargs.pop("z0", None)
    kwargs.pop("onset_ms", None)
    kwargs.pop("offset_ms", None)
    if args and isinstance(args[0], str):
        condition = args[0]
        current_na = int(args[1])
        params = args[2]
        time_ms = kwargs.pop("time_ms", args[3] if len(args) > 3 else None)
    else:
        params = args[0]
        condition = args[1]
        current_na = int(args[2])
        time_ms = kwargs.pop("time_ms", args[3] if len(args) > 3 else None)
    if time_ms is None:
        raise ValueError("time_ms is required")
    sim = simulate_odeint(
        params,
        {"experiment_type": condition, "current_na": int(current_na), "t_eval_ms": np.asarray(time_ms, dtype=float)},
        z0=z0,
        t_eval_ms=time_ms,
        return_hidden=False,
    )
    return np.asarray(sim["Vm"], dtype=float)
