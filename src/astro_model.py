from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.integrate import odeint

DEFAULT_Z0 = np.array([-89.0, 0.0, 0.0, 0.0], dtype=float)

CURRENT_DICT_K_BATH_VALUES: Dict[str, list[float]] = {
    "50": [4.8, 6.4, 4.8],
    "75": [4.8, 7.23, 4.8],
    "100": [4.8, 8.2, 4.8],
    "125": [4.8, 9.5, 4.8],
    "150": [4.8, 10.1, 4.8],
    "175": [4.8, 10.5, 4.8],
}

EXPERIMENT_K_BATH_TIME_MS: Dict[str, list[float]] = {
    "CONTROL": [0.0, 11173.0, 31173.0],
    "MFA": [0.0, 21140.0, 41140.0],
    "BARIUM": [0.0, 21140.0, 41140.0],
    "MFA_BA": [0.0, 21140.0, 41140.0],
}


def normalize_flat_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    aliases = {
        "Zth": "zth",
        "Z_th": "zth",
        "Zs": "zs",
        "Z_s": "zs",
        "gki": "gki",
        "pk": "pk",
        "gt": "gt",
        "gs": "gs",
        "d": "d",
        "ca": "ca",
        "wo_middle": "wo_middle",
        "w_o_middle": "wo_middle",
        "epsilon_middle": "eps_middle",
        "eps_middle": "eps_middle",
        "K_bath_value_middle": "K_bath_value_middle",
    }
    for old, new in aliases.items():
        if old in p and new not in p:
            p[new] = p[old]
    p.setdefault("w_a", 2000.0)
    p.setdefault("wo_middle", 1.0)
    p.setdefault("eps_middle", 1.0)
    p.setdefault("switching_function", "sigmoid")
    return p


def build_paramdict(experiment_type: str, current_na: int, flat_params: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    p = normalize_flat_params(flat_params)
    experiment_type = str(experiment_type).upper().replace("MFA+BA", "MFA_BA")
    current_key = str(int(current_na))
    k_bath_values = np.asarray(CURRENT_DICT_K_BATH_VALUES[current_key], dtype=float).copy()
    if "K_bath_value_middle" in p:
        k_bath_values[1] = float(p["K_bath_value_middle"])

    astro: Dict[str, Any] = {
        "Cm_a": float(p.get("ca", 400.0)),
        "g_kir": float(p.get("gki", 1.0)),
        "A": float(p.get("A", 1.0)),
        "g_k_a": float(p.get("g_k_a", 0.0)),
        "gl_a": float(p.get("gl_a", 0.01)),
        "w_a": float(p.get("w_a", 2000.0)),
        "K_a0": float(p.get("K_a0", 135.0)),
        "Sig_a": float(p.get("Sig_a", 1600.0)),
        "gama_t": float(p.get("gt", 6.0)),
        "gama_s": float(p.get("gs", 6.5)),
        "Z_th": float(p.get("zth", 0.2)),
        "Z_s": float(p.get("zs", 0.05)),
        "Va_0": float(p.get("Va_0", -89.0)),
        "Va_s": float(p.get("Va_s", -90.0)),
        "Va_l": float(p.get("Va_l", -70.0)),
        "P_k": float(p.get("pk", 3e-5)),
        "d_gap": float(p.get("d", 1.0)),
        "F": float(p.get("F", 96485.0)),
        "R": float(p.get("R", 8.314)),
        "T": float(p.get("T", 298.0)),
        "switching_function": str(p.get("switching_function", "sigmoid")),
    }
    if astro["switching_function"] == "hill":
        astro["hill_coefficient"] = float(p.get("hill_coefficient", 2.0))
        astro["K_d"] = float(p.get("K_d", 1.0))

    external: Dict[str, Any] = {
        "K_o0": float(p.get("K_o0", 4.8)),
        "w_o": float(p.get("wo", 1500.0)),
        "epsilon": float(p.get("eps", 1e-3)),
        "K_bath": {"time": np.asarray(EXPERIMENT_K_BATH_TIME_MS[experiment_type], dtype=float), "value": k_bath_values},
    }
    if "eps_middle" in p:
        external["epsilon_middle"] = float(p["eps_middle"])
    if "wo_middle" in p:
        external["w_o_middle"] = float(p["wo_middle"])
    return {"Astrocyte": astro, "external": external}


def _safe_exp(x: float) -> float:
    return float(np.exp(np.clip(x, -700.0, 700.0)))


def model(z: Sequence[float], t: float, paramdict: Mapping[str, Mapping[str, Any]]):
    Cm_a = paramdict["Astrocyte"]["Cm_a"]
    g_kir = paramdict["Astrocyte"]["g_kir"]
    A = paramdict["Astrocyte"]["A"]
    g_k_a = paramdict["Astrocyte"]["g_k_a"]
    gl_a = paramdict["Astrocyte"]["gl_a"]
    w_a = paramdict["Astrocyte"]["w_a"]
    K_a0 = paramdict["Astrocyte"]["K_a0"]
    Sig_a = paramdict["Astrocyte"]["Sig_a"]
    gama_t = paramdict["Astrocyte"]["gama_t"]
    gama_s = paramdict["Astrocyte"]["gama_s"]
    Z_th = paramdict["Astrocyte"]["Z_th"]
    Z_s = paramdict["Astrocyte"]["Z_s"]
    Va_0 = paramdict["Astrocyte"]["Va_0"]
    Va_s = paramdict["Astrocyte"]["Va_s"]
    Va_l = paramdict["Astrocyte"]["Va_l"]
    P_k = paramdict["Astrocyte"]["P_k"]
    d_gap = paramdict["Astrocyte"]["d_gap"]
    F = paramdict["Astrocyte"]["F"]
    R = paramdict["Astrocyte"]["R"]
    T = paramdict["Astrocyte"]["T"]
    K_o0 = paramdict["external"]["K_o0"]
    w_o = paramdict["external"]["w_o"]
    epsilon = paramdict["external"]["epsilon"]
    idx = np.where(paramdict["external"]["K_bath"]["time"] <= t)[0][-1]
    K_bath = paramdict["external"]["K_bath"]["value"][idx]
    switching_function = paramdict["Astrocyte"].get("switching_function", "sigmoid")

    if "epsilon_middle" in paramdict["external"] and idx == 1:
        epsilon = epsilon * paramdict["external"]["epsilon_middle"]
    if "w_o_middle" in paramdict["external"] and idx == 1:
        w_o = w_o * paramdict["external"]["w_o_middle"]

    Va = float(z[0])
    DK_a_t = float(z[1])
    K_a_s = float(z[2])
    Kg = float(z[3])

    DK_a = DK_a_t + K_a_s
    K_a = K_a0 + DK_a
    DK_o_a = -(w_a / w_o) * DK_a_t
    K_o = K_o0 + DK_o_a + Kg
    K_ratio = K_o / K_a
    if K_ratio <= 0:
        K_ratio = 1e-8
    E_k_a = 25.7 * np.log(K_ratio)
    I_k_a = g_k_a * (Va - E_k_a)
    I_Kir = g_kir * np.sqrt(np.abs(K_o)) * (Va - E_k_a) * (1.0 / (1.0 + _safe_exp((Va - E_k_a) / 19.2)))
    PH_a = 0.04 * (Va - Va_s)
    P_kgap = d_gap * P_k

    exp_neg_PH_a = _safe_exp(-PH_a)
    denominator = -1.0 + _safe_exp(-PH_a)
    if denominator == 0:
        denominator = 1e-8
    I_kgap = P_kgap * F * PH_a * (1.0 / denominator) * ((K_a * exp_neg_PH_a) - K_a0)

    I_l_a = gl_a * (Va - Va_l)
    if switching_function == "sigmoid":
        Th_s = DK_a / (1.0 + _safe_exp((Z_th - DK_a_t) * Z_s))
    elif switching_function == "tanh":
        Th_s = DK_a * (0.5 * (1.0 + np.tanh((DK_a_t - Z_th) * Z_s)))
    elif switching_function == "hill":
        n = paramdict["Astrocyte"].get("hill_coefficient", 2)
        K_d = paramdict["Astrocyte"].get("K_d", 1)
        Th_s = DK_a * ((DK_a_t ** n) / (K_d ** n + DK_a_t ** n))
    else:
        raise ValueError(f"Unknown switching function type: {switching_function}")

    dVa = (-1.0 / Cm_a) * (I_Kir + I_k_a + I_l_a + I_kgap)
    dDK_a_t = -(gama_t * Sig_a / (w_a * F)) * (I_Kir + I_k_a)
    dK_a_s = -Th_s * (gama_s * Sig_a / (w_a * F)) * I_kgap
    dKg = epsilon * (K_bath - K_o)

    return np.asarray([dVa, dDK_a_t, dK_a_s, dKg], dtype=float)


def compute_rhs_and_currents(z: Sequence[float], t_ms: float, params: Mapping[str, Mapping[str, Any]], return_currents: bool = False) -> Dict[str, Any]:
    dzdt = model(z, t_ms, params)
    astro = params["Astrocyte"]
    external = params["external"]
    idx = np.where(external["K_bath"]["time"] <= t_ms)[0][-1]
    epsilon = float(external["epsilon"])
    w_o = float(external["w_o"])
    if "epsilon_middle" in external and idx == 1:
        epsilon = epsilon * float(external["epsilon_middle"])
    if "w_o_middle" in external and idx == 1:
        w_o = w_o * float(external["w_o_middle"])
    Va = float(z[0]); DK_a_t = float(z[1]); K_a_s = float(z[2]); Kg = float(z[3])
    DK_a = DK_a_t + K_a_s
    K_a = float(astro["K_a0"]) + DK_a
    K_o = float(external["K_o0"]) - (float(astro["w_a"]) / w_o) * DK_a_t + Kg
    K_ratio = K_o / K_a
    if K_ratio <= 0:
        K_ratio = 1e-8
    E_k_a = 25.7 * np.log(K_ratio)
    I_k_a = float(astro["g_k_a"]) * (Va - E_k_a)
    I_Kir = float(astro["g_kir"]) * np.sqrt(np.abs(K_o)) * (Va - E_k_a) * (1.0 / (1.0 + _safe_exp((Va - E_k_a) / 19.2)))
    PH_a = 0.04 * (Va - float(astro["Va_s"]))
    P_kgap = float(astro["d_gap"]) * float(astro["P_k"])
    exp_neg_PH_a = _safe_exp(-PH_a)
    denominator = -1.0 + _safe_exp(-PH_a)
    if denominator == 0:
        denominator = 1e-8
    I_kgap = P_kgap * float(astro["F"]) * PH_a * (1.0 / denominator) * ((K_a * exp_neg_PH_a) - float(astro["K_a0"]))
    I_leak = float(astro["gl_a"]) * (Va - float(astro["Va_l"]))
    out: Dict[str, Any] = {"dzdt": np.asarray(dzdt, dtype=float)}
    if return_currents:
        out.update({
            "currents": {"I_Kir": float(I_Kir), "I_kgap": float(I_kgap), "I_leak": float(I_leak), "I_k_a": float(I_k_a)},
            "derived": {"K_o": float(K_o), "DK_a": float(DK_a), "P_gap_eff": float(P_kgap), "epsilon_eff": float(epsilon), "w_o_eff": float(w_o)},
        })
    return out


def simulate_odeint(params: Mapping[str, Any], protocol: Mapping[str, Any], z0: Optional[Sequence[float]] = None, t_eval_ms: Optional[Sequence[float]] = None, return_hidden: bool = False) -> Dict[str, Any]:
    experiment_type = str(protocol.get("experiment_type", "CONTROL")).upper().replace("MFA+BA", "MFA_BA")
    current_na = int(protocol.get("current_na", 100))
    paramdict = build_paramdict(experiment_type, current_na, params)
    z0_arr = np.asarray(DEFAULT_Z0 if z0 is None else z0, dtype=float)
    t_arr = np.asarray(protocol.get("t_eval_ms") if t_eval_ms is None else t_eval_ms, dtype=float)
    states = odeint(model, z0_arr, t_arr, args=(paramdict,))
    if not np.isfinite(states).all():
        raise ValueError("Non-finite values detected in ODE solution")
    out: Dict[str, Any] = {"t_ms": t_arr, "states": states, "Vm": states[:, 0].copy(), "params": paramdict, "protocol": dict(protocol)}
    if return_hidden:
        currents = {k: np.empty(len(t_arr), dtype=float) for k in ["I_Kir", "I_kgap", "I_leak", "I_k_a"]}
        derived = {k: np.empty(len(t_arr), dtype=float) for k in ["K_o", "DK_a", "P_gap_eff", "epsilon_eff", "w_o_eff"]}
        for i, (tt, zz) in enumerate(zip(t_arr, states)):
            hidden = compute_rhs_and_currents(zz, float(tt), paramdict, return_currents=True)
            for k in currents:
                currents[k][i] = hidden["currents"][k]
            for k in derived:
                derived[k][i] = hidden["derived"][k]
        out["currents"] = currents
        out["derived"] = derived
    return out


def reference_model_rhs(z: Sequence[float], t_ms: float, paramdict: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    """Direct reference implementation of the reviewer-facing model equations.

    This mirrors the notebook model shared for manuscript review discussions and
    is used only for validation checks. The production helpers
    :func:`compute_rhs_and_currents` and :func:`simulate_odeint` should remain
    numerically equivalent to this reference implementation for supported
    switching functions (sigmoid, tanh, hill).
    """
    astro = paramdict["Astrocyte"]
    external = paramdict["external"]

    Cm_a = astro["Cm_a"]
    g_kir = astro["g_kir"]
    g_k_a = astro["g_k_a"]
    gl_a = astro["gl_a"]
    w_a = astro["w_a"]
    K_a0 = astro["K_a0"]
    Sig_a = astro["Sig_a"]
    gama_t = astro["gama_t"]
    gama_s = astro["gama_s"]
    Z_th = astro["Z_th"]
    Z_s = astro["Z_s"]
    Va_s = astro["Va_s"]
    Va_l = astro["Va_l"]
    P_k = astro["P_k"]
    d_gap = astro["d_gap"]
    F = astro["F"]

    K_o0 = external["K_o0"]
    w_o = external["w_o"]
    epsilon = external["epsilon"]
    idx = np.where(np.asarray(external["K_bath"]["time"], dtype=float) <= t_ms)[0][-1]
    K_bath = np.asarray(external["K_bath"]["value"], dtype=float)[idx]

    if "epsilon_middle" in external and idx == 1:
        epsilon = epsilon * external["epsilon_middle"]
    if "w_o_middle" in external and idx == 1:
        w_o = w_o * external["w_o_middle"]

    Va = float(z[0])
    DK_a_t = float(z[1])
    K_a_s = float(z[2])
    Kg = float(z[3])

    DK_a = DK_a_t + K_a_s
    K_a = K_a0 + DK_a
    DK_o_a = -(w_a / w_o) * DK_a_t
    K_o = K_o0 + DK_o_a + Kg

    K_ratio = K_o / K_a
    if K_ratio <= 0:
        K_ratio = 1e-8
    E_k_a = 25.7 * np.log(K_ratio)
    I_k_a = g_k_a * (Va - E_k_a)
    I_Kir = g_kir * np.sqrt(np.abs(K_o)) * (Va - E_k_a) * (1 / (1 + np.exp((Va - E_k_a) / 19.2)))
    PH_a = 0.04 * (Va - Va_s)
    P_kgap = d_gap * P_k
    exp_neg_PH_a = np.exp(-PH_a)
    denominator = -1 + np.exp(-PH_a)
    if denominator == 0:
        denominator = 1e-8
    I_kgap = P_kgap * F * PH_a * (1 / denominator) * ((K_a * exp_neg_PH_a) - K_a0)
    I_l_a = gl_a * (Va - Va_l)

    switching_function = astro.get("switching_function", "sigmoid")
    if switching_function == "sigmoid":
        Th_s = DK_a / (1 + np.exp((Z_th - DK_a_t) * Z_s))
    elif switching_function == "tanh":
        Th_s = DK_a * (0.5 * (1 + np.tanh((DK_a_t - Z_th) * Z_s)))
    elif switching_function == "hill":
        n = astro.get("hill_coefficient", 2)
        K_d = astro.get("K_d", 1)
        Th_s = DK_a * ((DK_a_t ** n) / (K_d ** n + DK_a_t ** n))
    else:
        raise ValueError(f"Unknown switching function type: {switching_function}")

    dVa = (-1.0 / Cm_a) * (I_Kir + I_k_a + I_l_a + I_kgap)
    dDK_a_t = -(gama_t * Sig_a / (w_a * F)) * (I_Kir + I_k_a)
    dK_a_s = -Th_s * (gama_s * Sig_a / (w_a * F)) * I_kgap
    dKg = epsilon * (K_bath - K_o)
    return np.asarray([dVa, dDK_a_t, dK_a_s, dKg], dtype=float)


def model_alignment_probe(flat_params: Mapping[str, Any], experiment_type: str, current_na: int, z: Sequence[float], t_ms_values: Sequence[float]) -> Dict[str, Any]:
    """Numerically compare production RHS against the reviewer-facing reference model."""
    paramdict = build_paramdict(experiment_type, current_na, flat_params)
    rows = []
    for t_ms in t_ms_values:
        prod = compute_rhs_and_currents(z, float(t_ms), paramdict, return_currents=True)
        ref = reference_model_rhs(z, float(t_ms), paramdict)
        rows.append({
            "t_ms": float(t_ms),
            "max_abs_rhs_delta": float(np.max(np.abs(prod["dzdt"] - ref))),
            "mean_abs_rhs_delta": float(np.mean(np.abs(prod["dzdt"] - ref))),
        })
    max_delta = max(row["max_abs_rhs_delta"] for row in rows) if rows else 0.0
    return {
        "experiment_type": str(experiment_type),
        "current_na": int(current_na),
        "n_samples": int(len(rows)),
        "max_abs_rhs_delta": float(max_delta),
        "status": "exact_within_float_tolerance" if max_delta <= 1e-12 else "mismatch",
        "rows": rows,
    }
