from __future__ import annotations

import numpy as np
import pytest

from src.astro_model import ASTRO_DEFAULTS, EXTERNAL_DEFAULTS, CURRENT_DICT_K_BATH_VALUES, model


def reference_model(z, t, paramdict):
    Cm_a  = paramdict["Astrocyte"]["Cm_a"]
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
    F  = paramdict["Astrocyte"]["F"]
    R  = paramdict["Astrocyte"]["R"]
    T  = paramdict["Astrocyte"]["T"]
    K_o0 =paramdict["external"]["K_o0"]
    w_o = paramdict["external"]["w_o"]
    epsilon = paramdict["external"]["epsilon"]
    idx = np.where(paramdict["external"]["K_bath"]["time"]<=t)[0][-1]
    K_bath = paramdict["external"]["K_bath"]["value"][idx]
    switching_function = paramdict["Astrocyte"].get("switching_function", "sigmoid")

    if "epsilon_middle" in paramdict["external"] and idx == 1:
      epsilon = epsilon*paramdict["external"]["epsilon_middle"]
    if "w_o_middle" in paramdict["external"] and idx == 1:
      w_o = w_o*paramdict["external"]["w_o_middle"]

    Va  = z[0]
    DK_a_t = z[1]
    K_a_s = z[2]
    Kg = z[3]

    DK_a = DK_a_t + K_a_s
    K_a  = K_a0 +DK_a
    DK_o_a = -(w_a/w_o)*DK_a_t
    K_o  = K_o0 + DK_o_a + Kg
    K_ratio = K_o / K_a
    if K_ratio <= 0: K_ratio = 1e-8
    E_k_a = 25.7 * np.log(K_ratio)
    I_k_a = g_k_a*(Va - E_k_a)
    I_Kir = g_kir * np.sqrt(np.abs(K_o))*(Va - E_k_a)*(1/(1+np.exp((Va - E_k_a)/19.2)))
    PH_a = 0.04*(Va - Va_s)
    P_kgap = d_gap*P_k

    exp_neg_PH_a = np.exp(-PH_a)
    denominator = -1 + np.exp(-PH_a)
    if denominator == 0: denominator = 1e-8
    I_kgap = P_kgap * F * PH_a * (1 / denominator) * ((K_a * exp_neg_PH_a) - K_a0)

    I_l_a  = gl_a*(Va - Va_l)
    if switching_function == "sigmoid":
        Th_s = DK_a / (1 + np.exp((Z_th- DK_a_t) * Z_s))
    elif switching_function == "tanh":
        Th_s = DK_a * (0.5 * (1 + np.tanh((DK_a_t - Z_th) * Z_s)))
    elif switching_function == "hill":
        n = paramdict["Astrocyte"].get("hill_coefficient", 2)
        K_d = paramdict["Astrocyte"].get("K_d", 1)
        Th_s = DK_a * ((DK_a_t ** n) / (K_d ** n + DK_a_t ** n))
    else:
        raise ValueError(f"Unknown switching function type: {switching_function}")
    dVa   = (-1.0/Cm_a)*(I_Kir + I_k_a +I_l_a +I_kgap)
    dDK_a_t = -(gama_t*Sig_a/(w_a*F))*(I_Kir + I_k_a)
    dK_a_s = -Th_s*(gama_s*Sig_a/(w_a*F))* I_kgap
    dKg   =  epsilon*(K_bath-K_o)

    return [dVa,dDK_a_t,dK_a_s,dKg]


def build_paramdict(switching_function='sigmoid'):
    astro = dict(ASTRO_DEFAULTS)
    astro.update({
        'Cm_a': 400.0,
        'g_kir': 32.0,
        'g_k_a': 0.0,
        'gl_a': 0.012,
        'w_a': 2000.0,
        'P_k': 2.7e-5,
        'gama_t': 7.5,
        'gama_s': 8.1,
        'Z_th': 0.18,
        'Z_s': 0.07,
        'Va_s': -90.0,
        'Va_l': -70.0,
        'd_gap': 0.8,
        'switching_function': switching_function,
    })
    if switching_function == 'hill':
        astro['hill_coefficient'] = 2.3
        astro['K_d'] = 0.4
    ext = dict(EXTERNAL_DEFAULTS)
    ext.update({
        'K_o0': 4.8,
        'w_o': 1500.0,
        'epsilon': 0.012,
        'epsilon_middle': 0.85,
        'w_o_middle': 1.15,
        'K_bath': {
            'time': np.array([0.0, 11173.0, 31173.0], dtype=float),
            'value': np.array(CURRENT_DICT_K_BATH_VALUES['100'], dtype=float),
        },
    })
    return {'Astrocyte': astro, 'external': ext}


@pytest.mark.parametrize('switching_function', ['sigmoid', 'tanh', 'hill'])
def test_reference_model_matches_repo_model(switching_function: str) -> None:
    z = np.array([-84.7, 0.11, 0.03, 0.42], dtype=float)
    t = 15000.0
    paramdict = build_paramdict(switching_function)
    expected = np.asarray(reference_model(z, t, paramdict), dtype=float)
    observed = np.asarray(model(z, t, paramdict), dtype=float)
    assert np.allclose(observed, expected, rtol=1e-10, atol=1e-10)
