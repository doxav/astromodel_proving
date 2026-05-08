from __future__ import annotations
import numpy as np
from src.astro_model import build_paramdict, model

def reference_model(z, t, paramdict):
    Cm_a  = paramdict['Astrocyte']['Cm_a']
    g_kir = paramdict['Astrocyte']['g_kir']
    g_k_a = paramdict['Astrocyte']['g_k_a']
    gl_a = paramdict['Astrocyte']['gl_a']
    w_a = paramdict['Astrocyte']['w_a']
    K_a0 = paramdict['Astrocyte']['K_a0']
    Sig_a = paramdict['Astrocyte']['Sig_a']
    gama_t = paramdict['Astrocyte']['gama_t']
    gama_s = paramdict['Astrocyte']['gama_s']
    Z_th = paramdict['Astrocyte']['Z_th']
    Z_s = paramdict['Astrocyte']['Z_s']
    Va_s = paramdict['Astrocyte']['Va_s']
    Va_l = paramdict['Astrocyte']['Va_l']
    P_k = paramdict['Astrocyte']['P_k']
    d_gap = paramdict['Astrocyte']['d_gap']
    F = paramdict['Astrocyte']['F']
    K_o0 = paramdict['external']['K_o0']
    w_o = paramdict['external']['w_o']
    epsilon = paramdict['external']['epsilon']
    idx = np.where(paramdict['external']['K_bath']['time'] <= t)[0][-1]
    K_bath = paramdict['external']['K_bath']['value'][idx]
    switching_function = paramdict['Astrocyte'].get('switching_function', 'sigmoid')
    if 'epsilon_middle' in paramdict['external'] and idx == 1:
        epsilon = epsilon * paramdict['external']['epsilon_middle']
    if 'w_o_middle' in paramdict['external'] and idx == 1:
        w_o = w_o * paramdict['external']['w_o_middle']
    Va, DK_a_t, K_a_s, Kg = z
    DK_a = DK_a_t + K_a_s
    K_a = K_a0 + DK_a
    K_o = K_o0 - (w_a / w_o) * DK_a_t + Kg
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
    if switching_function == 'sigmoid':
        Th_s = DK_a / (1 + np.exp((Z_th - DK_a_t) * Z_s))
    elif switching_function == 'tanh':
        Th_s = DK_a * (0.5 * (1 + np.tanh((DK_a_t - Z_th) * Z_s)))
    elif switching_function == 'hill':
        n = paramdict['Astrocyte'].get('hill_coefficient', 2)
        K_d = paramdict['Astrocyte'].get('K_d', 1)
        Th_s = DK_a * ((DK_a_t ** n) / (K_d ** n + DK_a_t ** n))
    else:
        raise ValueError
    return np.asarray([
        (-1.0 / Cm_a) * (I_Kir + I_k_a + I_l_a + I_kgap),
        -(gama_t * Sig_a / (w_a * F)) * (I_Kir + I_k_a),
        -Th_s * (gama_s * Sig_a / (w_a * F)) * I_kgap,
        epsilon * (K_bath - K_o),
    ], dtype=float)

def test_expected_model_alignment() -> None:
    z = np.array([-80.0, 0.5, 0.2, 0.1], dtype=float)
    flat = {'gki': 90.0, 'pk': 3e-4, 'd': 0.05, 'gt': 2.0, 'gs': 10.0, 'zth': 70.0, 'zs': 2.5, 'eps': 0.002, 'eps_middle': 1.0, 'wo': 1400.0, 'wo_middle': 1.0, 'ca': 500.0, 'gl_a': 5.0, 'Va_l': -70.0, 'Va_s': -92.0, 'switching_function': 'sigmoid', 'w_a': 2000.0}
    pdict = build_paramdict('CONTROL', 75, flat)
    for t in [0.0, 11173.0, 12000.0, 21140.0]:
        assert np.allclose(model(z, t, pdict), reference_model(z, t, pdict), atol=1e-12, rtol=0.0)
