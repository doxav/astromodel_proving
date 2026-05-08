from __future__ import annotations

import numpy as np


def summarize_hidden_outputs(t_ms, currents, derived, effective_params):
    t_s = np.asarray(t_ms, dtype=float) / 1000.0
    def integral(x):
        return float(np.trapezoid(np.abs(np.asarray(x, dtype=float)), t_s))
    I_Kir = np.asarray(currents["I_Kir"], dtype=float)
    I_kgap = np.asarray(currents["I_kgap"], dtype=float)
    I_leak = np.asarray(currents["I_leak"], dtype=float)
    K_o = np.asarray(derived["K_o"], dtype=float)
    gap_int = integral(I_kgap)
    kir_int = integral(I_Kir)
    leak_int = integral(I_leak)
    total = gap_int + kir_int + leak_int + 1e-12
    ratio = gap_int / max(kir_int, 1e-12)
    if ratio > 2.0:
        classification = "gap_dominant"
    elif ratio < 0.5:
        classification = "kir_dominant"
    else:
        classification = "mixed"
    return {
        "I_Kir_integral": kir_int,
        "I_kgap_integral": gap_int,
        "I_leak_integral": leak_int,
        "I_Kir_peak_abs": float(np.max(np.abs(I_Kir))),
        "I_kgap_peak_abs": float(np.max(np.abs(I_kgap))),
        "gap_to_kir_integral_ratio": ratio,
        "gap_fraction": gap_int / total,
        "kir_fraction": kir_int / total,
        "leak_fraction": leak_int / total,
        "gap_kir_classification": classification,
        "K_o_baseline": float(K_o[0]),
        "K_o_peak": float(np.max(K_o)),
        "K_o_final": float(K_o[-1]),
        "K_o_delta_peak": float(np.max(K_o) - K_o[0]),
        "K_o_recovery_error": float(abs(K_o[-1] - K_o[0])),
        **{k: float(v) for k, v in effective_params.items()},
    }
