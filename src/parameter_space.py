"""Central effective-parameter coordinate transforms.

The raw model parameters contain structurally coupled quantities.  This module
provides one source of truth for the effective coordinates used by post-fit
summaries, identifiability diagnostics, and Step 04 fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .astro_model import build_paramdict, normalize_flat_params
from .contracts import canonical_condition, protocol_condition

EFFECTIVE_COORDINATES = ("P_gap_eff", "gamma_t_eff", "gamma_s_eff", "volume_ratio_wa_wo")
SIG_A_DEFAULT = 1600.0
F_DEFAULT = 96485.0
POSITIVE_COORDINATES = set(EFFECTIVE_COORDINATES) | {
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
}
BASE_CONDITION_DEFAULTS: dict[str, dict[str, Any]] = {
    "CONTROL": dict(ca=400.0, gl_a=0.01, Va_l=-70.0, Va_s=-90.0, switching_function="sigmoid", w_a=2000.0, eps_middle=1.0, wo_middle=1.0, g_k_a=0.0),
    "MFA": dict(ca=400.0, gl_a=0.01, Va_l=-70.0, Va_s=-90.0, switching_function="tanh", w_a=2000.0, eps_middle=1.0, wo_middle=1.0, g_k_a=0.0),
    "MFA_BA": dict(ca=400.0, gl_a=0.01, Va_l=-70.0, Va_s=-90.0, switching_function="tanh", w_a=2000.0, eps_middle=1.0, wo_middle=1.0, g_k_a=0.0),
}


@dataclass(frozen=True)
class EffectiveParams:
    P_gap_eff: float
    gamma_t_eff: float
    gamma_s_eff: float
    volume_ratio_wa_wo: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def effective_from_flat(params: Mapping[str, Any], condition: str = "CONTROL", current_na: int = 100) -> EffectiveParams:
    """Compute effective coordinates from a flat parameter mapping."""

    paramdict = build_paramdict(protocol_condition(condition), current_na, params)
    astro = paramdict["Astrocyte"]
    external = paramdict["external"]
    w_a = float(astro["w_a"])
    sig_a = float(astro["Sig_a"])
    f_const = float(astro["F"])
    return EffectiveParams(
        P_gap_eff=float(astro["d_gap"] * astro["P_k"]),
        gamma_t_eff=float(astro["gama_t"] * sig_a / (w_a * f_const)),
        gamma_s_eff=float(astro["gama_s"] * sig_a / (w_a * f_const)),
        volume_ratio_wa_wo=float(w_a / float(external["w_o"])),
    )


def _eff_mapping(eff: EffectiveParams | Mapping[str, Any]) -> Mapping[str, Any]:
    return eff.as_dict() if isinstance(eff, EffectiveParams) else eff


def flat_from_effective(condition: str, eff: EffectiveParams | Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create a flat parameter mapping whose raw values realize ``eff``."""

    cond = canonical_condition(condition)
    values = _eff_mapping(eff)
    out = dict(BASE_CONDITION_DEFAULTS[cond])
    if extra:
        out.update(extra)
    w_a = float(out.get("w_a", 2000.0))
    sig_a = SIG_A_DEFAULT
    f_const = F_DEFAULT
    d = float(out.get("d", 1.0))
    out["d"] = d
    out["pk"] = float(values["P_gap_eff"]) / max(abs(d), 1e-12)
    out["gt"] = float(values["gamma_t_eff"]) * w_a * f_const / sig_a
    out["gs"] = float(values["gamma_s_eff"]) * w_a * f_const / sig_a
    volume_ratio = float(values["volume_ratio_wa_wo"])
    if volume_ratio <= 0:
        raise ValueError("volume_ratio_wa_wo must remain positive")
    out["wo"] = w_a / volume_ratio
    return out


def coordinate_value(params: Mapping[str, Any], coordinate: str, condition: str = "CONTROL", current_na: int = 100) -> float:
    """Read either an effective coordinate or a raw flat parameter value."""

    if coordinate in EFFECTIVE_COORDINATES:
        return float(effective_from_flat(params, condition=condition, current_na=current_na).as_dict()[coordinate])
    normalized = normalize_flat_params(params)
    if coordinate not in normalized:
        raise KeyError(coordinate)
    return float(normalized[coordinate])


def set_coordinate(params: Mapping[str, Any], coordinate: str, value: float) -> dict[str, Any]:
    """Return a copy of ``params`` with one raw/effective coordinate changed."""

    out = normalize_flat_params(params)
    val = float(value)
    if coordinate in POSITIVE_COORDINATES and val <= 0:
        raise ValueError(f"{coordinate} must remain positive")
    w_a = float(out.get("w_a", 2000.0))
    if coordinate == "P_gap_eff":
        d = float(out.get("d", 1.0))
        out["pk"] = val / max(abs(d), 1e-12)
    elif coordinate == "gamma_t_eff":
        current = effective_from_flat(out).gamma_t_eff
        if current > 0 and "gt" in out:
            out["gt"] = float(out["gt"]) * val / max(current, 1e-30)
        else:
            out["gt"] = val * w_a * F_DEFAULT / SIG_A_DEFAULT
    elif coordinate == "gamma_s_eff":
        current = effective_from_flat(out).gamma_s_eff
        if current > 0 and "gs" in out:
            out["gs"] = float(out["gs"]) * val / max(current, 1e-30)
        else:
            out["gs"] = val * w_a * F_DEFAULT / SIG_A_DEFAULT
    elif coordinate == "volume_ratio_wa_wo":
        out["wo"] = w_a / max(val, 1e-12)
    else:
        out[coordinate] = val
    return out
