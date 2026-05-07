"""Perturbation sweeps and homeostasis-stability classification.

Source -> target merge rationale
--------------------------------
- Existing ``Filtered_basline_sweep`` already contains ``run_parameter_sweeps``:
  one-at-a-time perturbation around accepted fits.
- Astrosim contributes the idea of worker/caching and transparent resting-state
  pass/fail classification.

This module initializes a protocol-level perturbation layer that operates on the
new ``simulate_with_hidden_outputs`` output and returns flat tables suitable for
reviewer-facing robustness checks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .astro_model import simulate_with_hidden_outputs
from .mechanisms import compute_flux_summary, compute_proxy_validity


def _get_nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = mapping
    for p in path:
        cur = cur[p]
    return cur


def _set_nested(mapping: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    cur: Dict[str, Any] = mapping
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value


def apply_perturbation(params: Mapping[str, Any], context: Mapping[str, Any], perturbation: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply one perturbation to params or protocol context.

    Supported perturbation schema
    -----------------------------
    ``{"param": "eps", "mode": "multiply", "value": 0.5}``
        Perturb a flat parameter.

    ``{"param_path": ["external", "epsilon"], "mode": "multiply", "value": 2}``
        Perturb a nested parameter dictionary.

    ``{"protocol": "stim_duration", "mode": "multiply", "value": 1.5}``
        Placeholder for protocol-level perturbations.  This initializes the API;
        project-specific context builders should implement exact timing changes.
    """

    new_params = deepcopy(dict(params))
    new_context = deepcopy(dict(context))
    mode = str(perturbation.get("mode", "multiply"))
    value = perturbation.get("value", 1.0)

    def transform(old: Any) -> Any:
        old_f = float(old)
        val_f = float(value)
        if mode == "multiply":
            return old_f * val_f
        if mode == "add":
            return old_f + val_f
        if mode == "set":
            return val_f
        raise ValueError("perturbation mode must be 'multiply', 'add', or 'set'")

    if "param" in perturbation:
        key = str(perturbation["param"])
        if key not in new_params:
            raise KeyError(f"Parameter {key!r} not found")
        new_params[key] = transform(new_params[key])
    elif "param_path" in perturbation:
        path = list(perturbation["param_path"])
        _set_nested(new_params, path, transform(_get_nested(new_params, path)))
    elif "protocol" in perturbation:
        # Exact protocol-timing edits are project-specific.  Store the requested
        # perturbation so the context_builder or downstream simulator can handle it.
        new_context.setdefault("protocol_perturbations", []).append(dict(perturbation))
    else:
        raise ValueError("Perturbation must define 'param', 'param_path', or 'protocol'")

    return new_params, new_context


def classify_homeostasis_stability(
    sim: Mapping[str, Any],
    thresholds: Mapping[str, float],
    require: str = "all",
) -> Dict[str, Any]:
    """Classify whether a simulation preserves K/Vm homeostasis.

    Source
    ------
    Adapted from astrosim's resting-state pass/fail pattern, but with astrocyte
    K-homeostasis criteria instead of neuronal final-voltage criteria.

    Threshold keys
    --------------
    ``Vm_final_abs_max_mV``
        Maximum allowed absolute final Vm deviation from first Vm value.
    ``K_o_final_abs_max_mM``
        Maximum allowed final K_o deviation from first K_o value.
    ``K_o_peak_max_mM``
        Maximum allowed K_o peak during the simulation.
    ``recovery_fraction_min``
        Minimum recovery fraction after K_o peak.
    """

    criteria: Dict[str, bool] = {}
    failed: list[str] = []

    finite_ok = np.isfinite(np.asarray(sim.get("Vm", []), dtype=float)).all()
    criteria["finite_ok"] = bool(finite_ok)

    Vm = np.asarray(sim.get("Vm", []), dtype=float)
    if len(Vm) and "Vm_final_abs_max_mV" in thresholds:
        criteria["Vm_final_ok"] = bool(abs(float(Vm[-1] - Vm[0])) <= float(thresholds["Vm_final_abs_max_mV"]))

    K_o = None
    try:
        K_o = np.asarray(sim.get("derived", {}).get("K_o"), dtype=float)
    except Exception:
        K_o = None
    if K_o is not None and len(K_o):
        baseline = float(K_o[0])
        peak = float(np.nanmax(K_o))
        final = float(K_o[-1])
        if "K_o_final_abs_max_mM" in thresholds:
            criteria["K_o_final_ok"] = bool(abs(final - baseline) <= float(thresholds["K_o_final_abs_max_mM"]))
        if "K_o_peak_max_mM" in thresholds:
            criteria["K_o_peak_ok"] = bool(peak <= float(thresholds["K_o_peak_max_mM"]))
        if "recovery_fraction_min" in thresholds:
            denom = max(abs(peak - baseline), 1e-12)
            recovery = 1.0 - abs(final - baseline) / denom
            criteria["recovery_ok"] = bool(recovery >= float(thresholds["recovery_fraction_min"]))

    for k, v in criteria.items():
        if not v:
            failed.append(k)

    if require == "all":
        stable = all(criteria.values()) if criteria else False
    elif require == "any":
        stable = any(criteria.values()) if criteria else False
    elif require.startswith("fraction:"):
        frac = float(require.split(":", 1)[1])
        stable = (sum(criteria.values()) / max(len(criteria), 1)) >= frac
    else:
        raise ValueError("require must be 'all', 'any', or 'fraction:<value>'")

    return {"homeostasis_stable": bool(stable), "criteria": criteria, "failed_criteria": failed}


def run_perturbation_sweep(
    accepted_fits: pd.DataFrame,
    perturbations: Sequence[Mapping[str, Any]],
    context_builder: Callable[[pd.Series, Optional[Mapping[str, Any]]], Mapping[str, Any]],
    solver: str = "odeint",
    n_workers: int = 1,
    cache: bool = False,
    stability_thresholds: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Run perturbation sweeps around accepted fits.

    Parameters
    ----------
    accepted_fits:
        Table of accepted parameter sets.  Parameter columns should use the flat
        names expected by ``build_paramdict``.
    perturbations:
        List of perturbation dictionaries; see :func:`apply_perturbation`.
    context_builder:
        Callable receiving ``(row, perturbation)`` and returning a context dict
        for that row/condition/current.  This keeps project-specific ATF/Optuna
        metadata outside the generic module.
    solver:
        ``odeint`` or ``rk4``.
    n_workers:
        Initialized for API compatibility.  The first implementation runs
        serially for determinism; parallel execution can be added later.
    cache:
        Passed to ``simulate_with_hidden_outputs``.
    stability_thresholds:
        Optional thresholds for :func:`classify_homeostasis_stability`.

    Returns
    -------
    pandas.DataFrame
        One row per baseline fit and perturbation, including simulation status,
        feature outputs, mechanism summaries, proxy validity, and stability flags.
    """

    if n_workers != 1:
        # Keep initialization explicit; parallel simulation should be added only
        # after disk caching/hash validation is stable.
        raise NotImplementedError("Initial run_perturbation_sweep is serial; set n_workers=1")

    records: list[Dict[str, Any]] = []
    for _, row in accepted_fits.iterrows():
        base_params = row.dropna().to_dict()
        trial_number = row.get("trial_number", row.get("baseline_trial_number", None))
        for perturbation in perturbations:
            name = str(perturbation.get("name", perturbation.get("param", perturbation.get("protocol", "perturbation"))))
            context = dict(context_builder(row, perturbation))
            try:
                perturbed_params, perturbed_context = apply_perturbation(base_params, context, perturbation)
                sim = simulate_with_hidden_outputs(perturbed_params, perturbed_context, solver=solver, cache=cache)
                flux = compute_flux_summary(sim)
                proxy = compute_proxy_validity(sim)
                stability = classify_homeostasis_stability(sim, stability_thresholds or {}, require="all") if stability_thresholds else {"homeostasis_stable": np.nan, "criteria": {}, "failed_criteria": []}
                rec: Dict[str, Any] = {
                    "trial_number": trial_number,
                    "perturbation_name": name,
                    "simulation_ok": True,
                    "homeostasis_stable": stability["homeostasis_stable"],
                    "failed_criteria": ";".join(stability.get("failed_criteria", [])),
                }
                rec.update({f"flux_{k}": v for k, v in flux.items() if not isinstance(v, (dict, list))})
                rec.update({f"proxy_{k}": v for k, v in proxy.items()})
                for k, v in sim.get("features", {}).items():
                    if np.isscalar(v):
                        rec[f"feature_{k}"] = v
                records.append(rec)
            except Exception as exc:
                records.append(
                    {
                        "trial_number": trial_number,
                        "perturbation_name": name,
                        "simulation_ok": False,
                        "homeostasis_stable": False,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(records)
