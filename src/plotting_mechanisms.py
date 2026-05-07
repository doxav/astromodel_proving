"""Mechanistic plotting helpers for reviewer-facing figures.

Source -> target merge rationale
--------------------------------
Astrosim's ``smooth_changes_degeneracy_per_kbath`` and
``plot_combined_degeneracy`` supplied the useful plotting workflow:

    group simulations -> choose same-function/different-mechanism representatives
    -> plot voltage and current panels.

This module rewrites that idea for the astrocyte K-buffering paper: accepted fits,
astrocyte hidden currents, effective parameters, K_o traces, and proxy validity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_sim_for_trial(simulations: Mapping[Any, Mapping[str, Any]], trial_number: Any) -> Mapping[str, Any]:
    if trial_number in simulations:
        return simulations[trial_number]
    key = str(trial_number)
    if key in simulations:
        return simulations[key]
    try:
        key_int = int(trial_number)
        if key_int in simulations:
            return simulations[key_int]
    except Exception:
        pass
    raise KeyError(f"No simulation found for trial_number={trial_number!r}")


def _arr(sim: Mapping[str, Any], container: str, key: str) -> np.ndarray:
    return np.asarray(sim.get(container, {}).get(key), dtype=float)


def plot_compensation_trajectory(
    reps: pd.DataFrame,
    simulations: Mapping[Any, Mapping[str, Any]],
    x: str = "P_gap_eff",
    y: str = "gamma_t_eff",
    color: str = "gap_to_kir_integral_ratio",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a Marder-style compensation trajectory for selected representatives.

    Source
    ------
    Adapted from astrosim ``smooth_changes_degeneracy_per_kbath`` but replacing
    K_bath/resting-state/neuron-current logic by accepted-fit/effective-parameter
    and astro-current logic.

    Parameters
    ----------
    reps:
        Output from ``select_mechanistically_diverse_representatives``.  Must
        contain ``trial_number`` and the columns named by ``x``, ``y`` and
        optionally ``color``.
    simulations:
        Mapping ``trial_number -> simulate_with_hidden_outputs output``.
    x, y, color:
        Columns defining the effective/mechanistic trajectory.
    output_path:
        Optional PNG/PDF/SVG path.

    Returns
    -------
    matplotlib.figure.Figure
    """

    if reps.empty:
        raise ValueError("reps is empty")
    if "trial_number" not in reps.columns:
        raise KeyError("reps must contain a 'trial_number' column")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    ax_scatter, ax_vm, ax_kir, ax_gap, ax_ko, ax_ratio = axes.ravel()

    cvals = reps[color] if color in reps.columns else np.arange(len(reps))
    sc = ax_scatter.scatter(reps[x], reps[y], c=cvals)
    ax_scatter.plot(reps[x], reps[y], linewidth=1, alpha=0.6)
    ax_scatter.set_xlabel(x)
    ax_scatter.set_ylabel(y)
    ax_scatter.set_title("Effective/mechanistic compensation")
    if color in reps.columns:
        fig.colorbar(sc, ax=ax_scatter, label=color)

    ratios = []
    labels = []
    for _, row in reps.iterrows():
        trial = row["trial_number"]
        sim = _get_sim_for_trial(simulations, trial)
        t_s = np.asarray(sim["t_ms"], dtype=float) / 1000.0
        label = f"trial {trial}"
        labels.append(label)
        ax_vm.plot(t_s, np.asarray(sim["Vm"], dtype=float), linewidth=1.2, label=label)
        if "currents" in sim:
            ax_kir.plot(t_s, _arr(sim, "currents", "I_Kir"), linewidth=1.0)
            ax_gap.plot(t_s, _arr(sim, "currents", "I_kgap"), linewidth=1.0)
        if "derived" in sim:
            ax_ko.plot(t_s, _arr(sim, "derived", "K_o"), linewidth=1.0)
        ratios.append(row.get(color, np.nan))

    ax_vm.set_title("Vm representatives")
    ax_vm.set_xlabel("time (s)")
    ax_vm.set_ylabel("Vm (mV)")
    ax_vm.legend(fontsize=8)
    ax_kir.set_title("Kir current")
    ax_kir.set_xlabel("time (s)")
    ax_kir.set_ylabel("I_Kir")
    ax_gap.set_title("Gap-junction flux/current")
    ax_gap.set_xlabel("time (s)")
    ax_gap.set_ylabel("I_kgap")
    ax_ko.set_title("Extracellular K proxy/state")
    ax_ko.set_xlabel("time (s)")
    ax_ko.set_ylabel("K_o (mM)")
    ax_ratio.bar(np.arange(len(ratios)), ratios)
    ax_ratio.set_xticks(np.arange(len(ratios)), labels, rotation=45, ha="right")
    ax_ratio.set_title(color)
    ax_ratio.set_ylabel(color)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_mechanistic_degeneracy_panel(
    accepted_fits: pd.DataFrame,
    simulations: Mapping[Any, Mapping[str, Any]],
    experimental_trace: Optional[Mapping[str, Sequence[float]]] = None,
    group_by: Sequence[str] = ("condition", "current_na"),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Create a composite mechanism panel for accepted-fit ensembles.

    Source
    ------
    Merges the astrosim composite-panel plotting idea with the previous notebook
    Vm overlays.  The panel content is changed to reviewer-relevant astrocyte
    mechanisms.

    Expected inputs
    ---------------
    ``accepted_fits`` should contain ``trial_number`` and preferably effective
    columns such as ``P_gap_eff`` and ``gamma_t_eff``.  ``simulations`` should map
    each trial number to the output of ``simulate_with_hidden_outputs``.

    Figure panels
    -------------
    A. Experimental Vm and accepted simulation band/lines.
    B. Effective parameter scatter.
    C. I_Kir ensemble.
    D. I_kgap ensemble.
    E. K_o ensemble.
    F. Flux partition if available in ``accepted_fits``.
    """

    if accepted_fits.empty:
        raise ValueError("accepted_fits is empty")
    if "trial_number" not in accepted_fits.columns:
        raise KeyError("accepted_fits must contain 'trial_number'")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    ax_vm, ax_scatter, ax_kir, ax_gap, ax_ko, ax_flux = axes.ravel()

    # Vm panel.
    if experimental_trace is not None:
        t_exp = np.asarray(experimental_trace.get("t_s", experimental_trace.get("t", [])), dtype=float)
        y_exp = np.asarray(experimental_trace.get("Vm", experimental_trace.get("vm", [])), dtype=float)
        if len(t_exp) and len(y_exp):
            ax_vm.plot(t_exp, y_exp, linewidth=2.0, label="experiment")

    for _, row in accepted_fits.iterrows():
        trial = row["trial_number"]
        try:
            sim = _get_sim_for_trial(simulations, trial)
        except KeyError:
            continue
        t_s = np.asarray(sim["t_ms"], dtype=float) / 1000.0
        ax_vm.plot(t_s, np.asarray(sim["Vm"], dtype=float), alpha=0.30, linewidth=0.9)
        if "currents" in sim:
            ax_kir.plot(t_s, _arr(sim, "currents", "I_Kir"), alpha=0.25, linewidth=0.8)
            ax_gap.plot(t_s, _arr(sim, "currents", "I_kgap"), alpha=0.25, linewidth=0.8)
        if "derived" in sim:
            ax_ko.plot(t_s, _arr(sim, "derived", "K_o"), alpha=0.25, linewidth=0.8)

    ax_vm.set_title("Vm accepted ensemble")
    ax_vm.set_xlabel("time (s)")
    ax_vm.set_ylabel("Vm (mV)")
    ax_kir.set_title("I_Kir ensemble")
    ax_kir.set_xlabel("time (s)")
    ax_kir.set_ylabel("I_Kir")
    ax_gap.set_title("I_kgap ensemble")
    ax_gap.set_xlabel("time (s)")
    ax_gap.set_ylabel("I_kgap")
    ax_ko.set_title("K_o ensemble")
    ax_ko.set_xlabel("time (s)")
    ax_ko.set_ylabel("K_o (mM)")

    # Effective parameter scatter.
    x = "P_gap_eff" if "P_gap_eff" in accepted_fits.columns else None
    y = "gamma_t_eff" if "gamma_t_eff" in accepted_fits.columns else None
    c = "gap_to_kir_integral_ratio" if "gap_to_kir_integral_ratio" in accepted_fits.columns else None
    if x and y:
        sc = ax_scatter.scatter(accepted_fits[x], accepted_fits[y], c=(accepted_fits[c] if c else None), alpha=0.8)
        ax_scatter.set_xlabel(x)
        ax_scatter.set_ylabel(y)
        if c:
            fig.colorbar(sc, ax=ax_scatter, label=c)
    else:
        ax_scatter.text(0.5, 0.5, "Add P_gap_eff and gamma_t_eff\nto enable mechanism scatter", ha="center", va="center")
    ax_scatter.set_title("Effective parameter space")

    # Flux partition.
    frac_cols = [c for c in ["kir_fraction", "gap_fraction", "leak_fraction"] if c in accepted_fits.columns]
    if frac_cols:
        means = accepted_fits[frac_cols].mean(numeric_only=True)
        ax_flux.bar(np.arange(len(means)), means.values)
        ax_flux.set_xticks(np.arange(len(means)), means.index, rotation=30, ha="right")
        ax_flux.set_ylim(0, max(1.0, float(np.nanmax(means.values)) * 1.1))
        ax_flux.set_ylabel("mean fraction")
    else:
        ax_flux.text(0.5, 0.5, "Run compute_flux_summary()\nto enable flux partition", ha="center", va="center")
    ax_flux.set_title("Flux partition")

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
