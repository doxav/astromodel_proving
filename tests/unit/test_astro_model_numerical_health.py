from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from src import astro_model, step04_cell_fits
from src.step04_cell_fits import Step04Config, SweepTrace


def _warning_odeint(func: Any, z0: np.ndarray, t: np.ndarray, args: tuple[Any, ...] = ()) -> np.ndarray:
    """Return finite states while emitting a scipy odeint warning."""

    warnings.warn("synthetic odeint warning", astro_model.ODEintWarning, stacklevel=2)
    return np.repeat(np.asarray(z0, dtype=float)[None, :], len(t), axis=0)


def test_simulate_odeint_records_solver_warnings_without_failing_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(astro_model, "odeint", _warning_odeint)
    params = astro_model.build_paramdict("CONTROL", 100, {})

    sim = astro_model.simulate_odeint(
        params,
        {"experiment_type": "CONTROL", "current_na": 100, "t_eval_ms": [0.0, 1.0, 2.0]},
    )

    assert sim["numerical_health"]["status"] == "warning"
    assert sim["numerical_health"]["odeint_warning_count"] == 1
    assert sim["numerical_health"]["odeint_warning_messages"] == ["synthetic odeint warning"]
    assert np.isfinite(sim["Vm"]).all()


def test_simulate_odeint_can_treat_solver_warnings_as_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(astro_model, "odeint", _warning_odeint)
    params = astro_model.build_paramdict("CONTROL", 100, {})

    with pytest.raises(astro_model.ODESolverWarningError, match="ODE solver warning"):
        astro_model.simulate_odeint(
            params,
            {"experiment_type": "CONTROL", "current_na": 100, "t_eval_ms": [0.0, 1.0, 2.0]},
            fail_on_warning=True,
        )


def test_step04_simulation_uses_strict_solver_warning_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, Any] = {}

    def fake_simulate_odeint(
        params: Mapping[str, Any],
        protocol: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        captured_kwargs["protocol"] = dict(protocol)
        t_eval_ms = np.asarray(kwargs["t_eval_ms"], dtype=float)
        return {"Vm": np.linspace(-80.0, -75.0, len(t_eval_ms), dtype=float)}

    def fake_extract_features_from_trace(
        time_s: np.ndarray,
        vm_mV: np.ndarray,
        *,
        onset_s: float | None = None,
        offset_s: float | None = None,
    ) -> dict[str, float]:
        return {"stim_onset_s": float(onset_s), "stim_offset_s": float(offset_s)}

    monkeypatch.setattr(step04_cell_fits, "simulate_odeint", fake_simulate_odeint)
    monkeypatch.setattr(step04_cell_fits, "extract_features_from_trace", fake_extract_features_from_trace)
    time_ms = np.linspace(0.0, 50_000.0, 40)
    sweep_trace = SweepTrace(
        file_id="synthetic_cell",
        region="VH",
        condition="MFA",
        sweep=1,
        current_na=50,
        time_ms_fit=time_ms,
        vm_fit=np.linspace(-80.0, -74.0, len(time_ms), dtype=float),
        time_s_full=time_ms / 1000.0,
        vm_full=np.linspace(-80.0, -74.0, len(time_ms), dtype=float),
        stim_onset_s=10.0,
        stim_offset_s=30.0,
        step_source="synthetic",
    )

    _, _, onset_s = step04_cell_fits._simulate_sweep({}, sweep_trace)

    assert captured_kwargs["fail_on_warning"] is True
    assert captured_kwargs["protocol"]["stim_onset_ms"] == pytest.approx(10_000.0)
    assert captured_kwargs["protocol"]["stim_offset_ms"] == pytest.approx(30_000.0)
    assert onset_s == pytest.approx(10.0)


def test_candidate_overlay_exposes_baseline_aligned_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    time_ms = np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0])
    observed_vm = np.array([-60.0, -60.0, -60.0, -60.0, -60.0, -55.0, -55.0])
    predicted_vm = np.array([-90.0, -90.0, -90.0, -90.0, -90.0, -85.0, -85.0])
    sweep_trace = SweepTrace(
        file_id="synthetic_cell",
        region="VH",
        condition="CONTROL",
        sweep=1,
        current_na=50,
        time_ms_fit=time_ms,
        vm_fit=observed_vm,
        time_s_full=time_ms / 1000.0,
        vm_full=observed_vm,
        stim_onset_s=5.0,
        stim_offset_s=6.0,
        step_source="synthetic",
    )

    def fake_simulate_odeint(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        return {"Vm": predicted_vm}

    monkeypatch.setattr(step04_cell_fits, "reconstruct_candidate_params", lambda candidate_row: {})
    monkeypatch.setattr(step04_cell_fits, "simulate_odeint", fake_simulate_odeint)

    overlay = step04_cell_fits.build_candidate_overlay_frame(
        {"file_id": "synthetic_cell", "candidate_id": "synthetic_candidate"},
        {"synthetic_cell": {1: sweep_trace}},
    )

    assert overlay["vm_observed_baseline_mV"].iloc[0] == pytest.approx(-60.0)
    assert overlay["vm_predicted_baseline_mV"].iloc[0] == pytest.approx(-90.0)
    assert overlay["vm_baseline_delta_pred_minus_obs_mV"].iloc[0] == pytest.approx(-30.0)
    assert overlay.loc[overlay["time_ms"] == 5000.0, "vm_observed_centered_mV"].iloc[0] == pytest.approx(5.0)
    assert overlay.loc[overlay["time_ms"] == 5000.0, "vm_predicted_centered_mV"].iloc[0] == pytest.approx(5.0)
    assert overlay.loc[overlay["time_ms"] == 5000.0, "vm_predicted_baseline_aligned_mV"].iloc[0] == pytest.approx(-55.0)


def test_step04_residual_vector_keeps_fixed_length_when_simulation_fails(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
) -> None:
    time_ms = np.linspace(0.0, 50_000.0, 40)
    sweep_trace = SweepTrace(
        file_id="synthetic_cell",
        region="VH",
        condition="MFA",
        sweep=1,
        current_na=50,
        time_ms_fit=time_ms,
        vm_fit=np.linspace(-80.0, -74.0, len(time_ms), dtype=float),
        time_s_full=time_ms / 1000.0,
        vm_full=np.linspace(-80.0, -74.0, len(time_ms), dtype=float),
    )
    cfg = Step04Config(project_root=project_root)
    x = np.zeros(9, dtype=float)

    def successful_simulation(
        params: Mapping[str, Any],
        trace: SweepTrace,
    ) -> tuple[np.ndarray, dict[str, float], float]:
        return trace.vm_fit.copy(), {"stim_onset_s": 10.0}, 10.0

    monkeypatch.setattr(step04_cell_fits, "_simulate_sweep", successful_simulation)
    monkeypatch.setattr(step04_cell_fits, "_feature_residuals", lambda *args, **kwargs: np.ones(4, dtype=float))
    monkeypatch.setattr(step04_cell_fits, "_binary_residuals", lambda *args, **kwargs: np.ones(2, dtype=float))
    success_residual = step04_cell_fits._residual_vector(
        x,
        "MFA",
        [1],
        {1: sweep_trace},
        {1: {}},
        pd.DataFrame(),
        cfg,
    )

    def failed_simulation(params: Mapping[str, Any], trace: SweepTrace) -> tuple[np.ndarray, dict[str, float], float]:
        raise astro_model.ODESolverWarningError("synthetic failure")

    monkeypatch.setattr(step04_cell_fits, "_simulate_sweep", failed_simulation)
    failed_residual = step04_cell_fits._residual_vector(
        x,
        "MFA",
        [1],
        {1: sweep_trace},
        {1: {}},
        pd.DataFrame(),
        cfg,
    )

    assert failed_residual.shape == success_residual.shape
    assert success_residual[-1] == pytest.approx(0.0)
    assert failed_residual[-1] > 0.0
