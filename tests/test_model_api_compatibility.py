import numpy as np

from src.astro_model import (
    VALID_CURRENTS,
    build_paramdict,
    compute_rhs_and_currents,
    model,
    model_alignment_probe,
    simulate_odeint,
    simulate_voltage_trace,
)


def test_model_public_entry_points_are_consistent():
    params = {
        "gki": 10.0,
        "pk": 1e-4,
        "d": 2.0,
        "gt": 5.0,
        "gs": 7.0,
        "wo": 1500.0,
        "eps": 1e-3,
        "gl_a": 0.01,
        "zth": 0.2,
        "zs": 0.05,
    }
    z = np.array([-84.0, 0.1, 0.05, 0.2])
    pdict = build_paramdict("CONTROL", 100, params)
    assert np.allclose(
        model(z, 12000.0, pdict),
        compute_rhs_and_currents(z, 12000.0, pdict, return_currents=True)["dzdt"],
    )
    probe = model_alignment_probe(params, "CONTROL", 100, z, [0.0, 12000.0, 32000.0])
    assert probe["status"] == "exact_within_float_tolerance"


def test_simulate_voltage_trace_supports_both_call_orders():
    params = {
        "gki": 10.0,
        "pk": 1e-4,
        "d": 2.0,
        "gt": 5.0,
        "gs": 7.0,
        "wo": 1500.0,
        "eps": 1e-3,
        "gl_a": 0.01,
        "zth": 0.2,
        "zs": 0.05,
    }
    t = np.linspace(0.0, 1000.0, 20)
    a = simulate_voltage_trace(params, "CONTROL", 100, time_ms=t)
    b = simulate_voltage_trace("CONTROL", 100, params, time_ms=t)
    assert a.shape == t.shape
    assert np.allclose(a, b)
    assert set(VALID_CURRENTS) == {50, 75, 100, 125, 150, 175}


def test_simulate_odeint_honors_protocol_timing_override():
    params = {
        "gki": 10.0,
        "pk": 1e-4,
        "d": 2.0,
        "gt": 5.0,
        "gs": 7.0,
        "wo": 1500.0,
        "eps": 1e-3,
        "gl_a": 0.01,
        "zth": 0.2,
        "zs": 0.05,
    }
    sim = simulate_odeint(
        params,
        {
            "experiment_type": "CONTROL",
            "current_na": 100,
            "t_eval_ms": np.linspace(0.0, 40_000.0, 5),
            "stim_onset_ms": 21_100.0,
            "stim_offset_ms": 41_100.0,
        },
        t_eval_ms=np.linspace(0.0, 40_000.0, 5),
    )
    assert np.allclose(sim["params"]["external"]["K_bath"]["time"], [0.0, 21_100.0, 41_100.0])
