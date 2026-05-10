from __future__ import annotations

import numpy as np
from src.astro_model import _switching_gate
from src.step07_assumption_sensitivity import (
    GATING_FAMILIES,
    Step07Config,
    _safe_corr,
    _scaled_rmse,
    build_gating_family_comparison,
    build_proxy_validity,
    load_step07_inputs,
)


def test_step07_inputs_preserve_identity_and_mechanism_labels(project_root):
    candidates = load_step07_inputs(project_root, Step07Config(max_candidates=2, write_outputs=False))
    required = {"file_id", "region", "condition", "candidate_id", "mechanism_cluster", "dominant_mechanism"}
    assert required.issubset(candidates.columns)
    assert candidates["region"].notna().all()


def test_step07_gating_families_have_explicit_status_and_contract(project_root):
    candidates = load_step07_inputs(project_root, Step07Config(max_candidates=1, write_outputs=False))
    cfg = Step07Config(max_candidates=1, time_points=25, write_outputs=False)
    gating = build_gating_family_comparison(candidates, cfg)
    assert set(GATING_FAMILIES).issubset(set(gating["gating_family"]))
    assert gating["identical_contract_id"].eq("step07_same_candidates_currents_timegrid_loss_v1").all()
    assert gating["simulation_status"].isin({"ok", "failed"}).all()
    assert gating.loc[gating["simulation_status"].eq("failed"), "failure_reason"].astype(str).str.len().gt(0).all()


def test_step07_proxy_metrics_are_auditable(project_root):
    candidates = load_step07_inputs(project_root, Step07Config(max_candidates=1, write_outputs=False))
    proxy = build_proxy_validity(candidates, Step07Config(max_candidates=1, time_points=25, write_outputs=False))
    assert {"pearson_r", "spearman_r", "scaled_rmse", "best_lag_samples", "proxy_validity_status"}.issubset(proxy.columns)
    ok = proxy[proxy["simulation_status"].eq("ok")]
    assert np.isfinite(ok[["pearson_r", "spearman_r", "scaled_rmse"]].to_numpy()).all()
    assert proxy["explicit_ecs_variant_required"].isin({True, False}).all()


def test_safe_corr_spearman_optimization():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    assert np.isclose(_safe_corr(x, y, method="spearman"), 1.0)

    z = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert np.isclose(_safe_corr(x, z, method="spearman"), -1.0)

    zeros = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.isnan(_safe_corr(x, zeros, method="spearman"))

    x_nan = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    y_nan = np.array([2.0, 4.0, np.nan, 8.0, 10.0])
    assert np.isclose(_safe_corr(x_nan, y_nan, method="spearman"), 1.0)


def test_switching_gate_double_sigmoid_bounds():
    astro_params = {
        "switching_function": "double_sigmoid",
        "Z_th": 1.0,
        "Z_s": 10.0,
        "Z_upper_delta": 4.0,
    }

    dk_a = 1.0
    assert _switching_gate(dk_a, 0.0, astro_params) < 0.1

    middle_activation = _switching_gate(dk_a, 3.0, astro_params)
    assert middle_activation > 0.9

    assert _switching_gate(dk_a, 10.0, astro_params) < 0.1


def test_astro_model_double_sigmoid_bounds():
    astro_params = {
        "switching_function": "double_sigmoid",
        "Z_th": 1.0,
        "Z_s": 0.0,
        "Z_upper_delta": 2.0,
    }

    res_low = _switching_gate(10.0, 0.0, astro_params)
    assert np.isfinite(res_low)

    res_mid = _switching_gate(10.0, 2.0, astro_params)
    assert np.isfinite(res_mid)

    res_high = _switching_gate(10.0, 5.0, astro_params)
    assert np.isfinite(res_high)


def test_step07_proxy_metrics_nan_handling():
    flat_array = np.array([1.0, 1.0, 1.0])
    target_array = np.array([2.0, 3.0, 4.0])

    corr = _safe_corr(flat_array, target_array)
    rmse = _scaled_rmse(flat_array, target_array)

    assert np.isnan(corr)
    assert np.isnan(rmse)
