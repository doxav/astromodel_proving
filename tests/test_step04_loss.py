from __future__ import annotations

import json

import numpy as np
import pytest

from src.step04_loss import (
    Step04LossConfig,
    TraceLossConfig,
    compute_loss,
    compute_trace_objective,
    config_hash,
    Step04OptimizerConfig,
    feature_columns_for_loss,
    write_optimization_config,
)


def test_compute_loss_l2_l1():
    z = np.array([0.0, 1.0, -2.0, 3.0])
    target = np.array([1.0, 0.0, -1.0, 1.0])
    residual = z - target

    assert compute_loss(z, target, "L2") == pytest.approx(np.sum(residual**2))
    assert compute_loss(z, target, "L1") == pytest.approx(np.sum(np.abs(residual)))


def test_compute_loss_huber():
    z = np.array([0.0, 0.5, 2.0, -3.0])
    target = np.zeros_like(z)
    delta = 1.0
    abs_resid = np.abs(z - target)
    expected = np.where(abs_resid <= delta, 0.5 * abs_resid**2, delta * (abs_resid - 0.5 * delta)).sum()

    assert compute_loss(z, target, "HUBER", delta_huber=delta) == pytest.approx(expected)


def test_compute_loss_log_cosh_is_finite_for_large_residuals():
    z = np.array([1000.0, -1000.0])
    target = np.zeros_like(z)

    loss = compute_loss(z, target, "LOG_COSH")

    assert np.isfinite(loss)
    assert loss > 0.0


def test_compute_loss_combined_matches_old_definition():
    z = np.array([0.0, 1.0, 3.0, 6.0])
    target = np.zeros_like(z)
    expected_l2 = np.sum((z - target) ** 2)
    expected_grad = 20.0 * np.sum(np.abs(np.gradient(z) - np.gradient(target)))

    assert compute_loss(z, target, "COMBINED") == pytest.approx(expected_l2 + expected_grad)


def test_compute_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        compute_loss(np.array([1.0, 2.0]), np.array([1.0]), "L2")


def test_compute_loss_combined_singleton_does_not_call_gradient():
    assert compute_loss(np.array([2.0]), np.array([1.0]), "COMBINED") == pytest.approx(1.0)


def test_feature_columns_for_loss_validates_mode():
    all_columns = feature_columns_for_loss("all")
    primary_columns = feature_columns_for_loss("primary_no_redundant")

    assert all_columns
    assert primary_columns
    assert set(primary_columns).issubset(set(all_columns))
    assert "stim_end_depolarization_mV" in primary_columns
    assert "peak_depolarization_mV" not in primary_columns
    assert "rise_tau_s" in primary_columns
    assert "decay_tau_s" in primary_columns
    assert "plateau_slope_mV_per_s" not in primary_columns
    with pytest.raises(ValueError, match="feature set"):
        feature_columns_for_loss("not_a_feature_set")


def test_optimizer_config_maps_default_multi_sampler_to_nsga2():
    cfg = Step04OptimizerConfig(backend="optuna_multi")
    assert cfg.optuna_sampler == "nsga2"


def test_optimizer_backend_accepts_user_facing_optuna_alias():
    cfg = Step04OptimizerConfig(backend="optuna")
    assert cfg.backend == "optuna_scalar"


def test_optimizer_config_rejects_invalid_scipy_loss():
    with pytest.raises(ValueError, match="scipy_loss"):
        Step04OptimizerConfig(scipy_loss="not-a-loss")


def test_loss_config_rejects_negative_or_non_finite_weights():
    with pytest.raises(ValueError, match="trace_weight"):
        Step04LossConfig(trace_weight=-1.0)
    with pytest.raises(ValueError, match="feature_weight"):
        Step04LossConfig(feature_weight=float("nan"))
    with pytest.raises(ValueError, match="binary_weight"):
        Step04LossConfig(binary_weight=float("inf"))


def test_optimizer_config_rejects_invalid_optuna_trial_controls():
    with pytest.raises(ValueError, match="optuna_n_trials"):
        Step04OptimizerConfig(backend="optuna_scalar", optuna_n_trials=0)
    with pytest.raises(ValueError, match="optuna_timeout_s"):
        Step04OptimizerConfig(backend="optuna_scalar", optuna_timeout_s=0)


def test_config_hash_rejects_non_finite_json_payloads():
    with pytest.raises(ValueError):
        config_hash({"bad": float("nan")})



def test_write_optimization_config_uses_strict_json(tmp_path):
    loss_config = Step04LossConfig()
    optimizer_config = Step04OptimizerConfig()

    payload = write_optimization_config(tmp_path, loss_config, optimizer_config)
    raw = (tmp_path / "optimization_config.json").read_text(encoding="utf-8")

    assert json.loads(raw)["optimization_config_hash"] == payload["optimization_config_hash"]
    assert "NaN" not in raw
    assert "Infinity" not in raw


def test_trace_and_optimizer_numeric_config_rejects_nan():
    with pytest.raises(ValueError, match="delta_huber"):
        TraceLossConfig(delta_huber=float("nan"))
    with pytest.raises(ValueError, match="gradient_loss_weight"):
        TraceLossConfig(gradient_loss_weight=float("nan"))
    with pytest.raises(ValueError, match="scipy_f_scale"):
        Step04OptimizerConfig(scipy_f_scale=float("nan"))


def test_compute_trace_objective_honors_mean_reduction():
    z = np.array([1.0, 2.0, 4.0])
    target = np.zeros_like(z)
    summed = compute_trace_objective(z, target, TraceLossConfig(loss_type="L2", reduction="sum"))
    meaned = compute_trace_objective(z, target, TraceLossConfig(loss_type="L2", reduction="mean"))
    assert meaned == pytest.approx(summed / z.size)


def test_config_hash_is_stable_and_sensitive():
    cfg = Step04LossConfig()
    same_cfg = Step04LossConfig()
    changed_cfg = Step04LossConfig(trace=TraceLossConfig(loss_type="L2"))

    assert config_hash(cfg) == config_hash(same_cfg)
    assert config_hash(cfg) != config_hash(changed_cfg)
