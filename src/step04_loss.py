from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .atf_features import FEATURE_COLUMNS

TRACE_LOSS_TYPES = ("L2", "L1", "HUBER", "LOG_COSH", "COMBINED")
TRACE_TARGET_MODES = ("default", "centered", "centered_scaled")
TRACE_SHAPE_OPTUNA_OBJECTIVES = tuple(
    f"trace_shape_{loss_type.lower()}_{target_mode}"
    for target_mode in TRACE_TARGET_MODES
    for loss_type in TRACE_LOSS_TYPES
)
OPTUNA_OBJECTIVES = (
    "metric_scalar",
    "balanced_residual",
    "acceptance_margin",
    "trace_shape",
    "trace_shape_l2",
    "trace_shape_l1",
    "trace_shape_huber",
    "trace_shape_log_cosh",
    *TRACE_SHAPE_OPTUNA_OBJECTIVES,
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all": tuple(FEATURE_COLUMNS),

    # One amplitude feature + robust kinetic/function features.
    "primary": (
        "peak_depolarization_mV",
        "rise_tau_s",
        "undershoot_magnitude_mV",
        "decay_tau_s",
    ),

    # Reviewer-facing default: avoid double-counting peak and stim-end amplitude.
    # Prefer stim_end_depolarization_mV because it is less peak/transient-driven.
    "primary_no_redundant": (
        "stim_end_depolarization_mV",
        "rise_tau_s",
        "undershoot_magnitude_mV",
        "decay_tau_s",
    ),

    "primary_plus_slopes": (
        "stim_end_depolarization_mV",
        "rise_slope_mV_per_s",
        "rise_tau_s",
        "undershoot_magnitude_mV",
        "decay_slope_mV_per_s",
        "decay_tau_s",
        "return_slope_mV_per_s",
    ),
}


@dataclass(frozen=True)
class TraceLossConfig:
    loss_type: str = "COMBINED"
    delta_huber: float = 1.0
    gradient_loss_weight: float = 20.0
    reduction: str = "sum"

    def __post_init__(self) -> None:
        loss_type = str(self.loss_type).upper()
        if loss_type not in TRACE_LOSS_TYPES:
            raise ValueError(f"invalid trace loss type {self.loss_type!r}; expected one of {TRACE_LOSS_TYPES}")
        if self.reduction not in {"sum", "mean"}:
            raise ValueError("TraceLossConfig.reduction must be 'sum' or 'mean'")
        if not np.isfinite(float(self.delta_huber)) or float(self.delta_huber) <= 0:
            raise ValueError("TraceLossConfig.delta_huber must be positive")
        if not np.isfinite(float(self.gradient_loss_weight)) or float(self.gradient_loss_weight) < 0:
            raise ValueError("TraceLossConfig.gradient_loss_weight must be finite and non-negative")
        object.__setattr__(self, "loss_type", loss_type)


@dataclass(frozen=True)
class Step04LossConfig:
    trace: TraceLossConfig = field(default_factory=TraceLossConfig)
    feature_set: str = "primary_no_redundant"

    trace_weight: float = 1.0
    feature_weight: float = 1.0
    binary_weight: float = 1.0
    prior_weight: float = 0.0
    hidden_weight: float = 0.0
    fail_weight: float = 1.0

    multi_objective_names: tuple[str, ...] = ("trace", "feature", "binary")

    def __post_init__(self) -> None:
        feature_columns_for_loss(self.feature_set)
        for name in self.multi_objective_names:
            if name not in {"trace", "feature", "binary", "prior", "hidden", "fail"}:
                raise ValueError(f"invalid objective component name {name!r}")
        for weight_name in (
            "trace_weight",
            "feature_weight",
            "binary_weight",
            "prior_weight",
            "hidden_weight",
            "fail_weight",
        ):
            value = float(getattr(self, weight_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{weight_name} must be finite and non-negative")


@dataclass(frozen=True)
class Step04OptimizerConfig:
    backend: str = "least_squares"  # least_squares | optuna_scalar | optuna_multi | hybrid

    scipy_loss: str = "linear"
    scipy_f_scale: float = 1.0

    optuna_n_trials: int = 100
    optuna_timeout_s: int | None = None
    optuna_sampler: str = "tpe"  # tpe | random | nsga2
    optuna_storage: str | None = None
    optuna_study_name: str | None = None
    optuna_preseed_candidate_csv: str | None = None
    optuna_preseed_candidate_limit: int = 0
    optuna_preseed_only_accepted: bool = True
    optuna_objective: str = "metric_scalar"
    optuna_adaptive_min_accepted_per_cell: int = 0
    optuna_adaptive_trial_step: int = 0
    optuna_adaptive_max_trials: int = 0
    optuna_include_extended_parameters: bool = False
    optuna_extend_parameter_space_when_large: bool = False

    hybrid_scipy_pre_nfev: int = 40
    hybrid_scipy_post_nfev: int = 20
    hybrid_refine_top_k: int = 3
    candidate_top_k: int = 500

    allow_optuna_fallback: bool = False
    run_holdout: bool = True
    holdout_backend: str = "least_squares"

    def __post_init__(self) -> None:
        backend = str(self.backend).lower()
        if backend == "optuna":
            backend = "optuna_scalar"
        if backend in {"scipy_optuna", "scipy_optuna_hybrid"}:
            backend = "hybrid"
        if backend not in {"least_squares", "optuna_scalar", "optuna_multi", "hybrid"}:
            raise ValueError(
                "invalid Step04 optimizer backend; expected least_squares, optuna/optuna_scalar, optuna_multi, or hybrid"
            )
        object.__setattr__(self, "backend", backend)
        if self.holdout_backend != "least_squares":
            raise ValueError("Step 04 holdout_backend currently supports only least_squares")
        if self.optuna_sampler not in {"tpe", "random", "nsga2"}:
            raise ValueError("invalid Optuna sampler; expected tpe, random, or nsga2")
        if self.optuna_objective not in OPTUNA_OBJECTIVES:
            raise ValueError(f"invalid Optuna objective; expected one of {OPTUNA_OBJECTIVES}")
        if self.scipy_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("invalid scipy_loss; expected linear, soft_l1, huber, cauchy, or arctan")
        if not np.isfinite(float(self.scipy_f_scale)) or float(self.scipy_f_scale) <= 0:
            raise ValueError("scipy_f_scale must be positive")
        if int(self.optuna_n_trials) < 1:
            raise ValueError("optuna_n_trials must be >= 1")
        if int(self.optuna_adaptive_min_accepted_per_cell) < 0:
            raise ValueError("optuna_adaptive_min_accepted_per_cell must be >= 0")
        if int(self.optuna_adaptive_trial_step) < 0:
            raise ValueError("optuna_adaptive_trial_step must be >= 0")
        if int(self.optuna_adaptive_max_trials) < 0:
            raise ValueError("optuna_adaptive_max_trials must be >= 0")
        if int(self.optuna_adaptive_max_trials) > 0 and int(self.optuna_adaptive_max_trials) < int(self.optuna_n_trials):
            raise ValueError("optuna_adaptive_max_trials must be >= optuna_n_trials")
        if int(self.optuna_adaptive_max_trials) > 0 and int(self.optuna_adaptive_trial_step) <= 0:
            raise ValueError("optuna_adaptive_trial_step must be > 0 when optuna_adaptive_max_trials is set")
        if self.optuna_timeout_s is not None and float(self.optuna_timeout_s) <= 0:
            raise ValueError("optuna_timeout_s must be positive when provided")
        if int(self.hybrid_scipy_pre_nfev) < 1:
            raise ValueError("hybrid_scipy_pre_nfev must be >= 1")
        if int(self.hybrid_scipy_post_nfev) < 1:
            raise ValueError("hybrid_scipy_post_nfev must be >= 1")
        if int(self.hybrid_refine_top_k) < 1:
            raise ValueError("hybrid_refine_top_k must be >= 1")
        if int(self.candidate_top_k) < 1:
            raise ValueError("candidate_top_k must be >= 1")
        if int(self.optuna_preseed_candidate_limit) < 0:
            raise ValueError("optuna_preseed_candidate_limit must be >= 0")

        # Use the standard multi-objective sampler unless the caller explicitly requests another.
        if self.backend == "optuna_multi" and self.optuna_sampler == "tpe":
            object.__setattr__(self, "optuna_sampler", "nsga2")


def feature_columns_for_loss(feature_set: str) -> tuple[str, ...]:
    try:
        return FEATURE_SETS[str(feature_set)]
    except KeyError as exc:
        raise ValueError(f"invalid Step 04 feature set {feature_set!r}; expected one of {tuple(FEATURE_SETS)}") from exc


def _stable_log_cosh(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0)


def _reduce(values: np.ndarray, reduction: str = "sum") -> float:
    if reduction == "sum":
        return float(np.sum(values))
    if reduction == "mean":
        return float(np.mean(values)) if values.size else 0.0
    raise ValueError("reduction must be 'sum' or 'mean'")


def compute_loss(
    z: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    loss_type: str = "COMBINED",
    delta_huber: float = 1.0,
    gradient_loss_weight: float = 20.0,
    reduction: str = "sum",
) -> float:
    loss_type = str(loss_type).upper()
    if loss_type not in TRACE_LOSS_TYPES:
        raise ValueError(f"invalid trace loss type {loss_type!r}; expected one of {TRACE_LOSS_TYPES}")
    z_arr = np.asarray(z, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if z_arr.shape != target_arr.shape:
        raise ValueError(f"z and target must have matching shapes, got {z_arr.shape} and {target_arr.shape}")
    residual = z_arr - target_arr

    if loss_type == "L2":
        return _reduce(residual**2, reduction)
    if loss_type == "L1":
        return _reduce(np.abs(residual), reduction)
    if loss_type == "HUBER":
        delta = float(delta_huber)
        if delta <= 0:
            raise ValueError("delta_huber must be positive")
        abs_resid = np.abs(residual)
        return _reduce(np.where(abs_resid <= delta, 0.5 * abs_resid**2, delta * (abs_resid - 0.5 * delta)), reduction)
    if loss_type == "LOG_COSH":
        return _reduce(_stable_log_cosh(residual), reduction)

    l2 = _reduce(residual**2, reduction)
    if z_arr.size < 2:
        grad = 0.0
    else:
        grad_values = np.abs(np.gradient(z_arr) - np.gradient(target_arr))
        grad = float(gradient_loss_weight) * _reduce(grad_values, reduction)
    return float(l2 + grad)


def compute_trace_objective(z: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray, trace_config: TraceLossConfig | None = None) -> float:
    cfg = trace_config or TraceLossConfig()
    return compute_loss(z, target, cfg.loss_type, cfg.delta_huber, cfg.gradient_loss_weight, cfg.reduction)


def scalarize_components(components: Mapping[str, Any], loss_config: Step04LossConfig) -> float:
    def finite(name: str) -> float:
        value = float(components.get(name, 0.0))
        return value if np.isfinite(value) else 1e12

    return float(
        loss_config.trace_weight * finite("trace")
        + loss_config.feature_weight * finite("feature")
        + loss_config.binary_weight * finite("binary")
        + loss_config.prior_weight * finite("prior")
        + loss_config.hidden_weight * finite("hidden")
        + loss_config.fail_weight * finite("fail")
    )


def objective_tuple(components: Mapping[str, Any], objective_names: Sequence[str]) -> tuple[float, ...]:
    allowed = {"trace", "feature", "binary", "prior", "hidden", "fail"}
    out: list[float] = []
    for name in objective_names:
        if name not in allowed:
            raise ValueError(f"invalid objective component {name!r}")
        value = float(components.get(name, 0.0))
        out.append(value if np.isfinite(value) else 1e12)
    return tuple(out)


def config_to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return config_to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): config_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [config_to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [config_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def config_hash(*configs: Any) -> str:
    payload = config_to_jsonable(configs[0] if len(configs) == 1 else configs)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def write_optimization_config(output_dir: str | Path, loss_config: Step04LossConfig, optimizer_config: Step04OptimizerConfig) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "loss_config": config_to_jsonable(loss_config),
        "optimizer_config": config_to_jsonable(optimizer_config),
    }
    payload["optimization_config_hash"] = config_hash(payload)
    (out / "optimization_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return payload
