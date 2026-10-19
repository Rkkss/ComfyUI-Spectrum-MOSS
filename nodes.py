# ComfyUI-Spectrum-MOSS
#
# Spectrum-style spectral feature forecasting for native ComfyUI MOSS-Image.
# Based on the method from arXiv:2603.01623 and informed by the GPL-3.0
# Comfyui-Spectrum-Qwen2.1 implementation by awdqwdasdg.
#
# SPDX-License-Identifier: GPL-3.0-only

"""
@author: Rkkss
@title: ComfyUI-Spectrum-MOSS
@nickname: Spectrum MOSS
@description: Spectrum-style hidden-state forecasting acceleration for native ComfyUI MOSS-Image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

import torch

import comfy.model_management as model_management
import comfy.patcher_extension
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.moss_image.model import modulated_rms_norm


LOG_TAG = "Spectrum-MOSS"
WRAPPER_KEY = "spectrum_moss_image"
_PREDICT_CHUNK = 1 << 22

# Validated Adaptive policy constants (MOSS-Image v0.6).
_ADAPTIVE_HISTORY_POINTS = 8
_ADAPTIVE_CHEBYSHEV_DEGREE = 4
_ADAPTIVE_RIDGE_LAMBDA = 0.1
_ADAPTIVE_BLEND_WEIGHT = 0.5
_ADAPTIVE_GUARD_HORIZON = 1
_ADAPTIVE_BASELINE_ALPHA = 0.12
_ADAPTIVE_MEDIUM_RATIO = 1.18
_ADAPTIVE_HIGH_RATIO = 1.35
_ADAPTIVE_SAFE_REFRESHES_TO_EXPAND = 2
_ADAPTIVE_REPAIR_ACTUAL_STEPS = 1
_ADAPTIVE_SAMPLE_TOKENS = 384
_ADAPTIVE_HIGH_COOLDOWN_REFRESHES = 1


def _log_debug(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{LOG_TAG}] {msg}", flush=True)


def _log_warning(msg: str) -> None:
    print(f"[{LOG_TAG}] WARNING: {msg}", flush=True)


@dataclass(frozen=True)
class SpectrumConfig:
    """Runtime configuration shared by the Manual and Adaptive nodes.

    Manual mode exposes the core Spectrum predictor/scheduler controls.
    Adaptive mode uses the validated MOSS-Image v0.6 policy: an early detail
    guard plus a local p95/p99 drift controller that adjusts forecast horizon.
    """

    mode: str = "manual"
    warmup_steps: int = 8
    tail_actual_steps: int = 2
    window_size: float = 2.0
    flex_window: float = 0.75
    max_consecutive_forecasts: int = 2  # Manual mode only.
    history_points: int = 8
    chebyshev_degree: int = 4
    ridge_lambda: float = 0.1
    blend_weight: float = 0.5
    cache_device: str = "main_device"
    force_actual_on_control: bool = True

    # Adaptive scheduling controls. Other adaptive constants are fixed above.
    detail_guard_steps: int = 8
    post_guard_horizon_cap: int = 6
    error_tolerance: float = 1.0

    debug: bool = False

    @property
    def adaptive(self) -> bool:
        return self.mode == "adaptive"

    def validate(self) -> None:
        if self.mode not in {"manual", "adaptive"}:
            raise ValueError("mode must be 'manual' or 'adaptive'")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.tail_actual_steps < 0:
            raise ValueError("tail_actual_steps must be >= 0")
        if self.window_size < 1.0:
            raise ValueError("window_size must be >= 1.0")
        if self.flex_window < 0.0:
            raise ValueError("flex_window must be >= 0.0")
        if self.max_consecutive_forecasts < 0:
            raise ValueError("max_consecutive_forecasts must be >= 0")
        if self.history_points < 2:
            raise ValueError("history_points must be >= 2")
        if self.chebyshev_degree < 1:
            raise ValueError("chebyshev_degree must be >= 1")
        if self.history_points < self.chebyshev_degree + 1:
            raise ValueError(
                "history_points must be at least chebyshev_degree + 1 "
                f"(got history_points={self.history_points}, degree={self.chebyshev_degree})"
            )
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be >= 0")
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        if self.cache_device not in {"main_device", "offload_device", "cpu"}:
            raise ValueError("cache_device must be main_device, offload_device, or cpu")

        if self.adaptive:
            if self.detail_guard_steps < 0:
                raise ValueError("detail_guard_steps must be >= 0")
            if self.post_guard_horizon_cap < 1:
                raise ValueError("post_guard_horizon_cap must be >= 1")
            if self.error_tolerance <= 0.0:
                raise ValueError("error_tolerance must be > 0")

    @property
    def window_growth_cap(self) -> float:
        return max(1.0, self.window_size, float(self.history_points))


def _normalize_step_position(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.0
    coord = 2.0 * float(step_index) / float(total_steps - 1) - 1.0
    return max(-1.0, min(1.0, coord))


def _chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.to(dtype=torch.float32, device="cpu")
    cols = [torch.ones_like(x)]
    if degree >= 1:
        cols.append(x)
    for _ in range(2, degree + 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return torch.stack(cols[: degree + 1], dim=-1)


@dataclass
class _Anchor:
    coord: float
    feature: torch.Tensor


class HistoryWeightChebyshevForecaster:
    """Ridge-regularized Chebyshev fit represented as weights over anchors.

    Instead of materializing a huge coefficient matrix over the hidden-state
    feature dimension, solve the tiny polynomial system on CPU and turn the
    prediction into a weighted sum of the stored real-step hidden states.
    """

    def __init__(
        self,
        degree: int,
        ridge_lambda: float,
        max_history: int,
        blend_weight: float,
    ) -> None:
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = max(2, int(max_history))
        self.blend_weight = float(min(1.0, max(0.0, blend_weight)))
        self._anchors: list[_Anchor] = []
        self._feature_shape: Optional[torch.Size] = None
        self._feature_dtype: Optional[torch.dtype] = None
        self._storage_device: Optional[torch.device] = None
        self._chol: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._anchors.clear()
        self._feature_shape = None
        self._feature_dtype = None
        self._storage_device = None
        self._chol = None

    def release_features(self) -> None:
        self._anchors.clear()
        self._chol = None

    def ready(self) -> bool:
        return self._feature_shape is not None and len(self._anchors) >= self.degree + 1

    @property
    def history_size(self) -> int:
        return len(self._anchors)

    @property
    def feature_shape(self) -> Optional[torch.Size]:
        return self._feature_shape

    def update(self, coord: float, feature: torch.Tensor) -> None:
        if self._feature_shape is None:
            self._feature_shape = torch.Size(feature.shape)
            self._feature_dtype = feature.dtype
            self._storage_device = feature.device
        else:
            if tuple(feature.shape) != tuple(self._feature_shape):
                raise ValueError(
                    f"feature shape changed {tuple(self._feature_shape)} -> {tuple(feature.shape)}"
                )
            if feature.dtype != self._feature_dtype:
                raise ValueError(
                    f"feature dtype changed {self._feature_dtype} -> {feature.dtype}"
                )
            if feature.device != self._storage_device:
                raise ValueError(
                    f"feature device changed {self._storage_device} -> {feature.device}"
                )

        self._anchors.append(_Anchor(float(coord), feature))
        if len(self._anchors) > self.max_history:
            self._anchors.pop(0)
        self._chol = None

    def _design(self) -> torch.Tensor:
        coords = torch.tensor([a.coord for a in self._anchors], dtype=torch.float32)
        return _chebyshev_basis(coords, self.degree)

    def _factorize(self) -> torch.Tensor:
        if self._chol is not None:
            return self._chol
        design = self._design()
        gram = design.T @ design
        p = gram.shape[0]
        eye = torch.eye(p, dtype=torch.float32)
        lhs = gram + self.ridge_lambda * eye
        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            scale = max(float(gram.diagonal().abs().mean().item()), 1.0)
            chol = None
            for m in (1e-8, 1e-7, 1e-6, 1e-5):
                try:
                    chol = torch.linalg.cholesky(lhs + scale * m * eye)
                    break
                except RuntimeError:
                    pass
            if chol is None:
                raise
        self._chol = chol
        return chol

    def _spectral_weights(self, coord: float) -> torch.Tensor:
        design = self._design()  # K x P
        chol = self._factorize()  # P x P
        phi = _chebyshev_basis(torch.tensor([float(coord)]), self.degree)  # 1 x P
        solved = torch.cholesky_solve(design.T, chol)  # P x K
        return (phi @ solved).reshape(-1)

    def _linear_weights(self, coord: float) -> torch.Tensor:
        k = len(self._anchors)
        w = torch.zeros(k, dtype=torch.float32)
        if k <= 1:
            if k == 1:
                w[0] = 1.0
            return w
        prev = self._anchors[-2].coord
        last = self._anchors[-1].coord
        spacing = last - prev
        if abs(spacing) <= 1e-12:
            w[-1] = 1.0
            return w
        ratio = (float(coord) - last) / spacing
        w[-2] = -ratio
        w[-1] = 1.0 + ratio
        return w

    def combined_weights(self, coord: float) -> torch.Tensor:
        if self.blend_weight <= 1e-9:
            return self._linear_weights(coord)
        spectral = self._spectral_weights(coord)
        if self.blend_weight >= 1.0 - 1e-9:
            return spectral
        linear = self._linear_weights(coord)
        return self.blend_weight * spectral + (1.0 - self.blend_weight) * linear

    def sampled_token_error_stats(self, coord: float, actual: torch.Tensor, sample_tokens: int) -> dict[str, float]:
        """Measure spatially local forecast error on sampled image tokens.

        MOSS hidden states are shaped [B, image_tokens, D].  A global scalar L2
        can hide a badly drifting hand/eye token region because that region is a
        tiny fraction of the image.  Here we predict only a deterministic set of
        spatial tokens and compute a relative L2 per token, then report tail
        percentiles (p95/p99).
        """
        if not self.ready():
            raise RuntimeError("forecaster is not ready")
        if actual.ndim != 3:
            raise ValueError(f"expected [B,T,D] feature, got shape={tuple(actual.shape)}")

        weights = self.combined_weights(coord).tolist()
        b, tokens, dim = map(int, actual.shape)
        if tokens <= 0 or dim <= 0:
            return {"median": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        count = min(tokens, max(32, int(sample_tokens)))
        device = actual.device
        if count >= tokens:
            tidx = torch.arange(tokens, device=device, dtype=torch.int64)
        elif count <= 1:
            tidx = torch.zeros(1, device=device, dtype=torch.int64)
        else:
            tidx = torch.arange(count, device=device, dtype=torch.int64)
            tidx = (tidx * (tokens - 1)) // (count - 1)
            tidx.clamp_(0, tokens - 1)

        target = actual.index_select(1, tidx).to(torch.float32)
        pred = torch.zeros_like(target, dtype=torch.float32)
        for weight, anchor in zip(weights, self._anchors):
            feat = anchor.feature
            if feat.ndim != 3 or tuple(feat.shape) != tuple(actual.shape):
                return {"median": float("inf"), "p95": float("inf"), "p99": float("inf"), "mean": float("inf")}
            sampled = feat.to(device=device).index_select(1, tidx).to(torch.float32)
            if weight != 0.0:
                pred.add_(sampled, alpha=float(weight))

        diff_norm = torch.linalg.vector_norm(pred - target, dim=-1)
        target_norm = torch.linalg.vector_norm(target, dim=-1).clamp_min_(1e-6)
        rel = (diff_norm / target_norm).reshape(-1)
        rel = torch.nan_to_num(rel, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        finite = torch.isfinite(rel)
        if not bool(finite.any()):
            return {"median": float("inf"), "p95": float("inf"), "p99": float("inf"), "mean": float("inf")}
        vals = rel[finite].detach().to("cpu", dtype=torch.float32)
        q = torch.quantile(vals, torch.tensor([0.50, 0.95, 0.99], dtype=torch.float32))
        return {
            "median": float(q[0].item()),
            "p95": float(q[1].item()),
            "p99": float(q[2].item()),
            "mean": float(vals.mean().item()),
        }

    def predict(self, coord: float) -> torch.Tensor:
        if not self.ready():
            raise RuntimeError("forecaster is not ready")
        assert self._feature_shape is not None
        assert self._feature_dtype is not None
        assert self._storage_device is not None

        weights = self.combined_weights(coord).tolist()
        out = torch.empty(
            self._feature_shape,
            dtype=self._feature_dtype,
            device=self._storage_device,
        )
        out_flat = out.view(-1)
        flats = [a.feature.contiguous().view(-1) for a in self._anchors]
        numel = out_flat.numel()

        finfo = torch.finfo(self._feature_dtype) if self._feature_dtype.is_floating_point else None
        for start in range(0, numel, _PREDICT_CHUNK):
            stop = min(start + _PREDICT_CHUNK, numel)
            acc = torch.zeros(stop - start, dtype=torch.float32, device=out.device)
            for weight, flat in zip(weights, flats):
                if weight != 0.0:
                    acc.add_(flat[start:stop].to(torch.float32), alpha=float(weight))
            if finfo is not None:
                torch.nan_to_num_(acc, nan=0.0, posinf=float(finfo.max), neginf=float(finfo.min))
                acc.clamp_(min=float(finfo.min), max=float(finfo.max))
            out_flat[start:stop] = acc.to(self._feature_dtype)
        return out


@dataclass
class SpectrumBranchState:
    config: SpectrumConfig
    forecaster: HistoryWeightChebyshevForecaster = field(init=False)
    consecutive_forecasts: int = 0
    current_window: float = 2.0
    actual_count: int = 0
    forecast_count: int = 0

    # Adaptive local-drift controller.
    baseline_p95: Optional[float] = None
    baseline_p99: Optional[float] = None
    baseline_observations: int = 0
    adaptive_horizon: int = 1
    safe_refreshes: int = 0
    repair_actual_steps_remaining: int = 0
    last_p95: Optional[float] = None
    last_p99: Optional[float] = None
    last_local_ratio: Optional[float] = None
    last_validation_tier: str = "learning"
    last_streak: int = 0

    # Temporary guard after a local anomaly, counted in validation refreshes.
    cooldown_refreshes_remaining: int = 0
    cooldown_horizon_cap: Optional[int] = None

    def __post_init__(self) -> None:
        self.current_window = float(self.config.window_size)
        self.adaptive_horizon = (
            max(1, self.config.post_guard_horizon_cap)
            if self.config.adaptive
            else max(1, self.config.max_consecutive_forecasts)
        )
        self.forecaster = HistoryWeightChebyshevForecaster(
            degree=self.config.chebyshev_degree,
            ridge_lambda=self.config.ridge_lambda,
            max_history=self.config.history_points,
            blend_weight=self.config.blend_weight,
        )

    def reset(self) -> None:
        self.forecaster.reset()
        self.consecutive_forecasts = 0
        self.current_window = float(self.config.window_size)
        self.actual_count = 0
        self.forecast_count = 0
        self.baseline_p95 = None
        self.baseline_p99 = None
        self.baseline_observations = 0
        self.adaptive_horizon = (
            max(1, self.config.post_guard_horizon_cap)
            if self.config.adaptive
            else max(1, self.config.max_consecutive_forecasts)
        )
        self.safe_refreshes = 0
        self.repair_actual_steps_remaining = 0
        self.last_p95 = None
        self.last_p99 = None
        self.last_local_ratio = None
        self.last_validation_tier = "learning"
        self.last_streak = 0
        self.cooldown_refreshes_remaining = 0
        self.cooldown_horizon_cap = None


@dataclass
class SpectrumRootState:
    config: SpectrumConfig
    branches: dict[Any, SpectrumBranchState] = field(default_factory=dict)
    last_sigmas_id: Optional[int] = None
    last_total_steps: Optional[int] = None
    last_step_index: Optional[int] = None
    completed_branches: set[Any] = field(default_factory=set)
    summary_logged: bool = False

    def reset_run(self) -> None:
        self.branches.clear()
        self.last_sigmas_id = None
        self.last_total_steps = None
        self.last_step_index = None
        self.completed_branches.clear()
        self.summary_logged = False

    def branch(self, key: Any) -> SpectrumBranchState:
        if key not in self.branches:
            self.branches[key] = SpectrumBranchState(self.config)
        return self.branches[key]


def _resolve_cache_target(hidden: torch.Tensor, cache_device: str) -> tuple[torch.device, torch.dtype]:
    if cache_device == "cpu":
        return torch.device("cpu"), hidden.dtype
    if cache_device == "offload_device":
        try:
            return torch.device(model_management.unet_offload_device()), hidden.dtype
        except Exception:
            return torch.device("cpu"), hidden.dtype
    return hidden.device, hidden.dtype


def _is_moss_core(obj: Any) -> bool:
    if obj is None:
        return False
    if type(obj).__name__ == "MOSSImage":
        return True
    required = (
        "layers",
        "time_embedding",
        "video_head_modulation",
        "final_norm_video",
        "final_linear_video",
        "latent_channels",
    )
    if not all(hasattr(obj, name) for name in required):
        return False
    layers = getattr(obj, "layers", None)
    return hasattr(layers, "__len__") and len(layers) > 0


def _find_step_index(sigmas: torch.Tensor, timestep: torch.Tensor) -> int:
    """Map the model timestep back to the sampler sigma index.

    Native MOSS is ModelType.FLOW, whose ModelSamplingDiscreteFlow passes
    ``timestep = sigma * multiplier`` (normally multiplier=1000) into the
    diffusion model.  Some other flow wrappers pass sigma directly.  Try both
    representations and keep the one that actually lands on the sigma schedule.
    """
    if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 1:
        return -1
    if not isinstance(timestep, torch.Tensor) or timestep.numel() < 1:
        return -1

    raw = timestep.reshape(-1)[0].float()
    s = sigmas.to(device=raw.device, dtype=torch.float32)
    candidates = [raw, raw / 1000.0]

    best_index = -1
    best_error = float("inf")
    scale = max(float(s.abs().max().item()), 1e-6)
    for target in candidates:
        diffs = (s - target).abs()
        idx = int(diffs.argmin().item())
        err = float(diffs[idx].item()) / scale
        if err < best_error:
            best_error = err
            best_index = idx

    return best_index


def _patch_signature(transformer_options: dict) -> tuple[str, ...]:
    """Separate histories when DiT replacement patches (e.g. SLG) change."""
    try:
        dit = transformer_options.get("patches_replace", {}).get("dit", {})
        if not isinstance(dit, dict) or not dit:
            return ()
        return tuple(sorted(repr(key) for key in dit.keys()))
    except Exception:
        return ()


def _adaptive_phase_horizon_cap(config: SpectrumConfig, step_index: int) -> int:
    """Hard horizon cap for the validated MOSS-Image adaptive policy.

    Warmup is handled separately. The following detail guard is intentionally
    conservative, then the scheduler may expand up to the post-guard cap.
    """
    guard_end = config.warmup_steps + max(0, config.detail_guard_steps)
    if step_index < guard_end:
        return _ADAPTIVE_GUARD_HORIZON
    return max(1, config.post_guard_horizon_cap)


def _choose_actual(
    state: SpectrumBranchState,
    step_index: int,
    total_steps: int,
    control_present: bool,
    config: SpectrumConfig,
) -> tuple[bool, str]:
    if step_index < 0:
        return True, "unknown_step"
    if step_index < config.warmup_steps:
        return True, "warmup"
    if config.tail_actual_steps > 0 and step_index >= max(0, total_steps - config.tail_actual_steps):
        return True, "tail"
    if control_present and config.force_actual_on_control:
        return True, "control"
    if not state.forecaster.ready():
        return True, "insufficient_history"

    if config.adaptive:
        if state.repair_actual_steps_remaining > 0:
            return True, "local_repair"

        # Adaptive mode interprets window_size as a forecast budget before the
        # next real refresh. Local drift and phase protection can only reduce it.
        window_cap = max(1, int(math.floor(state.current_window)))
        phase_cap = _adaptive_phase_horizon_cap(config, step_index)
        safety_cap = min(max(1, state.adaptive_horizon), phase_cap)
        if state.cooldown_refreshes_remaining > 0 and state.cooldown_horizon_cap is not None:
            safety_cap = min(safety_cap, max(1, state.cooldown_horizon_cap))
        effective_cap = min(window_cap, safety_cap)
        if state.consecutive_forecasts >= effective_cap:
            return True, "window" if window_cap <= safety_cap else "local_guard"
        return False, "forecast"

    # Manual mode uses the original Spectrum-style expanding-window scheduler.
    if state.consecutive_forecasts >= config.max_consecutive_forecasts:
        return True, "horizon_refresh"

    interval = max(1, int(math.floor(state.current_window)))
    if (state.consecutive_forecasts + 1) % interval == 0:
        return True, "window"
    return False, "forecast"

def _note_decision(state: SpectrumBranchState, actual: bool, reason: str, config: SpectrumConfig) -> None:
    if actual:
        state.consecutive_forecasts = 0
        state.actual_count += 1
        if reason == "window":
            state.current_window = min(
                state.current_window + config.flex_window,
                config.window_growth_cap,
            )
        if reason == "local_repair":
            state.repair_actual_steps_remaining = max(0, state.repair_actual_steps_remaining - 1)
    else:
        state.consecutive_forecasts += 1
        state.forecast_count += 1


def _capture_last_block_feature(
    config: SpectrumConfig,
    captured: dict,
    image_tokens: int,
):
    def hook(_module: Any, _args: tuple, output: Any) -> None:
        if not torch.is_tensor(output) or output.ndim != 3:
            return
        hidden = output[:, :image_tokens]
        target_device, target_dtype = _resolve_cache_target(hidden, config.cache_device)
        try:
            captured["feature"] = hidden.detach().to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=(target_device.type == hidden.device.type),
            ).contiguous()
        except RuntimeError:
            # VRAM/offload target failed: prefer a slower CPU anchor to breaking the run.
            try:
                captured["feature"] = hidden.detach().to("cpu").contiguous()
            except RuntimeError:
                return
    return hook


def _adaptive_validate_and_update(
    config: SpectrumConfig,
    branch: SpectrumBranchState,
    coord: float,
    actual_feature: Optional[torch.Tensor],
    allow_action: bool,
    allow_baseline_update: bool,
    preceding_forecast_streak: int,
) -> None:
    """Update the local p95/p99 drift controller after an actual refresh."""
    if not config.adaptive or actual_feature is None or not branch.forecaster.ready():
        return
    try:
        stats = branch.forecaster.sampled_token_error_stats(
            coord, actual_feature, _ADAPTIVE_SAMPLE_TOKENS
        )
    except Exception as exc:
        _log_debug(config.debug, f"local validation failed ({type(exc).__name__}: {exc})")
        return

    p95 = stats["p95"]
    p99 = stats["p99"]
    branch.last_p95 = p95
    branch.last_p99 = p99
    branch.last_streak = int(preceding_forecast_streak)

    baseline_eligible = allow_baseline_update and preceding_forecast_streak <= 1
    if branch.baseline_p95 is None or branch.baseline_p99 is None:
        branch.baseline_p95 = p95
        branch.baseline_p99 = p99
        branch.baseline_observations = 1
        ratio = 1.0
        tier = "learning"
    else:
        r95 = p95 / max(branch.baseline_p95, 1e-8)
        r99 = p99 / max(branch.baseline_p99, 1e-8)
        ratio = max(r95, r99)
        medium_threshold = _ADAPTIVE_MEDIUM_RATIO * config.error_tolerance
        high_threshold = _ADAPTIVE_HIGH_RATIO * config.error_tolerance
        if branch.baseline_observations < 3:
            tier = "learning"
        elif ratio >= high_threshold:
            tier = "high"
        elif ratio >= medium_threshold:
            tier = "medium"
        else:
            tier = "normal"

        # Repair/tail observations do not update the baseline. A very clean
        # repair sample can otherwise make later normal refreshes look anomalous.
        if baseline_eligible and tier in {"learning", "normal"}:
            a = _ADAPTIVE_BASELINE_ALPHA
            branch.baseline_p95 = (1.0 - a) * branch.baseline_p95 + a * p95
            branch.baseline_p99 = (1.0 - a) * branch.baseline_p99 + a * p99
            branch.baseline_observations += 1

    branch.last_local_ratio = ratio
    branch.last_validation_tier = tier

    if allow_action and branch.baseline_observations >= 3:
        if tier == "high":
            observed_cap = max(1, preceding_forecast_streak - 1)
            branch.adaptive_horizon = max(
                1,
                min(branch.adaptive_horizon - 1, observed_cap),
            )
            branch.safe_refreshes = 0
            branch.current_window = float(config.window_size)
            branch.repair_actual_steps_remaining = max(
                branch.repair_actual_steps_remaining, _ADAPTIVE_REPAIR_ACTUAL_STEPS
            )
            branch.cooldown_refreshes_remaining = max(
                branch.cooldown_refreshes_remaining, _ADAPTIVE_HIGH_COOLDOWN_REFRESHES
            )
            branch.cooldown_horizon_cap = max(
                1, min(2, branch.adaptive_horizon)
            )
        elif tier == "medium":
            if preceding_forecast_streak >= 2:
                branch.adaptive_horizon = max(1, branch.adaptive_horizon - 1)
            branch.safe_refreshes = 0
            branch.cooldown_horizon_cap = max(1, min(3, branch.adaptive_horizon))
        else:
            if branch.cooldown_refreshes_remaining > 0:
                branch.cooldown_refreshes_remaining -= 1
                if branch.cooldown_refreshes_remaining <= 0:
                    branch.cooldown_horizon_cap = None
            elif preceding_forecast_streak >= max(1, branch.adaptive_horizon - 1):
                branch.safe_refreshes += 1
                if branch.safe_refreshes >= _ADAPTIVE_SAFE_REFRESHES_TO_EXPAND:
                    max_h = max(1, config.post_guard_horizon_cap)
                    branch.adaptive_horizon = min(max_h, branch.adaptive_horizon + 1)
                    branch.safe_refreshes = 0

    _log_debug(
        config.debug,
        f"local tier={tier} p95={p95:.5f} p99={p99:.5f} "
        f"base95={branch.baseline_p95 if branch.baseline_p95 is not None else float('nan'):.5f} "
        f"base99={branch.baseline_p99 if branch.baseline_p99 is not None else float('nan'):.5f} "
        f"ratio={ratio:.3f} tol={config.error_tolerance:.2f} streak={preceding_forecast_streak} "
        f"horizon={branch.adaptive_horizon} repair={branch.repair_actual_steps_remaining} "
        f"cooldown={branch.cooldown_refreshes_remaining} "
        f"coolcap={branch.cooldown_horizon_cap if branch.cooldown_horizon_cap is not None else 'none'}",
    )

def _run_actual(
    executor: Any,
    core: Any,
    config: SpectrumConfig,
    branch: SpectrumBranchState,
    reason: str,
    step_index: int,
    total_steps: int,
    coord: float,
    branch_key: Any,
    args_tuple: tuple,
    kwargs: dict,
    image_tokens: int,
) -> Any:
    captured: dict[str, torch.Tensor] = {}
    handle = core.layers[-1].register_forward_hook(
        _capture_last_block_feature(config, captured, image_tokens)
    )
    try:
        out = executor(*args_tuple, **kwargs)
    finally:
        handle.remove()

    feature = captured.get("feature")
    if feature is not None:
        tail_start = max(0, total_steps - config.tail_actual_steps) if config.tail_actual_steps > 0 else total_steps
        adaptive_action_start = config.warmup_steps
        if config.adaptive:
            # Learn baselines throughout the protected band, but only let the
            # controller change horizon after the detail guard.
            adaptive_action_start += max(0, config.detail_guard_steps)
        allow_adaptive_action = (
            step_index >= adaptive_action_start
            and step_index < tail_start
            and reason != "local_repair"
        )
        preceding_forecast_streak = branch.consecutive_forecasts
        allow_baseline_update = reason not in {"local_repair", "tail"}
        _adaptive_validate_and_update(
            config, branch, coord, feature, allow_adaptive_action,
            allow_baseline_update, preceding_forecast_streak
        )
        try:
            branch.forecaster.update(coord, feature)
        except ValueError:
            branch.reset()
            try:
                branch.forecaster.update(coord, feature)
            except ValueError:
                _log_warning("could not record last-block hidden state after reset")
    else:
        _log_debug(config.debug, "last-block hook did not capture a feature; staying on real steps")

    _log_debug(
        config.debug,
        f"step {step_index + 1}/{total_steps} branch={branch_key} "
        f"mode=actual reason={reason} history={branch.forecaster.history_size} "
        f"p95={branch.last_p95 if branch.last_p95 is not None else 'n/a'} "
        f"p99={branch.last_p99 if branch.last_p99 is not None else 'n/a'} "
        f"ratio={branch.last_local_ratio if branch.last_local_ratio is not None else 'n/a'} "
        f"tier={branch.last_validation_tier} horizon={branch.adaptive_horizon} "
        f"cooldown={branch.cooldown_refreshes_remaining}",
    )
    return out


def _run_moss_head(core: Any, x: torch.Tensor, timestep: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """Run MOSS' timestep-conditioned output head on a forecast last-block state."""
    dtype = x.dtype
    device = x.device
    batch, _, height, width = x.shape

    hidden = feature.to(device=device, dtype=dtype, non_blocking=(feature.device.type == device.type))
    vec = core.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    head_shift, head_scale = (
        core.video_head_modulation(vec)
        .unflatten(-1, (2, -1))
        .unsqueeze(2)
        .unbind(1)
    )
    hidden = modulated_rms_norm(
        hidden,
        core.final_norm_video,
        head_scale,
        head_shift,
    )
    out = core.final_linear_video(hidden)
    return out.transpose(1, 2).reshape(batch, int(core.latent_channels), height, width)


def _run_forecast(
    core: Any,
    config: SpectrumConfig,
    branch: SpectrumBranchState,
    step_index: int,
    total_steps: int,
    coord: float,
    branch_key: Any,
    x: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    feature = branch.forecaster.predict(coord)
    out = _run_moss_head(core, x, timestep, feature)
    _log_debug(
        config.debug,
        f"step {step_index + 1}/{total_steps} branch={branch_key} "
        f"mode=forecast history={branch.forecaster.history_size} "
        f"ratio={branch.last_local_ratio if branch.last_local_ratio is not None else 'n/a'} "
        f"tier={branch.last_validation_tier} horizon={branch.adaptive_horizon} "
        f"cooldown={branch.cooldown_refreshes_remaining}",
    )
    return out


def _make_spectrum_wrapper(config: SpectrumConfig):
    root = SpectrumRootState(config)

    def spectrum_moss_wrapper(
        executor: Any,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Any = None,
        transformer_options: Any = None,
        **kwargs: Any,
    ) -> Any:
        core = getattr(executor, "class_obj", None)
        if not isinstance(transformer_options, dict):
            transformer_options = {}

        args_tuple = (x, timestep, context, transformer_options)

        # Fail closed on anything that is not native MOSS-Image.
        if not _is_moss_core(core):
            return executor(*args_tuple, **kwargs)

        sigmas = transformer_options.get("sample_sigmas")
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 2:
            return executor(*args_tuple, **kwargs)

        step_index = _find_step_index(sigmas, timestep)
        total_steps = max(1, int(sigmas.numel()) - 1)

        # New-run detection. The sigma schedule identity is normally unique per queue run;
        # a backwards step also catches repeated runs if the tensor is reused.
        if (
            (root.last_sigmas_id is not None and root.last_sigmas_id != id(sigmas))
            or (root.last_total_steps is not None and root.last_total_steps != total_steps)
            or (root.last_step_index is not None and step_index >= 0 and step_index < root.last_step_index)
        ):
            root.reset_run()

        cond_key = tuple(transformer_options.get("cond_or_uncond") or ())
        patch_key = _patch_signature(transformer_options)
        branch_key = (cond_key, patch_key)
        branch = root.branch(branch_key)

        image_tokens = int(x.shape[-2]) * int(x.shape[-1])
        known_shape = branch.forecaster.feature_shape
        if known_shape is not None and (
            len(known_shape) != 3
            or int(known_shape[0]) != int(x.shape[0])
            or int(known_shape[1]) != image_tokens
        ):
            branch.reset()

        control_present = (
            kwargs.get("control") is not None
            or transformer_options.get("control") is not None
        )
        actual, reason = _choose_actual(
            branch,
            step_index,
            total_steps,
            control_present,
            config,
        )
        coord = _normalize_step_position(step_index, total_steps)

        try:
            if actual:
                out = _run_actual(
                    executor,
                    core,
                    config,
                    branch,
                    reason,
                    step_index,
                    total_steps,
                    coord,
                    branch_key,
                    args_tuple,
                    kwargs,
                    image_tokens,
                )
            else:
                try:
                    out = _run_forecast(
                        core,
                        config,
                        branch,
                        step_index,
                        total_steps,
                        coord,
                        branch_key,
                        x,
                        timestep,
                    )
                except Exception as exc:
                    # A numerical/shape/API problem should never ruin a generation.
                    actual = True
                    reason = "forecast_failed"
                    _log_debug(config.debug, f"forecast failed ({type(exc).__name__}: {exc}); using real step")
                    out = _run_actual(
                        executor,
                        core,
                        config,
                        branch,
                        reason,
                        step_index,
                        total_steps,
                        coord,
                        branch_key,
                        args_tuple,
                        kwargs,
                        image_tokens,
                    )
        finally:
            _note_decision(branch, actual, reason, config)
            root.last_step_index = step_index
            root.last_sigmas_id = id(sigmas)
            root.last_total_steps = total_steps

        # Release anchors after all branches that exist at the final step have completed.
        if step_index >= 0 and step_index >= total_steps - 1:
            root.completed_branches.add(branch_key)
            if len(root.completed_branches) >= len(root.branches):
                if config.debug and not root.summary_logged:
                    actual_calls = sum(b.actual_count for b in root.branches.values())
                    forecast_calls = sum(b.forecast_count for b in root.branches.values())
                    total_calls = actual_calls + forecast_calls
                    ratio = (float(total_calls) / float(actual_calls)) if actual_calls else 0.0
                    _log_debug(
                        True,
                        f"run complete: branches={len(root.branches)} actual={actual_calls} "
                        f"forecast={forecast_calls} calls={total_calls} theoretical_transformer_speedup={ratio:.2f}x",
                    )
                    root.summary_logged = True
                for b in root.branches.values():
                    b.forecaster.release_features()

        return out

    spectrum_moss_wrapper.root_state = root  # type: ignore[attr-defined]
    return spectrum_moss_wrapper


def apply_spectrum_moss(model: Any, config: SpectrumConfig) -> Any:
    config.validate()
    clone = model.clone()
    core = clone.get_model_object("diffusion_model")
    if not _is_moss_core(core):
        raise ValueError(
            "Spectrum (MOSS-Image) requires the native ComfyUI MOSSImage core. "
            "Update ComfyUI to a build with native MOSS-Image support and load the "
            "MOSS diffusion model with Load Diffusion Model."
        )

    # Re-applying Spectrum to an already Spectrum-patched MODEL must replace
    # the previous Spectrum wrapper rather than stack another nested wrapper.
    # ModelPatcher.clone() carries model_options forward, while ComfyUI's
    # add_wrapper_with_key() appends under an existing key.
    transformer_options = clone.model_options.get("transformer_options")
    if isinstance(transformer_options, dict):
        wrappers = transformer_options.get("wrappers")
        if isinstance(wrappers, dict):
            by_type = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL)
            if isinstance(by_type, dict):
                by_type.pop(WRAPPER_KEY, None)

    wrapper = _make_spectrum_wrapper(config)
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        wrapper,
        clone.model_options,
        is_model_options=True,
    )
    return clone



class SpectrumMOSSImageManual:
    """Manual Spectrum controls for native ComfyUI MOSS-Image."""

    DESCRIPTION = (
        "Manual Spectrum scheduling for MOSS-Image. Forecasts hidden states to skip full "
        "transformer evaluations. Higher forecast settings improve speed but may reduce image quality."
    )
    CATEGORY = "model/optimization"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "warmup_steps": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Initial steps that always run the full transformer. Higher = safer quality, lower = more speed. Default 8.",
                }),
                "tail_actual_steps": ("INT", {
                    "default": 2, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Final steps that always run the full transformer. Helps preserve late-stage details. Higher = safer, lower = faster. Default 2.",
                }),
                "window_size": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 16.0, "step": 0.25,
                    "tooltip": "Starting forecast window. Controls how quickly full transformer refreshes occur. Higher = potentially more speed, but greater forecast error risk.",
                }),
                "flex_window": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Amount the forecast window can grow after successful refreshes. Higher = ramps toward longer forecast streaks faster. Lower = more conservative.",
                }),
                "max_consecutive_forecasts": ("INT", {
                    "default": 2, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Maximum forecasts allowed in a row before forcing a full transformer refresh. Main speed/quality control: higher = faster but riskier. Default 2 is conservative.",
                }),
                "history_points": ("INT", {
                    "default": 8, "min": 2, "max": 32, "step": 1,
                    "tooltip": "Number of previous actual hidden states used for forecasting. More history uses more memory. Usually leave at default 8.",
                }),
                "chebyshev_degree": ("INT", {
                    "default": 4, "min": 1, "max": 12, "step": 1,
                    "tooltip": "Polynomial degree used by the spectral predictor. Higher is not necessarily better and may become unstable. Usually leave at default 4.",
                }),
                "ridge_lambda": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Regularization strength for the spectral predictor. Helps prevent unstable extrapolation. Usually leave at default 0.1.",
                }),
                "blend_weight": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mix between spectral prediction and linear extrapolation. 0 = linear only, 1 = spectral only. Default 0.5 is recommended.",
                }),
                "cache_device": (["main_device", "offload_device", "cpu"], {
                    "default": "main_device",
                    "tooltip": "Where actual hidden-state history is stored. main_device is fastest but uses VRAM. CPU/offload saves VRAM but may reduce speed.",
                }),
                "force_actual_on_control": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force full transformer evaluations when a control input is present. Safer for compatibility and quality. Recommended: enabled.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print Spectrum scheduling, forecast error and speed statistics to the console.",
                }),
            }
        }

    def patch(self, model, warmup_steps, tail_actual_steps, window_size, flex_window,
              max_consecutive_forecasts, history_points, chebyshev_degree, ridge_lambda,
              blend_weight, cache_device, force_actual_on_control, debug):
        config = SpectrumConfig(
            mode="manual",
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            window_size=window_size,
            flex_window=flex_window,
            max_consecutive_forecasts=max_consecutive_forecasts,
            history_points=history_points,
            chebyshev_degree=chebyshev_degree,
            ridge_lambda=ridge_lambda,
            blend_weight=blend_weight,
            cache_device=cache_device,
            force_actual_on_control=force_actual_on_control,
            debug=debug,
        )
        return (apply_spectrum_moss(model, config),)


class SpectrumMOSSImageAdaptive:
    """Validated adaptive policy for native ComfyUI MOSS-Image."""

    DESCRIPTION = (
        "Experimental error-aware Spectrum scheduler for MOSS-Image. It measures forecast error "
        "at actual refreshes and automatically adjusts the forecast horizon. Speed varies with prompt and trajectory."
    )
    CATEGORY = "model/optimization"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "warmup_steps": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Initial steps that always run the full transformer. Higher = safer quality, lower = more potential speed. Default 8.",
                }),
                "detail_guard_steps": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Protected steps immediately after warmup. At most one forecast is allowed between actual refreshes. Higher = safer early details but slower. Default 8.",
                }),
                "tail_actual_steps": ("INT", {
                    "default": 2, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Final steps that always run the full transformer. Helps preserve late-stage details. Higher = safer, lower = faster. Default 2.",
                }),
                "post_guard_horizon_cap": ("INT", {
                    "default": 6, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Maximum forecast streak allowed after the detail guard. The adaptive error controller may automatically use a lower value. Higher = more potential speed but greater quality risk.",
                }),
                "window_size": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 16.0, "step": 0.25,
                    "tooltip": "Starting forecast budget before the next full transformer refresh. The adaptive safety controller can reduce it when forecast error increases.",
                }),
                "flex_window": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "How quickly the forecast budget can grow after successful refreshes. Higher = more aggressive speed ramp. Lower = more conservative.",
                }),
                "cache_device": (["main_device", "offload_device", "cpu"], {
                    "default": "main_device",
                    "tooltip": "Where actual hidden-state history is stored. main_device is fastest but uses VRAM. CPU/offload saves VRAM but may reduce speed.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print adaptive decisions, p95/p99 forecast error, horizon changes and final actual/forecast counts to the console.",
                }),
                # Keep new widgets appended after all v0.6 widgets. ComfyUI stores
                # workflow widget values positionally, so inserting this in the
                # middle would shift values when an old v0.6 workflow is loaded.
                "error_tolerance": ("FLOAT", {
                    "default": 1.0, "min": 0.75, "max": 1.5, "step": 0.05,
                    "tooltip": "Controls how much forecast error the adaptive scheduler tolerates before reducing the forecast horizon. Higher = faster but riskier. Lower = safer but slower. 1.0 uses the default.",
                }),
            }
        }

    def patch(self, model, warmup_steps, detail_guard_steps, tail_actual_steps,
              post_guard_horizon_cap, window_size, flex_window, cache_device, debug, error_tolerance):
        cap = max(1, int(post_guard_horizon_cap))
        config = SpectrumConfig(
            mode="adaptive",
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            window_size=window_size,
            flex_window=flex_window,
            history_points=_ADAPTIVE_HISTORY_POINTS,
            chebyshev_degree=_ADAPTIVE_CHEBYSHEV_DEGREE,
            ridge_lambda=_ADAPTIVE_RIDGE_LAMBDA,
            blend_weight=_ADAPTIVE_BLEND_WEIGHT,
            cache_device=cache_device,
            force_actual_on_control=True,
            detail_guard_steps=detail_guard_steps,
            post_guard_horizon_cap=cap,
            error_tolerance=error_tolerance,
            debug=debug,
        )
        return (apply_spectrum_moss(model, config),)


NODE_CLASS_MAPPINGS = {
    "SpectrumMOSSImageManual": SpectrumMOSSImageManual,
    "SpectrumMOSSImageAdaptive": SpectrumMOSSImageAdaptive,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumMOSSImageManual": "Spectrum MOSS-Image (Manual)",
    "SpectrumMOSSImageAdaptive": "Spectrum MOSS-Image (Adaptive)",
}
