# ComfyUI-Spectrum-MOSS
# Native ER-SDE solver-space bridge for Spectrum MOSS-Image.
#
# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

import contextvars
import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

import comfy.patcher_extension


LOG_TAG = "Spectrum-MOSS"
ER_SDE_BRIDGE_SAMPLER_WRAPPER_KEY = "spectrum_moss_er_sde_bridge_sampler"
ER_SDE_BRIDGE_PREDICT_WRAPPER_KEY = "spectrum_moss_er_sde_bridge_predict"
_SUPPORTED_ER_SDE_SAMPLER = "sample_er_sde"
_VERIFIED_SAMPLERS = frozenset({"sample_euler", "sample_er_sde"})
_ER_SDE_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "spectrum_moss_er_sde_active", default=False
)


def _debug(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[{LOG_TAG}] {message}", flush=True)


def _warning(message: str) -> None:
    print(f"\033[93m[{LOG_TAG}] WARNING: {message}\033[0m", flush=True)


def sampler_name(sampler: Any) -> str:
    fn = getattr(sampler, "sampler_function", None)
    return str(getattr(fn, "__name__", type(sampler).__name__))


def er_sde_active() -> bool:
    """True while native sample_er_sde is executing this Spectrum model."""
    return bool(_ER_SDE_ACTIVE.get())


def _native_default_noise_scaler(value: torch.Tensor) -> torch.Tensor:
    # Current native sample_er_sde default. Keep this mathematically identical.
    return value * ((value ** 0.3).exp() + 10.0)


@dataclass(slots=True)
class _DenoisedAnchor:
    step_id: int
    value: torch.Tensor


@dataclass(slots=True)
class _StepDecision:
    modes: list[bool]  # True = actual, False = forecast


class ERSDESolverBridge:
    """Solver-space bridge for native ComfyUI sample_er_sde.

    Spectrum still decides which transformer calls are actual vs forecast. The
    bridge leaves native ER-SDE RNG/noise ownership untouched, records exact
    guided denoised outputs on actual steps, and replaces forecast observations
    with a causal two-anchor dense output in native ER-SDE lambda coordinates.

    Mixed actual/forecast CFG branches (for example at a partial SLG boundary)
    are never passed raw to ER-SDE; they use the same clean-anchor dense output.
    """

    def __init__(self, *, debug: bool, warn_unverified_sampler: bool) -> None:
        self.debug = bool(debug)
        self.warn_unverified_sampler = bool(warn_unverified_sampler)
        self.active = False
        self.sampler_name: Optional[str] = None
        self.supported = False
        self.bridge_enabled = False
        self.bypass_spectrum = False
        self._base_noise_scaler: Callable[[torch.Tensor], torch.Tensor] = _native_default_noise_scaler
        self._solver_coordinates: dict[int, torch.Tensor] = {}
        self._anchors: list[_DenoisedAnchor] = []
        self._decisions: dict[int, _StepDecision] = {}
        self._latest_decision_step: Optional[int] = None
        self._scaler_step: Optional[int] = None
        self._scaler_calls = 0
        self.actual_observations = 0
        self.forecast_observations = 0
        self.hold_predictions = 0
        self.lambda_predictions = 0
        self.lambda_continuation_predictions = 0
        self.bridge_fallbacks = 0
        self.mixed_observations = 0
        self.mixed_dense_predictions = 0
        self.mixed_raw_fallbacks = 0

    def reset_runtime(self) -> None:
        self._solver_coordinates.clear()
        self._anchors.clear()
        self._decisions.clear()
        self._latest_decision_step = None
        self._scaler_step = None
        self._scaler_calls = 0
        self.actual_observations = 0
        self.forecast_observations = 0
        self.hold_predictions = 0
        self.lambda_predictions = 0
        self.lambda_continuation_predictions = 0
        self.bridge_fallbacks = 0
        self.mixed_observations = 0
        self.mixed_dense_predictions = 0
        self.mixed_raw_fallbacks = 0
        self.bypass_spectrum = False

    def begin_sampler(self, sampler: Any) -> None:
        self.reset_runtime()
        self.active = True
        self.sampler_name = sampler_name(sampler)
        self.supported = self.sampler_name == _SUPPORTED_ER_SDE_SAMPLER
        self.bridge_enabled = self.supported
        if self.warn_unverified_sampler and self.sampler_name not in _VERIFIED_SAMPLERS:
            _warning(f"Unverified sampler: {self.sampler_name}.")

    def end_sampler(self) -> None:
        if self.debug and self.supported:
            retained_bytes = sum(a.value.numel() * a.value.element_size() for a in self._anchors)
            _debug(
                True,
                "ER-SDE bridge summary: "
                f"actual_observations={self.actual_observations} "
                f"forecast_observations={self.forecast_observations} "
                f"latest_hold={self.hold_predictions} "
                f"lambda_dense={self.lambda_predictions} "
                f"lambda_continue={self.lambda_continuation_predictions} "
                f"mixed={self.mixed_observations} "
                f"mixed_dense={self.mixed_dense_predictions} "
                f"mixed_raw_fallback={self.mixed_raw_fallbacks} "
                f"fallbacks={self.bridge_fallbacks} "
                f"retained_anchor_bytes={retained_bytes}",
            )
        self.active = False
        self.sampler_name = None
        self.supported = False
        self.bridge_enabled = False
        self.bypass_spectrum = False
        self._decisions.clear()
        self._latest_decision_step = None
        self._scaler_step = None
        self._scaler_calls = 0

    def set_base_noise_scaler(self, scaler: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self._base_noise_scaler = scaler or _native_default_noise_scaler

    def note_branch_decision(
        self,
        step_id: int,
        _total_steps: int,
        actual: bool,
        _reason: str,
        _branch_key: Any,
    ) -> None:
        if not self.active or not self.bridge_enabled or step_id < 0:
            return
        item = self._decisions.get(step_id)
        if item is None:
            item = _StepDecision(modes=[])
            self._decisions[step_id] = item
        item.modes.append(bool(actual))
        self._latest_decision_step = step_id

    def _consume_mode(self) -> tuple[int, Optional[bool]] | None:
        step_id = self._latest_decision_step
        if step_id is None:
            return None
        item = self._decisions.pop(step_id, None)
        self._latest_decision_step = None
        if item is None or not item.modes:
            return None
        if any(mode != item.modes[0] for mode in item.modes[1:]):
            self.mixed_observations += 1
            _debug(
                self.debug,
                f"ER-SDE bridge step {step_id + 1}: mixed actual/forecast branches -> dense output",
            )
            return step_id, None
        return step_id, item.modes[0]

    @staticmethod
    def _same_tensor_contract(a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.shape == b.shape and a.device == b.device and a.dtype == b.dtype

    def _observe_actual(self, step_id: int, denoised: torch.Tensor) -> None:
        retained = denoised.detach().clone(memory_format=torch.contiguous_format)
        self._anchors.append(_DenoisedAnchor(step_id=int(step_id), value=retained))
        if len(self._anchors) > 2:
            del self._anchors[:-2]
        self.actual_observations += 1
        _debug(
            self.debug,
            f"ER-SDE bridge step {step_id + 1}: actual solver anchor; "
            f"anchors={[a.step_id + 1 for a in self._anchors]}",
        )

    def _latest_hold(self, step_id: int, raw: torch.Tensor, mode: str) -> torch.Tensor:
        latest = self._anchors[-1]
        if not self._same_tensor_contract(latest.value, raw):
            self.bridge_fallbacks += 1
            _warning(f"ER-SDE bridge tensor contract changed at step {step_id + 1}.")
            return raw
        self.hold_predictions += 1
        _debug(
            self.debug,
            f"ER-SDE bridge step {step_id + 1}: mode={mode} anchor={latest.step_id + 1}",
        )
        return latest.value.clone(memory_format=torch.contiguous_format)

    def _predict_f2_continuation(self, step_id: int, raw: torch.Tensor) -> torch.Tensor:
        """Continue F2 from the two latest ACTUAL anchors in ER-SDE lambda space.

        Forecast outputs never become anchors. F3+ remains on the conservative
        latest-actual hold path until a longer continuation horizon is validated.
        """
        latest = self._anchors[-1]
        if len(self._anchors) < 2:
            return self._latest_hold(step_id, raw, "continuation_hold_no_second_anchor")
        previous = self._anchors[-2]
        if (
            not self._same_tensor_contract(previous.value, latest.value)
            or not self._same_tensor_contract(latest.value, raw)
        ):
            return self._latest_hold(step_id, raw, "continuation_hold_contract_guard")

        previous_coord = self._solver_coordinates.get(previous.step_id)
        latest_coord = self._solver_coordinates.get(latest.step_id)
        target_coord = self._solver_coordinates.get(step_id)
        if previous_coord is None or latest_coord is None or target_coord is None:
            return self._latest_hold(step_id, raw, "continuation_hold_missing_lambda")

        denominator = latest_coord - previous_coord
        advance = target_coord - latest_coord
        alpha = advance / denominator
        scale = torch.maximum(
            torch.maximum(previous_coord.abs(), latest_coord.abs()),
            torch.ones_like(latest_coord),
        )
        # Validated F2 cases need at most ~1.4x continuation. Keep a conservative
        # 1.5x rail; longer forecast runs are not extrapolated by this path.
        valid = (
            torch.isfinite(alpha)
            & torch.isfinite(denominator)
            & (denominator.abs() > 1e-12 * scale)
            & (denominator * advance >= 0.0)
            & (alpha >= -1e-6)
            & (alpha <= 1.5 + 1e-6)
        )
        if not bool(valid.item() if valid.numel() == 1 else valid.all().item()):
            return self._latest_hold(step_id, raw, "continuation_hold_invalid_lambda")

        weight = alpha.clamp(0.0, 1.5).to(device=latest.value.device, dtype=latest.value.dtype)
        predicted = torch.lerp(latest.value, previous.value, -weight)
        self.lambda_continuation_predictions += 1
        alpha_scalar = float(alpha.detach().reshape(-1)[0].cpu())
        _debug(
            self.debug,
            f"ER-SDE bridge step {step_id + 1}: mode=lambda_actual_anchor_continuation "
            f"anchors=({previous.step_id + 1},{latest.step_id + 1}) alpha={alpha_scalar:.4f}",
        )
        return predicted

    def _predict_dense(self, step_id: int, raw: torch.Tensor, *, mixed: bool = False) -> torch.Tensor:
        if not mixed:
            self.forecast_observations += 1
        if not self._anchors:
            self.bridge_fallbacks += 1
            if mixed:
                self.mixed_raw_fallbacks += 1
            _warning(f"ER-SDE bridge has no clean anchor at step {step_id + 1}.")
            return raw

        latest = self._anchors[-1]
        mode_prefix = "mixed_" if mixed else ""
        if latest.step_id != step_id - 1:
            if not mixed and latest.step_id == step_id - 2:
                return self._predict_f2_continuation(step_id, raw)
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold_nonadjacent")
        if len(self._anchors) < 2:
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold")

        previous = self._anchors[-2]
        if (
            not self._same_tensor_contract(previous.value, latest.value)
            or not self._same_tensor_contract(latest.value, raw)
        ):
            if mixed and not self._same_tensor_contract(latest.value, raw):
                self.mixed_raw_fallbacks += 1
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold_contract_guard")

        previous_coord = self._solver_coordinates.get(previous.step_id)
        latest_coord = self._solver_coordinates.get(latest.step_id)
        target_coord = self._solver_coordinates.get(step_id)
        if previous_coord is None or latest_coord is None or target_coord is None:
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold_missing_lambda")

        denominator = latest_coord - previous_coord
        advance = target_coord - latest_coord
        alpha = advance / denominator
        scale = torch.maximum(
            torch.maximum(previous_coord.abs(), latest_coord.abs()),
            torch.ones_like(latest_coord),
        )
        valid = (
            torch.isfinite(alpha)
            & torch.isfinite(denominator)
            & (denominator.abs() > 1e-12 * scale)
            & (denominator * advance >= 0.0)
            & (alpha >= -1e-6)
            & (alpha <= 1.0 + 1e-6)
        )
        weight = torch.where(valid, alpha.clamp(0.0, 1.0), torch.zeros_like(alpha))
        weight = weight.to(device=latest.value.device, dtype=latest.value.dtype)
        predicted = torch.lerp(latest.value, previous.value, -weight)
        self.lambda_predictions += 1
        if mixed:
            self.mixed_dense_predictions += 1

        mode_name = "mixed_lambda_bounded_extrapolation" if mixed else "lambda_bounded_extrapolation"
        _debug(
            self.debug,
            f"ER-SDE bridge step {step_id + 1}: mode={mode_name} "
            f"anchors=({previous.step_id + 1},{latest.step_id + 1})",
        )
        return predicted

    def consume_guided_denoised(self, denoised: torch.Tensor) -> torch.Tensor:
        """Bridge the guided denoised result before native ER-SDE sees it."""
        if not self.active or not self.bridge_enabled or not torch.is_tensor(denoised):
            return denoised
        decision = self._consume_mode()
        if decision is None:
            return denoised
        step_id, actual = decision

        if actual is True:
            bridged = denoised
            self._observe_actual(step_id, denoised)
        elif actual is False:
            bridged = self._predict_dense(step_id, denoised)
        else:
            bridged = self._predict_dense(step_id, denoised, mixed=True)

        # Native sample_er_sde calls noise_scaler after the model result returns.
        # Arm coordinate capture for exactly this solver step.
        self._scaler_step = int(step_id)
        self._scaler_calls = 0
        return bridged

    def noise_scaler(self, value: torch.Tensor) -> torch.Tensor:
        result = self._base_noise_scaler(value)
        step_id = self._scaler_step
        if (
            step_id is not None
            and self._scaler_calls < 2
            and torch.is_tensor(value)
            and value.numel() == 1
        ):
            coordinate = value.detach().clone()
            if self._scaler_calls == 0:
                # sample_er_sde evaluates noise_scaler(er_lambda_t) first.
                self._solver_coordinates[step_id + 1] = coordinate
            else:
                self._solver_coordinates[step_id] = coordinate
            self._scaler_calls += 1
        elif step_id is not None:
            self._scaler_calls += 1
        return result


def make_predict_noise_wrapper(bridge: ERSDESolverBridge):
    def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
        result = executor(x, timestep, model_options or {}, seed)
        return bridge.consume_guided_denoised(result)

    return predict_noise_wrapper


def make_sampler_sample_wrapper(bridge: ERSDESolverBridge):
    def sampler_sample_wrapper(
        executor,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        sampler = executor.class_obj
        bridge.begin_sampler(sampler)
        name = bridge.sampler_name

        if name != _SUPPORTED_ER_SDE_SAMPLER:
            try:
                return executor(
                    model_wrap,
                    sigmas,
                    extra_args,
                    callback,
                    noise,
                    latent_image,
                    denoise_mask,
                    disable_pbar,
                )
            finally:
                bridge.end_sampler()

        token = _ER_SDE_ACTIVE.set(True)
        try:
            options = dict(getattr(sampler, "extra_options", {}) or {})
            custom_noise_sampler = options.get("noise_sampler")
            custom_noise_scaler = options.get("noise_scaler")

            # The validated bridge uses native ER-SDE noise ownership. A custom
            # noise sampler has not been validated, so fail closed internally;
            # this is a structural bridge safeguard, not the old sampler fallback policy.
            if custom_noise_sampler is not None:
                bridge.bridge_enabled = False
                bridge.bypass_spectrum = True
                _warning("ER-SDE custom noise_sampler is not bridged; using actual steps.")
                return executor(
                    model_wrap,
                    sigmas,
                    extra_args,
                    callback,
                    noise,
                    latent_image,
                    denoise_mask,
                    disable_pbar,
                )

            if custom_noise_scaler is not None and not callable(custom_noise_scaler):
                bridge.bridge_enabled = False
                bridge.bypass_spectrum = True
                _warning("ER-SDE noise_scaler is not callable; using actual steps.")
                return executor(
                    model_wrap,
                    sigmas,
                    extra_args,
                    callback,
                    noise,
                    latent_image,
                    denoise_mask,
                    disable_pbar,
                )

            # The tracked sampler copy below is required to inject coordinate
            # capture without changing native RNG/noise. Do not silently bypass
            # another sampler-level extension.
            wrappers = getattr(executor, "wrappers", None)
            if isinstance(wrappers, (list, tuple)) and len(wrappers) != 1:
                bridge.bridge_enabled = False
                bridge.bypass_spectrum = True
                _warning("ER-SDE sampler-wrapper conflict; using actual steps.")
                return executor(
                    model_wrap,
                    sigmas,
                    extra_args,
                    callback,
                    noise,
                    latent_image,
                    denoise_mask,
                    disable_pbar,
                )

            bridge.set_base_noise_scaler(custom_noise_scaler)
            tracked_sampler = copy.copy(sampler)
            tracked_options = dict(options)
            tracked_options["noise_scaler"] = bridge.noise_scaler
            tracked_sampler.extra_options = tracked_options

            _debug(
                bridge.debug,
                "ER-SDE solver bridge active: native RNG/noise untouched; "
                "forecast denoised -> two-anchor lambda dense output",
            )
            return tracked_sampler.sample(
                model_wrap,
                sigmas,
                extra_args,
                callback,
                noise,
                latent_image,
                denoise_mask,
                disable_pbar,
            )
        finally:
            _ER_SDE_ACTIVE.reset(token)
            bridge.end_sampler()

    return sampler_sample_wrapper


def install_bridge_wrappers(model_options: dict, bridge: ERSDESolverBridge) -> None:
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        ER_SDE_BRIDGE_PREDICT_WRAPPER_KEY,
        make_predict_noise_wrapper(bridge),
        model_options,
        is_model_options=True,
    )
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        ER_SDE_BRIDGE_SAMPLER_WRAPPER_KEY,
        make_sampler_sample_wrapper(bridge),
        model_options,
        is_model_options=True,
    )


__all__ = [
    "ER_SDE_BRIDGE_PREDICT_WRAPPER_KEY",
    "ER_SDE_BRIDGE_SAMPLER_WRAPPER_KEY",
    "ERSDESolverBridge",
    "er_sde_active",
    "install_bridge_wrappers",
]
