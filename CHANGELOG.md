# Changelog

## 0.8.0
- Added automatic native ComfyUI `er_sde` detection to the two public nodes.
- Added the validated ER-SDE solver-space bridge: native RNG/noise stays owned by ER-SDE while Spectrum forecast observations are replaced with causal two-anchor lambda-space dense output.
- Added mixed actual/forecast branch protection for partial SLG ranges so mixed guided observations do not leak raw Spectrum forecasts into ER-SDE.
- Manual keeps its scheduling controls under ER-SDE. `max_consecutive_forecasts` remains a ceiling, not a forced cadence; values above 1 may allow consecutive forecasts when the scheduler has room.
- Under ER-SDE, Manual predictor-shape knobs (`history_points`, `chebyshev_degree`, `ridge_lambda`, `blend_weight`) use the validated internal bridge configuration and are ignored by the UI values.
- Under ER-SDE, Adaptive keeps warmup/detail-guard/tail/error-tolerance behavior but caps consecutive forecasts at 1; `post_guard_horizon_cap`, `window_size`, and `flex_window` do not control the ER-SDE cadence.
- Removed the old sampler hard-fallback option. Added `warn_unverified_sampler` (default ON), which only prints a short yellow console warning and does not change sampler behavior.
- Cleaned tooltips: removed old 50-step-specific default/recommendation text and added short ER-SDE notes only where behavior changes.
- Removed all ER-SDE test nodes, oracle diagnostics, Smooth3/quadratic/low-frequency experiments, and temporary validation code from the public build.

## 0.7.0
- Added hover tooltips and node descriptions for Manual and Adaptive controls.
- Added `error_tolerance` to Adaptive.
- Added Comfy Registry / ComfyUI-Manager metadata.
- Preserved the v0.6 Adaptive widget order and appended `error_tolerance` so older workflows do not have positional widget values shifted on load.
- Re-applying a Spectrum node replaces an existing Spectrum diffusion wrapper instead of stacking another Spectrum wrapper.

## 0.6.0-public
- Public cleanup release for native MOSS-Image.
- Retained only Manual and Adaptive public nodes.
- Removed older experimental scheduler/cache/profiler branches.
