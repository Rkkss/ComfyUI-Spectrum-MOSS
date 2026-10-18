# ComfyUI-Spectrum-MOSS

Spectrum-style hidden-state forecasting for **native ComfyUI MOSS-Image**.

This public build intentionally contains only two nodes:

- **Spectrum MOSS-Image (Manual)**
- **Spectrum MOSS-Image (Adaptive)**

The experimental branches used during development (older adaptive variants, diagnostics, RLS, SEA scheduling, partial-block cache experiments, profiler nodes, etc.) are not included.

## Nodes

### Spectrum MOSS-Image (Manual)

Manual access to the core Spectrum predictor and scheduler.

Safer defaults used by this release:

- `warmup_steps = 8`
- `tail_actual_steps = 2`
- `max_consecutive_forecasts = 2`
- `window_size = 2.0`
- `flex_window = 0.75`
- `history_points = 8`
- `chebyshev_degree = 4`
- `ridge_lambda = 0.1`
- `blend_weight = 0.5`

Use this node when you want direct control over the predictor/scheduler behavior.

### Spectrum MOSS-Image (Adaptive)

Production-oriented adaptive policy derived from the tested v0.6 MOSS-Image configuration.

Exposed controls are intentionally limited to settings that materially affect scheduling:

- `warmup_steps = 8`
- `detail_guard_steps = 8`
- `tail_actual_steps = 2`
- `post_guard_horizon_cap = 6`
- `window_size = 2.0`
- `flex_window = 0.5`
- `cache_device`
- `debug`

The predictor and local-drift detector are pinned internally to the tested values instead of exposing redundant knobs.

Default scheduling behavior:

1. First 8 steps are actual transformer evaluations.
2. The next 8-step detail guard allows at most one forecast between actual refreshes.
3. After the guard, the scheduler can expand the forecast horizon up to the configured cap.
4. A sampled-token p95/p99 error controller can shorten the horizon and request an actual repair refresh when local drift spikes.
5. The final 2 steps are actual evaluations.

Histories are separated by conditioning/DiT patch signature, so SLG-patched and normal branches do not share forecast state.

## How it works

On actual evaluations, the node captures the image-token hidden state after the final MOSS transformer block. A small ridge-regularized Chebyshev model predicts future final-block hidden states from recent actual anchors. Forecast calls skip the full transformer stack and run only the normal MOSS output head at the current timestep.

The node fails closed: unsupported model state, missing sampler schedule information, forecast errors, shape changes, or configured control inputs fall back to a real model evaluation rather than intentionally returning a broken forecast.

## Recommended / validated setup

The main development and regression path was:

- Native ComfyUI `MOSSImage`
- 50 sampling steps
- Euler
- `sgm_uniform`
- CFG/negative branch in normal ComfyUI execution
- Skip Layer Guidance (SLG) tested with a separate patched branch

Other samplers and schedulers may run, but they are **not validated to have the same speed/quality tradeoff**. Samplers with multi-stage, predictor-corrector, stochastic, or solver-history behavior can interact differently with hidden-state forecasting.

## Installation

1. Remove older experimental copies of this node to avoid duplicate class registrations.
2. Put the `ComfyUI-Spectrum-MOSS` folder in:

   `ComfyUI/custom_nodes/`

3. Restart ComfyUI.

Requires a ComfyUI build with native MOSS-Image support.

No additional Python package is required beyond the dependencies already used by ComfyUI.

## Node IDs

Public node IDs are intentionally clean and version-independent:

- `SpectrumMOSSImageManual`
- `SpectrumMOSSImageAdaptive`

Display names:

- `Spectrum MOSS-Image (Manual)`
- `Spectrum MOSS-Image (Adaptive)`

Older private/test workflow IDs such as `SpectrumMOSSImage` and `SpectrumMOSSImageAdaptiveV6` are not kept as aliases in this clean public package.

## Credits

Spectrum method / paper:

- **Spectrum: Spectral Feature Forecasting for Accelerating Diffusion Transformers** — arXiv:2603.01623
- Official implementation: https://github.com/hanjq17/Spectrum

ComfyUI implementation work was also informed by:

- https://github.com/awdqwdasdg/Comfyui-Spectrum-Qwen2.1

MOSS-Image support is provided by native ComfyUI.

## License

GPL-3.0-only. See `LICENSE`.

SPDX-License-Identifier: GPL-3.0-only
