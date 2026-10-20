# ComfyUI-Spectrum-MOSS

Spectrum-style hidden-state forecasting acceleration for **native ComfyUI MOSS-Image**.

The public build contains only two nodes:

- **Spectrum MOSS-Image (Manual)**
- **Spectrum MOSS-Image (Adaptive)**

Development-only nodes and diagnostics are not included.

## Nodes

### Spectrum MOSS-Image (Manual)

Manual access to the Spectrum scheduler and predictor controls.

For ordinary samplers, the controls behave as the standard Spectrum path. When native ComfyUI **ER-SDE** is detected, the node automatically enables the ER-SDE solver bridge. Scheduling controls such as warmup, tail, window, flex window and `max_consecutive_forecasts` remain user-controlled.

`max_consecutive_forecasts` is only a ceiling. Setting it above 1 does **not** force consecutive forecasts; F-F can occur only when the Manual scheduler has enough room. The ER-SDE bridge validation path used a ceiling of 1. Higher values are intentionally left available for users who want to experiment with the speed/quality tradeoff.

Under ER-SDE, these Manual predictor knobs use the internally validated bridge configuration instead of the UI value:

- `history_points`
- `chebyshev_degree`
- `ridge_lambda`
- `blend_weight`

### Spectrum MOSS-Image (Adaptive)

Error-aware scheduling based on local forecast drift measured at actual refreshes.

Under ER-SDE, Adaptive keeps warmup, detail guard, tail protection and `error_tolerance`, while consecutive forecasts are capped at 1. Because of that ER-SDE cap, these controls do not change the ER-SDE cadence:

- `post_guard_horizon_cap`
- `window_size`
- `flex_window`

They remain available for non-ER-SDE samplers.

## ER-SDE bridge

Native ER-SDE is detected automatically. No separate node is required.

On actual steps, Spectrum records the exact guided denoised output as a solver anchor. On forecast steps, the raw Spectrum denoised observation is not passed directly into ER-SDE. Instead, the bridge uses the two latest clean actual anchors and native ER-SDE lambda coordinates to produce bounded causal dense output.

The bridge keeps native ER-SDE RNG/noise ownership unchanged.

Partial SLG ranges can temporarily make guided branches disagree on actual vs forecast cadence. The bridge detects that mixed state and uses the same clean-anchor dense output rather than passing the mixed raw guided result into ER-SDE.

## Sampler warning

`warn_unverified_sampler` is enabled by default. An unvalidated sampler only prints a short yellow console message:

```text
[Spectrum-MOSS] WARNING: Unverified sampler: <sampler_name>.
```

The warning does **not** change sampler behavior. Disable the node toggle if you do not want the message.

## Validation scope

Regression work has covered two main paths:

- The earlier Spectrum path: native MOSS-Image with Euler / `sgm_uniform` at 50 steps.
- The current native MOSS template path: 25 steps, ER-SDE / `simple`, including SLG full-range and partial-range branch transitions.

ER-SDE validation focused on preserving solver consistency while accelerating transformer calls. Other samplers may run, but their speed/quality behavior may differ.

## How the standard Spectrum path works

On actual evaluations, the node captures the image-token hidden state after the final MOSS transformer block. A small ridge-regularized Chebyshev model predicts future final-block hidden states from recent actual anchors. Forecast calls skip the full transformer stack and run only the normal MOSS output head at the current timestep.

The implementation keeps separate histories when conditioning or DiT replacement patches change, so patched and unpatched branches do not share hidden-state history.

## Installation

1. Remove older experimental copies of this node to avoid duplicate class registrations.
2. Put the `ComfyUI-Spectrum-MOSS` folder in:

   `ComfyUI/custom_nodes/`

3. Restart ComfyUI.

Requires a ComfyUI build with native MOSS-Image support. No additional Python package is required beyond ComfyUI's existing dependencies.

## Node IDs

- `SpectrumMOSSImageManual`
- `SpectrumMOSSImageAdaptive`

Display names:

- `Spectrum MOSS-Image (Manual)`
- `Spectrum MOSS-Image (Adaptive)`

New widgets are appended to preserve positional compatibility with older public workflows where possible.

## Credits

Spectrum method / paper:

- **Spectrum: Spectral Feature Forecasting for Accelerating Diffusion Transformers** — arXiv:2603.01623
- Official implementation: https://github.com/hanjq17/Spectrum

ComfyUI implementation work was also informed by:

- https://github.com/awdqwdasdg/Comfyui-Spectrum-Qwen2.1
- https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3 (solver-space ER-SDE bridge approach)

MOSS-Image support is provided by native ComfyUI.

## License

GPL-3.0-only. See `LICENSE`.

SPDX-License-Identifier: GPL-3.0-only
