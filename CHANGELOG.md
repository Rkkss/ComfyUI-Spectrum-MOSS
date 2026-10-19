# Changelog
## 0.7.0
- Added hover tooltips and node descriptions for Manual and Adaptive controls.
- Added `error_tolerance` to Adaptive: values above 1.0 tolerate more forecast error for more speed; values below 1.0 are more conservative. `1.0` preserves the previous default thresholds.
- Added Comfy Registry / ComfyUI-Manager metadata (`pyproject.toml`, `node_list.json`, legacy header metadata).
- Preserved the v0.6 Adaptive widget order and appended `error_tolerance` at the end so existing v0.6 workflows do not have positional widget values shifted on load.
- Re-applying a Spectrum node now replaces an existing Spectrum diffusion wrapper on the cloned model instead of stacking a second Spectrum wrapper.


## 0.6.0-public

- Public cleanup release for native MOSS-Image.
- Renamed the two retained nodes to:
  - `Spectrum MOSS-Image (Manual)`
  - `Spectrum MOSS-Image (Adaptive)`
- Renamed node class IDs to `SpectrumMOSSImageManual` and `SpectrumMOSSImageAdaptive`.
- Manual defaults set to warmup 8, tail 2, max consecutive forecasts 2.
- Adaptive node keeps the validated v0.6 phase guard + p95/p99 local-drift controller.
- Removed obsolete v0.4/v0.5 scheduler branches and version-specific internal flags.
- Removed experimental nodes and dead configuration fields from the public build.
- Removed an unused error-sampling constant.
- Added public-facing installation, validation scope, sampler caveats, credits, and licensing notes.
