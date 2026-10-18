# Changelog

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
