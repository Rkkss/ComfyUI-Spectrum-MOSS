# ComfyUI-Spectrum-MOSS

Spectrum-style hidden-state forecasting ComfyUI node for **MOSS-Image**.
vibe-coded

This build contains two nodes:

- **Spectrum MOSS-Image (Manual)** - Manually choose forecast steps after warmup
- **Spectrum MOSS-Image (Adaptive)** - Experimental node. It will try to adapt the forecast steps with forecast error estimation to maintain quality. Speed can be vary

Default (adaptive) scheduling behavior:

1. First (warmup) steps are actual transformer evaluations.
2. The next few steps (detail_guard) allow at most 1 forecast between actual refreshes.
3. After the guard, the scheduler can expand the forecast horizon up to the configured cap (post_guard_cap).
4. The final (tail) steps are actual evaluations.

## Recommended settings
 OpenMOSS Official's suggestion 50 steps/euler+sgm_uniform 


- er_sdm is supported through a solver bridge, but er_sde is a non-deterministic sampler; results may vary. Max_consecutive_forecasts = 1 is the recommended setting for this sampler. Adaptive node will force max forecast =1 when er_sde is detected.
- If you use SkiplayerGuidance, set start & end to 0.000-1.000. 0.000-0.500 will force spectrum relearn mid sigma, thus reducing forecast potential 
- Other schedulers should run. Other samplers may not run well.

Speed gain is about x2 - x2.5 with the default setup

Default settings are on the safer side. If you want more speed, increase max_consecutive_forecasts, and/or reduce warmup_steps/tail_actual_steps. 

Detail_guard allows at most 1 forecast during this window. A higher error_tolerance value allows more "room for error" after guard steps.  Both values may also affect speed.

## Examples 
<p align="center">
  <img src="examples/1.png" width="48%">
  <img src="examples/2.png" width="48%">
   <img src="examples/3.png" width="48%">
    <img src="examples/4.png" width="48%">

</p>

## Credits
Spectrum method / paper:

- **Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration** — arXiv:2603.01623
- Official implementation: https://github.com/hanjq17/Spectrum
- - https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3 (solver-space ER-SDE bridge approach)


## License

GPL-3.0-only. See `LICENSE`.

SPDX-License-Identifier: GPL-3.0-only
