# CityGaussian 12-camera performance attribution

Campaign ID: `citygs_12camera_attribution_20260824_222048`  
Status: **complete**  
Evidence class: `CITYGS-12CAM-ATTRIBUTION-*`

## Outcome

B — the isolated backend ratio varies materially with camera. Camera-dependent backend behavior and high-level integration behavior both contribute; neither should be treated as a complete single-cause explanation.

The study uses the exact frozen camera indices `[0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439]`, 1920×1080 resolution, released Nyquist selections, immutable activated Gaussian inputs, and the same direct rasterizer boundary as the preceding backend microbenchmark. Every camera/backend cell has three fresh processes, ten excluded warmups, and 100 CUDA-event samples. Historical per-camera values are means and sample SDs across the three already-completed formal repetitions; neither 1,440-frame campaign was rerun.

## Per-camera direct timing

| Camera | Visible cells | Selected Gaussians | GraphDECO GPU ms | gsplat GPU ms | Ratio | GraphDECO wall ms | gsplat wall ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 1,957,671 | 6.460 ± 0.009 | 4.640 ± 0.001 | 1.392× | 6.946 ± 0.057 | 4.985 ± 0.009 |
| 131 | 19 | 4,545,269 | 7.804 ± 0.007 | 6.471 ± 0.005 | 1.206× | 8.349 ± 0.005 | 6.960 ± 0.041 |
| 262 | 16 | 4,405,467 | 10.609 ± 0.016 | 7.462 ± 0.004 | 1.422× | 11.322 ± 0.029 | 7.999 ± 0.022 |
| 392 | 17 | 3,823,434 | 8.444 ± 0.009 | 6.332 ± 0.014 | 1.334× | 9.004 ± 0.017 | 6.782 ± 0.004 |
| 523 | 13 | 4,256,076 | 9.321 ± 0.010 | 6.866 ± 0.004 | 1.357× | 9.947 ± 0.029 | 7.345 ± 0.005 |
| 654 | 11 | 3,241,519 | 9.278 ± 0.007 | 6.458 ± 0.016 | 1.437× | 9.908 ± 0.038 | 6.916 ± 0.009 |
| 785 | 21 | 5,285,212 | 12.325 ± 0.013 | 8.518 ± 0.008 | 1.447× | 13.146 ± 0.059 | 9.099 ± 0.030 |
| 916 | 12 | 2,965,440 | 6.326 ± 0.023 | 4.094 ± 0.011 | 1.545× | 6.761 ± 0.034 | 4.391 ± 0.010 |
| 1047 | 13 | 2,964,347 | 15.880 ± 0.004 | 7.790 ± 0.004 | 2.039× | 16.923 ± 0.025 | 8.316 ± 0.008 |
| 1177 | 13 | 3,310,269 | 3.918 ± 0.006 | 2.863 ± 0.004 | 1.369× | 4.222 ± 0.022 | 3.080 ± 0.003 |
| 1308 | 23 | 5,902,505 | 11.232 ± 0.017 | 9.529 ± 0.018 | 1.179× | 12.001 ± 0.052 | 10.161 ± 0.047 |
| 1439 | 35 | 6,211,124 | 13.996 ± 0.005 | 12.864 ± 0.004 | 1.088× | 14.946 ± 0.013 | 13.784 ± 0.113 |


Across cameras, the ratio mean/median/min/max is `1.401 / 1.380 / 1.088 / 2.039`. Camera 392 is at approximately the `29.2` percentile of the 12-camera ratios.

## Clean official-wrapper replay and stage probe

The unchanged official wrapper was replayed on only the twelve oracle cameras in three fresh processes under the clean GPU condition. Its mean render/CUDA/end-to-end time across the twelve camera means is `17.134 / 16.037 / 25.717 ms`, corresponding to a `38.89 s⁻¹` end-to-end throughput equivalent for this sparse oracle sample. The historical canonical gsplat values for the same twelve camera indices average `15.597 ms` (`64.11 s⁻¹`), a clean sparse-sample end-to-end ratio of `1.649×`.

The historical official CUDA stalls of roughly 378 ms at camera 785 and 349 ms at camera 1047 did not reproduce: clean replay values are about 24 and 31 ms. Focused stage probes show sub-millisecond screen-space tensor creation, activated attribute views, and rasterizer-object construction. First target-shaped rasterizer calls are about 26 and 32 ms; warmed calls return to about 12 and 16 ms. The original stalls are preserved as raw evidence but cannot be treated as stable wrapper or backend cost. External contention or another nonstationary trajectory/system state remains plausible and cannot be distinguished retrospectively.

## Validation and interpretation boundaries

Every prepared active set matches the official released-LoD oracle and the canonical reproduction records exactly in visible cells, LoD assignments, per-level counts, and total count. Every process re-hashed the immutable input before upload. Cross-backend image parity passes for all cameras; details are in `tables/image_parity.json`.

The twelve serialized canonical `.pt` inputs were a 7.0 GiB temporary execution cache, not source evidence. They were removed after validation; their exact file/tensor hashes, frozen source-asset hashes, full selection metadata, per-process input hashes, output arrays, and reconstruction harnesses remain durable in this package.

The historical high-level/direct differences in `tables/high_level_vs_backend.json` are diagnostic rather than exact accounting identities. Historical frames are single trajectory encounters averaged across three repetitions, while the direct calls repeatedly render a warmed, preassembled input. Positive-radii definitions differ between rasterizers, and correlation does not establish causality.

## Evidence map

- `manifest.json`: frozen sources, binaries, protocol, and evidence relationships.
- `environment/`: idle GPU/window baseline, build records, packages, and harness copies.
- `validation/prepared_input_validation.json`: hashes, camera checks, and exact oracle comparisons.
- `validation/package_validation.json`: completeness and parity gate for 72 processes and 7,200 samples.
- `microbench/`: raw 100-sample JSONL, summaries, images, diagnostics, memory, and provenance.
- `tables/per_camera_backend.json`: direct timings and internal diagnostics.
- `tables/high_level_vs_backend.json`: extracted historical camera values and conservative differences.
- `tables/attribution_summary.json`: ratio distribution, camera-392 placement, and correlations.
- `tables/image_parity.json`: finite, RGB/alpha, PSNR, SSIM, and coverage results.
- `tables/wrapper_stage_probe.json`: synchronized stage probe for cameras 392, 785, and 1047.
- `tables/clean_official_oracle_replay.json`: historical versus clean official-wrapper camera values.
- `figures/`: five thesis-ready diagnostic plots.

No backend, kernel, method, asset, camera, resolution, visibility, or LoD behavior was optimized or changed. The pinned CityGaussian reference checkout and thesis LaTeX were not modified.
