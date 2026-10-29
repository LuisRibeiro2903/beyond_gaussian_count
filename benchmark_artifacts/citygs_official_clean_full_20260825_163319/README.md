# Clean full-trajectory official CityGaussian runtime

Campaign ID: `citygs_official_clean_full_20260825_163319`  
Evidence class: `CITYGS-OFFICIAL-CLEAN-FULL-*`  
Status: **complete**

## Outcome

B — the full-trajectory stalls recur under the controlled condition; the historical campaign remains non-canonical for current performance because the new clean campaign provides the representative three-run estimate.

The clean three-process result is `74.778 ± 15.459 ms` end-to-end (mean and sample SD of run means), or `13.37 FPS`. All stalls remain in the primary aggregate. This supersedes the historical `52.516 ms / 19.04 FPS` value only as the representative official-runtime performance estimate; the historical package remains authoritative semantic-parity and historical-observation evidence.

## Controlled condition and protocol

Before each repetition, Computer Use showed only Codex open: no Twitch/browser video, media player, Spotify, game, or launcher. NVIDIA reported P8 idle clocks, 39–44 °C, 154–889 MiB, 10–11% display utilization, and no compute process. The elevated display utilization was observable with only Codex/the desktop present; no system, driver, power, or clock setting was changed.

Each fresh process loaded the exact three direct SH2 PLY levels, rendered trajectory frame 0 once as an excluded warmup, synchronized/reset peak state, then measured frames 0–1439 in unchanged order at 1920×1080. No camera/workload-shape prewarming was performed.

## Three-run timing

| Repeat | Mean | Median | p95 | p99 | Max | FPS from mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.689 | 23.736 | 503.043 | 985.701 | 1640.566 | 12.71 |
| 2 | 87.905 | 31.226 | 532.837 | 1028.041 | 1518.613 | 11.38 |
| 3 | 57.738 | 22.290 | 313.259 | 672.701 | 1233.390 | 17.32 |


Across-run end-to-end CV is `20.67%`; run means span `57.738–87.905 ms`.

## Stage decomposition

| Stage | Mean of run means (ms) | Between-run SD (ms) |
| --- | ---: | ---: |
| Visibility + LoD selection | 1.487 | 0.035 |
| Active-set assembly | 9.179 | 0.849 |
| Render call | 64.111 | 14.646 |
| CUDA rasterization boundary | 63.938 | 14.617 |
| End-to-end | 74.778 | 15.459 |

CUDA is nested within render call and is not additive. The large variation is inside the synchronized GraphDECO rasterization boundary; selection remains stable, while assembly also spikes in the same difficult trajectory region.

## Stall behavior

The hundreds-of-milliseconds behavior recurs under the controlled condition. Cameras 785 and 1047 exceed 250 ms in all three repetitions. Full threshold frame lists, decompositions, recurring ranges, workload associations, and allocator-reservation transitions are preserved in `tables/stall_summary.json` and `tables/framewise_nonstationarity.json`.

Pairwise frame-time correlations are `0.967`, `0.971`, and `0.978`. Stalls therefore recur at strongly similar camera regions rather than migrating randomly, although their magnitude changes substantially.

PyTorch reserved-memory transitions recur near the stall regions and the official process finishes near whole-device capacity. Reserved counters exceed physical VRAM under WSL/custom-extension accounting, so this is a diagnostic association, not proof that allocator behavior causes the stalls.

## Comparison with canonical gsplat

| Metric | Official clean full | Canonical gsplat | Official / gsplat |
| --- | ---: | ---: | ---: |
| End-to-end (ms) | 74.778 | 15.673 | 4.771× |
| FPS | 13.37 | 63.80 | 4.771× gsplat |
| Selection/assembly or preparation (ms) | 10.667 | 8.126 | 1.313× |
| Render call (ms) | 64.111 | 7.547 | 8.495× |
| CUDA (ms) | 63.938 | 7.480 | 8.547× |

Both retain `25,494,686` resident Gaussians and select the identical mean `4,214,145.394` Gaussians. All 36 repeat-camera semantic checks pass.

## Interpretation

Confirmed recurring costs are synchronized visibility/LoD selection, active-set assembly, camera-dependent raw backend differences, natural first/workload-shape effects, and—only in full sequential official execution—large recurring GraphDECO rasterization-boundary stalls. Static warmed isolated calls do not exhibit those stalls, so the final gap must not be described solely as a generic backend ratio or vague wrapper overhead.

The exact low-level cause of the sequential stalls remains unresolved. They align strongly across runs and with recurrent reservation-state transitions/workload regions, but allocator counters are unreliable and causality was not instrumented. External GPU contention is not supported by a competing process or visible avoidable application in this campaign.

## Evidence map

- `runs/repeat_*/frames.jsonl`: all 4,320 measured frame records.
- `tables/performance_summary.json`: independent-run distributions and across-run statistics.
- `tables/stage_decomposition.json`: synchronized stage means.
- `tables/stall_summary.json`: all thresholds, frame IDs, major records, recurrence, and diagnostics.
- `tables/framewise_nonstationarity.json`: three-run comparison for every trajectory frame.
- `tables/historical_comparison.json`: historical, sparse replay, isolated, and new full-run values.
- `tables/official_vs_gsplat.json`: controlled comparison with the frozen canonical gsplat campaign.
- `validation/selected_set_gate.json`: 36/36 exact oracle-camera semantic checks.
- `validation/package_validation.json`: completeness and system/semantic gates.
- `environment/`: per-run system baselines plus copied harness, package, source,
  patch, and native-build records.
- `figures/`: six thesis-ready figures.

No backend, kernel, method, asset, camera, resolution, visibility, LoD policy, or thesis LaTeX was changed. The pinned CityGaussian reference checkout remains untouched.
