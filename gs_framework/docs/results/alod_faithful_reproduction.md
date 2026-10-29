# ALoD Faithful Public-Source Reproduction

Date: 2026-08-22

Status: `PASS-ALOD-FAITHFUL-REPRODUCTION`

Campaign: `/home/luis/tese/benchmark_artifacts/alod_faithful_reproduction_001`

## Result

The project reproduced the pinned public ALoD hierarchy-selection path exactly
on the 16 frozen correctness cameras, evaluated the source-faithful no-cache
renderer over the 71 held-out MatrixCity cameras, executed every paper-style
ablation that the release supports without inventing missing behavior, and
measured held-out quality. The milestone is a faithful **public-source
reproduction on the frozen formal proxy**, not an exact reproduction of the
paper's assets, configurations, hardware, or reported numbers.

The authoritative hierarchy remains unchanged:

| Field | Value |
| --- | ---: |
| Asset | `ALOD-REPRO-MC-MIXED-01-FORMAL-TRAIN-009.dhier` |
| SHA-256 | `38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b` |
| Total / hierarchy / skybox rows | 7,072,627 / 6,972,627 / 100,000 |
| SH degree / maximum depth | 1 / 55 |
| Official source | `/home/luis/tese/LoDOfGaussians` at `73f547341547ae021ff036ad74fdf73a1240b450` |

No `ALOD-EXT` component was enabled. In particular, the campaign used no
project prematerialization filter, CUDA materialization filter, 525k or 160k
project cache, or optimized project selector.

## Frozen ladder and fidelity

| Experiment | Classification | Exact scope and limitation |
| --- | --- | --- |
| `ALOD-REPRO-HOST-NOCACHE` | Source-faithful project reproduction | Exact released evaluator HSPT/LoD and four-plane frustum selection; full selected non-skybox properties gathered from host and copied every frame; no cross-frame cache. Exact selection, raster order, and properties are verified, while the benchmark harness and renderer integration are instrumented project adaptations. |
| `ALOD-REPRO-NOFRUSTUM` | Source-defined component ablation | Changes only the released early four-plane upper-hierarchy frustum predicate. It retains no-cache host behavior. This is not the paper Table 3 no-frustum row because that paper row retained the unreleased rendering cache. The released false branch required a non-methodological CUDA boolean-device correction. |
| `ALOD-REPRO-CUT-HSPT` | Exact released cut path | Uses the evaluator's released upper traversal and CUDA SPT cut on the 16 diagnostic cameras. Transfer, property assembly, and rendering are excluded. |
| `ALOD-REPRO-CUT-BFS` | Minimal source-faithful reconstruction | Uses the released full-hierarchy BFS traversal primitive with the same LoD and scale-proxy frustum intent. No official BFS evaluation executable exists. This is an ablation reconstruction, not a project optimization. |
| `ALOD-REPRO-PREVIOUS-REUSE` | Exact released viewer reuse semantics in the project harness | Reuses a previous SPT cut only when `0.3 < previous_distance/current_distance < 1.5`; retains whole compatible prior cuts and loads newly required properties. It has no capacity, multi-frame retention, LRU, or bounded cache. |
| Paper `Render Full` | Unavailable | No released rendering entry point exposes the paper's complete bounded rendering-cache configuration. No substitute was fabricated. |

PAPER-SOURCE-DIVERGENCE: no released rendering entry point exactly implements the paper-reported full bounded rendering-cache configuration.

## Frozen evaluation protocol

The camera fixture contains all 71 held-out evaluation records in ascending
`(dataset_index, dataset_filename)` order. It was frozen before performance
execution and has SHA-256
`4c77176a4f9e72baf806ceb358c8cad800941c1e7fb5036244a003d2e35d4ceb`.
Aerial cameras render at 960 x 540 and street cameras at 500 x 500, both at the
formal divisor-two evaluation resolution.

Each rendering mode ran in three fresh processes. Each process performed 10
ordered warmup frames followed by one measured pass over all 71 cameras. Thus
each table distribution pools 213 synchronized measured frames. Total FPS is
computed from `end_to_end_frame_time_ms`; the raw `frame_time_ms` field is the
raster-only interval and is not reported as total FPS.

The 16 pre-frozen diagnostic cameras remained correctness-only. They were not
chosen or replaced using observed LoD behavior.

## Correctness

The bounded authoritative path passed all 16 cameras with:

- exact ordered selected global IDs against the official oracle;
- exact raster-ID order, including the 100,000-row skybox prefix;
- exact pre-raster means, scales, rotations, opacities, and SH properties; and
- complete selected non-skybox H2D loading on every no-cache frame.

The gates are `PASS-ALOD-CPU-REFERENCE-SELECTION` and
`PASS-ALOD-FORMAL-FAITHFUL-PRE-RASTER`. This establishes exact reproduction of
official hierarchy selection and assembly semantics within the verified
boundary. It does not make the instrumented benchmark an untouched official
executable.

## Rendering performance

| Configuration | End-to-end mean / median / p95 | FPS from mean | Preparation mean | Raster mean | Mean selected hierarchy | Mean H2D rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALOD-REPRO-HOST-NOCACHE` | 186.77 / 184.87 / 207.52 ms | 5.35 | 182.34 ms | 4.15 ms | 3.432M | 3.432M |
| `ALOD-REPRO-NOFRUSTUM` | 188.02 / 186.16 / 210.13 ms | 5.32 | 183.63 ms | 4.07 ms | 3.443M | 3.443M |
| `ALOD-REPRO-PREVIOUS-REUSE` | 134.21 / 131.58 / 160.44 ms | 7.45 | 129.80 ms | 4.03 ms | 3.456M | 0.088M |

Mean component timings for the faithful no-cache path were 0.60 ms camera
preparation, 1.08 ms upper/frustum traversal, 34.22 ms SPT cut, 38.09 ms total
selection, 93.05 ms host gather, 19.27 ms H2D transfer, 3.85 ms pre-raster
assembly, and 4.15 ms rasterization. Preparation and property movement, not
rasterization, dominate the 186.77 ms end-to-end result.

The mean no-cache count decomposition was 6,972,627 total hierarchy rows,
3,432,141 selected hierarchy rows, 100,000 skybox rows, and 3,532,141 raster
input rows. The released packed renderer reported 496,182 mean packed-
projection survivors. These are projection survivors, not final pixel
contributors. The compact HSPT had 335 upper rows, and the upper traversal
produced 20.97 mean candidate SPTs (the logged upper candidates).

## Memory and host transfer

| Configuration | Host properties | Host hierarchy metadata | Persistent GPU Gaussian properties max | Frame-local GPU properties max | PyTorch frame peak allocated / reserved max | Whole-device VRAM max | Process RSS max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALOD-REPRO-HOST-NOCACHE` | 0.606 GiB | 0.236 GiB | 0.009 GiB | 0.605 GiB | 1.281 / 2.387 GiB | 3.963 GiB | 3.810 GiB |
| `ALOD-REPRO-NOFRUSTUM` | 0.606 GiB | 0.236 GiB | 0.009 GiB | 0.605 GiB | 1.284 / 2.363 GiB | 4.005 GiB | 3.816 GiB |
| `ALOD-REPRO-PREVIOUS-REUSE` | 0.606 GiB | 0.236 GiB | 0.307 GiB | 0.421 GiB | 1.284 / 2.961 GiB | 4.548 GiB | 3.922 GiB |

In the project no-cache integration, HSPT/hierarchy selection metadata remains
on CPU (`0` persistent GPU hierarchy-metadata bytes are logged); only the
100,000-row skybox contributes the 9.2 MB persistent GPU Gaussian-property
store. Current selected properties are frame-local. This does bound persistent
GPU property residency relative to the 650.7 MB full host property table, but
it costs a mean 3,432,141 rows / 315.8 MB H2D per frame plus 93.05 ms host
gather and 19.27 ms H2D time. Allocator intermediates make total PyTorch and
whole-device peaks larger than the property tensors alone.

Previous-working-set reuse retained 3,368,739 rows on average (97.47% mean
retention ratio), loaded only 87,570 new rows / 8.06 MB per frame on average,
and saved 52.57 ms end-to-end relative to no-cache. Its mean frame time fell by
28.14%. The trade-off is persistent prior-working-set residency and higher
reserved/whole-device memory. This is released viewer reuse, not paper `Render
Full`, LRU, or a bounded cache.

## Frustum ablation

Across the 213 paired samples, early frustum rejection avoided a mean 10,782
semantic hierarchy rows and H2D rows (median 733, p95 1,746, maximum 704,075),
or only 0.313% of the no-frustum selected hierarchy on average. It eliminated
5.03 candidate SPTs on average and added 0.372 ms of upper-traversal cost. It
avoided essentially no later packed projection work: 0.155 survivors on
average, with median zero and maximum two.

The mean paired savings were 0.36 ms host gather, 0.002 ms H2D, -0.03 ms
pre-raster assembly, -0.08 ms rasterization, and 1.25 ms end-to-end. Frustum-on
was 0.664% faster by pooled mean, but the paired per-camera effect ranged from
-42.60 to +70.30 ms and lies within process/timing variability. Therefore the
paper's strong cached-renderer frustum trend is **not established** on this
formal proxy. The result is consistent with the paper's own observation that
frustum overhead can outweigh savings on easier scenes.

The image-equivalence check found 64/71 images bit-identical. The remaining
seven had a maximum RGB absolute difference of 0.00035188 and mean absolute
difference at most `4.09e-7`, consistent with benign floating-point/order
effects after equivalent off-screen work was removed.

## HSPT versus BFS cut

The cut-only comparison used the 16 diagnostic cameras, one warmup, five
CUDA-synchronized repetitions per camera, and median-per-camera timing.

| Measure | HSPT | BFS | BFS / HSPT |
| --- | ---: | ---: | ---: |
| Mean per-camera median cut time | 36.00 ms | 91.54 ms | 2.543x |
| Median per-camera median cut time | 34.30 ms | 89.93 ms | — |
| Logical GPU metadata | 83,685,344 B | 396,067,112 B | 4.733x |

The HSPT speed and memory trend reproduces: the reconstructed full BFS was
2.54x slower and required 4.73x the logical GPU metadata. The selected sets did
**not** preserve identity. Mean/median/minimum/maximum Jaccard were
0.0970/0.0244/0.00615/0.3073, and BFS selected 3,099,291 fewer rows on average.
This negative structural result is retained, not normalized away.

The difference follows from released-source semantics on this hierarchy. Full
BFS evaluates the released Gaussian scale proxy at every considered hierarchy
node, whereas HSPT performs BFS only on the compact upper tree and then uses
SPT root distance plus ordered min/max-distance metadata for each lower cut.
That approximation does not guarantee selected-set identity. Consequently the
local timing/memory trend supports the paper's HSPT motivation, but the local
cut identity does not reproduce the intended equivalence and the BFS result is
classified as a source-faithful reconstruction rather than an exact official
mode.

## Held-out quality

Quality used the pinned official evaluator selection and released packed
`render_gsplat` path for all 71 frozen held-out cameras. It used the existing
official metric semantics: per-channel PSNR then channel mean, 11x11 Gaussian
SSIM, and bundled calibrated LPIPS 0.1 with ImageNet VGG16. The only LPIPS
correction adds the batch dimension required by the bundled implementation.
The official evaluator's alpha behavior, including its second multiplication
of already-masked ground truth, was preserved and documented.

| Split | Cameras | PSNR mean / median | SSIM mean / median | LPIPS-VGG mean / median |
| --- | ---: | ---: | ---: | ---: |
| All | 71 | 21.495 / 21.183 | 0.6903 / 0.6913 | 0.3972 / 0.4157 |
| Aerial | 26 | 23.255 / 23.775 | 0.7352 / 0.7781 | 0.3088 / 0.3095 |
| Street | 45 | 20.478 / 20.882 | 0.6644 / 0.6718 | 0.4483 / 0.4593 |

The full ranges were 15.547-27.457 PSNR, 0.4726-0.8737 SSIM, and
0.1529-0.6058 LPIPS-VGG. The frozen weight hashes are recorded in campaign
provenance.

## Full-evaluation LoD characterization

The supplementary 71-camera scan did not tune or redefine the evaluation set.
Selected hierarchy count ranged from 2,606,659 to 3,478,844, a 33.46% range
relative to the maximum, with mean 3,432,141. Much of that difference reflects
camera coverage and early visibility rather than a strongly changing depth
cut.

Every camera reached depth 55; every median selected depth was 22; per-camera
mean depth occupied only 22.8484-23.0579; and p90 depth was 26 or 27. Maximum
all-pairs depth-distribution TVD was 0.02358 (mean 0.00602). Adjacent-camera
Jaccard had minimum 0.7745 and mean 0.9813. Camera-to-scene-centre distance was
associated with selected count (Spearman rho -0.608) and mean selected depth
(-0.652), but the depth distributions remained narrowly concentrated.

Thus the broader evaluation set confirms the limitation rather than rescuing
the diagnostic subset:

> The public-source mixed-scale proxy and trained hierarchy reproduce ALoD correctly but do not exhibit the strong view-dependent hierarchy-depth variation expected from the paper's larger/unreleased evaluation setup.

The original 16-camera oracle remains `WEAK-LOD-ORACLE`: a strong correctness
fixture and weak evidence of strong adaptive LoD. No training or
`SPT_root_volume` retuning followed this observation.

## Paper comparison

| Mode | Paper Table 3, four scenes | This campaign | Interpretation |
| --- | --- | ---: | --- |
| Render Full | 48.1 / 68.3 / 47.1 / 83.2 ms | unavailable | Missing released bounded rendering-cache entry point. |
| w/o cache | 119.4 / 127.4 / 92.6 / 222.3 ms | 186.77 ms mean | Closest public evaluator-style proxy; not directly comparable. |
| w/o frustum | 52.3 / 89.3 / 38.3 / 110.2 ms | 188.02 ms mean | Local no-cache source-component ablation; paper row retained unreleased cache. |
| HSPT cut | 31.9 / 45.7 / 31.3 / 36.5 ms | 36.00 ms | Exact released cut path; trend-only comparison. |
| BFS cut | 47.8 / OOM / 40.0 / 53.7 ms | 91.54 ms | Minimal reconstructed source-faithful path; trend-only comparison. |

These numbers are separated deliberately. GPU, dataset, scene scale, training
envelope, hierarchy, camera set, and harness differ, and some paper assets and
configuration code were not released. The campaign reproduces defensible
public-source behavior and qualitative trends, not paper-number equality.

## CityGaussian methodological context

The earlier frozen CityGaussian V1 campaign used the same RTX 5080-class host
and project `gsplat` infrastructure but a different author-provided aerial
representation, 1,440-camera navigation path, 1920 x 1080 resolution, Gaussian
counts, and selection organization. Its released LoD result was 7.84 ms mean
at 4.21M selected Gaussians with all 25.49M level Gaussians GPU-resident; the
faithful ALoD no-cache result here is 186.77 ms at 3.43M selected hierarchy
rows, with the full property table in host RAM and per-frame transfer. This is
methodological context only, not evidence of numerical superiority by either
method. A dedicated frozen cross-method comparison is the next phase.

## Artifact map and reproducibility

| Evidence | Path under campaign root |
| --- | --- |
| Provenance and file/config hashes | `provenance.json` |
| Frozen process schedule | `checks/formal_render_schedule.json` |
| Selection and pre-raster correctness | `checks/selection_oracle_parity.json`, `checks/formal_pre_raster_parity.json` |
| Per-frame formal metrics | `runs/ALOD-REPRO-*/replicate-*/frames.jsonl` |
| Rendering, frustum, reuse tables | `tables/` |
| Memory and H2D summaries | `memory_summary.json`, `h2d_summary.json` |
| HSPT/BFS records and selected IDs | `hspt_bfs/` |
| 71-camera LoD scan | `lod_coverage_71/` |
| Quality metrics | `quality/results/` |

The aggregation entry point is
`scripts/alod/aggregate_faithful_reproduction.py`. Configuration and camera
fixture hashes are sealed in `provenance.json`. All nine formal rendering runs,
the quality run, and the HSPT/BFS run completed with finite output and the
pinned official checkout clean.

## Verification and protected state

| Gate | Result |
| --- | --- |
| Full project unit discovery | 214 tests passed; 2 CUDA-only unit smokes skipped in the sandboxed CPU invocation |
| External ALoD integrations | 14/14 passed, including GPU-backed rendering, residency, materialization, oracle, and topology checks |
| Explicit CityGaussian regressions | 63/63 passed |
| `git diff --check` | passed |
| Pinned official checkout | clean at `73f547341547ae021ff036ad74fdf73a1240b450` |

The protected hierarchy hash remained
`38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b`;
the formal dataset manifest remained
`eaf3cb228234efd92441d28ea56482dcf4676c5a318a6d6beef1020ee27c5f7a`;
and the frozen diagnostic-camera file remained
`35ae984baad7bd5ebba2db2abdb9c3ac9c3f17a93090f00969c8e18c11e771a3`.
The formal oracle was read-only throughout. The CUDA integration run used the
documented CUDA 12.9 compiler and repaired only generated gsplat JIT objects
after system CUDA 12.0 rejected RTX 5080 `sm_120`; no protected source or
experimental result changed.

## Methodological gate

`PASS-ALOD-FAITHFUL-REPRODUCTION` means:

- training and strict formal-hierarchy validation completed;
- the official selection oracle is frozen and project parity is exact;
- source-faithful no-cache rendering and released viewer reuse were evaluated;
- the reproducible no-frustum and HSPT/BFS ablations were executed;
- held-out quality was measured;
- the weak LoD result and HSPT/BFS identity mismatch are explicit; and
- all paper/source divergences are documented.

It does not mean exact paper-number reproduction or a reproduction of paper
`Render Full`.

Next thesis phase:

> compare the frozen faithful CityGaussian and ALoD results, then evaluate ALOD-EXT project-specific optimizations against the faithful ALoD baseline and determine the final thesis contribution set

That extension comparison was not started in this task.
