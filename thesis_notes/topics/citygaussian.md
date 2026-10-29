# CityGaussian V1 Evidence Note

This note is the Layer-2 entry point for the CityGaussian dissertation chapter.
It separates final evidence from implementation history and does not duplicate
formal metric tables. Every exact project result must be read from the linked
canonical artifact and field.

## Method and reproduction objective

- Method: CityGaussian V1, including divide-and-conquer training, released
  runtime-cell organization, released visibility selection, and adaptive LoD.
- Paper/source audit:
  [`citygaussian-v1-method-card.md`](../../gs_framework/docs/papers/citygaussian-v1-method-card.md).
- Paper conversion for passage-level verification:
  [`citygaussian-v1-paper.md`](../../gs_framework/docs/papers/sources/citygaussian-v1-paper.md).
- Pinned released source: `/home/luis/tese/CityGaussian`, branch `V1-original`,
  commit `e16c526f78ab01d377b13b3e6862dcc93ddbe953`.
- Runtime-cell source trace:
  [`citygaussian_runtime_cell_construction.md`](../../gs_framework/docs/citygaussian_runtime_cell_construction.md).

The paper is motivated by the difficulty of training large-scale 3DGS and
rendering it in real time across camera scales without excessive computation
and memory. Its response combines divide-and-conquer training with block-wise
LoD so that runtime work is selected at block granularity. This project focuses
on independently reproducing and instrumenting the released viewer-side
runtime behavior, not claiming those source algorithms as project inventions.

The project objective was to reconstruct and validate the inspected released
viewer/runtime behavior incrementally: flat finest asset, official in-memory
cell organization, released visibility selection, and released adaptive LoD.
It was not a reproduction of training, LightGaussian compression, or the
complete paper evaluation. A later controlled campaign executed the released
runtime core with its author-native GraphDECO renderer under the frozen project
cameras; this remains distinct from the paper's evaluation protocol.

## Reproducibility status

| Dimension | Status | Defensible statement |
| --- | --- | --- |
| Implementation reproducibility | Strong for the inspected released runtime paths | Cell assignment, visibility-selected sets, and released Nyquist LoD-selected sets have source traces, tests, frozen fixtures, and passing formal oracle checks. |
| Experimental reproducibility | Partial | Released MatrixCity assets, the direct three-level PLY path, the released GraphDECO backend, and a frozen project camera/timing protocol were exercised. Training, the paper camera path, VQ, and the complete author evaluation regime were not recreated. |
| Result reproducibility | Partial | The packages support local source-behavior, controlled-transition, official-runtime parity, and performance-attribution findings. They do not reproduce the paper's FPS, VRAM, PSNR, SSIM, or LPIPS results. |

Classification: reproduction of inspected released-code runtime behavior. The
canonical stage campaign adapts it to the project `gsplat` runtime; the later
official-parity campaign executes the released GraphDECO core separately.

## Evidence boundary

Keep these evidence classes distinct in the chapter:

| Evidence class | Authority | What it may support |
| --- | --- | --- |
| Paper claim | CityGaussian publication, checked passage by passage | Claimed training pipeline, fixed-distance LoD, paper datasets/protocol, and paper-reported results. |
| Released-source behavior | Pinned `V1-original` source and repository README | Runtime cell construction, projected-corner visibility, Nyquist selector, assembly, residency assumptions, and paper/source divergence. |
| Project reproduction | Current `gs_framework` CityGaussian paths, fixtures, checks, and formal package | Selection parity and locally measured behavior under the frozen project backend/protocol. |
| Project measurement | Raw formal JSONL and validated summaries in the campaign package | Exact timing, workload, memory, and sampled pairwise image-difference results. |
| Official-runtime parity | Isolated clone of the pinned source plus the `CITYGS-OFFICIAL-PARITY-*` package | Direct-PLY execution of the released core, exact selected-set parity, GraphDECO-versus-gsplat image similarity, and high-level call-boundary performance comparison. |
| Backend microbenchmark | Preassembled same-input tensors plus pinned GraphDECO and gsplat public rasterizer calls | Isolated fixed-camera backend timing, input/image parity, build verification, and qualification of the earlier high-level attribution. |
| Clean full official runtime | The `CITYGS-OFFICIAL-CLEAN-FULL-*` package under a recorded no-competing-compute/no-avoidable-media condition | Representative full-trajectory official-runtime performance under the project protocol, recurrence/nonstationarity of stalls, and controlled comparison with the frozen canonical gsplat campaign. |
| Project extension or unrelated baseline | `BlockManager`, custom partitions, host-residency/cache work, ALoD, FlashGS plans | Separate comparisons only; never evidence for faithful CityGaussian reproduction. |

## Final numerical evidence

Campaign ID: `citygs_canonical_20260824_135615`.

- Package:
  [`citygs_canonical_20260824_135615`](../../benchmark_artifacts/citygs_canonical_20260824_135615/README.md).
- Identity, commands, environment, run classes, and declared limitations:
  [`manifest.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/manifest.json).
- Per-run resolved configurations, summaries, and raw frames:
  `benchmark_artifacts/citygs_canonical_20260824_135615/runs/<CONFIG>/repeat_<N>/`.
- Aggregate configuration and timing fields:
  [`config_summary.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/tables/config_summary.json).
- Selected/resident workload fields:
  [`selected_workload_summary.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/tables/selected_workload_summary.json).
- Allocator, NVML, and process-memory fields:
  [`memory_summary.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/tables/memory_summary.json).
- Corrected controlled transitions:
  [`controlled_transitions.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/tables/controlled_transitions.json).
- Durable validation, including same-asset image equivalence:
  [`validation/`](../../benchmark_artifacts/citygs_canonical_20260824_135615/validation/).

Every configuration has three fresh-process repetitions of 1,440 measured
frames after one excluded deterministic warmup frame. The package validates
30,240 measured records. Source commit
`cfc2136459dd63cc47dc3acfde55a3d72a4d8bba` was clean; the manifest hashes the
trajectory, four assets, and seven configuration files. Camera order remained
fixed from frame 0 through 1439, while configuration order was rotated across
the three repetitions as recorded in the manifest.

Frozen input SHA-256 identities, copied from the canonical manifest for
discovery, are:

- fused asset: `06e76f13e2805396713f587da05e991048be9ca019f63121e11772fc2a1f8357`;
- finest released LoD asset: `70e0401c629f98f13bb55f8f6dc8579a4ba5dffe686c638b090565aabcd974bb`;
- middle released LoD asset: `57904c2602a077f5593d236a0c72203b89456cf700060774c1184d307163c77a`;
- coarsest released LoD asset: `7142e855e4367b352269a1de9e6b7f3a1ab98909099f6ad2ba4b272489fd0c0d`;
- frozen trajectory: `c20ea29cfacd8331a808eec08840cf814d908de7e7e2f8715763b89d5e1605b9`.

The seven configuration-file hashes remain indexed only in
[`manifest.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/manifest.json)
to avoid maintaining a second complete hash table.

Canonical timing names are `preparation_time_ms`, `render_call_time_ms`,
`cuda_rasterization_time_ms`, and `end_to_end_frame_time_ms`. Primary FPS is
derived from mean end-to-end time. The July 30 package and its generic
frame-time memo are preserved as superseded evidence; they must not be used for
overall runtime claims.

## Official implementation parity evidence

Campaign ID: `citygs_official_parity_20260824_191215`.

- Package and source-call-chain audit:
  [`README.md`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/README.md).
- Identity, pinned commits, asset/trajectory hashes, environment, run order,
  timing semantics, and patch hash:
  [`manifest.json`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/manifest.json).
- Camera and exact selected-set gates:
  [`camera_parity.json`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/validation/camera_parity.json) and
  [`selected_set_parity.json`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/validation/selected_set_parity.json).
- Three-run official aggregates and same-asset flat backend control:
  [`official_config_summary.json`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/tables/official_config_summary.json) and
  [`flat_backend_comparison.json`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/tables/flat_backend_comparison.json).
- Full released-LoD timing attribution:
  [`released_lod_comparison.json`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/tables/released_lod_comparison.json).
- Twelve-camera cross-backend RGB/alpha similarity:
  [`cross_backend_image_comparison.json`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/comparisons/cross_backend_image_comparison.json).
- Complete instrumentation and CUDA 12.9 compatibility diff:
  [`instrumentation.patch`](../../benchmark_artifacts/citygs_official_parity_20260824_191215/environment/instrumentation.patch).

The released core executed through `GaussianModelLOD.load_ply`,
`BlockedGaussian`, `render_lod`, `in_frustum`, the released Nyquist selector,
official active-set concatenation, and the original GraphDECO rasterizer. It
matched all 12 frozen cameras exactly for visible cells, selected LoD
indices/labels, per-cell/per-level/total selected counts, nearest cell, and
fallback state. Each formal configuration used three fresh processes and 1,440
measured frames after one excluded frame-0 warmup; package validation covers
8,640 formal records.

The official and project LoD runs have identical resident count and identical
mean selected workload. The authoritative timing table records the official
runtime at `52.516 ± 3.337 ms` end-to-end (`19.04 FPS`) and the canonical
project run at `15.673 ± 0.738 ms` (`63.80 FPS`). The measured end-to-end gap is
`36.843 ms`: `35.452 ms` appears in the render-call difference and `1.391 ms`
in the preparation difference. The same-asset finest-flat control independently
records a large GraphDECO-versus-gsplat high-level call-boundary gap. The later
isolated backend microbenchmark below shows that this does **not** support a
strong claim that the pinned rasterizer itself accounts for most of the gap.
The official-parity timing fields remain valid as observations from their
recorded processes and boundaries, but their performance attribution is
superseded by the isolated, sparse-replay, and clean full-trajectory evidence
below. The flat GraphDECO
CUDA boundary included substantial per-frame input preparation inside
`render()`, and two very large official-LoD oracle-camera stalls from the full
trajectory did not reproduce under the later sparse clean replay but did recur
in all three clean full sequential runs.

Cross-backend images are finite and visually consistent. Across the 12 frozen
cameras, the authoritative comparison records mean raw RGB absolute difference
`1.9315e-5`, mean clipped SSIM `0.9999968`, and mean coverage-mask Jaccard
`0.999999959`. These are implementation-similarity metrics, not reconstruction
quality against ground truth.

## Isolated backend microbenchmark evidence

Campaign ID: `citygs_backend_microbench_20260824_213420`.

- Package, backend-input audit, exact protocol, interpretation, and limitations:
  [`README.md`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/README.md).
- Identity, source/binary hashes, cases, and timing boundary:
  [`manifest.json`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/manifest.json).
- Three-process timing summaries and ratios:
  [`backend_timing.json`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/tables/backend_timing.json) and
  [`relative_speed.json`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/tables/relative_speed.json).
- Deterministic tensor/camera and image gate:
  [`input_output_parity.json`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/tables/input_output_parity.json) and
  [`package_validation.json`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/validation/package_validation.json).
- Reconciliation with the earlier camera-392 timing boundaries:
  [`prior_boundary_comparison.json`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/tables/prior_boundary_comparison.json).
- Native `sm_120` and release-build verification:
  [`graphdeco_build.txt`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/environment/graphdeco_build.txt) and
  [`gsplat_build.txt`](../../benchmark_artifacts/citygs_backend_microbench_20260824_213420/environment/gsplat_build.txt).

The focused experiment used frozen camera 392 for both principal cases. The
finest-flat input contains exactly `11,694,810` Gaussians. The preassembled
released-LoD input contains exactly `3,823,434` Gaussians with the same 17
visible cells and per-level counts `[2,286,715, 176,751, 1,359,968]` already
validated by official parity. Both backends re-hashed identical activated
means, normalized quaternions, scales, opacities, SH2 coefficients, and camera
tensors before upload. Loading, selection, gathering, concatenation,
activation, camera/static setup, warmup, diagnostics, and serialization were
outside timing. Each backend/case used three fresh processes, ten excluded
warmups, and 100 synchronized CUDA-event samples.

GraphDECO versus gsplat direct GPU time is `10.769 ± 0.052 ms` versus
`9.738 ± 0.267 ms` for finest flat (`1.106x`) and `8.989 ± 0.229 ms` versus
`6.652 ± 0.021 ms` for representative released LoD (`1.351x`). These are
backend throughput measurements, not end-to-end viewer FPS. Image equivalence
passes in both cases: mean raw RGB absolute difference is approximately
`9.13e-6`, clipped SSIM is at least `0.99999809`, and coverage Jaccard is
`1.0`.

This is prespecified Outcome B: the earlier approximately 4--5x flat gap does
not survive direct fixed-camera isolation. The previous strong
backend-dominance interpretation is therefore superseded. Source and
camera-392 evidence identify one concrete contaminant: the old official flat
CUDA event began before `render()` and included full screen-space tensor
initialization plus per-frame opacity/scale/rotation activation and SH feature
concatenation. Its camera-392 mean was `23.751 ms`, compared with `10.769 ms`
for the preactivated direct GraphDECO call. The corresponding official LoD
event was already placed immediately around `GaussianRasterizer` after active
assembly and is much closer (`10.507 ms` earlier versus `8.989 ms` direct).

The defensible conclusion is narrower: renderer call-path and timing-boundary
differences dominate much of the official-parity gap, but the isolated pinned
rasterizers differ modestly at this fixed camera. Camera-dependent projection,
tile-overlap, duplication/sort workload, allocator history, and the moving
trajectory remain alternative explanations for the older trajectory average.
GraphDECO does not expose tile/sort counts through its pinned public API, so no
kernel-efficiency claim is made. This is a controlled project measurement, not
a CityGaussian paper-result reproduction.

## Twelve-camera performance-attribution evidence

Campaign ID: `citygs_12camera_attribution_20260824_222048`.

- Complete protocol, result table, clean replay, and interpretation:
  [`README.md`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/README.md).
- Frozen identities, binary hashes, clean GPU baseline, and evidence relations:
  [`manifest.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/manifest.json).
- Direct timing, historical high-level extraction, and ratio/correlation summary:
  [`per_camera_backend.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/tables/per_camera_backend.json),
  [`high_level_vs_backend.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/tables/high_level_vs_backend.json), and
  [`attribution_summary.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/tables/attribution_summary.json).
- Clean official-wrapper replay and focused stage probe:
  [`clean_official_oracle_replay.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/tables/clean_official_oracle_replay.json) and
  [`wrapper_stage_probe.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/tables/wrapper_stage_probe.json).
- Exact input/oracle and image gates:
  [`prepared_input_validation.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/validation/prepared_input_validation.json),
  [`image_parity.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/tables/image_parity.json), and
  [`package_validation.json`](../../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/validation/package_validation.json).

The direct study covers the frozen camera indices `[0, 131, 262, 392, 523,
654, 785, 916, 1047, 1177, 1308, 1439]`. Every prepared active set exactly
matches the official and canonical released-LoD records. Each camera/backend
cell uses three fresh processes, ten excluded warmups, and 100 CUDA-event
samples, for 72 processes and 7,200 validated samples. All twelve image pairs
pass the finite, PSNR, SSIM, and coverage gate.

The GraphDECO/gsplat direct GPU-time ratio has mean `1.401`, median `1.380`,
minimum `1.088`, and maximum `2.039`. Ten of twelve cameras are at or below
`1.5x`; camera 916 is `1.545x`, while camera 1047 reaches `2.039x`. Camera 392
is `1.334x` in the clean campaign, approximately the 29th percentile and only
`0.047` below the median, so it is broadly representative of the central
distribution but misses the difficult camera-1047 tail. The ratio is
camera-dependent and is not explained by selected count alone: in this small
sample it is negatively associated with count, while gsplat time itself rises
strongly with count. No causal interpretation is assigned, and the two
rasterizers' positive-radii counts are not definitionally interchangeable.

The unchanged official wrapper was also replayed on only these twelve cameras
in three fresh processes after confirming no Twitch/video/browser GPU workload
was active. Its mean render/CUDA/end-to-end time across the twelve camera means
is `17.134 / 16.037 / 25.717 ms`, or `38.89 s^-1` end-to-end throughput
equivalent. The historical canonical gsplat values on the same twelve indices
average `15.597 ms` (`64.11 s^-1`), giving a clean sparse-sample end-to-end
ratio of `1.649x`. These throughput equivalents are not a new full-trajectory
FPS campaign.

The historical official CUDA stalls of approximately `378 ms` at camera 785
and `349 ms` at camera 1047 did not reproduce in this sparse clean replay; the
replay values are approximately `24 ms` and `31 ms`. Focused instrumentation
shows sub-millisecond screen-space tensor creation, activated-attribute views,
and rasterizer-object construction. First target-shaped rasterizer calls are
approximately `26 ms` and `32 ms`, while warmed calls settle near `12 ms` and
`16 ms`. Ordinary target-shape/allocation warmup therefore explains part of
the clean high-level/direct difference, but not the older 350--380 ms stalls.
At this stage, external GPU contention or another nonstationary
trajectory/system state could not be distinguished retrospectively. The later
clean full-trajectory campaign below resolves the key ambiguity: the large
stalls recur without a competing compute process, but only after preserving
the complete sequential camera history.

This sparse study is prespecified Outcome B for its own attribution question.
It establishes a modest but camera-dependent warmed raw-backend difference and
ordinary first-shape costs, but its `1.649x` sparse end-to-end ratio is not a
full-trajectory estimate. Its earlier interpretation that the large stalls
were non-reproduced is superseded by the clean sequential campaign below; the
restriction against a single-cause renderer-dominance claim remains valid.

## Clean full-trajectory official-runtime evidence

Campaign ID: `citygs_official_clean_full_20260825_163319`.

- Complete protocol, result, stall interpretation, and evidence map:
  [`README.md`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/README.md).
- Frozen identities, environment snapshots, source/binary identities, and
  supersession boundaries:
  [`manifest.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/manifest.json).
- Independent-run statistics and synchronized stage decomposition:
  [`performance_summary.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/tables/performance_summary.json) and
  [`stage_decomposition.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/tables/stage_decomposition.json).
- Full threshold lists, outlier decompositions, cross-run recurrence, and
  framewise nonstationarity:
  [`stall_summary.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/tables/stall_summary.json) and
  [`framewise_nonstationarity.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/tables/framewise_nonstationarity.json).
- Historical/sparse/isolated reconciliation and frozen canonical comparison:
  [`historical_comparison.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/tables/historical_comparison.json) and
  [`official_vs_gsplat.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/tables/official_vs_gsplat.json).
- Full-run oracle and package gates:
  [`selected_set_gate.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/validation/selected_set_gate.json) and
  [`package_validation.json`](../../benchmark_artifacts/citygs_official_clean_full_20260825_163319/validation/package_validation.json).

This campaign reused the unchanged official PLY LoD harness, compiled
GraphDECO backend, three direct SH2 PLY levels, validated camera conversion,
1920x1080 trajectory, and timing semantics. Each of three fresh processes used
one excluded frame-0 warmup followed by frames 0--1439 in normal order. Before
each run only Codex/the desktop was visible; NVIDIA recorded P8, 39--44 degrees
Celsius, 154--889 MiB used, 10--11% display utilization, and no compute
process. No system performance setting was changed.

The representative official result is `74.778 ± 15.459 ms` end-to-end (mean
and sample standard deviation of three run means), or `13.37 FPS`, with a
between-run CV of `20.67%`. The three run means are `78.689`, `87.905`, and
`57.738 ms`; their maxima are `1640.566`, `1518.613`, and `1233.390 ms`.
Selection is stable at `1.487 ms`, assembly averages `9.179 ms`, and the
synchronized render/CUDA boundaries average `64.111/63.938 ms`. Thus most
variation is directly located inside the GraphDECO rasterization boundary,
with smaller recurring assembly spikes; it is not unlocalized generic wrapper
overhead.

This is Outcome B: hundreds-of-milliseconds stalls recur under the controlled
full sequential condition. Ninety-five frame indices exceed `250 ms` in all
three runs, concentrated in ranges `783--801`, `809--815`, `836--859`, frame
`895`, `975--977`, `979--1008`, and `1045--1055`. Pairwise per-frame timing
correlations are `0.967--0.978`. Cameras 785 and 1047 exceed `250 ms` in every
new run, even though their sparse replay and warmed isolated calls did not.
The evidence therefore supports a trajectory/state-dependent runtime effect,
not historical external contamination as the sole explanation.

Against frozen canonical gsplat `CG-LOD-RELEASE`, the clean official run is
`4.771x` slower end-to-end and in FPS; preparation is `1.313x`, render call is
`8.495x`, and CUDA is `8.547x`. Both runtimes retain `25,494,686` Gaussians and
select the identical mean `4,214,145.394`; all 36 repeat-camera semantic checks
pass. Static warmed direct backend ratios of `1.088--2.039x` therefore cannot
explain the full sequential gap.

The exact low-level stall cause remains unresolved. Recurrent reservation
state transitions and near-capacity whole-device usage are associated with the
same regions, but WSL/custom-extension PyTorch reserved counters exceed
physical VRAM and are not representation-neutral. The package makes no
allocator-causality, kernel-efficiency, or external-contention claim. This
campaign supersedes the historical `19.04 FPS` value only as the representative
official-runtime performance estimate; the historical package remains valid
semantic-parity and historical-observation evidence.

## Final correctness evidence

| Gate | Final canonical artifact | Status and boundary |
| --- | --- | --- |
| Finest visibility | [`finest_visibility_oracle.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/validation/finest_visibility_oracle.json) | Passed for all 12 frozen cameras. |
| Fused visibility | [`fused_visibility_oracle.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/validation/fused_visibility_oracle.json) | Passed for all 12 frozen cameras. |
| Released adaptive LoD | [`release_lod_oracle.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/validation/release_lod_oracle.json) | Passed for all 12 frozen cameras, including selected cell/level/count and fallback state. |
| Same-asset flat/cell rendering | [`finest_flat_vs_all_cells/`](../../benchmark_artifacts/citygs_canonical_20260824_135615/validation/finest_flat_vs_all_cells/) and [`fused_flat_vs_all_cells/`](../../benchmark_artifacts/citygs_canonical_20260824_135615/validation/fused_flat_vs_all_cells/) | Twelve-camera equivalence checks; finest has tiny RGB drift and zero alpha drift, fused is exact. |
| Package/schema | [`package_validation.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/validation/package_validation.json) | Passed for seven configurations, three repetitions, and 30,240 measured frames. |

## Implemented released runtime behavior

Architectural claims for this campaign refer to the `gs_framework` repository
at clean commit `cfc2136459dd63cc47dc3acfde55a3d72a4d8bba`, not implicitly to
the mutable primary working tree. At the 2026-08-24 evidence audit, the primary
working tree was on branch `codex/alod-reproduction-audit` at later commit
`154c3a12db93495956a985bd809b5d77327c0b0f` with substantial uncommitted ALoD
and benchmark work, and its benchmark file already differed materially from
the campaign source. The clean worktree at
`/home/luis/tese/.citygs_canonical_worktree` preserves the campaign commit and
the later memo-only commit `0c977626885384ad8b2e73ba83e8d57a08d2188e`.

- At the frozen commit, `scene_runtime/citygaussian.py` reconstructs the official
  contraction, x-fastest 6x6x1 cell assignment, half-open intervals, MAD bounds,
  projected-corner visibility with nearest-cell fallback, and released strict
  Nyquist selection.
- `citygaussian_assets.py` audits the author-provided fused and LoD
  assets.
- `scene_runtime/loader.py`, `renderer.py`, and `types.py` provide
  shared loading, rendering, and instrumentation.
- `benchmarks/benchmark_real_scene.py` records preparation,
  render-call, CUDA-event rasterization, workload, and memory fields.
- `benchmarks/configs/cg_*.json` and
  `flat_citygs_full.json` define the formal configuration roles.
- `scripts/extract_citygaussian_*_oracle.py` and
  `validate_citygaussian_*_oracle.py` provide source-sliced oracle extraction
  and validation.

The inspected official call chain loads each whole-scene LoD model, constructs
`BlockedGaussian` cells in memory, selects visible cells and one LoD per cell,
concatenates selected tensors, and invokes the GraphDECO rasterizer. No
pre-generated chunk directory or `lod-meta.json` participates in this V1 Python
runtime path.

## Controlled experiment graph

Final valid parent/child relations are frozen in
[`EXPERIMENT_LADDER.md`](../../gs_framework/docs/EXPERIMENT_LADDER.md),
[`BENCHMARK_PROTOCOL.md`](../../gs_framework/docs/BENCHMARK_PROTOCOL.md), and
the formal resolved configurations.

| Branch | Transition | Valid interpretation |
| --- | --- | --- |
| Finest released representation | `CG-FINEST-FLAT -> CG-FIXED-FINEST-ALL` | Same asset; adds official in-memory cell organization. Use end-to-end timing to include assembly. |
| Finest released representation | `CG-FIXED-FINEST-ALL -> CG-FIXED-FINEST-VISIBILITY` | Same asset/cells; adds released visibility selection. |
| Released LoD family | `CG-FIXED-FINEST-VISIBILITY -> CG-LOD-RELEASE` | Adds per-visible-cell released Nyquist LoD and middle/coarsest resident assets. Controlled runtime transition with a residency/representation caveat. |
| Fused no-LoD representation | `FLAT-CITYGS-FULL -> CG-FUSED-CELLS-ALL` | Same fused SH3 asset; adds official in-memory cell organization. |
| Fused no-LoD representation | `CG-FUSED-CELLS-ALL -> CG-FUSED-VISIBILITY` | Same fused SH3 asset/cells; adds released visibility selection. |

`FLAT-CITYGS-FULL -> CG-LOD-RELEASE` is not an isolated-method transition:
representation, SH degree, total Gaussian count, visibility, and LoD all change.

## Corrected campaign findings

These are qualitative indexes to the canonical transition and configuration
tables, not a second numerical source:

- Same-asset cell organization leaves renderer-call and CUDA-rasterization cost
  effectively unchanged in both finest SH2 and fused SH3 branches, but the
  frozen campaign's gsplat integration pays substantial per-frame active-set
  preparation/concatenation cost. Therefore the supported claim is
  **negligible renderer-call overhead**, not negligible overall overhead.
- Released visibility reduces selected Gaussians and improves renderer-call and
  end-to-end time on the finest SH2 branch.
- On the fused SH3 branch, visibility also reduces selected Gaussians, but
  renderer-call/CUDA and end-to-end time worsen. Selected count alone therefore
  does not predict performance in this unpacked gsplat integration.
- Released adaptive LoD improves renderer-call and end-to-end time relative to
  finest visibility while increasing resident and allocated memory because all
  three LoD assets remain GPU resident.
- Across-run variation is recorded from three independent run means; the large
  directional effects above persist, while the tiny same-asset renderer deltas
  are treated as negligible rather than as speedups or regressions.

Authoritative values are in
[`config_summary.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/tables/config_summary.json),
[`controlled_transitions.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/tables/controlled_transitions.json),
and [`memory_summary.json`](../../benchmark_artifacts/citygs_canonical_20260824_135615/tables/memory_summary.json).

## Paper/source/project divergences

1. The paper describes fixed distance intervals for MatrixCity; the inspected
   released V1 README/source uses a Nyquist rule. `CG-LOD-RELEASE` reproduces
   released source, not the paper policy. The provisional paper-policy fixture
   is not an implemented or validated paper reproduction.
2. The canonical project campaign uses `gsplat`, while the separate official
   parity campaign executes the released GraphDECO backend. Their controlled
   comparison is local performance and image-similarity evidence, not paper
   protocol or paper-number reproduction.
3. The formal campaign loads PLYs and does not reproduce the released VQ loading
   path or LightGaussian pruning/distillation/vector quantization.
4. All three LoD levels remain GPU resident. Reduced selected workload is not
   evidence of streaming, host residency, or a GPU cache.
5. The fused SH3 branch and compressed SH2 LoD family are different
   author-provided representations with unverified exact generation lineage.
6. The project camera path is a Viser/nerfview trajectory with a deterministic
   conversion, not the paper's test-camera protocol.
7. The pinned reference checkout remains untouched. Exact submodules were
   initialized only in an isolated clone. CUDA 12.9 required two include-only
   compatibility patches (`<cstdint>` and `<cfloat>`), recorded separately from
   runtime instrumentation; neither changes algorithm semantics.

## Evidence limitations and verification register

### Remaining required limitations

- No ground-truth PSNR/SSIM/LPIPS campaign exists. The recorded PSNR values are
  pairwise parent/child image differences on 12 cameras, not reconstruction
  quality against MatrixCity ground truth.
- The official parity campaign uses the frozen project trajectory and timing
  protocol, not the paper's cameras or complete author evaluation regime.
- The isolated backend evidence now covers all 12 frozen oracle cameras but not
  every frame of the trajectory. It establishes a camera-dependent direct
  ratio and supersedes strong rasterizer-dominance attribution, but does not
  turn sparse warmed calls or the clean oracle replay into full-trajectory FPS.
- The clean full-trajectory campaign reproduces the original large stall
  regions across three processes, while the clean sparse replay does not. The
  exact sequence-state/memory mechanism remains unresolved; the new full-run
  estimate supersedes the original 19 FPS value for representative official
  performance, while preserving it as historical evidence.
- No paper fixed-distance LoD evaluation exists.
- No formal interactive-viewer, temporal popping, or motion-preview-quality
  campaign exists.
- The official-vs-gsplat panel is a backend-similarity figure, not a render/GT
  reconstruction-quality panel.
- NVML process accounting remains unreliable in WSL. PyTorch allocated stages
  are primary; raw reserved counters also need caution because visibility runs
  report values at or above physical device capacity.
- Earlier oracle extraction used minimal source slices; the later official
  executable parity gate now independently validates the selected sets, while
  training and the full interactive/paper stack remain outside scope.

### Optional verification, only if the chapter needs stronger claims

- Interactively exercise `viewer.py` only if the chapter needs UI/event-loop,
  moving-resolution, or presentation behavior; controlled headless execution
  of the released runtime core is complete.
- Implement and validate the paper fixed-distance policy with confirmed units.
- Reproduce VQ loading and/or LightGaussian asset generation.
- Run a fixed ground-truth quality campaign and a temporal LoD artifact study.

## Implementation history, without duplicate numerical authority

| Stage | Historical value | Final status / supersession |
| --- | --- | --- |
| Spec 001 asset validation | Established official versus project-generated asset roles and local LoD availability. | Retain the spec/method-card audit; formal resolved configs are the final asset-use record. |
| Spec 002 fixed-finest CUDA validation | Established the flat and all-cells parents and resolved the RTX 5080 CUDA 12.9 JIT environment. | `/tmp/cg_*` outputs are missing; all short-run numbers require verification. The formal run/comparison supersedes them. |
| Spec 003 visibility implementation | Added released visibility, source-sliced oracles, camera conversion, and fused/finest controlled branches. | Commit `021b054` preserves the tracked visibility milestone. Missing `/tmp/citygs003_*` timing outputs are non-canonical; formal checks/runs supersede them. |
| Spec 004 oracle-only stage | Created independent released-policy and provisional paper-policy fixtures before runtime LoD existed. | Retain as validation-method history. Statements that runtime LoD was unimplemented are superseded. |
| Spec 004 released-LoD implementation | Added `CG-LOD-RELEASE`, instrumentation, and 12-camera validation. | Missing `/tmp/citygs004_*` results are non-canonical; formal LoD check/run/comparison supersede them. |
| Interactive viewer exposure | Exposed the runtime mode and identified nerfview moving-preview resolution/JPEG behavior. | Qualitative debugging only; no registered viewer result. |
| First formal freeze | Preserved the full July 30 package and memo outside `/tmp`. | Superseded for performance claims by the corrected repeated campaign because generic frame time was renderer-only and provenance/warmup/repetition evidence was incomplete. |
| First corrected-campaign attempt | `/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_132000` contains 21 completed raw run summaries from clean commit `7583985...`, but aggregation stopped because flat-scene `selected_gaussian_count` was non-finite. | Preserved as failed evidence only (`manifest.status=initialized`, no package validation). Post-hoc normalization/reuse was rejected because residency/activity repair was a pre-run requirement. Flat metrics were repaired to report resident/selected/active counts explicitly, aggregation validation now rejects non-finite counts, and all 21 processes were rerun from the corrected commit. |
| Corrected canonical campaign | Repaired timing names, metadata, provenance, warmup, independent repetition, hashing, validation, tables, figures, and memo. | `citygs_canonical_20260824_135615` is the canonical CityGaussian numerical campaign; commit `cfc2136...`, clean worktree. |
| Official-runtime PLY parity campaign | Executed the pinned released core and GraphDECO backend under the canonical project assets, trajectory, resolution, warmup, repetition, and timing semantics. | `citygs_official_parity_20260824_191215` is the separate `CITYGS-OFFICIAL-PARITY-*` evidence class; exact 12-camera semantic parity and six formal processes passed. |
| Isolated same-input backend microbenchmark | Preassembled the exact finest-flat and camera-392 released-LoD populations once, then called pinned GraphDECO and gsplat rasterizers directly with three fresh processes and 100 samples each. | `citygs_backend_microbench_20260824_213420` passed input/image/build gates. The old 4--5x flat gap shrank to 1.106x and the selected-set gap to 1.351x; strong rasterizer-dominance attribution is superseded. |
| Twelve-camera attribution and clean replay | Extended the immutable same-input direct comparison to all frozen oracle cameras, extracted three-repeat historical camera values, and replayed the unchanged official wrapper without avoidable video/browser GPU workload. | `citygs_12camera_attribution_20260824_222048` passed 72-process/7,200-sample input and image gates. Direct ratios span 1.088--2.039x; the clean sparse official/canonical end-to-end ratio is 1.649x. The old 350--380 ms official stalls did not reproduce in sparse replay; the later clean full campaign supersedes the inference that they were non-representative contamination. |
| Clean full official runtime | Reused the unchanged official runtime for three fresh full 1,440-frame processes under recorded clean-system conditions, preserving natural sequential first-occurrence and state effects. | `citygs_official_clean_full_20260825_163319` passes 4,320-frame and 36/36 semantic gates. Large stalls recur at strongly aligned trajectory regions; `74.778 ± 15.459 ms` / `13.37 FPS` supersedes the old 19 FPS value as representative official performance, while the cause remains unresolved. |
| One-frame fused feasibility | Demonstrated that the fused asset could reach a render on the target system. | The reported approximately 1431 ms value has no recoverable JSON and must not be used; formal raw frames are authoritative. |

Detailed implementation chronology remains available in Specs 001--004,
`BENCHMARK_PROTOCOL.md`, `EXPERIMENT_LADDER.md`, the runtime-cell trace, and
the source history. This table replaces repeated conversation-level narratives;
it does not erase them.

## Citation and dissertation use

- The full dissertation already contains the paper key
  `liu2024citygaussianrealtimehighqualitylargescale`, but final metadata should
  be normalized against the chosen ECCV proceedings or arXiv version.
- Add a separate bibliographic/source reference for the pinned CityGaussian
  repository/commit when discussing released-code behavior.
- Cite LightGaussian and MatrixCity where their methods/assets/dataset claims
  are used.
- Do not copy the condensed dissertation's stale citation key
  `liu2024citygaussian`; the condensed document is not the writing target.

Likely dissertation use: methodology/reproducibility chapter, CityGaussian
case study, stage-by-stage controlled evaluation, paper/source divergence, and
threats to validity. Use the corrected memo and canonical machine-readable
tables for exact claims.
