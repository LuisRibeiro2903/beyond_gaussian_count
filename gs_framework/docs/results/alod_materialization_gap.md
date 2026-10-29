# ALoD Selected-to-Materialized Gap

Status: `PASS-ALOD-PREMATERIALIZATION-CULLING`

Artifact root:
`/home/luis/tese/benchmark_artifacts/alod_materialization_gap_001`

Asset: Run005 `final.dhier`, SHA-256
`6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c`

## Scope and protected semantics

This gate adds an opt-in post-selection, pre-property-gather filter. It does
not change HSPT construction, four-plane frustum semantics, the LoD predicate,
candidate SPTs, per-SPT cuts, semantic hierarchy IDs, semantic ordering, the
official oracle, the Run005 asset, the formal trajectory, or the 525,000-row
cache configuration.

The frame identity sets are now distinguished as follows:

1. `selected_hierarchy_global_node_ids`: unchanged production ALoD hierarchy
   cut, excluding the fixed skybox prefix;
2. `semantic_selected_global_node_ids`: the existing semantic raster candidate
   order, namely skybox prefix followed by the hierarchy cut;
3. `materialization_global_node_ids`: conservative subset for which complete
   render properties are gathered;
4. `raster_global_node_ids`: ordered input actually passed to gsplat, currently
   equal to the materialization set;
5. `backend_projection_survivor_global_node_ids`: packed gsplat projection
   survivors mapped back to serialized `.dhier` row IDs.

`global_node_id` remains the serialized `.dhier` row index at every stage.
The semantic selector implementation in `scene_runtime/alod/selection.py` is
unchanged.

## Exact meaning of the historical backend-visible metric

The historical project field `visible_gaussians` is a **gsplat projection
survivor count**, not a final pixel-contributor count.

The exact trace for the installed gsplat 1.5.3 environment is:

- `scene_runtime/renderer.py`, `GsplatRenderer.render`: calls
  `gsplat.rendering.rasterization`;
- `scene_runtime/renderer.py`, `_visible_count`: with the formal
  `packed=true` setting, counts unique `info["gaussian_ids"]`;
- installed `gsplat/rendering.py`, `rasterization`: receives packed IDs from
  `fully_fused_projection`, stores them in `info`, then performs
  `isect_tiles` and pixel rasterization;
- installed
  `gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu`: emits a packed Gaussian ID
  after the projection tests below.

For the tracked pinhole/classic settings, a packed ID survives:

1. center depth is within `[near_plane, far_plane]` (the kernel rejects
   `z < near` or `z > far`);
2. blurred projected covariance determinant is positive;
3. opacity is at least `ALPHA_THRESHOLD = 1/255`;
4. the opacity-aware projected radius is not clipped by `radius_clip`;
5. the projected ellipse bounding rectangle overlaps the image.

The metric is produced **before tile intersection**. It therefore does not
mean nonzero tile count, a rasterized alpha sample, or nonzero final pixel
contribution. For backwards compatibility, historical `visible_gaussians`
and `rendered_gaussian_count` outputs are retained. New benchmark output also
uses the exact alias `backend_projection_survivor_count`.

On all ten validation cameras, every projection survivor also had a nonzero
tile-intersection count. That equality is measured behavior for this asset and
trajectory, not the definition of the metric.

## Selected-to-projection funnel

The complete ten-camera frozen trajectory was evaluated. Counts below include
the 100,000-row skybox prefix except where explicitly labeled hierarchy-only.

| Stage | Ten-camera total | Mean/frame | Percent of semantic input |
|---|---:|---:|---:|
| Semantic hierarchy cut only | 4,822,520 | 482,252.0 | N/A |
| Semantic input including skybox | 5,822,520 | 582,252.0 | 100.000% |
| Center-depth gate survivor | 3,940,230 | 394,023.0 | 67.672% |
| Projection survivor with opacity omitted | 171,813 | 17,181.3 | 2.951% |
| Actual opacity-aware backend projection survivor | 168,566 | 16,856.6 | 2.895% |
| Nonzero tile-intersection survivor | 168,566 | 16,856.6 | 2.895% |
| Conservative pre-materialization survivor | 188,419 | 18,841.9 | 3.236% |

Backend-supported rejection decomposition across the ten cameras:

| Rejection group | Rows | Percent of semantic input | Evidence boundary |
|---|---:|---:|---|
| Center outside backend depth interval | 1,882,290 | 32.328% | Camera-space center depth |
| Projected-covariance/radius/image rejection after depth, with opacity omitted | 3,768,417 | 64.728% | gsplat projection call with identical geometry and no opacity |
| Additional opacity-aware projection rejection | 3,247 | 0.056% | Difference between no-opacity and actual projection IDs |
| Actual projection survivors | 168,566 | 2.895% | Packed `gaussian_ids` |

The installed public metadata does not expose a per-row rejection enum.
Consequently, determinant, radius, and image-bound failures cannot be split
reliably without changing or instrumenting the CUDA kernel. With
`radius_clip=0`, positive `eps2d=0.3`, finite validated inputs, and positive
scales, the combined post-depth group is expected to be dominated by image
non-intersection, but it remains labeled as the exact recoverable combined
category rather than an invented finer breakdown.

## Field dependencies

| Criterion or stage | Minimum exact fields | Color/SH needed? | Notes |
|---|---|---:|---|
| Center-depth gate | mean, view matrix, near/far | No | Exact first backend gate |
| Projected covariance determinant | mean, quaternion + scale or covariance, camera, `eps2d` | No | Camera-model dependent |
| Opacity threshold/tight radius | opacity plus projected covariance | No | Threshold is `1/255` |
| Radius clipping | mean, covariance representation, camera, `eps2d`, opacity, `radius_clip` | No | Opacity changes the tight radius |
| Image overlap | projected mean and radii | No | Full exact inputs are mean, covariance representation, opacity, camera |
| Tile intersection | projected mean, radii, depth, tile/image dimensions | No | Runs after the historical metric |
| Pixel contribution | projection/tile data, opacity, color/SH, ordering | Yes | Not exposed as one Gaussian visibility set |
| Conservative implemented bound | mean + padded maximum scale, camera | No | Ignores opacity tightening and rotation |

An exact duplicate of gsplat projection using quaternion/scale properties would
require mean (12 bytes), quaternion (16), scale (12), and opacity (4): 44 bytes
per row, or 46.16 MiB for this asset. A covariance alternative is 40 bytes per
row, or 41.96 MiB. Both omit SH but are too close to the current 54.84 MiB
bounded full-property cache to be an attractive redundant persistent table.

The implemented conservative representation is 16 logical bytes per row:
float32 mean plus a padded float32 maximum scale. This is 17,599,984 bytes
(16.78 MiB) logically. Means already exist in the hierarchy source, so only
the 4,399,996-byte (4.20 MiB) maximum-scale array is incremental CPU memory.
No compact metadata is resident on the GPU and no compact metadata H2D traffic
is hidden from the accounting.

## gsplat primitive audit

| Candidate | Classification | Reason |
|---|---|---|
| `fully_fused_projection` | Usable directly for exact diagnosis; unsuitable as the chosen prefilter | Exact projection IDs, but requires geometry and opacity tensors on GPU before filtering |
| `isect_tiles` | Too late for the historical metric | Requires projected means/radii/depths and runs after projection |
| Separate covariance/projection wrappers | Usable with adaptation for diagnosis | Still require full covariance information and do not remove the property dependency |
| Pure PyTorch reference projection | Diagnostic only | Does not guarantee installed fused-kernel float identity |
| CPU mean/max-scale analytic bound | Implemented | Uses existing CPU metadata, no custom CUDA, no full-property gather, zero observed false negatives |
| GPU mean/max-scale analytic bound | Viable future adaptation | Would add either 16.78 MiB persistent GPU metadata or selected compact H2D/index traffic |

No dependency was added or upgraded, and no custom CUDA was written.

## Implemented conservative filter

`scene_runtime/alod/materialization.py` implements the filter after
`select_production()` and before resident gather or host/cache `prepare()`.
For each semantic row it:

1. applies the backend center-depth necessary condition with padding;
2. treats `max(scale)` as an isotropic covariance upper bound;
3. applies the installed pinhole projection Jacobian and gsplat's maximum
   extent `3.33` to form over-bounded X/Y radii;
4. retains the row if that over-bounded rectangle can intersect the image.

Opacity tightening is deliberately ignored. Scale padding of `1e-4`, depth
padding, and two pixels of screen padding cover CPU/CUDA evaluation-order
differences. Unsupported camera modes and UT/eval3d modes fall back to identity
materialization rather than risk a false negative.

The feature is opt-in through:

```text
alod.materialization_filter = "conservative_pinhole"
```

The existing configurations retain `none`; the bounded diagnostic uses
`ALOD-HOST-BOUNDED-CACHE-PREMATERIALIZATION`. The 525,000-row cache policy is
unchanged.

## Correctness evidence

`checks/prematerialization_correctness.json` compares existing resident,
filtered resident, and filtered bounded-cache paths for all ten frozen cameras.

- semantic selected hierarchy IDs and order: exact for all cameras;
- semantic selected raster IDs including skybox: exact for all cameras;
- materialization set: subset in original order with stable IDs;
- backend projection false negatives: zero over 168,566 survivor instances;
- surviving means/quaternions/scales/opacities/SH: bitwise equal;
- backend projection survivor IDs and counts: exact;
- resident filtered RGB and alpha: bitwise equal to existing resident;
- host/cache filtered RGB and alpha: bitwise equal to existing resident.

The diagnostic prefilter produced zero backend false negatives on the same
complete trajectory. No image tolerance was used.

## Materialization reduction

| Count | Mean | Median | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| Semantic hierarchy rows | 482,252.0 | 496,155.5 | 498,755.1 | 364,357 | 499,030 |
| Semantic rows including skybox | 582,252.0 | 596,155.5 | 598,755.1 | 464,357 | 599,030 |
| Materialization rows | 18,841.9 | 19,227.5 | 42,944.3 | 2,018 | 51,074 |
| Backend projection survivors | 16,856.6 | 17,551.0 | 38,542.8 | 1,782 | 46,173 |
| Materialization minus backend | 1,985.3 | 1,894.5 | 4,401.5 | 236 | 4,901 |

The per-frame semantic-including-skybox to materialization reduction averages
96.82% (range 91.29%-99.57%). The materialization/backend ratio averages
1.13:1, down from the reported 28.6:1 hierarchy-selected/backend ratio. Thus
almost all of the original gap is removed before complete render-property
materialization while semantic selection remains unchanged.

## Bounded deterministic memory and timing comparison

One ten-frame, ten-warmup bounded comparison was run for the unchanged 525k
cache. This is a characterization, not a replacement four-mode formal
campaign.

| Metric | Existing bounded cache | Filtered bounded cache | Change |
|---|---:|---:|---:|
| Mean frame-local property bytes | 93.40 MiB | 3.31 MiB | -96.46% |
| Mean frame-local index bytes | 3.68 MiB | 0.14 MiB | -96.10% |
| Maximum PyTorch allocated | 198.83 MiB | 103.07 MiB | -48.16% |
| Maximum PyTorch reserved | 336.00 MiB | 118.00 MiB | -64.88% |
| Mean before-raster allocated | 167.13 MiB | 65.81 MiB | -60.63% |
| Mean after-raster allocated | 181.79 MiB | 80.16 MiB | -55.91% |
| Maximum whole-device NVML | 1,934.18 MiB | 1,716.30 MiB | -217.88 MiB / -11.26% |
| Persistent full-property allocation | 54.84 MiB | 54.84 MiB | unchanged by design |
| Incremental compact CPU metadata | 0 | 4.20 MiB | +4.20 MiB |
| Mean filter time | 0 | 18.79 ms | +18.79 ms |
| Mean residency lookup | 3.23 ms | 0.19 ms | -3.04 ms |
| Mean cache management | 2.05 ms | 0.45 ms | -1.60 ms |
| Mean selected GPU gather | 0.64 ms | 0.18 ms | -0.46 ms |
| Mean renderer call | 1.43 ms | 1.44 ms | +0.01 ms |
| Mean end-to-end frame | 18.09 ms | 29.68 ms | +64.09% |

The success signal is memory, not speed: frame-local materialization and peak
whole-device VRAM both fall materially. The current NumPy CPU filter is too
slow for the final interactive path and is now the dominant frame stage.

Process-level NVML accounting remains unreliable under WSL (it reports zero
despite active PyTorch allocations), so only whole-device NVML and PyTorch
allocator measurements are used. Peak RSS differed between the two fresh
processes by more than the 4.20 MiB compact table and is not attributed solely
to the filter.

## Transfer and cache impact

The existing and filtered bounded runs both have zero measured steady-state
H2D after the ten-camera warmup. The amount needed to reach that state changes:

- first cold camera: 364,357 rows / 31.97 MiB becomes 2,018 rows / 0.177 MiB;
- ten-camera cache union after warmup: 518,746 rows becomes 151,621 rows;
- inferred full-property bytes loaded for those unions: 45.51 MiB becomes
  13.30 MiB, a 70.77% reduction.

The formal cache allocation remains 525,000 rows. The old maximum selected
hierarchy working set and minimum feasible current-frame capacity were 499,030
rows. The new maximum materialized hierarchy request is 50,936 rows, which is
the new minimum capacity able to hold every individual frozen frame. A cache
of 151,621 rows could retain the entire observed materialization union and
reproduce the current zero-steady-H2D behavior on this exact trajectory. These
smaller capacities are recommendations for a separate experiment, not changes
made by this gate.

## Read-only CPU selection profile

`profiles/cpu_selection_profile.json` profiles 300 unchanged production
selections over repeated complete trajectories. A separately instrumented
diagnostic replica was required for sub-function stages and was checked for
exact output parity on every call. It does not replace production selection.

| Stage | Mean | P95 |
|---|---:|---:|
| Camera preparation / torch-to-NumPy | 0.073 ms | 0.282 ms |
| Frustum plane construction | 0.040 ms | 0.054 ms |
| Upper bound preparation | 0.126 ms | 0.145 ms |
| Upper frustum tests | 0.574 ms | 0.721 ms |
| Upper node classification | 0.107 ms | 0.134 ms |
| Upper distance/threshold work | 0.179 ms | 0.218 ms |
| Upper stack concatenation | 0.056 ms | 0.072 ms |
| Candidate-SPT construction | 0.011 ms | 0.017 ms |
| Other upper cut/output assembly | 0.023 ms | 0.030 ms |
| SPT binary interval searches | 0.051 ms | 0.064 ms |
| Per-SPT index creation and predicate gathers | 4.736 ms | 5.157 ms |
| SPT Python result assembly | 0.034 ms | 0.043 ms |
| Combined SPT concatenation | 0.169 ms | 0.209 ms |
| Final hierarchy/skybox/global-ID assembly | 0.302 ms | 0.360 ms |
| Unmodified `select_production` wall time | 6.404 ms | 7.128 ms |

`cProfile` independently attributes about 4.97 ms/call cumulatively to
`cut_spts` and 1.39 ms/call to `traverse_upper`; it is not used alone for the
native NumPy timing claim.

The allocation audit observed a 17,579,956-byte traced peak for one complete
trajectory and 12,461,644 bytes of returned NumPy arrays for a representative
first-frame selection when per-SPT emitted arrays and combined outputs are
both retained. Major allocation/copy patterns are per-SPT `np.arange`, repeated
advanced indexing and boolean gathers, duplicated per-SPT/final emitted IDs,
per-level upper stack concatenation, final hierarchy/skybox concatenation, and
camera tensor-to-NumPy copies.

No selector optimization was implemented. Future candidates, in measured
priority order, are:

1. fuse/vectorize per-SPT interval, predicate, and output compaction;
2. eliminate duplicated per-SPT and final emitted-ID storage with persistent
   output buffers;
3. reuse upper traversal stacks and cut buffers;
4. migrate HSPT/SPT selection to GPU, potentially with a separate custom
   kernel task;
5. investigate stronger algorithmic hierarchy pruning only after preserving
   oracle semantics.

## Limitations and next bounded task

The filter is proven against the installed gsplat backend and frozen Run005
validation trajectory. Its unsupported renderer modes intentionally retain all
rows. The CPU implementation trades substantial latency for memory and should
not yet be described as the final interactive solution.

Measured evidence selects branch **D** next: evaluate reduced full-property
cache capacities before scaling. The bounded task should compare at least
50,936 rows (minimum complete-frame capacity) and approximately 151,621 rows
(observed all-trajectory union), with explicit headroom, while keeping the
semantic selector and filter fixed. In parallel or immediately afterward, the
same conservative mean/max-scale test should be evaluated with existing GPU
PyTorch operations to remove its 18.79 ms CPU cost; that is a filter-location
change, not a CPU selector optimization.

Do not scale to a larger hierarchy until the smaller-cache experiment confirms
the expected persistent-memory reduction and the filter execution location no
longer dominates frame time.
