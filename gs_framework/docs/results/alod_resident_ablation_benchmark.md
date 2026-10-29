# ALoD Resident Ablation Benchmark

Date: 2026-08-08

Status: `PASS-ALOD-RESIDENT-BENCHMARK`

## Scope

This campaign measures the production resident ALoD ladder before any host
streaming or GPU working-set cache exists. All serialized Gaussian properties
remain GPU resident in every configuration. The result is therefore a
**resident representation baseline**, not the final memory-scalable runtime.

The external campaign package is:

```text
/home/luis/tese/benchmark_artifacts/alod_resident_benchmark_001
```

It contains raw per-frame JSONL, resolved configurations, per-process
summaries, correctness checks, all-camera quality comparisons, CSV/JSON and
Markdown tables, figures, commands, and a campaign manifest.

Explicitly excluded throughout:

```text
host property offload, on-demand H2D transfer, GPU cache, LRU/write-back,
previous-working-set reuse, asynchronous transfer, prefetch, eviction,
dynamic residency, and custom CUDA kernels
```

## Frozen identities

| Item | Identity |
|---|---|
| Campaign | `ALOD-RESIDENT-BENCHMARK-001` |
| Hierarchy | Run005 `final.dhier` |
| `.dhier` SHA-256 | `6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c` |
| Serialized rows | 1,099,999 |
| Skybox rows | 100,000 |
| Hierarchy rows | 999,999 |
| Finest hierarchy leaves | 500,000 |
| Oracle | `ALOD-RUN005-SELECTION-ORACLE-001` |
| Oracle semantic SHA-256 | `20bb87caf1694ede18ac2d2a71cdc146c002325ddbf8f6060625f18feda70591` |
| Project commit | `25bbca34c2a4951bb2ff29cf7a251b6cbc55cd6a` plus the documented dirty working tree |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB reported by NVML |
| Renderer | shared project `GsplatRenderer`, gsplat 1.5.3, packed mode |
| Resolution | 960 × 540 |

The configuration file SHA-256 values were frozen in the campaign manifest.
The tracked trajectory file SHA-256 is
`c6d607b8baee0cc30f3c74121e9d19ba045a820fc87c8550a11c3dbc0c92968d`.

## Configuration semantics

### `ALOD-HIER-FINEST-RESIDENT`

The finest representation is exactly the 500,000 serialized hierarchy rows
whose hierarchy `child_count` is zero, preceded by the 100,000 serialized
skybox rows. It does not render all hierarchy nodes simultaneously.

All ten cameras passed the following checks:

- exact leaf identity;
- unique hierarchy IDs;
- exactly one selected node on every root-to-leaf path;
- no selected ancestor/descendant pair;
- exact skybox prefix `0..99,999`;
- raster order equal to skybox followed by selected hierarchy IDs.

The production implementation recomputes the finest leaf identity each frame.
That measured selection cost is included; no asset-static cache was introduced
for the benchmark.

### `ALOD-HIER-LOD-RESIDENT`

This mode uses the identical Run005 HSPT, distance multiplier, target
granularity, upper LoD predicate, native SPT interval behavior, and output
ordering as the production configuration. Four-plane rejection is bypassed,
so all 23 SPT candidates are traversed and the upper traversal emits 633 rows
for every camera.

The source-faithful HSPT output is a cut in the official HSPT/SPT execution
semantics, but it is not a strict antichain in the original serialized binary
tree: official-equivalent output may include an original-tree ancestor and
descendant simultaneously. This was measured and preserved rather than
silently redefining the selector. Global IDs remain unique, all leaf paths are
covered without frustum rejection, and the exact production/oracle parity gate
remains the correctness authority.

### `ALOD-HIER-LOD-FRUSTUM`

This is the primary source-faithful resident configuration:

```text
left/right/bottom/top planes only
near/far planes disabled
bounding spheres disabled
bound = 3 * exp(max(log-scale components))
identical HSPT, LoD predicate, SPT cut, and ordering
```

Across the ten cameras it produced 22 or 23 candidate SPTs, 88 to 215 upper
emitted rows, and 108 to 168 upper frustum rejections.

Before timing, `PASS-ALOD-PRODUCTION-RESIDENT` was rerun. All ten cameras again
matched the CPU reference and official oracle exactly for HSPT arrays,
candidate and per-SPT order, ordered and sorted selected IDs, raster IDs, and
all pre-raster tensor/camera hashes.

## Trajectory and timing protocol

The nearest existing deterministic asset-compatible trajectory was retained:

```text
trajectory_id = alod_run005_frozen_evaluation_v1
source evaluation indices = [0,2,4,6,7,8,9,10,12,14]
camera count = 10
resolution scale = 0.5
resolution = 960 x 540
```

Camera order, intrinsics, poses, resolution, background, render settings, and
seeds were identical across configurations. The trajectory is the frozen
`WEAK-LOD-ORACLE`; no camera was added or changed after observing results.

Each configuration ran in three independent processes. Each process rendered
one full ten-camera warmup followed by ten complete measured trajectory
repetitions, giving 100 measured frames per process and 300 per configuration.
The nine process runs used a balanced interleaving so every configuration
occupied each run-order position. No run was rejected.

CUDA was synchronized around preparation and rendering. GPU events measured
rasterization. Primary total frame time is synchronized preparation plus
renderer wall time; GPU-event rasterization is reported separately. Quality
metrics ran in a separate pass.

Formal cold initialization was approximately 3.07 to 3.25 seconds per process.
HSPT construction was 196 to 210 ms, GPU property activation/residency transfer
was approximately 97 to 111 ms, and the ten-frame warmup was approximately 635
to 751 ms. The correctness gate incurred a separate first-use gsplat extension
setup of 62.99 seconds; it completed before all formal configurations and was
not charged selectively to one mode.

## Performance

All values below pool 300 measured frames per configuration. FPS is derived
from mean end-to-end frame time.

| Configuration | Mean total ms | Median ms | P95 ms | FPS | Total selection ms | Gather ms | GPU raster ms | Selected hierarchy | Total raster rows | Backend-visible | Peak allocated MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ALOD-HIER-FINEST-RESIDENT` | 6.940 | 6.833 | 7.729 | 144.09 | 3.631 | 1.011 | 1.287 | 500,000 | 600,000 | 16,905 | 196.7 | 1,547.3 |
| `ALOD-HIER-LOD-RESIDENT` | 9.326 | 9.227 | 10.025 | 107.23 | 6.041 | 1.032 | 1.280 | 494,534 | 594,534 | 16,858 | 195.2 | 1,559.6 |
| `ALOD-HIER-LOD-FRUSTUM` | 9.709 | 9.703 | 10.745 | 103.00 | 6.425 | 1.017 | 1.275 | 482,252 | 582,252 | 16,857 | 198.1 | 1,587.3 |

The standard deviation of the three replicate mean total-frame times was
0.185 ms for finest, 0.042 ms for LoD resident, and 0.041 ms for LoD plus
frustum. Pooled per-frame standard deviations were 0.442, 0.462, and 0.688 ms.

Mean index creation was 0.314, 0.321, and 0.317 ms respectively. Mean
pre-raster metrics assembly was 0.019, 0.018, and 0.018 ms. Gather and assembly
therefore did not materially improve when selection removed rows on this
trajectory.

## Finest versus LoD

LoD removed only 5,465.6 hierarchy rows on average, or 1.09%. Selected counts
ranged from 482,583 to 499,530, with median 496,628.5. Backend-visible rows
changed from 16,904.9 to 16,858.4 on average.

Rasterization saved only 0.0067 ms (0.52%). Total selection increased by
2.4107 ms, while gather increased by 0.0209 ms. The net effect was a 2.3858 ms
increase in total frame time: LoD resident was 34.38% slower than the resident
finest cut.

The answer to the primary performance question is therefore **no**: on this
weak-LoD trajectory, resident hierarchical LoD does not reduce rasterization
enough to offset selection and preparation overhead.

## LoD versus LoD plus frustum

Frustum rejection removed another 12,282.4 hierarchy rows on average, but the
distribution was highly skewed: minimum 418, median 514, and maximum 118,226
additional removals. The first camera accounts for most of the mean reduction.

Total selection increased by 0.3838 ms. Gather saved 0.0147 ms and GPU
rasterization saved only 0.0058 ms. Total frame time increased by 0.3826 ms,
or 4.10%.

The answer is again **no** for this trajectory: official-equivalent frustum
rejection did not improve total frame time after accounting for traversal.
Selected counts fell, but backend-visible counts remained essentially
unchanged.

## Selection-depth behavior

The finest leaf set is asset-static: 500,000 hierarchy rows at depths 16 to 23,
median depth 19, with Jaccard 1.0 between cameras.

Both LoD modes selected depths 2 through 23 with per-camera median depth 19.
LoD-resident selected counts were 482,583 / 496,628.5 / 499,530
(minimum/median/maximum). LoD plus frustum selected counts were
364,357 / 496,155.5 / 499,030. Minimum sequential selected-set Jaccard was
0.9246 without frustum and 0.7069 with frustum.

The maximum selected depth and median depth never changed. This preserves the
original `WEAK-LOD-ORACLE` finding: the campaign is valid for runtime cost,
correctness, and resident-ablation structure, but it is not final evidence of
strong multi-scale LoD adaptation.

## Resident memory

| Component | Bytes | MiB | Behavior |
|---|---:|---:|---|
| GPU Gaussian properties | 101,199,908 | 96.51 | Identical in all modes |
| GPU global-ID identity / hierarchy mapping | 8,799,992 | 8.39 | Identical in all modes |
| CPU serialized Gaussian properties | 101,199,908 | 96.51 | Identical in all modes |
| CPU hierarchy plus HSPT metadata | 38,458,208 | 36.68 | Identical in all modes |
| Mean selected index | 4,800,000 / 4,756,275 / 4,658,016 | 4.58 / 4.54 / 4.44 | Small working-set change |

Mean steady PyTorch allocation was approximately 189.65, 188.89, and
188.35 MB (decimal) for finest, LoD, and LoD plus frustum. Peak allocated
memory was approximately 195 to 208 MB and did not track selected-count
reduction monotonically. Whole-device peaks were approximately 1.91 to
1.94 GiB and include the CUDA context and other device allocations.

Selected-count reduction therefore does **not** reduce resident property VRAM.
The persistent Gaussian property store and global identity tensor remain
dominant and invariant; only small index and transient working-set allocations
change. This is the baseline against which host-resident properties and a
bounded GPU working set must later be measured.

## Same-asset image comparison

Images were clamped consistently to `[0,1]`. PSNR, SSIM, LPIPS-AlexNet, mean
absolute error, and maximum absolute error were computed on all ten cameras.
The finest valid cut is a same-asset reference, not photographic ground truth.

| Comparison | Min PSNR dB | Mean finite PSNR dB | Mean SSIM | Mean LPIPS | Mean abs. error | Max abs. error |
|---|---:|---:|---:|---:|---:|---:|
| Finest → LoD resident | 33.803 | 68.160 | 0.99837817 | 0.00106260 | 0.00391178 | 0.240898 |
| Finest → LoD + frustum | 33.803 | 63.224 | 0.99837817 | 0.00106260 | 0.00391182 | 0.240898 |
| LoD resident → LoD + frustum | 110.726 | 126.422 | 0.99999840 | 2.50e-10 | 3.67e-08 | 8.30e-05 |

Two finest/LoD frames and one finest/frustum frame were exactly equal after
clamping. Seven of ten LoD/frustum pairs were exactly equal. Frustum output is
therefore not bitwise unchanged across the complete set, but it is numerically
equivalent at the image level under the reported error envelope.

## Bottleneck and interpretation

The measured bottleneck is CPU-side selection/preparation, not rasterization.
The finest implementation itself spends 3.63 ms rebuilding its static leaf
identity each frame. LoD raises selection to 6.04 ms, and frustum raises it to
6.43 ms. GPU rasterization remains approximately 1.28 ms because the
backend-visible count is nearly invariant even when active/raster rows fall.

This reproduces the earlier CityGaussian systems lesson in a different
representation: reducing selected rows is not a performance result by itself.
Per-frame traversal, indexing, gathering, concatenation/allocation, and the
renderer-visible workload determine the net outcome.

For scalability, two distinct bottlenecks remain:

1. resident property memory remains fixed because every Gaussian property is
   still on the GPU; and
2. CPU selection dominates steady frame time on this weak-LoD trajectory.

The current evidence does not justify changing the selector, hierarchy, or
trajectory. A future mixed-scale scene/trajectory is required for strong LoD
adaptation evidence.

## Validation and protected state

The campaign completion gate includes full unit discovery, external ALoD
integration gates, the exact production parity gate, benchmark package/schema
checks, `git diff --check`, and final protected-state fingerprints. The
official checkout already contained generated build artifacts and a documented
Blackwell CMake change before this task; validation requires unchanged status
and hashes, not a false clean-checkout claim.

No historical CityGaussian result was modified.

## Next bounded task

Design and implement ALoD host-resident Gaussian properties with a bounded GPU
working-set/cache, using this resident benchmark as the performance and memory
baseline.

Do not infer that streaming/cache work was implemented by this result.
