# ALoD-EXT natural Previous working-set completion

Protocol: `ALOD-EXT-NATURAL-PREVIOUS-COMPLETION-001`

Status: `PASS-ALOD-EXT-NATURAL-PREVIOUS-COMPLETION`

Classification: project-specific `ALOD-EXT` policy-completion experiment. It
is not released ALoD and not the paper's `Render Full` mode.

## Scientific answer

Previous working-set reuse improves end-to-end performance relative to the
Released Evaluator (No Cache) on both frozen natural trajectories, but the
union-covering Bounded policy is faster on both. Campus Previous reduces mean
frame time by 25.97% (1.351x), from 90.250 to 66.812 ms, while Bounded reaches
65.667 ms. Sci-Art Previous reduces it by 27.28% (1.375x), from 521.086 to
378.909 ms, while Bounded reaches 366.420 ms.

The synthetic observation that Previous can outperform Bounded while moving
more bytes therefore does **not** appear on either natural completion
workload. Previous is 1.145 ms slower than Bounded on Campus and 12.489 ms
slower on Sci-Art. This is a workload- and trajectory-locality result, not a
consequence of the frozen LoD class alone.

## Frozen scope and correctness

The benchmark uses the final hierarchy and complete held-out camera sequence
from the canonical natural association campaign:

| Dataset | Hierarchy | LoD class | Frozen cameras and order | Resolution |
| --- | --- | --- | --- | ---: |
| H3DGS Campus | `ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-TRAIN-001` | `VALID-NONTRIVIAL-LOD` | all 24 cameras in `ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-EVAL-24-PUBLIC-HELDOUT-V1` | 1436x1075 |
| UrbanScene3D Sci-Art | `ALOD-PREV-US3D-SCIART-01-TRAIN-001` | `VALID-WEAK-LOD` | all 21 cameras in `ALOD-PREV-US3D-SCIART-01-EVAL-21-PUBLIC-VAL-V1` | 1216x912 |

No hierarchy, camera, selector, filter, cache policy, timing boundary, or
dataset preparation changed. Exact asset, camera, oracle, implementation, and
protocol hashes are sealed in `protocol.json` in the artifact package.

The established correctness barriers were not rerun unnecessarily. The
post-ordered-multiset natural regression is byte-for-byte identical to the
canonical Campus and Sci-Art correctness JSON and NPZ traces. Across all 24
Campus and 21 Sci-Art cameras, it proves exact official semantic IDs/order,
zero conservative-filter false negatives, exact CPU/CUDA filtered IDs/order,
bitwise retained properties/order, exact packed survivors, and bitwise RGB and
alpha agreement. Its authority is
`benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/checks/natural_regression_summary.json`.

## Timing protocol and rerun decision

Each policy ran in three fresh processes. Every process rendered the first ten
canonical cameras as ordered warmup, retained reuse/cache state, restarted the
full sequence at camera zero, and measured it once. End-to-end timing is
CUDA-synchronized and includes semantic selection, filtering,
preparation/residency, assembly, and rendering. Timing columns below are
nested diagnostics and must not be summed.

The old natural package already contained Previous rows, but it predates the
ordered-multiset runtime and current benchmark recorder. The implementation
hashes therefore differ. Correctness is unchanged, but timing equivalence is
not established, so Released and Bounded were rerun with Previous for a direct
same-state comparison. The original canonical artifacts were not overwritten.
The Bounded capacities remain the prospectively fixed 621,377 Campus rows and
2,666,887 Sci-Art rows; both cover the retained trajectory union and incur zero
evictions.

## Performance

Means pool every measured camera and all three fresh processes. Renderer wall
time is `frame_time_ms`; nested rasterization is the gsplat rasterizer timing.
Dynamic-property H2D bytes pool the measured sequence across three processes
and exclude warmup, the persistent skybox, compact selector metadata,
ID/index traffic, and device-to-device copies.

| Dataset | Mode | End-to-end ms | FPS | Select ms | Filter ms | Inclusive prepare ms | Host gather ms | H2D ms | Renderer wall ms | Nested raster ms | Steady dynamic H2D |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Campus | Released | 90.250 | 11.08 | 6.788 | 0.000 | 86.177 | 8.522 | 3.299 | 4.072 | 3.856 | 2,849,422,620 B |
| Campus | Previous | 66.812 | 14.97 | 6.745 | 48.026 | 62.864 | 2.546 | 1.620 | 3.949 | 3.757 | 703,375,512 B |
| Campus | Bounded | 65.667 | 15.23 | 6.757 | 49.850 | 62.185 | 0.262 | 0.451 | 3.482 | 3.303 | 22,038,876 B |
| Sci-Art | Released | 521.086 | 1.92 | 29.353 | 0.000 | 509.460 | 68.222 | 16.456 | 11.626 | 11.146 | 14,192,560,320 B |
| Sci-Art | Previous | 378.909 | 2.64 | 29.106 | 301.053 | 369.530 | 13.001 | 6.875 | 9.378 | 8.993 | 2,741,495,856 B |
| Sci-Art | Bounded | 366.420 | 2.73 | 28.694 | 300.791 | 356.598 | 0.809 | 0.542 | 9.822 | 9.402 | 47,909,184 B |

The conservative filter time now includes ordered-multiset compatibility work.
On these duplicate-free natural inputs, stable unique-ID extraction alone
averages 42.919/44.363 ms for Campus Previous/Bounded and
281.485/280.726 ms for Sci-Art. The mapping is the identity and no occurrence
reindex gather is required, but the extraction cost remains included in filter,
preparation, and end-to-end time. This is one reason the absolute values in
this package must not silently replace the older pre-multiset natural timing
rows.

## Previous reuse and transfer behavior

| Dataset | Previous pooled row-hit rate | Mean per-frame retention | Bounded pooled row-hit rate | Previous steady H2D reduction vs Released | Previous cold-inclusive H2D reduction vs Released | Previous / Bounded steady H2D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Campus | 25.19% | 24.71% | 97.66% | 75.32% | 74.98% | 31.92x |
| Sci-Art | 37.36% | 40.73% | 98.91% | 80.68% | 81.51% | 57.22x |

Campus Previous requests a mean 141,947 dynamic hierarchy rows per frame,
hits 35,761, and misses 106,186. Sci-Art requests 755,162, hits 282,164, and
misses 472,998. Previous therefore exploits meaningful consecutive-frame
overlap, but discards rows that recur after a gap. The persistent Bounded cache
retains the complete trajectory union, yielding much higher hit rates after
the fixed warmup and reducing both host gather and transfer latency.

On Campus, Bounded saves 2.284 ms of host gather and 1.168 ms of H2D time
relative to Previous, partly offset by 1.824 ms more filter time; its net mean
advantage is 1.145 ms. On Sci-Art, filter time is effectively equal, while
Bounded saves 12.191 ms of host gather and 6.332 ms of H2D time; its net mean
advantage is 12.489 ms. Transfer volume and latency matter, but neither alone
determines end-to-end performance.

## Workload funnel

The semantic and materialization funnel is unchanged from the canonical
correctness evidence.

| Dataset | Mean semantic rows | Mean materialized rows | Mean packed survivors | Materialized / semantic | Survivors / semantic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Campus | 530,166.46 | 158,787.33 | 146,633.21 | 29.9505% | 27.6580% |
| Sci-Art | 2,548,681.90 | 767,118.05 | 732,242.76 | 30.0986% | 28.7303% |

## Four-workload policy comparison

Values are mean end-to-end frame time in milliseconds. Run009 and the
synthetic control are existing canonical evidence, while Campus and Sci-Art
are the new same-state completion runs. Different assets and trajectories make
this a policy-outcome summary, not a cross-dataset speed ranking.

| Dataset | LoD class | Released | Previous | Bounded | Fastest EXT policy |
| --- | --- | ---: | ---: | ---: | --- |
| Run009 | historical `VALID-WEAK-LOD` | 186.774 | 71.657 | 73.480 | Previous |
| Campus | `VALID-NONTRIVIAL-LOD` | 90.250 | 66.812 | 65.667 | Bounded |
| Sci-Art | `VALID-WEAK-LOD` | 521.086 | 378.909 | 366.420 | Bounded |
| Synthetic strong | `STRONG CONTROL` | 17.679 | 9.577 | 15.161 | Previous |

The outcome is workload dependent. Previous wins on canonical Run009 and the
controlled synthetic path, while Bounded wins on both natural completion
paths. Campus and Sci-Art have different frozen LoD classes yet the same policy
winner; Run009 and Sci-Art share a weak class yet have different winners.
These observations directly reject LoD class as a sufficient explanation.
The more defensible mechanism is temporal locality: immediate working-set
overlap benefits Previous, whereas rows reused after longer gaps retain value
only in the persistent Bounded cache.

## Limitations

- The comparison establishes behavior on the frozen canonical camera order;
  Previous is order-sensitive and no reordered trajectory was tested here.
- Absolute current natural times differ substantially from the older
  pre-multiset campaign. Only the three current same-state rows within each
  natural dataset are treated as the direct completion comparison.
- Dynamic-property byte totals are explicit accounting, not whole-bus traffic.
- The union-covering Bounded capacities test persistent residency without
  pressure; they are not claimed to be FPS-optimal capacities.
- Three replicates characterize this host and execution state, not wider
  hardware or population variability.

## Canonical evidence

All paths below are relative to
`benchmark_artifacts/alod_ext_natural_previous_completion_001/`:

- prospective machine protocol: `protocol.json`;
- administrative hash-transcription correction (no scientific or runtime
  change): `protocol_amendment_001.json`;
- raw executions: `runs/{campus,sciart}/{nocache,previous,bounded}/replicate-*`;
- complete schedule: `checks/schedule.json`;
- aggregate and four-workload comparison: `tables/completion_summary.json`;
- package gate: `checks/package_validation.json`;
- immutable file inventory: `manifest.json`.

The human-readable prospective protocol is
`thesis_notes/ALOD_EXT_NATURAL_PREVIOUS_COMPLETION_PROTOCOL.md`.
