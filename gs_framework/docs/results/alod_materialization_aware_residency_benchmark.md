# Formal Materialization-Aware ALoD Residency Benchmark

Status: `PASS-ALOD-MATERIALIZATION-AWARE-BENCHMARK`

Campaign: `ALOD-MATERIALIZATION-AWARE-RESIDENCY-BENCHMARK-001`

Artifacts: `/home/luis/tese/benchmark_artifacts/alod_materialization_aware_residency_benchmark_001`

## Scope and protocol

This campaign measures the production Run005 ALoD runtime after the validated
CUDA conservative pre-materialization filter reduced full-property
materialization. It does not change HSPT construction, LoD/frustum predicates,
upper traversal, SPT cuts, selected global IDs or order, the conservative
filter, skybox behavior, cache policies, or renderer semantics.

- Asset: `final.dhier`, SHA-256 `6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c`.
- Trajectory: unchanged `alod_run005_frozen_evaluation_v1`, SHA-256 `c6d607b8baee0cc30f3c74121e9d19ba045a820fc87c8550a11c3dbc0c92968d`.
- Resolution: 960×540.
- Per process: ten-camera warmup, then ten complete trajectory repetitions (100 measured frames).
- Replication: three fresh processes/configuration; 300 measured frames/configuration and 1,200 total.
- Ordering: exact A/B/C/D substitution into the accepted 12-process host-residency schedule.
- Timing: synchronous completed work with CUDA events; semantic selection and the CUDA prefilter remain separate.
- Optional no-cache: not repeated because the previous formal campaign already established its extreme traffic and the current four modes answer the direct comparison.

`WEAK-LOD-ORACLE` still applies. These are formal runtime, materialization,
residency, traffic, and memory results on the frozen trajectory, not strong
multi-scale LoD evidence.

## Correctness gate

The ten-camera preflight returned
`PASS-ALOD-MATERIALIZATION-AWARE-PREFLIGHT` before timing began.

- semantic hierarchy and raster IDs/order were exact;
- CUDA materialization IDs exactly matched the validated CPU reference;
- backend-projection false negatives were zero;
- surviving means, quaternions, scales, opacities, and SH were bitwise exact;
- backend survivor IDs were exact;
- RGB and alpha were bitwise identical;
- global-node-to-slot mappings were exact;
- predefined alternative histories produced no stale-slot read or render change.

The previous-reuse, 525k, and 160k paths therefore differ only in residency and
cache state, not semantic selection, materialization, or image output.

## Primary formal results

Memory units are powers of two. H2D totals below cover the 300 accepted
steady-state frames/configuration; cold and warm population are separate.

| Configuration | Mean ms | Median ms | P95 ms | FPS | Selection ms | Filter ms | Residency/gather/H2D ms | Assembly ms | Raster ms | Semantic rows | Materialized rows | Backend survivors | Full properties MiB | Compact MiB | Peak allocated MiB | Peak device MiB | H2D MiB | Hit/reuse | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Resident | 10.708 | 10.773 | 11.835 | 93.39 | 6.573 | N/A | N/A | 0.217 | 1.271 | 582,252 | 582,252 | 16,857 | 96.51 | N/A | 230.76 | 1,946.17 | 0 | N/A | N/A |
| Fast + previous reuse | 12.880 | 13.002 | 14.524 | 77.64 | 6.474 | 2.230 | 1.640 | 0.161 | 1.393 | 582,252 | 18,842 | 16,857 | 10.43 mean | 16.78 | 93.21 | 1,758.38 | 433.16 | 12.56% | N/A |
| Fast + 525k cache | 12.465 | 12.518 | 13.823 | 80.23 | 6.586 | 2.240 | 0.874 | 0.138 | 1.308 | 582,252 | 18,842 | 16,857 | 54.84 | 16.78 | 139.89 | 1,804.06 | 0 | 100% | 0 |
| Fast + 160k cache | 12.018 | 12.021 | 13.657 | 83.21 | 6.590 | 2.283 | 0.590 | 0.148 | 1.319 | 582,252 | 18,842 | 16,857 | 22.81 | 16.78 | 108.45 | 1,783.88 | 0 | 100% | 0 |

The semantic total consists of 482,252 hierarchy rows and 100,000 skybox rows
on average. The filtered total consists of 18,819.9 hierarchy rows and 22.0
skybox rows. The candidate SPT count averages 22.4.

For every filtered configuration, ratios of mean counts are:

- semantic selected/materialized: 30.902:1;
- materialized/backend survivor: 1.118:1;
- semantic selected/backend survivor: 34.541:1.

The filter thus removes 96.76% of semantically selected rows before full
properties are assembled while retaining all backend survivors.

## Stage timing and bottlenecks

The 160k path decomposes into 6.590 ms semantic selection (54.83% of the mean
frame), 2.283 ms prefilter (19.00%), 0.590 ms residency/gather/H2D (4.91%),
0.148 ms assembly (1.23%), 1.319 ms rasterization (10.98%), and 1.088 ms other
measured overhead (9.05%). The ranking is therefore:

1. semantic CPU HSPT/SPT selection;
2. CUDA pre-materialization filtering;
3. rasterization;
4. other frame overhead;
5. residency/gather/H2D;
6. assembly.

The same conclusion holds in all four configurations. CPU semantic selection
is the dominant controllable stage at 6.47–6.59 ms/frame. Upper traversal is
about 1.14–1.21 ms and per-SPT cut work is about 5.02–5.09 ms, consistent with
the earlier approximately 6.40/4.74 ms profile. This campaign did not optimize
the selector.

For the 160k path, the fast-filter sub-stages average 0.134 ms selected-ID
preparation, 0.289 ms selected-ID H2D, 0.135 ms compact metadata indexing,
1.120 ms projection/culling, 0.285 ms mask compaction, 0.169 ms retained-index
D2H, and 0.032 ms output-ID production. The 2.283 ms total includes the small
remaining orchestration/synchronization overhead.

## Cache trace, cold population, and steady state

The formal campaign independently reproduced the reduced-cache sizing inputs:

- maximum single-frame hierarchy materialization: 50,936 rows;
- ten-camera hierarchy union: 151,621 rows;
- fixed capacity: 160,000 rows;
- headroom: 8,379 rows, or 5.526% of the measured union;
- evictions: zero.

Both bounded caches reach 95% of their useful trajectory occupancy after nine
warmup cameras and the full 151,621-row union after camera ten. Per fresh
process they upload 151,621 property rows / 13.30 MiB during the complete
warmup, including only 2,018 rows / 0.177 MiB on the first cold camera. They
then have 300/300 zero-transfer measured frames per configuration.

The bounded-cache hit rate is 19.44% during initial population, 100% over the
accepted steady frames, and 92.68% when all three warmups and all 300 measured
frames are combined. The earlier trace-simulation value of 91.94% is therefore
not copied as a formal measurement; the independently measured full-campaign
combined value is 92.68%. Zero evictions holds in every phase.

Because the full trajectory union fits, this campaign says nothing about
replacement-policy quality. It establishes capacity/memory mechanics and
zero-transfer steady reuse only for this frozen trajectory.

Previous-working-set reuse is not LRU and behaves very differently after
filtering. It achieves only a 12.56% steady exact-row reuse rate, transfers
433.16 MiB over 300 frames, and has no zero-transfer navigation frame.

One resident replicate performed the one-time 61.98-second gsplat CUDA
extension setup during its first warmup frame. That initialization outlier is
preserved in the artifact but excluded from every accepted steady statistic.
The other resident cold frames were 466.82 and 468.17 ms; host-mode cold frame
medians are approximately 533–550 ms. Cache-population rows and bytes are not
affected by the compilation outlier.

## Memory decomposition

| Configuration | Selection metadata MiB | Compact culling MiB | Skybox properties MiB | Dynamic full-property MiB | Persistent total MiB | Mean frame-local properties MiB | Mean selected/materialization indices MiB | Peak allocated MiB | Peak reserved MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Resident | 8.39 | N/A | included | included | 104.90 | N/A | 4.44 | 230.76 | 288.00 | 1,606.48 |
| Fast + previous reuse | 0 | 16.78 | 8.77 | 1.65 mean | 27.21 mean | 3.10 | 4.75 | 93.21 | 114.00 | 1,888.38 |
| Fast + 525k cache | 0 | 16.78 | 8.77 | 46.06 | 71.62 | 3.31 | 4.73 | 139.89 | 160.00 | 1,906.30 |
| Fast + 160k cache | 0 | 16.78 | 8.77 | 14.04 | 39.60 | 3.31 | 4.73 | 108.45 | 134.00 | 1,907.89 |

The compact table is exactly 17,599,984 bytes: 16 logical bytes for each of
1,099,999 Gaussians. It is reported separately and never counted as full
Gaussian properties. Host modes also retain the 101,199,908-byte canonical
property store in CPU RAM. Residency maps are CPU-side in this implementation;
their GPU allocation is zero. Maximum inferred rasterizer temporary allocation
increments are about 57–60 MiB.

PyTorch peak allocation provides the cleanest same-process memory result. The
525k-to-160k change saves 33,580,000 persistent full-property bytes (32.02 MiB)
and reduces peak PyTorch allocation by 32,960,512 bytes (31.43 MiB). PyTorch
reservation falls from 160 to 134 MiB.

Whole-device NVML peaks also fall in every matched 525k/160k replicate pair,
by roughly 20–26 MiB. The maximum-over-campaign values are 1,804.06 versus
1,783.88 MiB. WSL process-level NVML accounting still reports zero and is
flagged unreliable, so the whole-device values are supporting isolated-run
evidence rather than exact process attribution. Peak RSS varies independently
of the GPU cache size and is not used to claim host-memory savings.

## Experimental answers

### Does 525k to 160k save memory without harming throughput or traffic?

Yes on this trajectory. The smaller cache saves 32.02 MiB of persistent full
properties and 31.43 MiB of peak PyTorch allocation, while both policies have
identical population traffic, zero steady H2D, a 100% steady hit rate, and zero
evictions. Mean frame time improves from 12.465 to 12.018 ms (0.447 ms, 3.59%)
rather than regressing. The improvement comes mainly from cache-management
time falling from 0.441 to 0.176 ms. This is not a replacement-policy result,
because neither cache replaces a row.

### Does bounded caching help over previous-set reuse?

Yes for this filtered workload, unlike the earlier unfiltered campaign. The
160k cache eliminates 454,202,160 bytes (433.16 MiB) of accepted steady H2D,
raises steady reuse from 12.56% to 100%, and reduces mean frame time by 0.862 ms
(6.70%). Its cost is approximately 12.39 MiB more mean persistent GPU memory
than previous reuse. Explicit bounded caching is therefore a useful
traffic/throughput trade on this trajectory, not an unconditional winner.

### What is the remaining bottleneck?

CPU semantic HSPT/SPT selection is clearly dominant. In the production 160k
configuration it consumes 54.83% of mean frame time, compared with 19.00% for
the fast prefilter and 4.91% for residency/gather/H2D. The next bounded task is
fork A: optimize the production CPU selection path while preserving exact
CPU-reference/oracle parity. That optimization is not performed here.

## Validation and reproducibility

- 211 full project unit tests passed;
- 14 opt-in external ALoD integration tests passed;
- 79 focused CityGaussian regressions passed;
- `git diff --check` passed;
- protected official repositories, Run005/oracles/references, prior ALoD
  artifacts, and CityGaussian behavior/results were unchanged;
- nothing was staged, committed, or pushed.

Machine-readable results live in `tables/config_summary.json`; detailed
memory, cache, cold/warm, and bottleneck tables are separate JSON files. All
1,200 accepted per-frame records and all 120 warmup records remain under
`runs/` with hashes in `manifest.json`.
