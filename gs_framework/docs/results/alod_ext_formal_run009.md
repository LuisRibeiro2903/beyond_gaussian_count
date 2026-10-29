# Formal Run009 ALOD-EXT Runtime Evaluation

Status: `PASS-ALOD-EXT-FORMAL-RUN009`

Campaign: `/home/luis/tese/benchmark_artifacts/alod_ext_formal_run009_001`

## Result

The best existing extension is `ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS` at 71.66 ms / 13.96 FPS. Relative to frozen `ALOD-REPRO-HOST-NOCACHE` (186.77 ms / 5.35 FPS), it saves 115.12 ms (61.63%) and is 2.61x faster.

All modes preserve exact semantic ALoD selection. The 16-camera gate found exact ordered oracle IDs, selected/touched SPT parity, bitwise-exact surviving properties, zero backend false negatives, and bitwise-exact RGB/alpha. CPU and CUDA filters produced exact ordered IDs on all 71 workload cameras.

## Frozen protocol and capacity sizing

The campaign reuses the faithful 71-camera order, per-camera resolution, packed gsplat backend, ten ordered warmups, synchronization, and three fresh processes. Run005 remains development evidence; this is the formal mixed-scale Run009 evaluation. Capacities were frozen from count traces before timing:

- maximum post-filter hierarchy working set: 1,599,906 rows;
- mean / p95: 509,957 / 1,242,482 rows;
- trajectory union: 3,503,048 rows;
- mean consecutive required-row overlap: 49.67%;
- minimum / practical / larger sensitivity capacities: 1,599,906 / 3,700,000 / 5,300,000 rows.

## Thesis-facing comparison

| Configuration | Mean / median / p95 ms | FPS | Semantic rows | Materialized rows | Projection survivors | Steady H2D MiB | Property + filter MiB | Peak allocated MiB | Whole-device peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALOD-REPRO-HOST-NOCACHE` | 186.77 / 184.87 / 207.52 | 5.35 | 3,532,141 | 3,532,141 | 496,182 | 64,140.5 | 8.8 | 1,312.1 | 4,058.2 |
| `ALOD-REPRO-PREVIOUS-REUSE` | 134.21 / 131.58 / 160.44 | 7.45 | 3,556,309 | 3,556,309 | 498,077 | 1,636.5 | 314.5 | 1,315.1 | 4,656.7 |
| `ALOD-EXT-ROW-LRU-PRACTICAL` | 134.54 / 132.68 / 145.60 | 7.43 | 3,532,141 | 3,532,141 | 496,182 | 12.3 | 333.4 | 1,691.2 | 4,290.9 |
| `ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL` | 220.50 / 213.79 / 265.99 | 4.54 | 3,532,141 | 525,049 | 496,182 | 774.2 | 333.4 | 984.4 | 2,735.5 |
| `ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS` | 71.66 / 66.72 / 124.03 | 13.96 | 3,532,141 | 525,049 | 496,182 | 3,057.6 | 257.1 | 768.1 | 2,888.9 |
| `ALOD-EXT-PREMATERIALIZATION-CUDA-MINIMUM` | 84.38 / 84.73 / 138.03 | 11.85 | 3,532,141 | 525,049 | 496,182 | 2,049.5 | 257.1 | 922.4 | 2,916.9 |
| `ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL` | 73.48 / 68.94 / 98.73 | 13.61 | 3,532,141 | 525,049 | 496,182 | 774.2 | 441.3 | 1,107.2 | 3,098.9 |
| `ALOD-EXT-PREMATERIALIZATION-CUDA-LARGE` | 76.29 / 71.99 / 101.34 | 13.11 | 3,532,141 | 525,049 | 496,182 | 774.2 | 581.7 | 1,247.1 | 3,238.9 |

## Gaussian-count funnel

On the fast practical-cache path, mean semantic selection is 3,532,141 rows, mean materialization is 525,049, and packed-projection survival is 496,182. The ratios are 6.727:1 semantic/materialized, 7.119:1 semantic/survivor, and 1.058:1 materialized/survivor.

The filter acts only after exact semantic selection. It reduces full-property materialization, not the LoD cut.

## Incremental attribution

| From → to | Saved ms | Time reduction | Steady H2D change MiB | Persistent property+filter change MiB |
| --- | ---: | ---: | ---: | ---: |
| `ALOD-REPRO-HOST-NOCACHE` → `ALOD-REPRO-PREVIOUS-REUSE` | 52.57 | 28.14% | -62,504.0 | 305.7 |
| `ALOD-REPRO-HOST-NOCACHE` → `ALOD-EXT-ROW-LRU-PRACTICAL` | 52.23 | 27.96% | -64,128.2 | 324.6 |
| `ALOD-EXT-ROW-LRU-PRACTICAL` → `ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL` | -85.96 | -63.89% | 761.9 | 0.0 |
| `ALOD-EXT-PREMATERIALIZATION-CPU-PRACTICAL` → `ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL` | 147.02 | 66.68% | 0.0 | 107.9 |
| `ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL` → `ALOD-EXT-PREMATERIALIZATION-CUDA-MINIMUM` | -10.90 | -14.83% | 1,275.3 | -184.3 |
| `ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL` → `ALOD-EXT-PREMATERIALIZATION-CUDA-LARGE` | -2.81 | -3.82% | 0.0 | 140.4 |
| `ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL` → `ALOD-EXT-PREMATERIALIZATION-CUDA-PREVIOUS` | 1.82 | 2.48% | 2,283.4 | -184.3 |

## H2D and memory

For the fast practical cache across three processes, cold first-frame, remaining warmup, and measured traffic are 14.5, 133.3, and 774.2 MiB. Steady hits/misses/evictions/insertions are 99,796,944 / 8,823,972 / 0 / 8,823,972; 66/213 frames transfer no property rows.

The practical and large caches have identical one-pass union coverage; the large cache is only a sensitivity control. The minimum cache lowers persistent property memory but incurs evictions and higher traffic. Allocated, reserved, whole-device VRAM, and RSS remain separate in the machine-readable table; whole-device claims use only measured NVML values and are not treated as process-attributed under WSL.

## Remaining bottleneck

For the fastest mode, semantic HSPT/SPT selection is the largest stage at 38.19 ms, or 53.30% of total frame time. The filter is 8.04 ms, residency/gather/H2D 20.17 ms, assembly 0.73 ms, and rasterization 3.50 ms.

## Thesis questions

- Q1/Q4: only 14.86% of semantically selected rows are materialized on average; the exact conservative decision sharply reduces frame-local property work.
- Q2: practical bounded residency reduces measured H2D by 98.79% versus faithful no-cache over the matched 213 frames.
- Q3: yes. Exact semantic selection is preserved with all three bounded capacities; capacity changes only residency/traffic.
- Q5: yes in persistent full-property bytes, but the 1.60M minimum cache trades the reduction for evictions and additional traffic; 3.70M is the practical zero-eviction union-sized choice.
- Q6: best formal improvement is 115.12 ms (61.63%), from 5.35 to 13.96 FPS.
- Q7: semantic selection remains dominant at 38.19 ms (53.30%).

## Interpretation boundary

Run009 is `WEAK-LOD`. These are the measured behaviors of the validated formal hierarchy and fixed workload; they do not establish unchanged performance for a stronger-LoD hierarchy. Official ALoD, the faithful campaign, Run005 history, the hierarchy, oracle, datasets, and cameras remain untouched. Every evaluated optimization is classified `ALOD-EXT-*`.

## Validation

- 217 project unit tests passed (2 skipped);
- all 14 opt-in external ALoD integration tests passed; the first combined invocation exposed the known system-CUDA `sm_120` JIT limitation, and the affected two-test fast-filter class passed on retry with the already-built pinned gsplat extension used by the campaign;
- 82 focused CityGaussian/shared-runtime regressions passed;
- Run009 correctness gate: `PASS-ALOD-EXT-FORMAL-RUN009-CORRECTNESS`;
- cache-sizing gate: `PASS-ALOD-EXT-RUN009-CACHE-SIZING`;
- `git diff --check` passed;
- nothing was staged, committed, or pushed.

## Next task

Decide whether the dominant remaining bottleneck justifies a new semantic-selection optimization, and if so design it as the next ALOD-EXT contribution with exact oracle parity as a hard constraint. No such optimization was implemented here.
