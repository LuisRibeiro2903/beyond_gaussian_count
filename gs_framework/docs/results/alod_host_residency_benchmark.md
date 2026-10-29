# Formal ALoD Host Residency Benchmark

Status: `PASS-ALOD-HOST-RESIDENCY-BENCHMARK`

Campaign: `ALOD-HOST-RESIDENCY-BENCHMARK-001`

Artifacts: `/home/luis/tese/benchmark_artifacts/alod_host_residency_benchmark_001`

## Scope and protocol

This campaign compares the validated Run005 ALoD hierarchy under four Gaussian-property residency modes. It changes neither the HSPT nor LoD/frustum selection, selected global-node order, skybox policy, cameras, renderer, nor raster settings.

- Asset: `final.dhier`, SHA-256 `6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c`
- Trajectory: the unchanged ten-camera `alod_run005_frozen_evaluation_v1`, SHA-256 `c6d607b8baee0cc30f3c74121e9d19ba045a820fc87c8550a11c3dbc0c92968d`
- Resolution: 960×540
- Per process: 10 warmup frames followed by 10 complete trajectory repetitions (100 measured frames)
- Replication: three independent processes/configuration, 300 measured frames/configuration and 1,200 accepted frames total
- Ordering: the 12-process deterministic interleaving was frozen before execution; each mode occupied three distinct order positions and no mode was always first or last
- Bounded cache: 525,000 hierarchy rows; minimum feasible frame working set 499,030; nominal headroom 25,970; 100,000 skybox rows resident separately
- Timing: synchronous completed GPU work; formal frames use CUDA events; the first existing warmup frame is separately recorded as cold and is not included in steady-state statistics

`WEAK-LOD-ORACLE` remains a limitation. The campaign establishes residency, reuse, timing, memory, and correctness behavior on this frozen trajectory, not strong multi-scale LoD adaptation.

## Correctness gate

The preflight returned `PASS-ALOD-HOST-RESIDENCY`. For all modes, selected IDs, raster order, pre-raster property hashes, and backend-visible counts match the resident path.

- 30 representative resident/host render comparisons were bitwise identical: maximum absolute error 0, PSNR infinite, SSIM 1.0.
- 97 renders reached through predefined alternative camera histories were bitwise identical with exact backend-visible counts: maximum absolute error 0, PSNR infinite, SSIM 1.0.
- No output depends on previous-working-set or bounded-cache history.

## Steady-state results

All memory units below are MiB/GiB (powers of two). H2D totals cover the 300 accepted steady-state frames/configuration; cold and warmup transfers are reported separately.

| Configuration | Mean ms | Median ms | P95 ms | FPS | Selection ms | Residency lookup ms | Cache mgmt ms | Host gather ms | H2D ms | Assembly ms | Raster ms | Selected hierarchy | Backend-visible | Persistent properties MiB | Peak device MiB | Peak RSS MiB | Total H2D GiB | Hit/reuse rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ALOD-HIER-LOD-FRUSTUM` | 9.415 | 9.396 | 10.554 | 106.21 | 6.366 | N/A | N/A | N/A | N/A | 0.020 | 1.224 | 482,252 | 16,857 | 96.51 | 1,828.57 | 1,565.07 | 0 | N/A |
| `ALOD-HOST-NOCACHE` | 21.466 | 20.132 | 28.462 | 46.59 | 6.681 | 0.146 | N/A | 8.633 | 3.069 | 0.194 | 1.325 | 482,252 | 16,857 | 8.77 | 1,840.57 | 1,709.79 | 12.396 | N/A |
| `ALOD-HOST-PREVIOUS-REUSE` | 14.558 | 14.368 | 17.215 | 68.69 | 6.347 | 1.257 | 1.391 | 0.677 | 0.805 | 0.185 | 1.206 | 482,252 | 16,857 | 51.09 mean | 1,831.88 | 1,691.63 | 0.544 | 95.61% |
| `ALOD-HOST-BOUNDED-CACHE` | 14.509 | 14.600 | 16.023 | 68.92 | 6.401 | 2.227 | 1.827 | 0.039 | 0.043 | 0.168 | 1.245 | 482,252 | 16,857 | 54.84 | 1,876.57 | 1,674.57 | 0 | 100% |

Selected hierarchy rows are not backend-visible rows. The mean selected-to-visible ratio is about 28.6:1 (482,252 versus 16,857), and all four modes have exactly the same counts.

## Memory

Resident properties occupy 101,199,908 bytes (96.51 MiB).

- No-cache keeps only the 9,200,000-byte skybox property prefix persistently on GPU: 91,999,908 bytes, or 90.91%, less than resident.
- Previous reuse uses 53,567,184 bytes on average (51.09 MiB), varying with the prior selected set: 47,632,724 bytes, or 47.07%, less than resident on average.
- The bounded cache uses 57,500,000 bytes (54.84 MiB): 43,699,908 bytes, or 43.18%, less than resident.
- Each host mode also stores the canonical 101,199,908 property bytes in CPU RAM. Pinned memory is disabled. CPU HSPT/hierarchy metadata is 38,458,208 bytes; residency metadata reaches about 3.99 MB no-cache, 12.79 MB previous reuse, and 21.19 MB bounded cache.

Persistent property memory falls as designed, but peak total device memory does not. Relative to resident's 1,828.57 MiB device peak, no-cache is about 12.0 MiB higher, previous reuse 3.3 MiB higher, and bounded cache 48.0 MiB higher. The host paths still materialize large frame-local pre-raster property arrays; bounded cache additionally holds its fixed cache while gathering the selected raster input. Therefore this implementation demonstrates bounded persistent residency, but not yet a lower total VRAM peak.

## Transfer and cache behavior

No-cache transfers 13,310,155,200 bytes across 300 measured frames: 44,367,184 bytes/frame on average (42.312 MiB), with no zero-transfer frames.

Previous reuse transfers 584,623,200 bytes (0.544 GiB), eliminating 12,725,532,000 bytes or 95.61% of no-cache traffic. Its overall exact-row reuse hit rate is 95.61%; mean previous/current retention is 95.67%. There are no zero-transfer navigation frames because each ordered camera transition changes at least some rows.

After the required ten-frame warmup traverses all ten cameras, the bounded cache contains 518,746 distinct hierarchy rows (98.81% occupancy). That union fits below 525,000, so all 300 measured frames are cache hits: zero measured H2D bytes and zero evictions. Its separately reported cold first frame still uploads 33,520,844 bytes. This is a consequence of the frozen trajectory and warmup—not evidence that larger scenes or trajectories will fit the same cache.

## Cold start

Mean host-store construction is 147.20 ms for no-cache, 131.69 ms for previous reuse, and 135.33 ms for bounded cache. Mean bounded-cache allocation is 5.47 ms; skybox transfer is roughly 2.1–2.3 ms.

The first host frame requests 364,357 hierarchy rows and uploads 33,520,844 bytes. Mean first-frame completed H2D time is 3.28 ms for no-cache, 62.02 ms for previous reuse, and 63.62 ms for bounded cache; the reuse/cache modes additionally construct persistent GPU state. First-frame GPU raster time, including first-use initialization, is 372.46, 339.97, and 327.25 ms respectively (resident 371.72 ms). These cold values are reported separately and excluded from steady-state timing.

## Answers to the research questions

### How much memory does host residency save?

Persistent Gaussian-property storage falls by 90.91% for no-cache, 47.07% on average for previous reuse, and 43.18% for the bounded cache. The current full-frame materialization path does not reduce peak total VRAM, so persistent-memory scaling and total-memory scaling must not be conflated.

### What traffic and performance cost does no-cache impose?

No-cache moves 12.396 GiB over 300 frames and raises mean frame time from 9.415 to 21.466 ms: +12.051 ms or +127.99%, reducing 106.21 FPS to 46.59 FPS. Host gather (8.633 ms) is its dominant measured stage, followed by selection (6.681 ms) and H2D (3.069 ms). It is a useful minimum-persistence baseline, not a desirable final policy.

### How much does temporal reuse recover?

Previous reuse removes 95.61% of no-cache H2D traffic and recovers 6.908 ms of the no-cache penalty. It runs at 14.558 ms / 68.69 FPS, still 5.143 ms (54.62%) slower than resident.

### Is the bounded cache worth its extra complexity here?

Not materially for frame time. Against previous reuse, it removes the remaining 584,623,200 measured H2D bytes and raises the hit rate by 4.39 percentage points, but uses 3,932,816 more persistent property bytes and adds about 1.407 ms of lookup/management time. Mean frame time improves by only 0.049 ms (0.34%), well inside the replicate-level variation. The bounded cache is valuable as an explicit hard residency bound and achieves zero steady transfers on this trajectory, but it does not materially outperform the simpler previous-set policy in throughput.

### What remains the bottleneck?

For resident, previous reuse, and bounded cache, CPU selection is the largest measured stage at about 6.35–6.40 ms/frame. No-cache is instead dominated by host gather. The 28.6:1 selected/backend-visible gap also explains why frame-local property materialization prevents total VRAM from falling despite bounded persistent storage.

## Measurement audit

Two complete preliminary attempts are preserved under `rejected/` and excluded from every accepted statistic:

1. `diagnostic_validation_enabled`: rejected because timed host frames included an expensive diagnostic duplicate/mapping validation with no resident equivalent.
2. `missing_cold_raster_event`: steady-state frames were valid, but the cold frame lacked the required CUDA-event raster metric.

The accepted campaign reran all four configurations from scratch after each instrumentation fix; no partial results were reused. The formal correctness gate still runs exhaustive mapping/property validation outside timed frames.

Final validation passed 192 full project unit tests (including the CUDA renderer smoke), 12 external ALoD integration tests, 63 focused CityGaussian regressions, and `git diff --check`. Preflight/final hashes match for all 13 protected paths and both clean official repositories (`PASS-PROTECTED-STATE-UNCHANGED`). Nothing was staged, committed, or pushed.

## Next bounded task

Do not scale the asset yet. The next bounded task should characterize and reduce the selected-to-backend-visible gap without changing LoD semantics—starting with visibility/workload accounting that can avoid materializing roughly 482k properties when only about 16.9k reach the backend—while separately profiling the 6.4 ms CPU HSPT/selection stage. The measured total-VRAM peak must fall before claiming end-to-end memory scalability.

Reproducible machine-readable results are in `tables/config_summary.json`; cold, transfer, cache, and memory summaries are separate JSON files, and the accepted per-frame JSONL files remain under `runs/`.
