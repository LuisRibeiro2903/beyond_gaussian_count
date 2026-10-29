# Benchmark Protocol

Generated: 2026-07-06

This protocol freezes the first real-scene benchmark configurations for future incremental rendering-method comparisons. It covers the current flat `gsplat` path, CityGaussian all-cells-active parents, CityGaussian V1 runtime visibility, and the released-code adaptive LoD validation configuration. It does not implement paper fixed-distance LoD, hierarchy traversal, FlashGS, host residency, or a new cache.

2026-07-08 update: CityGaussian V1 runtime visibility is now implemented as a non-formal validation configuration. Adaptive LoD, hierarchy traversal, FlashGS, host residency, streaming, VQ reconstruction, and runtime cache behavior remain out of scope for these configs.

2026-07-28 update: `CG-LOD-RELEASE` implements the released V1 Nyquist policy with all three PLY levels resident. Paper fixed-distance LoD, streaming, host residency, VQ reconstruction, hierarchy traversal, and FlashGS remain out of scope.

2026-07-30 update: the released-code CityGaussian V1 branch completed the formal 1,440-frame campaign. The persistent package is `/home/luis/tese/benchmark_artifacts/citygs_formal_bench_20260730_121148/`, and the thesis memo is `docs/results/citygaussian_v1_released_benchmark_memo.md`.

2026-08-20 update: the formal materialization-aware ALoD residency campaign
completed as `ALOD-MATERIALIZATION-AWARE-RESIDENCY-BENCHMARK-001`. It preserves
the accepted 12-process/1,200-frame Run005 protocol while comparing resident,
CUDA-filtered previous-working-set reuse, CUDA-filtered 525k cache, and
CUDA-filtered 160k cache modes. Artifacts are under
`/home/luis/tese/benchmark_artifacts/alod_materialization_aware_residency_benchmark_001/`;
the thesis memo is `docs/results/alod_materialization_aware_residency_benchmark.md`.

2026-08-22 update: the public-source faithful ALoD ladder completed as
`ALOD-FAITHFUL-REPRODUCTION-001`. It freezes three 71-camera, three-replicate
rendering modes (`ALOD-REPRO-HOST-NOCACHE`, `ALOD-REPRO-NOFRUSTUM`, and
`ALOD-REPRO-PREVIOUS-REUSE`), official-path quality, a supplementary LoD scan,
and the HSPT/BFS cut ablation. The persistent package is
`/home/luis/tese/benchmark_artifacts/alod_faithful_reproduction_001/`, and the
thesis memo is `docs/results/alod_faithful_reproduction.md`. No `ALOD-EXT`
configuration is part of this ladder, and paper `Render Full` remains
unavailable from the released rendering entry points.

2026-08-23 update: `ALOD-EXT-FORMAL-RUN009-001` formally evaluates the existing
project runtime extensions on the same Run009 hierarchy and unchanged faithful
71-camera workload. Six modes each use three fresh processes, ten ordered
warmups, one measured 71-camera pass, packed gsplat, and synchronized timing.
The minimum/practical/larger property-cache capacities (1,599,906 / 3,700,000 /
5,300,000 rows) were frozen from the pre-timing working-set trace. Artifacts are
under `/home/luis/tese/benchmark_artifacts/alod_ext_formal_run009_001/`; the
memo is `docs/results/alod_ext_formal_run009.md`. This campaign does not modify
semantic selection and remains wholly `ALOD-EXT`.

## Configurations

### `FLAT-REAL-FIT`

Purpose:

- operational real-scene flat baseline;
- validates scripted camera replay, output files, metrics, and benchmark semantics;
- uses the current `SceneLoader.load(..., scene_type="flat")` path and `GsplatRenderer`;
- activates every loaded Gaussian every frame.

Tracked preset:

```text
benchmarks/configs/flat_real_fit.json
```

External asset path is supplied by `--asset` or `GS_FRAMEWORK_FLAT_REAL_FIT_ASSET`.

Observed short validation asset:

```text
/home/luis/tese/gs_framework/cleanup/splatfacto_cleanup/pruned.pt
```

This is a project-trained flat checkpoint with `1,227,788` Gaussians. The path is not stored in the tracked preset.

### `FLAT-CITYGS-FULL`

Purpose:

- feasibility test for an author-provided CityGaussian fused PLY as one fully resident flat scene;
- motivates spatial visibility, LoD, hierarchy, or other large-scene methods when the flat path cannot handle the asset cleanly;
- does not partition the PLY, reduce precision, use LoD, use block culling, or fall back to CPU rendering.

Tracked preset:

```text
benchmarks/configs/flat_citygs_full.json
```

External asset path is supplied by `--asset` or `GS_FRAMEWORK_FLAT_CITYGS_FULL_ASSET`.

Observed feasibility asset:

```text
/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply
```

The PLY header reports `23,389,621` vertices and the file size is `5,800,627,541` bytes. The path is not stored in the tracked preset.

### `CG-FINEST-FLAT`

Purpose:

- same-asset flat reference for the finest released CityGaussian V1 LoD PLY;
- loads the `mc_aerial_c36_light_50_vq/point_cloud.ply` asset through the existing flat scene path;
- activates every Gaussian every frame;
- performs no CityGaussian cell selection, visibility, adaptive LoD, host residency, hierarchy traversal, streaming, or VQ reconstruction.

Tracked preset:

```text
benchmarks/configs/cg_finest_flat.json
```

External asset path is supplied by `--asset` or `GS_FRAMEWORK_CG_FINEST_FLAT_ASSET`.

Expected asset:

```text
/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq/point_cloud.ply
```

The PLY header reports `11,694,810` vertices and SH degree 2. The path is not stored in the tracked preset.

### `CG-FIXED-FINEST-ALL`

Purpose:

- same-asset official-cell reference for the finest released CityGaussian V1 LoD PLY;
- loads the same PLY as `CG-FINEST-FLAT`;
- constructs official runtime cells with `block_dim=[6,6,1]`, AABB `[-3.5,-4,-10,4.5,2,10]`, contraction to unisphere, x-fastest cell IDs, MAD bounds with `n=4`, and average activated scale;
- activates all 36 cells every frame, so selected Gaussian count equals total Gaussian count;
- performs no projected visibility testing, adaptive LoD, host residency, hierarchy traversal, streaming, custom project partitioning, or VQ reconstruction.

Tracked preset:

```text
benchmarks/configs/cg_fixed_finest_all.json
```

External asset path is supplied by `--asset` or `GS_FRAMEWORK_CG_FIXED_FINEST_ALL_ASSET`.

### `CG-FIXED-FINEST-VISIBILITY`

Purpose:

- child of `CG-FIXED-FINEST-ALL`;
- uses the same finest released LoD PLY, same official 36 cells, same camera path, same SH degree, same `gsplat` renderer, and same render options;
- changes only official CityGaussian V1 projected-corner visibility selection;
- keeps all Gaussian data resident and passes only selected cell tensors to the renderer;
- performs no adaptive LoD, host residency, streaming, VQ reconstruction, custom project partitioning, or FlashGS.

Tracked preset:

```text
benchmarks/configs/cg_fixed_finest_visibility.json
```

External asset path is supplied by `--asset` or `GS_FRAMEWORK_CG_FIXED_FINEST_VISIBILITY_ASSET`.

### `CG-FUSED-CELLS-ALL`

Purpose:

- controlled all-cells parent for the fused no-LoD CityGaussian PLY used by `FLAT-CITYGS-FULL`;
- constructs the same official `block_dim=[6,6,1]` cells from the fused PLY and activates all 36 cells;
- performs no visibility, no LoD, no host residency, no streaming, and no VQ reconstruction.

Tracked preset:

```text
benchmarks/configs/cg_fused_cells_all.json
```

External asset path is supplied by `--asset` or `GS_FRAMEWORK_CG_FUSED_CELLS_ALL_ASSET`.

### `CG-LOD-RELEASE`

Purpose:

- controlled child of `CG-FIXED-FINEST-VISIBILITY`;
- loads released `light_50_vq`, `light_66_vq`, and `light_75_vq` PLY levels in that order;
- uses coarsest-level MAD bounds for the same official visibility path;
- applies strict first-match released Nyquist LoD per visible cell, with coarsest fallback;
- keeps all 25,494,686 Gaussians resident and assembles only the selected cell tensors for `GsplatRenderer`;
- performs no paper fixed-distance LoD, VQ reconstruction, host residency, streaming, `BlockManager`, or FlashGS.

Tracked preset:

```text
benchmarks/configs/cg_lod_release.json
```

Supply external paths with `--lod-finest-asset`, `--lod-middle-asset`, and `--lod-coarsest-asset`, or the three environment variables recorded in the preset.

### `CG-FUSED-VISIBILITY`

Purpose:

- child of `CG-FUSED-CELLS-ALL`;
- uses the same fused 23,389,621-Gaussian PLY, official cells, camera path, SH degree 3, renderer, and render options;
- changes only official CityGaussian V1 projected-corner visibility selection;
- keeps all Gaussian data resident and passes only selected cell tensors to the renderer;
- performs no adaptive LoD, host residency, streaming, VQ reconstruction, custom project partitioning, or FlashGS.

Tracked preset:

```text
benchmarks/configs/cg_fused_visibility.json
```

External asset path is supplied by `--asset` or `GS_FRAMEWORK_CG_FUSED_VISIBILITY_ASSET`.

## Camera Paths

Tracked camera paths:

```text
benchmarks/camera_paths/matrixcity_1m2_navigation_v1.json
benchmarks/camera_paths/citygaussian_23m_navigation_v1.json
```

Both are copied from existing manual Viser exports under `results/camera_paths/`. Original values are preserved. At runtime, the benchmark applies the deterministic Viser-export conversion already used by the external parent benchmark script: flip camera Y/Z axes and multiply translation by `scale_ratio`.

| Configuration | Source file | Frames | Resolution | FOV | Keyframes | Compatibility |
|---|---|---:|---:|---:|---:|---|
| `FLAT-REAL-FIT` | `results/camera_paths/simplemc_aerial_nav_v1.json` | 1080 | `1920 x 1080` | 50 deg vertical | 19 | Authored for the approximately 1.2M MatrixCity/splatfacto scene. |
| `FLAT-CITYGS-FULL` | `results/camera_paths/realmc_aerial_nav_v1.json` | 1440 | `1920 x 1080` | 50 deg vertical | 25 | Authored for the larger CityGaussian MatrixCity aerial asset. |
| `CG-FINEST-FLAT` | `results/camera_paths/realmc_aerial_nav_v1.json` | 1440 | `1920 x 1080` | 50 deg vertical | 25 | Same trajectory as `FLAT-CITYGS-FULL`, used for the finest-LoD same-asset parent. |
| `CG-FIXED-FINEST-ALL` | `results/camera_paths/realmc_aerial_nav_v1.json` | 1440 | `1920 x 1080` | 50 deg vertical | 25 | Must use the same trajectory indices as `CG-FINEST-FLAT`. |

The two scenes use different spatial measurements, scale, or coordinate conventions. Do not force them to share camera matrices. Every future configuration using the same underlying asset must replay the exact same trajectory.

## Entry Point

```bash
python benchmarks/benchmark_real_scene.py \
  --config-id FLAT-REAL-FIT \
  --asset /path/to/external/checkpoint.pt \
  --output-dir /path/to/output
```

Short non-formal validation:

```bash
python benchmarks/benchmark_real_scene.py \
  --config-id FLAT-REAL-FIT \
  --asset /path/to/external/checkpoint.pt \
  --output-dir /tmp/flat_real_fit_validation \
  --num-warmup 1 \
  --max-frames 3
```

Feasibility attempt:

```bash
python benchmarks/benchmark_real_scene.py \
  --config-id FLAT-CITYGS-FULL \
  --asset /path/to/point_cloud.ply \
  --output-dir /tmp/flat_citygs_full_feasibility \
  --num-warmup 0 \
  --max-frames 1
```

Same-asset fixed-finest validation:

```bash
python benchmarks/benchmark_real_scene.py \
  --config-id CG-FINEST-FLAT \
  --asset /path/to/mc_aerial_c36_light_50_vq/point_cloud.ply \
  --output-dir /tmp/cg_finest_flat_validation \
  --num-warmup 1 \
  --max-frames 3

python benchmarks/benchmark_real_scene.py \
  --config-id CG-FIXED-FINEST-ALL \
  --asset /path/to/mc_aerial_c36_light_50_vq/point_cloud.ply \
  --output-dir /tmp/cg_fixed_finest_all_validation \
  --num-warmup 1 \
  --max-frames 3

python benchmarks/compare_citygaussian_same_asset.py \
  --asset /path/to/mc_aerial_c36_light_50_vq/point_cloud.ply \
  --output-dir /tmp/cg_fixed_finest_image_compare \
  --frame-index 0 \
  --frame-index 1 \
  --frame-index 2
```

The benchmark requires CUDA. It fails clearly when CUDA or the external asset is unavailable.

RTX 5080 validation note: in the current host environment, use the conda CUDA 12.9 compiler and headers when `gsplat` needs to JIT its CUDA extension. System `/usr/bin/nvcc` is CUDA 12.0 and rejects `compute_120`.

```bash
env \
  CUDA_HOME=/home/luis/miniconda3/envs/nerfstudio2 \
  PATH=/home/luis/miniconda3/envs/nerfstudio2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  CPATH=/home/luis/miniconda3/envs/nerfstudio2/targets/x86_64-linux/include \
  TORCH_EXTENSIONS_DIR=/tmp/torch_ext_citygs002 \
  /home/luis/miniconda3/envs/nerfstudio2/bin/python ...
```

Observed same-asset fixed-finest validation on 2026-07-06, not a formal benchmark:

- `CG-FINEST-FLAT`: 3 measured frames, finite RGB/alpha, 11,694,810 active Gaussians, mean frame time `10.463 ms`, mean visible/rendered Gaussians `2,212,332.33`, PyTorch peak allocated VRAM `2,780,806,144` bytes.
- `CG-FIXED-FINEST-ALL`: 3 measured frames, finite RGB/alpha, 36 active cells, 11,694,810 selected Gaussians, mean frame time `10.541 ms`, mean preparation time `11.273 ms`, cell construction `870.297 ms`, mean visible/rendered Gaussians `2,212,332.33`, PyTorch peak allocated VRAM `4,652,319,232` bytes.
- Same-frame comparison: frames `[0, 1, 2]`, RGB max/mean absolute diff `0.0`, alpha max/mean absolute diff `0.0`, visible counts exactly matched per frame.

Observed Spec 003 visibility validation on 2026-07-08, not a formal benchmark:

- Five-Gaussian CUDA smoke passed on the host RTX 5080 with finite RGB/alpha and deterministic RGB/alpha max absolute difference `0.0`.
- Runtime selected-set validation against `finest_visibility_oracle_v1.json`: 12/12 frames passed; selected cells `9..35`; selected Gaussians `3,248,918..11,694,702`; fallback case trajectory index `1177` passed.
- Runtime selected-set validation against `fused_visibility_oracle_v1.json`: 12/12 frames passed; selected cells `10..35`; selected Gaussians `6,422,577..23,389,336`; fallback cases trajectory indices `654` and `785` passed.
- `FLAT-CITYGS-FULL` vs `CG-FUSED-CELLS-ALL` over the 12 oracle cameras: total and assigned Gaussian count `23,389,621`, unassigned `0`, duplicated `0`, all 36 cells nonempty, RGB/alpha max absolute difference `0.0`.
- `CG-FIXED-FINEST-VISIBILITY` render sample against all-cells parent: finite RGB/alpha on all 12 frames, max RGB absolute difference `0.2814`, max mean RGB absolute difference `0.03694`, minimum PSNR `22.29`.
- `CG-FUSED-VISIBILITY` render sample against all-cells parent: finite RGB/alpha on all 12 frames, max RGB absolute difference `0.4926`, max mean RGB absolute difference `0.03625`, minimum PSNR `22.31`.
- 12-frame sampled benchmark summaries under `/tmp/citygs003_bench_*` record resident/selected/active counts, visibility time, assembly time, rasterization time, and PyTorch/NVML memory fields. These are non-formal validation runs, not thesis benchmark results.

Observed Spec 004 validation on 2026-07-28, not a formal benchmark:

- release selected-set oracle: 12/12 exact matches, including fallback state; level-use totals `[94, 14, 94]`; selected count `1,957,671..6,211,124`;
- `CG-LOD-RELEASE`: finite RGB/alpha for all 12 cameras, mean selected count 4,072,361, resident count 25,494,686;
- separate parent/child samples: mean end-to-end time `87.33 -> 73.33 ms`, FPS `11.45 -> 13.64`, post-transfer allocated memory `1.97 -> 4.29 GB`, render peak allocated `5.22 -> 6.07 GB`;
- image comparison: RGB max/mean absolute difference `0.8699`/`0.00548`, alpha max/mean `0.8810`/`0.00255`, minimum/mean PSNR `29.72`/`39.74` dB, finite output throughout.

Observed formal CityGaussian V1 benchmark campaign on 2026-07-30:

- output package: `/tmp/citygs_formal_bench_20260730_121148/`;
- persistent package: `/home/luis/tese/benchmark_artifacts/citygs_formal_bench_20260730_121148/`;
- thesis memo: `docs/results/citygaussian_v1_released_benchmark_memo.md`;
- full formal 1,440-frame runs with `record_kind="formal_benchmark"`: `CG-FINEST-FLAT`, `CG-FIXED-FINEST-ALL`, `CG-FIXED-FINEST-VISIBILITY`, `CG-LOD-RELEASE`, `FLAT-CITYGS-FULL`, `CG-FUSED-CELLS-ALL`, and `CG-FUSED-VISIBILITY`;
- formal samples: none; the 23M fused branch completed the full trajectory;
- required checks passed: full `unit_tests` discovery (`118` tests, `1` skipped), five-Gaussian CUDA smoke on RTX 5080, visibility selected-set oracles for finest and fused assets, released-LoD selected-set oracle, package validation, and `git diff --check`;
- visibility oracle selected sets matched exactly; projected-box validation used documented float tolerances (`3e-5` finest, `1.2e-4` fused);
- released-LoD branch headline: `CG-FIXED-FINEST-VISIBILITY -> CG-LOD-RELEASE` changed mean frame time `11.06 -> 7.84 ms`, FPS `90.40 -> 127.48`, and mean selected Gaussians `5.94M -> 4.21M`; render peak allocated VRAM increased `5.24 -> 6.19 GB` because all LoD levels stay resident;
- fused branch headline: `CG-FUSED-CELLS-ALL -> CG-FUSED-VISIBILITY` changed mean selected Gaussians `23.39M -> 11.83M` but mean frame time `18.87 -> 43.52 ms`, so this fused SH3 visibility path is not a speed win under the measured backend;
- summary tables live in `/home/luis/tese/benchmark_artifacts/citygs_formal_bench_20260730_121148/tables/`; figures live in `/home/luis/tese/benchmark_artifacts/citygs_formal_bench_20260730_121148/figures/`.

Feasibility timing note: the earlier approximately `1431 ms` one-frame `FLAT-CITYGS-FULL` feasibility timing is non-formal. The old exact JSON was not found during the 2026-07-30 freeze check; the saved formal run shows a first-frame outlier of `681.77 ms` followed by frame 1 at `16.18 ms`, so one-frame no-warmup timings are vulnerable to first-frame/lazy setup effects. Use the formal package as the thesis timing source.

## Benchmark Kinds

`record_kind: "benchmark_validation"` means a short operational run. Its timing is validation metadata, not a formal thesis benchmark.

`record_kind: "formal_benchmark"` means a full protocol run with the preset warmup, measured-frame policy, camera path, and output schema. Do not claim a formal result unless the full protocol was actually run.

`record_kind: "feasibility_result"` means a large-model feasibility attempt. Failed frames are not recorded as timing samples.

## Timing Policy

Warmup frames:

- render the first `warmup_frames` cameras;
- do not record warmup frames;
- exclude first-frame initialization effects where the preset says so.

Measured frames:

- replay the frozen camera path in order;
- `--max-frames` may cap measured frames for short validation;
- `num_repeat` repeats the camera list for formal runs.

Synchronization:

- CUDA is synchronized before active-set preparation timing;
- CUDA is synchronized before and after `GsplatRenderer.render(...)` because the renderer currently synchronizes when measuring CUDA renders;
- optional CUDA event timing can be requested with `--log-gpu-events`, but it is not required for baseline validation.

FPS:

- per-frame FPS is `1000 / frame_time_ms`;
- summary FPS is derived consistently as `1000 / mean_frame_time_ms`.

Percentiles:

- p95 and p99 frame time use NumPy percentile over measured frame times only.

## Memory Policy

Primary thesis VRAM metric:

- `torch_peak_allocated_bytes` for the relevant measured phase. This is the PyTorch CUDA allocator peak for tensors actually allocated by the benchmark process. For per-frame rendering it is reset immediately before rasterization; for scene construction it is reset before asset loading/GPU activation. `whole_run_peak` is reported as the maximum of the recorded scene-construction and render phase peaks because PyTorch exposes one resettable peak counter per device.

Supporting PyTorch allocator metrics:

- `torch_current_allocated_bytes`;
- `torch_current_reserved_bytes`;
- `torch_peak_allocated_bytes`;
- `torch_peak_reserved_bytes`.

Legacy aliases `torch_allocated_bytes` and `torch_reserved_bytes` remain in frame rows for compatibility and mirror `torch_current_allocated_bytes` and `torch_current_reserved_bytes`.

`allocated` means live tensor memory tracked by the PyTorch allocator. `reserved` means memory held by the PyTorch caching allocator, including free cached blocks that can be reused by PyTorch but are still reserved from the CUDA driver. These do not include all driver, context, compiler, NVML, display, or non-PyTorch allocations.

Allocator metadata:

- `allocator_backend`;
- `active_device`;
- `active_device_index`.

Whole-device metrics use NVML through the already available `pynvml` package where supported:

- `nvml_device_total_bytes`;
- `nvml_device_used_bytes`;
- `nvml_device_free_bytes`;
- `nvml_process_used_bytes`;
- `nvml_other_processes_used_bytes`;
- `nvml_process_accounting_supported`;
- `nvml_process_accounting_reliable`;
- `nvml_process_accounting_note`.

In the current WSL environment, NVML whole-device memory is available, but process-level accounting can report `0` bytes for the benchmark process while PyTorch has active allocations. When this happens, `nvml_process_accounting_reliable=false` and the result must not use `nvml_process_used_bytes` as the process VRAM metric. Whole-device before/after deltas are only supporting evidence when the GPU is otherwise isolated.

Memory stages:

- `process_start`;
- `after_cuda_initialization`;
- `before_asset_load`;
- `after_cpu_asset_load`;
- `before_gpu_transfer`;
- `after_gpu_transfer`;
- `before_rasterization`;
- `after_rasterization`;
- `result_collection`.

Reset boundaries:

- scene-construction peak: reset before asset load/GPU activation and record after GPU transfer;
- render peak: reset immediately before each rasterization call;
- whole-run peak: maximum of recorded phase peaks, not an uninterrupted raw PyTorch counter.

CPU RAM metrics:

- process RSS where `psutil` is available;
- system used RAM where `psutil` is available.

## Result Files

Successful `FLAT-REAL-FIT` validation or formal runs write:

```text
frames.jsonl
summary.json
resolved_config.json
```

`FLAT-CITYGS-FULL` feasibility attempts write:

```text
feasibility_result.json
resolved_config.json
```

`benchmarks/compare_citygaussian_same_asset.py` writes:

```text
frames.jsonl
image_equivalence_summary.json
resolved_config.json
```

Outputs must go to an explicit output directory outside tracked source.

## Output Schema

Frame samples include:

- `record_kind: "frame_sample"`;
- configuration, frame, repeat, and camera identifiers;
- width and height;
- frame time, FPS, preparation time, and optional rasterization time;
- total, active, visible, and rendered Gaussian counts;
- Torch allocator metrics;
- NVML whole-device and process-accounting metrics where available;
- CPU RAM metrics;
- flat-path integrity flags.
- CityGaussian fixed-all rows additionally include total/active cell count, selected Gaussian count, cell-construction time, frame preparation/concatenation time, and flags confirming no visibility, adaptive LoD, or streaming.

Successful summaries include:

- `record_kind`;
- frame count;
- mean, median, p95, p99, minimum, and maximum frame time;
- FPS derived from mean frame time;
- rasterization time where valid;
- preparation time where valid;
- peak allocated and reserved VRAM;
- CPU RAM where available;
- total and active Gaussian counts;
- radii-derived visible Gaussian count;
- resolution;
- gsplat, PyTorch, CUDA, and GPU metadata;
- Git commit and dirty-state indicator;
- configuration ID;
- checkpoint identifier;
- trajectory identifier.
- CityGaussian fixed-all summaries additionally include block dimensions, AABB, cell IDs, per-cell Gaussian counts, exact-once assignment diagnostics, MAD bounds, average scales, and cell-construction time.

Feasibility results include:

- `record_kind: "feasibility_result"`;
- `configuration_id: "FLAT-CITYGS-FULL"`;
- `status: "success" | "oom" | "failed"`;
- `failure_stage`;
- checkpoint file size;
- PLY vertex count where available;
- CPU RAM before and after;
- CPU RAM after PLY parsing and after CPU tensor construction where reached;
- GPU allocated/reserved memory before transfer, after transfer, and before rasterization where reached;
- GPU allocated/reserved/peak memory before failure reporting;
- PyTorch allocator backend and active device;
- NVML whole-device memory where available;
- NVML process-accounting support and reliability flags;
- `scene_construction_peak`, `render_peak`, and `whole_run_peak`;
- final successful stage;
- exception type and message;
- environment metadata.

The current `gsplat` visible count is radii-derived from rasterizer output. It is not automatically equivalent to a paper's definition of visibility, selected Gaussians, rendered Gaussians, or intersected Gaussians.

## OOM And Failure Handling

The feasibility path records the stage active when an exception occurs:

- `asset_validation`: path and lightweight asset metadata checks;
- `ply_parsing`: PLY header validation and memory-map setup;
- `cpu_tensor_construction`: CPU splat tensors are constructed from checkpoint or PLY fields;
- `bounds_or_metadata_computation`: metadata-only scene bounds are computed from CPU positions;
- `gpu_transfer`: raw flat splat tensors are activated onto the target GPU;
- `gpu_scene_construction`: GPU-resident active splat object and flat scene stats are finalized;
- `rasterizer_initialization`: `GsplatRenderer` is constructed;
- `active_set_preparation`: `scene.prepare_frame(...)`;
- `rasterization`: `GsplatRenderer.render(...)` and transient rasterizer allocation.
- `result_collection`: render outputs and memory metrics are collected.

An OOM is an acceptable `FLAT-CITYGS-FULL` feasibility result. Do not deliberately repeat OOM attempts many times.

## Benchmark Integrity

For `FLAT-REAL-FIT` and `FLAT-CITYGS-FULL`, the benchmark asserts or records:

- scene type is flat;
- all loaded Gaussians are active whenever a frame successfully reaches rendering;
- `BlockManager` is not used;
- host-residency logic is not used;
- no LoD selection is used;
- no hierarchy traversal is used;
- no CityGaussian runtime selection is used;
- the existing `GsplatRenderer` wrapper is used.

## Cross-Asset Limitations

`FLAT-REAL-FIT` and `FLAT-CITYGS-FULL` use different assets and different manually authored trajectories. Comparisons across them are cross-asset or partially controlled. Do not attribute differences between them solely to viewer optimization or rendering-system methodology.

Duplicated NxM grids of the 1.2M checkpoint are not primary real-scene baselines. They remain candidates for a later synthetic scalability branch such as `SYNTHETIC-SCALE-1X1`, `SYNTHETIC-SCALE-2X2`, and `SYNTHETIC-SCALE-3X3`.

## ALoD Resident Correctness Characterization

The tracked `ALOD-HIER-FINEST-RESIDENT`, `ALOD-HIER-LOD-RESIDENT`, and
`ALOD-HIER-LOD-FRUSTUM` configurations use the same `.dhier` asset role,
resident property representation, and shared gsplat renderer. Their intended
selection differences are valid finest cut, hierarchical LoD without frustum
rejection, and hierarchical LoD with the pinned four-plane frustum.

These presets currently use `record_kind="correctness_characterization"`.
Their ten-camera timings are not formal benchmark results: the frozen oracle is
classified `WEAK-LOD-ORACLE`, and cold initialization is intentionally visible.

ALoD records separate HSPT build, camera preparation, upper traversal, SPT cut,
selected-index, gather, assembly, and raster timings. Counts and bytes separate
asset, skybox, hierarchy, resident, upper, candidates, selected hierarchy,
raster rows, CPU/GPU hierarchy metadata, and CPU/GPU Gaussian properties.
Streaming, caching, prefetching, async transfer, and working-set reuse remain
disabled for the resident baseline.

The formal resident-only campaign completed on 2026-08-08 as
`ALOD-RESIDENT-BENCHMARK-001`. It used three independent processes per
configuration, one complete ten-camera warmup per process, and ten complete
measured trajectory repetitions per process (300 measured frames per
configuration). The persistent package is
`/home/luis/tese/benchmark_artifacts/alod_resident_benchmark_001/`; the
thesis-facing memo is `docs/results/alod_resident_ablation_benchmark.md`.

Headline result: mean end-to-end frame time was 6.94 ms for finest resident,
9.33 ms for LoD resident, and 9.71 ms for LoD plus frustum. Resident property
memory was identical. On this weak-LoD trajectory, selected-row reductions did
not reduce backend-visible work enough to offset CPU selection.
