# CityGaussian V1 Full-Detail Reference

## Objective

Define bounded no-adaptive-LoD references for CityGaussian V1 evaluation. The fused no-LoD PLY remains the flat `CityGS(noLoD)` reference. The finest released compressed LoD, `mc_aerial_c36_light_50_vq`, is now also represented by two same-asset references:

- `CG-FINEST-FLAT`: the finest released LoD PLY loaded as one flat Gaussian set.
- `CG-FIXED-FINEST-ALL`: the same PLY organized into official CityGaussian V1 runtime cells with all 36 cells active.

## Thesis Relevance

This gives the CityGaussian branch a controlled parent for later runtime selection work and preserves the distinction between `CityGS(noLoD)` and compressed LoD levels.

## Current State

`FLAT-CITYGS-FULL` already validates that `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply` can render finite output as one flat scene. Asset validation verified that `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq/point_cloud.ply` is the finest official compressed LoD, with 11,694,810 SH-degree-2 Gaussians, matching VQ metadata, and all 36 official runtime cells populated.

`CG-FINEST-FLAT` and `CG-FIXED-FINEST-ALL` are implemented as benchmark configurations. `CG-FIXED-FINEST-ALL` uses a CityGaussian-specific runtime boundary in `scene_runtime/citygaussian.py`; it does not use `BlockManager`, host residency, visibility selection, adaptive LoD, streaming, VQ reconstruction, or FlashGS.

CUDA validation completed on 2026-07-06 on the host RTX 5080 environment with PyTorch `2.11.0.dev20260128+cu129`, CUDA `12.9`, and `gsplat 1.5.3`. The validation used the existing conda CUDA 12.9 compiler and headers because system `/usr/bin/nvcc` is CUDA 12.0 and cannot compile for `compute_120`.

## Paper And Reference Repository

- Paper title: `CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians`
- Authors/year: Yang Liu et al.; ECCV 2024
- Reference code URL and commit/tag: `https://github.com/DekuLiuTesla/CityGaussian.git`, `V1-original`, `e16c526f78ab01d377b13b3e6862dcc93ddbe953`
- License notes: CC BY-NC-SA 4.0.

## Reproduction Scope

Use the author-provided fused PLY as the no-LoD full-detail reference. Treat the finest compressed LoD as a separate fixed-detail parent for visibility experiments. Do not claim either reproduces official adaptive LoD.

## Training Scope

No training. Training and block fusion are author-provided/out of scope.

## Offline Construction Scope

No construction. Validation may confirm the fused PLY header, finest LoD PLY/VQ metadata, official YAML order, camera JSON, and 36-cell population.

## Runtime Scope

Use existing flat scene loading and `GsplatRenderer` for `CG-FINEST-FLAT`. Use `CityGaussianFixedAllScene` and `GsplatRenderer` for `CG-FIXED-FINEST-ALL`. This is a project backend reference, not the official GraphDECO rasterizer.

## Explicitly Excluded Work

No adaptive LoD selection, no visibility culling, no host residency, no LightGaussian processing, and no generation of LoD assets.

## Methodology Steps

1. Keep `FLAT-CITYGS-FULL` as the fused no-LoD PLY reference.
2. Use `CG-FINEST-FLAT` for the finest released LoD PLY as a flat same-asset parent.
3. Use `CG-FIXED-FINEST-ALL` for the same PLY divided into official 36 runtime cells, with all cells active and visibility/adaptive LoD disabled.
4. Compare the two same-asset references by Gaussian assignment diagnostics and image difference metrics.

## Mapping To Reference Code

Reference `merge.py::blockMerge` for direct fusion semantics and paper Sec. 4.2/Supplement Table S1 for `CityGS(noLoD)`.

## Permitted Deviations

Rendering through `gsplat` is permitted only when labeled as a backend deviation from official V1.

## Required Inputs

- Fused PLY: `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply`
- Finest compressed LoD: `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq`
- Camera path: existing `benchmarks/camera_paths/citygaussian_23m_navigation_v1.json`

## Architecture Touchpoints

`SceneLoader.load(..., scene_type="flat")`, `FlatGaussianScene`, `SceneLoader.load(..., scene_type="citygaussian_fixed_all")`, `CityGaussianFixedAllScene`, `GsplatRenderer.render`, `benchmarks/benchmark_real_scene.py`, and `benchmarks/compare_citygaussian_same_asset.py`.

## Invariants

For `FLAT-CITYGS-FULL`, all 23,389,621 fused Gaussians are active. For `CG-FINEST-FLAT` and `CG-FIXED-FINEST-ALL`, all 11,694,810 finest compressed LoD Gaussians are active. In `CG-FIXED-FINEST-ALL`, every source Gaussian must belong to exactly one official cell, all 36 cells must be active every frame, and selected Gaussian count must equal total Gaussian count. No visibility, adaptive LoD, hierarchy, cache, or streaming code path is enabled.

## Failure Cases

If the fused PLY or finest LoD asset is absent, report `BLOCKED - missing asset`. If CUDA cannot render it, report feasibility failure without changing methodology.

## Performance Constraints

Use existing benchmark protocol and 1920x1080 camera path. Do not add new formal timing claims without a full protocol run.

## Acceptance Criteria

- Configuration labels distinguish no-LoD fused PLY from finest compressed LoD.
- `CG-FINEST-FLAT` and `CG-FIXED-FINEST-ALL` use the same finest-LoD PLY, trajectory, render settings, `gsplat` backend, and SH degree.
- `CG-FIXED-FINEST-ALL` reports cell IDs, per-cell counts, exact-once assignment diagnostics, MAD bounds, average scales, cell-construction time, all-active selected count, and concatenation/preparation time.
- Output metrics include total/active/selected Gaussians, frame time, VRAM, and visible/projected count proxy.
- Completion report states this is not official visibility or adaptive LoD.

## Required Tests

Focused tests cover official contraction behavior, boundary points, cell assignment, exact-once validation, empty cells, MAD bounds, average scale, all-cells-active preparation, source-index equivalence, flat-vs-cell attribute equivalence, disabled visibility/adaptive-LoD/streaming flags, no `BlockManager` dependency, and result serialization.

## Required Benchmark Evidence

Short validation should run both `CG-FINEST-FLAT` and `CG-FIXED-FINEST-ALL` with the same finest-LoD PLY and frame indices, plus `benchmarks/compare_citygaussian_same_asset.py` for numerical image equivalence. Formal benchmark evidence follows `docs/BENCHMARK_PROTOCOL.md`.

Observed short validation, not a formal benchmark:

- Five-Gaussian CUDA smoke passed with finite image/alpha and deterministic image/alpha max absolute difference `0.0`; log: `/tmp/cg_smoke_cuda129_cpath.log`.
- `CG-FINEST-FLAT`, `/tmp/cg_finest_flat_validation/summary.json`: 1 warmup frame, 3 measured frames, 11,694,810 active Gaussians, finite image/alpha for every measured frame, mean frame time `10.463 ms`, mean preparation time `0.0146 ms`, mean visible/rendered Gaussians `2,212,332.33`, PyTorch peak allocated VRAM `2,780,806,144` bytes, peak reserved VRAM `3,145,728,000` bytes.
- `CG-FIXED-FINEST-ALL`, `/tmp/cg_fixed_finest_all_validation/summary.json`: 1 warmup frame, 3 measured frames, 36 total/active cells, 11,694,810 selected Gaussians every frame, finite image/alpha for every measured frame, cell construction `870.297 ms`, mean frame time `10.541 ms`, mean preparation time `11.273 ms`, mean visible/rendered Gaussians `2,212,332.33`, PyTorch peak allocated VRAM `4,652,319,232` bytes, peak reserved VRAM `4,693,426,176` bytes.
- Same-frame comparison, `/tmp/cg_fixed_finest_image_compare/image_equivalence_summary.json`: frames `[0, 1, 2]`, all finite, max RGB absolute difference `0.0`, mean RGB absolute difference `0.0`, max alpha absolute difference `0.0`, mean alpha absolute difference `0.0`; visible counts matched exactly per frame (`2,187,326`, `2,212,751`, `2,236,920`).
- Integrity flags confirm no `BlockManager`, host residency, hierarchy traversal, visibility selection, adaptive LoD, streaming, or VQ reconstruction path was used.

## Required Completion Report

Report whether the task used the existing config or added a new alias, all deviations from official V1, and confirmation no LoD/chunk claims were made.

## Fable Readiness

COMPLETE for the same-asset fixed-finest parent validation. Visibility remains Spec 003.
