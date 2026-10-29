# CityGaussian V1 LoD

## Objective

Reproduce official CityGaussian V1 adaptive LoD selection on top of validated official visibility behavior. `CG-LOD-RELEASE` is implemented and validated against the authoritative released-code oracle.

## Thesis Relevance

This isolates the additional impact of blockwise LoD after visibility selection is already working and measured.

## Current State

The method card verifies two LoD descriptions: paper fixed distance intervals and inspected source Nyquist selection. Asset validation verified the three official MatrixCity LoD directories under `/home/luis/tese/mc_aerial_c36/LoD`, in YAML order: `light_50_vq` finest, `light_66_vq` middle, and `light_75_vq` coarsest. Spec 003 provides the validated visibility prerequisite: official runtime cells, shared camera conversion, selected-cell oracle agreement, and finite render samples without LoD. Spec 004 now adds independent selected-LoD oracle fixtures under `unit_tests/fixtures/citygaussian_lod/`.

## Paper And Reference Repository

- Paper title: `CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians`
- Authors/year: Yang Liu et al.; ECCV 2024
- Reference code URL and commit/tag: `https://github.com/DekuLiuTesla/CityGaussian.git`, `V1-original`, `e16c526f78ab01d377b13b3e6862dcc93ddbe953`
- License notes: CC BY-NC-SA 4.0; LightGaussian dependency/license must be checked when submodule is available.

## Reproduction Scope

Adaptive LoD selection per visible block, using the same official visibility path validated by Spec 003. The first implementation should be source-faithful Nyquist LoD against `release_lod_oracle_v1.json`. Paper fixed intervals must remain a separate provisional policy until scale semantics are resolved.

## Training Scope

No training. Use the validated author-provided LoD assets. Generating LoD levels remains a separate documented offline task.

## Offline Construction Scope

Do not implement LightGaussian compression in this task. The released LoD outputs are present and should be treated as author-provided complete-scene alternatives, not as locally generated assets.

## Runtime Scope

Load all official LoD levels, compute per-block average scale and MAD bounds, select one LoD per visible block, concatenate selected block tensors, render, and expose LoD metrics.

## Explicitly Excluded Work

No visibility rewrite beyond Spec 003, no blending/hysteresis, no streaming/cache additions, no rasterizer backend comparison. Paper-interval mode is only allowed as a separately labeled provisional ablation.

## Methodology Steps

1. Use source-faithful Nyquist as the primary release policy; keep paper fixed distance intervals separate and provisional.
2. Validate LoD ordering and SH degree from official config and asset manifest.
3. Compare selected LoD indices per block against official source on fixture cameras.
4. Add metrics for selected LoD distribution, selected Gaussians per level, selection time, concatenation time, and rasterization time.

## Mapping To Reference Code

- `config/mc_aerial_c36_lod.yaml`
- `render_large_lod.py::render_sets`
- `scene/gaussian_model.py::BlockedGaussian`
- `utils/large_utils.py::in_frustum`
- `gaussian_renderer/__init__.py::render_lod`

## Permitted Deviations

`gsplat` rendering is permitted only after selected-set equivalence. Any fixed-interval paper mode must be a separate labeled variant from source-faithful Nyquist mode.

## Required Inputs

- `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq`
- `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_66_vq`
- `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_75_vq`
- `/home/luis/tese/CityGaussian/config/mc_aerial_c36_lod.yaml`
- Selected-LoD oracle fixtures:
  - `unit_tests/fixtures/citygaussian_lod/release_lod_oracle_v1.json`
  - `unit_tests/fixtures/citygaussian_lod/paper_lod_oracle_v1.json`

## Architecture Touchpoints

New CityGaussian scene/selector, `SceneLoader`, `GaussianScene.prepare_frame`, `ActiveSplats`, `SceneFrameMetrics`, metrics CSV/schema, `GsplatRenderer`.

## Invariants

Visibility path is identical to Spec 003. One LoD per visible block. No blending or hysteresis. Selected LoD ordering matches official config.

## Failure Cases

Missing LoD directory, selected-set mismatch against the release oracle, or accidental mixing of paper and release policies must block implementation. Do not silently fall back to fused PLY or project partitions.

## Performance Constraints

Report LoD selection and concatenation separately from rasterization. Peak VRAM must distinguish fully resident loaded LoD tensors from active selected tensors where measurable.

## Acceptance Criteria

- LoD assets validated and counted.
- Selected LoD per visible block matches `release_lod_oracle_v1.json` on fixture cameras.
- `CG-FIXED-FINEST-VISIBILITY -> CG-LOD-RELEASE` uses the same finest-level asset, cameras, visibility path, and renderer settings; the child additionally loads the middle and coarsest released levels required by adaptive selection.
- Metrics include per-level block and Gaussian counts.
- Image output is compared against official renderer or labeled as `gsplat` backend deviation.

## Completion Evidence

**COMPLETE for `CG-LOD-RELEASE` on 2026-07-28.**

- All 12 frozen cameras exactly match `release_lod_oracle_v1.json` for visible cells, LoD indices/labels, per-cell/per-level counts, totals, and fallback state.
- Selected cell-use totals are 94 finest, 14 middle, and 94 coarsest; selected Gaussians range from 1,957,671 to 6,211,124.
- All 25,494,686 Gaussians remain resident; selected and active counts are equal.
- Parent and child render samples are finite. The child sample reduced mean selected count by 29.04% and mean end-to-end frame time by 16.03%, while FPS increased 19.09%.
- Fully resident post-transfer allocated memory increased by 2.32 GB and render peak allocated memory increased by 0.86 GB versus `CG-FIXED-FINEST-VISIBILITY`.
- Parent/child `gsplat` comparison recorded RGB max/mean absolute difference `0.8699`/`0.00548`, alpha max/mean `0.8810`/`0.00255`, minimum PSNR `29.72` dB, and finite output. Visual inspection found no missing regions or invalid output.
- `CG-LOD-PAPER`, VQ reconstruction, streaming, host residency, and `BlockManager` are absent from this configuration.

## Required Tests

Unit tests for Nyquist equation, fallback behavior, LoD ordering, selected-set comparison, and no-blend/no-hysteresis invariants.

## Required Benchmark Evidence

Short validation over fixed camera path with LoD distributions and active counts. Formal quality/performance run only after selected-set validation against `release_lod_oracle_v1.json`.

## Required Completion Report

Report LoD rule chosen, asset lineage, selected-set oracle, deviations from paper/source, metrics, and unresolved artifacts.

## Fable Readiness

COMPLETE FOR SOURCE-FAITHFUL RELEASED-CODE POLICY; PAPER POLICY PROVISIONAL AND UNIMPLEMENTED - fixed distance intervals need scale review before paper-faithful claims.
