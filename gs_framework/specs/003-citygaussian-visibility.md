# CityGaussian V1 Visibility

## Objective

Reproduce the official CityGaussian V1 viewer-side block visibility rule without adaptive LoD. This specification now has independent selected-set oracle fixtures for both the released finest fixed-detail branch and the fused no-LoD branch.

## Thesis Relevance

This isolates the effect of official block visibility on selected blocks, active Gaussian count, VRAM, and frame time before adding adaptive LoD.

## Current State

The official rule is documented in `docs/papers/citygaussian-v1-method-card.md`: `BlockedGaussian.cell_divider` creates MAD bounds, `utils.large_utils.in_frustum` selects visible blocks, and `render_lod` later activates whole selected blocks. Asset validation verified that `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq` covers all 36 official MatrixCity cells. `CG-FINEST-FLAT` and `CG-FIXED-FINEST-ALL` provide the same-asset fixed-finest parent representations; the current project `BlockManager` is still not official CityGaussian V1 behavior.

Spec 002 CUDA validation completed on 2026-07-06: `CG-FIXED-FINEST-ALL` activated all 36 official cells and all 11,694,810 finest-LoD Gaussians, rendered finite RGB/alpha over the short validation path, and matched `CG-FINEST-FLAT` exactly for RGB/alpha on frames `[0, 1, 2]`.

Selected-set oracle fixtures were added on 2026-07-06:

- `unit_tests/fixtures/citygaussian_visibility/finest_visibility_oracle_v1.json`
- `unit_tests/fixtures/citygaussian_visibility/fused_visibility_oracle_v1.json`

Both fixtures execute the official `utils/large_utils.py::in_frustum` behavior through a minimal source slice because direct import of `utils.large_utils` is blocked by the uninitialized/missing `simple_knn` dependency chain. The source slice is read from `/home/luis/tese/CityGaussian` at branch `V1-original`, commit `e16c526f78ab01d377b13b3e6862dcc93ddbe953`.

Visibility is implemented in `scene_runtime` as `CityGaussianVisibilityScene` and validated against the independent finest and fused selected-set oracle fixtures.

## Paper And Reference Repository

- Paper title: `CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians`
- Authors/year: Yang Liu et al.; ECCV 2024
- Reference code URL and commit/tag: `https://github.com/DekuLiuTesla/CityGaussian.git`, `V1-original`, `e16c526f78ab01d377b13b3e6862dcc93ddbe953`
- License notes: CC BY-NC-SA 4.0; avoid copying substantial source without license review.

## Controlled Branches

### Released Finest Branch

```text
CG-FIXED-FINEST-ALL
-> CG-FIXED-FINEST-VISIBILITY
```

Both configurations use the same `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq/point_cloud.ply` asset with 11,694,810 Gaussians. `CG-FIXED-FINEST-ALL` activates all 36 official cells. `CG-FIXED-FINEST-VISIBILITY` changes only the active cell set according to the official visibility oracle. This branch is the parent for later adaptive LoD.

### Fused Full-Model Branch

```text
FLAT-CITYGS-FULL
-> CG-FUSED-CELLS-ALL
-> CG-FUSED-VISIBILITY
```

All three configurations use the same `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply` fused PLY with 23,389,621 Gaussians. `CG-FUSED-CELLS-ALL` constructs official cells for the fused SH-degree-3 model and activates all 36 cells. `CG-FUSED-VISIBILITY` changes only the visibility-selected active cells. This branch isolates visibility relative to the original 23M flat baseline.

`CG-FUSED-CELLS-ALL` and `CG-FUSED-VISIBILITY` are implemented tracked configurations. The all-cells parent is image-equivalent to `FLAT-CITYGS-FULL` over the 12 oracle cameras, and the visibility child matches the fused oracle selected sets.

## Reproduction Scope

Block/cell-level visibility only: MAD bounds, projection to screen, `z > 0.2` front-corner filtering, positive-area screen intersection with `[-1,1] x [-1,1]`, and nearest-block fallback. Adaptive LoD must be disabled.

## Training Scope

No training. Use the official finest compressed LoD asset validated by Spec 001.

## Offline Construction Scope

No offline construction is needed for official cells: the V1 runtime constructs them in memory from the loaded LoD model. Project-generated XY partitions cannot be used to claim official visibility.

## Runtime Scope

Implement a new CityGaussian-specific scene path or selector at `GaussianScene.prepare_frame`, returning `ActiveSplats` and metrics. Do not route through `BlockManager` unless an adapter proves equivalent behavior. Do not implement host residency, streaming, or adaptive LoD in Spec 003.

## Explicitly Excluded Work

No LoD, no LightGaussian, no host-residency/cache behavior, no paper-interval LoD, no CUDA rasterizer changes.

## Oracle Fixtures

Selected trajectory indices are deterministic and shared by both oracle fixtures:

```text
[0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439]
```

Selection policy: `np.linspace` over the 1,440-frame frozen trajectory, rounded to nearest integer, deduplicated, sorted, and including first and final frames.

Camera conversion:

- source JSON stores row-major Viser `camera_to_world` matrices;
- oracle conversion flips camera Y/Z axes and multiplies translation by `scale_ratio`, matching the project benchmark trajectory conversion;
- intrinsics use the project benchmark convention that JSON `fov` is vertical FOV;
- `FoVx` and `FoVy` are derived from image width, height, and focal length;
- official `world_view_transform = inverse(converted camera_to_world).T`;
- official projection matrix matches `utils.graphics_utils.getProjectionMatrix(...).T`;
- official `full_proj_transform = world_view_transform @ projection_matrix`;
- camera center is `inverse(world_view_transform)[3, :3]`;
- front-facing corners use official camera-space `z > 0.2`.

Fixture contents include visible cell IDs, 36-entry visibility mask, visible-cell count, selected Gaussian count, nearest-cell ID, fallback flag, official per-visible-cell distances, projected boxes, front-corner counts, cell-bound checksum, projection checksum, official commit, asset identifier, and Gaussian count.

## Methodology Steps

1. Use `CG-FIXED-FINEST-ALL` as the fixed-detail parent: load only `mc_aerial_c36_light_50_vq`, construct official 36 runtime cells, and activate all cells.
2. Validate `CG-FIXED-FINEST-VISIBILITY` selected block IDs, nearest fallback behavior, distances, and selected Gaussian counts against `finest_visibility_oracle_v1.json`.
3. Optionally add `CG-FUSED-CELLS-ALL` for the fused PLY with all 36 cells active.
4. Validate `CG-FUSED-VISIBILITY` selected block IDs, nearest fallback behavior, distances, and selected Gaussian counts against `fused_visibility_oracle_v1.json`.
5. Log selected blocks and selected Gaussian count separately from rendered/projected count.

## Mapping To Reference Code

- `scene/gaussian_model.py::BlockedGaussian.cell_divider`
- `utils/large_utils.py::block_filtering`
- `utils/large_utils.py::in_frustum`
- `gaussian_renderer/__init__.py::render_lod` only for assembly semantics, not LoD selection

## Permitted Deviations

Rendering through `gsplat` is allowed only after selected-block equivalence is established and labeled as a backend deviation.

## Required Inputs

- `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq`
- `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply` for the proposed fused branch
- `/home/luis/tese/CityGaussian/config/mc_aerial_c36_lod.yaml`
- `benchmarks/camera_paths/citygaussian_23m_navigation_v1.json`
- `unit_tests/fixtures/citygaussian_visibility/finest_visibility_oracle_v1.json`
- `unit_tests/fixtures/citygaussian_visibility/fused_visibility_oracle_v1.json`

## Architecture Touchpoints

`scene_runtime/citygaussian.py`, `SceneLoader`, `ActiveSplats`, `SceneFrameMetrics`, `GsplatRenderer.render`, viewer metrics CSV, and benchmark summary schema.

## Invariants

Adaptive LoD disabled. Selected detail source fixed. Visibility selection binary per block. All Gaussians from selected visible blocks are active.

## Failure Cases

If an oracle fixture is absent, report `BLOCKED - missing visibility validation oracle`. If the fixed-finest parent regresses, return to Spec 002 before claiming visibility results. If the official finest LoD or fused PLY asset is absent in a future environment, report `BLOCKED - missing asset`. If selected-set IDs differ from the oracle, do not render or benchmark the configuration until the camera conversion, cell bounds, and visibility implementation are reconciled.

## Performance Constraints

Selection time must be measured separately from selected-cell assembly/concatenation and rasterization.

## Acceptance Criteria

- Exact selected cell IDs match the relevant oracle fixture for every fixture camera.
- Exact nearest-cell fallback behavior matches the relevant oracle fixture.
- Selected Gaussian counts equal the oracle selected Gaussian counts.
- Per-visible-cell camera distances match the oracle within `1e-5` absolute tolerance after the documented float32 conversion.
- Active Gaussian counts equal the sum of official selected cell counts.
- Output image and alpha are finite.
- Visibility-only output is compared against its all-cells parent where semantically valid.
- Visibility-selection timing is recorded separately from selected-cell assembly and rasterization.
- No adaptive LoD, host residency, streaming, or runtime cache behavior is enabled.

## Required Tests

Unit tests for corner projection, screen intersection, nearest-block fallback, selected-set fixture comparison, deterministic distance output, malformed camera matrix rejection, and mismatched cell-count rejection.

Focused oracle tests currently live in `unit_tests/test_citygaussian_visibility_oracle.py`.

## Required Benchmark Evidence

Short validation over the CityGaussian camera path with selected block counts, selected Gaussian counts, active Gaussian counts, visibility-selection time, selected-cell assembly time, rasterization time, frame time, VRAM, and image sanity. Formal benchmark later follows the protocol.

Future visibility configurations must record:

- total cell count;
- visible/selected cell count;
- selected cell IDs;
- total Gaussian count;
- selected Gaussian count;
- active Gaussian count;
- radii-derived visible Gaussian count;
- visibility-selection time;
- selected-cell assembly time;
- rasterization time;
- end-to-end frame time and FPS;
- post-transfer and render-peak VRAM;
- finite output;
- image difference against the all-cells parent.

## Completion Evidence

Spec 003 completed on 2026-07-08 with non-formal CUDA validation under `/tmp/citygs003_*`.

- Runtime selected-set oracle validation passed for all 12 frozen cameras on `mc_aerial_c36_light_50_vq/point_cloud.ply`: visible cell IDs, visible-cell counts, nearest-cell IDs, nearest fallback flags, and selected Gaussian counts matched exactly. Selected cells ranged `9..35`; selected Gaussians ranged `3,248,918..11,694,702`; the fallback-changing case at trajectory index `1177` passed.
- Runtime selected-set oracle validation passed for all 12 frozen cameras on the fused `point_cloud/iteration_30000/point_cloud.ply`: visible cell IDs, visible-cell counts, nearest-cell IDs, nearest fallback flags, and selected Gaussian counts matched exactly. Selected cells ranged `10..35`; selected Gaussians ranged `6,422,577..23,389,336`; fallback-changing cases at trajectory indices `654` and `785` passed.
- Distance outputs matched the fixtures within `2e-6`; projected boxes matched within fixture/float tolerance (`2.9e-5` finest, `1.15e-4` fused).
- `FLAT-CITYGS-FULL` and `CG-FUSED-CELLS-ALL` rendered RGB/alpha-equivalent output over the 12 oracle cameras with max absolute difference `0.0`; fused official-cell construction assigned all `23,389,621` Gaussians exactly once with zero unassigned and zero duplicated assignments, and all 36 cells were nonempty.
- `CG-FIXED-FINEST-VISIBILITY` and `CG-FUSED-VISIBILITY` rendered finite RGB/alpha for all 12 oracle cameras. Visibility outputs intentionally differ from all-cells parents because cells are removed; minimum sampled PSNRs against all-cells parents were `22.29` finest and `22.31` fused.
- Runtime metrics distinguish `resident_gaussian_count = total_gaussian_count` from `selected_gaussian_count` and `active_gaussian_count`; for Spec 003, active equals selected.
- No adaptive LoD, fixed-distance LoD, Nyquist LoD, VQ reconstruction, streaming, host residency, `BlockManager`, FlashGS, dependency installation, staging, committing, pushing, or training was used.

## Required Completion Report

Report asset lineage, selected-set oracle, deviations, metrics exposed, and confirmation that no adaptive LoD was enabled.

## Fable Readiness

COMPLETE. Runtime selected cells exactly match the independent oracle for both finest and fused assets, both branches render finite output, parent-child metrics were collected, and no LoD behavior is present.
