# CityGaussian V1 Method Card

Generated: 2026-07-06

Evidence labels:

- **Verified from paper**: checked in `docs/papers/sources/citygaussian-v1-paper.md`.
- **Verified from official source**: checked in `/home/luis/tese/CityGaussian` at `e16c526f78ab01d377b13b3e6862dcc93ddbe953`.
- **Verified from released asset**: checked from local author-provided assets without rendering or full data loading.
- **Verified from current project**: checked in `/home/luis/tese/gs_framework`.
- **Inferred**: reasoned from the verified sources.
- **Unresolved**: not established by the available paper, source, or local assets.

This is a research-audit and reproduction-status document. Specs 001-004 now implement and validate the stated viewer-side released-code components; it does not claim training or full-paper reproduction.

## 1. Exact Reference

| Field | Status |
|---|---|
| Paper title | **Verified from paper/source.** `CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians`. |
| Authors | **Verified from paper.** Yang Liu, He Guan, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, Zhaoxiang Zhang. |
| Venue/year | **Verified from official source.** README labels the paper ECCV 2024; repository BibTeX also contains a Springer ECCV proceedings entry published in 2025. |
| Paper version | **Verified from current project/source.** Local PDF is `2404.01133v3.pdf`; official README says the camera-ready version was available on 2024-07-18. |
| Official repository | **Verified from official source.** `https://github.com/DekuLiuTesla/CityGaussian.git`. |
| Local repository path | **Verified from official source.** `/home/luis/tese/CityGaussian`. |
| Inspected branch/commit | **Verified from official source.** Branch `V1-original`, commit `e16c526f78ab01d377b13b3e6862dcc93ddbe953`, commit subject `Update citygaussianv1_readme.md`, clean checkout. |
| Relevant local refs | **Verified from official source.** Branches `V1-original`, `main`; tags `CityGaussian_V1.0`, `CityGaussian_V1.1`, `CityGaussian_V1.2`, `CityGaussian_V1.5`, `CityGaussian_V2.0`. |
| License | **Verified from official source.** `LICENSE.md` and README state Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. This permits noncommercial sharing/adaptation under attribution and share-alike terms; integration into this thesis codebase needs license-aware handling. |
| Dependencies | **Verified from official source.** README: conda Python 3.9, PyTorch 2.0.1, CUDA 11.8 example, `pip install -r requirements.txt`, tailored LightGaussian for LoD. `requirements.txt`: `lightning==2.0.9.post0`, `plyfile==0.8.1`, `numpy==1.26.2`, `viser==0.1.29`, `opencv-python==4.8.*`, local 3DGS rasterizer and simple-knn submodules. |
| Submodules | **Verified from official source.** Local submodules are not initialized: `LargeLightGaussian`, `SIBR_viewers`, `submodules/diff-gaussian-rasterization`, `submodules/simple-knn` show missing commits in `git submodule status`. |
| Known local author assets | **Verified from released asset.** `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply` is a binary little-endian Gaussian PLY with 23,389,621 vertices and SH-degree-3-style fields; `/home/luis/tese/mc_aerial_c36/cameras.json` exists with 5,621 camera entries; `/home/luis/tese/mc_aerial_c36/LoD/` contains `mc_aerial_c36_light_50_vq`, `mc_aerial_c36_light_66_vq`, and `mc_aerial_c36_light_75_vq`. |
| Known missing local author assets | **Verified from released asset/source.** No local `lod-meta.json` or pre-generated official chunk directory is required or present for the inspected V1 Python runtime path; runtime cells are constructed in memory by `BlockedGaussian.cell_divider`. |

## 2. Claimed Contributions

- **Training scalability. Verified from paper.** CityGS trains a coarse global Gaussian prior, then partitions the scene and trains blocks independently to reduce single-GPU memory pressure.
- **Scene partitioning and data assignment. Verified from paper/source.** The coarse prior is contracted into a bounded cubic space and uniformly divided into blocks. A camera is assigned to a block if the block contributes enough visible content by an SSIM-loss criterion or if the camera center lies inside the block.
- **Model fusion. Verified from paper/source.** Fine-tuned block Gaussians are filtered to their spatial bounds and fused by direct concatenation.
- **LoD asset generation. Verified from paper/source.** The paper states that detail levels are generated from fused Gaussians using LightGaussian compression. Official README commands use `LargeLightGaussian` pruning, distillation, and vector-quantization scripts, but that submodule is absent locally.
- **Runtime visibility selection. Verified from paper/source.** Runtime operates at block/cell granularity. MAD-derived block bounds are projected, intersected with the screen rectangle, and visible blocks are selected.
- **Runtime LoD selection. Verified from paper/source with conflict.** The paper describes fixed distance intervals; the current V1 README and code use a Nyquist-sampling rule comparing per-block average scale to `2 * distance / focal_length`.
- **Rendering/viewer behavior. Verified from official source.** Selected block tensors from selected LoD levels are concatenated every frame and passed to the original 3DGS rasterizer. The web viewer path can load either a PLY or a LoD YAML config.

## 3. Complete Pipeline

| Step | Input | Operation | Output | Phase | Paper reference | Official source | Parameters/defaults |
|---|---|---|---|---|---|---|---|
| Coarse global prior | COLMAP/Blender-style scene data | Train normal 3DGS on all observations | Coarse Gaussian PLY/checkpoint | Training | Sec. 3.2, Fig. 2 | `train_large.py`, `scene.LargeScene`; coarse configs such as `config/mc_aerial_coarse.yaml` | 30,000 iterations; MatrixCity coarse position LR `0.00008`, scaling LR `0.0025`, densify 500-15,000, interval 100. |
| Contracted partition | Coarse prior positions and AABB | Normalize AABB to `[-1,1]`, contract unbounded points, uniformly divide contracted `[0,1]^3` | Block IDs and masks | Offline/training prep | Sec. 3.2, Eq. 2 | `utils/large_utils.py::contract_to_unisphere`, `block_filtering`, `which_block` | `ord=torch.inf`; MatrixCity `block_dim=[6,6,1]`, `aabb=[-3.5,-4,-10,4.5,2,10]`. |
| Data assignment | Cameras, coarse prior, block masks | Assign camera if inside block or if removing the block changes the prior render by SSIM loss above threshold | `data_partitions/{partition}.npy` camera/block mask | Offline/training prep | Sec. 3.2, Eq. 3-5 | `data_partition.py::block_partitioning` | MatrixCity `ssim_threshold=0.05`; `num_threshold` source default unresolved without generated `cfg_args`; bounds grow by `0.01` while under threshold. |
| Block fine-tuning | Coarse prior, assigned cameras per block | Train each block independently using 3DGS L1 + SSIM loss | Per-cell block outputs | Training | Sec. 3.2 | `train_large.py --block_id`, `scene.LargeScene` partition loading | Fine-tune 30,000 iterations; MatrixCity position LR `0.000064`, scaling LR `0.004`, `lambda_dssim=0.2`, densify 500-15,000 interval 200. |
| Spatial filtering | Fine-tuned block Gaussians | Keep only Gaussians whose contracted positions lie inside the block | `point_cloud_blocks/scale_1.0/.../point_cloud.ply` | Offline construction | Sec. 3.2 | `scene/__init__.py::LargeScene.save` | `save_block_only=True` default in `utils/general_utils.py`. |
| Fusion | Filtered block PLYs | Directly concatenate block Gaussian tensors | Fused `point_cloud/iteration_30000/point_cloud.ply` | Offline construction | Sec. 3.2, Appendix D | `merge.py::blockMerge` | Falls back to iteration 1 for a missing block iteration; no blending. |
| LoD generation | Fused dense Gaussians | LightGaussian compression / pruning / distillation / vector quantization | Multiple compressed model directories | Offline construction | Sec. 3.3, Fig. 3; Sec. 4.3 | README `LargeLightGaussian` commands; YAML `lod_configs` | Paper MatrixCity rates: 50%, 34%, 25% named LoD2, LoD1, LoD0. Local config paths: `*_light_50_vq`, `*_light_66_vq`, `*_light_75_vq`; exact rate semantics unresolved. |
| Block packing and bounds | One loaded LoD model at a time | Partition model into blocks, pack `[xyz, opacity, SH, scale/rot or covariance]`, estimate MAD bounds and per-block average scale | `BlockedGaussian` object | Runtime initialization | Sec. 3.3, Eq. 6 | `scene/gaussian_model.py::BlockedGaussian.cell_divider` | MAD multiplier `n=4`; `avg_scalings=max(mean(scale per block), dim=-1)`. |
| Visibility | Camera, block corners | Project front-facing corners, form 2D box, intersect with `[-1,1]^2`; force nearest block visible | Runtime | Sec. 3.3 | `utils/large_utils.py::in_frustum` | Near/front test `z > 0.2`; binary block mask; no padding in code. |
| LoD selection | Visible blocks and all LoD `BlockedGaussian`s | Choose one LoD per visible block by Nyquist rule | Selected block feature tensors | Runtime | Paper Sec. 3.3 fixed intervals; README/code Nyquist update | `gaussian_renderer/__init__.py::render_lod` | `nyquist_scalings = 2 * distance3D / focal_length`; `focal_length = 0.5 * image_width / tan(FoVx/2)`. |
| Render assembly | Selected block tensors | Concatenate selected block features, unpack Gaussian fields, render | Image/radii | Runtime | Sec. 3.3 direct concatenation | `render_lod`, `render_large_lod.py`, `viewer.py` | Original 3DGS rasterizer, SH degree from YAML; background from `white_background`. |

## 4. Training Methodology

**Verified from paper/source.** CityGS training is a two-stage pipeline. Stage one trains a coarse global prior on all observations for 30,000 iterations. Stage two uses that prior for both partition/data assignment and initialization of per-block fine-tuning. Fine-tuning is under original uncontracted space, not the contracted coordinates.

**Verified from paper/source.** Spatial contraction follows Eq. 2 in the paper. The implementation maps AABB coordinates to `[0,1]` after contraction: normalize to `[-1,1]`, apply the infinity-norm contraction, then compute `x / 4 + 0.5`. This implementation detail matters because official block IDs are in `[0,1]^3`, not in local arbitrary XY bounds.

**Verified from paper/source.** Data assignment has two criteria: contribution-based assignment with `loss = 1 - ssim(render_without_block, render_with_all)` and an in-block camera-center rule. The implementation renders the full coarse prior and the complement of the candidate block; if the SSIM loss exceeds `args.ssim_threshold`, the camera is assigned. For MatrixCity the config threshold is `0.05`.

**Verified from paper/source.** Blocks are trained independently with the 3DGS weighted L1/SSIM loss and ordinary densification/pruning. The official MatrixCity fine-tune config uses `sh_degree=3`, 30,000 iterations, `lambda_dssim=0.2`, `densify_from_iter=500`, `densify_until_iter=15000`, and `densification_interval=200`.

**Excluded from viewer-focused thesis. Inferred.** Reproducing coarse-prior training, data partitioning renders, per-block optimization, and LightGaussian training is expensive and not required for a viewer-side reproduction if official assets are available. The thesis can use author-provided assets for runtime behavior, but it must label training quality as author-provided.

**Important non-equivalence. Verified from current project.** The local 6x6 XY partition in `gs_framework/partitioned/mc_aerial_6x6.pt` is project-generated from the fused PLY and does not reproduce CityGS training partitioning, data assignment, block fine-tuning, or LightGaussian LoD construction.

## 5. Offline Rendering-Asset Construction

**Verified from paper/source.** The no-LoD representation is a fused PLY produced by direct concatenation of per-block filtered PLY files. This is consistent with the local 23.39M PLY, but it does not prove the local fused PLY is the exact parent of any missing local LoD asset.

**Verified from official source.** The LoD viewer/render path expects YAML with:

- `model_config.name: "GaussianModelLOD"`;
- `sh_degree: 2` for compressed LoD assets;
- `block_dim`, `aabb`, and `lod_configs`;
- `lod_configs` as ordered model directories, e.g. MatrixCity `output/mc_aerial_c36_light_50_vq`, `output/mc_aerial_c36_light_66_vq`, `output/mc_aerial_c36_light_75_vq`.

**Verified from paper/source.** The paper says MatrixCity uses three detail levels: LoD2 finest, LoD1 middle, LoD0 coarsest. Paper Sec. 4.3 names compression rates 50%, 34%, and 25%; the README command names use LightGaussian prune/distill/vector-quantize steps. The local repository does not contain initialized LightGaussian scripts, so exact pruning/distillation/quantization details are unresolved locally.

**Verified from official source.** The runtime block asset is not a persisted `lod-meta.json`. `BlockedGaussian` constructs cell feature tensors and MAD bounds in memory from each loaded LoD model. The block count is `block_dim[0] * block_dim[1] * block_dim[2]`; MatrixCity config uses 36 cells.

**Verified from released asset.** Local official LoD assets are present under `/home/luis/tese/mc_aerial_c36/LoD/`. Their PLY counts are 11,694,810 for `light_50_vq`, 7,952,471 for `light_66_vq`, and 5,847,405 for `light_75_vq`, all with SH degree 2. Each adjacent `extreme_saving` representation has `metadata.input_pc_num` equal to the PLY count, stores `xyz` and `other_attribute` for every Gaussian, and vector-quantizes SH color features through `codebook.npz` and `vq_indexs.npz`.

**Verified from released asset/source.** The `50`, `66`, and `75` name components denote pruning percentages, not retained percentages: counts are approximately 50%, 34%, and 25% retained relative to the 23,389,621 fused PLY. This matches `doc/custom_dataset.md`, which says higher `prune_percents` means lower model size.

## 6. Runtime Visibility Behavior

**Verified from paper/source.** The visibility unit is a block/cell, not a Gaussian. Each block is represented by eight MAD-bounded corners after packing.

**Verified from official source.** `in_frustum` projects each block's corners with `viewpoint_cam.full_proj_transform`, keeps only corners with camera-space `z > 0.2`, builds a 2D projected box, and checks positive-area intersection with the normalized screen rectangle `[-1,-1,1,1]`. It also forces the nearest block by minimum corner distance to be visible.

**Verified from official source.** Visibility is binary per block. All Gaussians from a selected block at the selected LoD are activated. Visibility is computed before LoD choice; the returned `distance3D` vector only contains visible blocks.

**Verified from current project/source slice/CUDA validation.** Independent selected-set oracle fixtures exist at `unit_tests/fixtures/citygaussian_visibility/finest_visibility_oracle_v1.json` and `unit_tests/fixtures/citygaussian_visibility/fused_visibility_oracle_v1.json`. Runtime `CityGaussianVisibilityScene` now matches those fixtures exactly for selected cell IDs, visible-cell counts, nearest-cell IDs, fallback flags, and selected Gaussian counts over deterministic trajectory indices `[0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439]`. Direct import of `utils.large_utils` is blocked locally by the missing `simple_knn` dependency chain, so the oracle source slice remains the independent reference.

**Verified from current project.** Oracle and runtime camera conversion now share the same projection/camera helper: start from the tracked Viser path, flip camera Y/Z axes, apply `scale_ratio`, derive `FoVx`/`FoVy` from the project benchmark vertical-FOV intrinsics, set `world_view_transform = inverse(converted camera_to_world).T`, use the official transposed projection matrix, and compute `full_proj_transform = world_view_transform @ projection_matrix`.

**Verified from official source.** There is no official runtime streaming in the inspected V1 Python path. All LoD models are loaded to GPU, converted to `BlockedGaussian` feature tensors, and selected block tensors are concatenated every frame. The official README also has a web viewer path; SIBR is listed as a submodule but is absent locally.

**Unresolved.** The code's screen-box projection is conservative for corners in front of the camera but can miss blocks whose corners are behind/near-plane while volume intersects the frustum. The paper does not specify a more robust volume/frustum clipping rule.

## 7. Runtime LoD Behavior

**Verified from paper.** The paper's standard MatrixCity setting uses fixed distance intervals: `[0,200]` meters for LoD2, `[200,400]` for LoD1, and `[400,infinity]` for LoD0. Other datasets use intervals from Table S3.

**Verified from official source.** The inspected `V1-original` source uses a Nyquist rule instead of those fixed intervals. For each visible block:

```text
focal_length = 0.5 * image_width / tan(FoVx / 2)
nyquist_scaling = 2 * min_corner_distance_to_camera / focal_length
lod_idx = first level whose avg_block_scaling > nyquist_scaling
fallback = last level when no level satisfies the comparison
```

`lod_list` order is the YAML `lod_configs` order. `render_large_lod.py` prints `Init LoD {len(lod_configs)-i}` for each path, so the first MatrixCity path is treated as LoD3/finest in the print convention, while the paper uses LoD2 as finest. The code uses integer indices rather than paper labels.

**Verified from official source.** There is no blending, no hysteresis, and no transition smoothing. One LoD is selected per visible block. All visible blocks use the same rule.

**Verified from current project/source slice on 2026-07-16.** The independent selected-LoD oracle is extracted at `unit_tests/fixtures/citygaussian_lod/release_lod_oracle_v1.json`. It uses the official source-sliced visibility function, the coarsest level's MAD bounds for visibility as in `render_lod`, and the source Nyquist rule. Over the 12 frozen trajectory cameras, release-policy selected Gaussian counts range from `1,957,671` to `6,211,124`.

**Provisional paper-policy oracle.** `unit_tests/fixtures/citygaussian_lod/paper_lod_oracle_v1.json` applies the MatrixCity intervals `0-200`, `200-400`, and `400+` as a clearly separated policy. Under the current project camera conversion all sampled visible-cell distances fall below 200, so it selects the finest level for every visible cell. This fixture is marked provisional because the paper interval units/scale compatibility have not been validated against official renders.

**Discrepancy. Human review required for paper-faithful claims.** A paper-faithful implementation must decide whether to reproduce the paper equations/tables or the released source/README behavior. For code-faithful reproduction of the inspected repository, use the Nyquist rule. For paper-figure reproduction, fixed intervals must be implemented and compared separately.

## 8. Rendering Backend and Data Assembly

**Verified from official source.** The official V1 path uses GraphDECO's 3DGS rasterizer (`diff-gaussian-rasterization`) through `GaussianRasterizer`, not `gsplat`.

**Verified from official source.** Gaussian features passed to the renderer are unpacked as:

- means: `feats[:, :3]`;
- opacity: `feats[:, 3]`;
- SH features: `feats[:, 4:feat_end_dim].reshape(-1, (degree+1)^2, 3)`;
- either precomputed covariance or `scale[3]` plus `rotation[4]`.

The no-LoD MatrixCity config uses `sh_degree=3`; the LoD config uses `sh_degree=2`, consistent with compressed assets reducing feature dimension.

**Verified from official source.** Camera render settings use `FoVx`, `FoVy`, `world_view_transform`, `full_proj_transform`, `camera_center`, image width/height, `prefiltered=False`, `scale_modifier=1.0`, and background `[0,0,0]` for MatrixCity because `white_background=False`.

**Verified from official source.** Selected block tensors are concatenated with `torch.cat(out_list, dim=0)` every frame before rasterization. The implementation assumes all loaded LoD resources are GPU resident; it does not copy selected blocks from CPU on demand.

**Inferred.** Rendering through `gsplat` can be a practical thesis backend, but image equivalence is not automatic because rasterizer implementation, SH layout expectations, antialiasing defaults, and camera matrix conventions may differ.

## 9. Released Asset Audit

| Asset | Status | Format/counts | Suitability |
|---|---|---|---|
| `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply` | **Verified from released asset.** Author-provided or author-derived CityGaussian fused PLY, source path used by current flat benchmark. | Binary little-endian PLY, 23,389,621 vertices, SH-degree-3-style fields, 5,800,627,541 bytes. | Valid for `FLAT-CITYGS-FULL`; valid no-LoD fused model for cross-representation comparisons; not equivalent to finest compressed LoD because the finest compressed level is pruned to 11,694,810 SH2 Gaussians. |
| `/home/luis/tese/mc_aerial_c36/cameras.json` | **Verified from released asset.** Local camera JSON beside fused PLY. | JSON camera list with width/height, position, rotation, `fx`, `fy`; file is about 2.28 MB. | Useful for official viewer orientation/camera convention checks; not yet validated against project camera paths. |
| `/home/luis/tese/CityGaussian/config/mc_aerial_c36_lod.yaml` | **Verified from official source.** Official config template/reference. | Expects 36 cells, AABB `[-3.5,-4,-10,4.5,2,10]`, YAML order `light_50_vq`, `light_66_vq`, `light_75_vq`, `sh_degree=2`. | Required for asset validation; released local LoD directories correspond by basename. |
| `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq` | **Verified from released asset.** Official finest compressed LoD. | PLY: binary little-endian, 11,694,810 vertices, SH degree 2, 1,917,949,848 bytes. VQ `metadata.input_pc_num=11694810`, `codebook=(8192,27)`, `non_vq_feats=(7016886,27)`, `xyz=(11694810,3)`, `other_attribute=(11694810,8)`. | Valid fixed-finest asset for `CG-FIXED-FINEST-*`; covers all 36 runtime cells. |
| `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_66_vq` | **Verified from released asset.** Official middle compressed LoD. | PLY: binary little-endian, 7,952,471 vertices, SH degree 2, 1,304,206,251 bytes. VQ `metadata.input_pc_num=7952471`, `codebook=(8192,27)`, `non_vq_feats=(4771482,27)`, `xyz=(7952471,3)`, `other_attribute=(7952471,8)`. | Valid middle LoD; covers all 36 runtime cells. |
| `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_75_vq` | **Verified from released asset.** Official coarsest compressed LoD. | PLY: binary little-endian, 5,847,405 vertices, SH degree 2, 958,975,427 bytes. VQ `metadata.input_pc_num=5847405`, `codebook=(8192,27)`, `non_vq_feats=(3508443,27)`, `xyz=(5847405,3)`, `other_attribute=(5847405,8)`. | Valid coarsest LoD; covers all 36 runtime cells. |
| `lod-meta.json` | **Verified from released asset/source.** Not present and not required by official V1 Python runtime. | Runtime constructs `BlockedGaussian` cells in memory from each loaded LoD model. | Do not design around this filename unless another asset supplies it. |
| `/home/luis/tese/gs_framework/partitioned/mc_aerial_6x6.pt` | **Verified from current project.** Project-generated partition. | PyTorch zip, about 5.52 GB; local blocked asset format, not official. | Useful legacy baseline only; explicitly non-official. |
| `/home/luis/tese/gs_framework/partitioned/citygaussian_matrixcity_aerial_10x10_bounds_clean.pt` | **Verified from current project.** Project-generated partition with cleanup metadata. | PyTorch zip, about 5.52 GB; cleanup metadata records 23,389,621 total Gaussians, 19,740,454 bounds-valid after opacity/scale filters. | Useful local bounds/culling experiment only; not CityGaussian V1 methodology. |

## 10. Controlled Experimental Configurations

### `CG-ASSET-VALIDATION`

**Status: implemented as metadata validation.** It validates official config/source expectations and local asset availability only: fused PLY, camera JSON, YAML config, LoD PLY/VQ paths, block dimensions, AABB, expected SH degree, and license/source metadata. It makes no runtime optimization claim.

### `CG-FULL-DETAIL`

**Status: partially valid with clarified meanings.** The fused PLY is a semantically valid no-LoD full-detail CityGS representation, consistent with paper `CityGS(noLoD)` and `FLAT-CITYGS-FULL`. It is distinct from the finest released compressed LoD, `mc_aerial_c36_light_50_vq`, which is an SH2 pruned/VQ-compatible complete-scene alternative with 11,694,810 Gaussians.

### `CG-VISIBILITY`

**Status: implemented and CUDA-validated for visibility selected-set work.** The finest official compressed LoD asset and fused no-LoD PLY can be divided into the same 36 runtime cells in memory with official-compatible cell construction, and every cell is nonempty. `CG-FIXED-FINEST-VISIBILITY` and `CG-FUSED-VISIBILITY` match the independent selected-set oracles over all 12 frozen cameras, including nearest-cell fallback cases. A local project partition still cannot be used to claim CityGaussian visibility.

Controlled branches for Spec 003:

- `CG-FIXED-FINEST-ALL -> CG-FIXED-FINEST-VISIBILITY` uses the same finest compressed LoD PLY and changes only official visibility-selected active cells.
- `FLAT-CITYGS-FULL -> CG-FUSED-CELLS-ALL -> CG-FUSED-VISIBILITY` uses the same fused PLY, first adding all official cells, then changing only visibility-selected active cells.

Validation snapshot from 2026-07-08:

- finest oracle: selected cells `9..35`, selected Gaussians `3,248,918..11,694,702`, fallback trajectory index `1177`;
- fused oracle: selected cells `10..35`, selected Gaussians `6,422,577..23,389,336`, fallback trajectory indices `654` and `785`;
- `FLAT-CITYGS-FULL` vs `CG-FUSED-CELLS-ALL`: RGB/alpha max absolute difference `0.0`, total/assigned Gaussian count `23,389,621`, unassigned `0`, duplicated `0`;
- visibility render samples are finite for both finest and fused branches, with expected nonzero image differences against all-cells parents because cells are intentionally removed.

### `CG-FINEST-FLAT`

**Status: implemented and CUDA-validated by short run.** This configuration loads the finest official compressed LoD PLY, `mc_aerial_c36_light_50_vq/point_cloud.ply`, as one flat Gaussian set through the project flat path and `GsplatRenderer`. It uses SH degree 2, the CityGaussian 23M-compatible trajectory, black background, and all Gaussians active every frame. It performs no cell construction, visibility, adaptive LoD, host residency, streaming, VQ reconstruction, or hierarchy traversal.

### `CG-FIXED-FINEST-ALL`

**Status: implemented and CUDA-validated by short run.** This configuration loads the same finest official compressed LoD PLY, constructs official runtime cells with `block_dim=[6,6,1]`, AABB `[-3.5,-4,-10,4.5,2,10]`, contraction to unisphere, x-fastest cell IDs, half-open cell intervals, MAD bounds with `n=4`, and per-cell average activated scale. It activates all 36 cells every frame, so selected Gaussian count equals total Gaussian count. It uses `gsplat` rather than the official GraphDECO rasterizer and does not implement visibility or adaptive LoD. Short validation on 2026-07-06 selected all 11,694,810 Gaussians and matched `CG-FINEST-FLAT` exactly for RGB and alpha over frames `[0, 1, 2]`.

### `CG-LOD-RELEASE`

**Status: released-code policy implemented as `CG-LOD-RELEASE` and CUDA-validated.** Official LoD assets are loaded as complete-scene alternatives in YAML order. Runtime visibility uses coarsest-level MAD bounds, and strict first-match Nyquist selection exactly matches the authoritative oracle for all 12 frozen cameras. All three levels remain resident; selected and active counts are equal. The paper fixed-distance policy remains a separate provisional, unimplemented ablation. `CG-FIXED-FINEST-VISIBILITY -> CG-LOD-RELEASE` uses the same visibility path, cameras, renderer, and settings, changing the per-visible-cell level selection and necessarily adding middle/coarsest resident assets.

## 11. Relationship to `FLAT-CITYGS-FULL`

**Partially controlled / cross-representation with current evidence.** `FLAT-CITYGS-FULL` uses the 23.39M fused PLY as a fully resident flat scene. The paper and source establish that CityGS no-LoD fusion is direct concatenation of trained block outputs, so the fused PLY is a valid CityGS no-LoD asset. The released compressed LoD levels are present and their counts match 50%, 66%, and 75% pruning relative to the fused count, but exact LightGaussian generation lineage remains author-provided rather than locally reproduced.

Differences between `FLAT-CITYGS-FULL` and future CityGS runtime modes may come from partitioning, MAD bounds, LoD compression, SH-degree changes, vector quantization, block filtering, and rasterizer differences. Do not claim that all improvement over the flat PLY comes from runtime culling.

## 12. Metrics Required

Required for `CG-ASSET-VALIDATION`: asset path, source status, file existence, file size, PLY vertex count, SH degree inferred from fields, config path, block count, AABB, expected LoD paths, LoD path existence, license/reference commit, and validation failures.

Required for `CG-FIXED-FINEST-VISIBILITY` and `CG-FUSED-VISIBILITY`: total Gaussians in representation, total blocks, selected/visible blocks, active Gaussian count, selected Gaussian count, visibility-selection time, concatenation/preparation time, rasterization time, end-to-end frame time, PyTorch allocated/reserved memory, image difference versus no-visibility reference where valid, and selected-block equivalence against official source for fixture cameras.

Required for `CG-LOD-RELEASE`: selected LoD per block, per-LoD selected block counts, per-LoD selected Gaussian counts, selected Gaussian count total, visibility time, LoD-selection time, concatenation time, rasterization time, frame time, VRAM, PSNR/SSIM/LPIPS on fixed cameras/GT, temporal popping/artifact notes, and image difference against official renderer output where available.

Now exposed for `CG-LOD-RELEASE`: separate resident/selected/active counts, logical cell IDs, selected LoD indices/labels, per-level selected cell/Gaussian counts, visibility/LoD/assembly/rasterization/end-to-end timing, radii-derived visible count, allocator memory, finite flags, exact selected-set oracle validation, and parent/child `gsplat` image comparison. Still not available: image difference against the official GraphDECO renderer and true transfer bytes, because the released V1 runtime assumes residency.

## 13. Faithfulness Requirements

Before claiming `viewer-side reproduction of CityGaussian V1 visibility`:

- use an official fused or LoD-derived asset, or document and validate an official-equivalent construction;
- use official `block_dim`, `aabb`, contracted block assignment, MAD bounds with `n=4`, and camera matrix conventions;
- implement the official projected-corner/screen-rectangle visibility rule including `z > 0.2` and nearest-block fallback;
- select at block granularity and activate all Gaussians from visible blocks at the chosen reference detail level;
- compare selected block IDs/counts against the official source on fixture cameras;
- keep rasterizer deviations labeled if rendering through `gsplat`.

Before claiming `viewer-side reproduction of CityGaussian V1 LoD`:

- satisfy all visibility requirements;
- use the official LoD asset levels and correct SH degree per level;
- decide and document paper fixed intervals versus source Nyquist rule;
- match LoD ordering, fallback, no-blend/no-hysteresis behavior, and per-block granularity;
- log selected LoD per visible block and compare against official source on fixture cameras;
- validate image output against official renderer or mark rasterizer-induced differences.

## 14. Open Questions and Conflicts

| Issue | Sources | Consequence | Defensible interpretation | Human review |
|---|---|---|---|---|
| Distance intervals vs Nyquist LoD | Paper Sec. 3.3/4.3/Table S3 vs README "LoD selection is now based on Nyquist sampling rate" and `render_lod` code | A paper-faithful and code-faithful implementation differ. | Implement source-faithful Nyquist for V1 repository reproduction; keep paper-interval ablation separate and provisional until scale semantics are resolved. | Required only for paper-faithful claims. |
| LoD asset availability | Local `/home/luis/tese/mc_aerial_c36/LoD` contains three official MatrixCity LoD levels matching YAML order | Asset validation is unblocked. | Use these official assets for fixed-finest visibility and LoD work after selected-set oracles are defined. | No. |
| LightGaussian exact processing | Paper says LightGaussian compression; local submodule absent, but released outputs are present | Can validate released model counts/VQ layout, but not reproduce the training/compression pipeline. | Treat LoD levels as author-provided complete alternatives; do not claim LightGaussian generation is reproduced. | Required before generation claims. |
| Fused PLY parentage | Local fused PLY and released LoD assets have counts matching 50/66/75 pruning semantics | Strong count-level evidence that names are pruning percentages; exact generation lineage is still author-provided. | Classify flat-vs-LoD as author-asset comparison; do not attribute differences solely to runtime selection. | Required before lineage claims. |
| Official viewer backend | V1 Python path uses 3DGS rasterizer; project uses `gsplat` | Image equivalence and timing differ by backend. | First validate selected sets; render through `gsplat` as a labeled backend deviation. | Required before quality claims. |
| Official runtime residency | Source loads all LoD models on GPU; paper claims VRAM savings | Savings come from active set/rasterization and perhaps compressed levels, not out-of-core streaming in this code. | Do not claim official V1 streaming. | No, unless adding memory streaming. |

## Reference-Code Mapping Table

| Methodology step | Class | Paper reference | Reference path | Class/function | Important parameters | Thesis scope | Verification |
|---|---|---|---|---|---|---|---|
| Coarse prior training | Training | Sec. 3.2 | `train_large.py`, `config/mc_aerial_coarse.yaml` | `training`, `LargeScene` | 30k iters, SH3, 3DGS loss | Out of viewer scope | Verified |
| Contracted partition | Offline/training | Eq. 2 | `utils/large_utils.py` | `contract_to_unisphere`, `block_filtering` | `ord=torch.inf`, AABB, block_dim | Needed for asset validation | Verified |
| Camera assignment | Offline/training | Eq. 3-5 | `data_partition.py` | `block_partitioning` | `ssim_threshold`, in-block rule, `num_threshold` | Document only | Verified |
| Block fine-tune | Training | Sec. 3.2 | `train_large.py`, `scene/__init__.py` | `training`, `LargeScene` | `--block_id`, 30k iters | Out of viewer scope | Verified |
| Block filtering | Offline | Sec. 3.2 | `scene/__init__.py` | `LargeScene.save` | contracted block mask | Needed to understand fused PLY | Verified |
| Fusion | Offline | Sec. 3.2, Appendix D | `merge.py` | `blockMerge` | direct tensor concat | Asset lineage only | Verified |
| LoD generation | Offline | Sec. 3.3 | README, absent `LargeLightGaussian` | external scripts | LightGaussian prune/distill/VQ | Blocked | Partly verified |
| LoD config loading | Runtime init | Fig. 3 | `viewer.py`, `render_large_lod.py` | `_load_model_from_file`, `load_gaussians` | YAML `lod_configs`, `load_vq` | Spec 001/004 | Verified |
| Block packing/MAD bounds | Runtime init | Eq. 6 | `scene/gaussian_model.py` | `BlockedGaussian.cell_divider` | `n=4`, avg scale | Spec 003/004 | Verified |
| Visibility | Runtime | Sec. 3.3 | `utils/large_utils.py` | `in_frustum` | projected box IoU, `z>0.2`, nearest fallback | Spec 003 | Verified |
| Visibility oracle fixtures | Validation | Sec. 3.3/source behavior | `scripts/extract_citygaussian_cell_oracle.py`, `unit_tests/fixtures/citygaussian_visibility/*.json` | source-sliced `in_frustum` | 12 deterministic trajectory frames, official cell ordering, per-asset MAD bounds | Spec 003 validation | Verified |
| LoD selection | Runtime | Sec. 3.3 conflict | `gaussian_renderer/__init__.py` | `render_lod` | Nyquist `2*d/focal`, first passing level | Spec 004 | Verified with conflict |
| Rasterization | Runtime | Sec. 3.3 | `gaussian_renderer/__init__.py` | `render_lod`, `GaussianRasterizer` | SH, scale/rot, covariance option | Project may use `gsplat` deviation | Verified |

## Fable Readiness

| Spec | Status | Recommendation |
|---|---|---|
| `001-citygaussian-asset-validation` | READY | Suitable for Codex first; Fable not required. |
| `002-citygaussian-full-detail-reference` | COMPLETE for fixed-finest parent | Same-asset flat and all-cell finest-LoD parents are implemented and CUDA-validated; formal benchmark remains separate. |
| `003-citygaussian-visibility` | COMPLETE | Runtime visibility matches finest and fused selected-set oracles, renders finite output, records parent-child metrics, and contains no LoD behavior. |
| `004-citygaussian-lod` | COMPLETE FOR RELEASED-CODE POLICY | `CG-LOD-RELEASE` exactly matches the release oracle and renders finite output through `gsplat`; paper fixed intervals remain provisional and unimplemented. |

Recommended first implementation task: `001-citygaussian-asset-validation`, preferably by Codex, because it is mostly path/config/header validation and readiness reporting.
