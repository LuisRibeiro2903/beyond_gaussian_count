# CityGaussian V1 Asset Validation

## Objective

Implement a read-only validator for locally configured CityGaussian V1 assets and reference metadata. The task proves what assets are present, what official source/config expectations apply, and which runtime reproduction tasks are blocked. It must not render, benchmark, or claim runtime optimization.

## Thesis Relevance

This supports the CityGaussian branch by establishing whether later visibility and LoD experiments use official assets, local project-generated assets, or missing/incomplete inputs.

## Current State

`docs/papers/citygaussian-v1-method-card.md` verifies the official repository at `/home/luis/tese/CityGaussian`, branch `V1-original`, commit `e16c526f78ab01d377b13b3e6862dcc93ddbe953`. Current project flat loading and PLY header handling live in `scene_runtime/ply_splats.py`, while benchmark presets live in `benchmarks/configs/`.

## Paper And Reference Repository

- Paper title: `CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians`
- Authors/year: Yang Liu, He Guan, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, Zhaoxiang Zhang; ECCV 2024
- Paper URL/DOI: arXiv `2404.01133`
- Reference code URL and commit/tag: `https://github.com/DekuLiuTesla/CityGaussian.git`, `V1-original`, `e16c526f78ab01d377b13b3e6862dcc93ddbe953`
- License notes: CC BY-NC-SA 4.0; keep adaptation and redistribution license implications explicit.

## Reproduction Scope

Validate asset presence, official config fields, PLY header semantics, file sizes, expected LoD paths, and local-vs-official classification. No visibility, LoD, active-set construction, rendering, or performance claim is included.

## Training Scope

No training changes. Existing author-provided or local assets are inspected only.

## Offline Construction Scope

No asset construction. The validator may parse lightweight metadata and headers, but must not create large derived assets.

## Runtime Scope

No runtime rendering. The only architecture interaction is an optional shared helper for PLY header parsing if it can reuse `scene_runtime/ply_splats.py` without forcing GPU imports.

## Explicitly Excluded Work

No renderer code, no `BlockedGaussian` implementation, no LoD selector, no `torch.load` of multi-GB assets unless a metadata-only path is proven safe, no package installation, no reference-repo modification.

## Methodology Steps

1. Add a CLI or script under `scripts/` or `benchmarks/` that accepts paths to the CityGaussian repo, fused PLY, camera JSON, and optional LoD config.
2. Read official config YAML fields for `block_dim`, `aabb`, `sh_degree`, `lod_configs`, `white_background`, and `source_path`.
3. Parse the fused PLY header only and report vertex count and property names.
4. Check existence and size of every expected LoD directory without loading Gaussian tensors.
5. Emit JSON with evidence labels and readiness flags.

## Mapping To Reference Code

Use `config/mc_aerial_c36.yaml`, `config/mc_aerial_c36_lod.yaml`, `doc/citygaussianv1_readme.md`, and `utils/large_utils.py` only as read-only references.

## Permitted Deviations

The validator may use project-native JSON output instead of official CityGaussian output because no rendering behavior is reproduced.

## Required Inputs

- `/home/luis/tese/CityGaussian`
- `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply`
- `/home/luis/tese/mc_aerial_c36/cameras.json`
- `/home/luis/tese/CityGaussian/config/mc_aerial_c36_lod.yaml`

## Architecture Touchpoints

Prefer lightweight utilities. Do not couple this validator to `BlockManager` or the experimental blocked-scene path.

## Invariants

Flat rendering remains independent. Generated reports go to explicit output paths outside tracked source unless they are tiny committed fixtures approved by the task.

## Failure Cases

Missing LoD paths are expected and should produce `BLOCKED - missing asset`, not an exception unless strict mode is requested.

## Performance Constraints

Runtime should be dominated by metadata/header reads. It must not read the full 5.8 GB PLY body.

## Acceptance Criteria

- Reports official repo path, branch, commit, dirty state, and license.
- Reports fused PLY vertex count `23,389,621` from header.
- Reports MatrixCity config block count `36`.
- Reports missing local LoD directories.
- Classifies project-generated partitions as non-official if inspected.

## Required Tests

Unit tests for PLY-header parsing, missing-path reporting, and config-field extraction using small fixtures.

## Required Benchmark Evidence

None. This is validation only.

## Required Completion Report

Report files changed, inputs inspected, readiness flags, and confirmation that no rendering/training/benchmarking/reference-repo modification occurred.

## Fable Readiness

READY, but this is more appropriate for Codex than Fable.
