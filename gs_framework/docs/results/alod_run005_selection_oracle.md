# Official ALoD Run005 Selection Oracle

Date: 2026-08-08

Status: `PASS-ALOD-SELECTION-ORACLE`

## Scope and artifacts

This result establishes a cold, deterministic oracle from the pinned official
ALoD evaluation runtime. It does not implement a project-side selector and it
does not measure image quality or performance.

The external oracle is
`/home/luis/tese/runs/alod/oracles/ALOD-RUN005-SELECTION-ORACLE-001`.
Its deterministic semantic manifest SHA-256 is
`20bb87caf1694ede18ac2d2a71cdc146c002325ddbf8f6060625f18feda70591`.
The tracked compact locator is
`unit_tests/fixtures/alod_selection_oracle/run005-selection-oracle-v1.json`.

The runtime used the unchanged Run005 `final.dhier` with SHA-256
`6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c`:
1,099,999 serialized Gaussian/node rows, degree-1 SH, a 100,000-row
serialized skybox prefix, and hierarchy root `global_node_id=100000`.
The official source commit is `73f547341547ae021ff036ad74fdf73a1240b450`.
The Run005 training correctness-patch identity is
`a7412fff7315153774c93b504105804c2a182308807a2c6ec8de34672b9cd7d6`.

Source modification existed only in a disposable `git archive` extraction.
The source insertion patch SHA-256 is
`930b3945b89a4f33128197e0e4b0603561aad1e7869324042fc65201d0ba3dbd`;
the project-owned observational hook SHA-256 is
`f85336b80f0a05270c84b6803c31d4f6766e904934027ef115f4df1125260b48`.

## Authoritative call chain

The authoritative path is `eval_hierarchy.py`, not `GaussianModel.get_SPT_cut`:

```text
load_dynamic_hierarchy
  -> GaussianModel.create_from_hier(scaffold_file="", device="cpu")
  -> compact_gaussians
  -> build_hierarchical_SPT(150, 2 / 2317.6449482429634,
                            min_SPT_size=100, use_bounding_spheres=False)
  -> extract_frustum_planes
  -> cut_hierarchy_on_condition(root_node=0)
  -> candidate SPT IDs and camera distances
  -> gaussian_hierarchy._C.get_spt_cut_cuda
  -> GetSPTCut
  -> Switching::getSPTCut
  -> binary_search_SPTs
  -> populate_SPT
  -> upper global IDs + CUDA SPT global IDs
  -> properties[global IDs]
  -> prepend serialized skybox rows 0..99999
  -> render_gsplat
  -> gsplat.rasterization(packed=True, render_mode="RGB")
```

`GaussianModel.get_SPT_cut` is not source-faithful here because it replaces its
computed LoD predicate with an all-true predicate. The inline evaluation chain
does not make that replacement. The evaluation path rebuilds selected tensors
for each camera and has no useful inter-camera cache, unlike viewer
previous-working-set reuse and fine-training retained/write-back cache behavior.

The official frustum contains only left, right, bottom, and top planes. Near and
far are commented out. Because `use_bounding_spheres=false`, the traversal uses
`3 * exp(max(log_scale_components))` bounds rather than constructed hierarchy
bounding spheres. No six-plane correction was made.

The effective distance multiplier is 1. For a node with activated scales
`sx,sy,sz`, the source builds its distance threshold from
`sqrt(sx*sy + sx*sz + sy*sz) / target_granularity`; the upper tree stores the
squared parent threshold and compares it with squared camera distance. Each
candidate SPT receives its Euclidean root-to-camera distance. The CUDA path
binary-searches the SPT maximum-distance array, then retains compacted entries
whose SPT minimum distance is below the camera distance. The full SPT min/max
arrays plus each camera's candidate roots and distances are exported, so
decision-boundary failures can be reconstructed without a separate massive
per-node trace.

## HSPT structural oracle and identity proof

The effective construction has `SPT_root_volume=150`, target granularity
`0.0008629449482830516`, 1,309 upper rows, 23 SPTs, 998,713 flattened SPT
metadata rows, and 1,000,022 total HSPT metadata rows. SPT sizes range from 111
to 249,619. The complete size histogram, upper-depth distribution, array
shapes/dtypes/ranges, raw hashes, and payload hashes are in `hspt.json`.

The external validator proved:

- all 1,309 upper mappings are unique serialized global IDs;
- all 998,713 flattened SPT mappings are unique serialized global IDs;
- their intersection is exactly the 23 SPT roots;
- their union is exactly serialized rows 100,000 through 1,099,998; and
- `upper_tree_nodes[:,5]`, `SPT_starts`, and each SPT interval root agree with
  the separately exported mappings.

The ID spaces are therefore:

| ID space | Mapping to serialized row | Cardinality and lifetime |
|---|---|---|
| `global_node_id` | Identity: `.dhier` row index | Stable, persistent, one-to-one |
| hierarchy-array index | Identity in this unchanged loaded asset | Stable, persistent, one-to-one |
| upper local ID | `upper_tree_nodes[local,5]` | Derived HSPT-local one-to-one subset |
| SPT ID | Interval `SPT_starts[id]:SPT_starts[id+1]`; root is its first mapped row | Derived, one SPT identity per root |
| SPT-local ID | `SPT_gaussian_indices[start + local]` | Derived one-to-one within the flattened SPT partition |
| combined HSPT row | Upper map or flattened SPT map | Derived; the 23 roots occur once in each representation |
| CUDA cut output row | Already a serialized global ID | Stable model identity in official order |
| selected working-set row | Explicit exported selected-ID vector | Frame-local one-to-one order |
| raster row | Explicit `[0..99999] + selected IDs` vector | Frame-local one-to-one order |

No cache slot, working-set offset, HSPT-local ID, SPT-local ID, or raster index is
used as the oracle identity.

## Cameras and selections

All frozen cameras are 1920x1080 with `fx=fy=2317.6449482429634` and centered
principal point. Official `--resolution 2` produces 960x540 with
`fx=fy=1158.82247412148`, `cx=480`, and `cy=270`. Each `camera.json` retains the
COLMAP registered image ID, camera ID, filename, stable indices, full frozen
world-to-camera pose, center, intrinsics, dimensions, effective official
matrices, category, and selection rationale. The set covers near/central/far,
low/middle/high, boundary/interior positions, and varied frozen orientations.

The `upper / SPT` column is emitted upper nodes followed by nodes returned by
the CUDA SPT cut. Raster count additionally includes exactly 100,000 skybox
rows. Every camera has selected depths 2, 16, 17, 18, 19, 20, 21, 22, and 23.

| Camera stable identity | Frozen center | Category | Candidate SPTs | Upper / SPT | Selected / raster | Ordered selected-ID SHA-256 | Sorted selected-ID SHA-256 |
|---|---|---|---:|---:|---:|---|---|
| `eval-0000-img-000001-cam-0001` | (-9.500000, -2.900000, 1.500000) | far/low/boundary | 22 | 199 / 364158 | 364357 / 464357 | `4f62b3ef15e767c218374bea91ad7826e1e560cadd2fa16483a8de76369cee17` | `ba5572ea58e82145ba7109a12314922da269a113ba6dddaf68a4c2437c5ae60a` |
| `eval-0002-img-000101-cam-0001` | (-3.000000, -2.500000, 1.500000) | central/low/interior | 23 | 167 / 498252 | 498419 / 598419 | `92b50aa4a64e758a929d58a9cdea8bcd375ec28dbfee35b6cc8c4e1d83357874` | `f311dcce60b6227870282d6cd5e88c2440c325f6dc36ed704d246dd4acdffc34` |
| `eval-0004-img-000201-cam-0001` | (0.188889, -4.500000, 5.000000) | far/high/interior | 23 | 215 / 495887 | 496102 / 596102 | `29caceb3b90ec5e1b2176d4fc0ae44798f392c2617001f6086f1c8114ab0deaa` | `2de1d7bd8faf144732fb73ad9c98442dc59b713f75e05e56bee28c385cc03dcf` |
| `eval-0006-img-000301-cam-0001` | (4.700000, -5.390000, 4.500000) | far/high/interior | 22 | 194 / 486475 | 486669 / 586669 | `ed06954feed0cf1bbe217ec5212183eddea9cc7bbf81860c893281ee90b02491` | `9c302f81fa1e2943cfad679df2a3dc08bbee6d8b74490b99955300ae3a581e94` |
| `eval-0007-img-000351-cam-0001` | (2.372222, -0.500000, 3.500000) | near/high/interior | 22 | 105 / 497927 | 498032 / 598032 | `ac77061045f3f3ff41abeeae180ab6c7eaa091d282220a47f4344e8e764a6c34` | `227dd7d7ecfe4994acbf578e2832b531764a04fc2a8d9ce06e617558c6b4fbc7` |
| `eval-0008-img-000401-cam-0001` | (-5.312500, 0.914286, 2.500000) | central/middle/interior | 22 | 88 / 494531 | 494619 / 594619 | `c929d0f6a5a0d827c8010f826b20cfb4feba3b18cc712f9a9ab0960f63da1909` | `6a4cb8d0f5051a7beea2506b013eb754e1d3b3fb4773722649715f9a21c8e4bc` |
| `eval-0009-img-000451-cam-0001` | (-2.184375, 3.400000, 2.500000) | near/middle/interior | 22 | 128 / 497640 | 497768 / 597768 | `226f6745001646b32ba06becaa1337f536c18b4b8a4f23163cc8e358d7f32a37` | `aaac9678f54bf372a2452938e7c70c168e194cf24296d741fa75ddf23ad72770` |
| `eval-0010-img-000501-cam-0001` | (0.496875, 2.571429, 2.500000) | near/middle/interior | 23 | 133 / 498897 | 499030 / 599030 | `eff7e7d4cede11434d524f1b094e114d4b0e70a1e4084661bd596808d3ad6dfd` | `65369c79b06e0241b89931d8b7d57085a18a5ad75b6d3250c756adb8c75c5d74` |
| `eval-0012-img-000601-cam-0001` | (5.859375, 0.914286, 2.500000) | central/middle/boundary | 23 | 193 / 491122 | 491315 / 591315 | `4b99fb6eb26765bae669d70704ab6538ea2a3b05ef011fde02d7759bf8760477` | `6cdacd574b46346326b678ed9ea4c46495826923f3f4272a2f31e835b97c86bb` |
| `eval-0014-img-000701-cam-0001` | (1.966667, 5.100000, 2.000000) | central/middle/boundary | 22 | 167 / 496042 | 496209 / 596209 | `06e08d013326634c6ac9adfabf9c2c88316007394bdb6c10faf44fc8b9a4cc94` | `f6583b59727d06557bc233e72581b91e5e012255ab795e5dafa447ede88eadd3` |

The exact per-camera depth distributions are in the deterministic manifest.
Each `pre-raster.json` records full shape, dtype, official device, raw CPU-byte
SHA-256, finite count, and range for means, quaternions, scales, opacities, SH
features, view matrices, intrinsics, and background. The deterministic manifest
contains all per-camera tensor hashes; the tracked locator contains one
pre-raster bundle hash per camera. Intrinsics and black-background hashes are
constant across cameras:

- intrinsics: `6c25a2f285a6c609ff399eee70959afb7f79d4d4f78e4d39d4e4d4555b37abf2`;
- background: `15ec7bf0b50732b49f8228e07d24365338f9e3ab994b00af08e5a3bffe55fd8b`.

For every camera, the validator loads the raster-ID payload and proves it is
exactly rows 0 through 99,999 followed by the ordered selected hierarchy IDs.
The instrumentation also compared the gathered stored properties to those
indexed by that vector and obtained bit-exact equality for means, log scales,
raw quaternions, opacity logits, DC features, and rest features.

## Determinism and LoD diagnostic

“Cold” means a fresh OS process, a fresh `GaussianModel`, a rebuilt HSPT, and no
prior working set, cache, or SPT cut. Every camera ran twice. Repetition one
wrote full payloads and repetition two wrote hashes only; HSPT hashes, camera
matrices, upper traversal order and sets, ordered and sorted selected global
IDs, selected-source-row hashes, working-set mapping, all pre-raster tensor
hashes, and the rendered-image hash matched exactly.

The observer only clones or transfers already-computed arrays, validates
mappings, hashes them, and writes files. It does not write official tensors or
change predicates, matrices, thresholds, traversal, CUDA inputs, gathering, or
raster arguments. An uninstrumented one-camera comparison is not directly
available without changing the official evaluator, because its unmodified path
runs the whole quality evaluation; the full-payload/hash-only agreement and
source-level mutation audit are the practical observational-equivalence checks.

Selected counts are minimum 364,357, median 496,155.5, and maximum 499,030.
The minimum pairwise selected-set Jaccard is 0.663362; sequential Jaccards are
0.706892, 0.986610, 0.949664, 0.945350, 0.969452, 0.982807, 0.990582,
0.961270, and 0.968093. Across all selected nodes, minimum depth is 2, median
depth is 19, and maximum depth is 23; every camera has the same 2/19/23 depth
summary. The maximum pairwise depth-distribution
total variation is only 0.019207.

The frozen criteria are:

- meaningful set variation: minimum pairwise Jaccard at most 0.95;
- material count variation: `(max-min)/median` at least 0.05;
- material depth variation: maximum selected depth varies or depth-distribution
  total variation is at least 0.05;
- `NONTRIVIAL` requires all three, `DEGENERATE` requires identical sets/counts/
  depth distributions, and otherwise the result is `WEAK`.

The data has nine selected depths, 27.14% relative count range, and meaningful
set variation, but invariant maximum depth and only 1.92% depth-distribution
variation. It is therefore `WEAK-LOD-ORACLE`. It remains valid for exact
correctness testing, but a future mixed-scale dataset is still required for
formal LoD evaluation.

## Validation

The project reader validates schemas, global-ID bounds and uniqueness, every
referenced NumPy payload/file/raw hash, HSPT coverage, selected/raster ID order,
skybox placement, tensor row counts, and source-row correspondence. Default
unit tests use synthetic tiny fixtures and require neither Run005 nor CUDA. The
external integration test validates the complete exported tree.

No training, quality evaluation, benchmark trajectory, hierarchy mutation,
CityGaussian execution, or project-side ALoD rendering was performed.

## Next bounded task

Implement the project-owned CPU reference HSPT/selection path and prove exact
selected-global-node-ID parity against this official Run005 oracle before any
optimized GPU/runtime integration.
