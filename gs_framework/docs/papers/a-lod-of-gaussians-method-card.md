# A LoD of Gaussians — Method Card

Generated: 2026-08-05

## Post-audit local validation — 2026-08-07

The first accepted local hierarchy is Run005, produced from the pinned source
plus `ALOD-RELOCATE-TOPOLOGY-FIX`. Its final degree-1 hierarchy contains
1,099,999 serialized Gaussian/node rows, including a 100,000-row skybox prefix,
and has SHA-256
`6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c`.
It is a **correctness-patched local reproduction**, not a bit-identical
pinned-source reproduction. The unpatched Run004 final remains rejected with
4,355 depth inconsistencies, one reverse-parent mismatch, and a corrupted
skybox metadata row caused by the same relocation bug.

The project now owns a CPU-only, read-only parser under `alod_dhier/`. It uses
serialized row/index as `global_node_id`, performs exact field-major byte
decoding, and has no dependency on CUDA, gsplat, CityGaussian, the official
Python package, or the official extension. The native loader is used only by
separate oracle-export tooling.

## Audit scope and evidence policy

This card documents the method needed to reproduce **A LoD of Gaussians** (ALoD) inside the thesis framework without changing the frozen CityGaussian implementation. It is a documentation-only audit: no code was compiled, installed, trained, rendered, or benchmarked.

Evidence labels used below:

- **Paper**: arXiv v4, *A LoD of Gaussians: Out-of-Core Training and Rendering for Seamless Ultra-Large Scene Reconstruction*, arXiv:2507.01110.
- **Official source**: `FelixWindisch/LoDOfGaussians`, commit `73f547341547ae021ff036ad74fdf73a1240b450` (2026-07-03).
- **Public-code behavior**: behavior observed in that commit, which may differ from the paper.
- **Project inference**: a conclusion or proposed adaptation for `gs_framework`, not a claim about the authors' implementation.

The paper title evolved across versions. The repository and project page still use the earlier “Unified Training and Rendering” wording in places. This audit pins arXiv v4 and the source commit above so that the reproduction target is unambiguous.

## Contribution in one paragraph

ALoD treats an ultra-large Gaussian scene as a binary hierarchy whose interior nodes are renderable approximations, not merely spatial-index nodes. A camera-dependent hierarchy cut chooses an appropriate level of detail, while Hierarchical Spatial Partitioning Trees (HSPTs) make the lower hierarchy searchable in parallel on the GPU. The complete Gaussian attributes and optimizer state remain in host memory during out-of-core training; a bounded GPU working set holds selected hierarchy cuts, persistent upper-tree/skybox data, and cache metadata. This couples representation, visibility, LoD selection, and memory management into one system rather than applying block culling to an otherwise flat model. **Evidence: Paper; Official source.**

## Method stages

### 1. Dataset ingestion

The public implementation consumes a conventional COLMAP scene with images and sparse reconstruction. `Scene` and the dataset readers load cameras, images, optional masks/depths, and the initial sparse points. Camera conventions are subsequently converted for the gsplat renderer. **Evidence: Official source** (`scene/__init__.py`, `scene/dataset_readers.py`, `utils/camera_utils.py`).

The released source is not a direct reader for the local MatrixCity NeRF-transform layout. A reproduction must either use a standard COLMAP dataset or create a separately validated conversion. **Evidence: Official source; Project inference.**

### 2. Coarse scaffold training

The sparse point cloud and skybox seed a conventional 3DGS scaffold. The paper describes a fully GPU-resident coarse phase without densification for 100,000 iterations. The public configurations differ: the small, default, and MatrixCity configurations use 30,000, 60,000, and 250,000 coarse iterations respectively. The source configuration used in an experiment therefore belongs in every result record. **Evidence: Paper; Public-code behavior** (`train_scaffold.py`, `configs/small.json`, `configs/default.json`, `configs/MatrixCity.json`).

The coarse scaffold is saved as a GraphDECO-style binary PLY and accompanied by skybox-count metadata. **Evidence: Official source** (`Scene.save`, `GaussianModel.save_ply_coarse`).

### 3. Initial binary hierarchy construction

`train.py` invokes the C++ `GaussianHierarchyCreator` after coarse training. The creator:

1. reads a fixed-layout, degree-1-SH binary PLY into `LessRichPoint` records;
2. removes invalid, overly large, or zero-opacity entries;
3. builds a point-based binary k-d tree by splitting the longest bounding-box axis at the median;
4. recursively merges children into renderable parents using opacity- and projected-ellipse-area-weighted attributes;
5. aligns rotations before covariance/scale merging; and
6. serializes the hierarchy as `.dhier`.

This stage creates an initialization, not a paper-quality final asset. Fine hierarchical optimization is part of the method and cannot be skipped for a faithful reproduction. A generic high-SH PLY is not automatically compatible with the fixed `LessRichPoint` reader even though richer structs exist elsewhere in the source. **Evidence: Official source** (`mainHierarchyCreator.cpp`, `Loader`, `PointbasedKdTreeGenerator`, `ClusterMerger`, `RotationAligner`, `Writer`).

### 4. Fine hierarchical optimization

The fine phase loads the hierarchy, moves the full mutable Gaussian-property table and optimizer state to CPU memory, constructs the HSPT metadata, and trains only the camera-selected hierarchy cut. Interior hierarchy nodes can receive gradients because they are renderable Gaussians. **Evidence: Paper; Official source** (`train_hierarchy.py`, `GaussianModel.create_from_hier`, `compact_gaussians`, `build_hierarchical_SPT`).

Hierarchy-aware densification spawns two children below a selected leaf. Low-opacity/dead nodes can be relocated below another selected node. The source supports classic gradient-driven densification and an MCMC option, but its own hyperparameter notes recommend the classic strategy for large scenes and state that the required per-primitive contribution signal is incompatible with the gsplat path. The HSPT is rebuilt after densification rather than stored durably. **Evidence: Official source** (`add_new_gs`, `relocate_gs`, training configuration and hyperparameter notes).

### 5. Upper hierarchy and HSPT construction

The hierarchy is split using a Gaussian-volume threshold (`SPT_root_volume`). Breadth-first traversal retains a relatively small upper hierarchy on the GPU. Large lower subtrees become independent SPTs. Within each SPT, arrays are ordered so that the camera-dependent cut can be found by binary search over parent-derived maximum-distance values. **Evidence: Paper; Official source** (`GaussianModel.build_hierarchical_SPT`).

The public implementation derives a Gaussian minimum-distance threshold as:

```text
sqrt(s1*s2 + s1*s3 + s2*s3) / target_granularity
```

where `s1..s3` are activated Gaussian scales. This source expression is the reproduction oracle when paper notation is ambiguous. **Evidence: Public-code behavior.**

HSPT arrays, SPT membership, distance bounds, and cache state are reconstructed from the serialized hierarchy and configuration. They are not persisted in `.dhier`. Consequently, asset hash alone is insufficient to identify a runtime hierarchy: the source commit and hierarchy-building configuration are also required. **Evidence: Official source.**

### 6. Camera-dependent hierarchy cut

The upper hierarchy is traversed breadth-first. For each node, the public code evaluates a distance/scale LoD condition equivalent to:

```text
min_distance_squared > camera_distance_squared * distance_multiplier
```

`cut_hierarchy_on_condition` descends while the condition requests more detail and emits a node when the condition no longer requests descent. Selected lower-tree roots and their camera distances are passed to the SPT CUDA routine. **Evidence: Official source** (`cut_hierarchy_on_condition`; inline cut logic in training, evaluation, and viewer code).

The helper `GaussianModel.get_SPT_cut` computes the condition and then overwrites it with an all-true mask. It is not the primary train/evaluation/viewer path and must not be copied as the oracle implementation. **Evidence: Public-code behavior.**

### 7. Frustum culling

The upper traversal rejects hierarchy subtrees using sphere/frustum tests. The paper describes a scale-derived proxy bound. The source can instead propagate subtree bounding spheres, but `use_bounding_spheres` defaults to false. Its frustum extraction currently uses left, right, top, and bottom planes; near and far plane extraction is commented out. These choices must be controlled explicitly in fidelity experiments. **Evidence: Paper; Public-code behavior** (`extract_frustum_planes`, `frustum_cull_spheres`, hierarchy configuration).

### 8. GPU SPT cut

For each selected SPT root, the CUDA path binary-searches the SPT's sorted distance arrays, populates candidate global node IDs, filters invalid entries, and returns the selected Gaussian IDs plus per-SPT starts/counts. **Evidence: Official source** (`get_spt_cut_cuda`, `GetSPTCut`, `Switching::getSPTCut`, `runtime_switching.cu::binary_search_SPTs`, `populate_SPT`).

The semantic request/cache unit is an SPT cut at a camera distance, not an individual Gaussian. Physical GPU storage is a set of concatenated Gaussian tensors with SPT boundary and identity metadata. **Evidence: Paper; Official source.**

### 9. Host-resident attributes and GPU working set

During fine training, the full scene properties and Adam state reside in CPU RAM. Selected global IDs index the CPU property table, and the resulting rows are copied synchronously to CUDA (`.cuda(non_blocking=False)`). Persistent GPU data include the skybox, upper-hierarchy/HSPT metadata, and the active/cache tensors. The final render input concatenates selected upper nodes, newly loaded SPT cuts, reused/cached cuts, and the skybox. **Evidence: Paper; Official source.**

Disk backing is described as possible but much slower; the intended practical external memory is CPU RAM. **Evidence: Paper.**

### 10. Caching policies

The paper and public programs expose three distinct policies that must not be conflated:

| Context | Observed policy | Fidelity classification |
|---|---|---|
| Fine training | Reuse SPT cuts within asymmetric distance-ratio tolerances; keep unrequested cuts; reduce toward a target capacity; write attributes and Adam moments back to CPU on eviction; periodic clearing | Closest public implementation of the paper's bounded write-back/LRU-like cache |
| Interactive viewer | Reuse compatible requested SPT cuts retained from the previous working set; discard unrequested cuts; no explicit byte/Gaussian budget, write-back, or periodic flush | Released viewer behavior, not full paper LRU |
| Evaluation script | Recompute selection and transfer the complete selected set for every camera | Cold/no-cache reference despite a cache-related comment |

Training uses reuse tolerances corresponding to approximately `0.5 < cached_distance/current_distance < 1.3`; viewer reuse uses approximately `0.3 < ratio < 1.5`. The training eviction ordering behaves as an approximate ordering-based LRU but has no explicit age counter. **Evidence: Public-code behavior** (cache sections in `train_hierarchy.py`, hierarchy viewer, and `eval_hierarchy.py`).

### 11. Rasterization

The public renderer assembles activated opacity, exponential scales, normalized quaternions, and DC/rest SH coefficients and calls `gsplat.rasterization`. It constructs intrinsics from FoV, adapts the camera matrix, uses `packed=True`, and requests RGB rendering. The wrapper currently ignores some exposed arguments (`anti_aliasing`, `scaling_modifier`, and the selected render-mode variable) and always follows its fixed gsplat path. **Evidence: Public-code behavior** (`gaussian_renderer/__init__.py::render_gsplat`).

The repository README states that the public release uses gsplat and is more memory-efficient than the implementation measured in the paper. Paper numbers and local public-code numbers therefore must be reported as different backends rather than treated as direct replications. **Evidence: Official README; Paper.**

## `.dhier` serialized artifact

The dynamic hierarchy is a native raw binary stream with no magic number, schema version, checksum, or endianness marker. The audited writer/loader use the following order:

| Field | Type and shape |
|---|---|
| Gaussian count | `int32` |
| SH degree | `int32` |
| Positions | `float32[N,3]` |
| Rotations | `float32[N,4]` |
| Log scales | `float32[N,3]` |
| Activated opacities | `float32[N,1]` |
| SH coefficients | `float32[N,(degree+1)^2,3]` |
| Node count | `int32` |
| Hierarchy nodes | `int32[node_count,6]` = depth, parent, child count, first child, next sibling, max-side/index field |

The writer's preorder traversal makes the Gaussian array index the durable global node ID for that exact asset. Runtime HSPT IDs, SPT IDs, cache slots, and concatenated working-set offsets are derived identities and are not stable across configurations or rebuilds. **Evidence: Official source** (`HierarchyWriter::writeDynamic`, `HierarchyLoader::loadDynamic`).

The public loader reads `node_count` and then replaces it with the Gaussian count, effectively assuming one Gaussian per hierarchy node. The initial creator temporarily uses the final integer field to retain an input index for leaves; the runtime loader later zeros that column. A safe importer must validate one-to-one counts and tree structure before trusting the artifact. **Evidence: Public-code behavior.**

The native stream has no byte-order marker. Current project decoding uses
explicit little-endian 32-bit integers/floats under a **PINNED-ASSET PLATFORM
ASSUMPTION**: the accepted assets were produced and consumed on x86-64 Linux.
Cross-endian portability is unsupported rather than implied by the format.

The initial degree-3 creator artifact has two narrowly scoped compatibility
properties that do not apply to a strict final hierarchy:

- `populateDynamicRec` assigns the current flattened-vector size to
  `first_child` before checking whether a node is a leaf. All 443,620 leaves
  therefore contain a positive unused value; the `creator-initial` validation
  profile reports these values and never traverses them.
- Creator input leaves are sigmoid-activated, but `ClusterMerger` computes an
  interior opacity proxy as accumulated weight divided by merged ellipse area.
  In the validated initial artifact, 92,926 interior values exceed one.
  `create_from_hier` subsequently clamps the complete opacity array to
  `[0, 0.99999]` before applying inverse-sigmoid. Strict Run005 validation
  still requires serialized opacity in `[0,1]`.

Official parity is separated into two layers. Layer A compares every raw
serialized field against `load_dynamic_hierarchy` with exact full-array hashes
and sampled bytes. Layer B independently reproduces the transformations in
`GaussianModel.create_from_hier`—SH splitting, opacity clamp/inverse-sigmoid,
and zeroing node column five—plus the exp/normalize/sigmoid render-ready views.
Both layers match exactly for the degree-3 initial hierarchy and degree-1
Run005 final hierarchy.

For SH degree 1, the serialized lower bound is 116 bytes per node/Gaussian plus a 12-byte header: 92 bytes of attributes and 24 bytes of node metadata. Approximate uncompressed sizes are 1.74 GB for 15 million, 6.96 GB for 60 million, and 17.4 GB for 150 million Gaussians. **Evidence: Source-format arithmetic; Project inference.**

## Memory model

The source's compact CPU training representation allocates property, first-moment, and second-moment storage. At SH degree 1 this is at least 276 bytes per Gaussian before node metadata and auxiliary densification/contribution arrays. The paper's measured planning rule is substantially larger: approximately 1 GB of host RAM per million Gaussians, with the hierarchy structure under 10% of total RAM. **Evidence: Official source; Paper.**

For a 60-million-Gaussian scene, the paper reports approximately 680 MB of SPT metadata and 24 MB for the upper hierarchy on GPU. These are in addition to active attributes, rasterizer buffers, cache contents, and framework allocations. **Evidence: Paper.**

The local machine has about 30 GiB system RAM and a 15.9 GiB RTX 5080. It cannot responsibly target the published 60M–150M training regime. The first credible target is a deliberately capped small hierarchy with measured cache and rasterizer headroom. **Evidence: Local environment inventory; Project inference.**

## Source-faithful behavior versus project adaptations

| Element | Classification |
|---|---|
| Binary renderable hierarchy and fine optimization of selected cuts | Paper/source-faithful requirement |
| Public HSPT construction and CUDA cut | Source-faithful requirement |
| Four-plane frustum and default non-propagated bounds | Public-code behavior; expose as explicit compatibility mode |
| gsplat `packed=True` RGB renderer | Public-release backend behavior |
| Viewer previous-working-set reuse | Public viewer behavior |
| Training write-back cache | Paper/training behavior, not viewer behavior |
| Fully resident hierarchy or flat-leaf rendering | Controlled thesis ablation |
| Project-native metrics and deterministic oracle exports | Engineering adaptation required for validation |
| Asynchronous prefetch, byte-budgeted caching, and new replacement policies | Future thesis contribution, not reproduction |

## Known risks and ambiguities

- The released repository has no pretrained `.dhier`, hierarchy checkpoint, or result bundle. Reproduction requires training a hierarchy.
- HSPT metadata and LoD thresholds are reconstructed, so configuration drift can change selection without changing the asset.
- The paper, fine-training cache, viewer reuse, and evaluation transfer behavior differ.
- The viewer hard-codes a base focal length of 1000, while evaluation/training derive it from training cameras. Camera-dependent oracle records must include the exact focal normalization.
- The source CMake hierarchy tools target CUDA architectures 70, 75, and 86; RTX 5080/Blackwell support is not declared.
- Tracked extension binaries target CPython 3.10 and cannot be reused by the project's Python 3.12 environment.
- The raw `.dhier` format has no self-description or corruption checks.
- A leaf-only cut is a useful finest-level baseline, but interior nodes received direct optimization; it is not automatically equivalent to a separately trained flat 3DGS model.
- Aerial views at near-constant altitude weakly exercise LoD. Mixed aerial/street or materially varying camera distance is needed for thesis-valid LoD evaluation.

## Reproduction acceptance criteria

The implementation is not considered source-faithful until it can demonstrate, on the same asset and cameras:

1. identical serialized hierarchy interpretation and stable global node IDs;
2. identical upper-tree decisions, SPT roots, and selected global Gaussian IDs;
3. separately characterized cold, viewer-reuse, and write-back-cache behavior;
4. attribute-level equality before rasterization;
5. gsplat input parity followed by bounded image-level error;
6. explicit peak host RAM, VRAM, selected/resident counts, transfer bytes, and stage timings; and
7. no modification or dependency on the frozen CityGaussian runtime.

## Primary references

- Paper: <https://arxiv.org/abs/2507.01110>
- Official project page: <https://felixwindisch.github.io/ALoDOfGaussians/>
- Official repository: <https://github.com/FelixWindisch/LoDOfGaussians>
- Audited source commit: `73f547341547ae021ff036ad74fdf73a1240b450`
