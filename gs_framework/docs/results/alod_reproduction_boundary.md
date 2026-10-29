# ALoD Faithful Reproduction Boundary

Status: `PASS-ALOD-REPRODUCTION-BOUNDARY`

Audit date: 2026-08-20

Pinned paper: *A LoD of Gaussians: Out-of-Core Training and Rendering for
Seamless Ultra-Large Scene Reconstruction*, arXiv:2507.01110v4

Pinned official source: `/home/luis/tese/LoDOfGaussians`, commit
`73f547341547ae021ff036ad74fdf73a1240b450`

Canonical project: `/home/luis/tese/gs_framework`

## 1. Decision and primary answer

The closest publicly reproducible equivalent of the released ALoD rendering
evaluation is not one program called "Render Full." It is a pair of official
runtime references:

1. `eval_hierarchy.py` is the authoritative source-faithful quality/evaluation
   path. It rebuilds the HSPT from the `.dhier`, selects an HSPT LoD cut with
   optional four-plane frustum rejection, keeps the full render-property table
   in CPU RAM, and transfers the complete selected non-skybox set for every
   camera. It has no useful persistent inter-camera Gaussian cache.
2. `hierarchy_viewer.py` is the released temporal-reuse path. It reuses
   compatible SPT cuts from the immediately previous working set, but it does
   not implement a bounded retained/LRU rendering cache.

The paper's Table 3 `Render Full` configuration combines HSPT LoD, early
frustum culling, external-memory property streaming, and a bounded rendering
cache. No released rendering program at the pinned commit implements that
combination exactly. The only released bounded retained/write-back cache is in
`train_hierarchy.py`, where mutable properties and Adam state are managed for
fine training. Therefore:

```text
Render Full -> PAPER-SOURCE-DIVERGENCE
```

The current project configuration closest to the official evaluator is
`ALOD-HOST-NOCACHE`. It matches the released evaluator's selection and
host/no-cache property-residency semantics where applicable, but uses project
camera, instrumentation, assembly, and renderer integration. It is a
**SOURCE-FAITHFUL PROJECT REPRODUCTION**, not an untouched official run.

The current 525k and 160k row caches are not substitutes for paper `Render
Full`. They are **PROJECT-SPECIFIC EXTENSIONS**.

The next formal task is **B: construct and train a mixed-scale ALoD
reproduction asset first**. Run005 remains the correctness oracle, but its
aerial-only, weak-LoD workload cannot strongly test the paper's cache,
multi-scale LoD, or large-hierarchy trends.

## 2. Evidence and vocabulary

Primary evidence used in this audit:

- arXiv v4 paper, including Sections 3-6, Table 3, Figures 3-5, and
  Supplementary Sections A.1-A.7;
- the clean pinned official checkout and its `eval_hierarchy.py`,
  `hierarchy_viewer.py`, `train_hierarchy.py`, `scene/gaussian_model.py`,
  `gaussian_renderer/__init__.py`, `utils/view_graph_utils.py`, configuration
  files, documentation, and Gaussian-hierarchy CUDA/C++ sources;
- existing project specifications, validation memos, configurations, runtime
  source, and preserved benchmark records.

Useful primary links:

- paper: <https://arxiv.org/abs/2507.01110v4>
- official source: <https://github.com/FelixWindisch/LoDOfGaussians>
- public MatrixCity source dataset: <https://github.com/city-super/MatrixCity>

This memo uses four top-level labels:

| Label | Meaning in this project |
|---|---|
| **OFFICIAL/PAPER ALoD** | A behavior stated by the paper or directly emitted by the pinned authors' program. Paper and source are identified separately when they disagree. |
| **SOURCE-FAITHFUL PROJECT REPRODUCTION** | A project implementation validated against the pinned released behavior on identical hierarchy/camera inputs. Backend and measurement adaptations must still be disclosed. |
| **SOURCE-INSPIRED ADAPTATION** | A controlled implementation that borrows a released idea but changes an execution semantic such as residency, reuse unit, tolerance, or cache behavior. |
| **PROJECT-SPECIFIC EXTENSION** | A new project policy absent from the paper and pinned source. It must be evaluated as a thesis extension, never as official ALoD. |

The component classification requested for existing work uses the equivalent
short labels:

```text
A = faithful reproduction
B = reproduction-support / validation infrastructure
C = source-inspired adaptation
D = project-specific extension
```

"Faithful" is scoped. A component can reproduce official selection exactly
without making a fully resident frame runtime a faithful out-of-core renderer.

## 3. Paper method relevant to rendering

### 3.1 Representation and HSPT

ALoD is one global binary hierarchy of renderable Gaussians. Interior nodes
approximate descendants; a view-dependent proper cut selects coarse nodes far
from the camera and fine nodes nearby. It is not a hierarchy over independently
trained spatial chunks.

The paper replaces a full per-frame hierarchy traversal with a Hierarchical
Sequential Point Tree (HSPT):

1. a Gaussian-volume threshold splits the hierarchy into a relatively small
   upper hierarchy and lower subtrees;
2. sufficiently large lower subtrees become Sequential Point Trees (SPTs);
3. the upper hierarchy is traversed with BFS, including early subtree frustum
   rejection; and
4. selected SPTs are cut in parallel using camera-to-SPT-root distance and
   ordered minimum/maximum distance metadata.

HSPT means **Hierarchical Sequential Point Tree**, not spatial partition tree.
An HSPT cut and a full-hierarchy BFS cut are alternative ways to compute the
same intended view-dependent hierarchy cut; BFS is not a finest-detail or
all-Gaussian rendering mode.

### 3.2 External-memory representation

During paper fine training, the complete Gaussian properties and optimizer
state remain in CPU RAM. GPU-resident state includes:

- the skybox;
- the compact HSPT/SPT cut metadata;
- the upper hierarchy;
- properties for the current selected set; and
- cached SPT-cut properties retained for reuse.

Selected global Gaussian rows are gathered from the host property table and
copied to the GPU. The paper treats CPU RAM as the practical external memory;
disk backing is possible but reported as about an order of magnitude slower.
The full CPU property table remains canonical while the GPU cache holds copies.

The paper reports, for a 60M-Gaussian hierarchy, approximately 680 MB of SPT
metadata and 24 MB of upper hierarchy on the GPU. Active and cached Gaussian
properties, rasterization/optimization intermediates, images, and framework
allocations are additional costs. Its empirical planning statement is roughly
1 GB of host RAM per million Gaussians.

### 3.3 Paper cache semantics

The paper caches SPT cuts, not independent Gaussian decisions. A cached entry
is reusable only when the current distance to the same SPT root is within an
allowed ratio of its cached cut distance. Reuse may therefore render a nearby
compatible LoD rather than recomputing and loading an exactly new cut.

For training, newly loaded SPTs enter a GPU cache. A bounded, LRU-described
write-back policy evicts retained entries to CPU RAM when a memory threshold is
exceeded, and the cache is periodically flushed to avoid training stagnation.
The paper also schedules spatially nearby training views using a camera k-NN
graph to increase cache locality, with periodic random exploration.

Supplementary Figure 15 describes rendering cache capacity in Gaussians: up to
`cache_size` Gaussians from previous frames may be reused. Performance rises
until capacity exceeds the usual frame working set. The paper's reported
experiments state a 15M-Gaussian cache, but the exact Table 3 render-cache code,
distance tolerances, and timing harness are not exposed as a released rendering
mode.

### 3.4 Frustum culling semantics

Early frustum rejection occurs during upper-hierarchy BFS, before selected
lower SPT properties are loaded from RAM. The paper tests a sphere around a
Gaussian/root, using three times maximal scale as a practical conservative
proxy for subtree extent. This early cull reduces HSPT work and host-to-device
loading. It is distinct from the rasterizer's later implicit projection or
tile rejection.

Disabling ALoD frustum culling must therefore leave rasterizer culling enabled.
The ablation asks whether early hierarchy rejection is useful, not whether
off-screen Gaussians must be rasterized into pixels.

### 3.5 Paper view scheduling

The paper describes a k-nearest-camera graph. Successive training views are
sampled locally with probability inversely related to camera distance, and a
random view is injected every 128 iterations. This affects training cache
locality; it is not part of test-image quality evaluation and should not be
silently imposed on a recorded rendering trajectory.

### 3.6 Paper rendering evaluation and ablations

Table 3 reports average frame times over recorded camera paths. The intended
controlled semantics are:

| Paper row | What changes | What remains fixed | Cache/property interpretation |
|---|---|---|---|
| `Render Full` | Nothing is ablated: HSPT cut, early frustum culling, and render cache are enabled. | Trained hierarchy, LoD threshold, camera path, resolution, rasterizer, and timing boundary. | Full properties remain host-resident; a bounded GPU cache retains reusable SPT-cut properties. Exact released render implementation is missing. |
| `Render w/o Cache` | Disable all inter-frame Gaussian/SPT property reuse. | HSPT, frustum, LoD, camera path, renderer, and timing boundary. | The complete selected set is loaded from RAM for each frame. This is conceptually closest to released `eval_hierarchy.py`. |
| `Render w/o Frustum Culling` | Disable early upper-hierarchy subtree rejection. | Render cache, HSPT, LoD rule, path, renderer, and timing boundary. | More SPTs/properties can be requested from host; rasterizer projection culling still occurs. |
| `Render Cut (HSPT)` | Measure HSPT visible-set/cut computation. | Hierarchy, camera, LoD rule, and intended frustum behavior. | Property transfer, cache management, and rasterization are outside the cut-only boundary. |
| `Render Cut (BFS)` | Replace HSPT cut computation with a full-hierarchy BFS. | Same intended hierarchy cut and camera conditions. | Full-hierarchy positions/scales must be GPU-accessible; this caused paper OOM on MC-small-city+. It is not full-detail rendering. |

The paper reports trends, not universal monotonicity. Frustum overhead can
outweigh its savings on a smaller/easier view sequence: the 38M Campus result
has `Render w/o Frustum Culling` faster than `Render Full`, while larger scenes
benefit. A faithful local target is the same methodological effect and trend
under comparable pressure, not a requirement that every row improve.

## 4. Released runtime modes at the pinned commit

### 4.1 Official evaluation: cold/no persistent cache

Authoritative path:

```text
eval_hierarchy.py
  -> GaussianModel.create_from_hier through Scene
  -> compact_gaussians(storage_device="cpu", training=False)
  -> build_hierarchical_SPT(...)
  -> four-plane upper traversal and optional frustum rejection
  -> get_spt_cut_cuda(...)
  -> host property gather for upper nodes plus SPT output
  -> complete selected-row CUDA transfer
  -> render_gsplat(...)
  -> PSNR / SSIM / VGG-LPIPS and image output
```

Persistent GPU Gaussian properties are the 100,000-row skybox prefix. The
non-skybox `means3D`, scales, rotations, opacity, and SH tensors are rebuilt
from the CPU property table for each camera. `load_from_disk_indices` is a
misleading variable name in this path: normal `storage_device="cpu"` means the
properties are loaded from CPU RAM, not secondary storage.

The evaluator honors `use_frustum_culling`. Its frustum plane extraction uses
left, right, bottom, and top planes only; near/far planes are commented out.
With default `use_bounding_spheres=false`, bounds are three times activated
maximum scale of the upper/root Gaussian, matching the released approximation
rather than a propagated subtree sphere.

The evaluator has no frame-time benchmark protocol and no persistent render
cache. It is the clean official source for fixed-camera selection and quality,
and the closest released implementation of `Render w/o Cache`.

### 4.2 Official viewer: previous-compatible-SPT reuse

The viewer builds the same CPU property table and GPU HSPT metadata, then
stores the previous frame's SPT IDs, cut distances, boundaries, global IDs, and
GPU tensors. An SPT cut is reused when it is requested again and:

```text
0.3 < previous_distance / current_distance < 1.5
```

Unrequested previous SPTs are discarded. Newly requested or incompatible SPTs
are cut and loaded. There is no configurable Gaussian/byte capacity, retained
multi-frame cache, LRU ordering, write-back, or periodic flush. The exposed
`reuse_SPT_tolerance` viewer option is not used by the hard-coded comparison.

Additional released-viewer details prevent treating it as a paper benchmark
harness:

- base focal length is hard-coded to 1000 rather than derived from a training
  camera;
- replay overwrites resolution to 1100 x 900 and adjusts FoV;
- the wall-clock frame interval is not explicitly CUDA-synchronized before it
  stops; and
- the pinned CLI call supplies arguments positionally in a way that appears to
  misroute replay/ID into the `checkpoint` and `replay` parameters.

These are source facts to reproduce or minimally correct with explicit
non-algorithmic instrumentation; they are not permission to relabel the project
exact-row reuse as official viewer reuse.

### 4.3 Official fine training: bounded retained/write-back cache

Fine training is the released path closest to the paper cache:

- full raw Gaussian properties and Adam first/second moments live in CPU RAM;
- skybox properties remain GPU-resident;
- the upper hierarchy and SPT metadata remain GPU-resident;
- requested SPT cuts may reuse cached cuts when approximately
  `0.5 < cached_distance/current_distance < 1.3`;
- compatible requested cuts are reused;
- unrequested previous cuts are retained as cache candidates;
- missing SPT cuts and selected upper nodes are loaded from host;
- the current render/optimization set is assembled from skybox, new loads,
  compatible reuse, and retained cache rows;
- when the complete GPU row list exceeds `cache_size`, old retained SPTs are
  removed toward `cache_size_after_reduction`;
- evicted mutable properties and Adam moments are written back to CPU; and
- every `clear_cache_interval` iterations a full retained-cache clear is
  attempted.

Capacity is counted in Gaussian rows, not bytes. The trigger counts the entire
`gaussian_indices` list, including skybox and the current working set, rather
than only inactive retained cache rows. Eviction uses the ordered tail of
retained SPTs as an approximate age/LRU order; there is no explicit per-entry
age counter.

The source and paper view schedulers differ. The source resets to a random view
every 100 iterations, not 128. `construct_distance_graph` also overwrites its
`k` argument with 100, and its transition uses `1/(distance+20)`. The pinned
MatrixCity config disables `graph_view_select`, while `default.json` enables
it. These are configuration/source divergences, not paper constants.

The paper reports a 15M cache for its experiments. The pinned `MatrixCity.json`
uses `cache_size=25M` and `cache_size_after_reduction=20M`. Consequently a
paper-comparison result must record the chosen budget and cannot cite the
released JSON as proof of the paper's exact rendering capacity.

### 4.4 Rasterizer boundary

All three released programs call `render_gsplat`, which converts the projective
camera and attributes and invokes `gsplat.rasterization` with `packed=True` and
RGB rendering. The wrapper ignores some exposed options and fixes several
rasterization choices. The official README explicitly warns that this released
gsplat backend is more memory-efficient than the implementation measured in the
paper. Local VRAM and timing results are therefore released-code reproduction
results, not direct paper-backend replications.

## 5. Paper-to-source mapping

| Paper concept | Paper semantics | Pinned released path | Correspondence |
|---|---|---|---|
| Global renderable hierarchy | One non-partitioned binary hierarchy; interior nodes are renderable approximations | `.dhier` loader and `GaussianModel` | Exact method concept; local Run005 is correctness-patched and capped |
| HSPT construction | Volume-cut upper hierarchy plus lower SPTs; compact metadata on GPU | `GaussianModel.build_hierarchical_SPT` | Public implementation exposed |
| HSPT cut | Upper BFS then parallel SPT cuts | inline train/eval/viewer selection plus `get_spt_cut_cuda` | Public implementation exposed |
| Full BFS cut | Traverse full Gaussian hierarchy with same LoD intent | generic hierarchy traversal building blocks exist, but no released render-ablation command | Partial; paper ablation not publicly packaged |
| Frustum culling | Early upper-BFS subtree rejection before property loading | train/eval/viewer four-plane sphere test | Partial: source omits near/far and defaults to scale proxy; explicit compatibility mode required |
| External-memory rendering | Complete properties in CPU RAM; selected properties streamed to GPU | eval/viewer `compact_gaussians(cpu)` plus selected gather | Exposed |
| Temporal coherence | Reuse compatible recent SPT cuts | viewer previous-working-set reuse; trainer retained cache | Two partial paths with different semantics |
| Bounded rendering cache | Up to `cache_size` previous-frame Gaussians; paper full rendering uses cache | no released render path | **PAPER-SOURCE-DIVERGENCE** |
| LRU/write-back | Bounded GPU cache, old SPTs returned to host | fine-training loop | Closest source path, but mutable training cache rather than released render cache |
| `Render Full` | HSPT + frustum + bounded render cache | none | **PAPER-SOURCE-DIVERGENCE** |
| `Render w/o Cache` | Same renderer/path without temporal reuse | `eval_hierarchy.py` conceptually | Closest released match; evaluator lacks paper timing harness |
| `Render w/o Frustum Culling` | Cache stays enabled; early frustum disabled | config toggle exists, but no bounded render-cache path | Partial; exact paper combination not exposed |
| HSPT vs BFS cut timing | Same cut intent; only data structure/traversal changes | HSPT exposed; BFS ablation command absent | Partial |
| View scheduling | k-NN-local training views plus random exploration every 128 | source random walk plus random reset every 100; config-dependent | Partial/source divergence |
| GPU metadata layout | Slim HSPT/SPT metadata and upper hierarchy resident | HSPT arrays and upper tensors on CUDA | Exposed |
| Paper memory/FPS numbers | Authors' paper backend, scenes, paths, and hardware | release uses gsplat and no assets/paths from Table 3 | Not directly reproducible from source alone |

Central conclusion: the public release can reproduce official cold evaluation,
official previous-working-set viewer behavior, and official fine-training cache
behavior separately. It cannot directly reproduce the paper's combined
`Render Full` row without a clearly labeled render-only reconstruction of the
cache path.

## 6. Run005 boundary and topology correction

Run005 is the accepted local hierarchy:

```text
1,099,999 serialized Gaussian/node rows
100,000-row skybox prefix
SH degree 1
SHA-256 6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c
```

Its label is:

```text
correctness-patched local reproduction
```

It is not bit-identical output of the untouched pinned source and is not a
paper-scale reproduction.

### 6.1 Official bug corrected

During `relocate_gs`, the official implementation promotes a surviving sibling
into the dead node's old parent and respawns the dead/sibling rows under a new
leaf. It copies the promoted node's forward `child_count` and `first_child`, but
then updates only two immediate children. This has two correctness failures:

1. when the promoted sibling is an interior subtree, deeper descendants retain
   stale reverse-parent and depth metadata; and
2. when the promoted sibling is a leaf, `first_child == 0`, so the direct-child
   update can write parent/depth metadata into serialized row 0 of the skybox.

The unpatched Run004 artifact demonstrated the failure: 4,355 depth
inconsistencies, a reverse-parent mismatch, and corrupted skybox metadata.

### 6.2 Repair semantics

`ALOD-RELOCATE-TOPOLOGY-FIX.patch` removes the unsafe two-child metadata update
and recursively rebuilds redundant parent/depth metadata below only the forward
roots affected by relocation. It validates ranges, sibling termination, and
cycles. It deliberately preserves:

- which nodes are declared dead;
- which alive nodes are sampled for relocation;
- Gaussian property updates;
- promoted forward `child_count`, `first_child`, and sibling chains;
- respawn placement and forward topology;
- losses, optimization, densification thresholds, cache behavior, selection,
  and renderer behavior.

The patch changes invalid redundant metadata to agree with the already chosen
forward topology. It is classified as reproduction-support/correctness
infrastructure, not an original rendering method.

## 7. Current project component classification

| Project component | Class | Faithfulness statement and reason |
|---|---:|---|
| Run005 training | **A, qualified** | Correctness-patched, resource-adapted reproduction of official coarse/creator/fine training. It establishes mechanism correctness, not untouched-source or paper-scale output. |
| Relocation-topology correctness patch | **B** | Necessary invariant repair preserving intended forward method semantics; not a paper feature or thesis rendering contribution. |
| `alod_dhier` parser/inspector | **B** | Independent read-only format, topology, numeric, and official-loader parity infrastructure. |
| `alod_dhier` fixtures and inspector gate | **B** | Validation evidence only. |
| Official HSPT/selection oracle | **B** | Instrumented output from pinned `eval_hierarchy.py`; establishes source truth but is not a renderer policy. |
| `alod_reference` CPU HSPT/frustum/selection | **B** | Independent reference model used to catch shared implementation errors. |
| Production HSPT builder and selector | **A at the selection boundary** | Exact ordered parity with official HSPT arrays, upper decisions, per-SPT cuts, selected IDs, and raster order on Run005. |
| `AlodResidentScene` as a complete runtime | **C** | Couples faithful selection to fully resident properties, removing ALoD external-memory execution. |
| `ALOD-HIER-FINEST-RESIDENT` | **C** | Project controlled finest-cut reference, not an official ALoD rendering configuration. |
| `ALOD-HIER-LOD-RESIDENT` | **C** | Faithful LoD selection under project full residency; intentionally disables frustum and streaming. |
| `ALOD-HIER-LOD-FRUSTUM` | **C overall** | Faithful selection/frustum component, but full-resident execution is not official/paper out-of-core rendering. |
| Resident benchmark campaign | **B evaluating C** | Reproducible benchmark infrastructure for resident controlled adaptations. |
| `ALOD-HOST-NOCACHE` | **A, qualified** | Closest project reproduction of official evaluator residency: host property table, resident skybox/HSPT state, synchronous selected-row transfer every frame, no reuse. Project renderer/timing remains adapted. |
| `ALOD-HOST-PREVIOUS-REUSE` | **C** | Exact-row intersection with the immediately previous selected set. Official viewer reuses distance-compatible SPT cuts, so identity/unit/LoD semantics differ. |
| `ALOD-HOST-BOUNDED-CACHE` (525k) | **D** | Read-only per-row deterministic LRU and trajectory-derived capacity. Official trainer caches SPT cuts with distance tolerances, mutable properties/moments, write-back, reduction target, and flush. |
| Host residency correctness/benchmark harness | **B evaluating A/C/D** | Measurement and validation infrastructure; experiment claims inherit the class of each mode. |
| Conservative CPU pre-materialization filter | **D** | New post-semantic-selection mean/max-scale screen/depth filter absent from paper/source. |
| Fast PyTorch CUDA pre-materialization filter | **D** | Project acceleration of the new filter; exactness against the CPU project reference does not make it official ALoD. |
| `ALOD-HOST-BOUNDED-CACHE-REDUCED` (160k) | **D** | Cache capacity chosen from the project materialized-row union; no paper/source capacity correspondence. |
| CPU-filter and fast-filter 525k configurations | **D** | Combine two project extensions even when the historical cache policy is preserved. |
| Materialization-aware benchmark/campaign code | **B evaluating D** | The harness is infrastructure; the policies and resulting performance are extension results. |
| Materialization-aware benchmark results | **D characterization** | Evidence for project-specific systems behavior, not an ALoD paper ablation. |

This classification protects the existing work: none of the C/D components is
removed, renamed, or weakened. They move to the thesis extension layer.

## 8. Frozen faithful-reproduction experiment ladder

The new namespace is conceptual until the corresponding configurations are
implemented and validated. Historical result IDs remain immutable.

### 8.1 Reproduction namespace

| Proposed ID | Asset and hierarchy | Selection/frustum | Property residency and cache | Camera/renderer/timing/memory boundary | Status |
|---|---|---|---|---|---|
| `ALOD-REPRO-OFFICIAL-EVAL-COLD` | Run005 initially; later formal mixed-scale `.dhier`; HSPT rebuilt with exact source config | Official HSPT LoD; released four-plane frustum on | Full render properties in CPU RAM; skybox and HSPT metadata GPU-resident; no persistent non-skybox cache | Official fixed test cameras; released `render_gsplat`; quality boundary includes render only per camera, plus separately instrumented completed selection/transfer/render; record RSS, allocator, and whole-device VRAM | Existing official oracle and project host-no-cache semantics are available; unified formal config missing |
| `ALOD-REPRO-OFFICIAL-EVAL-NOFRUSTUM` | Same | Identical LoD; only early four-plane frustum off | Same no-cache host behavior | Same cameras/backend/timing/memory | Missing formal experiment |
| `ALOD-REPRO-OFFICIAL-VIEWER-PREVIOUS` | Same `.dhier` and exact viewer HSPT config | Official viewer HSPT/frustum | Reuse only requested SPT cuts satisfying official viewer distance ratio; discard unrequested cuts; no capacity/LRU | Frozen recorded path; exact viewer focal/resolution behavior or separately disclosed non-algorithmic correction; CUDA-completed timing required in addition to source wall clock | Missing; current exact-row reuse is not this mode |
| `ALOD-REPRO-PAPER-RENDER-FULL-APPROX` | Formal mixed-scale `.dhier` | HSPT LoD and early frustum on | CPU canonical properties; bounded retained SPT-cut GPU cache reconstructed from paper plus fine-training source; capacity and tolerances explicit | Frozen continuous path; released gsplat project-compatible renderer; selection, transfer, cache, render, end-to-end, H2D, CPU RAM, and GPU memory boundaries | Missing; mandatory `PAPER-SOURCE-DIVERGENCE` label |
| `ALOD-REPRO-PAPER-RENDER-NOCACHE` | Same as paper-full approximation | Same HSPT/frustum | Disable all cross-frame reuse only | Exactly same path/backend/timing/memory protocol as paper-full approximation | Missing; do not substitute a different camera/test loop |
| `ALOD-REPRO-PAPER-RENDER-NOFRUSTUM` | Same | Disable only early frustum; same HSPT LoD | Keep the same bounded cache as paper-full approximation | Same path/backend/timing/memory | Missing |
| `ALOD-REPRO-CUT-HSPT` | Same hierarchy | HSPT visible-set computation; frustum fixed | No property transfer/cache/render in timed boundary | Same cameras; CUDA-completed cut-only timing and metadata VRAM | HSPT implementation exists; formal isolation missing |
| `ALOD-REPRO-CUT-BFS` | Same hierarchy | Full hierarchy BFS with same LoD/frustum intent | Full positions/scales required on GPU as in paper comparison; no cache/render in cut timing | Same cameras and cut timing protocol; report OOM honestly | Missing public-ablation reconstruction |
| `ALOD-REPRO-QUALITY` | Same trained hierarchy | Official evaluator selection | Cold/no-cache, matching quality evaluator | Frozen held-out GT cameras; PSNR, SSIM, VGG-LPIPS; renderer/backend recorded | Run005 lacks a strong formal quality regime |

### 8.2 Extension namespace

New experiments using the existing project policies should use `ALOD-EXT-*`,
for example:

```text
ALOD-EXT-ROW-LRU-525K
ALOD-EXT-PREMATERIALIZATION-CPU-525K
ALOD-EXT-PREMATERIALIZATION-CUDA-525K
ALOD-EXT-PREMATERIALIZATION-CUDA-160K
```

These are mapping labels, not destructive renames. Published artifacts and
historical IDs stay unchanged:

| Historical ID | Conceptual layer |
|---|---|
| `ALOD-HIER-FINEST-RESIDENT` | source-inspired controlled parent |
| `ALOD-HIER-LOD-RESIDENT` | source-inspired resident LoD |
| `ALOD-HIER-LOD-FRUSTUM` | source-inspired resident runtime with faithful selector |
| `ALOD-HOST-NOCACHE` | reproduction precursor for `ALOD-REPRO-OFFICIAL-EVAL-COLD` |
| `ALOD-HOST-PREVIOUS-REUSE` | extension/source-inspired exact-row reuse; not official viewer mode |
| `ALOD-HOST-BOUNDED-CACHE` | `ALOD-EXT-ROW-LRU-525K` |
| `ALOD-HOST-BOUNDED-CACHE-PREMATERIALIZATION` | `ALOD-EXT-PREMATERIALIZATION-CPU-525K` |
| `ALOD-HOST-BOUNDED-CACHE-FAST-PREMATERIALIZATION` | `ALOD-EXT-PREMATERIALIZATION-CUDA-525K` |
| `ALOD-HOST-PREVIOUS-REUSE-FAST-PREMATERIALIZATION` | combined source-inspired row reuse plus project filter |
| `ALOD-HOST-BOUNDED-CACHE-REDUCED` | `ALOD-EXT-PREMATERIALIZATION-CUDA-160K` |

### 8.3 Prohibited substitutions

The following cannot appear in an `ALOD-REPRO-*` result unless a new primary
source establishes equivalent official behavior:

- conservative CPU pre-materialization filtering;
- fast PyTorch CUDA pre-materialization filtering;
- the 525k per-row project cache;
- the 160k per-row project cache; or
- materialized-union-derived capacity selection.

## 9. Current Run005 dataset boundary

Run005 is a deliberately small, capped, aerial-only reproduction asset with
approximately 1.1M serialized rows and weak LoD variation. Its ten frozen
cameras are an exact selection oracle, not a paper-like navigation workload.

It can test strongly:

- `.dhier` format and topology correctness;
- official loader transformations and property identity;
- HSPT construction and ordered SPT contents;
- upper-tree, frustum, SPT-cut, and final selected-ID correctness;
- host versus resident property equivalence;
- deterministic cache mapping, transfer accounting, and eviction correctness;
- renderer integration and pre-raster property parity; and
- instrumentation, timing-field, and memory-field correctness.

It can test only weakly or not at all:

- multi-scale aerial-to-street LoD adaptation;
- deep hierarchy transitions across a wide camera-distance range;
- paper-like cache pressure and temporal-coherence benefit;
- HSPT versus BFS scaling at tens or hundreds of millions of nodes;
- large-scene host/GPU capacity behavior;
- paper-level rendering throughput;
- paper MatrixCity PSNR/SSIM/LPIPS;
- paper training duration or convergence; and
- seams/quality comparisons against CityGaussian on a matched formal asset.

Existing Run005 benchmarks remain valid engineering characterizations. Their
negative or saturated trends cannot be generalized into claims against ALoD:
the bounded 525k cache holds almost the trajectory union, and selected LoD rows
change little across the aerial path.

## 10. Next formal ALoD reproduction asset

### 10.1 Exact paper dataset availability

The paper's `MC-small-city+` is not reconstructible exactly from the pinned
public repository alone. The supplement states 33,006 public street views,
7,672 public aerial views, and 533 newly generated high-altitude views, which
sum to 41,211 training views, plus 4,228 test views. The main result table labels
the scene "42.2k images"; the paper does not reconcile that count with the
supplementary component counts. It merges the public street and aerial sparse
point clouds and randomly downsamples them by five, without using ground-truth
depths.

The public ALoD repository supplies a generic `MatrixCity.json`, not the exact
camera identity lists, conversion manifest, random point-cloud sample seed,
recorded rendering paths, trained `.dhier`, or the 533 added high-altitude
images/poses. No public download for the augmented `+` dataset was found during
this audit. Therefore a local asset must not be called `MC-small-city+` unless
those missing author inputs are released and verified.

### 10.2 Defensible public-data reproduction

Freeze a new asset provisionally described as:

```text
MatrixCity Small City public mixed-scale ALoD reproduction
```

Required construction:

1. acquire the public MatrixCity Small City aerial and street RGB images,
   poses, and sparse point clouds under their original terms;
2. choose one contiguous geographic region, not an image stride or a
   cherry-picked camera list;
3. include all public street cameras in that region plus aerial cameras at all
   available heights whose footprints cover the region;
4. preserve a held-out public test set before any conversion or training;
5. convert poses/intrinsics to standard COLMAP with a versioned, reversible
   camera-identity manifest and numeric round-trip validation;
6. merge the public aerial and street sparse point clouds and reproduce the
   paper's factor-five random downsample with a frozen seed;
7. do not use MatrixCity ground-truth depths or dense geometry;
8. record camera-distance, height, focal-length, footprint, and nearest-neighbor
   distributions proving that both modalities and wide scale variation remain;
9. use the official coarse/creator/fine pipeline plus the already disclosed
   relocation correction; and
10. retain Run005 separately as the small correctness oracle.

The paper's 15.1k-image `MC-smaller-city+` is a useful scale reference, but its
exact one-third spatial crop is not released. A local contiguous mixed-scale
crop must use a new name and publish its geographic/camera selection rule.

### 10.3 Current hardware envelope

The local host has about 30 GiB RAM and the GPU has 16 GB VRAM. Paper-scale
60M-150M training is not defensible. Using the paper's empirical host rule,
30M alone would consume about the whole host before images, Python, hierarchy,
data-loader workers, and OS/WSL overhead.

The largest defensible initial envelope is a **measured, provisional ceiling
of about 20M final Gaussian/node rows**, not a target chosen to produce a
desired result. Twenty million corresponds to roughly 20 GB under the paper's
empirical rule and leaves limited but necessary host headroom. Acceptance must
still be governed by preflight and runtime measurements:

- keep peak process-group RSS below a frozen safe fraction of physical RAM;
- do not depend on sustained swap;
- load images from disk as the paper does;
- measure HSPT metadata and active/cache peaks before committing to the cap;
- lower the cap if the measured safe boundary is crossed; and
- never describe the result as paper-scale capacity.

Dataset semantics come first: a smaller but genuinely mixed-scale contiguous
scene is preferable to a larger aerial-only scene that cannot exercise ALoD.

## 11. Paper-result reproducibility on current hardware/data

| Result dimension | Classification | Reason |
|---|---|---|
| Exact paper render frame time/FPS | **NOT CURRENTLY REPRODUCIBLE** | Different GPU, missing paper paths/assets, released gsplat/backend difference, and missing `Render Full` code. |
| Cache-versus-no-cache methodological trend | **TREND REPRODUCIBLE** | Requires the formal mixed-scale asset and clearly labeled bounded SPT render-cache approximation; Run005 saturates too easily. |
| Frustum-versus-no-frustum trend | **TREND REPRODUCIBLE** | Source toggle exists, but benefit depends on scene/view pressure and must be compared under one cache/path protocol. |
| HSPT-versus-BFS cut trend | **TREND REPRODUCIBLE** | HSPT is public; a controlled BFS ablation must be reconstructed. Local bounded scale may not reproduce paper OOM. |
| Released evaluator selected IDs/properties | **DIRECTLY REPRODUCIBLE** | Already exact on Run005 through official oracle, CPU reference, and production selection gates. |
| Released evaluator cold host transfer | **DIRECTLY REPRODUCIBLE** | `ALOD-HOST-NOCACHE` reproduces the source residency behavior where applicable. |
| Official viewer temporal reuse | **NOT CURRENTLY REPRODUCIBLE** | Current project reuse is exact-row, not distance-compatible SPT-cut reuse; viewer replay/timing wiring also needs controlled handling. |
| Paper render-cache implementation | **NOT CURRENTLY REPRODUCIBLE** | Not released. Only an explicit paper/source approximation is possible. |
| VRAM absolute values | **NOT CURRENTLY REPRODUCIBLE** | Hardware and rasterizer differ; paper README itself warns release memory differs. |
| VRAM scaling trend | **TREND REPRODUCIBLE** | Can compare identical local asset/path configurations with consistent allocator and whole-device measurement. |
| CPU RAM absolute values | **NOT CURRENTLY REPRODUCIBLE** | Asset scale and runtime differ. |
| CPU RAM per-row/scaling trend | **TREND REPRODUCIBLE** | Host property, hierarchy, cache metadata, and process RSS can be measured separately. |
| H2D traffic and loaded-row counts | **DIRECTLY REPRODUCIBLE** | Exact bytes/rows can be counted for local modes; paper absolute counts require paper assets/paths. |
| PSNR/SSIM/LPIPS on Run005 | **NOT CURRENTLY REPRODUCIBLE** | Current asset is a contaminated/capped aerial oracle and not the paper evaluation regime. |
| PSNR/SSIM/LPIPS on a new public mixed-scale asset | **TREND REPRODUCIBLE** | It would be a valid local quality evaluation, but not exact `MC-small-city+` without missing high views and manifests. |
| Exact paper training time | **NOT CURRENTLY REPRODUCIBLE** | Different hardware, missing exact data/config, and infeasible scale. |
| Large-scale 60M-150M capacity | **NOT CURRENTLY REPRODUCIBLE** | Approximate 30 GiB host RAM is below the paper regime. |

No future report should promise absolute FPS parity across RTX 5080, RTX 3090,
H200, different gsplat versions, or the paper's unreleased renderer.

## 12. CityGaussian comparison plan

The thesis should compare two faithful reproduction branches before comparing
project extensions:

```text
CityGaussian source-faithful reproduction
vs
ALoD source-faithful reproduction
```

### 12.1 Dimensions that can be shared

- physical GPU/CPU and software inventory;
- deterministic camera identities when both trained representations cover the
  same coordinate system and held-out images;
- output resolution, intrinsics, background, near/far conventions, and image
  preprocessing;
- warmup, repeat count, cold/continuous sequence definitions, synchronization,
  and timing boundaries;
- CPU RSS, PyTorch allocated/reserved, whole-device VRAM, persistent property
  bytes, transient bytes, H2D bytes, and selected/rendered counts;
- PSNR, SSIM, LPIPS implementation and GT masks;
- project gsplat backend for a controlled selected-set comparison where both
  methods can supply equivalent activated tensors; and
- author-native/released backends as separate reproduction results.

### 12.2 Dimensions that cannot be forced equal

- Gaussian count or hierarchy/node count;
- trained parameter identity, topology, LoD representation, pruning, or VQ;
- cache unit and residency policy;
- training iterations when the methods define different staged optimization;
- source-native rasterizer behavior;
- training cameras or GT cameras if a method's faithful dataset construction
  is incompatible; and
- quality numbers from different assets or coordinate systems.

The existing CityGaussian branch is aerial MatrixCity, while the next useful
ALoD asset must be mixed aerial/street. These cannot support a direct quality or
FPS winner claim merely because both names contain MatrixCity. Two defensible
tiers are required:

1. **native-method reproduction and characterization**, each on its faithful
   asset/regime; and
2. **controlled shared-data evaluation** only after both trained
   representations can render the same frozen public cameras and GT images.

If CityGaussian cannot be trained faithfully on the mixed public crop, report
the methods in separate native sections and restrict direct comparisons to
clearly common aerial cameras or systems mechanisms. Do not normalize away
representation differences that are the methods' substance.

## 13. Thesis framing

### Section 1: CityGaussian reproduction and characterization

- author asset and source/version boundary;
- released visibility and Nyquist LoD behavior;
- resident representation and released backend caveats;
- quality, workload, frame-time, and memory characterization.

### Section 2: ALoD reproduction and characterization

- correctness-patched training provenance;
- `.dhier`, global identity, HSPT, frustum, SPT cut, and external-memory model;
- official cold evaluation and official viewer reuse as separate source modes;
- explicit `PAPER-SOURCE-DIVERGENCE` for bounded render cache;
- mixed-scale formal asset and paper-style cache/frustum/HSPT-BFS trends.

### Section 3: Project-specific scalable rendering extensions

Currently supported project systems contributions, without a novelty claim:

- materialization-aware host residency;
- conservative post-selection pre-materialization culling;
- a CUDA/PyTorch implementation preserving the project conservative filter;
- row-level bounded property caches with explicit capacities;
- trace-derived reduced cache sizing; and
- instrumentation separating semantic selection, materialization, backend
  visibility, transfer, persistent residency, and peak memory.

It is defensible to call these **project-specific systems contributions** or
**extensions evaluated in this thesis**. It is not yet defensible to call them
research-novel without a separate related-work and novelty audit.

## 14. Recommended execution sequence

1. Freeze this memo and preserve all historical experiment IDs.
2. Acquire/validate public MatrixCity street data and build a versioned
   mixed-scale dataset manifest; do not train yet during the construction task.
3. Quantify camera-scale distributions and choose a contiguous formal crop.
4. Freeze COLMAP conversion, held-out cameras, point-cloud merge/downsample,
   resource policy, and provisional <=20M cap.
5. Train the formal mixed-scale asset with the pinned method plus the disclosed
   topology correction; validate every hierarchy barrier.
6. Re-run official evaluator selection/property/quality oracles.
7. Implement `ALOD-REPRO-OFFICIAL-EVAL-COLD` and no-frustum under one formal
   protocol.
8. Reproduce official viewer compatible-SPT reuse separately.
9. Reconstruct the render-only bounded SPT cache as
   `ALOD-REPRO-PAPER-RENDER-FULL-APPROX`, retaining the divergence label.
10. Run cache/no-cache/frustum ablations on one continuous path.
11. Add cut-only HSPT/BFS comparison with identical selection intent.
12. Only then compare the reproduction branch with `ALOD-EXT-*` materialization
    and reduced-cache extensions.

## 15. Frozen boundary statements

The following statements are now frozen for future work:

```text
OFFICIAL/PAPER ALoD
    global renderable hierarchy; HSPT; early frustum; host properties;
    SPT-cut temporal cache; paper bounded Render Full semantics

SOURCE-FAITHFUL PROJECT REPRODUCTION
    exact Run005 HSPT/selection; official cold host/no-cache evaluator behavior;
    correctness-patched training with explicit provenance

SOURCE-INSPIRED ADAPTATION
    fully resident ALoD experiments; exact-row previous-working-set reuse

PROJECT-SPECIFIC EXTENSION
    525k/160k row caches; conservative CPU/CUDA pre-materialization filters;
    materialization-aware cache-sizing and extension benchmarks
```

No runtime optimization, cache change, culling change, CUDA kernel work,
training, hierarchy generation, or formal benchmark was performed during this
audit.

`PASS-ALOD-REPRODUCTION-BOUNDARY`
