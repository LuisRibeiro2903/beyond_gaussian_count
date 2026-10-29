# A LoD of Gaussians — Reproduction and Integration Plan

Generated: 2026-08-05

Status update 2026-08-07: the isolated environment, correctness-patched Run005
oracle asset, and standalone read-only `.dhier` inspector are validated. The
historical staged plan below is retained; the next bounded task is official
hierarchy/HSPT/selection oracle instrumentation, not project runtime selection.

## Objective

Reproduce the released ALoD hierarchy selection and rendering behavior inside `gs_framework`, validate it against the official source on identical assets and cameras, and then expose controlled ablations for visibility, LoD, residency, caching, and memory behavior on one consumer GPU.

The reproduction is a systems contribution: it must explain what is selected, what is resident, what moves between CPU and GPU, how long each stage takes, and how those choices affect image quality and frame time. A visually plausible hierarchy without official selection/attribute parity is insufficient.

## Non-goals for the reproduction phase

- Do not redesign the hierarchy, loss, optimizer, densification, or CUDA cut before the oracle passes.
- Do not add asynchronous prefetching, predictive navigation, compression, or a new cache policy and label it reproduction.
- Do not modify the official checkout in place.
- Do not modify or couple to the frozen CityGaussian implementation.
- Do not use published image metrics from a different asset, camera set, renderer backend, or source version as local replication evidence.
- Do not train at published 60M–150M scale on the current machine.

## Fidelity vocabulary

| Label | Meaning |
|---|---|
| Official reference | Output emitted by the pinned official source and environment |
| Source-faithful | Matches audited public-code behavior, including documented deviations from the paper |
| Paper-faithful | Implements behavior described in the paper when the public viewer differs |
| Backend adaptation | Same intended method semantics but different renderer/library behavior |
| Resource adaptation | Lower caps/budgets/dataset scale required by local hardware |
| Controlled ablation | Intentionally disables one component while holding asset/cameras fixed |
| Thesis extension | New behavior beyond the reproduction; must be evaluated separately |

## Workstream 0 — Preserve repository and method boundaries

Deliverables:

- ALoD code under a new method-specific namespace when implementation is authorized;
- ALoD-specific fixtures, configurations, and benchmark outputs;
- no imports from or edits to frozen method-specific runtimes;
- external source and build trees outside `gs_framework`; and
- local commits in the functional `gs_framework/.git` only after each authorized implementation task.

Gate: a path-level diff audit must show that an ALoD task touched only approved ALoD or genuinely shared files. A shared abstraction change requires explicit review before implementation.

## Workstream 1 — Reproduce the official environment

This is the next task and precedes all runtime integration.

1. Create a disposable Python 3.10 environment outside the current `nerfstudio2` environment.
2. Resolve the official cu126 PyTorch command to exact versions and store a lock.
3. Build `simple-knn`, `gaussianhierarchy`, fused SSIM, and the CMake creator/merger tools from the pinned commit.
4. Target the detected RTX 5080 architecture explicitly; retain any minimal build-only patch separately.
5. Verify imports and official command help/metadata paths.
6. Exercise only the smallest upstream-supported dummy/metadata operation.
7. Record compiler, CUDA, driver, GPU, package, binary, and patch hashes.

Acceptance:

- environment recreation succeeds from the recorded instructions;
- no package in the canonical project environment changes;
- no tracked prebuilt `.so` is mistaken for a locally compatible build;
- no algorithmic source edit is required; and
- the official checkout remains clean.

## Workstream 2 — Create the first official oracle asset

Select a small, legal, standard COLMAP scene. Start below the official `small.json` resource envelope with explicit resource-adaptation overrides. Preserve a fixed evaluation camera set before training.

Run the official pipeline:

```text
COLMAP input
  -> coarse scaffold training
  -> C++ initial hierarchy
  -> fine hierarchy training
  -> final .dhier
```

Required controls:

- hard Gaussian cap and cache target compatible with 30 GiB RAM / 15.9 GiB VRAM;
- process-RSS, swap, VRAM, output-size, and wall-time monitoring;
- deterministic seeds where supported;
- hashes for dataset manifest, coarse PLY, initial hierarchy, final hierarchy, and configuration;
- creator and loader invariant checks after each serialization; and
- an abort threshold before memory pressure destabilizes WSL/the host.

Acceptance:

- final hierarchy loads in the same official environment;
- Gaussian count equals node count;
- all hierarchy links and attributes validate;
- fixed cameras render in the official evaluation path; and
- full provenance sidecar is complete.

## Workstream 3 — Build read-only hierarchy inspector

Implement a standalone, versioned `.dhier` inspector before a project runtime loader. It should not import CUDA or construct a renderer.

Checks:

- exact header and computed byte length;
- supported SH degree and coefficient count;
- finite positions, rotations, scales, opacities, and SH;
- normalized/nonzero quaternion diagnostics;
- Gaussian/node count equality;
- root count, parent/child/sibling index ranges;
- child-count consistency, reachability, cycles, and preorder expectations;
- depth consistency and leaf/interior counts;
- bounding box, scale/opacity distributions, and estimated resident sizes; and
- SHA-256 plus a schema-versioned JSON report.

The reader must default to rejection when format assumptions fail because `.dhier` has no magic or version marker.

Acceptance: its report agrees with the official loader/exporter on the oracle asset and a deliberately corrupted/truncated copy is rejected.

## Workstream 4 — Add official oracle exporter

Add a non-algorithmic exporter to a separate official-source worktree. Export versioned JSON metadata and NPZ tensors at these points:

1. hierarchy immediately after `.dhier` load;
2. rebuilt upper hierarchy and every SPT;
3. upper traversal and frustum decision;
4. raw CUDA SPT cut;
5. cache/reuse/load/eviction result;
6. final activated, ordered pre-rasterization tensors; and
7. rendered image plus gsplat metadata.

### Stable identity rules

- `global_node_id`: serialized preorder index; primary stable ID for one asset.
- `upper_local_id`: derived; always export its global-node mapping.
- `spt_id`: derived from asset, source commit, and HSPT configuration; export root global ID and member global IDs.
- `cache_slot` and `frame_gaussian_index`: transient; never compare without global IDs.

### Hierarchy oracle payload

- asset hash, byte size, SH degree, node/Gaussian counts;
- raw attributes and activated attributes;
- node records, leaf flags, roots, depths, and global IDs;
- all resolved HSPT parameters;
- upper-to-global mapping;
- SPT roots, starts/counts, sorted global IDs, min/max distance arrays, and bounds; and
- focal/distance normalization inputs.

### Per-camera selection payload

- dataset/camera ID and camera sequence index;
- image width/height, FoV/focal values, K, view/projection matrices, camera center;
- exact distance multiplier;
- upper nodes tested, frustum-rejected, descended, and emitted;
- requested/rejected SPT roots and request distances;
- raw SPT-cut global IDs and starts/counts;
- final ordered global IDs and hierarchy-depth histogram; and
- selected upper, SPT, skybox, and total counts.

### Cache oracle sequences

Export separate deterministic traces for:

- isolated cold cameras;
- repeated identical camera;
- slow adjacent navigation;
- stationary camera;
- alternating near/far camera;
- large teleport; and
- cache-capacity pressure.

For every frame record requested/reused/loaded/retained/evicted SPTs, their root global IDs, Gaussian counts, bytes, distance ratios, cache occupancy, transferred bytes, and write-back bytes. Keep three oracle modes separate: official evaluation no-cache, official viewer prior-working-set reuse, and fine-training write-back cache.

## Workstream 5 — Three-layer equivalence testing

### Layer A: hierarchy and selection equivalence

For the same `.dhier`, resolved configuration, and camera:

- exact global-ID equality after normalizing only documented ordering differences;
- exact upper and SPT root decisions;
- exact selected counts and depth histograms;
- exact frustum decision masks; and
- exact cache trace on deterministic sequences.

Any mismatch must be localized before rendering.

### Layer B: pre-rasterization attribute equivalence

Compare final ordered tensors for the common global IDs:

- means;
- normalized quaternions;
- activated scales;
- activated opacities;
- DC/rest SH;
- background/skybox placement; and
- camera matrices and intrinsics.

Report maximum absolute error, relative error, and mismatch count per field. Define tolerances only after measuring official CPU/GPU dtype behavior; do not choose permissive thresholds in advance.

### Layer C: render equivalence

First run the same gsplat version and settings. Compare:

- image dimensions and color space;
- per-pixel absolute and relative error;
- PSNR/SSIM/LPIPS between official and project render;
- rasterizer metadata, projected/visible counts, and Gaussian IDs where available; and
- cold and warmed timing separately.

Only after same-backend parity passes may the project compare alternative backend/settings. Such results are backend adaptations.

## Configuration ladder

All configurations use the same oracle asset, camera set, attribute activation, background, and rasterizer unless the row explicitly changes one component. The ladder branches at host caching because the public viewer and paper/training cache are genuinely different policies.

| ID | Parent | One intentional change | Classification | Required oracle | Current status |
|---|---|---|---|---|---|
| `ALOD-OFFICIAL-REFERENCE` | none | Pinned official program produces source-of-truth outputs | Official reference | Environment, asset, all exporter payloads | Blocked by environment/asset |
| `ALOD-OFFICIAL-SET-RENDER` | official reference | Freeze a deterministic camera set and export selection, attributes, images | Validation fixture, no method change | Camera manifests and NPZ/JSON/image bundle | Blocked |
| `ALOD-FLAT-LEAVES` | set render | Load serialized leaves plus skybox as a flat finest-level set | Controlled ablation | Official leaf IDs and activated attributes | Proposed |
| `ALOD-HIER-FINEST-RESIDENT` | flat leaves | Traverse the hierarchy/HSPT but force the finest valid cut; keep attributes resident | Controlled hierarchy ablation | Forced-finest official exporter | Proposed |
| `ALOD-HIER-LOD-RESIDENT` | hierarchy finest resident | Enable camera-dependent LoD cut; keep all attributes resident | Controlled LoD ablation | Exact upper/SPT selected IDs | Proposed |
| `ALOD-HIER-LOD-FRUSTUM` | hierarchy LoD resident | Enable the audited official frustum rejection | Source-faithful selection | Frustum masks and selected IDs | Proposed |
| `ALOD-HOST-NOCACHE` | hierarchy LoD frustum | Keep full attributes in host memory and transfer every selected frame set | Source-faithful to evaluation path | Transfer and attribute oracle | Proposed |
| `ALOD-HOST-PREVIOUS-REUSE` | host no-cache | Reuse only compatible requested SPT cuts from retained prior working set | Source-faithful to released viewer | Viewer reuse trace | Proposed |
| `ALOD-OFFICIAL-FULL` | host previous reuse | Match remaining released-viewer camera/raster/assembly details | Released-viewer fidelity target | Full three-layer oracle | Proposed |
| `ALOD-HOST-LRU` | host no-cache | Add bounded retained SPT cache with training tolerances, reduction, and eviction/write-back semantics | Paper/fine-training cache branch | Training cache trace | Proposed |
| `ALOD-PAPER-CACHE` | host LRU | Match remaining paper/training cache controls, including flush and navigation sequence | Paper/training fidelity target | Full cache and timing oracle | Proposed |

Important interpretations:

- `ALOD-FLAT-LEAVES` is not a separately trained flat 3DGS baseline. It is a same-asset forced-leaf ablation.
- `ALOD-HIER-FINEST-RESIDENT` retains hierarchy interpretation and selection machinery, isolating hierarchy traversal from LoD.
- `ALOD-HIER-LOD-RESIDENT` will only fit small fixtures; its value is causal isolation, not scalability.
- `ALOD-HOST-NOCACHE` corresponds most closely to public evaluation behavior.
- `ALOD-HOST-PREVIOUS-REUSE` and `ALOD-HOST-LRU` are sibling policies. One must not be used as evidence for the other.
- Async transfer/prefetch, a byte-native cache, predictive policies, quantization, and alternative bounds are future thesis extensions with new IDs.

## Proposed project architecture

Implementation remains conditional, but the anticipated separation is:

```text
scene_runtime/alod/
  format.py          read-only .dhier parser and validation
  hierarchy.py       stable nodes and derived HSPT representation
  selection.py       upper traversal, frustum, SPT cuts
  residency.py       host store and frame working set
  cache.py           explicit viewer-reuse and write-back policies
  adapter.py         activated tensors and gsplat input assembly
  oracle.py          fixture readers/comparators
```

Keep format parsing, semantic selection, residency, and rendering separable. This permits exact ID/attribute testing without a GPU and prevents cache policy from becoming entangled with hierarchy correctness. Final locations should be reviewed against the repository's current module organization before coding.

## Metrics mapping and instrumentation plan

### Existing metrics that can be reused or precisely derived

| ALoD metric | Existing project field/source | Note |
|---|---|---|
| Total selected Gaussians | `active_gaussians` / selected count | Rename only in ALoD reporting; preserve semantic definition |
| Resident Gaussians | `resident_gaussians` | Existing general block path field; add ALoD producer |
| Newly uploaded / evicted count | `newly_uploaded_gaussians`, `evicted_gaussians` | Count must derive from global IDs/SPT membership |
| Cache hit/miss and rate | `cache_hits`, `cache_misses`, `cache_hit_rate` | Define whether one request means one SPT, not one Gaussian |
| Host-to-device time | `gpu_upload_ms` | ALoD-specific stage timing should feed/alias this field |
| GPU/CPU resident bytes | `gpu_resident_bytes`, `cpu_resident_bytes` | Split into components below while retaining total |
| Rasterization time | `RenderResult.gpu_render_time_ms` / benchmark rasterization time | Same synchronization policy required |
| End-to-end frame time | benchmark `end_to_end_frame_time_ms` | Primary interactive systems result |
| Visible/projected count | `visible_gaussians` and rasterizer metadata | Do not confuse with selected count |
| Torch VRAM | current/peak allocated/reserved benchmark fields | Reset/synchronize consistently per run |
| Process RAM | existing process-RSS capture | Record peak and steady-state |
| Device/process VRAM | NVML metrics | Retain reliability warning for per-process reporting |

### New ALoD-specific metrics required

Selection and hierarchy:

- upper nodes tested, descended, cut, and frustum-rejected;
- requested SPT count and selected SPT count;
- selected upper/SPT/skybox Gaussian counts;
- selected hierarchy-depth histogram;
- HSPT build time and bytes;
- upper traversal time, frustum time, and GPU SPT-cut time; and
- selection churn: added/removed global IDs and SPT roots between frames.

Transfer and cache:

- requested, reused, loaded, retained, and evicted SPT counts;
- host gather time, host-to-device enqueue/synchronized time, and write-back time;
- host-to-device and device-to-host bytes;
- cache occupancy in SPTs, Gaussians, and bytes;
- reuse distance-ratio distribution;
- compulsory, capacity, tolerance, and flush misses; and
- cache-policy name and capacity unit.

Memory components:

- CPU raw hierarchy bytes;
- CPU property and optimizer-state bytes;
- GPU upper hierarchy bytes;
- GPU HSPT/SPT metadata bytes;
- GPU persistent skybox bytes;
- GPU cache/working-set attribute bytes;
- GPU frame-concatenation temporary bytes; and
- rasterizer/allocator remainder inferred from observed peaks.

Timing names must state their synchronization boundary. CPU enqueue time must not be reported as completed GPU transfer or rasterization time.

## Benchmark protocol

### Fixed camera sequences

Use at least:

- fixed isolated quality cameras;
- stationary repeat;
- slow local translation;
- continuous approach/retreat through LoD thresholds;
- lateral frustum sweep;
- alternating near/far views;
- large teleport; and
- mixed aerial/street transition once a suitable scene exists.

Save exact camera matrices and order. Cache results are sequence-dependent and cannot be compared from unordered image sets.

### Run classes

- **Cold**: process/cache newly initialized for each measured camera or sequence.
- **Warm deterministic**: one documented warm-up traversal, followed by recorded repetitions.
- **Continuous navigation**: cache state intentionally carries across the full sequence.
- **Capacity stress**: budget selected to force eviction without out-of-memory behavior.

Quality results use the fixed ground-truth camera set. Systems results report median, p95, p99, and worst frame time plus stage breakdowns. Record active, visible, resident, and transferred counts per frame rather than averages alone.

### Resource safeguards

- start with one small oracle hierarchy and conservative cache;
- estimate serialized, CPU, and GPU component sizes before load;
- refuse a configuration whose static estimate leaves inadequate VRAM headroom;
- terminate on sustained swap or threshold breach;
- preserve partial logs on failure; and
- never turn an OOM-derived reduced configuration into an undocumented default.

## Correctness gates before performance claims

1. `.dhier` inspector agrees with official loader.
2. Global node identity and activated attributes agree.
3. Rebuilt HSPT metadata agrees.
4. Upper traversal and SPT selection agree on every fixture camera.
5. Cold/no-cache assembled tensor bundle agrees.
6. Released-viewer reuse trace agrees.
7. Fine-training cache trace agrees separately.
8. Same-backend image equivalence passes.
9. Metrics account for selected/resident/transferred work without semantic ambiguity.
10. Only then may performance and quality trade-offs be published.

## Result-reporting matrix

Every experiment row must include:

- configuration ID and parent;
- official source commit and project commit;
- asset/dataset/camera-set hashes;
- renderer and exact gsplat/PyTorch/CUDA versions;
- hierarchy/HSPT/focal/frustum configuration;
- cache policy, tolerance, capacity, and initial state;
- total/leaf/interior/selected/visible/resident Gaussian counts;
- transfer and cache statistics;
- frame/raster/selection/transfer timings;
- peak RSS and VRAM components;
- PSNR, SSIM, LPIPS for quality runs; and
- fidelity classification and any deviation.

## Decision gates and current disposition

| Gate | Requirement | Current disposition |
|---|---|---|
| G1 Source | Complete mapped official pipeline | Static pass |
| G2 Asset | Official pretrained asset or official small training | Small training required |
| G3 Environment | Reproducible isolated official build | Blocked; next task |
| G4 Format | Inspector validated on real final hierarchy | Static map only |
| G5 Oracle | Official selection/cache/attribute/render fixtures | Blocked by G2/G3 |
| G6 Same asset | Flat/finest/LoD cuts proven semantically comparable | Conditional |
| G7 Resources | Measured safe RAM/VRAM envelope | Small-only; published scale no-go |
| G8 Isolation | No frozen-method or environment contamination | Pass for documentation audit |

## Exact next task

Reproduce the pinned official source's **smoke environment only** in a disposable Python 3.10 environment. Resolve exact dependencies, build the native hierarchy and KNN extensions for RTX 5080, verify imports and the smallest official metadata/dummy path, and publish an environment/build record. Do not train, implement a project loader, or change any existing runtime in that task.

If the smoke environment passes, the following task is a separately approved capped training run that produces the first real `.dhier` oracle and provenance manifest.
