# A LoD of Gaussians — Official Source Map

Generated: 2026-08-05

## Project format-inspection status — 2026-08-07

`alod_dhier/` is the independent project format boundary. It decodes the
official field-major native stream with read-only NumPy memory maps, validates
numeric attributes and binary-tree topology, and generates deterministic JSON
reports. It deliberately lives outside `scene_runtime`: importing the current
`scene_runtime` package eagerly imports CityGaussian and rendering dependencies,
which would violate the inspector's isolation boundary.

Compact official-loader fixtures live under
`unit_tests/fixtures/alod_dhier/`. Their Layer-A hashes come directly from
`gaussian_hierarchy._C.load_dynamic_hierarchy`; Layer B comes from
`GaussianModel.create_from_hier(..., scaffold_file="", device="cpu")` so model
transformations are isolated from external scaffold/skybox prepending. The
official source is not called by the parser or normal CLI.

## Pinned upstream

| Item | Audited value |
|---|---|
| Repository | `https://github.com/FelixWindisch/LoDOfGaussians.git` |
| Local read-only checkout | `/home/luis/tese/LoDOfGaussians` |
| Branch | `main` |
| Commit | `73f547341547ae021ff036ad74fdf73a1240b450` |
| Commit date | 2026-07-03 |
| Registered Git submodules | None |
| Vendored native dependencies | `submodules/gaussianhierarchy`, `submodules/simple-knn` |
| Tags/releases at audit time | No Git tags; no GitHub release assets found |

The checkout is deliberately outside `gs_framework` and was inspected read-only. Although two directories are named `submodules`, the repository has no `.gitmodules` and `git submodule status --recursive` reports none. There are therefore no submodule commit IDs to pin independently; their content is pinned by the main repository commit.

## End-to-end call graph

```text
COLMAP scene
  -> Scene / dataset_readers / CameraDataset
  -> train_scaffold.training
  -> coarse GraphDECO PLY + skybox count
  -> GaussianHierarchyCreator (C++)
       -> LessRichPoint PLY loader
       -> point-based binary k-d tree
       -> recursive Gaussian parent merging
       -> initial .dhier
  -> GaussianModel.create_from_hier
       -> compact CPU properties + optimizer state
       -> build_hierarchical_SPT
  -> per-camera upper hierarchy traversal + frustum culling
  -> selected SPT roots and camera distances
  -> CUDA SPT binary search/filter
  -> global Gaussian IDs
  -> cache decision + CPU gather + host-to-device copy
  -> concatenate upper nodes, SPT cuts, reuse/cache, skybox
  -> gaussian_renderer.render_gsplat
  -> gsplat.rasterization(packed=True)
  -> training gradients, write-back, densification, HSPT rebuild
  -> final .dhier
```

## Stage-to-source map

| Stage | File | Primary symbol or region | Reproduction role |
|---|---|---|---|
| Orchestration | `train.py` | main training flow | Runs scaffold training, hierarchy creator, then fine training |
| Scene loading | `scene/__init__.py` | `Scene.__init__` | Resolves dataset, camera splits, model paths |
| COLMAP parsing | `scene/dataset_readers.py` | `readColmapSceneInfo`, `readColmapCameras` | Produces scene/camera records and sparse initialization |
| Camera/image loading | `utils/camera_utils.py` | `CameraDataset`, `loadCam` | Loads pixels and creates runtime camera objects |
| Coarse training | `train_scaffold.py` | `training` | Conventional scaffold optimization |
| Point initialization | `scene/gaussian_model.py` | `GaussianModel.create_from_pcd` | Creates initial Gaussians from sparse points |
| Coarse save | `scene/__init__.py`, `scene/gaussian_model.py` | `Scene.save`, `save_ply_coarse` | Writes hierarchy-creator input and skybox metadata |
| C++ creator entry | hierarchy creator sources | `mainHierarchyCreator.cpp::main` | Loads PLY and drives tree construction/serialization |
| Fixed PLY load | hierarchy creator sources | `Loader::loadPly<LessRichPoint>` | Defines accepted binary PLY property layout |
| Binary spatial tree | hierarchy creator sources | `PointbasedKdTreeGenerator::generate`, `recKdTree` | Median split on longest bounding-box axis |
| Parent approximation | hierarchy creator sources | `ClusterMerger::merge`, `mergeRec` | Produces renderable interior Gaussians |
| Rotation handling | hierarchy creator sources | `RotationAligner::align` | Aligns child rotations for stable merging |
| Initial serialization | hierarchy creator sources | `Writer::writeDynamicHierarchy`, `populateDynamicRec` | Emits preorder `.dhier` |
| Fine training | `train_hierarchy.py` | `training` | Hierarchical optimization and out-of-core working set |
| Dynamic load | `scene/gaussian_model.py` | `create_from_hier` | Loads `.dhier` and activates model state |
| CPU compaction | `scene/gaussian_model.py` | `compact_gaussians` | Packs properties and optimizer moments in host RAM |
| HSPT construction | `scene/gaussian_model.py` | `build_hierarchical_SPT` | Builds upper hierarchy and SPT arrays |
| Upper cut | hierarchy utilities / call sites | `cut_hierarchy_on_condition` plus inline train/eval/viewer logic | Selects upper nodes and lower SPT requests |
| Frustum culling | hierarchy utilities | `extract_frustum_planes`, `frustum_cull_spheres` | Rejects upper subtrees |
| SPT Python binding | Gaussian hierarchy extension | `get_spt_cut_cuda` | Exposes GPU cut operation |
| SPT C++ binding | extension C++ | `GetSPTCut`, `Switching::getSPTCut` | Dispatch and shape management |
| SPT CUDA | `runtime_switching.cu` | `binary_search_SPTs`, `populate_SPT` | Finds and compacts selected global IDs |
| Densification | `scene/gaussian_model.py` | `add_new_gs`, `relocate_gs` | Extends/relocates hierarchy nodes |
| Final save | `scene/gaussian_model.py` | `save_hierarchy` | Sends optimized hierarchy to writer extension |
| Dynamic writer binding | extension binding | `WriteDynamicHierarchy` | Python/C++ bridge |
| Dynamic writer | hierarchy serializer | `HierarchyWriter::writeDynamic` | Writes final `.dhier` fields |
| Dynamic loader binding | extension binding | `LoadDynamicHierarchy` | Python/C++ bridge |
| Dynamic loader | hierarchy serializer | `HierarchyLoader::loadDynamic` | Reads `.dhier`; assumes nodes equal Gaussians |
| Public renderer | `gaussian_renderer/__init__.py` | `render_gsplat` | Converts attributes/camera and calls gsplat |

## Runtime-path differences

### Fine-training path

The fine trainer performs upper-tree and SPT selection, reuses compatible cached SPT cuts, loads misses from CPU, retains unrequested cuts until capacity reduction, and writes evicted attributes and Adam state back to host memory. Cache reduction and periodic clearing occur in the training loop. This is the source path closest to the paper's bounded out-of-core cache.

Key regions to instrument in a future oracle exporter:

- upper hierarchy tested, rejected, descended, and emitted nodes;
- requested SPT roots and camera distances;
- reuse-ratio decisions;
- new global IDs gathered from `gaussians.properties`;
- cache concatenation order and SPT boundaries;
- eviction/write-back global IDs; and
- selected attributes immediately before `render_gsplat`.

### Interactive-viewer path

The viewer retains compatible requested SPT cuts from the prior working set and drops unrequested ones. It does not implement the training cache's explicit capacity reduction, write-back, or periodic flush. It also hard-codes a base focal length of 1000. A viewer-fidelity reproduction must match this policy rather than label it LRU.

### Evaluation path

`eval_hierarchy.py` rebuilds the selected tensors and transfers them for each camera. It does not preserve a useful inter-frame cache. This makes it the cleanest public cold/no-cache selection and rendering oracle once an official environment and `.dhier` asset exist.

## Data and identity flow

```text
serialized global node ID (preorder index)
  -> upper-tree global-ID mapping OR SPT sorted global-ID array
  -> per-camera selected global IDs
  -> cache lookup keyed by SPT identity + cut distance
  -> rows in CPU properties / optimizer tensors
  -> concatenated runtime slot
  -> gsplat Gaussian index for this frame
```

Only the first identity is durable. The runtime slot is not a model identity and must never be written into benchmark records as if it were one. Oracle artifacts should always retain the serialized global node ID alongside any upper-tree, SPT, cache, or frame-local index.

## Serialized hierarchy source map

| Concern | Writer behavior | Loader behavior | Validation requirement |
|---|---|---|---|
| Byte order/schema | Native raw writes, no header magic/version | Native raw reads | Inspector checks exact expected length and rejects trailing/truncated data |
| Node order | Preorder traversal | Read as stored | Preserve array index as global node ID |
| Attribute activation | Stores log scale and activated opacity | Applies inverse sigmoid to opacity for optimization | Compare raw and activated attributes separately |
| Counts | Writes Gaussian and node counts | Reads node count, then uses Gaussian count | Require equality before import |
| Tree links | Six `int32` fields | Direct tensor load | Validate ranges, roots, parents, sibling chains, child counts, cycles, and reachability |
| SH | Degree and all coefficients serialized | Allocates from degree | Validate coefficient count against file length |
| HSPT/cache | Not serialized | Rebuilt | Record commit and all hierarchy-build settings |

Creator-specific qualification: initial leaves store activated opacity, while
`ClusterMerger.cpp` can produce merged interior opacity proxies above one.
The Python model clamps all loaded opacity values to `[0, 0.99999]` before
inverse-sigmoid. The initial creator also leaves an unused positive
`first_child` payload on leaves because `writer.cpp::populateDynamicRec` assigns
the field before observing `children.size() == 0`. Neither behavior weakens
strict final-hierarchy validation.

## Public renderer contract

`render_gsplat` expects activated/render-ready tensors:

- means in world coordinates;
- normalized quaternions;
- exponential scales;
- sigmoid opacities;
- DC plus rest SH coefficients;
- a camera matrix converted from the source camera convention; and
- intrinsics reconstructed from image size and FoV.

It calls `gsplat.rasterization` with `packed=True` and RGB rendering and returns gsplat metadata including Gaussian IDs, radii, and projected means. A reproduction should first compare the complete pre-rasterization tensor bundle; an image-only comparison cannot distinguish selection, activation, camera, ordering, and rasterizer differences.

## Configuration evidence

| Configuration | Coarse iterations | Fine iterations | Maximum Gaussians | Cache / reduced target | Notes |
|---|---:|---:|---:|---:|---|
| `configs/small.json` | 30,000 | 100,000 | 15M | 25M / 20M | Smallest shipped configuration; densification limit exceeds fine duration |
| `configs/default.json` | 60,000 | 250,000 | 100M | 22M / 18M | Graph-view sampling enabled |
| `configs/MatrixCity.json` | 250,000 | 500,000 | 150M | 25M / 20M | Far beyond local host-memory budget |

These public settings differ from the paper's described 100,000-iteration coarse phase. Every future fixture must include the complete resolved configuration rather than just a configuration filename.

## Environment and build surface

| Component | Official release expectation | Local project state | Risk |
|---|---|---|---|
| Python | 3.10 | 3.12.12 | Tracked `.so` files and extensions are not ABI-compatible |
| CUDA toolkit | 12.6 | PyTorch CUDA 12.9; conda nvcc 12.9; system nvcc 12.0 | Requires isolated, pinned toolchain |
| PyTorch | cu126 wheel instructions, version not pinned | 2.11 nightly + cu129 | API/build drift |
| gsplat | Requirements entry, unpinned | 1.5.3 | Exact upstream environment must record resolved version |
| C++ hierarchy tools | CMake architectures 70, 75, 86 | RTX 5080/Blackwell | Missing declared target architecture |
| Python extensions | Source plus tracked CPython 3.10 binaries | Project Python 3.12 | Must rebuild in isolation; never import tracked binary blindly |

The reproduction environment must be separate from `nerfstudio2`. Resolving Blackwell architecture support is an environment task, not permission to edit the official checkout in place.

## Static source deviations to preserve in tests

- `GaussianModel.get_SPT_cut` overwrites its calculated condition with all true; primary call sites do not rely on this helper.
- Near/far frustum planes are not active in the audited extraction code.
- Viewer focal normalization is hard-coded; training/evaluation derive it differently.
- Viewer reuse, training cache, and evaluation behavior are different programs.
- `render_gsplat` fixes packed RGB behavior and ignores several wrapper arguments.
- Dynamic loader assumes one Gaussian per hierarchy node.
- HSPT and cache structures are reconstructed and absent from `.dhier`.
- Source distributions include built extension artifacts, but those are not portable evidence of local compatibility.

## First oracle insertion points

Once the isolated official environment and a small trained `.dhier` exist, add a separate exporter without changing algorithmic decisions. The minimum insertion points are:

1. immediately after dynamic hierarchy load;
2. immediately after HSPT construction;
3. after upper traversal/frustum culling;
4. after CUDA SPT cut and before cache processing;
5. after cache processing and CPU-to-GPU gathering;
6. immediately before `render_gsplat`; and
7. after rasterization metadata returns.

The exporter should write versioned JSON metadata plus NPZ tensor payloads and include repository commit, asset SHA-256, dataset/camera identifiers, resolved configuration, and software versions.
