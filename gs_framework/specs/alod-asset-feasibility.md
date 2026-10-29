# A LoD of Gaussians — Asset and Feasibility Audit

Generated: 2026-08-05

Status: **Proceed only with an isolated small-training reproduction; no official pretrained hierarchy is available.**

## Decision summary

| Question | Decision | Evidence |
|---|---|---|
| Is the official method source available? | Yes, statically auditable | Official repository at commit `73f547341547ae021ff036ad74fdf73a1240b450` |
| Is an official pretrained `.dhier` available? | No asset found | Repository tree, tags, releases, project links, and tracked files audited |
| Can a viewer-only reproduction start now? | No | It requires a trained `.dhier`, compatible extensions, and matching dataset/cameras |
| Can the local project load the official binaries? | No | Official artifacts target CPython 3.10; project uses Python 3.12 and a different CUDA stack |
| Is published-scale training feasible locally? | No | About 30 GiB host RAM and 15.9 GiB VRAM versus published 60M–150M regimes |
| Is a bounded reproduction feasible? | Conditionally yes | Isolated environment, Blackwell-compatible builds, small COLMAP scene, conservative cap |
| Is runtime integration authorized now? | No | This task is documentation-only and all runtime gates remain closed |

The operative gate result is:

```text
PROCEED-ALOD-SMALL-TRAIN
```

This means “prepare and validate an isolated official environment, then train the smallest credible hierarchy.” It does not mean that the current framework may implement an inferred hierarchy or claim paper reproduction.

## Official upstream inventory

### Repository

- URL: `https://github.com/FelixWindisch/LoDOfGaussians`
- Audited commit: `73f547341547ae021ff036ad74fdf73a1240b450`
- Local read-only checkout: `/home/luis/tese/LoDOfGaussians`
- Registered Git submodules: none
- Git tags found: none
- GitHub release assets found: none
- Repository hierarchy assets found: none

The source tree contains no `.dhier`, pretrained hierarchy, coarse/fine checkpoint, packaged HSPT metadata, evaluation render bundle, or ground-truth/result archive. The only PLY located during the asset scan is a tiny dummy fixture under `utils/dummy_data`; it is not a trained model.

The source includes several camera-path files. They may be useful for viewer replay, but they are not documented as the paper's evaluation-camera set and are not substitutes for a model plus matching scene images. Their dataset mapping must be verified before use.

### Project-linked data

The official project page links:

- a MatrixCity extension hosted on Google Drive; and
- the Uni10K dataset hosted on Hugging Face.

Uni10K is a dataset, not a pretrained ALoD model. Its public card describes a roughly 53.8 GB, mixed aerial/street COLMAP dataset with about 9,810 images, 6.2 million sparse points, and CC BY 4.0 data terms. It is much too large for the first local smoke test, though its mixed-scale views are relevant to later LoD evaluation.

The MatrixCity extension link was visible, but its folder contents, exact size, and license could not be established reliably from the accessible listing. It is classified **needs asset verification** and must not be described as a released trained model.

### Asset conclusion

No official viewer-side model path exists at audit time. The smallest defensible path is to train an official-source hierarchy and preserve it as a pinned local oracle asset, including source commit, dataset digest, resolved configuration, environment lock, and `.dhier` SHA-256.

## Licensing and redistribution

The top-level code license inherits research/non-commercial conditions from the original 3D Gaussian Splatting and Hierarchical 3DGS codebases. `simple-knn` carries the original Gaussian-Splatting research/non-commercial terms. Vendored Eigen contains its own MPL2 and component-specific licensing. Python and CUDA dependencies retain their independent licenses.

Practical rules for this thesis project:

- use the upstream source and derived models for research evaluation only unless the applicable terms are reviewed again;
- retain upstream notices in any redistributed patch or fixture;
- do not copy the official source tree into `gs_framework`;
- do not publish third-party dataset images or trained assets without checking their dataset-specific terms;
- prefer hashes, download instructions, and locally generated tiny metadata fixtures over redistributing full source/data/model artifacts; and
- conduct a dependency-license inventory before any public binary or container release.

This is an engineering audit, not legal advice.

## Official software environment versus local environment

| Dimension | Official instructions/source | Current thesis environment | Assessment |
|---|---|---|---|
| Python | 3.10 | 3.12.12 | Incompatible with tracked CPython 3.10 extensions |
| CUDA | Toolkit 12.6 / cu126 packages | PyTorch cu129, conda nvcc 12.9, system nvcc 12.0 | Must isolate and pin |
| PyTorch | cu126 install command, version not locked | 2.11.0 nightly | Reproduction risk from version drift |
| gsplat | Unpinned dependency | 1.5.3 | Resolve and record official environment version |
| Native hierarchy build | CMake lists SM 70/75/86 | RTX 5080 | Blackwell architecture absent from declared list |
| GPU | Published experiments use larger-memory devices/configurations | RTX 5080, about 15.9 GiB | Small working set only |
| Host memory | Paper reports about 1 GB per million Gaussians | About 30 GiB RAM and 16 GiB swap | 15M cap is already aggressive; 60M is out |

The official checkout contains compiled files, but they are not portable evidence: their Python ABI and likely CUDA target do not match the thesis environment. They must not be imported or copied into the current environment.

An isolated environment is mandatory. It should begin from the official Python 3.10/cu126 instructions, capture exact resolved versions, and compile both native components for the detected GPU. If upstream code changes are required solely to express a Blackwell architecture, they should live as a minimal, reviewable patch outside the frozen checkout or in a disposable worktree; the exact patch must be archived with oracle results.

## Dataset suitability

### Smoke-test requirements

A smoke scene must:

- use the standard COLMAP layout expected by the official source;
- have valid images, registered cameras, sparse points, and sufficient view overlap;
- be legally usable and small enough to train well below local RAM/VRAM limits;
- include materially different camera distances if it is used to test LoD semantics; and
- preserve a deterministic training/evaluation split.

The repository's dummy data may support parser/build diagnostics, but it is not documented as a meaningful end-to-end training scene. It cannot establish rendering quality or LoD behavior.

### Candidate progression

| Candidate | Intended use | Strength | Limitation | Decision |
|---|---|---|---|---|
| Official dummy fixture | Import/parser smoke only | Tiny and bundled | Not a credible reconstructed scene | Use only if official program supports it without invention |
| Small standard COLMAP scene | First real hierarchy train | Bounded and directly supported | May have weak scale variation | Preferred Gate-3/Gate-2 validation scene |
| Local MatrixCity Block_A | Thesis continuity | Existing baseline and aerial geometry | Current stable form uses NeRF transforms; aerial-only views weakly exercise LoD | Convert and validate later, not first environment smoke |
| Small mixed aerial/street subset | LoD semantics and navigation | Exercises large distance/scale changes | Requires careful subset construction and camera coverage | Preferred post-smoke oracle scene |
| Full Uni10K / large MatrixCity | Published-scale study | Closest target domain | Far beyond local host-memory budget | No-go on this machine |

The source's smallest shipped `small.json` still permits 15 million Gaussians and a cache target larger than that cap. It is an official configuration, not automatically a safe local configuration. The first local run must use an explicitly documented lower cap/cache budget and be labeled a **resource adaptation**, not an unmodified paper reproduction.

## Storage estimates

For a final SH-degree-1 `.dhier`, source-format arithmetic gives approximately 116 bytes per Gaussian/node:

| Count | Serialized `.dhier` lower bound |
|---:|---:|
| 1M | 116 MB |
| 5M | 580 MB |
| 15M | 1.74 GB |
| 60M | 6.96 GB |
| 150M | 17.4 GB |

Training memory is the limiting factor. The source representation alone stores three copies of the optimizable attribute width for values and Adam moments, plus nodes and auxiliary arrays. The paper's empirical planning estimate of roughly 1 GB host RAM per million Gaussians is the safer bound:

| Count | Planning host RAM | Local feasibility |
|---:|---:|---|
| 1M | about 1 GB plus process/data overhead | Plausible |
| 5M | about 5 GB plus overhead | Plausible with measurement |
| 15M | about 15 GB plus images/build/raster overhead | Risky but potentially bounded |
| 60M | about 60 GB plus overhead | No-go |
| 150M | about 150 GB plus overhead | No-go |

GPU planning must include upper hierarchy, HSPT arrays, resident Gaussian attributes, optimizer/gradient state during training, cache contents, gsplat temporary buffers, images, and the CUDA/PyTorch allocator. The paper's 60M example uses roughly 680 MB for SPT metadata plus 24 MB for upper hierarchy before active data and rasterization. A cache count alone is therefore not a VRAM budget.

## Feasibility gates

### Gate 1 — Source completeness

**Status: PASS for static analysis.**

Training orchestration, hierarchy creation, dynamic serialization, HSPT build, selection, caching, viewer/evaluation, and gsplat rendering paths are present and mapped. Runtime behavior is still unverified locally.

### Gate 2 — Official asset availability

**Status: PROCEED-ALOD-SMALL-TRAIN.**

No pretrained official hierarchy was found. Do not build an asset loader and call it a reproduction before an official-source training run creates a real `.dhier` oracle.

### Gate 3 — Reproducible environment

**Status: BLOCKED pending an isolated smoke environment.**

Required evidence:

- exact conda/pip lock and toolchain versions;
- successful source builds targeting RTX 5080;
- imports of `simple-knn`, hierarchy extension, fused SSIM, gsplat, and renderer;
- a metadata-only or official dummy path that does not modify the canonical project environment; and
- clean teardown/recreation instructions.

### Gate 4 — Serialized-format validation

**Status: STATIC PASS; NEEDS REAL-ASSET VERIFICATION.**

The byte layout and preorder global-ID convention are mapped. A standalone read-only inspector must validate a real final `.dhier`, exact file length, node/Gaussian count equality, finite attributes, SH dimensions, and tree invariants before `gs_framework` gains a loader.

### Gate 5 — Official oracle export

**Status: BLOCKED by Gates 2 and 3.**

The source has clear insertion points for selection, cache, attributes, and render outputs. Export cannot be validated without a runnable official environment and trained hierarchy.

### Gate 6 — Same-asset ablations

**Status: CONDITIONAL.**

Serialized leaves can be identified structurally, so flat-leaf and forced-finest cuts appear possible. Their semantic validity must be confirmed against official forced-finest selection and pre-rasterization attribute oracles. Interior nodes are trained renderable content, so a leaf-only result is an ablation rather than a separately trained flat baseline.

### Gate 7 — Local resource budget

**Status: NO-GO for published scale; GO only for bounded smoke scale.**

The first run must use hard stop thresholds for host RAM, process RSS, VRAM, Gaussian count, output size, and wall time. Swap use is a failure signal, not an acceptable route to a successful training result.

### Gate 8 — Method isolation

**Status: PASS for this audit.**

All proposed files are ALoD-specific documentation. No existing CityGaussian file, shared runtime abstraction, test, benchmark, or environment was modified.

## Required provenance record for a trained oracle asset

Every generated `.dhier` admitted to the project must have a sidecar manifest containing:

- upstream repository URL and exact commit;
- any applied patch as a file and hash;
- complete environment lock and compiler/CUDA/GPU inventory;
- dataset name, license reference, source URL, and content hash/manifest;
- exact train/evaluation split and camera order;
- resolved configuration, including all command-line overrides;
- random seeds and deterministic-mode settings;
- coarse PLY, final `.dhier`, and relevant checkpoint hashes/sizes;
- training duration, peak RSS, peak VRAM, final Gaussian/node counts, and failure/restart history; and
- whether the artifact may be redistributed or only regenerated locally.

## Abort criteria for the first training attempt

Abort rather than improvise if any of the following occurs:

- the native extension requires invasive algorithmic edits to run on Blackwell;
- official source selection cannot run unchanged after a minimal architecture/build patch;
- estimated peak RSS approaches 26 GiB before the configured Gaussian cap;
- sustained swap activity begins;
- allocated/reserved VRAM leaves insufficient rasterization headroom or approaches device capacity;
- hierarchy count/tree invariants fail after creator or final serialization;
- the selected dataset has no meaningful camera-distance variation for the claimed LoD test; or
- output provenance cannot be made deterministic and auditable.

## Exact next bounded task

Create a separate, disposable **official-source smoke environment** for commit `73f547341547ae021ff036ad74fdf73a1240b450`. Do not train yet. Pin Python 3.10 and resolved cu126-compatible packages, build the native extensions for RTX 5080, and verify imports plus the smallest officially supported metadata/dummy path. Record the lockfile, build commands, architecture flags, and any minimal patch. Keep `nerfstudio2`, the official checkout, and all existing method runtimes unchanged.

Only after that task passes should the project choose a small COLMAP scene and run a capped official-source training job to create the first `.dhier` oracle.

## References

- Official project: <https://felixwindisch.github.io/ALoDOfGaussians/>
- Official source: <https://github.com/FelixWindisch/LoDOfGaussians>
- Paper: <https://arxiv.org/abs/2507.01110>
- Uni10K dataset: <https://huggingface.co/datasets/ethz/Uni10K>
