# ALoD Production Resident Runtime

Status: `PASS-ALOD-PRODUCTION-RESIDENT` (2026-08-08).

## Scope

This result establishes the first renderer-oriented ALoD hierarchy path in
`gs_framework`:

```text
official implementation
  -> frozen official Run005 oracle
  -> independent project CPU reference
  -> production gs_framework resident runtime
  -> shared GsplatRenderer
```

It deliberately contains no host offloading, GPU cache, working-set reuse,
streaming, asynchronous transfer, prefetching, or residency budget. All
render properties are resident on the target device; HSPT selection is CPU
side and gathers rows from those resident device tensors.

## Runtime architecture

`alod_dhier` remains the read-only binary layer and `alod_reference` remains
the slow correctness authority. Production code lives under
`scene_runtime/alod/` and does not import or invoke `alod_reference`.

`AlodResidentScene.prepare_frame()` converts the current `FrameCamera` to the
pinned four-plane selection camera, performs independent upper traversal and
SPT cuts, creates the ordered device index tensor, gathers resident properties,
and returns the existing `ActiveSplats` contract. Rendering remains the shared
`GsplatRenderer` call to `gsplat.rendering.rasterization()`.

Stable identity is always:

```text
global_node_id = serialized .dhier row index
```

Upper-local IDs, SPT offsets, selected-index rows, and raster rows are never
used as persistent identities.

## Resident representation and property conversion

Run005 contains 1,099,999 total rows: a 100,000-row skybox prefix and 999,999
hierarchy rows. The valid finest cut is the 500,000 serialized hierarchy leaf
rows plus the skybox, for 600,000 raster rows; it is not every hierarchy node.

The project-owned parser supplies serialized field-major arrays. The runtime
matches the previously validated official load/evaluation sequence:

- positions and SH coefficients are copied unchanged;
- raw rotations are transferred and normalized on the render device;
- log scales are transferred and exponentiated on the render device;
- activated opacity is clamped to `[0, 0.99999]` and inverse-sigmoid is
  computed on CPU, matching official `create_from_hier` plus
  `compact_gaussians`; selected logits are then transferred and sigmoid is
  evaluated on the render device.

The CPU-to-GPU opacity boundary is intentional: moving the inverse-sigmoid to
CUDA changes float32 hashes even though the values are visually equivalent.

## HSPT and selection

The evaluation configuration is supplied explicitly rather than embedded as
runtime defaults:

```text
root_global_node_id = 100000
skybox_count = 100000
SPT_root_volume = 150
min_SPT_size = 100
target_granularity = 0.0008629449482830516
distance_multiplier = 1
```

The production HSPT independently derives exactly:

| Structure | Rows/count |
|---|---:|
| Upper hierarchy | 1,309 |
| SPTs | 23 |
| Flattened SPT rows | 998,713 |

Every HSPT array matches the CPU reference exactly, including mappings and raw
hashes. Production selection retains the source's left/right/top/bottom-only
frustum, scale-derived bounds, stable traversal/SPT order, strict distance
predicate, and validated interval/exclusive-prefix quirks.

For all ten frozen oracle cameras, exact equality holds for candidate SPT
order, every ordered per-SPT cut, combined ordered selected hierarchy IDs,
sorted selected IDs, and the final skybox-plus-hierarchy raster vector.
Selected hierarchy counts range from 364,357 to 499,030; raster counts range
from 464,357 to 599,030.

## Pre-raster and render parity

All ten cameras achieve exact SHA-256 parity for:

- means, normalized quaternions, activated scales, activated opacities, and SH
  features in selected raster order;
- view matrices, intrinsics, and background;
- final raster global-node-ID vectors.

The ten-camera production render smoke completed with finite outputs. Image
hashes are not exact. Across the ten images, maximum absolute error was at
most `9.7885728e-4`, while the largest per-image mean absolute error was
`1.3340947e-8`. Because selected IDs and every pre-raster input hash are exact,
the remaining difference is isolated to the rasterizer/toolchain layer. The
official oracle used PyTorch `2.11.0+cu128`/CUDA 12.8; the production run used
PyTorch `2.11.0.dev20260128+cu129`/CUDA 12.9. Both report gsplat 1.5.3.

## Characterization

This is a ten-camera correctness characterization, not a formal benchmark.

| Metric | Result |
|---|---:|
| Production HSPT build | 206.67 ms |
| Reference HSPT build | 205.39 ms |
| Production selection, mean (min-max) | 7.91 ms (5.40-14.05) |
| Reference selection, mean (min-max) | 7.94 ms (5.12-14.76) |
| Selected gather, mean (min-max) | 6.36 ms (0.96-52.76) |
| Pre-raster assembly, mean | 0.025 ms |
| First raster call/JIT | 400.65 ms |
| Later raster calls, mean (maximum) | 1.54 ms (2.10) |

The first gather and first render include cold initialization and are retained
as observed rather than treated as steady-state benchmark numbers.

| Memory category | Bytes |
|---|---:|
| Serialized Gaussian properties mapped on CPU | 101,199,908 |
| Source hierarchy metadata + derived HSPT on CPU | 38,458,208 |
| Activated Gaussian properties on GPU | 101,199,908 |
| Resident global-ID identity tensor on GPU | 8,799,992 |
| PyTorch current allocated after run | 185,187,328 |
| PyTorch peak allocated | 255,300,096 |
| PyTorch peak reserved | 297,795,584 |

Selected index bytes are recorded per frame. Whole-device/process memory is
also available through the existing benchmark/viewer instrumentation where
NVML is reliable.

## Configurations and instrumentation

Three tracked correctness-characterization configurations establish the
resident ablation ladder:

- `ALOD-HIER-FINEST-RESIDENT`: skybox plus valid hierarchy leaf cut;
- `ALOD-HIER-LOD-RESIDENT`: hierarchical LoD with frustum rejection disabled;
- `ALOD-HIER-LOD-FRUSTUM`: official-equivalent LoD plus four-plane frustum.

One non-formal first-camera smoke through `benchmark_real_scene.py` completed
for every configuration with finite output:

| Configuration | Selected hierarchy | Raster rows | Backend-visible rows |
|---|---:|---:|---:|
| `ALOD-HIER-FINEST-RESIDENT` | 500,000 | 600,000 | 1,787 |
| `ALOD-HIER-LOD-RESIDENT` | 482,583 | 582,583 | 1,788 |
| `ALOD-HIER-LOD-FRUSTUM` | 364,357 | 464,357 | 1,782 |

These values characterize configuration semantics only and are not formal
performance results.

Scene/frame metrics separate HSPT build, camera preparation, upper/frustum
traversal, SPT cut, index creation, selected gather, pre-raster assembly, and
rasterization. Counts distinguish total asset, skybox, hierarchy, resident,
upper, candidate SPT, selected hierarchy, total raster, and backend-visible
rows. CPU/GPU property and hierarchy bytes are separate.

## Validation command

```bash
/home/luis/miniconda3/bin/conda run -n nerfstudio2 python \
  scripts/alod/validate_production_resident.py \
  --dhier /home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/ALOD-SMOKE-TRAIN-005/output/final.dhier \
  --oracle-manifest /home/luis/tese/runs/alod/oracles/ALOD-RUN005-SELECTION-ORACLE-001/manifest.json \
  --device cuda --output /tmp/alod-production-cuda.json
```

The opt-in integration gate uses
`GS_FRAMEWORK_RUN_ALOD_PRODUCTION_INTEGRATION=1`.

## Limitations and next step

The frozen cameras remain a `WEAK-LOD-ORACLE`: they are authoritative for
selection correctness but insufficient for formal LoD-depth behavior. CPU
selection and fresh per-frame gathers are correctness-first implementation
choices, not the final optimized design. There is no streaming/cache behavior.

The next bounded task is to run the formal resident ALoD ablation/benchmark
ladder to quantify finest-vs-LoD-vs-frustum behavior and establish the
resident baseline before adding host streaming/cache execution.
