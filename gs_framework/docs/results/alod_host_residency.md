# ALoD Host-Resident Properties and Bounded GPU Working Set

Date: 2026-08-19  
Status: `PASS-ALOD-HOST-RESIDENCY`

## Scope and motivation

The frozen resident campaign kept all Run005 Gaussian properties on the GPU:
101,199,908 bytes for 1,099,999 rows. Its LoD/frustum trajectory selected
364,357 to 499,030 hierarchy rows while the rasterizer ultimately reported
roughly 16.9k backend-visible rows on average. This gate addresses property
residency only. It does not change HSPT construction, selection, visibility,
LoD, rasterization, the official checkout, or any CityGaussian path.

The validated asset remains:

```text
/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/ALOD-SMOKE-TRAIN-005/output/final.dhier
SHA-256 6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c
```

The machine-readable validation report is
`/tmp/alod-host-residency-validation.json`. It is a correctness artifact, not
the next formal benchmark campaign.

## Property and identity layout

`AlodHostProperties` is the canonical complete property table. It stores
render-ready `float32` means, normalized quaternions, activated scales,
activated opacities, and SH features in CPU structure-of-arrays tensors.
Run005 SH degree 1 consumes exactly 92 bytes per row and 101,199,908 bytes in
total. Pageable memory and synchronous transfers are used; pinned-host cost is
zero.

CUDA construction activates bounded 131,072-row chunks on the render device
and copies the results back once. This preserves the resident path's exact
device-visible normalization, exponential, and sigmoid results without
retaining the complete table on GPU or reactivating selected rows per frame.
The `.dhier` itself remains a read-only memory map for serialized values and
selection/topology input.

Stable identity is always:

```text
global_node_id == serialized .dhier row == canonical host property row
```

The bounded cache maintains dense CPU `global_node_id -> slot` and reverse
`slot -> global_node_id` arrays. A GPU slot, previous-set row, selected row,
and raster row are transient identities. Validation checks mapping
bijectivity, selected-slot ownership, duplicate requests, range constraints,
and stale-slot exclusion.

## Allocation classification

| Class | Placement | Run005 behavior |
|---|---|---|
| Selection metadata | CPU | `.dhier` positions/log scales/nodes plus HSPT upper/SPT arrays; 38,458,208 bytes for node metadata plus rebuilt HSPT |
| Canonical render properties | CPU | Complete render-ready SoA; 101,199,908 bytes |
| Skybox render properties | GPU | First 100,000 rows, separate and non-evictable; 9,200,000 bytes |
| Dynamic cache properties | GPU | Mode-dependent; fixed 48,300,000 bytes for 525,000 bounded slots |
| Residency metadata | CPU | Dense maps, reverse maps, LRU stamps, and previous selected IDs; persistent GPU cache-map bytes are zero |
| Frame-local working data | CPU/GPU | Selected IDs, ordered slot vector, transferred misses, contiguous selected gather, and skybox-plus-hierarchy raster arrays |
| Rasterizer temporary data | GPU | Measured per render as peak allocator increment; retained separately in the validation JSON |

The current selector is CPU-only, so moving HSPT metadata to GPU would add a
copy without serving production selection. The ordered frame slot vector is
the only cache index transferred to GPU.

## Skybox policy

The 100,000 skybox rows are requested and prepended every frame. They are
therefore transferred once, remain permanently GPU-resident, use 9.2 MB, have
no ordinary cache slots, and cannot be evicted. Dynamic capacity and eviction
statistics exclude them.

## Three host modes

### `ALOD-HOST-NOCACHE`

This is source-faithful evaluation-style behavior where applicable: every
frame gathers the exact selected hierarchy rows from the host store, copies
all of them synchronously to GPU, prepends the resident skybox, and renders.
It performs no cross-frame property reuse and is not called a cache.

### `ALOD-HOST-PREVIOUS-REUSE`

This is a source-inspired adaptation of released viewer behavior. The released
viewer can retain compatible SPT cuts under a distance-ratio tolerance; using
that tolerance literally could change the already validated production cut.
The project mode instead retains only exact global IDs also present in the
immediately previous selected set, loads new IDs, and discards unrequested
rows. It has no explicit budget, no LRU history, no write-back, and no
optimizer state.

### `ALOD-HOST-BOUNDED-CACHE`

This is a project rendering cache, not a claim of exact training-cache
fidelity. It owns fixed read-only GPU property slots, performs hit/miss/load
accounting, and deterministically evicts the least-recently-used unrequested
slot. Current-frame hit slots are protected. Rendering is read-only, so the
official training cache's mutable-property, Adam-state, and write-back
behavior is intentionally absent.

The shared gsplat interface still requires contiguous selected arrays. The
bounded mode therefore gathers ordered cache slots into a frame-local SoA and
then prepends the skybox. Those copies are reported as frame-local memory and
assembly time, not persistent cache residency.

## Capacity derivation

The ten frozen LoD/frustum cameras select at most 499,030 hierarchy rows. That
is the minimum feasible dynamic capacity because one complete pre-raster
working set must fit. The first production capacity is 525,000 rows:

| Quantity | Rows |
|---|---:|
| Minimum feasible capacity | 499,030 |
| Chosen first capacity | 525,000 |
| Headroom | 25,970 (5.20%) |
| Separate permanent skybox | 100,000 |

The ~16.9k backend-visible count is not used to derive capacity. A dedicated
stress sequence temporarily uses the 499,030-row minimum and observed 64,153
total deterministic evictions. The normal 525,000-row cache reached 518,746
occupied rows and did not need eviction over the ten-camera frozen order.

## Correctness and rendering

The existing resident production/reference/oracle gate was rerun first. For
each of the three host modes and all ten cameras:

- ordered selected hierarchy global IDs exactly matched resident;
- skybox-plus-hierarchy raster global IDs exactly matched resident;
- means, quaternions, scales, opacities, SH features, view matrices,
  intrinsics, and background had exact SHA-256 parity;
- all selected IDs resolved to exactly one correct property row/slot; and
- rendered images were bitwise exact against the resident runtime in the same
  toolchain (`30/30` host-mode camera renders, zero maximum/mean error).

## Transfer and persistent-memory characterization

These are ten-camera correctness-trajectory totals, not formal timing results.

| Mode | Persistent GPU property bytes | Dynamic policy | Ten-camera H2D bytes | Row hit rate |
|---|---:|---|---:|---:|
| Resident | 101,199,908 | Complete asset | 0 per frame | n/a |
| Host no-cache | 9,200,000 | Skybox only | 443,671,840 | 0% |
| Previous reuse | 42,720,844 to 55,110,760 | Previous selected hierarchy set plus skybox | 52,106,316 | 88.26% |
| Bounded cache | 57,500,000 | 525,000 hierarchy slots plus skybox | 47,724,632 | 89.24% |

The bounded cache controls persistent Gaussian property memory at 57.5 MB,
43.18% below the 101.2 MB resident table. No-cache demonstrates the minimum
persistent path at the 9.2 MB skybox cost. Previous reuse transfers 11.74% and
bounded cache 10.76% of no-cache trajectory traffic after including their cold
first frames.

The largest explicitly accounted frame-local property/index allocations were
101,021,520/0 bytes for no-cache, 76,241,688/7,960,568 bytes for previous
reuse, and 113,891,124/3,992,240 bytes for bounded cache. These are temporary
contiguous transfer/gather/assembly arrays and are deliberately not described
as persistent cache residency.

The maximum per-render PyTorch peak increments attributed to rasterizer
execution were 25,444,864 bytes for no-cache, 25,576,448 bytes for previous
reuse, and 25,321,472 bytes for bounded cache. End-of-gate allocated/reserved
bytes were 96,763,904/283,115,520, 95,726,592/304,087,040, and
146,466,816/247,463,936 respectively. Whole-device NVML snapshots were about
1.89--1.95 GB, but WSL process attribution reported unreliable and the modes
ran sequentially through one process/allocator pool, so those snapshots are
supporting context rather than a controlled cross-mode result.

Cold means the first request after `reset_residency`, with no valid reusable
mapping. Warm is the second request. Steady state begins at the third request.
For a repeated camera, previous reuse and bounded cache transferred all
364,357 hierarchy rows cold and zero rows warm/steady; no-cache transferred
all 364,357 rows on every repetition.

Per-frame metrics include upper traversal, SPT cut, requested/retained/new/
discarded rows, hits, misses, evictions, occupancy, H2D rows/bytes/time, host
gather time, lookup time, slot-index construction, GPU selected gather,
pre-raster assembly, frame-local property/index bytes, backend-visible count,
raster time, total time, PyTorch allocation/reservation, process RAM, and NVML
whole-device context where reliable.

Here retained/new/discarded describe physical residency-policy actions:
no-cache retains zero rows, previous reuse retains exact previous-set hits, and
bounded cache retains all cache hits (including rows older than one frame).
Selected-set symmetric-difference churn is reported independently.

## Known costs and limitations

- Correct synchronous residency is the target; no FPS speedup is claimed.
- Dense CPU uniqueness/mapping work and unavoidable gsplat gathers are visible
  costs. They were measured, not optimized.
- The complete render-ready host table is additional committed host memory;
  serialized `.dhier` fields remain file-backed memory maps for validation and
  selection.
- Pageable transfers are used. Pinned memory, non-blocking streams, prefetch,
  asynchronous overlap, and custom CUDA are not implemented.
- The frozen trajectory is still `WEAK-LOD-ORACLE`.
- The selected-versus-backend-visible gap remains a separate future problem.
  Stronger culling, visibility-aware cuts, spatial organization, occlusion,
  and GPU selection are explicitly outside this gate.
- The complete 900-frame resident-vs-host formal campaign was not run.

## Validation

```bash
/home/luis/miniconda3/bin/conda run -n nerfstudio2 python \
  scripts/alod/validate_host_residency.py \
  --dhier /home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/ALOD-SMOKE-TRAIN-005/output/final.dhier \
  --oracle-manifest /home/luis/tese/runs/alod/oracles/ALOD-RUN005-SELECTION-ORACLE-001/manifest.json \
  --device cuda --output /tmp/alod-host-residency-validation.json
```

The opt-in test gate is
`GS_FRAMEWORK_RUN_ALOD_HOST_RESIDENCY_INTEGRATION=1`.

## Next bounded task

Run the formal resident-vs-host-no-cache-vs-previous-reuse-vs-bounded-cache
campaign and quantify GPU-memory reduction, H2D traffic, cache reuse,
frame-time cost, and remaining bottlenecks before scaling to a larger or
mixed-scale ALoD asset.
