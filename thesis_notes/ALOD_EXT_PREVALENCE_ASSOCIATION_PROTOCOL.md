# ALoD-EXT natural-hierarchy association protocol

Protocol ID: `ALOD-EXT-PREVALENCE-ASSOCIATION-001`

Freeze date: 2026-09-23

Status: `FROZEN-BEFORE-NEW-EXT-RESULTS`

Classification: project-specific `ALOD-EXT`; not released ALoD and not the
paper's `Render Full`.

## Question and datasets

The primary descriptive comparison asks whether the already validated
ALoD-EXT execution paths retain a material execution advantage on the natural
Campus `VALID-NONTRIVIAL-LOD` hierarchy and how its semantic-to-materialization
opportunity differs from the Sci-Art `VALID-WEAK-LOD` hierarchy.

- `ALOD-EXT-PREVALENCE-CAMPUS-001` uses the immutable Campus TRAIN-001
  hierarchy and all 24 frozen held-out cameras in recorded order.
- `ALOD-EXT-PREVALENCE-SCIART-001` uses the immutable Sci-Art TRAIN-001
  hierarchy and all 21 frozen held-out cameras in recorded order.
- Run009 is historical `VALID-WEAK-LOD` context only. It is not rerun unless a
  later, separately controlled harmonization is required.

The three Stage-A labels remain unchanged. No Stage-B training or synthetic
control is part of this protocol.

## Frozen implementation and modes

All modes use the validated project HSPT and official semantic selector with
the dataset-specific frozen `SPT_root_volume`, effective evaluator
`min_SPT_size=100`, resolved two-pixel target granularity, frustum convention,
SH degree 1, packed gsplat backend, black background, and original held-out
resolution. ALoD-EXT runs only after the official semantic cut.

The frozen three-mode set is:

1. **Released Evaluator (No Cache):** host/no-cache materialization of every
   official semantic raster row; no pre-materialization filter.
2. **ALoD-EXT CUDA Previous Reuse:** the established conservative CUDA
   pre-materialization filter followed by exact previous-working-set property
   reuse. This is project extension behavior, not released viewer semantic-cut
   reuse.
3. **ALoD-EXT CUDA Bounded Cache:** the same conservative CUDA filter followed
   by the established deterministic row-LRU cache.

No selector, filter mathematics, hierarchy property, replacement policy, or
render setting may be retuned. The released viewer-style semantic-cut reuse is
not pooled with these modes.

## Frozen bounded-capacity rule

Before timing, correctness/count tracing obtains the complete unique union
`U` of filtered hierarchy rows over the canonical held-out trajectory. Let
`H` be the hierarchy-row count. The bounded capacity is

```text
min(H, ceil(U * 3,700,000 / 3,503,048)).
```

The rational factor transfers the exact Run009 practical-capacity-to-union
ratio (5.62% headroom before rounding) without choosing a Campus or Sci-Art
capacity from observed FPS. Capping at `H` avoids allocating nonexistent
rows. This rule guarantees union coverage when the correctness trace passes;
it is not a latency optimum and no capacity sweep is authorized.

## Compatibility amendment before correctness rerun

The initial Sci-Art correctness launch stopped at camera 0 before timing because
the generic benchmark camera reconstructed selector values from the public
COLMAP record. Stage-A official selection used the already frozen
`camera_center_float32` and `full_projection_transform` stored in each oracle
camera record. The benchmark camera container is therefore amended only to
forward those existing exact selector-camera values to the runtime field that
already supports them. Rendering pose, intrinsics, resolution, camera
membership and order remain the frozen effective official workload. The same
path is used for Campus and Sci-Art. This is a compatibility correction, not a
selector, filter, cache, threshold, or workload change. New implementation
hashes are sealed in `protocol_amendment_001.json` before the correctness
rerun.

## Correctness gate before timing

Every held-out camera must pass:

- exact official ordered semantic raster IDs and hierarchy IDs;
- zero conservative-filter false negatives relative to faithful packed
  projection;
- exact ordered CPU/CUDA conservative-filter IDs;
- bitwise retained Gaussian properties and order;
- exact faithful packed-projection survivor IDs;
- bitwise RGB and alpha.

A failed required barrier stops canonical performance execution for that
dataset. Numerical or instrumentation failures must be diagnosed rather than
relaxed.

## Timing and state

Each mode uses three fresh processes in a balanced cyclic schedule. A process
builds a fresh scene, HSPT, renderer, and empty dynamic residency state. It
renders the first ten canonical cameras as ordered warmup, retains resulting
cache/reuse state, then restarts the complete canonical trajectory at camera
zero for the measured window. Campus therefore measures 24 frames per
replicate and Sci-Art 21.

CUDA is synchronized around active-set preparation. Renderer GPU events and
the existing synchronized end-to-end frame measurement are retained. The
primary runtime is synchronized end-to-end frame time (selection, filtering,
property preparation/residency, assembly, and rasterization). Warmup is
excluded from primary timing but recorded for cold-inclusive dynamic-property
traffic. No-cache and extension modes use identical camera order, resolution,
renderer, and synchronization boundaries.

## Frozen measurements and interpretation

Per camera, replicate, mode, and dataset, retain semantic rows, materialized
rows, packed-projection survivors, selection/filter/preparation/residency/
rasterization/total times, H2D rows and dynamic-property bytes, and cache
counters. Report:

- materialized / semantic-selected rows;
- packed-projection survivors / semantic-selected rows;
- ALoD-EXT / Released Evaluator frame time;
- dynamic-property H2D-byte reduction;
- semantic-selection share of total frame time;
- useful absolute count, time, traffic, and FPS summaries.

Comparisons are direct within each dataset and descriptive across Campus,
Sci-Art, and compatible historical Run009 evidence. They do not establish a
population relationship, causal effect of weak/nontrivial LoD, semantic
quality improvement, or a paper `Render Full` result.
