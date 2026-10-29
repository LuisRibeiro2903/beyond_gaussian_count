# ALoD CPU Reference HSPT and Selection Parity

Date: 2026-08-08

Status: `PASS-ALOD-CPU-REFERENCE-SELECTION`

## Scope

This result covers the first project-owned CPU correctness reference for ALoD
HSPT construction, four-plane upper traversal, LoD decisions, SPT cutting, and
global-row assembly. It implements no rendering, gsplat integration, CUDA,
official extension call, residency, cache, streaming, viewer, or production
runtime behavior.

The core implementation is isolated under `alod_reference/`. It consumes only
project-owned `alod_dhier` parser output plus explicit configuration. Oracle
payloads are opened only after the project HSPT has been constructed.

## Asset

| Field | Value |
|---|---|
| Hierarchy | Run005 `final.dhier` |
| SHA-256 | `6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c` |
| Serialized rows | 1,099,999 |
| Skybox rows | 100,000 (`0..99,999`) |
| Hierarchy root | `global_node_id=100000` |
| Hierarchy rows | 999,999 |

The stable identity is always the serialized `.dhier` row index.

## Oracle

| Field | Value |
|---|---|
| Oracle | `ALOD-RUN005-SELECTION-ORACLE-001` |
| Deterministic semantic manifest SHA-256 | `20bb87caf1694ede18ac2d2a71cdc146c002325ddbf8f6060625f18feda70591` |
| Official source commit | `73f547341547ae021ff036ad74fdf73a1240b450` |
| Frozen cameras | Registered image IDs `1, 101, 201, 301, 351, 401, 451, 501, 601, 701` |

## Configuration

```text
root_global_node_id = 100000
skybox_count = 100000
SPT_root_volume = 150
min_SPT_size = 100
target_granularity = 0.0008629449482830516
distance_multiplier = 1
effective_resolution = 960 x 540
frustum_planes = left, right, bottom, top
near/far planes = disabled
bounding_sphere mode = disabled
bounds = 3 * exp(max(log_scale_components))
```

## HSPT parity

The project derives the Run005 counts; they are not constants in the builder.

| Structure | Project | Oracle | Exact |
|---|---:|---:|---|
| Upper hierarchy rows | 1,309 | 1,309 | Yes |
| SPT count | 23 | 23 | Yes |
| Flattened SPT rows | 998,713 | 998,713 | Yes |
| Total HSPT metadata rows | 1,000,022 | 1,000,022 | Yes |

Selected raw-array hashes:

| Array | Project/oracle raw SHA-256 |
|---|---|
| `upper_tree_nodes` | `6a8b82043cb012ea148c75f6c5f55ff2594b61f4a5b54a615e19f65bed6292e0` |
| `upper_to_global_node_id` | `5f364afaaffc7054dab7214bd9ad019b52d80a8e5229ce782cc8255f5f15f1a2` |
| `upper_min_distance_squared` | `9fba6f3a071004749bb502b9c22fca3d1e52a5487438132cef237eed3c5d78ab` |
| `spt_starts` | `c0da2eed26d2c09c4b07c3979e4e9831b54829d7103b941009e7af6d826cd338` |
| `spt_min` | `8842d3e96b9181ba8837c430474e8db27036b0d4172db4b59b45e04ad6fce9f8` |
| `spt_max` | `c81aceed7821cfcd2ccc7bd99b5ad074bf4a1d19c72a8946aeae0ff2d7548a45` |
| `spt_to_global_node_id` | `7de0b5581154b03a24e75331db9fa035ecf56e5ceaa18568f45a5c8f58e54c6e` |
| `spt_root_global_node_ids` | `0193ccb725fef4de20b6a8e5ce567ff6ac468fb3f914f2464e13643e1e4544e6` |
| `spt_sizes` | `24654d14e688394445339cc29fddb0bb8ffcd4beed369672245e54488856f08b` |

All remaining exported HSPT arrays (`upper_tree_xyz` and
`upper_tree_log_scaling`) also match in shape, dtype, order, values, and raw
hash. Upper/SPT mappings retain the oracle's exact overlap-at-SPT-roots and
complete hierarchy-row coverage.

Stable descending `SPT_max` ordering is implemented with an explicitly stable
sort, preserving breadth-first source order for equal keys. It does not depend
on an unspecified default tie rule.

## Camera parity

Every `Yes` below means exact ordered array equality, not count-only or set-only
agreement. `Upper` includes the four-plane result, coarse cut, emitted upper
rows, candidate SPT IDs/roots/order, and candidate distances. `SPT` includes
each individual ordered SPT cut and their official concatenation order.

| Image ID | Candidate SPTs | Selected hierarchy | Raster rows | Upper | Per-SPT | Ordered IDs | Sorted IDs | Raster order |
|---:|---:|---:|---:|---|---|---|---|---|
| 1 | 22 | 364,357 | 464,357 | Yes | Yes | Yes | Yes | Yes |
| 101 | 23 | 498,419 | 598,419 | Yes | Yes | Yes | Yes | Yes |
| 201 | 23 | 496,102 | 596,102 | Yes | Yes | Yes | Yes | Yes |
| 301 | 22 | 486,669 | 586,669 | Yes | Yes | Yes | Yes | Yes |
| 351 | 22 | 498,032 | 598,032 | Yes | Yes | Yes | Yes | Yes |
| 401 | 22 | 494,619 | 594,619 | Yes | Yes | Yes | Yes | Yes |
| 451 | 22 | 497,768 | 597,768 | Yes | Yes | Yes | Yes | Yes |
| 501 | 23 | 499,030 | 599,030 | Yes | Yes | Yes | Yes | Yes |
| 601 | 23 | 491,315 | 591,315 | Yes | Yes | Yes | Yes | Yes |
| 701 | 22 | 496,209 | 596,209 | Yes | Yes | Yes | Yes | Yes |

The ordered and sorted selected-ID hashes for every camera equal the published
oracle hashes in `docs/results/alod_run005_selection_oracle.md`. Camera 1 also
has a compact regression fixture for its exact 364,158-node ordered SPT output:
raw SHA-256
`007f60abb4fd75957fa21775c950acab0d7bf0bddc52a1d09a0d49c6bdda427c`.

For every camera, the final reference raster identity is exactly:

```text
int64 range(0, 100000)
+ ordered selected hierarchy global_node_ids
```

and equals the oracle raster-row payload element-for-element.

## Source-sensitive semantics

### Mathematical algorithm parity

The project independently implements the same breadth-first HSPT decomposition,
volume cut, parent-derived minimum distances, stable SPT ordering, four-plane
sphere rejection, strict upper LoD predicate, native SPT interval search,
exclusive-prefix population behavior, nonzero compaction, and upper-then-SPT
assembly order.

### Float32 operation-order parity

Topology and identity remain NumPy arrays. CPU PyTorch is narrowly used for the
HSPT float32 expressions that the official code also evaluates on CPU:
activated scales, volume products, surface-area distance thresholds, center
distances, capping, and squaring. This is required for bit parity: a NumPy-only
calculation differs on 428 of 1,309 Run005 upper thresholds, while CPU PyTorch
matches the complete oracle threshold hash exactly.

The official per-camera path evaluates three-component reductions on CUDA. The
reference explicitly reproduces the observed float32 reduction order
`(x + z) + y` for plane normalization, camera distances, and signed plane
tests. This produces bit-exact frustum planes and candidate-SPT distances for
all ten cameras. No epsilon is used.

The scale-derived bound expression itself runs through CPU PyTorch rather than
CUDA `exp`; its last-bit representation is not claimed as a CUDA tensor-byte
replica. The oracle does not provide that bound payload for element-wise
comparison. Crucially, every resulting four-plane visibility decision, rejected
branch effect, upper cut, candidate SPT, and final discrete selection is exact.

### Discrete selection parity

The native cut's unusual behavior is preserved literally rather than cleaned
up: `high - start - 1`, minimum interval size one, exclusive-prefix boundaries,
the first-candidate offset-zero test, subsequent offset-one starts, strict
`SPT_min < camera_distance`, and zero-ID compaction. Per-SPT equality is checked
before combined selection equality.

## Performance

Diagnostic CPU-only measurement on the local machine; no optimization was
performed. The process built one HSPT and selected all ten cameras without
loading expected selection payloads.

| Image ID | Upper traversal (ms) | SPT cut (ms) | Total selection (ms) |
|---:|---:|---:|---:|
| 1 | 1.070 | 3.819 | 5.917 |
| 101 | 1.058 | 8.570 | 14.359 |
| 201 | 1.048 | 5.137 | 11.508 |
| 301 | 1.065 | 4.916 | 6.184 |
| 351 | 1.131 | 5.112 | 6.458 |
| 401 | 1.082 | 5.234 | 7.097 |
| 451 | 1.033 | 5.490 | 6.748 |
| 501 | 1.032 | 5.060 | 6.287 |
| 601 | 1.128 | 4.905 | 6.220 |
| 701 | 1.029 | 5.009 | 6.234 |

- HSPT build wall time: approximately `0.214 s`.
- OS-observed process peak RSS: `739,616 KiB` (approximately `722.3 MiB`).
- Timings are engineering diagnostics, not a production performance claim.

## Validation

Synthetic unit tests cover single leaves, two-leaf roots, multiple depths,
small and multiple SPTs, equal-key stable ordering, all four frustum exits,
plane contact, subtree rejection, near/far cuts, strict `< / == / >` LoD
boundaries, float64/NumPy-float32/CPU-PyTorch divergence, literal native
interval behavior, and CUDA-use rejection.

Validation commands passed:

```text
python -m unittest unit_tests.test_alod_reference_hspt \
  unit_tests.test_alod_reference_frustum \
  unit_tests.test_alod_reference_selection

GS_FRAMEWORK_RUN_ALOD_REFERENCE_INTEGRATION=1 \
  python -m unittest integration_tests.test_alod_reference_selection_external

python -m unittest discover -s unit_tests
```

The focused synthetic run passed 14 tests. The external Run005 gate passed its
two tests, including the camera-1 ordering regression. Full unit discovery
passed 164 tests with one existing CUDA-conditional skip.

## Result

`PASS-ALOD-CPU-REFERENCE-SELECTION`

The pinned official ALoD evaluation implementation, the frozen Run005 selection
oracle, and the independent project CPU reference agree on the exact serialized
hierarchy nodes and exact raster-row order for all ten frozen cameras. This is
a stronger identity/order result than visual or rendered-image parity and is
the correctness authority for a future optimized ALoD runtime.

The next bounded task is to design a fully resident production ALoD HSPT/LoD
runtime against this reference before adding host streaming or cache behavior.
