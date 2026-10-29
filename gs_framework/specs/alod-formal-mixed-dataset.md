# Formal mixed-scale ALoD reproduction dataset

Frozen: 2026-08-20  
Dataset: `ALOD-REPRO-MC-MIXED-01`  
Status: `PASS-ALOD-FORMAL-DATASET` and
`PASS-ALOD-MIXED-SCALE-CAMERA-COVERAGE`

## Decision and boundary

The frozen asset is a **mixed-scale MatrixCity proxy for faithful
public-source reproduction; not the unreleased exact ALoD paper dataset**.
The reproduction-boundary audit established that the paper's additional
cameras, crops, lists, seeds, and trained assets are not all public. This
dataset therefore freezes a reproducible public proxy and makes no exact-data
claim.

The external root is
`/home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01`. Project metadata is in
`datasets/alod/ALOD-REPRO-MC-MIXED-01`. Image payload stays outside Git.

## Source audit and coordinate evidence

| Modality | Public identity and local source | Images used | Resolution / intrinsics | Pose and frame |
|---|---|---:|---|---|
| Aerial | `BoDai/MatrixCity`, Small City aerial `block_all`; `/home/luis/tese/data/matrix_city/aerial` | 335 of 6,362 registered views | 1920x1080 PINHOLE, `fx=fy=2317.644948`, principal point at image centre | Released NeRF c2w; MatrixCity normalized world, 100 m/unit |
| Street | `BoDai/MatrixCity`, `aerial_street_fusion/street.tar`; preserved under the external dataset `source/` | all 330 train + 45 test | 1000x1000 PINHOLE, `fx=fy=499.999978`, principal point at image centre | Released fusion NeRF c2w in the same normalized world |
| Sparse geometry | Public MatrixCity aerial COLMAP `points3D.bin` | 2,317,128 cropped points | XYZ/RGB/error unchanged; obsolete source tracks removed | Same aerial COLMAP frame |

MatrixCity data is CC BY-NC 4.0. The source archive URL and all source hashes
are frozen in `source_provenance.json`.

No manual or visual alignment was used. MatrixCity's released
`merge_aerial_street.py` concatenates its normalized aerial and street c2w
poses without a transform. Independently, all 335 selected aerial centres were
matched to the released aerial COLMAP registrations after the exact
NeRF-to-COLMAP axis conversion. The maximum centre discrepancy is
`6.712405e-7` normalized units against a `1e-5` threshold. The alignment
transform is therefore identity.

## Evidence-driven selection

All 375 available street views are retained because they define the close
regime and are only 52.8% of the final cameras. Aerial cameras are restricted
to a reproducible overlap envelope around the street trajectory. The builder
tests multiples of the street XY footprint diagonal and accepts the smallest
radius with at least three times the requested train/eval headroom and both
low (`<=2`) and high (`>=5`) aerial heights. The first accepted multiplier is
2.0, radius `4.767986768` normalized units (about 477 m).

Within this candidate pool, train and evaluation aerial views are selected
independently. Each is divided into equal near/mid/far 3D-distance strata,
then deterministic farthest-point sampling is performed in standardized
position plus view-direction space. This yields 309 aerial train and 26 aerial
evaluation views. Selection does not use images, renders, quality, or later
model behavior.

The sparse geometry is cropped by the selected-camera XY envelope with a
deterministic padding derived by the builder. It reduces 3,826,641 public
source points to 2,317,128 while preserving point values. Image observations
and point tracks are empty because the frozen camera set is constructed from
released poses rather than the original registration identities; this avoids
dangling references without fabricating feature correspondences.

## Frozen split and identities

The 710 filenames begin with a zero-padded stable dataset index. The 71 source
test views occupy indices `0, 10, ..., 700`; the 639 source train views occupy
all other indices. This realizes the pinned official loader's lexicographic
`llff_hold=10` behavior exactly while preserving released source partitions.

| Count | Value |
|---|---:|
| Total | 710 |
| Aerial / street | 335 / 375 |
| Train / evaluation | 639 / 71 |
| Aerial train / evaluation | 309 / 26 |
| Street train / evaluation | 330 / 45 |
| Near / mid / far aerial | 112 / 112 / 111 |
| Central / boundary | 473 / 237 |

`split.json` is the authoritative per-camera record: source identity, source
and dataset filenames, image SHA-256, registered dataset image ID, intrinsic
identity, c2w, COLMAP w2c, camera centre, modality, stratum, region, and split.
The COLMAP model contains two PINHOLE camera records and 710 registered images.

## Quantified mixed-scale coverage

Distances are in MatrixCity normalized units. Camera height is relative to the
sampled sparse-geometry fifth-percentile ground proxy.

| Statistic | Aerial min / median / p90 / max | Street min / median / p90 / max |
|---|---|---|
| Camera-to-scene-centre distance | 2.191 / 3.989 / 5.196 / 6.846 | 0.00177 / 0.430 / 0.949 / 1.095 |
| Camera height | 1.494 / 2.494 / 2.494 / 4.994 | 0.0239 / 0.0239 / 0.0239 / 0.0239 |
| Nearest-scene-distance proxy | 0.196 / 0.896 / 1.536 / 2.188 | 0.0195 / 0.0354 / 0.0693 / 0.0868 |
| Projected scene-scale proxy | 338.5 / 580.9 / 786.9 / 1057.8 | 456.6 / 1163.4 / 6086.5 / 282093.7 |
| Focal length, pixels | 2317.64 throughout | 500.00 throughout |

The ranges, multiple aerial heights, broad yaw/pitch support, close street
trajectory, and near/mid/far aerial quotas establish materially stronger
camera-scale variation than the aerial-only `WEAK-LOD-ORACLE`. This is only a
dataset-level coverage result; `NONTRIVIAL-LOD-ORACLE` remains unavailable
until a trained hierarchy is evaluated by the actual selector.

## Diagnostic evaluation set

The 16 identities are frozen before training:

```text
000060__aerial__test__0049.png  far / boundary
000090__aerial__test__0072.png  near / central
000100__aerial__test__0084.png  mid / boundary
000130__aerial__test__0129.png  near / central
000150__aerial__test__0000.png  far / boundary
000220__aerial__test__0098.png  near / central
000240__aerial__test__0124.png  mid / central
000250__aerial__test__0147.png  far / boundary
000320__street__test__0006.png  boundary
000350__street__test__0009.png  central
000370__street__test__0011.png  boundary
000480__street__test__0022.png  boundary
000500__street__test__0024.png  boundary
000520__street__test__0026.png  central
000580__street__test__0032.png  central
000650__street__test__0039.png  central
```

Coverage is exactly aerial near/mid/far = 3/2/3 and street central/boundary =
4/4. Full pose and metric records are in `diagnostic_cameras.json`.

## Official-ingestion acceptance

The CPU-only validator called the real pinned
`scene.dataset_readers.readColmapSceneInfo` and `utils.camera_utils.loadCam` at
commit `73f547341547ae021ff036ad74fdf73a1240b450`. The pinned import allocates a
CUDA relocation table at module import, so validation installed an in-memory,
fail-closed `utils.reloc_utils.compute_relocation_cuda` stub. Relocation is not
reachable in ingestion; no official file was changed.

Acceptance results:

- 639 train and 71 evaluation cameras, exactly;
- 2 camera records, 710 image records, and 2,317,128 points;
- all images resolve and representative aerial/street train/eval tensors are
  finite float32;
- resolution factor 2 produces 960x540 aerial and 500x500 street tensors;
- finite poses, no duplicate filenames/IDs/source identities, and no dangling
  references;
- manifest-to-official camera-centre maximum error `6.922367e-15`;
- CUDA remained uninitialized.

The report status is `PASS-ALOD-FORMAL-DATASET`.

## Frozen hashes and reconstruction

Manifest SHA-256:
`eaf3cb228234efd92441d28ea56482dcf4676c5a318a6d6beef1020ee27c5f7a`.

COLMAP SHA-256 values:

```text
cameras.bin  55a1ccae4f8c2cea30535b1323b036c09b27182c8bc56495b132932464e50966
images.bin   4a2079ee1a2fa2a10922287904962dc0b5ff40af92fe264228bfd4dc1db52d35
points3D.bin cbd4affef7138370ddbbd4c5b3dccb221d317fe7bd112c625cbb0cf612b02791
points3D.ply fc77affd4849e469e32071476bf475934e546aaba614e171190dea91451631ef
```

Use the exact `construction_command` in `manifest.json`. The builder is
deterministic and refuses any non-empty output directories; the frozen root
should now be treated as immutable.
