# ALOD-ORACLE-MC-AERIAL-TEST-01

> **Formal-evaluation contamination warning:** This asset is derived from MatrixCity's original aerial test reconstruction. Because most of its cameras are repurposed for ALoD training, it must not be used as an untouched MatrixCity test set or compared directly with formal results that preserve the original MatrixCity train/test protocol.

## Role

This is the first bounded official-source A LoD of Gaussians training-smoke and oracle dataset. It is intended to support:

```text
official coarse-training smoke
-> initial hierarchy generation
-> capped fine-hierarchy training
-> final .dhier validation
-> official selection, tensor, rendering, and cache-trace oracles
```

It is not a performance baseline and is not the formal MatrixCity or formal LoD evaluation dataset. Later baselines will be rendering configurations produced from the trained hierarchy, such as official-reference, flat-leaf, or forced-finest configurations.

## Provenance and changed partition semantics

```text
source_dataset: MatrixCity Small City
source_modality: aerial
source_partition: original MatrixCity/CityGaussian test reconstruction
source_registered_images: 741
project_use: repurposed for ALoD smoke training and oracle generation
original_test_semantics_are_not_preserved: true
```

The MatrixCity images are published under CC BY-NC 4.0. The released CityGaussian COLMAP reconstruction is published under CC BY-NC-SA 4.0. Use is restricted here to non-commercial research with attribution, and reconstruction-derived redistribution must respect ShareAlike. Repository metadata contains hashes and regeneration instructions; it does not contain third-party images or COLMAP payloads.

Sources:

- MatrixCity: <https://city-super.github.io/matrixcity/>
- MatrixCity data: <https://huggingface.co/datasets/BoDai/MatrixCity>
- CityGaussian: <https://github.com/Linketic/CityGaussian>
- CityGaussian released COLMAP results: <https://huggingface.co/datasets/dylanebert/CityGaussian>

## External asset

The payload is external to Git:

```text
/home/luis/tese/data/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/
├── images -> /home/luis/tese/data/matrix_city/aerial/test/block_all_test/input
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    └── points3D.ply
```

The three COLMAP binaries are copied so that later official-loader side effects cannot alter the source dataset. `images` is an absolute symbolic link; the 3,030,193,206-byte image set is not duplicated. The frozen asset therefore depends on the original external MatrixCity image path. Validation resolves the link, requires every registered image to exist, hashes all 741 files, verifies every PNG, and fully decodes the required representative loader-check set.

`points3D.ply` is a deterministic CPU-only conversion of `points3D.bin` matching `scene.dataset_readers.storePly` at official ALoD commit `73f547341547ae021ff036ad74fdf73a1240b450`. It contains 443,620 vertices with `x/y/z` and zero `nx/ny/nz` as float32 plus `red/green/blue` as uint8. Repeated generation produced the identical SHA-256:

```text
db177e4b1e5ed20b3f58ff6e3afe24c5d808f7aec2798b48304d64d56929b5fa
```

## Frozen ALoD local oracle split

The split reproduces the pinned official `readColmapSceneInfo` behavior exactly:

1. read registered records from `sparse/0/images.bin`;
2. stably sort by registered image name;
3. use zero-based sorted indices;
4. with resolved `eval=true` and `llff_hold=50`, assign an image to evaluation when `index % 50 == 0`;
5. assign every remaining image to training with `train_test_exp=false`.

Counts:

```text
training:   726
evaluation: 15
```

Evaluation registered image IDs are:

```text
1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 701
```

These are a new **ALoD local oracle split**, not the original MatrixCity train/test split. In particular, 726 cameras from the original MatrixCity/CityGaussian test reconstruction are now designated for local ALoD training. Any hierarchy trained from this asset has therefore seen most of the original test-partition cameras.

Stable identity is the tuple of COLMAP registered image ID, COLMAP camera ID, and relative image name. Every record also preserves pose, camera centre, intrinsics, stable sorted index, split assignment, file size, and SHA-256. Filenames alone are not treated as oracle identities.

## Measured dataset properties

| Property | Value |
|---|---:|
| Registered images | 741 |
| Sparse points | 443,620 |
| Camera model | one `PINHOLE` camera |
| Image format | 741 RGBA PNG files |
| Resolution | 1920 x 1080 for every image |
| Focal length | 2317.644948 px, fixed |
| Total image bytes | 3,030,193,206 |
| Median image bytes | 4,142,106 |
| Maximum image bytes | 4,741,666 |
| Camera-centre extent | 16.70 x 13.00 x 3.50 scene units |
| Camera height | 1.50–5.00 scene units |
| Camera-to-centre distance | min 2.25, median 5.90, p90 8.21, max 10.65 |
| Nearest-camera distance | min 0.141, median 0.400, p90 0.414, max 0.894 |

The robust scene centre is the coordinate-wise median of the sparse COLMAP points.

## CPU-only official image-path check

Compatibility passed without CUDA, rendering, or full `Scene` initialization:

- every image opened and passed Pillow PNG integrity verification;
- all images are RGBA at 1920 x 1080;
- the pinned official `utils.general_utils.PILtoTorch` converted all 15 evaluation images plus early, middle, and late representatives;
- the default `resolution=-1` path resizes 1920 x 1080 inputs to 1600 x 900 using LANCZOS;
- tensors are float32 CHW with shape `4 x 900 x 1600`;
- channels `0:3` become RGB and channel `3` becomes the alpha mask;
- `Camera` multiplies RGB by alpha without background compositing; and
- representative masked RGB tensors were finite and remained in `[0,1]`.

Full official `Scene` initialization was intentionally not invoked because the pinned source import graph initializes CUDA. The official low-level COLMAP parser and the exact CPU image conversion function were used instead, as required by the no-CUDA validation rule.

## Allowed uses

- official ALoD coarse-training smoke;
- initial hierarchy generation;
- capped fine-hierarchy training;
- `.dhier` format validation;
- official loader validation;
- source-selected-set oracle extraction;
- pre-rasterization tensor parity;
- initial same-asset runtime integration; and
- deterministic cache-trace development.

## Prohibited uses

- formal MatrixCity train/test quality reporting;
- direct PSNR/SSIM/LPIPS comparison with existing CityGaussian formal results;
- claims that the model was trained on the official MatrixCity training split;
- city-scale scalability claims;
- final aerial-versus-street LoD conclusions; and
- generalization claims on held-out MatrixCity test images.

## Suitability and limitations

| Use | Suitability |
|---|---|
| Training pipeline validation | Suitable |
| `.dhier` validation | Suitable |
| Hierarchy-selection oracle creation | Conditionally suitable |
| Cache behavior development | Suitable |
| Formal LoD evaluation | Insufficient |
| City-scale claims | Insufficient |
| Mixed-scale navigation claims | Insufficient |

The hierarchy-selection oracle remains conditional until a trained hierarchy demonstrates non-trivial depth/LoD variation over the 15 cameras and later deterministic trajectories. The scene uses one fixed-focal `PINHOLE` camera, aerial-only imagery, heights of about 1.5–5.0 units, and camera-to-centre distances of about 2.25–10.65 units. It has no street-level approach/retreat sequence and no large aerial-to-ground transition. If official selection stays at essentially one hierarchy depth, retain this asset for pipeline validation and select a second mixed-scale oracle.

A later formal dataset is still required. It must provide at least one of proper original training/test separation, mixed aerial/street views, materially broader camera-distance variation, or an untouched quality-evaluation set.

## Metadata and regeneration

Repository metadata files are:

- `dataset-manifest.json`: provenance, licenses, use restrictions, hashes, split and generation command;
- `registered-images.json`: all 741 stable identities, cameras, poses, images, and image hashes;
- `camera-statistics.json`: scene, camera, focal, orientation, resolution, and loader statistics;
- `evaluation-cameras.json`: complete 15-camera oracle records and selection categories;
- `train-images.txt`: exact 726 training identities; and
- `evaluation-images.txt`: exact 15 evaluation identities.

Regenerate them with the `manifest_generation_command` stored in `dataset-manifest.json`. The generator reads the source dataset but writes only to the approved external ALoD directory and repository metadata directory. It imports neither CityGaussian nor the ALoD renderer/trainer.
