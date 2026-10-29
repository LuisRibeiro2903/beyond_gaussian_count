# A LoD of Gaussians — First Oracle Dataset Selection

Generated: 2026-08-05

> **Formal-evaluation contamination warning:** This asset is derived from MatrixCity's original aerial test reconstruction. Because most of its cameras are repurposed for ALoD training, it must not be used as an untouched MatrixCity test set or compared directly with formal results that preserve the original MatrixCity train/test protocol.

## 1. Purpose

This task selects and freezes the first bounded dataset for the pinned official A LoD of Gaussians source at commit `73f547341547ae021ff036ad74fdf73a1240b450`.

The selected asset is intended for:

```text
official-source ALoD training smoke
+ initial and final .dhier generation/validation
+ format, hierarchy-selection, pre-rasterization, rendering, and cache-trace oracles
```

It is not a performance baseline. A later baseline will be a rendering configuration derived from the trained hierarchy, such as an official-reference, flat-leaf, or forced-finest configuration. It is also not the formal MatrixCity or formal LoD evaluation dataset and cannot support formal quality comparisons with the completed CityGaussian campaign.

No training, hierarchy construction, `.dhier` generation, rendering, compilation, CUDA work, or runtime integration occurred in this task.

## 2. Verified official-source dataset requirements

The pinned source was inspected at:

```text
/home/luis/tese/LoDOfGaussians
commit 73f547341547ae021ff036ad74fdf73a1240b450
```

`Scene.__init__` recognizes only a source directory containing `sparse/`. `readColmapSceneInfo` then attempts:

```text
sparse/0/cameras.bin
sparse/0/images.bin
sparse/0/points3D.bin
```

and falls back to the corresponding text files. The following behavior is verified directly from `scene/__init__.py`, `scene/dataset_readers.py`, `utils/camera_utils.py`, `scene/cameras.py`, and `utils/general_utils.py`:

- only `PINHOLE` and `SIMPLE_PINHOLE` camera models are accepted; distorted COLMAP models are rejected;
- registered image names are resolved under the configured image directory, with same-stem `.jpg` and `.png` fallbacks;
- images are opened by Pillow and converted to float CHW tensors by `PILtoTorch`;
- an RGBA image supplies RGB channels `0:3` and alpha channel `3`; RGB is multiplied by alpha without background compositing;
- with default `resolution=-1`, input widths above 1600 pixels are downscaled to 1600 pixels, so this dataset becomes 1600 x 900 at load time;
- masks are optional and otherwise searched under a sibling `rectified` layout;
- inverse-depth PNGs and `sparse/0/depth_params.json` are optional;
- `xyz.pt` plus `rgb.pt` are attempted first for sparse initialization; otherwise `points3D.ply` is loaded;
- if `points3D.ply` is absent, the reader converts `points3D.bin` or `points3D.txt` into a binary PLY in the dataset directory;
- the camera list is stably sorted by registered image name;
- the reader forces evaluation behavior internally; with `llff_hold > 0`, zero-based sorted indices satisfying `index % llff_hold == 0` are test views;
- `train_test_exp=false` excludes those views from training; and
- the source `test.txt` branch is not reachable under the audited conditional structure, so an arbitrary text split is not the reproduction path.

The shipped `small.json` uses a very large hold value and would retain only the first sorted camera. This freeze instead records `llff_hold=50` for a 15-camera official-compatible local oracle split. This is a future resource-adapted setting, not a training configuration defined by this task.

Full `Scene` initialization was not invoked because the pinned import graph initializes CUDA. The task followed the required fallback: the official low-level binary parser was used for all three COLMAP files, and the exact official CPU `PILtoTorch` function was used for representative image conversion.

## 3. Candidate inventory

The relevant local data roots `/home/luis/tese/data`, `/home/luis/tese/gsplat_data`, and the official dummy fixture were inspected without traversing unrelated vendor trees or hashing large rejected assets. Remote discovery was limited to official dataset/project sources.

Candidates investigated:

- MatrixCity Small City aerial `block_all_test` standard reconstruction;
- locally generated `aerial_subset_4` and `aerial_subset_2` reconstructions;
- full MatrixCity Small City aerial training reconstruction;
- the official ALoD dummy fixture;
- the official 3DGS Tanks & Temples/Deep Blending scene archive;
- Mip-NeRF 360;
- the frozen 23.4M fused CityGaussian Gaussian PLY, mentioned only as a rejected future extension candidate.

## 4. Candidate comparison

| Candidate | Source/license | Format and measured scale | Camera variation and storage | Conversion/risk | Decision |
|---|---|---|---|---|---|
| MatrixCity aerial `block_all_test` | Local MatrixCity images, CC BY-NC 4.0; CityGaussian released reconstruction, CC BY-NC-SA 4.0 | Complete standard COLMAP; 741 registered RGBA PNGs; one supported `PINHOLE`; 443,620 points; official low-level parser passed | 1920 x 1080; 3.03 GB images; centre-distance 2.25–10.65; height 1.5–5.0 | No camera conversion; external sparse PLY generation only; original test semantics become contaminated by local training | **SELECTED** |
| `aerial_subset_4` | Local derivative of the same source terms | 1,406 registrations and 1,385,150 points, but every image symlink is broken and 1,421,492 image-point references are dangling | Corrected image payload would be 5.63 GB; aerial-only | Requires a separate validated subset reconstruction; explicitly not repaired here | **REJECTED-MISSING-DATA** |
| `aerial_subset_2` | Local derivative of the same source terms | 2,811 registrations and 2,486,159 points, but every image symlink is broken and 1,121,075 image-point references are dangling | Larger than subset 4; aerial-only | Requires a separate validated subset reconstruction; explicitly not repaired here | **REJECTED-MISSING-DATA** |
| Full MatrixCity aerial train | Local MatrixCity/CityGaussian source terms | Complete standard COLMAP; 5,621 registrations; 3,826,641 points | 22.53 GB images; similar aerial height/distance range | Unnecessarily large for the first oracle | **REJECTED-SCALE**, deferred rather than intrinsically unsuitable |
| Official ALoD dummy | Pinned ALoD repository terms | One image and one sparse point | 14.7 KB image; no meaningful camera distribution | Cannot validate training, hierarchy, LoD, or oracle behavior | **REJECTED-MISSING-DATA** |
| Tanks & Temples/Deep Blending scenes | Official 3DGS 650 MB archive; Tanks & Temples terms permit research but restrict redistribution | Standard 3DGS/COLMAP ecosystem candidate; not locally acquired | Bounded remote fallback, but exact extracted scene size and metadata require acquisition validation | Official download, archive checksum, extracted size, and selected-scene loader check still required | **ACCEPTABLE REMOTE FALLBACK / NEEDS-ACQUISITION** |
| Mip-NeRF 360 | Official project distribution | Standard posed dataset, but no sufficiently explicit dataset-specific use/redistribution terms were established | Several scenes are practical in scale | Legal-use gate unresolved | **REJECTED-LICENSE** |
| Fused 23.4M CityGaussian PLY | Frozen CityGaussian result | Finished degree-3-SH Gaussian model, not an image/COLMAP training scene | 23,389,621 primitives; about 5.8 GB PLY | Incompatible with the fixed degree-1 coarse PLY contract and unsafe hierarchy scale | **REJECTED-AS-FIRST-ALOD-ORACLE** |

The full aerial training reconstruction may later support a formal, properly partitioned MatrixCity experiment, but only after a separate validated subset-construction task. It is not rejected as intrinsically invalid.

## 5. Selected dataset

```text
dataset_id: ALOD-ORACLE-MC-AERIAL-TEST-01
dataset_name: MatrixCity Small City aerial block_all_test reconstruction
external_path: /home/luis/tese/data/alod/ALOD-ORACLE-MC-AERIAL-TEST-01
```

Required provenance semantics:

```text
source_dataset: MatrixCity Small City
source_modality: aerial
source_partition: original MatrixCity/CityGaussian test reconstruction
source_registered_images: 741
project_use: repurposed for ALoD smoke training and oracle generation
original_test_semantics_are_not_preserved: true
```

The asset was selected because it is the smallest complete, legal, internally consistent, loader-compatible local reconstruction found. It requires no download or camera-format conversion, has enough initial geometry to plausibly grow into a bounded hierarchy, and has modest but useful camera-position/height variation.

The frozen external directory contains copied COLMAP binaries and a generated sparse PLY. Its `images` entry is an absolute symbolic link to:

```text
/home/luis/tese/data/matrix_city/aerial/test/block_all_test/input
```

Images were not duplicated because they occupy 3,030,193,206 bytes and remain third-party payload. The manifest records the absolute link text, resolved target, external dependency, and broken-link validation policy. All 741 images are individually hashed.

## 6. Rejected and deferred alternatives

The broken `aerial_subset_2` and `aerial_subset_4` assets were not repaired. Their broken absolute symlinks and inconsistent point references make them unsuitable as frozen standard COLMAP scenes in their current state.

The official dummy fixture is an import/parser diagnostic, not an oracle-training dataset. The full training reconstruction is deferred because its 5,621 images, 3.83M sparse points, and 22.53 GB image payload add time and resource risk without helping the first pipeline gate.

Tanks & Temples remains the preferred remote fallback if this local asset later proves unusable. No dataset was downloaded in this task. Mip-NeRF 360 remains rejected until dataset-specific legal terms can be established.

The approximately 23.4M fused CityGaussian MatrixCity PLY is classified:

```text
REJECTED-AS-FIRST-ALOD-ORACLE
```

It was not opened, copied, converted, cropped, benchmarked, or passed to an ALoD tool. It is rejected because:

- it is a finished Gaussian model rather than an image/COLMAP training scene;
- it stores degree-3 SH while the audited initial hierarchy creator expects its fixed degree-1 coarse PLY layout;
- binary hierarchy construction would approximately double the node count before fine-stage growth;
- that scale exceeds the current safe host-memory envelope; and
- using it would mix completed CityGaussian training with an ALoD project adaptation.

It may be reconsidered only as a separately scoped future extension, never as this official training oracle.

## 7. Frozen ALoD local oracle split

The split is named:

```text
ALoD local oracle split
```

It exactly reproduces `scene.dataset_readers.readColmapSceneInfo` at commit `73f547341547ae021ff036ad74fdf73a1240b450`:

```text
registered image records
-> stable sort using image_name as the official key
-> zero-based sorted index
-> eval=true
-> llff_hold=50
-> evaluation when index % 50 == 0
-> training otherwise, with train_test_exp=false
```

Counts are:

```text
training:   726
evaluation: 15
```

Evaluation registered image IDs:

```text
1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 701
```

The complete sorted identity order, exact 726 training IDs, exact 15 evaluation IDs, and every image name are stored in `dataset-manifest.json` and `registered-images.json`. The text lists preserve zero-based sorted index, registered image ID, camera ID, and name.

This is not the original MatrixCity train/test split. It repurposes 726 cameras from the original aerial test reconstruction for ALoD training. A hierarchy trained on it cannot be reported as generalizing to the untouched MatrixCity test partition.

## 8. Camera and image statistics

All statistics below are measured from the frozen COLMAP model. The robust scene centre is the coordinate-wise median of the 443,620 sparse points.

| Quantity | Minimum | Median | p90 | Maximum |
|---|---:|---:|---:|---:|
| Camera-to-centre distance | 2.253 | 5.903 | 8.210 | 10.648 |
| Nearest-camera distance | 0.141 | 0.400 | 0.414 | 0.894 |
| Camera height | 1.500 | 2.500 | 4.500 | 5.000 |
| Focal x/y in pixels | 2317.645 | 2317.645 | 2317.645 | 2317.645 |
| Image bytes | 2,847,755 | 4,142,106 | 4,464,624 | 4,741,666 |

Camera-centre bounds and extent:

```text
min:    [-9.5000, -7.1000, 1.5000]
max:    [ 7.2000,  5.9000, 5.0000]
extent: [16.7000, 13.0000, 3.5000]
```

Every image is an RGBA PNG at 1920 x 1080. The scene has one `PINHOLE` camera and a fixed focal length. Camera rotations and world optical axes are stored per record and summarized in `camera-statistics.json`. The evaluation cameras cover low, middle, and high altitude bands plus central and boundary positions, but there is no focal-length variation.

Stable identity checks passed with zero duplicate registered image IDs, filenames, name/ID sort keys, resolved files, or case-folded path collisions. COLMAP consistency checks passed with zero dangling image-point references, missing track images, invalid point indices, or point/track identity mismatches.

CPU-only image-loader compatibility also passed:

- all 741 PNGs opened and passed integrity verification;
- all 15 evaluation images plus early, middle, and late representatives fully decoded;
- the exact official `PILtoTorch` path produced float32 RGBA CHW tensors;
- default resizing produced `4 x 900 x 1600` tensors;
- RGB and alpha extraction matched the official `Camera` logic;
- RGB was multiplied by alpha without compositing; and
- no CUDA context was initialized.

## 9. Resource assessment

These are evidence-labelled planning estimates, not measured training memory or runtime.

- **Measured external input:** 3.03 GB linked image payload, 216.6 MB copied COLMAP binaries, and an 11.98 MB generated sparse PLY.
- **Estimated image working data:** one default-resolution 1600 x 900 RGBA float32 tensor is about 23.0 MB. Eight one-sample prefetched workers imply roughly 184 MB of raw image tensor values, before Python, Pillow, worker, RGB/alpha, and page-cache overhead. The loader is lazy; it does not intentionally materialize all 3.03 GB of compressed images as tensors at once.
- **Measured initialization geometry:** 443,620 sparse points. With the source default of 100,000 skybox points, the initial source population is expected to be roughly 543,620 primitives before coarse-stage evolution.
- **Estimated hierarchy scale:** a binary hierarchy around a roughly 0.5M-leaf scaffold is naturally in the vicinity of one million nodes before later fine-stage growth. A conservative future cap around 1–2M final nodes is therefore plausible, but remains unmeasured and must be enforced by the separately approved training configuration.
- **Estimated final `.dhier`:** SH-degree-1 source-format arithmetic gives about 116–232 MB for 1–2M nodes.
- **Estimated host memory:** the paper-level planning rule is approximately 1–2 GB for 1–2M Gaussians before image workers, Python, hierarchy metadata, optimizer auxiliaries, and safety headroom. This is not a measured peak.
- **Estimated output storage:** allow several GB for scaffold PLYs, initial/final hierarchies, checkpoints, logs, and partial failure artifacts even though the final `.dhier` alone should be much smaller.
- **Measured free disk before freeze:** about 712 GB, so storage is not the limiting resource.

The dataset is credible for a later capped attempt, but this task deliberately does not define the final cap, cache target, abort thresholds, or training command.

## 10. Suitability and limitations

| Claim or task | Classification |
|---|---|
| Environment validation | Suitable |
| Training pipeline validation | Suitable |
| `.dhier` validation | Suitable |
| Hierarchy-selection oracle creation | Conditionally suitable |
| Official loader and pre-rasterization oracle work | Suitable |
| Cache behavior development | Suitable |
| Formal LoD evaluation | Insufficient |
| City-scale claims | Insufficient |
| Mixed-scale navigation claims | Insufficient |

The hierarchy-selection classification is conditional because non-trivial selected-depth/LoD variation cannot be known before a hierarchy exists. If all 15 cameras and future deterministic trajectories select essentially the same depth, this dataset remains valid for pipeline, format, loader, and cache-development work, but a second mixed-scale oracle must be selected.

Specific limitations:

```text
single PINHOLE camera
fixed focal length
aerial-only imagery
camera heights limited to approximately 1.5–5.0 scene units
camera-to-centre distances limited to approximately 2.25–10.65 units
no street-level approach/retreat sequence
no large aerial-to-ground transition
```

Successful completion of this freeze does not remove the need for a later formal dataset. A later asset must provide at least one of:

```text
proper original training/test separation
mixed aerial and street-level views
materially broader camera-distance variation
an untouched quality-evaluation set
```

That selection is a separate task after the first `.dhier` and official/runtime oracle path exist.

## 11. Provenance and reproducibility

External payload:

```text
/home/luis/tese/data/alod/ALOD-ORACLE-MC-AERIAL-TEST-01
```

Tracked metadata:

```text
datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/dataset-manifest.json
datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/registered-images.json
datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/camera-statistics.json
datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/evaluation-cameras.json
datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/train-images.txt
datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/evaluation-images.txt
datasets/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/README.md
```

The manifest records source URLs and licenses, local source and frozen paths, acquisition date, file counts, relative model paths, complete registered identities, split order and assignments, camera records, image dimensions and bytes, per-image SHA-256 hashes, COLMAP/PLY hashes, excluded payloads, and the exact generation command.

Frozen model hashes:

| File | SHA-256 |
|---|---|
| `cameras.bin` | `c16791c2d5e7e4bca7e6a7165438fd3ec9906276799d7720bf3d027306e533bc` |
| `images.bin` | `78847ca139b896f44ea6e6193b732072ca9702786bab1829a1b8d5c9f03e6cdf` |
| `points3D.bin` | `7adbc31c2dbb6303153b18e6359e8d26fe253f150eaaf0fc334c7c13a909cf18` |
| `points3D.ply` | `db177e4b1e5ed20b3f58ff6e3afe24c5d808f7aec2798b48304d64d56929b5fa` |

The copied binary hashes exactly match the source reconstruction. Repeated sparse-Ply generation and complete metadata regeneration were byte-identical. No archive was present; the manifest records the official sources and complete local content identity instead.

No image, COLMAP binary, point cloud, checkpoint, Gaussian PLY, `.dhier`, archive, render, or training output entered Git.

## 12. Decision

The selected dataset is legally usable for non-commercial research; has reproducible source and complete hashes; supplies the official loader's supported COLMAP metadata; contains valid registered images and sparse points; fits the intended bounded next task; has an exact source-compatible split and fixed evaluation cameras; requires no unresolved camera conversion; and was frozen without modifying CityGaussian, the official checkout, environments, runtime, tests, or benchmarks.

```text
PASS-ALOD-ORACLE-DATASET
```
