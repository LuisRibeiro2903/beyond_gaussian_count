# Formal Mixed-Scale ALoD Selection Oracle

Date: 2026-08-22

Status: `PASS-ALOD-FORMAL-SELECTION-ORACLE`

LoD classification: `WEAK-LOD-ORACLE`

## Result

The pinned official ALoD cold-evaluation path was executed twice in fresh
processes for all 16 diagnostic cameras frozen before formal training. Exact
ordered selection, raster identity, pre-raster tensors, and render endpoint
hashes were deterministic. The project CPU reference selector then reproduced
the official HSPT and every selection layer exactly for all 16 cameras.

The formal mixed-scale asset does not meet the predeclared `NONTRIVIAL` LoD
criteria. It has meaningful selected-set variation, but count variation and
depth-distribution variation remain below their frozen thresholds. Street
cameras select more nodes than aerial cameras on average, but do not select a
systematically finer/deeper cut in the depth-distribution sense.

No FPS benchmark, paper-style ablation, PSNR, SSIM, or LPIPS evaluation was
performed.

## Frozen artifacts

The external oracle is:

```text
/home/luis/tese/runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001
```

Its deterministic semantic manifest SHA-256 is:

```text
98615f6d2fabeaa33f67ce3dfd0cb1dae04b1f3fb5764a79ae83356c2c5262e7
```

The detailed formal analysis SHA-256 is:

```text
282af5da9bcc32d8fd6436ed5b2cd7fad513c40a8488c021e8eb6cbf6a47fdd4
```

The compact tracked locator is
`unit_tests/fixtures/alod_selection_oracle/formal-mixed-run009-selection-oracle-v1.json`
with SHA-256
`0395463e455e7d613f77e31db0b0d93fb9fac9a17c72e868e8e590d957899c6b`.
Large ordered ID arrays and HSPT arrays remain external. Every NumPy payload
has a file hash and a raw canonical-array hash checked by the project oracle
reader.

## Asset and source identity

The unchanged formal hierarchy is:

```text
/home/luis/tese/runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/output/ALOD-REPRO-MC-MIXED-01-FORMAL-TRAIN-009.dhier
```

Strict read-only parsing reconfirmed:

| Property | Value |
|---|---:|
| SHA-256 | `38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b` |
| Serialized rows | 7,072,627 |
| Skybox rows | 100,000 |
| Hierarchy rows | 6,972,627 |
| SH degree | 1 |
| Root global node ID | 100,000 |
| Root parent / depth | -1 / 0 |
| Reachable hierarchy rows | 6,972,627 |
| Maximum depth | 55 |

The formal dataset manifest remained
`eaf3cb228234efd92441d28ea56482dcf4676c5a318a6d6beef1020ee27c5f7a`.
The pinned official checkout remained clean at commit
`73f547341547ae021ff036ad74fdf73a1240b450`.

The Run005 insertion patch was applied only to a disposable `git archive`
extraction. Its SHA-256 is
`930b3945b89a4f33128197e0e4b0603561aad1e7869324042fc65201d0ba3dbd`.
The generalized observational hook has SHA-256
`d33cfa31fe0c2bdd9f1205468662eca849616fedcb8e050d2c57cdf700ed5686`.
The generalization changes only fail-closed asset dimensions and expected
camera resolution fields supplied through the oracle environment; Run005
defaults remain unchanged.

## Authoritative evaluation semantics

The authority remains `eval_hierarchy.py`:

```text
load_dynamic_hierarchy
  -> compact_gaussians
  -> build_hierarchical_SPT
  -> four-plane frustum extraction
  -> upper cut_hierarchy_on_condition(root_node=0)
  -> ordered candidate SPT IDs and root distances
  -> gaussian_hierarchy._C.get_spt_cut_cuda
  -> ordered upper IDs + ordered CUDA SPT IDs
  -> gather serialized properties by global ID
  -> prepend serialized skybox rows 0..99,999
  -> render_gsplat(packed=True, render_mode="RGB")
```

Each camera repetition used a fresh OS process, fresh model, rebuilt HSPT, no
prior SPT cut, no viewer previous-working-set reuse, and no fine-training or
project cache. `global_node_id` is always the serialized `.dhier` row index.

Evaluation-time HSPT configuration was derived from captured official values:

| Setting | Evaluation value |
|---|---:|
| `SPT_root_volume` | 4.75 |
| Conceptual target granularity | 2 pixels |
| Base focal length | 2317.6449482429634 source pixels |
| Resolved target granularity | 0.0008629449482830516 |
| Effective `min_SPT_size` | 100 |
| Distance multiplier | 1.0 |
| Frustum planes | left, right, bottom, top |
| Near/far planes | inactive in pinned source |
| Bounding-sphere mode | false |
| Bounds | `3 * exp(max(log_scale_components))` |

The formal training configuration declared `SPT_root_volume=4.75`, two-pixel
target granularity, `min_SPT_size=256`, frustum culling enabled, and bounding
spheres disabled. The initial `train_hierarchy.py` build explicitly passes
256. Its densification rebuild call omits that argument and therefore invokes
the source default 100. HSPT is transient and is not serialized in `.dhier`.
Evaluation independently rebuilds it and also omits the argument, so its
source-defined effective value is 100. This is the documented training versus
evaluation difference; no parameter was changed for the oracle.

## Evaluation HSPT structure

| Property | Value |
|---|---:|
| Upper hierarchy rows | 335 |
| SPT count | 26 |
| Flattened SPT rows | 6,972,318 |
| Total HSPT metadata rows | 6,972,653 |
| Unique hierarchy rows covered | 6,972,627 |
| Shared upper/SPT roots | 26 |

The upper and flattened SPT mappings intersect only at the 26 SPT roots and
their union is exactly serialized hierarchy rows 100,000 through 7,072,626.
The ordered SPT-root hash is
`acc576221a964469a6a7fb236fe5f30b77ae4f16a88a3d11297953aa924dcbdd`.

SPT sizes are highly skewed:

| min | median | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|
| 119 | 1,512 | 1,223,118 | 1,752,133 | 1,857,171 | 1,874,473 |

`hspt.json` contains ordered mappings, sizes, starts, thresholds, coordinates,
scales, and raw/payload hashes for every deterministic structural array.

## Frozen cameras and official selections

Aerial views use source intrinsics
`fx=fy=2317.6449482429634, cx=960, cy=540` at 1920x1080 and effective
intrinsics `fx=fy=1158.8224741214817, cx=480, cy=270` at 960x540. Street
views use source intrinsics approximately
`fx=fy=499.999978, cx=cy=500` at 1000x1000 and effective intrinsics
approximately `fx=fy=249.999989, cx=cy=250` at 500x500.

The identity column is `dataset index / evaluation index / registered image ID /
camera ID`. Depth summary is `min / p10 / p25 / median / mean / p75 / p90 /
max`. Full source pose, runtime matrices, camera centre, and exact source and
effective intrinsics are retained per camera in `formal-analysis.json` and
`camera.json`.

| Frozen filename | Identity | Class / stratum | Camera centre | Candidate SPTs | Selected hierarchy | Depth summary |
|---|---|---|---|---:|---:|---|
| `000060__aerial__test__0049.png` | 60/6/61/1 | aerial/far | (-6.900,-0.900,1.500) | 24 | 3,325,366 | 2/21/21/22/22.9127/24/27/55 |
| `000090__aerial__test__0072.png` | 90/9/91/1 | aerial/near | (-5.600,-0.500,1.500) | 19 | 3,415,808 | 2/21/21/22/22.9485/24/27/55 |
| `000100__aerial__test__0084.png` | 100/10/101/1 | aerial/mid | (-5.167,-2.100,1.500) | 23 | 3,388,342 | 2/21/21/22/22.9415/24/27/55 |
| `000130__aerial__test__0129.png` | 130/13/131/1 | aerial/near | (-2.567,-0.900,1.500) | 22 | 3,466,127 | 2/21/21/22/22.9533/24/27/55 |
| `000150__aerial__test__0000.png` | 150/15/151/1 | aerial/far | (-0.700,-1.700,5.000) | 16 | 3,338,135 | 2/21/21/22/22.8802/24/27/55 |
| `000220__aerial__test__0098.png` | 220/22/221/1 | aerial/near | (-1.738,2.571,2.500) | 23 | 3,470,520 | 2/21/21/22/22.9527/24/27/55 |
| `000240__aerial__test__0124.png` | 240/24/241/1 | aerial/mid | (-0.397,1.743,2.500) | 17 | 3,475,238 | 2/21/21/22/22.9532/24/27/55 |
| `000250__aerial__test__0147.png` | 250/25/251/1 | aerial/far | (0.944,2.157,2.500) | 23 | 3,453,200 | 2/21/21/22/22.9386/24/27/55 |
| `000320__street__test__0006.png` | 320/32/321/2 | street | (-4.815,1.600,0.030) | 20 | 3,453,244 | 2/21/21/22/22.9521/24/27/55 |
| `000350__street__test__0009.png` | 350/35/351/2 | street | (-3.597,1.600,0.030) | 21 | 3,472,441 | 2/21/21/22/22.9536/24/27/55 |
| `000370__street__test__0011.png` | 370/37/371/2 | street | (-2.785,1.600,0.030) | 22 | 3,478,228 | 2/21/21/22/22.9541/24/27/55 |
| `000480__street__test__0022.png` | 480/48/481/2 | street | (-3.191,1.600,0.030) | 21 | 3,475,852 | 2/21/21/22/22.9539/24/27/55 |
| `000500__street__test__0024.png` | 500/50/501/2 | street | (-4.815,1.600,0.030) | 17 | 3,452,525 | 2/21/21/22/22.9525/24/27/55 |
| `000520__street__test__0026.png` | 520/52/521/2 | street | (-4.003,1.600,0.030) | 17 | 3,466,970 | 2/21/21/22/22.9536/24/27/55 |
| `000580__street__test__0032.png` | 580/58/581/2 | street | (-3.777,1.128,0.030) | 21 | 3,472,059 | 2/21/21/22/22.9536/24/27/55 |
| `000650__street__test__0039.png` | 650/65/651/2 | street | (-3.777,2.057,0.030) | 22 | 3,467,703 | 2/21/21/22/22.9531/24/27/55 |

Every raster ID vector is exactly:

```text
[0, 1, ..., 99,999] + official ordered selected hierarchy global IDs
```

Thus total raster rows are selected hierarchy count plus 100,000. For each
touched SPT, `formal-analysis.json` records its candidate-order index, SPT ID,
root global ID, camera distance, input interval size, selected count, ordered
hash, and sorted hash. The official arrays preserve upper traversal order,
candidate SPT order, raw CUDA prefix starts, per-SPT outputs, combined order,
and sorted identity separately.

## Aerial versus street

| Measure | Aerial (8) | Street (8) |
|---|---:|---:|
| Mean selected hierarchy count | 3,416,592.00 | 3,467,377.75 |
| Median selected hierarchy count | 3,434,504 | 3,469,881 |
| Count min / max | 3,325,366 / 3,475,238 | 3,452,525 / 3,478,228 |
| Mean candidate SPT count | 20.875 | 20.125 |
| Median candidate SPT count | 22.5 | 21 |
| Pooled median selected depth | 22 | 22 |
| Pooled mean selected depth | 22.93544 | 22.95332 |
| Pooled p90 selected depth | 27 | 27 |
| Maximum selected depth | 55 | 55 |
| Fraction at depth >=30 | 4.59896% | 4.62239% |
| Fraction at depth >=37 | 0.49616% | 0.49199% |

Street cameras therefore select about 1.49% more hierarchy rows on average,
but the median, p90, maximum, and deep-tail fractions do not show a systematic
finer/deeper cut. The pooled mean-depth difference is only about 0.018 depth
levels and the deep-tail fraction is slightly lower for street views.

## Cross-camera variation and classification

Selected-set Jaccard:

| Pair class | Minimum | Median | Maximum |
|---|---:|---:|---:|
| All 120 pairs | 0.911268 | 0.979254 | 0.999792 |
| Aerial/aerial | 0.911681 | 0.949692 | 0.993302 |
| Street/street | 0.983238 | 0.992376 | 0.999792 |
| Aerial/street | 0.911268 | 0.970916 | 0.995048 |

Depth-distribution total variation distance:

| Pair class | Minimum | Median | Maximum |
|---|---:|---:|---:|
| All 120 pairs | 0.000072 | 0.003147 | 0.018369 |
| Aerial/aerial | 0.000920 | 0.009062 | 0.018160 |
| Street/street | 0.000072 | 0.001424 | 0.003812 |
| Aerial/street | 0.000171 | 0.004536 | 0.018369 |

The criteria were frozen in the Run005 methodology before these formal values
were inspected:

- meaningful set variation: minimum pairwise Jaccard <= 0.95;
- material count variation: `(max-min)/median >= 0.05`;
- material depth variation: maximum selected depth varies or maximum pairwise
  depth-distribution TVD >= 0.05;
- `NONTRIVIAL` requires all three; `DEGENERATE` requires identical sets,
  counts, and distributions; otherwise classification is `WEAK`.

Observed evidence is:

- minimum Jaccard 0.911268: meaningful set variation passes;
- selected counts 3,325,366 to 3,478,228 around median 3,466,548.5, a 4.4096%
  relative range: material count variation fails;
- maximum selected depth is 55 for every camera and maximum depth TVD is
  0.018369: material depth variation fails.

The formal result is therefore `WEAK-LOD-ORACLE`. It is not degenerate: sets,
counts, candidate SPTs, and distributions do change. But it is not a
`NONTRIVIAL-LOD-ORACLE` because the mixed aerial/street scale does not shift
the official cut deeply enough under the source evaluation configuration.

Run005 was also `WEAK-LOD-ORACLE`, with minimum Jaccard 0.663362, 27.14% count
range, and maximum depth TVD 0.019207. This comparison is correctness/LoD
characterization only, not a performance comparison. The formal hierarchy has
many more possible and selected depths (through 55), yet its camera-dependent
depth distributions are no more materially separated under the frozen 5%
criterion.

## Frozen boundary examples

These examples were chosen only after classification from the existing frozen
set and do not affect it:

| Role | Camera | Selected | SPTs | Depth 21 | Depth 22 | Depth >=30 | Mean depth |
|---|---|---:|---:|---:|---:|---:|---:|
| Far aerial | `000060__aerial__test__0049.png` | 3,325,366 | 24 | 1,114,956 | 762,007 | 154,223 | 22.91274 |
| Near aerial | `000220__aerial__test__0098.png` | 3,470,520 | 23 | 1,139,337 | 822,395 | 160,140 | 22.95273 |
| Representative street | `000580__street__test__0032.png` | 3,472,059 | 21 | 1,139,583 | 822,672 | 160,289 | 22.95361 |

The near aerial and representative street cuts are nearly indistinguishable
in depth distribution despite different modality and runtime intrinsics. The
far aerial example mainly reduces selected count and changes SPT coverage.

## Determinism and pre-raster identity

All 16 cameras ran twice. Each repetition matched exactly for:

- all captured HSPT structural hashes;
- effective camera matrices;
- upper traversal and emitted upper order;
- candidate SPT IDs and order;
- raw ordered CUDA SPT results and prefix starts;
- combined ordered and sorted hierarchy IDs;
- selected serialized source-row hashes;
- raster global-ID vector;
- means, quaternions, activated scales, activated opacities, SH/features,
  view matrix, intrinsics, dimensions, and black background;
- rendered RGB endpoint hash.

The render hash is diagnostic only. No image-quality result was computed.

## Project faithful-reproduction parity

`scripts/alod/compare_reference_selection_oracle.py` independently loaded the
formal `.dhier`, rebuilt HSPT from public project code, and consumed the frozen
oracle only after construction. Its report SHA-256 is
`f50e03e820a587020fce0653fba2dbc491577476dbc0484eac537e783b357394`.

It returned `PASS-ALOD-CPU-REFERENCE-SELECTION` with exact parity for:

- all 11 HSPT structural arrays and mappings;
- effective camera centre and projection input;
- four-plane frustum;
- upper traversal;
- candidate SPT IDs, roots, distances, and ordering;
- every individual ordered SPT cut;
- combined ordered hierarchy selection;
- sorted hierarchy selection;
- final skybox-plus-selection raster vector.

No ALOD-EXT prematerialization filter, cache extension, host-cache mode, or
optimized selector was used.

## Validation and protected state

The project oracle validator checked all 16 cameras, 326 payload references,
128 stage JSON hashes, HSPT coverage, unique/global ID bounds, ordered and
sorted hashes, skybox placement, tensor row correspondence, and two-run
determinism records. The freeze gate also reconfirmed the official source was
clean, the Run005 semantic manifest remained
`20bb87caf1694ede18ac2d2a71cdc146c002325ddbf8f6060625f18feda70591`,
and the formal hierarchy and dataset hashes were unchanged.

Run009, Run002, Run006/Run007 history, Run005, the Run005 oracle, the formal
dataset/split/camera identities, CityGaussian, and ALOD-EXT results were not
modified. The canonical official checkout was not modified. Nothing was
staged, committed, or pushed.

## Limitations

- The oracle establishes source-faithful selection correctness, not rendering
  quality or performance.
- The 16 cameras are diagnostic samples, not an exhaustive trajectory.
- `WEAK` is a conclusion about camera-dependent cut variation under the pinned
  official configuration, not about hierarchy validity.
- The formal mixed-scale construction changes both modality and intrinsics,
  but official cuts remain dominated by similar depth bands.

## Next bounded task

Freeze and execute the `ALOD-REPRO` paper-style evaluation ladder on the formal
mixed-scale hierarchy, including source-faithful no-cache rendering, frustum
ablation, HSPT/BFS cut comparison, memory/performance characterization, and
quality evaluation where publicly reproducible.
