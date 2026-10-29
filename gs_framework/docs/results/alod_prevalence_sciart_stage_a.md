# Sci-Art Stage-A ALoD prevalence result

Status: `VALID-WEAK-LOD` under frozen protocol
`ALOD-LOD-PREVALENCE-DESIGN-001`.

## Scope and identity

The dataset is `ALOD-PREV-US3D-SCIART-01`, the UrbanScene3D Sci-Art
reconstruction prepared through the public COLMAP path. Training ID is
`ALOD-PREV-US3D-SCIART-01-TRAIN-001`; oracle ID is
`ALOD-PREV-US3D-SCIART-01-ORACLE-001`. The frozen public split contains
2,998 training and 21 validation images. Every validation camera was evaluated
twice in a fresh official process at the source-documented factor-4 resolution,
1216x912. The oracle adapter interleaves 42 registered cameras so the pinned
official `llff_hold=2` path selects exactly the 21 public validation names;
the training dataset and hierarchy are unchanged.

The pinned official source commit is
`73f547341547ae021ff036ad74fdf73a1240b450`. Only the already-audited
relocation/topology correctness support and observation-only oracle
instrumentation are in scope. No ALoD-EXT selector, cache, or materialization
variant contributes to this classification.

## Training and hierarchy

The dataset preflight, 30k coarse stage, hierarchy creator, structure-only SPT
sweep, official HSPT/cache barrier, and fine-admission gate all passed.
Outcome-blind calibration selected `SPT_root_volume=1228800`, the largest
stable passing candidate around 614400/1228800/2457600; 4915200 failed the
frozen nondegeneracy criteria. The official barrier built 51 SPTs with
0.9996831171 hierarchy coverage and required 731,560 cache rows, below the
5,000,000-row reduced cache.

Fine training completed 100,000 iterations and exited zero. The independent
strict parser found:

| Property | Value |
| --- | ---: |
| Final asset SHA-256 | `acc49a863e022a6871391a224dfa01e2de24133a168c3e4498c2c8b83afa3010` |
| File bytes | 588,605,472 |
| Total rows | 5,074,185 |
| Skybox rows | 100,000 |
| Reachable hierarchy rows | 4,974,185 |
| Maximum depth | 48 |
| SH degree | 1 |
| Topology/numeric errors | 0 |

The 12M cap retained 6,925,815 rows (57.715125%) of final headroom. The fine
stage lasted 65,472.63 sampled seconds (18.19 h), below the frozen 20 h limit.
Sampled peaks were 6.944 GiB process-group RSS and 5.024 GiB whole-device VRAM;
minimum available system RAM was 19.890 GiB; maximum swap growth was
0.0257 GiB. Maximum monitored output was 1.880 GiB. GPU memory is a
whole-device observation, not reliable process attribution.

## Official oracle and independent parity

The canonical official oracle completed 21 cameras times two cold repetitions
with exact determinism. Its integrity validator passed all 21 cameras, 410
payload references, and 168 stage-JSON hashes. HSPT mapping covered all
4,974,185 hierarchy rows: 413 unique upper rows, 4,973,834 unique flattened
SPT rows, and 62 shared SPT roots.

The independent CPU reference rebuilt the exact official HSPT and matched every
camera's effective input, frustum, upper traversal, candidate SPTs, ordered
per-SPT cuts, combined selection, sorted selection, and final raster rows. Its
status is `PASS-ALOD-CPU-REFERENCE-SELECTION`. The CPU HSPT build took
1.078 s; per-camera reference selection ranged from 0.0297 to 0.0433 s. These
are validation timings, not released renderer performance.

Canonical evidence hashes:

- semantic manifest:
  `5bf84d3ee1a6a1f88210ed6f0e260fcf9617bccc37b0eae9660d16c77ee55cac`;
- CPU parity:
  `8d64ea9302eb83433ce437e8068cdcc6ff74c11574f75036ec808bbc64101139`;
- prevalence response:
  `5f6d085524630f185f1936365e6f3b9e7a90323218121ea22751bda7f72283cf`;
- frozen camera manifest:
  `b43b3d018a926497e73e0634775bcad7835a54cc0a4ea3e4b18fb790da509def`.

## Frozen classification

The prospective `criteria_v1` values are:

| Frozen predicate | Sci-Art value | Threshold | Result |
| --- | ---: | ---: | --- |
| Selected-count relative range | 0.024205443210447316 | >= 0.05 | fail |
| Minimum pairwise selected-set Jaccard | 0.9388327852940161 | <= 0.95 | pass |
| Maximum pairwise depth-distribution TVD | 0.013039350956579489 | >= 0.05, or varying maximum depth | fail |

Selected hierarchy counts range from 2,413,401 to 2,472,825, with median
2,454,985. Every camera reaches maximum selected depth 48. Mean selected depth
ranges only from 21.8498663 to 21.8778551, and median selected depth is 22 for
every camera. The exact degenerate predicate is false because selected sets,
counts, and distributions differ. Only meaningful set variation passes, so the
frozen label is `VALID-WEAK-LOD`.

The nearest-geometry projected-scale proxy has Spearman rho 0.1000 and a
10,000-resample camera-bootstrap interval spanning zero. The scene-centre
sensitivity proxy has rho 0.833766, with bootstrap interval
[0.535827, 0.956780]. That is a strong descriptive association for this proxy,
but the absolute normalized-depth change is small (top-minus-bottom quintile
median 0.0003914), and it cannot override the prospectively frozen binary
criteria.

## Attempt chronology and boundaries

A power interruption occurred after coarse training and hierarchy creation.
Both completed artifacts were hash-verified before continuation; no partial
training checkpoint was resumed. Several preflight retries corrected local
evidence-tool assumptions (the creator row estimate, tracked CPython bytecode,
and downsampled HSPT validation). They did not relax resource, topology,
selection, or outcome thresholds.

The first oracle attempt completed four cameras deterministically, then an
observation-only camera-centre assertion rejected camera 4. Exact reproduction
showed that the official DataLoader worker performs a CPU float32 matrix
inverse: its 2.8706e-5 error exceeded the two-ULP-scaled audit bound by 0.8%.
All 21 poses were checked, and a four-coordinate-scaled-ULP bound passed with a
maximum tolerance fraction of 0.5042. The failed logs were preserved, the
incomplete 411 MiB output was removed, and the canonical oracle restarted from
camera 0 with one instrumentation hash. Dataset, hierarchy, official selector,
camera set, and prevalence thresholds were unchanged.

No quality metrics, FPS benchmark, paper-number comparison, ALoD-EXT run,
Stage-B repeat, or synthetic-control result is claimed here.

## Canonical artifacts

- Training:
  `runs/alod/ALOD-PREV-US3D-SCIART-01/ALOD-PREV-US3D-SCIART-01-TRAIN-001/`.
- Frozen cameras:
  `gs_framework/datasets/alod/ALOD-PREV-US3D-SCIART-01/evaluation-cameras.json`.
- Oracle:
  `runs/alod/oracles/ALOD-PREV-US3D-SCIART-01-ORACLE-001/`.
- Final validation:
  `validation/final-hierarchy-independent.json` under the training directory.
- Independent parity and outcome:
  `validation/cpu-reference-parity.json` and
  `validation/prevalence-lod-response.json` under the oracle directory.
