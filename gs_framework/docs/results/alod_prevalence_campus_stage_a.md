# Campus Stage-A ALoD prevalence result

Status: `VALID-NONTRIVIAL-LOD` under frozen protocol
`ALOD-LOD-PREVALENCE-DESIGN-001`. This is one valid natural-dataset outcome, not
the completed Stage-A prevalence tally.

## Scope and identity

The dataset is `ALOD-PREV-H3DGS-CAMPUS-CHUNK-01`, the standalone H3DGS Campus
chunk. Training ID is `ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-TRAIN-001`; oracle ID is
`ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-ORACLE-001`. The public pre-existing held-out
split has 24 images; all 24 were evaluated twice in cold official processes at
the archive's 1436×1075 resolution. The source commit is
`73f547341547ae021ff036ad74fdf73a1240b450`, with the already-audited
Run009 relocation/topology correctness and compatibility support for training.
No ALOD-EXT selector, cache, or materialization variant contributed to the
classification.

This is a project-scale, correctness-patched reproduction of released-source
behavior under the frozen local 100k-iteration/12M-row method. It does not
reproduce the paper's full dataset, training regime, quality numbers, or
rendering performance. Campus download rights were not established for
redistribution; the local source is retained for research/evaluation only.

## Training and validation

The dataset preflight and 30k coarse stage passed. The creator made a valid
718,949-node hierarchy; with 100,000 skybox rows, the fine initial state had
818,949 rows. Outcome-blind structural calibration selected
`SPT_root_volume=9600` from a stable 4800/9600/19200 neighbourhood. All frozen
fine-admission gates passed. The 100k fine stage exited zero, without resource
warnings or hard stops, and wrote the final SH1 hierarchy. Its independent
strict parser found 1,143,221 total rows, 1,043,221 non-skybox hierarchy rows,
maximum depth 44, and no topology or numeric errors. Final asset SHA-256:
`8743b7d34f4c61b638f5e6528539b8dca021b9752ddd88d076a66608641815c4`.

The sampled fine-stage peaks were 5.871 GiB process-group RSS and 2.548 GiB
whole-device VRAM; minimum available system RAM was 21.652 GiB; swap growth
was zero. All 51 recorded HSPT/cache barriers were feasible. These are
watchdog measurements, not allocator-only or process-attributed GPU memory.

## Official selection and frozen classification

The official cold oracle completed two deterministic evaluations per held-out
camera. Its integrity validator passed all 24 cameras, 467 payload references,
and 192 stage-JSON hashes. The independent project CPU selector exactly matched
official HSPT and every camera's effective input, frustum, upper traversal,
candidate SPTs, individual and combined cuts, and skybox-plus-selection raster
rows. The deterministic semantic manifest SHA-256 is
`171bb1c12b48402ebd78465fe466def2d1b929580f6ecf5b06e262feba2801d6`.

All three prospective `criteria_v1` predicates pass, with no rounding before
comparison:

| Frozen predicate | Campus value | Threshold |
| --- | ---: | ---: |
| Selected-count relative range | 0.25237410403028177 | >= 0.05 |
| Minimum pairwise selected-set Jaccard | 0.44869920162834015 | <= 0.95 |
| Maximum pairwise depth-distribution TVD | 0.07197548966251402 | >= 0.05, or varying maximum selected depth |

The selected hierarchy count ranges from 350,849 to 460,527 (median 434,585).
Maximum selected depth is 44 for every camera, so it is the TVD branch that
passes the third predicate. Thus the valid label is `VALID-NONTRIVIAL-LOD`.
This label means the *selected cuts vary materially by the frozen three-part
rule*; it does not imply a large or monotonic scale-to-depth effect. The
nearest-sparse-geometry scale proxy has Spearman rho -0.0713, and its
descriptive 10,000-resample camera-bootstrap interval spans zero. The scene-
centre proxy has rho 0.3383 and likewise an interval spanning zero. The full
per-camera summaries, response curves, and intervals remain in the analysis
JSON, rather than being maintained here.

## Attempt chronology and boundaries

The first evaluation launch was blocked before camera evaluation because the
sandbox could not see the NVIDIA driver. A GPU-enabled attempt completed two
cameras but stopped when an observation-only 2e-6 absolute camera-centre check
rejected normal float32 GPU inverse roundoff on camera 2. All 24 frozen camera
centres were checked against the official GPU calculation; the check was
scaled to float32 coordinate magnitude, with a maximum observed error only
0.631 of its permitted bound. Both failed attempts are retained outside the
canonical oracle directory. The final oracle restarted from camera 0 with one
instrumentation version. Training, dataset, selection algorithm, camera set,
and prevalence thresholds were unchanged.

No PSNR/SSIM/LPIPS, FPS, paper-number comparison, `ALOD-EXT` transfer/cache
test, Stage-B repeat, Sci-Art run, or synthetic control is claimed here. A
nontrivial natural Campus result does not retrospectively alter Run009's weak
oracle label or validate Run009 extension gains on this hierarchy.

## Canonical artifacts

- Frozen design: `thesis_notes/ALOD_LOD_PREVALENCE_STUDY_DESIGN.md`.
- Training directory: `runs/alod/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-TRAIN-001/`.
- Fine admission and independent final hierarchy: `validation/fine-admission.json`, `validation/final-hierarchy-independent.json` under that training directory.
- Frozen cameras: `gs_framework/datasets/alod/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01/evaluation-cameras.json`.
- Oracle directory: `runs/alod/oracles/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-ORACLE-001/`.
- Oracle identity and official cuts: `semantic-manifest.json` and `cameras/` under the oracle directory.
- Independent parity and outcome: `validation/cpu-reference-parity.json` and `validation/prevalence-lod-response.json` under the oracle directory.
