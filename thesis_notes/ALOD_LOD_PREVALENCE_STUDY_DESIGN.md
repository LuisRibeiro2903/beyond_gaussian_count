# ALoD adaptive-LoD prevalence study: frozen design

Status: `FROZEN-DESIGN-PENDING-EXECUTION`

Protocol ID: `ALOD-LOD-PREVALENCE-DESIGN-001`

Freeze date: 2026-09-21

Expensive training authorized by this document: **no**

## 1. Question, estimand, and interpretation boundary

The primary question is:

> Across distinct public large-scene datasets, how often does a
> correctness-patched, source-faithful ALoD pipeline produce a valid hierarchy
> with nontrivial camera-dependent LoD under the same declared consumer-hardware
> method and resource envelope?

The unit of observation is one completed and validated training run. The first
stage estimates an outcome frequency over three deliberately different
datasets, not a population prevalence with useful statistical precision:
existing MatrixCity Run009 plus one H3DGS Campus standalone chunk run and one
UrbanScene3D Sci-Art run. Dataset-level camera and scale measures are then used
for descriptive association analysis. They are not causes, treatments, or
proof that commodity hardware causes weak LoD.

The design accepts every prospective outcome. In particular, all admissible
natural hierarchies may remain weak, or one or both may be nontrivial. Dataset
selection, classification thresholds, and the hardware gates must not be
changed after looking at a new LoD cut.

Run005 is historical mechanism/correctness support only. Uni10K is a
`RESOURCE-FEASIBILITY-NEGATIVE` result under its frozen 15M-row preflight, not a
LoD observation. Neither enters the Stage-A natural-outcome denominator.

## 2. Frozen LoD-adaptability outcome

### 2.1 Evaluation workload

For every new valid hierarchy, the primary workload is **every camera in the
public, pre-existing held-out split**, in its recorded order. If a source has
no official split, a deterministic split must be committed before coarse
training and before any hierarchy exists; it may not be chosen from LoD
outcomes. The 16-camera Run009 oracle remains its exact correctness authority,
while all 71 Run009 held-out cameras are the prevalence-study workload.

A geometry-defined near/far stress subset may be frozen before training as a
secondary sensitivity analysis. It cannot override the all-held-out primary
classification.

For camera \(i\), let \(S_i\) be the hierarchy row IDs returned by the
unchanged official ALoD selector, \(n_i=|S_i|\), and \(h_i(d)\) the number of
selected rows at tree depth \(d\). Define

\[
p_i(d)=\frac{h_i(d)}{n_i}.
\]

All values use semantic selector output before frustum filtering, packing,
materialization, caching, or ALoD-EXT.

### 2.2 Exact existing metrics and thresholds (`criteria_v1`)

The following definitions are copied from
`gs_framework/scripts/alod/export_official_selection_oracle.py` and are frozen:

1. **Selected-count relative range**

   \[
   R_n=\frac{\max_i n_i-\min_i n_i}{\operatorname{median}_i n_i}.
   \]

   `material_count_variation := R_n >= 0.05`.

2. **Selected-set Jaccard** for each distinct pair

   \[
   J_{ij}=\frac{|S_i\cap S_j|}{|S_i\cup S_j|}.
   \]

   An empty union has identity 1.0.
   `meaningful_set_variation := min_(i<j) J_ij <= 0.95`.

3. **Selected-depth distribution TVD**

   \[
   \operatorname{TVD}_{ij}=\frac{1}{2}\sum_d|p_i(d)-p_j(d)|.
   \]

   `material_depth_variation := (max_i dmax_i > min_i dmax_i) OR
   (max_(i<j) TVD_ij >= 0.05)`, where \(dmax_i\) is the greatest selected
   depth with nonzero count.

4. **Selected-depth summaries** are reported for every camera:

   - weighted mean \(\mu_i=\sum_d d h_i(d)/n_i\);
   - weighted sample median: the middle order statistic for odd \(n_i\), or the
     arithmetic mean of the two middle depth order statistics for even \(n_i\);
   - maximum: the greatest nonzero selected depth.

5. **Classification**

   - `NONTRIVIAL-LOD`: all three variation predicates are true;
   - `DEGENERATE-LOD`: all selected sets, selected counts, and complete depth
     distributions are exactly identical;
   - `WEAK-LOD`: a valid result that is neither nontrivial nor degenerate.

`DEGENERATE-LOD` is retained as a separate diagnostic subtype and is counted
as a valid weak outcome in the binary weak-versus-nontrivial prevalence tally.
No rounding is applied before a threshold comparison. Exactly 0.95, 0.05, and
0.05 pass their respective predicates.

### 2.3 New continuous measures frozen before new outcomes

These descriptive measures supplement, and cannot veto or replace,
`criteria_v1`:

- camera scale \(x_i=\log_2(q_i/\operatorname{median}(q))\), where
  \(q_i=f_i/r_i\), \(f_i\) is focal length in pixels at study resolution and
  \(r_i\) is distance from the camera centre to its nearest sampled sparse
  point;
- normalized depth response \(y_i=\mu_i/D\), where \(D\) is final hierarchy
  maximum depth;
- Spearman \(\rho(x,y)\);
- Theil--Sen slope of \(y\) per projected-scale octave;
- median \(y\) in the top camera-scale quintile minus median \(y\) in the
  bottom quintile;
- the same three statistics with distance to the median sparse-scene centre as
  a sensitivity proxy;
- hierarchy-scale response curves: median and interquartile range of \(y\) and
  \(n_i/N\) in ten equal-count bins of \(x\), where \(N\) is the number of
  non-skybox hierarchy rows.

Ties, missing values, confidence intervals, and small-sample handling are fixed
as follows: use all finite held-out cameras; report `NA` if fewer than five
distinct \(x_i\); use a two-sided camera bootstrap with 10,000 resamples and
seed `20260921` for 95% percentile intervals; do not use these intervals as a
second categorical threshold. Because cameras within a scene are correlated,
the intervals are descriptive, not independent-sample inferential claims.

### 2.4 Frozen Run009 comparison anchor

The formal 16-camera oracle has 3,325,366 / 3,466,548.5 / 3,478,228
minimum/median/maximum selected rows, hence \(R_n=0.0440963108\); minimum
pairwise Jaccard 0.9112684765; maximum pairwise depth TVD 0.0183692116; pooled
selected-depth median 22; and maximum selected depth 55 for every camera. It
passes set variation but fails count and depth variation, so its exact frozen
class is `WEAK-LOD-ORACLE`.

Across all 71 held-out cameras, selected counts are 2,606,659 / 3,467,266 /
3,478,844. Recomputed with the frozen median-denominator definition,
\(R_n=0.2515483381\). (The older 33.4599% descriptive field divides by the
minimum and must not be substituted for `criteria_v1`.) Maximum selected depth
remains 55 for all cameras, maximum TVD is 0.0235823454, and mean selected
depth spans only 22.8484--23.0579. Thus the depth predicate still fails even
though the wider workload passes count variation.

On the separately frozen 15-camera geometry scale-stress trajectory, selected
depth median is 22 and maximum is 55 for every camera, with maximum TVD about
0.01834. Existing descriptive associations on all 71 are modest: projected
nearest-geometry scale versus mean selected depth has Pearson \(r=0.3077\) and
Spearman \(\rho=0.3797\); the projected-centre proxy has Spearman
\(\rho=0.4904\). These are the comparison anchors, not newly chosen thresholds.

## 3. Pre-outcome candidate audit

### 3.1 Method and provenance

The audit was performed without training a hierarchy or inspecting ALoD cuts.
The reproducible reader is
[`tools/audit_alod_dataset_candidates.py`](../tools/audit_alod_dataset_candidates.py).
It streams public COLMAP binaries, hashes them, reads camera centres, and
samples at most 250,000 sparse points for nearest-geometry queries. Its two
scale proxies are:

\[
q_{near}=f_{study}/r_{nearest},\qquad q_{centre}=f_{study}/r_{centre}.
\]

They are dimensionless projected-scale indicators, not reconstructed physical
units. Cross-dataset raw metre-like extents are not compared because COLMAP
coordinate gauges differ. Robust within-dataset ratios are comparable.

The exact initial-row planning estimate is

\[
H_0=2N_{sparse}-1+100,000,
\]

assuming a full binary creator tree and the frozen 100k skybox. Actual creator
output, topology, HSPT construction, and cache feasibility still require a
formal preflight.

The CityGaussian-provided COLMAP archive was used for Mill19 and UrbanScene3D,
matching the data path used by ALoD for those scenes. Images remain subject to
their original public dataset licences. The H3DGS Campus standalone archive
supplies one combined COLMAP model and an existing 24-image `test.txt`. Audit
artifacts preserve archive and COLMAP-file SHA-256 values:

- [`alod_prevalence_candidate_audit.json`](data/alod_prevalence_candidate_audit.json);
- [`alod_prevalence_campus_candidate_audit.json`](data/alod_prevalence_campus_candidate_audit.json);
- [`alod_prevalence_campus_full_candidate_audit.json`](data/alod_prevalence_campus_full_candidate_audit.json).

Primary public provenance is the [official ALoD repository](https://github.com/FelixWindisch/LoDOfGaussians),
the [official H3DGS dataset host](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/),
the [official Mega-NeRF/Mill19 repository](https://github.com/cmusatyalab/mega-nerf),
and the [official UrbanScene3D repository](https://github.com/Linxius/UrbanScene3D).

Camera-region overlap was not assigned a numerical score. The available
prepared train/eval models do not provide one consistent cross-dataset region
partition, and a nearest-point or view-direction surrogate would be presented
with unjustified precision. Before execution, each selected dataset will
instead receive a non-outcome-bearing coverage report: held-out cameras in the
training camera convex hull, nearest training-camera distance, and viewing-ray
direction distance. This is an eval-support diagnostic, not a selection or LoD
threshold.

### 3.2 Quantitative comparison

`q p90/p10` and `q max/min` are the robust and full-range spreads of
\(q_{near}\). Resolutions are the frozen training resolution. `H0` is the
estimated initial fine state including skybox. `Headroom` is under the proposed
12M-row cap.

| Candidate | Cameras train/eval | Study resolution | Sparse points | Camera extent diagonal | Height p10/med/p90 | Centre distance p10/med/p90 | Nearest geometry p10/med/p90 | q-near p10/med/p90 | q spread p90/p10; max/min | H0; 12M headroom | Pre-outcome assessment |
|---|---:|---|---:|---:|---|---|---|---|---|---|---|
| **Run009 MatrixCity mixed** | 639/71 | 500×500 aerial and 960×540 street | 2,317,128 | 13.956 | 0.024 / 0.024 / 2.494 | 0.166 / 1.019 / 4.641 | 0.027 / 0.076 / 1.244 | 1,862.9 / 7,130.0 / 18,452.9 | **9.905; 24.216** | 4,734,255; 60.55% | Existing completed reference; mixed synthetic camera regimes |
| **H3DGS Campus standalone** | 1,166/24 | 1436×1075 | 359,475 | 84.840 | 1.829 / 2.171 / 4.830 | 4.328 / 14.203 / 27.616 | 0.251 / 0.578 / 1.076 | 670.7 / 1,248.8 / 2,871.7 | 4.282; **299.214** | 818,949; **93.18%** | Ground/handheld-style path; strongest full-range natural scale excursion; low row risk |
| H3DGS Campus full | 22,042/no public split in archive | ≈1436×1075 | 4,726,131 | 326.276 | 5.139 / 11.165 / 16.962 | 23.962 / 62.863 / 130.925 | 0.779 / 1.547 / 1.841 | 392.6 / 467.1 / 927.0 | 2.361; 223.662 | 9,552,261; **20.40%** | Borderline: only 47,739 rows beyond the 20%-headroom boundary, no frozen held-out split, and cache still unknown |
| **UrbanScene3D Sci-Art** | 2,998/21 | 1216×912 (official ALoD factor 4) | 2,133,151 | 495.147 | 50.452 / 63.922 / 88.947 | 60.798 / 109.472 / 159.400 | 7.809 / 17.845 / 36.362 | 25.4 / 51.8 / 118.4 | **4.656; 77.378** | 4,366,301; **63.61%** | Largest robust spread among admissible ALoD paper aerial candidates; distinct urban scene |
| UrbanScene3D Residence | 2,561/21 | 1368×912 (factor 4) | 1,951,399 | 571.939 | 82.766 / 96.071 / 111.395 | 92.690 / 141.544 / 226.471 | 24.360 / 38.867 / 53.995 | 17.0 / 23.7 / 37.8 | 2.217; 11.170 | 4,002,797; 66.64% | Feasible but narrower camera-scale range than Sci-Art |
| Mill19 Building | 1,920/20 | 1152×864 (factor 4) | 1,603,125 | 510.601 | 42.821 / 44.638 / 45.866 | 59.684 / 122.643 / 203.550 | 9.194 / 14.111 / 25.291 | 29.5 / 52.8 / 81.1 | 2.751; 15.882 | 3,306,249; 72.45% | Feasible; near-constant aerial height, a documented ALoD worst case |
| Mill19 Rubble | 1,657/21 | 1152×864 (factor 4) | 1,694,315 | 321.841 | 33.184 / 34.376 / 35.801 | 42.144 / 76.415 / 111.792 | 14.463 / 21.947 / 26.494 | 28.1 / 33.9 / 51.5 | 1.832; 7.128 | 3,488,629; 70.93% | Feasible; narrowest scale variation |
| Uni10K full | 8,805/1,258 | 2K study resolution from 4K source | 6,170,655 | 14.908 | −0.327 / 2.117 / 4.387 | 2.100 / 3.415 / 4.965 | 0.034 / 0.246 / 0.741 | 1,820.7 / 5,483.4 / 39,828.0 | **21.875; 2,663.31** | 12,441,309; negative under 12M | Strongest robust spread, but already failed its more permissive 15M/20% gate (17.05794%); no LoD label |

The height sign for Uni10K reflects the sparse-ground proxy and arbitrary
reconstruction gauge, not cameras physically below ground. The Campus full
range is tail-sensitive; the robust p90/p10 measure is reported alongside it
to prevent one close camera from being mistaken for broad coverage. Conversely,
Run009 has the broadest admissible robust ratio, so this audit does **not**
promise that a natural candidate has a uniformly broader scale distribution by
every statistic. Campus tests much closer natural viewpoints (full range), and
Sci-Art provides the widest admissible robust range among the aerial candidates.

Preprocessing burden was also frozen qualitatively and by known staged bytes:

| Candidate group | Staged public calibration/archive | Required preparation | Burden |
|---|---:|---|---|
| Run009 | already complete | none for Stage A | none |
| Campus standalone | 4.1 GiB archive | verify hashes; expose existing COLMAP model/images and `test.txt`; no reconstruction | low--medium |
| Campus full | 10.9 GiB archive | extract aligned 22,042-camera model; define and seal an outcome-independent split; validate 22k image links | high |
| Mill19 Building/Rubble | shared 2.2 GiB CityGaussian COLMAP archive, excluding original imagery | acquire/link original Mill19 images; verify train/val names; apply documented factor 4 | medium |
| UrbanScene3D Residence/Sci-Art | same shared 2.2 GiB COLMAP archive, excluding original imagery | acquire/link original UrbanScene3D images under its licence; verify train/val names; apply factor 4 | medium |
| Uni10K | source and validated derived proxy already local | no new preparation; existing source-semantic observation/track stripping is already validated | complete but fine-inadmissible |

These labels cover data staging only; the common audit, formal validation, and
training/evaluation work is budgeted separately below.

### 3.3 Candidates not selected

- **Uni10K:** retained as the resource-feasibility negative control. Reducing
  its points or images now would create a new proxy selected after failure and
  would not answer the source-faithful question.
- **Mill19 Building/Rubble:** ALoD explicitly describes the nearly regular,
  same-height aerial imagery as a worst case for LoD. They are useful future
  weak-regime controls but add less diversity than Campus and Sci-Art.
- **UrbanScene3D Residence:** feasible, but Sci-Art dominates it on audited
  scale spread and supplies more cameras.
- **Full H3DGS Campus:** deferred from Stage A. Its public calibration package
  is about 10.9 GB, contains 22,042 registered cameras, and has no public
  held-out list in that archive. Its 9,552,261-row estimate passes the 9.6M
  admission boundary by only 47,739 rows before an actual creator/cache
  preflight, and the ALoD paper's full-scene schedule is substantially heavier.
  The standalone chunk gives a public fixed split and isolates the desired
  natural close/far ground-view regime with much safer headroom. Full Campus
  may become a separately pre-registered Stage-C scale-up, never a replacement
  chosen after observing Campus-chunk LoD.

## 4. Selected Stage-A datasets

### Dataset B: `ALOD-PREV-H3DGS-CAMPUS-CHUNK-01`

Select the public H3DGS Campus standalone chunk. It is independent of the
MatrixCity construction, has a ground-dominated natural trajectory, public
COLMAP input, an existing fixed test list, a 299.2× full projected-scale range,
and an initial hierarchy estimate only 6.82% of the 12M cap. It tests whether a
close/far ground-camera tail can elicit adaptive depth where Run009 did not.

Limitation: the current archive has 1,190 registered images, whereas the ALoD
paper labels its Campus chunk as about 1.5k images. The study therefore names
and hashes this exact public archive and does not claim bit-identical paper data.

### Dataset C: `ALOD-PREV-US3D-SCIART-01`

Select UrbanScene3D Sci-Art through the public CityGaussian/ALoD-compatible
COLMAP path. It is a large, independent real urban reconstruction with 3,019
registered cameras, 2.13M sparse points, a 4.656 robust projected-scale ratio,
and 63.61% cap headroom. It differs from Campus in capture regime and scene
extent, preventing the initial sample from becoming two variants of the same
ground path. The factor-4 image scale is source-documented by ALoD for
Mill19/UrbanScene3D and is frozen before outcomes.

This pair prioritizes capture-regime diversity, public provenance, and resource
admissibility. It is not selected because either dataset is known to yield a
desired LoD result.

## 5. Frozen training and hardware method

### 5.1 Common software/method

- official repository commit:
  `73f547341547ae021ff036ad74fdf73a1240b450`;
- only the already-audited relocation/topology correctness patch and
  compatibility support used by Run009; every patch hash recorded per run;
- same coarse trainer, hierarchy creator, fine trainer, official selector,
  hierarchy convention, skybox semantics, and validation tools;
- 30,000 coarse and 100,000 fine iterations; SH degree 1;
- classic densification every 1,500 iterations, active from 10,000 through
  85,000; `cap_max=12,000,000`;
- `graph_view_select=false`, bounding spheres off, frustum culling on,
  `prune_unused=false`, CPU hierarchy storage;
- two-pixel target granularity, initial `min_SPT_size=256`, distance multiplier
  variation on, exposure optimization off;
- Run009 optimizer/loss fields and learning rates copied byte-for-byte from
  `ALOD-REPRO-MC-MIXED-01-formal-target-v3-run009.json`;
- GPU cache 6.25M rows, reduced target 5.0M rows, clear interval 1,000;
- same official evaluator/selector semantics; no ALoD-EXT during classification.

Permitted dataset calibration is limited to input paths, official/source-
documented resolution (factor 4 for Mill19/UrbanScene3D; archive resolution for
Campus), pre-existing train/eval split, vertical/gauge metadata, and
`SPT_root_volume`. `SPT_root_volume` is selected by a structure-only sweep:
choose the largest value in a stable neighbouring range that passes coverage,
SPT-size, cache, topology, and resource gates. No image render, quality metric,
or LoD cut may inform it. `min_SPT_size` starts and remains 256 unless the
official creator cannot form a valid hierarchy; any change requires a new
prospective protocol and excludes the run from this frozen study.

The fixed 100k-fine-iteration budget deliberately differs from paper schedules
that scale with image count. It makes the local consumer-envelope comparison
coherent; any under-convergence is reported as a limitation and is not repaired
after seeing LoD behavior.

### 5.2 Machine and watchdog

The fixed machine is the Run009 host: RTX 5080 (16,303 MiB VRAM), 30.46 GiB host
RAM, 16 GiB swap, and 8 logical CPUs. Sample every five seconds. Threshold
duration means consecutive violating samples, with the first sample counting.

| Resource | Warning | Hard stop |
|---|---|---|
| Process-group RSS | >20 GiB for 30 s | >24 GiB for 15 s |
| Available system RAM | <8 GiB immediately | <5 GiB for 15 s |
| Swap growth from stage baseline | >128 MiB for 15 s | >256 MiB for 15 s, or six strictly increasing samples totalling ≥128 MiB |
| Whole-device VRAM | >12 GiB for 10 s | >13.5 GiB for 10 s |
| Training-process VRAM, when reliable PID attribution exists | >11.5 GiB for 10 s | >13 GiB for 10 s |
| Output directory | >2 GiB | >5 GiB |
| No progress | >900 s | >1,800 s |
| Fine rows | ≥11.4M diagnostic | configuration-enforced 12M cap |

Stage warning/hard wall times are coarse 1/2 h, creator 1/3 h, cache preflight
0.5/1 h, fine 14/20 h, and whole pipeline 18/24 h. Warnings do not authorize
relaxation. PyTorch allocated/reserved memory is diagnostic only. If reliable
process attribution is unavailable, whole-device VRAM remains authoritative.

Run009 observed peaks (7.163 GiB process RSS, 6.501 GiB whole-device VRAM,
zero swap growth, 0.764 GiB output, 15.21 h fine training) are planning evidence,
not substitutions for the gates.

## 6. Admission, stopping, and retry rules

### 6.1 Before coarse training

Require: pinned archive/model hashes and licences; immutable split; all images
resolvable; COLMAP cameras supported; finite poses/intrinsics; disk free at
least 64 GiB after input staging; live GPU identity; zero competing compute
processes; watchdog dry run; software/patch/config hashes sealed. Failure is
`IMPLEMENTATION-FAILURE` if caused by ingestion/code, or
`RESOURCE-INFEASIBLE` if the declared machine cannot stage the public input
within disk/RAM/time gates.

### 6.2 Before fine training

All conditions must pass:

1. complete 30k coarse output and creator output; no partial artifact reused;
2. strict binary-tree/topology, row-property, skybox, finite-value, and official
   loader validation;
3. initial rows \(H_{actual}<12,000,000\);
4. growth headroom satisfies both
   \(12,000,000-H_{actual}\ge1,000,000\) and
   \((12,000,000-H_{actual})/12,000,000\ge0.20\). Thus the binding prospective
   bound is \(H_{actual}\le9,600,000\);
5. HSPT coverage at least 95%, at least two nondegenerate SPTs, and
   `skybox + upper hierarchy + largest SPT + 100,000 reserve <= 5,000,000`;
6. the structure-only `SPT_root_volume` sweep has a stable passing neighbour,
   not a single isolated value;
7. coarse/creator/cache stages stayed inside all hard resource and wall-time
   limits.

The simple `H0` estimates pass for Campus and Sci-Art. That is planning
feasibility, not admission; actual creator/HSPT results control.

### 6.3 During fine training

Terminate on any hard watchdog condition; CUDA OOM/illegal access/device-side
assert; NaN/Inf loss; unhandled traceback; topology/cache barrier failure;
12M cap breach attempt; or whole-pipeline hard time. Dynamic growth may proceed
only within the frozen cap/cache. The cap, cache, resolution, SPT layout, or
optimizer may not be increased in response to growth or a preliminary LoD cut.

Preserve logs and partial outputs, label them invalid barrier inputs, and never
run the LoD classifier on them.

### 6.4 Retry policy

One automatic restart from the beginning is allowed for a documented external
failure (power loss, WSL/service interruption, unrelated process collision) if
inputs and every method/configuration hash are unchanged. A second external
failure pauses the dataset for manual audit; it does not become weak LoD or
resource infeasible. OOM, watchdog, cap, cache, output, or wall-time stops are
resource outcomes and are not retried with relaxed settings. A reproducible
traceback or correctness failure is an implementation failure. No resume from
an unvalidated partial checkpoint is allowed.

## 7. Outcome state machine

Classification occurs in this order:

1. **`RESOURCE-INFEASIBLE`**: a prospective row/headroom/cache gate fails, or a
   hard RAM/VRAM/swap/disk/output/no-progress/wall-time/dynamic-growth gate
   stops the run under the pinned method. No LoD label is assigned.
2. **`IMPLEMENTATION-FAILURE`**: loader, patch, code, creator topology,
   selector parity, artifact-integrity, or reproducible runtime correctness
   fails despite admitted resources. No LoD label is assigned.
3. **`VALID-DEGENERATE-LOD`**: complete hierarchy and selector correctness pass,
   and exact degenerate predicate holds. Counted under valid weak in the binary
   prevalence tally, while reported separately.
4. **`VALID-WEAK-LOD`**: complete/correct result is neither degenerate nor
   `criteria_v1` nontrivial.
5. **`VALID-NONTRIVIAL-LOD`**: complete/correct result passes all three frozen
   variation predicates. “Strong” is descriptive shorthand only for this
   class, never for a favourable continuous correlation alone.

An unresolved external interruption is `EXTERNAL-EXECUTION-FAILURE`, reported
outside the four scientific outcome classes and eligible only for the exact
retry rule above.

## 8. Staged replication and randomness audit

### Stage A: dataset coverage first

- retain completed `ALOD-REPRO-MC-MIXED-01` Run009;
- execute one admitted Campus standalone training;
- execute one admitted Sci-Art training;
- validate, freeze all-held-out cuts, classify, and report each before any
  repeat is chosen.

The Stage-A report gives counts over valid natural hierarchies and separately
lists resource/implementation failures. With only three datasets, use exact
fractions and the full outcome list; do not market a precise population rate.

### Randomness audit before Stage B

The official path calls a global `safe_state` seed (currently zero), but
stochasticity still enters through shuffled training-camera order, random
coarse backgrounds, fine distance-multiplier draws, densification/noise, and
potential nondeterministic CUDA kernels. Record Python, NumPy, and Torch seeds,
DataLoader worker count, determinism flags, CUDA/PyTorch versions, and the first
100 sampled image IDs. First repeat the exact seed/configuration to measure
execution nondeterminism. Do not describe identical nominal seed-zero repeats
as independent seeds.

If distinct seeds are required, add a minimal seed plumbing patch, validate it
on a non-training smoke test, hash it, and freeze `{0, 20260922}` **before**
either repeated LoD outcome is inspected. This instrumentation may change only
RNG initialization, not algorithms or thresholds.

### Stage B: limited replication

- repeat one Stage-A weak dataset once;
- if a naturally nontrivial dataset exists, repeat one such dataset once;
- if all three are weak, repeat the natural dataset with the largest continuous
  depth response and, separately, Run009 once only if the original data and
  full training parents can be reproduced under identical preprocessing.

The rule selects repeats from scientific classes/continuous sensitivity, not
from a desired narrative. No 3--5 seed MatrixCity campaign is planned.

### Experiment IDs

| Purpose | Stage A | Stage B repeat |
|---|---|---|
| Design | `ALOD-LOD-PREVALENCE-DESIGN-001` | — |
| Candidate geometry audit | `ALOD-LOD-PREVALENCE-CANDIDATE-AUDIT-001` | — |
| Existing MatrixCity training | `ALOD-FORMAL-TRAIN-009` | `ALOD-PREV-MC-MIXED-TRAIN-002` if selected |
| Existing MatrixCity oracle | `ALOD-MIXED-RUN009-SELECTION-ORACLE-001` | `ALOD-PREV-MC-MIXED-ORACLE-002` |
| Campus dataset/training/oracle | `ALOD-PREV-H3DGS-CAMPUS-CHUNK-01`; `...-TRAIN-001`; `...-ORACLE-001` | `...-TRAIN-002`; `...-ORACLE-002` |
| Sci-Art dataset/training/oracle | `ALOD-PREV-US3D-SCIART-01`; `...-TRAIN-001`; `...-ORACLE-001` | `...-TRAIN-002`; `...-ORACLE-002` |
| Study aggregation | `ALOD-LOD-PREVALENCE-STAGE-A-001` | `ALOD-LOD-PREVALENCE-STAGE-B-001` |
| Later ALoD-EXT | `ALOD-EXT-PREVALENCE-<DATASET>-001` | only after natural classification |
| Later synthetic control | `ALOD-SYNTH-STRONG-LOD-CONTROL-001` | design/implementation kept separate |

Ellipses in the table mean the literal dataset prefix, not an untracked alias.
Every execution receives an immutable manifest with the full expanded ID.

## 9. Expected burden

Existing timing anchors are Run009 coarse 901.11 s (15.0 min), creator 5.12 s,
and fine 54,766.97 s (15.21 h). Uni10K preflight coarse took 2,006.11 s
(33.4 min) and creator 36.42 s before its gate stopped execution.

Plan **16--20 GPU-hours per admitted Stage-A dataset**, bounded by the frozen
20 h fine and 24 h whole-pipeline hard stops, plus CPU audit, download,
preprocessing, validation, and evaluation. Campus and Sci-Art together are
approximately 32--40 GPU-hours. Stage B adds 16--40 GPU-hours depending on
whether one or two repeats are triggered. These are planning ranges, not
performance promises; actual times and every resource peak must be reported.

Storage planning must reserve input archives/images separately and keep each
monitored output below 5 GiB. Source downloads are not counted as training
output. No expensive training is part of this design task.

## 10. Later ALoD-EXT association study

Only after natural classification, choose at least one valid weak hierarchy and,
if obtained, one valid nontrivial hierarchy. Run the already frozen ALoD-EXT
implementation without tuning it. Report, per hierarchy and camera, normalized:

- materialized / semantic selected rows;
- packed-projection / semantic selected rows;
- ALoD-EXT / released-evaluator frame time;
- host-to-device byte reduction;
- semantic-selection share of total frame time.

Relate these continuous outcomes to the frozen LoD response measures. With so
few datasets, use plots, within-dataset camera summaries, and effect ranges;
avoid causal or population claims.

## 11. Planned synthetic strong-LoD control

`ALOD-SYNTH-STRONG-LOD-CONTROL-001` is a later known-positive mechanism control,
not a natural prevalence observation and not a replacement if all natural
hierarchies are weak. Its future protocol will:

1. construct valid Gaussian leaves and mathematically derived hierarchical
   parents/extents in a strict valid tree;
2. freeze near-to-far cameras and unchanged two-pixel selector semantics;
3. run the unmodified official selector and demand observed, not manually
   assigned, depth shifts;
4. pass the same selector/topology/parity checks;
5. evaluate ALoD-EXT with the same normalized workload measurements.

No camera-to-level mapping, threshold alteration, or hand-selected cut is
allowed. If all natural runs are weak but the synthetic control is nontrivial,
the defensible conclusion is that the selector/runtime can express strong LoD
while the tested source-faithful natural trainings did not. If the synthetic
control also fails, investigate selector/control construction before drawing
runtime conclusions. Synthetic results never enter the natural-dataset
frequency numerator or denominator.

## 12. Reporting commitments

The final Stage-A table must show every attempted dataset, admission result,
resource peaks, completion/validity, hierarchy rows/depth, all frozen LoD
metrics, continuous response measures, and exact outcome label. It must report
all weak outcomes if all valid runs are weak, and all nontrivial outcomes if
they occur. Failed admissions remain visible. Dataset, method, or threshold
changes made after outcome inspection require a new protocol ID and may not be
silently pooled with this study.

Primary internal authorities:

- [`alod_formal_mixed_selection_oracle.md`](../gs_framework/docs/results/alod_formal_mixed_selection_oracle.md);
- [`alod_ext_run009_closure.md`](../gs_framework/docs/results/alod_ext_run009_closure.md);
- [`alod_formal_mixed_training_run009.md`](../gs_framework/docs/results/alod_formal_mixed_training_run009.md);
- [`alod_uni10k_preflight.md`](../gs_framework/docs/results/alod_uni10k_preflight.md);
- [`formal-mixed-resource-policy-authorized.json`](../gs_framework/environments/alod-official/formal-mixed-resource-policy-authorized.json);
- [`ALOD-REPRO-MC-MIXED-01-formal-target-v3-run009.json`](../gs_framework/configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target-v3-run009.json).
