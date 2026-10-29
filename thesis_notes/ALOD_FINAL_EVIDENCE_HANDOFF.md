# ALoD Final Thesis-Evidence Handoff

This is the consolidated Layer-2 entry point for writing the ALoD dissertation
case study. It is not dissertation prose and does not replace canonical JSON,
raw records, the paper, or pinned source. Before using an exact number, reopen
the authority listed in the artifact register. For deeper implementation and
history details, use [`topics/alod.md`](topics/alod.md) and the rules in
[`THESIS_WRITING_CONTEXT.md`](../THESIS_WRITING_CONTEXT.md).

Final verdict: **READY WITH DECLARED LIMITATIONS**.

The completed evidence supports a faithful **public-source** reproduction and
a separate project-extension evaluation. It does not support exact paper-data,
paper-cache, paper-scale, or paper-number reproduction. No remaining experiment
blocks writing if those limits remain explicit.

## 1. Reproduction scope

The faithful branch covers:

- independent decoding and strict validation of the released `.dhier` format;
- official coarse, hierarchy-creator, and fine-training execution on a frozen
  public MatrixCity proxy;
- a disclosed topology-correction patch required for valid densification;
- exact official/project HSPT, frustum, ordered selection, raster-order, and
  pre-raster property parity;
- source-faithful evaluator-style host/no-cache rendering;
- released viewer-compatible previous-SPT reuse;
- a source-defined no-frustum component ablation;
- exact released HSPT cut versus a minimal source-faithful BFS reconstruction;
- held-out PSNR, SSIM, and LPIPS-VGG using released evaluator semantics;
- timing, count, host-transfer, allocator, whole-device VRAM, and RSS evidence.

Strictly excluded from the faithful claim:

- the exact paper `MC-small-city+` asset, added views, merged sparse geometry,
  camera paths, sampling seed, or trained hierarchy;
- the paper's full configuration, hardware, scale, renderer, or numerical
  results;
- paper `Render Full`, because no released rendering entry point implements
  its bounded retained cache;
- any project row-LRU, trace-derived capacity, or pre-materialization filter;
- an assertion that the relocation repair equals an unreleased author fix.

Use the labels **paper claim**, **released-source behavior**, **`ALOD-REPRO`**,
**source-inspired adaptation**, and **`ALOD-EXT`**. Reproduction-support code
such as parsers, oracles, shims, and corrections is neither a paper claim nor a
project rendering method.

## 2. Implementation and experiment chronology

1. The 2026-08-05 paper/source audit mapped the complete public pipeline and
   found no pretrained `.dhier` or paper result bundle.
2. An isolated Python/CUDA environment plus launcher-only compatibility fixes
   resolved DrvFS multiprocessing, Blackwell JIT, and PyTorch optimizer API
   failures in smoke Runs001--003.
3. Smoke Run004 finished training but produced invalid reverse topology.
   `ALOD-RELOCATE-TOPOLOGY-FIX` repaired redundant parent/depth metadata;
   smoke Run005 became the first accepted local hierarchy.
4. Run005 enabled format parity, an official selection oracle, an independent
   CPU reference, resident runtime, host residency, and early materialization
   experiments. These are historical correctness/development evidence.
5. The 2026-08-20 reproduction-boundary audit separated released evaluator,
   viewer, and trainer cache semantics and froze the missing paper `Render
   Full` path as `PAPER-SOURCE-DIVERGENCE`.
6. `ALOD-REPRO-MC-MIXED-01` froze a mixed aerial/street public proxy, official
   ingestion, split, evaluation cameras, and diagnostic cameras.
7. Formal Run001 stopped before optimization because its inherited WSL monitor
   could not enforce per-process VRAM.
8. Formal Run002 completed 30k coarse training and hierarchy creation. Literal
   `SPT_root_volume=150` produced one oversized SPT and failed pre-fine cache
   feasibility. Its outputs became immutable parent inputs.
9. A source-documented, result-blind SPT sweep selected 4.75.
10. Formal Runs003--005 failed due to compilation process accounting, forked
    worker RSS accounting, and provenance writing. They are infrastructure
    history, not method results.
11. Formal Run006 stopped at iteration 13,500 when densification exceeded the
    1.6M reduced cache by 1,132 rows; topology and machine limits passed.
12. A resource calibration increased only the released fine-training cache
    capacities to 6.25M/5.0M. Run007 was interrupted by a WSL restart; Run008
    by destruction of a login-scoped user service manager. Neither failure was
    methodological.
13. Run009 restarted from the frozen Run002 parents, completed fine training,
    and passed independent and official hierarchy gates. A final snapshot-name
    collision was resolved by read-only revalidation.
14. The final 16-camera official oracle was deterministic and exactly matched
    by the project CPU reference. It classified the hierarchy
    `WEAK-LOD-ORACLE`.
15. The 71-camera faithful campaign measured no-cache, no-frustum, viewer
    reuse, quality, HSPT/BFS, and LoD coverage.
16. The final Run009 `ALOD-EXT` campaign evaluated six project-specific
    materialization/residency variants against frozen faithful baselines.

Chronology authority: [`topics/alod.md`](topics/alod.md), Run009
[`provenance.json`](../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/provenance.json),
and the final faithful/extension package summaries.

## 3. Pinned identities

### Source

- Paper: arXiv:2507.01110v4.
- Official repository: `/home/luis/tese/LoDOfGaussians`.
- Commit: `73f547341547ae021ff036ad74fdf73a1240b450`.
- Interpretation authorities:
  [`method card`](../gs_framework/docs/papers/a-lod-of-gaussians-method-card.md),
  [`source map`](../gs_framework/docs/papers/a-lod-of-gaussians-source-map.md),
  and [`reproduction boundary`](../gs_framework/docs/results/alod_reproduction_boundary.md).

### Dataset and cameras

- Dataset ID: `ALOD-REPRO-MC-MIXED-01`.
- Manifest SHA-256:
  `eaf3cb228234efd92441d28ea56482dcf4676c5a318a6d6beef1020ee27c5f7a`.
- Manifest counts: 710 cameras, 639 train, 71 held out, 335 aerial, 375
  street, and 2,317,128 sparse points.
- Evaluation trajectory:
  `ALOD-REPRO-MC-MIXED-01-EVAL-71-DATASET-INDEX-V1`.
- Evaluation-camera SHA-256:
  `4c77176a4f9e72baf806ceb358c8cad800941c1e7fb5036244a003d2e35d4ceb`.
- Diagnostic-camera SHA-256:
  `35ae984baad7bd5ebba2db2abdb9c3ac9c3f17a93090f00969c8e18c11e771a3`.

Authorities: dataset [`manifest`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/manifest.json),
[`split`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/split.json),
[`source provenance`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/source_provenance.json),
[`formal cameras`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/formal_evaluation_cameras.json),
and [`diagnostic cameras`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/diagnostic_cameras.json).

### Configuration and hierarchy

- Run009 config SHA-256:
  `f737e50e023c6cea6dfb33454007a026f3e2f241721cdc978866ac0731eaf2eb`.
- Final hierarchy SHA-256:
  `38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b`.
- Final structure: 7,072,627 total rows; 100,000 skybox; 6,972,627
  hierarchy; SH1; root global ID 100,000; maximum depth 55.
- Executed schedule: 30k coarse, 100k fine, classic densification, 12M cap.
- Calibrated training settings: `SPT_root_volume=4.75`, cache 6.25M/5.0M.
- Evaluation HSPT: effective `min_SPT_size=100`, 335 upper rows, 26 SPTs,
  6,972,318 flattened SPT rows.

Authorities: [`Run009 config`](../gs_framework/configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target-v3-run009.json),
[`final hierarchy`](../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-hierarchy-independent.json),
and oracle [`formal analysis`](../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/formal-analysis.json).

## 4. Faithful reproduction stages

| Stage | Final classification | Acceptance evidence |
| --- | --- | --- |
| `.dhier` decoding and official-load transformations | Reproduction support | Independent format/topology parser and official loader parity. |
| Coarse training and hierarchy creation | `ALOD-REPRO`, proxy scoped | Frozen Run002 parent artifacts reused by Run009. |
| Fine hierarchical training | `ALOD-REPRO`, qualified | Released mechanism with dataset/resource calibration, compatibility support, and topology correction. |
| Final hierarchy | `ALOD-REPRO` artifact | Strict independent topology plus official load/compact/HSPT round trip. |
| HSPT/frustum/ordered selection | `ALOD-REPRO` boundary | Exact 16-camera official oracle and independent CPU reference parity. |
| `ALOD-REPRO-HOST-NOCACHE` | Source-faithful project reproduction | Released evaluator selection and per-frame host gather/transfer; project instrumentation disclosed. |
| `ALOD-REPRO-NOFRUSTUM` | Source-defined component ablation | Only early four-plane predicate changes; not paper Table 3 because no bounded cache. |
| `ALOD-REPRO-PREVIOUS-REUSE` | Released viewer behavior | Strict compatible-SPT reuse from immediate previous working set; no capacity/LRU/write-back. |
| `ALOD-REPRO-CUT-HSPT` | Exact released cut path | Cut-only timing on diagnostic cameras. |
| `ALOD-REPRO-CUT-BFS` | Minimal source-faithful reconstruction | No released BFS evaluator; identity mismatch retained. |
| `ALOD-REPRO-QUALITY` | Source-faithful quality path | Released selection, packed renderer, and metric/alpha semantics on 71 held-out cameras. |

The authoritative faithful mode definitions and corrections are in
[`alod_faithful_reproduction.md`](../gs_framework/docs/results/alod_faithful_reproduction.md)
and the campaign [`provenance`](../benchmark_artifacts/alod_faithful_reproduction_001/provenance.json).

## 5. Paper/source divergences

Preserve all of these in dissertation drafting:

- The paper dataset is not reconstructible: exact lists, 533 added
  high-altitude views, crop, merged point cloud, factor-five sample seed,
  trained hierarchy, and paths are missing. Its supplementary training-view
  components total 41,211 while the main table says 42.2k images.
- The local proxy uses cropped released aerial sparse geometry, not the paper's
  merged aerial/street cloud.
- Paper coarse training is described as 100k iterations; public configs use
  30k/60k/250k. The proxy used 30k.
- No released rendering program implements paper `Render Full` with a bounded
  retained SPT cache. Evaluation, viewer, and training expose different cache
  semantics.
- Paper cache capacity is 15M; source MatrixCity is 25M/20M; Run009 training
  is 6.25M/5.0M. These are not interchangeable.
- Training and viewer reuse tolerances differ (approximately 0.5--1.3 versus
  strict 0.3--1.5); evaluator has none.
- Released training uses ordered retained SPTs without explicit age counters,
  counts the assembled row list in its trigger, and writes mutable state back.
  Viewer has no bounded retention, write-back, or flush.
- Source frustum uses only left/right/top/bottom planes; near/far are inactive;
  propagated spheres default off and the scale proxy is used.
- Paper view exploration is every 128 iterations; source uses 100, overwrites
  `k` with 100, uses `1/(distance+20)`, and disables graph scheduling for
  MatrixCity/formal training.
- Released fixed packed-RGB gsplat is documented as more memory-efficient than
  the paper renderer and ignores some wrapper arguments.
- Viewer focal/replay/timing/CLI behavior prevents treating it as the paper
  benchmark harness.
- No official BFS evaluation executable exists; local BFS is reconstructed and
  does not preserve HSPT selected sets.
- HSPT is transient. Initial training passes `min_SPT_size=256`, while source
  rebuild/evaluation calls default to 100.
- `get_SPT_cut` overwrites its condition with all true; inline train/eval/viewer
  paths, not that helper, define the oracle.
- `.dhier` has no magic, version, checksum, or endian marker; the loader assumes
  equal Gaussian/node counts and zeroes metadata column five.
- Creator-only artifacts can contain unused positive leaf `first_child` values
  and interior opacity above one before load-time clamp; final hierarchy
  validation is strict.
- No pretrained `.dhier`, exact paper result bundle, or Table-3 paths were
  released.

Detailed source authority:
[`alod_reproduction_boundary.md`](../gs_framework/docs/results/alod_reproduction_boundary.md).

## 6. Topology-correction boundary

Unpatched `relocate_gs` can leave stale reverse parent/depth metadata below a
promoted interior sibling and can corrupt skybox metadata when a promoted leaf
routes through row zero. Smoke Run004 recorded 4,355 depth mismatches, one
reverse-parent mismatch, and skybox corruption in
[`final-barrier-failure.json`](../runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/ALOD-SMOKE-TRAIN-004/failure/final-barrier-failure.json).

[`ALOD-RELOCATE-TOPOLOGY-FIX.patch`](../gs_framework/patches/alod/ALOD-RELOCATE-TOPOLOGY-FIX.patch)
recursively repairs redundant reverse metadata only below affected forward
roots. Differential and real-CUDA validation preserve properties, sampling,
respawn behavior, forward topology, objectives, optimizer/densification,
HSPT/cache, selection, and rendering.

Required wording: Run009 is a **correctness-patched public-source
reproduction**. The patch restores validated internal invariants while
preserving observed forward method decisions. It is not known to be identical
to an unreleased author fix.

## 7. Canonical artifact register

| Exact claim family | Authoritative artifact |
| --- | --- |
| Dataset counts/identity | [`manifest.json`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/manifest.json), [`split.json`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/split.json) |
| Source/coordinate/geometry provenance | [`source_provenance.json`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/source_provenance.json) |
| Camera identities/order/resolutions | [`formal_evaluation_cameras.json`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/formal_evaluation_cameras.json), [`diagnostic_cameras.json`](../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/diagnostic_cameras.json) |
| Executed training config | [`formal-target-v3-run009.json`](../gs_framework/configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target-v3-run009.json) |
| Training completion/resources | [`Run009 provenance`](../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/provenance.json), [`completion audit`](../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-completion-audit.json) |
| Hierarchy bytes/rows/topology | [`final-hierarchy-independent.json`](../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-hierarchy-independent.json) |
| Official load/compact/HSPT validation | [`final-official-loader-hspt.json`](../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-official-loader-hspt.json) |
| Exact final selection parity | [`completion-report.json`](../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/completion-report.json), [`project-reference-parity.json`](../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/project-reference-parity.json) |
| 16-camera LoD classification | [`formal-analysis.json`](../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/formal-analysis.json) |
| Faithful completion/protocol | [`ALOD-FAITHFUL-REPRODUCTION.json`](../benchmark_artifacts/alod_faithful_reproduction_001/ALOD-FAITHFUL-REPRODUCTION.json), [`provenance.json`](../benchmark_artifacts/alod_faithful_reproduction_001/provenance.json) |
| Faithful runtime/counts | [`rendering_summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json) |
| Faithful memory/H2D | [`memory_summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/memory_summary.json), [`h2d_summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/h2d_summary.json) |
| Frustum | [`frustum_ablation.json`](../benchmark_artifacts/alod_faithful_reproduction_001/tables/frustum_ablation.json) |
| HSPT/BFS | [`hspt_bfs/summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/hspt_bfs/summary.json) |
| Held-out quality | [`quality/results/summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/quality/results/summary.json) |
| 71-camera LoD characterization | [`lod_coverage_71/summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/lod_coverage_71/summary.json) |
| Extension correctness/capacity | [`correctness.json`](../benchmark_artifacts/alod_ext_formal_run009_001/checks/correctness.json), [`cache_sizing.json`](../benchmark_artifacts/alod_ext_formal_run009_001/checks/cache_sizing.json) |
| Final extension results/attribution | [`formal_summary.json`](../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json) |

Never use a prose memo to settle a numerical conflict when the listed JSON is
available. In particular, selected hierarchy counts exclude the 100,000
skybox rows; semantic/raster totals may include them.

## 8. Faithful results

Protocol: three fresh processes per rendering mode, ten ordered warmups, and
71 measured cameras per process, for 213 measured frames per mode. The exact
protocol and completion gate are in
[`ALOD-FAITHFUL-REPRODUCTION.json`](../benchmark_artifacts/alod_faithful_reproduction_001/ALOD-FAITHFUL-REPRODUCTION.json).

| Mode | Mean / median / p95 end-to-end | FPS | Mean selected hierarchy | Mean H2D rows | Authority |
| --- | ---: | ---: | ---: | ---: | --- |
| `ALOD-REPRO-HOST-NOCACHE` | 186.77 / 184.87 / 207.52 ms | 5.35 | 3,432,141 | 3,432,141 | [`rendering_summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json) |
| `ALOD-REPRO-NOFRUSTUM` | 188.02 / 186.16 / 210.13 ms | 5.32 | 3,442,923 | 3,442,923 | same |
| `ALOD-REPRO-PREVIOUS-REUSE` | 134.21 / 131.58 / 160.44 ms | 7.45 | 3,456,309 | 87,570 | same |

No-cache spends 182.34 ms in preparation and 4.15 ms in rasterization; host
gather is 93.05 ms, H2D 19.27 ms, and semantic selection 38.09 ms. Mean raster
input is 3,532,141 including skybox; packed projection survivors average
496,182. Viewer reuse saves 52.57 ms (28.14%) but retains prior properties.
Authority: [`rendering summary`](../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json)
and [`H2D summary`](../benchmark_artifacts/alod_faithful_reproduction_001/h2d_summary.json).

No-cache persistent property residency is the 0.009 GiB skybox versus a 0.606
GiB host property table; frame-local properties peak at 0.605 GiB. Maximum
PyTorch frame peak is 1.281 GiB allocated and 2.387 GiB reserved; whole-device
VRAM is 3.963 GiB and RSS 3.810 GiB. Authority:
[`memory_summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/memory_summary.json).

Frustum avoids 10,782 semantic rows (0.313%) and 5.03 candidate SPTs on
average, but 0.155 packed survivors. The pooled 1.25 ms/0.664% saving lies
inside large paired variability; the strong paper trend is not established.
Authority: [`frustum_ablation.json`](../benchmark_artifacts/alod_faithful_reproduction_001/tables/frustum_ablation.json).

HSPT/BFS cut time is 36.00/91.54 ms and logical metadata
83,685,344/396,067,112 bytes, but mean selected-set Jaccard is 0.0970. Report
the favorable speed/memory direction and failed cut equivalence together.
Authority: [`hspt_bfs/summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/hspt_bfs/summary.json).

| Split | Cameras | PSNR mean/median | SSIM mean/median | LPIPS-VGG mean/median | Authority |
| --- | ---: | ---: | ---: | ---: | --- |
| All | 71 | 21.495 / 21.183 | 0.6903 / 0.6913 | 0.3972 / 0.4157 | [`quality summary`](../benchmark_artifacts/alod_faithful_reproduction_001/quality/results/summary.json) |
| Aerial | 26 | 23.255 / 23.775 | 0.7352 / 0.7781 | 0.3088 / 0.3095 | same |
| Street | 45 | 20.478 / 20.882 | 0.6644 / 0.6718 | 0.4483 / 0.4593 | same |

Across 71 cameras, selected hierarchy count ranges 2,606,659--3,478,844, yet
all cameras reach depth 55 and have median depth 22; mean depth occupies only
22.8484--23.0579. This is final `WEAK-LOD` evidence, not strong adaptive-depth
evidence. Authority: [`lod_coverage_71/summary.json`](../benchmark_artifacts/alod_faithful_reproduction_001/lod_coverage_71/summary.json).

## 9. Project-extension results — strictly separate

Every mode here is `ALOD-EXT`. None is paper `Render Full` or faithful ALoD.
The extensions preserve the exact semantic cut and optionally reduce
full-property materialization after selection.

Correctness: exact oracle IDs, touched SPTs, bitwise surviving properties and
RGB/alpha, zero backend false negatives, and exact CPU/CUDA filtered IDs on all
71 workload cameras. Authority:
[`correctness.json`](../benchmark_artifacts/alod_ext_formal_run009_001/checks/correctness.json).

Mean count funnel: 3,532,141 semantic rows including skybox, 525,049
materialized rows, and 496,182 packed survivors; semantic/materialized is
6.727:1. Authority:
[`formal_summary.json`](../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json).

| `ALOD-EXT` mode | Mean / median / p95 | FPS | Authority |
| --- | ---: | ---: | --- |
| Row-LRU practical | 134.54 / 132.68 / 145.60 ms | 7.43 | [`formal summary`](../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json) |
| CPU filter practical | 220.50 / 213.79 / 265.99 ms | 4.54 | same |
| CUDA filter + previous | 71.66 / 66.72 / 124.03 ms | 13.96 | same |
| CUDA filter + minimum | 84.38 / 84.73 / 138.03 ms | 11.85 | same |
| CUDA filter + practical | 73.48 / 68.94 / 98.73 ms | 13.61 | same |
| CUDA filter + large | 76.29 / 71.99 / 101.34 ms | 13.11 | same |

The fastest measured extension, CUDA filter plus previous-set reuse, saves
115.12 ms (61.63%) and is 2.606x faster than faithful no-cache. Semantic
selection remains the bottleneck at 38.19 ms, 53.30% of total. This conclusion
is limited to the frozen `WEAK-LOD` workload.

Capacities were fixed before timing from the post-filter trace: maximum frame
1,599,906 rows, trajectory union 3,503,048, practical cache 3,700,000, and
large sensitivity cache 5,300,000. Authority:
[`cache_sizing.json`](../benchmark_artifacts/alod_ext_formal_run009_001/checks/cache_sizing.json).

## 10. Supersession chain

1. Initial audits remain source-interpretation authorities only.
2. Smoke Runs001--004 are failure evidence; smoke Run005 supersedes them for
   the first valid local hierarchy.
3. Run005 remains the small historical correctness oracle. Run009 supersedes
   it for final hierarchy identity; the Run009 oracle and faithful/extension
   packages supersede it for final results.
4. Formal volume 150 is superseded by 4.75. The initial 2.0M/1.6M training
   cache is superseded by 6.25M/5.0M. Neither pair identifies paper rendering
   cache behavior.
5. Formal Runs001--008 are failed/interrupted stages. Run009 is the final
   hierarchy authority.
6. A one-camera `DEGENERATE` smoke classification is superseded by the complete
   16-camera `WEAK-LOD-ORACLE` result.
7. Historical exact-row previous reuse is not final
   `ALOD-REPRO-PREVIOUS-REUSE`, which uses released compatible-SPT semantics.
8. Historical Run005 525k/160k capacities and associated results remain
   engineering history. Run009 `cache_sizing.json` and `formal_summary.json`
   are final extension authorities.
9. Prose-rounded numbers never supersede canonical JSON. A 100,000-row
   difference usually indicates whether skybox rows are included.

## 11. Safe claims

After reopening the cited authority, the dissertation may state that:

- the pinned public evaluator's hierarchy selection and pre-raster assembly
  were reproduced exactly on 16 frozen cameras;
- a correctness-patched official pipeline produced and validated the Run009
  hierarchy on a frozen public proxy;
- evaluator-style no-cache execution is dominated by selection, host gather,
  and transfer rather than rasterization;
- released viewer-compatible SPT reuse reduces transfer and end-to-end time at
  the cost of persistent prior-working-set residency;
- released early frustum culling removes semantic work but did not establish a
  robust end-to-end benefit locally;
- HSPT was faster and required less logical metadata than reconstructed BFS,
  while their selected sets differed materially;
- the hierarchy is structurally valid but weak in camera-dependent depth-cut
  variation;
- the project CUDA pre-materialization extension preserves semantic selection
  and reduces full-property work under the frozen protocol.

## 12. Claims requiring qualification

- Every exact result is specific to the public proxy, frozen cameras,
  released packed gsplat path, local RTX 5080/WSL environment, and declared
  corrections.
- Run009 training is public-source faithful only with source-documented
  dataset/resource calibration and the topology correction.
- Quality uses the official evaluator's alpha and metric semantics; it is not
  the paper dataset result.
- `WEAK-LOD` limits generalization to stronger hierarchies and wider camera
  scale ranges.
- PyTorch allocated/reserved, tensor bytes, whole-device NVML, and RSS are
  different measurements.
- The extension speedup is relative to frozen faithful project no-cache, not
  paper `Render Full`.
- The no-frustum mode includes a minimal Boolean-device execution correction.
- HSPT/BFS is trend evidence, not an exact controlled semantic substitution.

## 13. Prohibited claims

Do not claim:

- complete paper reproduction or equality with paper quality/FPS/memory;
- that the proxy is `MC-small-city+` or uses its merged geometry;
- that paper `Render Full` was implemented or measured;
- that row-LRU, trace-derived caches, or materialization filters are faithful
  ALoD;
- that Run009 is unchanged-source or bit-identical official output;
- that the topology patch is the authors' intended fix;
- that the final proxy demonstrates strong adaptive hierarchy-depth behavior;
- that local HSPT and BFS produced equivalent cuts;
- that frustum culling significantly accelerates this proxy;
- that packed projection survivors are final pixel contributors;
- that selected, semantic, materialized, resident, transferred, projected, and
  rendered counts are interchangeable;
- that Run005 capacities/performance are final Run009 evidence;
- that local absolute results reproduce paper-scale 60M--150M behavior;
- an unconditional CityGaussian-versus-ALoD ranking from current native runs.

## 14. Limitations and threats to validity

- Exact paper assets, paths, added views, preprocessing, and random sampling
  are unavailable.
- Proxy sparse geometry differs materially from the paper construction.
- Evidence uses one trained hierarchy and one local machine/runtime stack.
- The formal hierarchy is `WEAK-LOD` despite mixed camera modalities.
- A larger Uni10K candidate reached a valid initial hierarchy and official
  HSPT reconstruction but failed the prospectively frozen fine-training
  resource-admission gate; stronger-LoD external validity remains untested.
- Hardware, renderer implementation, and software differ from paper results.
- WSL complicates process-attributed GPU accounting; NVML is whole-device.
- Training required a correctness patch and non-methodological compatibility
  support.
- The paper bounded rendering cache is unreleased.
- The BFS comparator is reconstructed and semantically non-equivalent locally.
- The no-frustum source branch required a device-placement correction.
- Paper-scale host/GPU capacity and absolute timing were not tested.
- Current `gs_framework` implementation files live in a dirty/untracked nested
  repository; reproducibility relies on formal package hashes and provenance,
  not the current nested HEAD alone.

## 15. Optional remaining experiments

None is required before writing the current reproduction and extension
chapters. Optional work, in decreasing evidential value:

1. obtain author assets, camera paths, or an official bounded render-cache
   entry point if released;
2. only if materially justified, redesign a resource envelope prospectively
   for another stronger-LoD/second-scene attempt; the Uni10K attempt already
   stopped at its frozen 15M-cap/20%-headroom gate and should not be repeated
   automatically;
3. reconstruct a clearly labelled paper-cache approximation;
4. repeat the formal campaign on another hardware/runtime environment;
5. evaluate temporal LoD quality, popping, and interactive motion;
6. optimize semantic selection as a new `ALOD-EXT` stage with exact oracle
   parity as a hard gate;
7. build a shared-asset/shared-camera cross-method experiment before any
   numerical CityGaussian/ALoD ranking.

## 16. Recommended dissertation narrative

1. Introduce ALoD's global renderable hierarchy, HSPT, view-dependent cut, and
   host/GPU working-set model as paper claims.
2. Audit the released evaluator, viewer, trainer cache, frustum, renderer, and
   format separately from the paper.
3. State the missing `Render Full` entry point and unavailable paper dataset
   before presenting local experiments.
4. Explain the public mixed-scale proxy and frozen identities.
5. Present the topology failure and scoped correction as a reproducibility
   finding, not a new method.
6. Describe the formal training attempts as a fail-closed progression from
   source-literal settings to documented dataset/resource calibration.
7. Establish exact official/project selection and pre-raster parity before
   performance claims.
8. Present evaluator-style no-cache as the faithful runtime baseline and show
   that property preparation/movement dominates.
9. Present viewer-compatible reuse as released behavior, distinct from paper
   bounded caching.
10. Report frustum's weak local effect and HSPT/BFS identity mismatch as
    negative results.
11. Report held-out quality and the 71-camera `WEAK-LOD` limitation together.
12. Report the Uni10K attempt briefly as a resource-feasibility negative
    preflight: structural/HSPT validation passed, but fine training and LoD
    evaluation were not reached.
13. Close the reproduction section with separate implementation,
    experimental, and result reproducibility assessments.
14. Begin a new, explicitly labelled project-extension section before any
    row-LRU or pre-materialization result.
15. Use the semantic-to-materialized-to-projected count funnel to motivate the
    extension, then report correctness, performance, memory/traffic trade-offs,
    and the remaining semantic-selection bottleneck.

Recommended method-level assessment:

- implementation reproducibility: **strong within the pinned released-source
  boundary**;
- experimental reproducibility: **strong for the local packages, partial
  versus the paper**;
- result reproducibility: **partial**.

## 17. Writing checklist

Before drafting or approving each ALoD passage:

1. Identify the sentence as paper claim, released behavior, `ALOD-REPRO`,
   source-inspired adaptation, reproduction support, or `ALOD-EXT`.
2. Reopen the paper for paper claims and the pinned source/audit for released
   behavior.
3. Reopen the exact JSON row in Section 7 for every numerical statement.
4. Verify dataset, source commit, hierarchy hash, config hash, camera fixture,
   resolution, and mode.
5. State whether counts include skybox and distinguish selected hierarchy,
   semantic, materialized, H2D, resident, and packed-survivor populations.
6. State the timing boundary and derive FPS only from end-to-end mean time.
7. Keep property bytes, PyTorch allocated/reserved, whole-device VRAM, and RSS
   separate.
8. Preserve the relocation and no-frustum corrections.
9. Preserve `PAPER-SOURCE-DIVERGENCE` for bounded rendering cache.
10. Pair favorable HSPT efficiency with the selected-set mismatch.
11. Pair all performance/quality results with the `WEAK-LOD` and proxy limits.
12. Put paper numbers and project numbers in separately labelled columns and
    treat them as paper-reference comparisons.
13. Do not use Run005 metrics/capacities for final Run009 claims.
14. Start a separate subsection before introducing any `ALOD-EXT` result.
15. Check [`COMPARISON_VALIDITY.md`](COMPARISON_VALIDITY.md) before any
    cross-method table or improvement percentage.
16. Do not edit the dissertation until the claim-to-artifact trace is explicit.

## 18. Final verdict

**READY WITH DECLARED LIMITATIONS**

There is no evidence blocker to writing. The unresolved paper bounded cache,
unavailable exact paper dataset/assets/paths, proxy geometry divergence,
`WEAK-LOD` hierarchy, reconstructed BFS mismatch, and local hardware/backend
differences are declared limitations rather than missing local validation.
The remaining experiments in Section 15 are optional and would broaden
external validity or paper comparability; they are not prerequisites for an
honest dissertation chapter based on the current evidence. The first larger
stronger-LoD candidate has already been tested through preflight and stopped as
a resource-feasibility negative result rather than being misclassified as weak
LoD.

## 19. Uni10K external-validity preflight freeze

`ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT` asked whether a larger public-source
candidate could reach the strong-LoD gate needed to retest Run009 `ALOD-EXT`.
RUN005 completed 30,000 coarse iterations, hierarchy creation, independent
validation, structure-only SPT calibration, and official-loader HSPT
reconstruction. Its 12,341,309-row initial hierarchy is valid; the calibrated
HSPT has 56 SPTs and 1,997 upper rows.

The experiment stopped before fine training. With the 100,000-row skybox, the
12,441,309-row fine initial state leaves 2,558,691 rows, or 17.05794%, under
the frozen 15M cap and therefore fails the predeclared 20% headroom gate. The
classification is `BLOCKED-ALOD-UNI10K-FINE-HEADROOM`: a resource-feasibility
negative preflight, not `WEAK-LOD` and not an algorithm or implementation
failure. Strong-LoD behavior and `ALOD-EXT` external validity remain untested.

Full dissertation placement: short negative-result subsection or paragraph in
ALoD evaluation limitations/external validity. Condensed-paper placement:
limitations/discussion only, at most one sentence. Safe claim: a larger
candidate passed structural/HSPT preflight but not the frozen fine-training
resource-admission criterion. Prohibited claim: Uni10K was weak-LoD, ALoD
failed, or `ALOD-EXT` was validated on a stronger hierarchy.

Authority: [`topics/alod.md`](topics/alod.md),
[`alod_uni10k_preflight.md`](../gs_framework/docs/results/alod_uni10k_preflight.md),
and machine-readable
[`closure.json`](../runs/alod/ALOD-REPRO-UNI10K-01/ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT-RUN005-CLOSURE/closure.json).
