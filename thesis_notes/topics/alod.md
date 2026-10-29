# ALoD Evidence Note

This is the detailed Layer-2 evidence index for *A LoD of Gaussians* (ALoD),
not dissertation prose. The writing-ready synthesis is
[`ALOD_FINAL_EVIDENCE_HANDOFF.md`](../ALOD_FINAL_EVIDENCE_HANDOFF.md). Reopen
the linked machine-readable authority before using an exact number, following
[`THESIS_WRITING_CONTEXT.md`](../../THESIS_WRITING_CONTEXT.md).

## 1. Final status and reproduction scope

Verdict: **READY WITH DECLARED LIMITATIONS**.

The project reproduced the **pinned public-source** ALoD hierarchy-selection
and cold-evaluation behavior on a frozen public MatrixCity mixed aerial/street
proxy. The evidence includes correctness-patched official training, strict
`.dhier` validation, exact official/project selection and pre-raster parity,
source-faithful no-cache rendering, released viewer previous-compatible-SPT
reuse, a source-defined no-frustum ablation, an HSPT versus reconstructed BFS
study, held-out quality, and a separate `ALOD-EXT` campaign.

This is not an exact reproduction of the paper's data, assets, paths,
hardware, renderer, configuration, scale, or results. No released rendering
entry point implements the paper's bounded-cache `Render Full` mode. The final
proxy is `WEAK-LOD`: it supports implementation and local systems claims, but
not strong hierarchy-depth adaptation or paper-scale generalization.

Canonical completion authorities:

- [`ALOD-FAITHFUL-REPRODUCTION.json`](../../benchmark_artifacts/alod_faithful_reproduction_001/ALOD-FAITHFUL-REPRODUCTION.json);
- [`ALOD-EXT-FORMAL-RUN009.json`](../../benchmark_artifacts/alod_ext_formal_run009_001/ALOD-EXT-FORMAL-RUN009.json);
- [`alod_reproduction_boundary.md`](../../gs_framework/docs/results/alod_reproduction_boundary.md).

## 2. Evidence labels and authority

| Label | Meaning |
| --- | --- |
| **Paper claim** | Statement or result from arXiv:2507.01110v4; the paper is authoritative. |
| **Released-source behavior** | Behavior of `FelixWindisch/LoDOfGaussians` at `73f547341547ae021ff036ad74fdf73a1240b450`; it may differ from the paper. |
| **`ALOD-REPRO`** | Project execution that reproduces a pinned released-source boundary with declared instrumentation/corrections. |
| **Source-inspired adaptation** | ALoD component used with changed residency, selection, or reuse semantics. |
| **`ALOD-EXT`** | Project-specific materialization, cache, capacity, or runtime policy absent from paper/source. |
| **Reproduction support** | Parsers, oracles, instrumentation, compatibility shims, and correctness repairs. |

Raw runs/source and canonical JSON outrank prose. Historical IDs remain
immutable even when their conceptual class is clarified.

## 3. Pinned identities

### Paper and source

- Paper: arXiv:2507.01110v4, *A LoD of Gaussians: Out-of-Core Training and
  Rendering for Seamless Ultra-Large Scene Reconstruction*.
- Source: `/home/luis/tese/LoDOfGaussians`, commit
  `73f547341547ae021ff036ad74fdf73a1240b450`.
- Audits: [`method card`](../../gs_framework/docs/papers/a-lod-of-gaussians-method-card.md)
  and [`source map`](../../gs_framework/docs/papers/a-lod-of-gaussians-source-map.md).

The title differs from older repository wording; always identify the pinned
paper version and source commit.

### Dataset and cameras

`ALOD-REPRO-MC-MIXED-01` is a public MatrixCity mixed aerial/street proxy.
Its canonical [`manifest`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/manifest.json),
SHA-256 `eaf3cb228234efd92441d28ea56482dcf4676c5a318a6d6beef1020ee27c5f7a`,
records 710 cameras: 639 train and 71 held out, 335 aerial and 375 street, and
2,317,128 cropped aerial sparse points. Exact identities and construction are
in the [`split`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/split.json),
[`source provenance`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/source_provenance.json),
and [`ingestion gate`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/ingestion_validation.json).

The 71-camera trajectory is
`ALOD-REPRO-MC-MIXED-01-EVAL-71-DATASET-INDEX-V1`, ordered by
`(dataset_index, dataset_filename)`. Its
[`camera fixture`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/formal_evaluation_cameras.json)
has SHA-256 `4c77176a4f9e72baf806ceb358c8cad800941c1e7fb5036244a003d2e35d4ceb`;
aerial renders are 960x540 and street renders 500x500. The independent
16-camera [`diagnostic fixture`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/diagnostic_cameras.json)
has SHA-256 `35ae984baad7bd5ebba2db2abdb9c3ac9c3f17a93090f00969c8e18c11e771a3`.
Both were frozen before result-driven selection.

### Configuration and final hierarchy

The final [`Run009 configuration`](../../gs_framework/configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target-v3-run009.json),
SHA-256 `f737e50e023c6cea6dfb33454007a026f3e2f241721cdc978866ac0731eaf2eb`,
specifies 30,000 coarse and 100,000 fine iterations, SH1, classic
densification, a 12M cap, `SPT_root_volume=4.75`, two-pixel granularity,
initial `min_SPT_size=256`, host properties, and the released fine-training
cache mechanism sized 6.25M/5.0M rows. The HSPT and capacity values are
dataset/resource calibrations, not paper constants or render-cache results.

The final `ALOD-REPRO-MC-MIXED-01-FORMAL-TRAIN-009.dhier` has SHA-256
`38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b`.
It contains 7,072,627 serialized rows: 100,000 skybox and 6,972,627 reachable
hierarchy rows; SH degree 1; root row 100,000; maximum depth 55. Authority:
[`final-hierarchy-independent.json`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-hierarchy-independent.json).
Run/config/resource identity is in the Run009
[`provenance`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/provenance.json).

Evaluation rebuilds transient HSPT metadata with volume 4.75, resolved
granularity `0.0008629449482830516`, effective `min_SPT_size=100`, distance
multiplier 1.0, four side planes, inactive near/far planes, and scale-proxy
bounds. It yields 335 upper rows, 26 SPTs, and 6,972,318 flattened SPT rows.
Authority: oracle [`formal-analysis.json`](../../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/formal-analysis.json).
The evaluation value 100 follows the source default because densification
rebuild/evaluation omit the initially explicit value 256.

## 4. Current implementation boundary

The current working tree was inspected. It separates:

- independent `.dhier` parsing/validation in [`alod_dhier/`](../../gs_framework/alod_dhier/);
- independent CPU HSPT/frustum/selection in [`alod_reference/`](../../gs_framework/alod_reference/);
- production HSPT/selection in [`hspt.py`](../../gs_framework/scene_runtime/alod/hspt.py)
  and [`selection.py`](../../gs_framework/scene_runtime/alod/selection.py);
- host properties and assembly in [`host.py`](../../gs_framework/scene_runtime/alod/host.py)
  and [`host_scene.py`](../../gs_framework/scene_runtime/alod/host_scene.py);
- no-cache, exact-row previous reuse, viewer-compatible SPT reuse, and bounded
  row-LRU in [`residency.py`](../../gs_framework/scene_runtime/alod/residency.py);
- released viewer reuse/order semantics in [`viewer_reuse.py`](../../gs_framework/scene_runtime/alod/viewer_reuse.py);
- optional CPU/CUDA post-selection filters in [`materialization.py`](../../gs_framework/scene_runtime/alod/materialization.py);
- explicit runtime modes and metrics in [`loader.py`](../../gs_framework/scene_runtime/loader.py)
  and [`benchmark_real_scene.py`](../../gs_framework/benchmarks/benchmark_real_scene.py).

Thus semantic selection, materialization, residency, assembly, and backend
projection counts are distinct. Resolved configuration/provenance determines
fidelity, not file reuse. The nested `gs_framework` tree is dirty/untracked;
formal packages pin relevant file/config hashes rather than relying only on
its current HEAD.

## 5. Chronological implementation and experiment history

1. **2026-08-05 audit.** Paper/source mapping identified the hierarchy,
   coarse/creator/fine pipeline, HSPT, runtime modes, renderer, and missing
   pretrained/results assets. See the method card, source map,
   [`asset feasibility`](../../gs_framework/specs/alod-asset-feasibility.md),
   and [`reproduction plan`](../../gs_framework/specs/alod-reproduction-plan.md).
2. **Environment/smoke bring-up.** An isolated Python 3.10/CUDA environment
   and SM120 build-only patch superseded the initial block. Smoke Runs001--003
   exposed, respectively, DrvFS multiprocessing sockets, system-CUDA 12.0 JIT,
   and a PyTorch 2.11 optimizer API rename. Validated launcher-only fixes are
   reproduction support, not results. See
   [`alod-official-smoke.md`](../../gs_framework/docs/environments/alod-official-smoke.md).
3. **Smoke Run004/005.** Run004 completed training but failed topology. The
   scoped relocation repair produced accepted aerial-only Run005, enabling
   `.dhier` parity, official selection oracle, CPU reference, and resident
   runtime validation.
4. **Run005 engineering campaigns.** Resident, host-residency,
   materialization-gap, fast-filter, and materialization-aware campaigns
   established mechanics. They remain historical source-inspired/extension
   evidence: [`resident`](../../gs_framework/docs/results/alod_resident_ablation_benchmark.md),
   [`host`](../../gs_framework/docs/results/alod_host_residency_benchmark.md),
   [`gap`](../../gs_framework/docs/results/alod_materialization_gap.md),
   [`fast filter`](../../gs_framework/docs/results/alod_fast_materialization_culling.md),
   and [`materialization-aware`](../../gs_framework/docs/results/alod_materialization_aware_residency_benchmark.md).
5. **2026-08-20 boundary audit.** Classified every mode, froze the
   `PAPER-SOURCE-DIVERGENCE`, and required a mixed-scale public proxy before
   final claims.
6. **Formal dataset freeze.** Froze source/coordinate checks, released-loader
   ingestion, 639/71 split, and 16 diagnostic cameras. Geometry is cropped
   aerial sparse reconstruction, not the paper's unavailable merged cloud.
7. **Formal Run001.** Stopped pre-optimization because inherited monitoring
   could not enforce per-process VRAM under WSL; no training result.
8. **Formal Run002.** Completed 30k coarse training and hierarchy creation,
   then stopped before fine training: literal volume 150 produced one
   4,634,255-row SPT and failed the cache gate. Its coarse PLY and initial
   hierarchy became frozen parents. Authority:
   [`Run002 memo`](../../gs_framework/docs/results/alod_formal_mixed_training_run002.md).
9. **SPT calibration.** A source-documented, result-blind sweep selected 4.75
   without cameras, rendering, quality, or FPS. Authority:
   [`calibration memo`](../../gs_framework/docs/results/alod_formal_spt_calibration.md)
   and linked JSON.
10. **Formal Runs003--005.** Run003 exceeded process-tree RSS during gsplat
    compilation; Run004 did so because forked workers inherited the parent
    address space; synchronous ordered loading was adopted; formal Run005 then
    failed writing provenance. These are infrastructure failures.
11. **Formal Run006.** At iteration 13,500, densification required 1,601,132
    rows against the frozen 1,600,000 reduced cache. The 1,132-row shortfall
    caused a fail-closed stop; topology/resources remained healthy. Authority:
    [`dynamic-cache-feasibility-gate.json`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-006/failure/dynamic-cache-feasibility-gate.json).
12. **Formal Runs007--008.** With the 6.25M/5.0M v3 capacity, Run007 passed to
    iteration 48,000 then an external WSL restart stopped it at 49,160. Run008
    stopped at 310 when a non-lingering user service manager exited. Neither
    was a method/resource/topology failure; neither resumed partial state.
13. **Formal Run009.** Restarted from frozen Run002 parents, completed fine
    training, serialized the final hierarchy, and passed strict validation.
    A final snapshot-label collision was repaired by no-training/no-mutation
    revalidation. Authority:
    [`final-completion-audit.json`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-completion-audit.json).
14. **Final oracle.** Two fresh official processes per diagnostic camera were
    deterministic; the CPU reference matched all layers exactly. Outcome:
    `WEAK-LOD-ORACLE`. Authority:
    [`completion-report.json`](../../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/completion-report.json).
15. **Faithful campaign.** Three processes per mode measured no-cache,
    no-frustum, and released viewer-compatible reuse; separate gates covered
    quality, HSPT/BFS, and 71-camera LoD coverage.
16. **Final extension campaign.** Six `ALOD-EXT` variants were compared to the
    frozen faithful baselines under the same 71-camera protocol.

## 6. Topology-correction boundary

Pinned `relocate_gs` can promote a surviving sibling while updating reverse
parent/depth metadata for only two immediate children. Deeper descendants can
stay stale; a promoted leaf can route an update through row zero and corrupt
skybox metadata. Unpatched smoke Run004 recorded 4,355 depth inconsistencies,
one reverse-parent mismatch, and corrupted skybox topology. Authority:
[`final-barrier-failure.json`](../../runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/ALOD-SMOKE-TRAIN-004/failure/final-barrier-failure.json).

[`ALOD-RELOCATE-TOPOLOGY-FIX.patch`](../../gs_framework/patches/alod/ALOD-RELOCATE-TOPOLOGY-FIX.patch)
rebuilds redundant parent/depth metadata below only affected forward roots.
Differential/CUDA gates show unchanged properties, sampling, respawn behavior,
forward topology, losses, optimization, densification, HSPT/cache, selection,
and rendering. Classification: reproduction-support correctness repair. It
does not establish equivalence to an unreleased author fix or make Run009 an
untouched-source output.

## 7. Paper/source divergences

| Topic | Paper | Released source / project consequence |
| --- | --- | --- |
| Dataset | `MC-small-city+`; supplement components total 41,211 training views while main table says 42.2k images. | Exact lists, 533 added views, crop, sample seed, trained asset, and paths unavailable; proxy has a new name. |
| Geometry | Merged aerial/street sparse clouds, random factor-five sample. | Proxy crops released aerial geometry; no paper geometry claim. |
| Coarse schedule | 100k iterations. | Released configs use 30k/60k/250k; proxy executes 30k. |
| `Render Full` | HSPT + frustum + host properties + bounded SPT cache. | No released renderer combines them. Evaluator is no-cache; viewer is immediate-previous compatible-SPT reuse; only trainer has bounded write-back state. |
| Capacity | Paper experiments report 15M. | Source MatrixCity config uses 25M/20M; Run009 training calibrates 6.25M/5.0M. None is paper render-cache identity. |
| Reuse | Compatible SPT cuts. | Trainer ratio about 0.5--1.3; viewer strict 0.3--1.5; evaluator none. |
| Cache mechanics | Bounded LRU-like write-back and flush. | Trainer uses ordered retained SPTs without explicit ages and counts full assembled rows; viewer has no capacity, multi-frame retention, write-back, or flush. |
| Frustum | Early sphere rejection; three-times-max-scale proxy. | Four side planes only; near/far inactive; propagated spheres optional/default off. |
| View scheduling | Random exploration every 128 iterations. | Source uses 100, overwrites `k=100`, uses `1/(distance+20)`, and MatrixCity/formal configs disable graph scheduling. |
| Renderer | Paper measurement backend. | Release fixes packed RGB gsplat and is documented as more memory-efficient; several wrapper arguments are ignored. |
| Viewer | Recorded path. | Hard-coded focal base 1000, replay 1100x900, unsynchronized end wall clock, suspicious positional CLI wiring; not a paper harness. |
| HSPT/BFS | Alternative computations for intended same cut. | No official BFS evaluator; reconstructed BFS differs materially in selected IDs. |
| HSPT identity | Method component. | Rebuilt, not serialized; training starts at min SPT 256, rebuild/evaluation default to 100. |
| Helper | Not distinguished. | `get_SPT_cut` overwrites its condition with all true; inline train/eval/viewer paths are authoritative. |
| Format/loader | Conceptual hierarchy asset. | Native raw `.dhier` has no magic/version/checksum/endian marker; loader assumes equal Gaussian/node counts and zeroes metadata column five. |
| Creator quirks | Unspecified. | Initial leaves have unused positive `first_child`; merged opacity can exceed one before load clamp. Strict final validation rejects these quirks. |
| Assets | Trained scenes/results. | No pretrained `.dhier`, exact bundle, or Table-3 paths were released. |

## 8. Component classification

| Component/mode | Classification | Boundary |
| --- | --- | --- |
| Hierarchy, HSPT, host properties, camera cut | Paper claim + released-source behavior | Attribute to authors. |
| Parser, fixtures, oracle, CPU reference, instrumentation | Reproduction support | Validation only. |
| Relocation patch and compatibility shims | Reproduction support | Disclose corrections. |
| Run009 official pipeline | `ALOD-REPRO`, qualified | Correctness-patched proxy/resource-calibrated training. |
| Production HSPT/frustum/selection | `ALOD-REPRO` boundary | Exact ordered official parity. |
| `ALOD-REPRO-HOST-NOCACHE` | `ALOD-REPRO` | Closest evaluator behavior; project harness disclosed. |
| `ALOD-REPRO-NOFRUSTUM` | `ALOD-REPRO` component ablation | Not paper Table 3; Boolean-device correction disclosed. |
| `ALOD-REPRO-PREVIOUS-REUSE` | `ALOD-REPRO` viewer behavior | Compatible prior SPTs; unbounded, not `Render Full`. |
| `ALOD-REPRO-CUT-HSPT` | `ALOD-REPRO` | Exact released cut. |
| `ALOD-REPRO-CUT-BFS` | Minimal source-faithful reconstruction | No official executable; mismatch retained. |
| Resident finest/LoD/frustum | Source-inspired adaptation | Faithful selector with full property residency. |
| Historical exact-row previous reuse | Source-inspired adaptation | Row identity, not compatible SPTs. |
| Row-LRU and trace capacities | `ALOD-EXT` | Read-only per-row cache. |
| CPU/CUDA pre-materialization filters | `ALOD-EXT` | Post-selection filter absent upstream. |

## 9. Canonical artifact register

| Claim family | Canonical authority |
| --- | --- |
| Dataset/split/source | [`manifest`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/manifest.json), [`split`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/split.json), [`source provenance`](../../gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01/source_provenance.json) |
| Final config/training/resources | [`config`](../../gs_framework/configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target-v3-run009.json), [`provenance`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/provenance.json), [`completion audit`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-completion-audit.json) |
| Hierarchy/topology | [`independent hierarchy`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-hierarchy-independent.json), [`official load/HSPT`](../../runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/validation/final-official-loader-hspt.json) |
| Exact selection | [`oracle completion`](../../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/completion-report.json), [`reference parity`](../../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/project-reference-parity.json), [`analysis`](../../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/formal-analysis.json) |
| Faithful completion/runtime | [`campaign summary`](../../benchmark_artifacts/alod_faithful_reproduction_001/ALOD-FAITHFUL-REPRODUCTION.json), [`rendering summary`](../../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json) |
| Faithful memory/traffic | [`memory`](../../benchmark_artifacts/alod_faithful_reproduction_001/memory_summary.json), [`H2D`](../../benchmark_artifacts/alod_faithful_reproduction_001/h2d_summary.json) |
| Frustum/HSPT-BFS/quality/LoD | [`frustum`](../../benchmark_artifacts/alod_faithful_reproduction_001/tables/frustum_ablation.json), [`HSPT-BFS`](../../benchmark_artifacts/alod_faithful_reproduction_001/hspt_bfs/summary.json), [`quality`](../../benchmark_artifacts/alod_faithful_reproduction_001/quality/results/summary.json), [`LoD coverage`](../../benchmark_artifacts/alod_faithful_reproduction_001/lod_coverage_71/summary.json) |
| Extension correctness/capacity/results | [`correctness`](../../benchmark_artifacts/alod_ext_formal_run009_001/checks/correctness.json), [`cache sizing`](../../benchmark_artifacts/alod_ext_formal_run009_001/checks/cache_sizing.json), [`formal summary`](../../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json) |

## 10. Final faithful results

The 16-camera oracle passed exact HSPT, camera, traversal, ordered SPT cuts,
raster IDs, and properties. The faithful campaign used three fresh processes
per mode, ten ordered warmups, and 71 measured cameras per process: 213 frames
per mode, 639 total. Authorities: oracle
[`completion report`](../../runs/alod/oracles/ALOD-MIXED-RUN009-SELECTION-ORACLE-001/completion-report.json)
and faithful [`campaign summary`](../../benchmark_artifacts/alod_faithful_reproduction_001/ALOD-FAITHFUL-REPRODUCTION.json).

| Mode | Mean / median / p95 end-to-end | FPS | Mean selected hierarchy | Mean H2D rows | Authority |
| --- | ---: | ---: | ---: | ---: | --- |
| `ALOD-REPRO-HOST-NOCACHE` | 186.77 / 184.87 / 207.52 ms | 5.35 | 3,432,141 | 3,432,141 | [`rendering_summary.json`](../../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json) |
| `ALOD-REPRO-NOFRUSTUM` | 188.02 / 186.16 / 210.13 ms | 5.32 | 3,442,923 | 3,442,923 | same |
| `ALOD-REPRO-PREVIOUS-REUSE` | 134.21 / 131.58 / 160.44 ms | 7.45 | 3,456,309 | 87,570 | same |

No-cache preparation is 182.34 ms versus 4.15 ms rasterization; host gather
is 93.05 ms, H2D 19.27 ms, and selection 38.09 ms. Mean raster input is
3,532,141 after adding skybox, while packed projection survivors are 496,182.
Previous-compatible-SPT reuse saves 52.57 ms (28.14%) but increases persistent
working-set residency. Authorities: [`rendering`](../../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json)
and [`H2D`](../../benchmark_artifacts/alod_faithful_reproduction_001/h2d_summary.json).

No-cache stores 0.606 GiB host properties and 0.236 GiB CPU hierarchy
metadata. Persistent GPU Gaussian properties are the 0.009 GiB skybox;
frame-local properties peak at 0.605 GiB. Maximum PyTorch frame peak is 1.281
GiB allocated/2.387 GiB reserved, whole-device VRAM 3.963 GiB, and RSS 3.810
GiB. Authority: [`memory_summary.json`](../../benchmark_artifacts/alod_faithful_reproduction_001/memory_summary.json).

Frustum avoided 10,782 semantic/H2D rows on average (0.313%) and 5.03 SPTs,
but only 0.155 packed survivors. Frustum-on was 1.25 ms (0.664%) faster by
pooled mean, within large paired variability; 64/71 images were bit-identical
and remaining maximum RGB error was 0.00035188. The strong paper cached-render
trend is not established. Authority: [`frustum_ablation.json`](../../benchmark_artifacts/alod_faithful_reproduction_001/tables/frustum_ablation.json).

HSPT/BFS mean per-camera median cut time was 36.00/91.54 ms (2.543x) and
logical metadata 83,685,344/396,067,112 bytes (4.733x). Mean Jaccard was only
0.0970 and BFS selected 3,099,291 fewer rows: speed/memory direction reproduced,
cut equivalence did not. Authority: [`hspt_bfs/summary.json`](../../benchmark_artifacts/alod_faithful_reproduction_001/hspt_bfs/summary.json).

| Split | Cameras | PSNR mean/median | SSIM mean/median | LPIPS-VGG mean/median | Authority |
| --- | ---: | ---: | ---: | ---: | --- |
| All | 71 | 21.495 / 21.183 | 0.6903 / 0.6913 | 0.3972 / 0.4157 | [`quality summary`](../../benchmark_artifacts/alod_faithful_reproduction_001/quality/results/summary.json) |
| Aerial | 26 | 23.255 / 23.775 | 0.7352 / 0.7781 | 0.3088 / 0.3095 | same |
| Street | 45 | 20.478 / 20.882 | 0.6644 / 0.6718 | 0.4483 / 0.4593 | same |

These preserve official evaluator metric and alpha semantics and are local
proxy results, not paper numbers.

Across 71 cameras, selected hierarchy count is 2,606,659--3,478,844 (mean
3,432,141), but all reach depth 55 and median depth 22; mean depth spans only
22.8484--23.0579 and maximum all-pairs depth TVD is 0.02358. The hierarchy is
valid but `WEAK-LOD`. Authority: [`lod_coverage_71/summary.json`](../../benchmark_artifacts/alod_faithful_reproduction_001/lod_coverage_71/summary.json).

## 11. Project-extension results — strictly separate

Everything here is `ALOD-EXT`. The protocol reuses Run009, the 71-camera
order/resolutions, packed gsplat, ten warmups, and three processes. Exact
oracle selection, touched SPTs, surviving properties/RGB/alpha, zero false
negatives, and CPU/CUDA filter parity passed. Authorities:
[`correctness.json`](../../benchmark_artifacts/alod_ext_formal_run009_001/checks/correctness.json)
and [`formal_summary.json`](../../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json).

The mean funnel is 3,532,141 semantic rows including skybox, 525,049
materialized, and 496,182 packed survivors: 6.727:1 semantic/materialized. The
filter is post-selection and does not alter the LoD cut.

| Extension | Mean / median / p95 | FPS | Authority |
| --- | ---: | ---: | --- |
| Row-LRU practical | 134.54 / 132.68 / 145.60 ms | 7.43 | [`formal summary`](../../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json) |
| CPU filter practical | 220.50 / 213.79 / 265.99 ms | 4.54 | same |
| CUDA filter + previous | 71.66 / 66.72 / 124.03 ms | 13.96 | same |
| CUDA filter + minimum | 84.38 / 84.73 / 138.03 ms | 11.85 | same |
| CUDA filter + practical | 73.48 / 68.94 / 98.73 ms | 13.61 | same |
| CUDA filter + large | 76.29 / 71.99 / 101.34 ms | 13.11 | same |

The fastest extension saves 115.12 ms (61.63%) and is 2.606x faster than
faithful no-cache; semantic selection remains largest at 38.19 ms (53.30%).
Capacities were frozen before timing: maximum post-filter frame 1,599,906,
union 3,503,048, practical 3,700,000, sensitivity 5,300,000 rows. Authority:
[`cache_sizing.json`](../../benchmark_artifacts/alod_ext_formal_run009_001/checks/cache_sizing.json).
Run005's historical 525k/160k values are not final Run009 capacities.

## 12. Supersession chain

1. Audits are interpretation authorities, not measurements.
2. Smoke Runs001--004 are failed evidence; Run005 is the first valid local
   hierarchy and historical correctness/development platform.
3. Run009 supersedes Run005 for final hierarchy claims; the Run009 oracle and
   faithful/extension packages supersede it for final results.
4. Volume 150 is superseded by calibrated 4.75; v2 2.0M/1.6M training cache is
   superseded by v3 6.25M/5.0M. None is paper render-cache identity.
5. Formal Runs001--008 are failed/interrupted history; Run009 is final.
6. One-camera `DEGENERATE` smoke is superseded by the 16-camera
   `WEAK-LOD-ORACLE` result.
7. Historical exact-row reuse is distinct from final viewer-compatible
   `ALOD-REPRO-PREVIOUS-REUSE`.
8. Historical Run005 capacities/results remain engineering history; final
   Run009 extension JSON supersedes them for thesis claims.
9. Selected hierarchy rows exclude skybox; semantic/raster totals used in the
   extension funnel include it. Never resolve this apparent 100,000-row
   difference by choosing prose over JSON.

## 13. Reproducibility assessment

- **Implementation reproducibility: strong within the pinned public
  boundary.** Format, loader transforms, HSPT, four-plane frustum, ordered
  selection, raster order, and pre-raster attributes are independently exact.
  Missing paper cache and an unknown author-side relocation fix are outside it.
- **Experimental reproducibility: strong locally, partial versus paper.**
  Source, data, split, cameras, hierarchy, config, schedules, metrics, gates,
  and raw records are frozen; paper assets/paths/backend/hardware/cache are not.
- **Result reproducibility: partial.** Public-source behavior and local
  quality/runtime/memory/traffic plus extension effects are measured. Exact
  paper numbers and 60M--150M findings are not; frustum is inconclusive and
  HSPT/BFS identity is negative.

## 14. Claim rules

Safe after reopening the cited artifacts:

- exact pinned-evaluator selection and pre-raster parity on 16 cameras;
- a strictly valid, correctness-patched Run009 hierarchy on the proxy;
- no-cache dominated by selection/gather/transfer rather than rasterization;
- released viewer reuse lowers transfer/time while increasing residency;
- frustum removes semantic work without a robust local timing result;
- HSPT is faster/lighter than reconstructed BFS but produces different cuts;
- valid hierarchy with weak view-dependent depth variation;
- `ALOD-EXT` CUDA filtering preserves semantic selection and lowers property
  materialization under the frozen protocol.

Qualification required: every number is proxy/RTX5080/WSL/released-gsplat and
camera-protocol specific; Run009 includes calibration and topology correction;
quality uses official evaluator semantics; `WEAK-LOD` limits generalization;
memory counters differ; extension speedup is versus faithful project no-cache,
not paper `Render Full`.

Prohibited claims:

- full paper reproduction or paper-number parity;
- calling the proxy `MC-small-city+` or claiming paper merged geometry;
- claiming paper `Render Full` was implemented/measured;
- calling row-LRU/materialization modes faithful ALoD;
- calling Run009 unchanged-source or the patch the authors' intended fix;
- strong adaptive hierarchy-depth evidence;
- local HSPT/BFS cut equivalence or significant frustum speedup;
- treating packed survivors as pixel contributors;
- conflating selected, materialized, resident, transferred, projected, or
  rendered counts;
- using Run005 performance/capacities as final Run009 evidence;
- direct CityGaussian/ALoD ranking from their native campaigns.

## 15. Limitations, optional work, and writing order

Threats: unavailable paper asset/paths, proxy geometry divergence, one local
hierarchy, `WEAK-LOD`, different hardware/backend, WSL memory accounting,
topology and no-frustum corrections, missing bounded paper cache,
reconstructed BFS, and no paper-scale capacity.

No experiment is required before writing. Optional work includes obtaining
author assets/cache code, a stronger-LoD or second-scene campaign, an explicitly
labelled paper-cache approximation, another hardware run, temporal/popping
evaluation, oracle-exact semantic-selection optimization, and a shared-data
cross-method comparison.

Recommended order: paper concept; released behavior; reproduction boundary;
correctness-patched training; oracle parity; faithful evaluation; negative and
weak-LoD findings; then strictly separate `ALOD-EXT`. Keep implementation,
experimental, and result reproducibility separate.

## 16. Run009 extension evidence closure

`ALOD-EXT-RUN009-CLOSURE-001` closes the earlier correctness-scope ambiguity
and characterizes cache locality without changing any semantic or policy
boundary. The pre-existing formal gate had exact CPU/CUDA filter equality on
all 71 held-out cameras, but direct faithful packed-projection inclusion,
property, and RGB/alpha checks on only 16 protected diagnostics. The closure
extends those direct checks to all 71: zero false negatives, exact ordered
CPU/CUDA IDs, bitwise retained properties, exact packed-survivor IDs, and
bitwise RGB/alpha.

Previous-working-set reuse is order-sensitive: measured row hit rate is
67.92% canonical, 11.75% under the fixed-seed shuffle, and 3.97% under the
geometry far-jump order. The 3.7M bounded cache exceeds the 3,503,048-row
trajectory union, has zero evictions in all three orders, and transfers one
union (307.4 MiB) from an empty dynamic cache. The canonical measured-only
98.79% transfer reduction remains valid, but writing must label its warm-up
exclusion; the matched cold-inclusive reduction is 98.73%.

The 15-camera geometry-only scale-stress path remains weak LoD: per-camera
median/maximum selected depths stay 22/55 and maximum depth-distribution TVD
is 0.01834. Canonical evidence and the complete claim-impact table are in
[`alod_ext_run009_closure.md`](../../gs_framework/docs/results/alod_ext_run009_closure.md)
and [`manifest.json`](../../benchmark_artifacts/alod_ext_run009_closure_001/manifest.json).

Evidence update: this topic note and the experiment registry; reproduction
matrix unchanged; classification `ALOD-EXT` project-specific evidence closure;
canonical artifact `benchmark_artifacts/alod_ext_run009_closure_001`.

## 17. Paper `Render Full` specification/source audit

`ALOD-REPRO-RENDER-FULL-SPEC-AUDIT-001` asked whether the paper's bounded
`Render Full` cache could be independently reconstructed from arXiv v4 and
released source commit `73f547341547ae021ff036ad74fdf73a1240b450`
without outcome-dominating assumptions. It was an `ALOD-REPRO` specification
and paper/source-boundary audit, not an `ALOD-EXT` experiment. No
implementation, training, correctness run, or benchmark was started.

### Question

Are cache entry identity, insertion, retention, eviction, capacity, state, and
timing semantics specified closely enough to justify a separately labelled
paper `Render Full` reconstruction?

### Answer

The paper explicitly specifies SPT-cut entries, cached root distance,
distance-compatible reuse, Gaussian-row capacity, LRU-style retention, and a
15M-row experimental cache. The release exposes evaluator no-cache, viewer
previous-working-set reuse, and bounded mutable fine-training cache behavior
in separate programs. It does not uniquely identify the render-specific
compatibility interval, capacity accounting, reduction timing, row ordering,
flush behavior, cache-state protocol, or Table-3 timing boundary.

Most decisively, the paper says compatible cached SPT cuts may change rendered
detail with cache state, while the predeclared reconstruction gate required
exact selected IDs and raster inputs relative to the fresh evaluator. An
exact-row or exact-cut cache would satisfy that gate but would not reproduce
the paper-described compatible-cut behavior. Combining viewer thresholds with
trainer retention would splice distinct released paths. The final gate is:

```text
STOP-RENDER-FULL-UNDERSPECIFIED
```

This means underspecified from the available public evidence under the stated
validation requirements, not impossible to implement in principle. A
paper-guided/source-inspired cache could be studied under a different label,
but must not be reported as faithful or author-identical `Render Full`.

### Supported conclusions

- The paper defines the cache concept and SPT-cut granularity, but several
  render-specific execution semantics are not uniquely recoverable.
- No pinned released renderer implements the complete bounded `Render Full`
  combination.
- Evaluator, viewer, trainer, and `ALOD-EXT` remain four distinct evidence
  classes.
- The paper-reported 15M threshold exceeds Run009's entire 7,072,627-row
  asset, so this workload could not validate bounded eviction pressure even if
  the other ambiguities were resolved.

### Unsupported / untested conclusions

- The audit does not show that the authors' cache is ineffective, that it
  cannot be implemented with additional evidence, or that paper timings are
  wrong.
- It does not show that an independently invented hybrid matches the authors'
  renderer.
- It produced no `Render Full` timing, H2D, hit-rate, occupancy, correctness,
  or quality result.
- It does not change any faithful Run009 or `ALOD-EXT` result and does not
  convert project row/materialization policies into paper behavior.

### Superseded interpretations

Earlier notes left an explicitly labelled paper-cache approximation as
optional future work. The audit narrows that position: the available evidence
does not support promoting such a hybrid to `ALOD-REPRO-RENDER-FULL-
RECONSTRUCTION`. It remains possible only as a clearly labelled paper-guided
or source-inspired adaptation unless new primary evidence resolves the cache
semantics and the correctness target.

### Reproducibility and writing relevance

Implementation reproducibility is partial for this component: granularity and
qualitative policy are reconstructible, but the complete renderer policy is
not uniquely determined. Experimental and result reproducibility were not
reached. The audit strengthens the paper/source-divergence record and leaves
ALoD-EXT external validity unchanged.

For the full dissertation, use a short subsection in the ALoD reproduction
methodology/paper-source divergence discussion, with a brief limitation
reminder near evaluation. For the condensed paper, include only a short
limitations/discussion statement explaining why released evaluator and viewer
boundaries are reported instead of paper `Render Full`.

Safe claim: the paper specifies compatible SPT-cut caching and LRU-style
retention, but the paper and pinned release do not uniquely determine the
complete `Render Full` execution policy required for an independent faithful
reconstruction. Prohibited claim: the project reproduced, benchmarked,
disproved, or established the performance of paper `Render Full`.

Canonical narrative and complete ledger:
[`alod_render_full_spec_audit.md`](../../gs_framework/docs/results/alod_render_full_spec_audit.md).
Machine-readable closure:
[`alod_render_full_spec_audit_closure.json`](../../gs_framework/docs/results/data/alod_render_full_spec_audit_closure.json).

Comparison impact: none of the canonical Run009 evaluator/viewer timings,
ALoD-EXT's 2.6064936690853813 speedup versus the faithful no-cache baseline,
or CityGaussian results changes. The audit strengthens the requirement that
the extension comparison is not described as a comparison against paper
`Render Full`. Authorities: faithful
[`rendering_summary.json`](../../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json)
and extension
[`formal_summary.json`](../../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json).

Evidence update: this topic note and experiment registry; reproduction matrix
unchanged; classification `ALOD-REPRO` specification/source audit with a
negative scientific-defensibility gate; canonical artifact
`gs_framework/docs/results/data/alod_render_full_spec_audit_closure.json`.

## 18. Optional Uni10K external-validity preflight

### Identity and classification

- Experiment: `ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT`.
- Completed attempt: `ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT-RUN005`.
- Purpose: test whether a larger public-source ALoD hierarchy could reach the
  strong-LoD gate needed for an external-validity check of the completed
  Run009 `ALOD-EXT` materialization/residency findings.
- Class: `ALOD-REPRO` supplementary external-validity **feasibility
  preflight**, not an `ALOD-EXT` result.
- Final gate: `BLOCKED-ALOD-UNI10K-FINE-HEADROOM`.
- Reached: validated dataset construction, 30,000-iteration coarse training,
  hierarchy creation, independent structural validation, structure-only SPT
  calibration, and official-loader HSPT reconstruction.
- Not reached: fine training, final-hierarchy validation, camera selection or
  LoD classification, correctness oracle, quality/runtime benchmarking, and
  any `ALOD-EXT` experiment.

### Question

Do the Run009 `ALOD-EXT` materialization and residency findings persist on a
public-source ALoD hierarchy with materially stronger natural view-dependent
LoD adaptation?

### Answer

This preflight could not answer that external-validity question. It established
that Uni10K yields a valid, non-degenerate, cache-feasible initial hierarchy,
but the candidate did not satisfy the predeclared resource-admission rule for
fine training. Because no final hierarchy or camera-dependent cut evaluation
exists, the strong-LoD outcome remains **untested**.

### Executed evidence and stopping gate

Public Uni10K at source commit
`811af1acf8d274ac5d59b498a89ba9dd8c523f86` supplied 10,063 aerial images
(8,805 train and 1,258 held out), four camera IDs, and 6,170,655 sparse points.
It contains no street/ground imagery and must not be described as a mixed
aerial/ground dataset. The ALoD execution used pinned source commit
`73f547341547ae021ff036ad74fdf73a1240b450` plus the existing correctness
patch; it did not change selector, hierarchy, densification, objective,
rasterizer, or `ALOD-EXT` semantics.

RUN005 completed coarse training and the creator with exit status zero. The
initial `.dhier` has 12,341,309 rows, SH degree 3, maximum depth 23, file size
3,208,740,352 bytes, and SHA-256
`bb0b58966b34e0855767eef69c2a9a97a118c72ec75508a91343de8a830b2429`.
Independent inspection found finite attributes, valid topology, and all rows
reachable from one root.

Outcome-independent structural calibration selected
`SPT_root_volume=50`, the largest tested value inside the passing stable
neighborhood 75/50/25. The official loader reproduced the structural
prediction: 56 SPTs, 1,997 upper rows, 12,339,368 flattened SPT rows,
1,542,663 maximum SPT rows, and a passing 1,744,660-row conservative working
set under the frozen 6.4M reduced-cache target.

Fine loading would prepend 100,000 skybox rows, giving 12,441,309 initial rows
under the frozen 15M cap. The remaining 2,558,691 rows pass the absolute 2.5M
minimum by 58,691 rows but provide only 17.05794% cap headroom, below the frozen
20% admission criterion. The experiment therefore stopped before fine
training without weakening the gate or introducing a second envelope.

### Supported conclusions

- A substantially larger public candidate passed dataset, coarse, creator,
  independent hierarchy, structural HSPT, and official-loader HSPT checks.
- Under the frozen 15M-row resource envelope, its initial fine state failed
  the predeclared 20% growth-headroom admission criterion.
- Runs002--004 were monitor-orchestration failures, not evidence of a training
  or algorithm defect: the monitor failed to parse live `tqdm` progress and
  injected the observed `KeyboardInterrupt`; RUN005 completed after that
  parser was corrected.
- Stopping before fine training is the scientifically valid outcome of this
  preflight and preserves the outcome-independent protocol.

### Unsupported / untested conclusions

- This is not a `WEAK-LOD` result; camera-dependent LoD was never evaluated.
- It does not establish that Uni10K can never be trained under a different,
  prospectively justified resource envelope or on different hardware.
- It does not establish an ALoD algorithm, hierarchy, dataset, or
  implementation failure.
- It does not confirm or refute whether the Run009 materialization gap,
  transfer reduction, cache behavior, or 2.606x extension speedup persists
  under stronger LoD.
- It does not reproduce or contradict paper-scale quality, memory, capacity,
  or performance results.

### Superseded interpretations

#### Earlier belief

The pretraining camera-count and scale-diversity audit made Uni10K a defensible
candidate for reaching the strong-LoD gate within the proposed local envelope.
Separately, repeated late `KeyboardInterrupt` failures initially looked like a
possible training/data instability.

#### Later evidence

The corrected monitor allowed RUN005 to finish coarse training and hierarchy
creation, disproving the training-instability interpretation. Exact hierarchy
size then showed that the fine initial state had only 17.05794% cap headroom,
so the frozen resource gate—not hierarchy validity—blocked continuation.

#### Final thesis position

Uni10K was a justified candidate and produced a valid initial ALoD structure,
but it was infeasible for **this predeclared fine-training envelope**. Treat it
as a resource-feasibility negative preflight, not as weak LoD, failed ALoD, or
evidence against the candidate under all possible envelopes.

### Reproducibility relevance

- **Implementation reproducibility:** adds supporting evidence that the pinned
  public coarse/creator/loader/HSPT path operates on this larger candidate; it
  does not change the existing method-level assessment.
- **Experimental reproducibility:** the completed preflight is fully indexed,
  but the intended full training/evaluation protocol was not admitted.
- **Result reproducibility:** no paper or strong-LoD result was tested.
- **ALoD-EXT external validity:** unchanged and unresolved; Run009 remains the
  sole canonical formal extension workload and remains `WEAK-LOD`.
- **Resource feasibility:** negative under the frozen 15M cap and 20% headroom
  rule; this is the material new evidence.
- **Paper/source divergence:** unchanged; this run did not audit or reconstruct
  missing paper semantics.

### Thesis-placement guidance

For the full dissertation, use a short subsection or compact negative-result
paragraph in the ALoD evaluation-limitations/external-validity discussion,
with detailed counts left in the result report or appendix. The message should
be that a larger candidate passed structural/HSPT preflight but did not satisfy
the prospectively frozen resource-admission rule for fine training.

For the condensed eight-page paper, use at most one limitations/discussion
sentence if space permits; omit the detailed structure and attempt chronology.
It is not a central performance result.

### Claims for the writing agent

Safe claims:

- The Uni10K candidate completed coarse training and produced an independently
  valid, non-degenerate initial ALoD hierarchy.
- It failed the frozen fine-training admission rule because 17.05794% cap
  headroom was below the required 20%.
- Fine training and strong-LoD/ALoD-EXT evaluation were not performed.

Claims requiring qualification:

- “Infeasible” must be scoped to the frozen 15M-row local resource envelope
  and its predeclared 20% rule.
- “Larger candidate” refers to this public Uni10K-derived dataset and initial
  hierarchy, not paper-scale equivalence.
- HSPT feasibility applies to the initial hierarchy and the frozen 6.4M
  reduced-cache structural gate, not to final training or rendering behavior.

Claims that must not be written:

- Uni10K or the candidate hierarchy is `WEAK-LOD`, `NONTRIVIAL-LOD`, or
  strong-LoD.
- ALoD failed on Uni10K, or Uni10K cannot be trained.
- The experiment validated or invalidated `ALOD-EXT` on a stronger hierarchy.
- The experiment changed Run009 timings, correctness, quality, or hierarchy
  classification, reproduced paper `Render Full`, or changed CityGaussian
  evidence.

### Comparison impact and provenance

No canonical comparison changes. Run009 faithful and extension measurements,
including the 2.606x extension-versus-faithful-no-cache result, retain their
existing values and `WEAK-LOD` qualification. This preflight neither strengthens
nor weakens a CityGaussian comparison and creates no new cross-method ranking.

Canonical narrative:
[`alod_uni10k_preflight.md`](../../gs_framework/docs/results/alod_uni10k_preflight.md).
Machine-readable closure:
[`closure.json`](../../runs/alod/ALOD-REPRO-UNI10K-01/ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT-RUN005-CLOSURE/closure.json).
Dataset identity:
[`manifest.json`](../../gs_framework/datasets/alod/ALOD-REPRO-UNI10K-01/manifest.json).
Frozen resource envelope:
[`uni10k-preflight-resource-policy.json`](../../gs_framework/environments/alod-official/uni10k-preflight-resource-policy.json).
Frozen launcher:
[`uni10k-preflight-command.sh`](../../gs_framework/environments/alod-official/uni10k-preflight-command.sh).
Primary generated artifact:
[`hierarchy.dhier`](../../runs/alod/ALOD-REPRO-UNI10K-01/ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT-RUN005/output/scaffold/point_cloud/iteration_30000/hierarchy.dhier).

Evidence update: this topic note and experiment registry; reproduction matrix
unchanged; classification `ALOD-REPRO` supplementary external-validity
preflight blocked by the frozen fine-capacity gate; canonical artifact
`runs/alod/ALOD-REPRO-UNI10K-01/ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT-RUN005-CLOSURE`.

## 17. Frozen natural-dataset prevalence: Campus Stage A

The first newly trained natural dataset under
[`ALOD-LOD-PREVALENCE-DESIGN-001`](../ALOD_LOD_PREVALENCE_STUDY_DESIGN.md) is
the standalone H3DGS Campus chunk (`ALOD-PREV-H3DGS-CAMPUS-CHUNK-01`). The
project-scale, correctness-patched pinned-source reproduction completed its
coarse, creator, fine, independent hierarchy, and all-24-held-out-camera
official oracle stages. Independent CPU selection matched the official HSPT
and every camera cut exactly; two cold repetitions per camera were
deterministic. The frozen three-part `criteria_v1` yields
`VALID-NONTRIVIAL-LOD` for this one hierarchy. The analysis has a 0.252374
selected-count relative range, 0.448699 minimum pairwise selected-set
Jaccard, and 0.071975 maximum pairwise selected-depth TVD. The maximum
selected depth itself is constant at 44; the depth predicate passes via TVD.

This establishes a valid nontrivial *selection-cut* outcome for one natural
scene under the declared local method. It does not establish a strong
monotonic scale-to-depth response, paper-scale result reproducibility, a
population prevalence rate, or `ALOD-EXT` performance on the Campus asset.
The prospective Stage-A Sci-Art result and any Stage-B repeat remain separate
future work; neither was executed here. Run009 and its weak-LoD extension
qualification remain unchanged. The failed sandbox-driver attempt and an
over-strict observation-only float32 camera check are preserved as attempt
history; the final oracle restarted all 24 cameras under one corrected
instrumentation version, without changing selection semantics or thresholds.

Canonical result narrative:
[`alod_prevalence_campus_stage_a.md`](../../gs_framework/docs/results/alod_prevalence_campus_stage_a.md).
Authoritative machine-readable outcome:
[`prevalence-lod-response.json`](../../runs/alod/oracles/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-ORACLE-001/validation/prevalence-lod-response.json).
Official deterministic cuts and identity:
[`semantic-manifest.json`](../../runs/alod/oracles/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-ORACLE-001/semantic-manifest.json).
Independent parity:
[`cpu-reference-parity.json`](../../runs/alod/oracles/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-ORACLE-001/validation/cpu-reference-parity.json).
Fine hierarchy and resource log:
[`final-hierarchy-independent.json`](../../runs/alod/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-TRAIN-001/validation/final-hierarchy-independent.json)
and [`fine.jsonl`](../../runs/alod/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01-TRAIN-001/metrics/fine.jsonl).

Reproducibility impact: this adds a validated released-source selection result
on another public large-scene dataset, within a project-specific training envelope. It
does not change the method-level implementation assessment, nor close the
paper's experimental or result-reproducibility gaps. The reproduction matrix
is unchanged.


## 18. Completed prevalence Stage A and ALoD-EXT association

Stage A is complete under the prospectively frozen criteria: Campus is
`VALID-NONTRIVIAL-LOD`, while Sci-Art and MatrixCity Run009 are
`VALID-WEAK-LOD`; there are no degenerate, infeasible, or unlabeled outcomes.
These labels remain selection-cut classifications and were not changed by the
later execution study. Before observing new extension results, the Stage-B
repeat pair was frozen as Sci-Art TRAIN/ORACLE-002 (weak) and Campus
TRAIN/ORACLE-002 (nontrivial). The randomness audit and both repeats remain
pending.

`ALOD-EXT-PREVALENCE-ASSOCIATION-001` then applied the already validated
post-semantic-selection execution extension to all 24 Campus and 21 Sci-Art
held-out cameras. Every camera passed exact official cut, zero-false-negative
filter, ordered CPU/CUDA, bitwise property, packed-survivor, and bitwise
RGB/alpha gates. Under the prospectively normalized union-covering capacities,
the bounded mode reduced mean end-to-end frame time by 25.90% on Campus and
51.69% on Sci-Art relative to Released Evaluator (No Cache), while reducing
steady dynamic-property H2D bytes by 99.23% and 99.66%, respectively.

The central association result is descriptive: Campus and Sci-Art have nearly
equal pooled materialized/semantic ratios (29.95% and 30.10%) despite their
different frozen LoD labels; historical weak Run009 is much lower at 14.86%.
The three points therefore do not support using weak versus nontrivial class
alone to predict the semantic-to-materialization gap or runtime benefit.
ALoD-EXT remains project-specific and is neither released ALoD nor paper
`Render Full`.

Canonical narrative:
[`alod_ext_prevalence_association.md`](../../gs_framework/docs/results/alod_ext_prevalence_association.md).
Machine aggregate:
[`association_summary.json`](../../benchmark_artifacts/alod_ext_prevalence_association_001/tables/association_summary.json).
Prospective execution protocol:
[`ALOD_EXT_PREVALENCE_ASSOCIATION_PROTOCOL.md`](../ALOD_EXT_PREVALENCE_ASSOCIATION_PROTOCOL.md).
Stage-B choice freeze:
[`ALOD_LOD_PREVALENCE_STAGE_B_REPEAT_FREEZE.md`](../ALOD_LOD_PREVALENCE_STAGE_B_REPEAT_FREEZE.md).


## 19. Synthetic strong-LoD positive control and extension barrier

`ALOD-SYNTH-STRONG-LOD-CONTROL-001` is a controlled artificial mechanism
test, excluded from the natural Stage-A denominator and not a Stage-B repeat.
Its frozen analytic `1024 x 512` Gaussian plane has 1,048,575 hierarchy rows,
maximum depth 19, mathematically aggregated parents, and a 659.83x median
six-sigma root-to-leaf extent progression. The unchanged official selector was
evaluated on a frozen 25-camera, eight-octave near-to-far path.

The control passes every prospective strong-LoD gate. `criteria_v1` reports a
761.606 selected-count relative range, minimum Jaccard 0.00003099, and maximum
depth TVD 0.999960. Projected scale versus normalized mean depth has Spearman
0.988589 and Theil-Sen slope 0.101357 per octave; near/middle/far cameras select
25,137/33/4 hierarchy rows at mean depths 14.725/4.879/1.500. Official and
independent selection IDs match exactly in order for all 25 cameras. The
classification is `PASS-SYNTH-STRONG-LOD-CONTROL`.

The mandatory ALoD-EXT phase was attempted immediately after that gate, but
no timing result exists. At cameras 18--24, exact official semantics emit
global ID 100001 twice: the SPT 0 root satisfies both the released upper-render
`first_child <= 0` path and, at these distances, the released candidate-SPT
`first_child >= 0` population path. The independent reference reproduces the
same ordered four-ID multiset. The validated extension residency contract
requires unique semantic IDs, so correctness stopped at camera 18. Neither the
cut nor residency was deduplicated or modified after observing the result.
The extension status is therefore
`BLOCKED-SYNTH-ALOD-EXT-OFFICIAL-DUPLICATE-SEMANTIC-IDS`, and Outcomes A/B/C
remain unresolved.

Reusable read-only margin instrumentation was also run on the natural frozen
oracles. Among nodes selected at least once, the selected state changes for
99.998% in the synthetic control, 63.837% in Campus, 9.199% in Sci-Art, and
11.588% in the 16-camera Run009 oracle. Root-to-terminal scale span and global
near-boundary incidence do not order the four LoD responses, so the comparison
does not support either as a sole causal explanation. It is consistent with
camera geometry, per-level scale organization, and HSPT organization
interacting, but remains descriptive.

Campus/Sci-Art ground-truth quality remains a separate unexecuted task. Its
cleanest frozen-input design reuses the pinned official quality metrics and
compares Released Evaluator and bounded ALoD-EXT against the same held-out GT;
existing exact correctness is not presented as PSNR/SSIM/LPIPS evidence.

Canonical narrative:
[`alod_synth_strong_lod_control.md`](../../gs_framework/docs/results/alod_synth_strong_lod_control.md).
Prospective protocol:
[`ALOD_SYNTH_STRONG_LOD_CONTROL_PROTOCOL.md`](../ALOD_SYNTH_STRONG_LOD_CONTROL_PROTOCOL.md).
Strong classification:
[`strong-lod-classification.json`](../../runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/strong-lod-classification.json).
Extension blocker:
[`ext_duplicate_blocker.json`](../../benchmark_artifacts/alod_synth_strong_lod_control_001/checks/ext_duplicate_blocker.json).
Natural mechanism comparison:
[`comparison.json`](../../benchmark_artifacts/alod_synth_strong_lod_control_001/natural_comparison/comparison.json).
Quality-closure plan:
[`alod_prevalence_quality_closure.md`](../../gs_framework/docs/planning/alod_prevalence_quality_closure.md).

## 20. Synthetic strong-LoD ordered-multiset EXT closure

`ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE-001` is a new prospective
project-specific extension experiment. It does not revise the parent
`ALOD-SYNTH-STRONG-LOD-CONTROL-001`, whose EXT attempt remains
`BLOCKED-SYNTH-ALOD-EXT-OFFICIAL-DUPLICATE-SEMANTIC-IDS`. The closure keeps the
same hierarchy, 25 cameras, official selector, HSPT, filter predicate, cache
policies, capacity rule, and schedule, but generalizes runtime representation
so unique physical residency is separate from ordered semantic occurrence
multiplicity.

Cameras 18--24 now pass the exact released multiset
`[100001, 100001, 100002, 624289]`: unique backing is
`[100001, 100002, 624289]`, the inverse is `[0,0,1,2]`, and renderer assembly
and packed projection preserve both occurrences. All 25 cameras pass exact
semantic, mapping, CPU/CUDA filter, zero-false-negative, bitwise property,
renderer-row, packed-survivor, and RGB/alpha gates. Campus and Sci-Art
regression JSON and NPZ traces are byte-for-byte identical to their canonical
pre-change evidence.

Across the synthetic trajectory, mean semantic occurrence count is 102,167.12,
mean unique semantic count is 102,166.84, mean unique materialization is
77,087.44, mean assembled renderer count is 77,087.72, and mean packed
survival is 2,167.12. The pooled unique-materialized/unique-semantic ratio is
75.4525%. The union-covering 52,454-row bounded capacity is derived from a
49,661-row unique retained union and has zero evictions.

Relative to Released Evaluator (No Cache), previous reuse reduces pooled mean
end-to-end time from 17.679 to 9.577 ms (45.82%, 1.846x; 104.41 versus 56.57
FPS). Bounded cache reaches 15.161 ms (14.24%, 1.166x; 65.96 FPS), reduces
steady dynamic-property H2D by 99.77%, and reduces cold-inclusive traffic by
54.03%. These means retain repeatable long-latency frames under the frozen
protocol. Duplicate support is a compatibility requirement, not the claimed
cause of the speedup.

The controlled result supports semantic LoD and execution-side
materialization as complementary on this synthetic path, while far cameras
show that strong hierarchy selection can leave almost no hierarchy-row
opportunity. It does not generalize to natural hierarchies or paper `Render
Full`.

Canonical narrative:
[`alod_synth_strong_lod_ext_multiset_closure.md`](../../gs_framework/docs/results/alod_synth_strong_lod_ext_multiset_closure.md).
Prospective protocol:
[`ALOD_SYNTH_STRONG_LOD_EXT_MULTISET_CLOSURE_PROTOCOL.md`](../ALOD_SYNTH_STRONG_LOD_EXT_MULTISET_CLOSURE_PROTOCOL.md).
Formal package:
[`manifest.json`](../../benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/manifest.json),
[`aggregate.json`](../../benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/tables/aggregate.json),
and
[`package_validation.json`](../../benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/checks/package_validation.json).

Evidence update: this topic note; registry added; reproduction matrix unchanged;
classification project-specific `ALOD-EXT` compatibility and controlled
execution experiment; canonical artifact
`benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/`.

## 21. Natural Previous working-set completion

`ALOD-EXT-NATURAL-PREVIOUS-COMPLETION-001` completes the current
Previous-versus-Bounded policy comparison on the frozen final Campus and
Sci-Art workloads. It is a project-specific extension experiment, not released
ALoD and not paper `Render Full`. The post-multiset natural correctness
regression is byte-for-byte identical to the canonical 24-camera Campus and
21-camera Sci-Art correctness artifacts, so the existing all-camera semantic,
filter, property, packed-survivor, and image gates were reused.

The old natural association timing predates the ordered-multiset runtime and
current recorder. All three policies were therefore rerun in the new package,
with three fresh processes per mode, ten ordered warmups, the unchanged camera
order, and the existing union-covering capacities. On Campus, Released,
Previous, and Bounded take 90.250, 66.812, and 65.667 ms; on Sci-Art they take
521.086, 378.909, and 366.420 ms. Previous improves over Released by 25.97%
(1.351x) on Campus and 27.28% (1.375x) on Sci-Art, but Bounded is faster than
Previous by 1.145 and 12.489 ms respectively.

Previous's pooled measured row-hit rate is 25.19% on Campus and 37.36% on
Sci-Art, versus 97.66% and 98.91% for the union-covering Bounded cache. The
synthetic observation that Previous can outperform Bounded despite moving more
bytes therefore does not appear on either natural completion workload. The
four-workload outcome is mixed: Previous wins on canonical Run009 and the
synthetic strong control, while Bounded wins on both current natural paths.
Weak/nontrivial/strong LoD class is not sufficient to explain the policy
winner; trajectory locality and persistence horizon are the defensible
mechanisms.

Current filter timing includes stable unique-ID extraction even on these
duplicate-free natural inputs. This averages roughly 43--44 ms on Campus and
281 ms on Sci-Art and is included in preparation and end-to-end time. The
current same-state policy rows are direct within each dataset, but their
absolute times must not silently replace or be pooled with the older
pre-multiset natural campaign.

Canonical narrative:
[`alod_ext_natural_previous_completion.md`](../../gs_framework/docs/results/alod_ext_natural_previous_completion.md).
Prospective protocol:
[`ALOD_EXT_NATURAL_PREVIOUS_COMPLETION_PROTOCOL.md`](../ALOD_EXT_NATURAL_PREVIOUS_COMPLETION_PROTOCOL.md).
Formal package:
[`manifest.json`](../../benchmark_artifacts/alod_ext_natural_previous_completion_001/manifest.json),
[`completion_summary.json`](../../benchmark_artifacts/alod_ext_natural_previous_completion_001/tables/completion_summary.json),
and
[`package_validation.json`](../../benchmark_artifacts/alod_ext_natural_previous_completion_001/checks/package_validation.json).

Evidence update: this topic note; registry added; reproduction matrix unchanged;
classification project-specific `ALOD-EXT` natural policy-completion
experiment; canonical artifact
`benchmark_artifacts/alod_ext_natural_previous_completion_001/`.


## 22. Campus/Sci-Art final-state provenance closure (2026-10-17)

The [provenance closure report](../ALOD_CAMPUS_SCIART_PROVENANCE_CLOSURE.md)
reconciles the historical association and final natural policy campaigns.
A CPU-only audit revalidated the existing post-multiset natural regression,
matched its runtime hashes to the final timing protocol, and independently
checked all final timing-frame funnel counts. Every per-camera selected,
filtered, and packed-survivor count agrees exactly with the historical gate.
The final-runtime regression therefore supplies the funnel evidence already;
no new training, hierarchy construction, GPU correctness run, or timing
campaign was required.

Use the final natural completion package for the main Campus/Sci-Art
three-policy comparison. Retain the earlier association timings as historical
pre-multiset implementation evidence. Inputs, resolved configurations, camera
order, timing definition, and per-frame dynamic transfer/cache counts agree;
measured runtime work differs. The final filtered paths record substantial
unique-ID extraction cost, and the reload path computes additional unique-count
metrics inside preparation without a separate timer. Do not infer timing
equivalence from semantic/funnel equality or claim that every millisecond of
the increase has been causally isolated. The older natural selection-bottleneck
interpretation must be scoped to its implementation state.

The report gives exact proposed replacements for both paper provenance
caveats and targeted dissertation table/discussion updates. Neither manuscript
was edited. Canonical evidence remains in the historical association,
post-multiset regression, and final natural completion packages; the new
[CPU audit](../data/alod_campus_sciart_provenance_audit.json) records hashes,
raw-derived counts/timings, and verification scope. The
[audit script](../data/audit_alod_campus_sciart_provenance.py) is reproducible
without importing Torch or executing a model.

Evidence update: this topic note and closure report; registry unchanged;
reproduction matrix unchanged; classification project-specific ALOD-EXT
evidence reconciliation. Run009 and synthetic findings remain unchanged.
