# ALoD `Render Full` Specification and Source Audit

Audit ID: `ALOD-REPRO-RENDER-FULL-SPEC-AUDIT-001`

Final gate: `STOP-RENDER-FULL-UNDERSPECIFIED`

Classification: `ALOD-REPRO` paper/source specification audit and negative
reproducibility result. This is not an `ALOD-EXT` experiment.

Audit date: 2026-09-18

## Question

Is the `Render Full` configuration described by the ALoD paper sufficiently
specified, together with the pinned released source, to support an independent
bounded rendering-cache reconstruction without introducing outcome-dominating
project assumptions?

## Answer

No, under the predeclared reconstruction and correctness requirements. The
paper specifies SPT-cut cache entries, distance-compatible reuse, Gaussian-row
capacity, LRU-style retention, and a 15-million-Gaussian experimental cache.
The release provides three different partial behaviors: evaluator no-cache,
viewer previous-working-set reuse, and a bounded mutable training cache. It
does not provide the render-specific compatibility thresholds, capacity
accounting, reduction mechanics, cache-state protocol, row ordering, or timing
boundary used for paper `Render Full`.

The decisive conflict is semantic. The paper states that compatible cached SPT
cuts can vary rendered detail with cache state, whereas the proposed
reconstruction gate required exact selected IDs and exact raster inputs versus
the fresh-cut evaluator. Preserving evaluator identity would require an
exact-cut or row-property cache that is not the paper-described compatible-cut
behavior. Combining the viewer thresholds with the trainer's bounded
retention would instead splice two distinct released paths. The available
evidence therefore supports a paper-guided adaptation, but not the requested
`ALOD-REPRO-RENDER-FULL-RECONSTRUCTION` classification.

No implementation, training, correctness run, or performance benchmark was
started. No new numerical result was produced.

## Scope and primary evidence

- Paper: [arXiv:2507.01110v4](https://arxiv.org/abs/2507.01110v4),
  *A LoD of Gaussians: Out-of-Core Training and
  Rendering for Seamless Ultra-Large Scene Reconstruction*, especially
  Sections 4 and 4.4, Equation 7, Table 3, Supplementary Figure 15, and
  Supplementary Sections A.1, A.4, and A.7.
- Released source: `/home/luis/tese/LoDOfGaussians`, commit
  `73f547341547ae021ff036ad74fdf73a1240b450`.
- Released evaluator: `eval_hierarchy.py`.
- Released viewer: `hierarchy_viewer.py`.
- Released bounded training analogue: `train_hierarchy.py`,
  `arguments/__init__.py`, `Docs/Hyperparameters.md`, and the released configs.
- Existing reproduction-boundary audit:
  [`alod_reproduction_boundary.md`](alod_reproduction_boundary.md).
- Machine-readable audit closure:
  [`data/alod_render_full_spec_audit_closure.json`](data/alod_render_full_spec_audit_closure.json).

Run009 facts are contextual rather than new audit outputs. Its canonical
hierarchy remains 7,072,627 total serialized rows, including 100,000 skybox
rows and 6,972,627 hierarchy rows, with SH degree 1 and maximum depth 55. The
audit did not alter this asset, its HSPT, its cameras, or any canonical result.

## Paper-only specification

| Behavior | Paper classification | Specification boundary |
| --- | --- | --- |
| External-memory properties | `EXPLICIT-PAPER` | Full Gaussian properties reside outside VRAM and selected properties are streamed on demand. |
| Cache entry | `EXPLICIT-PAPER` | One SPT cut and its cached camera-to-root distance; individual Gaussian lookup is explicitly avoided. |
| Reuse | `EXPLICIT-PAPER` | Equation 7 accepts the cached cut when the current/cached distance ratio lies within `D_min` and `D_max`; numerical values are absent. |
| Semantic effect | `EXPLICIT-PAPER` | Compatible reuse may change rendered detail according to cache state. |
| Insertion | `EXPLICIT-PAPER` | Newly loaded SPTs are added after use. |
| Retention | `IMPLIED-PAPER` | LRU and previous-frames language imply multi-frame retention, but not an executable ordering algorithm. |
| Eviction | `EXPLICIT-PAPER` intent | LRU-style write-back occurs after a memory threshold is exceeded; exact trigger and reduction mechanics are absent. |
| Capacity unit | `EXPLICIT-PAPER` | Cache size is expressed in Gaussians; the general experiment description reports 15 million. |
| Reduced capacity | `NOT-SPECIFIED` | No render reduction target is stated. |
| Capacity accounting | `NOT-SPECIFIED` | The paper does not say whether active rows, retained rows, skybox, or upper nodes count. |
| Warm-up and timing | `NOT-SPECIFIED` for Table 3 | Initial cache state, fill inclusion, synchronization, and exact timing boundary are not stated. |

Supplementary Figure 15 evaluates rendering FPS over 1,000 camera-path frames
for several capacities and says that up to `cache_size` Gaussians from previous
frames may be reused. A separate LoD sensitivity experiment is explicitly
pre-warmed; that statement does not define the Table 3 or Figure 15 cache-state
protocol.

## Released-source mapping

| Behavior | Source support | Finding |
| --- | --- | --- |
| SPT-cut identity and cached distance | `EXACT-SOURCE-MATCH` | Viewer and trainer retain complete SPT cuts with their cut distance. |
| Distance-compatible reuse | `PARTIAL-SOURCE-MATCH` | Viewer uses `0.3 < previous/current < 1.5`; trainer uses approximately `0.5 < previous/current < 1.3`; neither is identified as the paper renderer setting. |
| Bounded multi-frame retention | `SOURCE-ANALOGUE` | Fine training retains unrequested cuts; the viewer retains only compatible cuts requested in the next frame. |
| LRU/reduction | `SOURCE-ANALOGUE` | Training uses ordered tail eviction without explicit ages and reduces toward a second row target. |
| 15M/12M | `PARTIAL-SOURCE-MATCH` | Default arguments use 15M/12M and the guide recommends about 20% hysteresis; released MatrixCity uses 25M/20M. |
| Immutable render eviction | `ABSENT` | Released write-back includes mutable Gaussian properties and optimizer moments. |
| Complete bounded renderer | `ABSENT` | No released renderer combines HSPT, frustum, host properties, bounded retention, and paper timing. |
| Raster-input ordering | `CONTRADICTORY` | Evaluator, viewer, and trainer assemble loaded, reused, and retained rows differently. |
| Cold/warm timing protocol | `ABSENT` | No released Table 3 benchmark harness exists. |

## Assumption ledger

### Fully specified

- Cache entries are SPT cuts rather than independent row decisions.
- Each cached cut stores a camera-to-SPT-root distance.
- Reuse is controlled by Equation 7 and may alter selected detail.
- Capacity is measured in Gaussian rows.
- The paper's general experimental capacity is 15 million rows.
- Replacement is intended to be LRU-style.

### Strongly source-supported reconstruction choices

- Retain unrequested SPT cuts across frames.
- Keep at most one cached cut per SPT.
- Treat newly loaded or reused entries as recent and evict whole SPT cuts from
  the old end of the order.
- Use 12 million as the reduction target for a 15-million threshold, based on
  released defaults and the 20% recommendation.
- Discard immutable inference entries instead of applying training write-back.

These choices are plausible source analogues, not evidence of the authors'
unreleased renderer.

### Necessary but ambiguous

- Viewer versus trainer compatibility interval.
- Active-plus-retained versus retained-only capacity accounting.
- Whether skybox and upper-hierarchy rows count against capacity.
- Immediate versus delayed reduction and temporary overshoot.
- Periodic flush or no flush during rendering.
- Evaluator order versus load/reuse/resident-slot raster order.
- Empty, warmed, or trajectory-carried initial state and timing inclusion.

### Unrecoverable for the requested reconstruction

- The numerical render compatibility interval used for Table 3.
- Exact render capacity/reduction/accounting semantics.
- A cache behavior that is simultaneously faithful to paper-compatible SPT
  reuse and exactly identical to the evaluator's fresh selected IDs.
- The paper benchmark's cold/warm and synchronization boundary.

These gaps are outcome-dominating because they can change cache hits, H2D
traffic, selected rows, image output, occupancy, and frame time.

## Supported conclusions

- The paper specifies SPT-cut caching, distance-compatible reuse, Gaussian-row
  capacity, and LRU-style retention for out-of-core execution.
- No released renderer entry point at the pinned commit implements the complete
  paper `Render Full` configuration.
- The evaluator, viewer, and fine trainer expose distinct no-cache,
  previous-working-set, and bounded mutable-cache behaviors and must not be
  conflated.
- Several render-specific semantics needed to reconstruct Table 3 are not
  uniquely recoverable from the paper and source.
- Under the predeclared exact-evaluator-semantic validation, an
  `ALOD-REPRO-RENDER-FULL-RECONSTRUCTION` is not scientifically defensible.
- The least speculative capacity pair would be 15M/12M rows, but Run009's
  entire 7,072,627-row asset is smaller than the threshold, so the frozen
  71-camera workload could not validate capacity pressure or eviction.

## Unsupported / untested conclusions

- This audit does not establish that the authors' paper cache is ineffective.
- It does not establish that `Render Full` is impossible to implement in
  principle or impossible to recover with additional author code or details.
- It does not establish that a newly designed hybrid cache would match the
  authors' renderer.
- It does not measure `Render Full` frame time, H2D traffic, hit rate,
  occupancy, correctness, or image quality.
- It does not invalidate the paper's headline results.
- It does not change the faithful evaluator/viewer measurements or any
  `ALOD-EXT` result.
- It does not convert ALoD-EXT row residency or materialization filtering into
  paper-described behavior.

## Superseded interpretations

### Earlier belief

An explicitly labelled bounded paper-cache approximation might be implementable
as a final optional experiment by combining the closest released components.

### Later evidence

The detailed audit found that the render-specific compatibility interval,
capacity accounting, reduction timing, ordering, and state protocol are not
fixed by the available evidence. More importantly, paper-compatible SPT reuse
may change the cut, conflicting with the proposed exact-evaluator-semantic
gate.

### Final thesis position

A paper-guided or source-inspired cache could still be studied if separately
authorized and clearly labelled, but it must not be reported as a faithful or
scientifically identified reconstruction of paper `Render Full`. The existing
evaluator, viewer, and ALoD-EXT classes remain separate.

## Reproducibility relevance

- **Implementation reproducibility:** partial. The cache unit and qualitative
  reuse/retention design are reconstructible; the full render policy is not
  uniquely determined.
- **Experimental reproducibility:** limited. The paper asset, camera paths,
  backend, cache-state protocol, and timing boundary remain unavailable.
- **Result reproducibility:** not reached. No `Render Full` implementation or
  benchmark was executed, so paper timings were not tested.
- **Paper/source divergence:** strengthened. The complete paper configuration
  is distributed only as partial analogues across released programs.
- **ALoD-EXT external validity:** unchanged. The audit improves the required
  qualification of extension comparisons but adds no stronger-LoD or
  paper-scale evidence.

## Full dissertation relevance

Recommended placement: a short subsection in the ALoD reproduction
methodology/paper-source divergence section, with a short reminder in the
evaluation limitations or discussion. The message should be that the paper
defines the cache concept but the available publication and source do not
uniquely identify the complete `Render Full` execution policy. Detailed tables
can remain in an appendix or evidence note.

## Condensed 8-page paper relevance

Include briefly in limitations/discussion only. State that the comparison uses
released evaluator and viewer boundaries because the reported bounded render
cache has no uniquely reconstructible released entry point. Do not spend space
on the complete assumption ledger unless reproducibility is a central paper
theme.

## Claims for the writing agent

### Safe claims

- The ALoD paper describes compatible SPT-cut caching with LRU-style bounded
  retention.
- The pinned release exposes related behavior separately in evaluator, viewer,
  and fine-training paths.
- No released entry point implements the complete paper `Render Full` mode.
- Render-specific semantics required for a unique reconstruction remain
  unavailable, so the audit stopped before implementation.

### Claims requiring qualification

- A 15M cache is paper-reported, but 12M reduction is source-derived rather
  than a verified paper-renderer parameter.
- The trainer is the closest bounded source analogue, but it manages mutable
  training properties and optimizer state.
- The viewer is the released rendering reuse analogue, but it is previous-set
  only and uses a different compatibility interval.
- The conclusion concerns the public paper/source evidence at the pinned
  versions, not all possible unreleased author implementations.

### Claims that must not be written

- The project reproduced, benchmarked, or disproved paper `Render Full`.
- The trainer cache is the authors' exact rendering cache.
- Viewer reuse plus trainer eviction is source-faithful `Render Full`.
- ALoD-EXT row caching or materialization reduction is paper-described.
- The paper's cache provides no benefit or the paper's timings are wrong.
- The 15M/12M candidate was validated under capacity pressure on Run009.

## Comparison impact

This audit changes no canonical result. Run009 faithful timings, including
186.77 ms no-cache and 134.21 ms released-viewer reuse, remain unchanged.
`ALOD-EXT-FORMAL-RUN009-001`, including its 2.6064936690853813 speedup versus
the faithful no-cache baseline, also remains unchanged. The audit strengthens
the qualification that this extension comparison is not versus paper
`Render Full` and cannot decompose the extension gain relative to an
unreconstructed paper cache. It creates no new CityGaussian comparison and
does not support a cross-method ranking.

Authorities: faithful
[`rendering_summary.json`](../../../benchmark_artifacts/alod_faithful_reproduction_001/tables/rendering_summary.json)
and extension
[`formal_summary.json`](../../../benchmark_artifacts/alod_ext_formal_run009_001/tables/formal_summary.json).

## Final disposition

```text
STOP-RENDER-FULL-UNDERSPECIFIED
```

Reserved but not activated:
`ALOD-REPRO-RENDER-FULL-RECONSTRUCTION-001`.

Future implementation would require new primary evidence or an explicit
reclassification as a paper-guided/source-inspired adaptation with revised
correctness semantics. It must remain separate from `ALOD-EXT` and from the
released evaluator/viewer results.
