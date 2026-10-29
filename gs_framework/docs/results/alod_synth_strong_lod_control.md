# ALoD synthetic strong-LoD positive control

Experiment: `ALOD-SYNTH-STRONG-LOD-CONTROL-001`

Status: synthetic semantic control complete; ALoD-EXT phase blocked before
timing by exact released-selector duplicate semantic IDs.

## Outcome

The prospectively frozen analytic hierarchy is a valid strong positive
control for the unchanged released ALoD selector. It passes every original
`criteria_v1` predicate, every additional ordered-response gate, exact
official/native loading, and exact ordered official/reference selection parity
on all 25 cameras. This result is synthetic, excluded from the natural
prevalence denominator, and does not replace Stage B.

The requested ALoD-EXT performance question is **not answered** by this run.
At cameras 18--24, the exact released cut contains global node `100001` twice.
The already validated ALoD-EXT residency path requires unique semantic global
IDs, so correctness stopped at camera 18 and performance timing never began.
No selector, cut, camera, hierarchy, filter, or residency policy was changed
to hide that result.

## Synthetic hierarchy

The scientific asset has a 100,000-row transparent official prefix and a
balanced `1024 x 512` analytic plane tree: 524,288 leaves, 1,048,575 hierarchy
rows, 1,148,575 total rows, maximum depth 19, and 133,234,712 serialized bytes.
Its SHA-256 is
`c74f075421b16583bebb2ead0f457e59f705e1276730ce28c6e06166de39da1a`.

Parents are derived from their two children with opacity-area weights,
weighted centroids and SH coefficients, covariance mixture moment matching,
deterministic eigendecomposition into scale/rotation, and a projected-mass
opacity rule. The maximum covariance reconstruction error was zero. The
six-sigma maximum-axis median extent falls from 110.851 at the root to 0.168
at the leaves, a 659.83x root-to-leaf progression. Binary level sizes double
from one root to 524,288 leaves; alternating spatial splits yield the expected
approximately 2x/1x median maximum-axis extent ratios across successive
levels.

The independent parser accepted all topology, reciprocal-link, depth,
reachability, finite-property, and prefix invariants. The official native
loader reproduced exact field hashes for positions, log scales, rotations,
opacities, SH coefficients, and nodes. The native and reference HSPT counts
were exact. A second fresh-process official run reproduced loader metadata,
HSPT counts, camera order, selected row counts, and all 25 ordered selected-ID
payloads exactly. Official near/middle/far renders completed.

## Frozen trajectory and strong-LoD response

The 25 fixed `1024 x 1024` cameras use `fx=fy=512`, look from `+z` to the
origin, and follow `r_j = 40 * 2^(j/3)` for eight near-to-far octaves. The
official selector uses the unchanged two-pixel target granularity,
`distance_multiplier=1`, four-plane frustum convention, and HSPT/SPT
predicates.

| Measure | Result |
| --- | ---: |
| Selected-count relative range | 761.6061 |
| Minimum pairwise Jaccard | 0.00003099 |
| Maximum depth-distribution TVD | 0.9999602 |
| Projected-scale / normalized-mean-depth Spearman | 0.988589 |
| Theil-Sen normalized depth per scale octave | 0.101357 |
| Near-minus-far quintile normalized depth | 0.597319 |
| Normalized mean-depth span | 0.696075 |

All three `criteria_v1` predicates pass. The five consecutive near-to-far
regime median normalized depths are `0.6763, 0.4729, 0.2568, 0.0947, 0.0789`,
so the frozen monotonic-regime gate also passes. All other positive-control
gates pass, giving `PASS-SYNTH-STRONG-LOD-CONTROL`.

| Representative | Distance | Selected hierarchy rows | Mean depth | Median depth |
| --- | ---: | ---: | ---: | ---: |
| Near, camera 0 | 40 | 25,137 | 14.725 | 15.0 |
| Middle, camera 12 | 640 | 33 | 4.879 | 5.0 |
| Far, camera 24 | 10,240 | 4 | 1.500 | 1.5 |

Exact ordered official/reference selected-ID parity holds for all 25 cameras.
The far row count is the released four-entry multiset, not four unique IDs.

## Selector-margin and natural-hierarchy comparison

The synthetic margin payload covers 26,214,350 camera/SPT-row predicate
decisions. Of all hierarchy nodes, 4.736% change selected state over the path;
among nodes selected at least once, 99.998% change state. Only 0.181% of all
SPT-row decisions lie within normalized absolute margin 0.05 of either
predicate boundary. This is a broad, orderly crossing of a regular hierarchy,
not instability caused by most decisions sitting narrowly on a threshold.

The same read-only instrumentation was applied to each frozen natural official
oracle:

| Hierarchy | Stage-A label | Ever-selected nodes changing | All nodes changing | Global near-boundary fraction | Root/terminal extent ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic | strong control | 99.998% | 4.736% | 0.181% | 659.83x |
| Campus | `VALID-NONTRIVIAL-LOD` | 63.837% | 39.537% | 4.785% | 31.92x |
| Sci-Art | `VALID-WEAK-LOD` | 9.199% | 4.711% | 0.669% | 101.32x |
| Run009, 16 protected cameras | `VALID-WEAK-LOD` | 11.588% | 6.003% | 0.734% | 9.74x |

Campus changes a much larger share of its selected identity support than the
two weak natural cases, agreeing with its Stage-A label, but its previously
measured nearest-scale/depth Spearman is only -0.0713 and it does not show the
synthetic ordered response. Sci-Art has a much larger root/terminal scale
ratio than Campus while remaining weak. The near-boundary fraction also does
not order the four LoD outcomes. These observations rule out a simple claim
that either total scale span or boundary proximity alone explains natural
weakness. They are consistent with camera geometry, per-level scale
organization, and HSPT organization interacting, but three natural
hierarchies cannot establish causality.

## ALoD-EXT correctness barrier

The strong gate triggered the mandatory extension phase. Cameras 0--17 passed
the established correctness path before camera 18 reached the uniqueness
barrier. Exact official and independent cuts agree that cameras 18--24 contain
the ordered semantic IDs `[100001, 100001, 100002, 624289]`.

This follows directly from overlapping released conditions at the first SPT:

- SPT 0 has root global ID `100001` and `first_child == 0`;
- the released upper-render condition accepts `first_child <= 0`;
- the released candidate-SPT condition accepts `first_child >= 0`; and
- once distance exceeds the root minimum threshold, SPT population emits that
  root again.

The official/reference agreement shows this is not a project selector bug.
Deduplicating the cut would change exact official semantic order; making
residency duplicate-aware would change the already validated ALoD-EXT
implementation; dropping the far cameras or changing HSPT structure would
amend frozen scientific inputs after observing cuts. None was authorized.
Therefore the run is
`BLOCKED-SYNTH-ALOD-EXT-OFFICIAL-DUPLICATE-SEMANTIC-IDS`: no materialization
funnel, H2D traffic, timing, FPS, or speedup result exists for this control.

## Scientific answer

The experiment demonstrates that the unchanged released ALoD selector can
produce deliberately strong, monotonic, materially varying semantic LoD on a
large mathematically derived hierarchy. It does **not** determine whether
ALoD-EXT still finds and exploits execution-side redundancy under that full
condition, because exact released semantics become a duplicate-ID multiset on
the far part of the frozen trajectory and violate the pre-existing extension
correctness contract before benchmarking. Outcomes A/B/C remain unresolved.

A future closure requires an explicit new study choice: either define and
validate duplicate-aware residency while preserving multiset render semantics,
or prospectively construct a second hierarchy whose official cut remains
unique. Either choice is a new protocol, not a repair to this result.

## Quality closure

Campus/Sci-Art ground-truth quality was not executed inside this synthetic
campaign. The isolated, frozen-input route is specified in
`docs/planning/alod_prevalence_quality_closure.md`. Existing exact correctness
records are not relabeled as PSNR/SSIM/LPIPS evidence.

## Canonical evidence

- prospective protocol: `thesis_notes/ALOD_SYNTH_STRONG_LOD_CONTROL_PROTOCOL.md`;
- scientific hierarchy construction:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/performance.construction.json`;
- independent validation:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/performance.validation.json`;
- exact selector parity:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/official-selector-parity.json`;
- fresh-process determinism:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/official-fresh-process-determinism.json`;
- strong classification:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/strong-lod-classification.json`;
- margin instrumentation:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/selector-margins.json` and
  `selector-margins.npz`;
- visual panel:
  `benchmark_artifacts/alod_synth_strong_lod_control_001/visual/near_mid_far_panel.png`;
- ALoD-EXT blocker:
  `benchmark_artifacts/alod_synth_strong_lod_control_001/checks/ext_duplicate_blocker.json`;
- natural comparison:
  `benchmark_artifacts/alod_synth_strong_lod_control_001/natural_comparison/comparison.json`.
