# ALoD synthetic strong-LoD EXT ordered-multiset closure protocol

Protocol and experiment ID: `ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE-001`

Freeze date: 2026-09-26

Status: `FROZEN-BEFORE-IMPLEMENTATION-CORRECTNESS-AND-TIMING`

Classification: prospective project-specific `ALOD-EXT` compatibility and
controlled execution experiment. It is not released ALoD, does not alter the
official selector, and does not amend
`ALOD-SYNTH-STRONG-LOD-CONTROL-001`.

## Historical boundary and question

The parent synthetic experiment remains frozen with semantic outcome
`PASS-SYNTH-STRONG-LOD-CONTROL` and EXT outcome
`BLOCKED-SYNTH-ALOD-EXT-OFFICIAL-DUPLICATE-SEMANTIC-IDS`. Its cameras 18--24
produce the exact released ordered cut `[100001, 100001, 100002, 624289]`.
This closure neither deduplicates that cut nor relabels the parent attempt.

The new question is:

> When the released ALoD selector exhibits deliberately strong, monotonic
> semantic LoD, does ordered-multiset-faithful ALoD-EXT still reduce unique
> physical materialization, transfer, and end-to-end frame time?

## Frozen scientific inputs

The closure reuses without modification the parent experiment's analytic
hierarchy, hierarchy SHA-256, 25 cameras, camera order, official selector,
HSPT, two-pixel target granularity, four-plane frustum convention, conservative
pre-materialization predicate, cache replacement policies, timing boundaries,
and image-agreement thresholds. The hierarchy and trajectory remain:

- asset:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/performance.dhier`;
- asset SHA-256:
  `c74f075421b16583bebb2ead0f457e59f705e1276730ce28c6e06166de39da1a`;
- camera path:
  `runs/alod/ALOD-SYNTH-STRONG-LOD-CONTROL-001/cameras.json`;
- 25 near-to-far cameras with `r_j = 40 * 2^(j/3)`;
- released source authority: commit
  `73f547341547ae021ff036ad74fdf73a1240b450`;
- `target_granularity=0.00390625`, `distance_multiplier=1`,
  `SPT_root_volume=1.5`, and evaluator-effective `min_SPT_size=100`.

No selector, hierarchy, camera, HSPT structure, LoD threshold, conservative
filter equation, cache policy, correctness tolerance, renderer semantic, or
benchmark schedule may change in this experiment.

## Frozen ordered-multiset runtime model

The runtime must keep five concepts distinct:

1. **Semantic occurrences:** the exact ordered official selector output,
   including repeated global node IDs.
2. **Unique semantic IDs:** a deterministic, first-occurrence-ordered set of
   backing global node IDs plus an occurrence-to-unique inverse mapping.
3. **Unique materialized/resident rows:** the distinct property records that
   pass the conservative filter and are physically gathered, transferred, or
   stored in residency.
4. **Assembled renderer occurrences:** the retained unique backing rows
   reindexed into exact semantic occurrence order and multiplicity.
5. **Packed-projection survivor occurrences:** the renderer/backend output,
   without deduplication.

For semantic occurrences `[100001, 100001, 100002, 624289]`, deterministic
unique backing is `[100001, 100002, 624289]`, the inverse is `[0, 0, 1, 2]`,
and renderer assembly must contain four property rows in the original order.
Cache capacity and occupancy count three unique physical rows, not four
semantic occurrences.

For an input with no duplicate IDs, unique extraction must preserve the input
order and the inverse must be the identity, so the established natural-data
path degenerates to its previous behavior.

## Conservative-filter equivalence argument

The frozen CPU and CUDA conservative predicates index compact metadata only by
global node ID. For each row they use its serialized mean and maximum scale,
the frame camera transform and intrinsics, render dimensions, near/far planes,
`eps2d`, and the frozen numerical padding. Neither implementation reads the
semantic occurrence index, neighbouring IDs, occurrence order, multiplicity,
cache state, or properties of another row. Consequently repeated occurrences
of one global ID have identical depth and screen decisions for a fixed frame.

The closure therefore evaluates the unchanged predicate once per unique ID and
broadcasts its retain/reject decision through the occurrence-to-unique mapping.
This is mathematically equivalent to evaluating identical duplicate property
records repeatedly. It changes filter work and representation, but not the
established predicate or the retained ordered semantic multiset. If a code
audit or test disproves this row-local property, the experiment must stop
before timing.

## Implementation boundary

Residency remains keyed by unique global node ID. Unique rows may be gathered
or transferred at most once per residency state. After residency, a GPU
gather/reindex operation reconstructs the exact retained semantic occurrence
sequence. Its time and index bytes are included in the existing inclusive
preparation boundary and reported explicitly as assembly/reindexing cost.

The implementation may change only reversible occurrence mapping, unique-ID
cache lookup plumbing, final assembly/reindexing, instrumentation, tests, and
orchestration. It may not change any frozen scientific input or algorithmic
policy listed above.

## Correctness barriers

Timing is forbidden until all barriers pass. Cameras 18--24 are tested first,
with camera 18 receiving a detailed trace. Only then may the complete
25-camera gate run.

For every tested camera require:

1. exact official ordered semantic occurrence parity, with no sorting,
   deduplication, or permutation;
2. deterministic unique extraction and exact reconstruction
   `unique_ids[semantic_to_unique] == semantic_occurrences`;
3. bitwise property parity for means, scales, quaternions, opacities,
   SH/colors, and global node IDs after reassembly; repeated IDs remain
   separate identical rows;
4. zero conservative-filter false negatives relative to faithful packed
   projection, exact CPU/CUDA decisions, and exact multiplicity propagation;
5. exact ordered renderer-input ID parity with the faithful Released Evaluator
   path;
6. exact packed-projection survivor ID order and multiplicity wherever the
   backend exposes them deterministically; and
7. evaluator-to-EXT mean absolute RGB and alpha error at most `1e-6`, and
   maximum absolute RGB and alpha error at most `0.01`. Output PSNR and bitwise
   equality are diagnostics.

The duplicate-camera trace records semantic occurrences, unique semantic IDs,
the inverse mapping, retained unique IDs, retained semantic occurrence IDs,
assembled renderer IDs, packed survivor IDs, and RGB/alpha agreement.

## Natural-workload regression

Before performance, run a focused regression on canonical Campus and Sci-Art
camera traces. At minimum, verify unchanged semantic IDs, identity
occurrence-to-unique mapping for duplicate-free inputs, unchanged unique
materialized IDs, unchanged assembled IDs, unchanged packed survivors, and
the established image-agreement bounds. No training or full natural-data
performance rerun is authorized by this protocol.

## Capacity and performance schedule

The bounded-cache capacity is frozen as

```text
min(hierarchy_rows, ceil(filtered_union_unique_rows * 3700000 / 3503048)).
```

`filtered_union_unique_rows` is the union of unique physical hierarchy rows.
Repeated semantic occurrences never consume extra cache slots. The capacity
is selected before timing and cannot be tuned for FPS.

After correctness and regression pass, run:

1. Released Evaluator (No Cache);
2. ALoD-EXT CUDA Previous Reuse;
3. ALoD-EXT CUDA Bounded Cache.

Use the original 25-camera near-to-far order, first ten cameras as ordered
warm-up, state survival from warm-up, measurement restarted at camera zero,
three fresh processes, the balanced cyclic mode schedule, and the established
CUDA synchronization boundaries.

## Required reporting

Per camera and in aggregate, retain separate counts for semantic occurrences,
unique semantic IDs, duplicate occurrences, unique retained/materialized rows,
assembled renderer occurrences, and packed survivor occurrences. Report both
`unique materialized rows / unique semantic IDs` and, where useful,
`unique materialized rows / semantic occurrences`; do not reuse an ambiguous
materialized/semantic label.

Report selection, filter, inclusive preparation, host gather, cache management,
H2D, assembly/reindexing, renderer wall, nested rasterization, end-to-end time,
FPS, steady dynamic-property H2D, cold-inclusive dynamic-property H2D, and
image agreement. Nested timing diagnostics must not be summed; end-to-end is
inclusive preparation wall time plus renderer wall time.

The final interpretation must treat duplicate support as a compatibility
requirement, include its overhead, and answer whether materialization remains
substantially beneficial, remains beneficial at a reduced level, or has
little/no benefit under this controlled strong-LoD workload. No conclusion may
be generalized beyond this synthetic hierarchy.

## Evidence package

Canonical new outputs belong under
`benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/` and must
not overwrite the parent package. The package includes a machine protocol,
code/config hashes, duplicate-camera trace, all-camera correctness summary,
Campus/Sci-Art regression summary, capacity record, fresh-process schedule,
per-camera run records, aggregate table, image-agreement report, and final
package validation. The result memo is
`gs_framework/docs/results/alod_synth_strong_lod_ext_multiset_closure.md`.
