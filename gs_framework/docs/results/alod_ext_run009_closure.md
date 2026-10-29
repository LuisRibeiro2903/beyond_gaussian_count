# Run009 ALoD-EXT Evidence Closure and Cache Robustness

Status: `PASS-ALOD-EXT-RUN009-EVIDENCE-CLOSURE`

Campaign: `/home/luis/tese/benchmark_artifacts/alod_ext_run009_closure_001`

Classification: project-specific `ALOD-EXT`; this is not the paper's
`Render Full` path and does not change any `ALOD-REPRO-*` result.

## Outcome

The reviewer concern was correct. The previous formal gate compared the CPU
and CUDA conservative filters on all 71 held-out cameras, but its direct
faithful packed-projection false-negative check, retained-property check, and
RGB/alpha comparison covered only the 16 protected diagnostic cameras. The
new all-71 gate closes that gap: every faithful packed-projection survivor was
retained on every camera, with zero false negatives. Ordered CPU/CUDA filter
IDs, retained properties, packed-survivor IDs, RGB, and alpha were exact; the
property and image criteria were bitwise.

The cache robustness result separates two behaviors. Previous-working-set
reuse depends strongly on order: its measured row hit rate falls from 67.92%
in the canonical order to 11.75% in the fixed shuffle and 3.97% in the
geometry far-jump order. The 3.7M-row bounded cache is robust for this camera
set because it exceeds the 3,503,048-row complete materialization union. It
therefore has no evictions and transfers exactly one union from an empty cache
for every order. Measured-only traffic changes with the ten warm-up cameras;
cold-inclusive traffic does not.

The frozen 15-camera projected-scale stress path does not produce materially
stronger LoD adaptation. Every camera still has median selected depth 22 and
maximum depth 55, and maximum pairwise depth-distribution TVD is 0.01834,
below the frozen 0.05 material-variation threshold. Run009 remains
`WEAK-LOD`.

## Phase 1: evidence audit

| Check | Previous camera scope | Camera set | Criterion | Previous authority | Closure result |
| --- | ---: | --- | --- | --- | --- |
| Official/source semantic selection | 16 | protected diagnostic oracle IDs listed in the audit | exact ordered IDs, upper traversal, per-SPT cuts and touched/selected SPTs | `alod_faithful_reproduction_001/checks/selection_oracle_parity.json` | unchanged; official-source parity remains 16-camera evidence |
| CPU reference filter versus CUDA filter | 71 | complete frozen evaluation path | exact ordered materialization IDs | `alod_ext_formal_run009_001/checks/correctness.json` | unchanged and reconfirmed in the all-71 gate |
| Faithful packed survivors retained by conservative filter | 16 | protected diagnostics | exact ID-set inclusion, zero false negatives | same correctness artifact | extended to all 71; zero false negatives |
| Retained full properties and order | 16 | protected diagnostics | bitwise equality | same correctness artifact | extended to all 71; bitwise exact |
| RGB and alpha | 16 | protected diagnostics | bitwise equality, zero absolute error | same correctness artifact | extended to all 71; bitwise exact |

The complete camera IDs and artifact paths are recorded in
`checks/evidence_audit.json`. The new camera-by-camera results are in
`checks/conservative_correctness_71.json`.

The all-71 gate compares the extension with the validated faithful project
path. It does not retroactively expand the official-evaluator oracle from 16
to 71 cameras.

## Phase 2: frozen cache semantics and accounting

The dynamic cache entry is one 92-byte, SH-degree-1 hierarchy property row,
identified by immutable `global_node_id`. The 100,000-row skybox is persistent
and separate. A hit means that the requested hierarchy ID maps to a live GPU
slot at frame entry; otherwise the row is a miss and its complete properties
are gathered and transferred. Misses use free slots first. When necessary,
the cache evicts deterministic least-recently-used rows, protects current
hits, and breaks equal-age ties by slot index. A capacity is infeasible if it
cannot hold the complete current-frame hierarchy request.

Every fresh process starts with an empty dynamic cache. Warm-up renders the
first ten cameras of that trajectory order. Cache state survives warm-up, and
the measured sequence then restarts at camera zero. Historical steady H2D
totals exclude warm-up, the initial persistent skybox upload, compact-filter
metadata, CUDA-filter selected-ID H2D, retained-index D2H, and GPU-to-GPU
copies. The closure adds warm-up and cold-inclusive dynamic-property totals;
even the latter still excludes those separately listed transfers. The frozen
contract is `checks/cache_contract.json`.

## Phases 4 and 5: order robustness and capacity pressure

The fixed shuffle uses seed `20260916`. The far-jump sequence was frozen before
cache results by greedily maximizing the sum of normalized camera-centre
displacement and viewing-direction angle from the current camera. No cache
hit, transfer, selection, or timing result entered its construction.

All numbers below are means per fresh process over three replicates. Requested,
hit, miss, eviction, and transfer counts are deterministic across replicates.
Occupancy and traffic concern dynamic hierarchy property rows only.

| Order and policy | ms / FPS | Requested rows | Hit rate | Miss rows | Evicted rows | Mean / peak occupancy | Steady H2D MiB | Cold-inclusive H2D MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical, faithful no-cache | 182.74 / 5.47 | 243,682,008 | 0.00% | 243,682,008 | 0 | 0 / 0 | 21,380.2 | 24,254.4 |
| canonical, previous reuse | 70.57 / 14.17 | 36,206,972 | 67.92% | 11,616,487 | 0 | 509,957 / 1,599,906 | 1,019.2 | 1,076.5 |
| canonical, bounded 3.7M | 72.13 / 13.86 | 36,206,972 | 91.88% | 2,941,324 | 0 | 2,378,285 / 3,503,048 | 258.1 | 307.4 |
| shuffle, faithful no-cache | 182.45 / 5.48 | 243,682,008 | 0.00% | 243,682,008 | 0 | 0 / 0 | 21,380.2 | 24,400.1 |
| shuffle, previous reuse | 79.66 / 12.55 | 36,206,972 | 11.75% | 31,954,451 | 0 | 509,957 / 1,599,906 | 2,803.6 | 3,258.4 |
| shuffle, bounded 3.7M | 70.32 / 14.22 | 36,206,972 | 98.92% | 392,808 | 0 | 3,381,703 / 3,503,048 | 34.5 | 307.4 |
| far-jump, faithful no-cache | 181.34 / 5.51 | 243,682,008 | 0.00% | 243,682,008 | 0 | 0 / 0 | 21,380.2 | 24,306.9 |
| far-jump, previous reuse | 80.97 / 12.35 | 36,206,972 | 3.97% | 34,769,708 | 0 | 509,957 / 1,599,906 | 3,050.6 | 3,109.9 |
| far-jump, bounded 3.7M | 72.03 / 13.88 | 36,206,972 | 91.78% | 2,974,440 | 0 | 2,537,402 / 3,503,048 | 261.0 | 307.4 |

No-cache has 3,696,619 unique requested semantic hierarchy rows. Filtered
previous/bounded modes have a 3,503,048-row complete trajectory union. The
same camera membership makes these union sizes invariant to order.

The bounded cache's apparently exceptional 99.84% measured-only reduction in
the shuffled order is a warm-up-placement effect: those ten cameras load
3,110,240 union rows before measurement. Cold-inclusive bounded traffic is
3,503,048 rows, or 307.4 MiB, in all three orders. Against order-matched
no-cache cold-inclusive traffic, the reductions are 98.73%, 98.74%, and 98.74%
for canonical, shuffled, and far-jump order. The original 98.79% claim remains
correct for the explicitly measured-only canonical accounting.

The capacity ladder was frozen structurally: the 1,599,906-row maximum
single-frame request, a 2.6M rounded midpoint between that maximum and the
union, the existing 3.7M practical point, and the existing 5.3M low-pressure
point.

| Order | Capacity rows | ms | Hit rate | Eviction frames / rows | Mean / peak occupancy | Steady / cold-inclusive H2D MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical | 1,599,906 | 82.64 | 78.49% | 47 / 6,748,242 | 1,345,010 / 1,599,906 | 683.2 / 732.5 |
| canonical | 2,600,000 | 88.53 | 83.21% | 38 / 4,042,408 | 1,977,945 / 2,600,000 | 533.5 / 582.8 |
| canonical | 3,700,000 | 72.13 | 91.88% | 0 / 0 | 2,378,285 / 3,503,048 | 258.1 / 307.4 |
| canonical | 5,300,000 | 76.69 | 91.88% | 0 / 0 | 2,378,285 / 3,503,048 | 258.1 / 307.4 |
| far-jump | 1,599,906 | 85.55 | 56.08% | 51 / 14,832,614 | 1,339,075 / 1,599,906 | 1,395.4 / 1,441.8 |
| far-jump | 2,600,000 | 94.45 | 83.80% | 47 / 3,795,895 | 2,025,651 / 2,600,000 | 514.8 / 561.2 |
| far-jump | 3,700,000 | 72.03 | 91.78% | 0 / 0 | 2,537,402 / 3,503,048 | 261.0 / 307.4 |
| far-jump | 5,300,000 | 75.37 | 91.78% | 0 / 0 | 2,537,402 / 3,503,048 | 261.0 / 307.4 |

Capacity has the expected monotonic effect on residency: increasing it reduces
eviction and property transfer until the complete union fits. Frame time is
not monotonic below that point; the 2.6M runs are slower than the minimum runs
despite lower transfer. This is an observed cache-management/runtime result,
not evidence that more transfer helps. At 3.7M, the cache is 5.62% larger than
the union, eliminates all evictions in both tested orders, and is the fastest
bounded point. Increasing to 5.3M changes neither hit rate nor traffic and is
3.34--4.56 ms slower. The 3.7M choice is therefore defensible as a
union-covering, no-eviction operating point with modest headroom, rather than
as a trace-tuned latency optimum. The experiment does not identify the exact
smallest no-eviction capacity between 2.6M and 3.7M.

## Phase 6: geometry-defined LoD stress

The supplementary path contains 15 cameras selected without ALoD outcomes:
the formal cameras were sorted by `projected_scale_proxy`, divided into three
equal low/middle/high bins, sampled at five fixed quantiles per bin, and
ordered low/high/middle. Its projected-scale proxy spans 345.26 to 65,254.10.

Mean semantic selection is 3,489,191 rows. Mean materialization is 590,238
rows (16.58% of semantic selection), and mean packed-projection survival is
561,319 rows. Median selected depth is 22 for every camera; maximum selected
depth is 55 for every camera. Maximum all-pairs depth TVD is 0.01834. Adjacent
selected-set Jaccard ranges from 0.7315 to 0.9998 (mean 0.9331). The practical
cache renders at 77.92 ms / 12.83 FPS with an 89.59% measured hit rate, zero
evictions, 2,391,865 mean and 3,151,331 peak occupied rows, 78.7 MiB steady
traffic, and 276.5 MiB cold-inclusive traffic.

The trajectory changes view scale and selected-set overlap but not enough of
the hierarchy depth distribution to satisfy the predeclared stronger-LoD
criterion. Training a second hierarchy remains outside this campaign.

## Conference-paper and dissertation claim impact

The following existing canonical-order claims remain numerically unchanged:

- 186.77 ms / 5.35 FPS for the released evaluator-style no-cache baseline;
- 71.66 ms / 13.96 FPS for the fastest previous-reuse extension;
- the 61.63% frame-time reduction and 2.606x speedup on that formal order;
- 73.48 ms / 13.61 FPS and 774.2 MiB pooled measured-only H2D for the practical
  bounded path;
- the 98.79% measured-only H2D reduction versus pooled no-cache;
- the materialization/count funnel and separation from semantic selection;
- `WEAK-LOD`, and the statement that paper `Render Full` is not exposed.

The following language should be revised or qualified:

1. A correctness statement must distinguish the 16-camera official-source
   selection oracle from the new all-71 extension-versus-faithful packed
   projection, property, and image proof.
2. The 98.79% transfer claim must say `measured-only after ten warm-up frames`.
   The matched canonical cold-inclusive dynamic-property reduction is 98.73%.
3. `Previous reuse is fastest` must be scoped to the canonical formal order.
   It degrades on shuffled and far-jump orders; the 3.7M bounded cache is the
   robust path across the tested orderings.
4. The 3.7M capacity should be described as union-covering with 5.62% headroom
   and zero observed evictions, not as an arbitrary or globally optimal size.
5. The Run009 limitation should mention that a frozen scale-stress subset also
   remains weak, rather than suggesting the formal order alone caused the
   weak-LoD observation.

## Remaining limitations

- One fixed shuffle seed and one geometry-defined far-jump heuristic were
  tested; no adversarial cache-outcome search was performed.
- Cold-inclusive totals remain explicit dynamic-property accounting, not
  whole-bus traffic; skybox, metadata, ID/index, and runtime transfers remain
  separately excluded.
- The LoD-stress path reuses formal held-out cameras. It spans a large geometry
  proxy range but is not a new synthetic near-to-far camera family.
- The capacity ladder is deliberately coarse and does not locate the exact
  no-eviction threshold.
- Timing uses the same RTX 5080 host and three fresh processes per mode; wider
  hardware generalization is untested.
- Run009 remains a weak-LoD hierarchy, so cache behavior under a genuinely
  stronger adaptive hierarchy still requires a separate justified campaign.
- The exact paper asset, renderer, and `Render Full` entry point remain
  unavailable; none of these extension results is a faithful paper result.

## Canonical artifacts

- `manifest.json`: package status, immutable asset identity, and artifact
  hashes;
- `checks/evidence_audit.json`: exact pre-existing evidence and the genuine
  gap;
- `checks/cache_contract.json`: frozen cache and accounting semantics;
- `checks/conservative_correctness_71.json`: camera-by-camera all-71 proof;
- `trajectories/trajectory_manifest.json`: frozen order algorithms, seed,
  camera IDs, and hashes;
- `tables/cache_robustness.json`: per-frame, per-replicate, order, capacity,
  warm-up, steady, and cold-inclusive metrics;
- `tables/lod_stress.json`: scale/depth/set/materialization/render/cache stress
  results;
- `checks/package_validation.json`: final accounting and package gate.

No semantic selector, hierarchy, trained property, materialization predicate,
cache replacement policy, formal camera membership, or source-faithful
baseline behavior was changed.
