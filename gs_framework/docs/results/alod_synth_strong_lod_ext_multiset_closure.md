# ALoD synthetic strong-LoD EXT ordered-multiset closure

Experiment: `ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE-001`

Status: `PASS-ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE`

Classification: project-specific `ALOD-EXT` compatibility and controlled
execution experiment. It is not released ALoD or paper `Render Full`.

## Outcome

Ordered-multiset support closes the compatibility gap exposed by the strong
synthetic control. The extension retains a measurable execution benefit under
the unchanged strong-LoD semantic cuts. Relative to Released Evaluator (No
Cache), previous-working-set EXT reduces pooled mean end-to-end frame time by
45.82% (1.846x), from 17.679 to 9.577 ms, and the bounded-cache mode reduces it
by 14.24% (1.166x), to 15.161 ms. The previous mode reaches 104.41 FPS and the
bounded mode 65.96 FPS, compared with 56.57 FPS for the released-evaluator
baseline.

This is not evidence that duplicate support made EXT faster. Duplicate support
is the compatibility requirement that allowed the frozen workload to execute.
The measured comparison remains Released Evaluator versus ALoD-EXT on the same
official strong semantic cuts, with occurrence reconstruction overhead included
in preparation and end-to-end time.

The controlled parent `ALOD-SYNTH-STRONG-LOD-CONTROL-001` remains unchanged.
Its historical EXT status is still
`BLOCKED-SYNTH-ALOD-EXT-OFFICIAL-DUPLICATE-SEMANTIC-IDS`; this new prospective
experiment supplies the closure.

## Compatibility implementation

The runtime now separates:

1. exact ordered semantic occurrences;
2. first-occurrence-ordered unique semantic IDs and an occurrence-to-unique
   inverse;
3. unique retained and resident property rows;
4. renderer rows reassembled in exact retained semantic order and
   multiplicity; and
5. packed-projection survivor occurrences.

The conservative filter runs once per unique property row. This preserves its
mathematics because both CPU and CUDA predicates depend only on the row's mean
and maximum scale plus camera and render settings. They contain no occurrence
index, ordering, multiplicity, cache, or neighbouring-row term. The resulting
unique retain mask is broadcast through the inverse mapping. Previous reuse and
bounded cache remain keyed by unique global ID; duplicate occurrences do not
consume additional cache slots. A final GPU gather restores occurrence
multiplicity. Duplicate-free inputs take the identity path and avoid this
additional gather.

The Released Evaluator (No Cache) path continues to gather and transfer
duplicate occurrences as separate rows, because it is the unchanged baseline,
not the optimized EXT residency path.

The execution protocol records exact SHA-256 identities. Principal hashes are:

| Component | SHA-256 |
| --- | --- |
| Configuration | `f40a7868d67934021c2ab5954c0521f87ed9c781834ca6ad9887695e4854a8b1` |
| Materialization | `b3b40be33c5c02cc3140e87417a8dbd46f201d3c90e283373156d7b4fdfabbb2` |
| Host-scene assembly | `b1fe33752ec72870e0f11411dfadca69332c98d84b491a08146a7f517bb093fa` |
| Residency | `297d026cc121c5be2b6e7f491e500c56bd1be126bb557b3c576d0fe6c3c7a473` |
| Benchmark recorder | `c789d854ec06ed629582d00fb46173b59102d885aa6d9611aa072d069c9c22c0` |
| Campaign runner | `f4715e771c33ba611f8f01723159a69771a1ca7974b49cb0ac5b463bc1936aee` |

## Correctness

The required cameras 18--24 were tested first. Every camera has the official
hierarchy occurrence sequence `[100001, 100001, 100002, 624289]`, unique
backing `[100001, 100002, 624289]`, and inverse `[0, 0, 1, 2]`. All three
unique rows are retained. Renderer assembly restores the exact four-row
sequence, and faithful packed projection reports the same four occurrences.
Means, scales, quaternions, opacities, SH/colors, and global node IDs are
bitwise exact for every occurrence. RGB and alpha are bitwise equal, with zero
mean and maximum absolute error. Camera 18's full trace is embedded in the
duplicate correctness JSON and all arrays are preserved in the NPZ trace.

The subsequent full 25-camera gate passes exact official semantic order,
deterministic unique reconstruction, CPU/CUDA filtering, zero multiset false
negatives, bitwise properties, renderer-input row order and multiplicity,
packed-survivor order and multiplicity, and image agreement. Across all 25
cameras, RGB and alpha remain bitwise equal and have zero measured mean and
maximum absolute error, exceeding the prospective numerical requirements.

## Natural-data regression

The canonical Campus 24-camera and Sci-Art 21-camera correctness workloads
were rerun after the runtime change. Both pass every existing semantic,
filter, property, packed-survivor, and image gate. Their new correctness JSON
and compressed trace files are byte-for-byte identical to the prior canonical
artifacts. The duplicate-free unique path is therefore the identity mapping in
these workloads, and no natural-data behavior changed. No training or natural
performance campaign was rerun.

## Capacity and workload funnel

The union contains 49,661 unique retained hierarchy rows. Applying the frozen
rule

```text
min(1,048,575, ceil(49,661 * 3,700,000 / 3,503,048))
```

gives a bounded capacity of 52,454 unique rows. It covers the union with zero
evictions in every measured process and was fixed before timing.

Counts below include the 100,000 semantic skybox prefix, consistent with the
renderer-input accounting used by the prior ALoD-EXT campaigns.

| Camera | Semantic occurrences | Unique semantic IDs | Unique materialized rows | Assembled renderer rows | Packed survivors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Near, 0 | 125,137 | 125,137 | 46,526 | 46,526 | 25,137 |
| Middle, 12 | 100,033 | 100,033 | 100,033 | 100,033 | 33 |
| Far, 24 | 100,004 | 100,003 | 100,003 | 100,004 | 4 |
| 25-camera mean | 102,167.12 | 102,166.84 | 77,087.44 | 77,087.72 | 2,167.12 |

The pooled physical materialization ratio is 75.4525% of unique semantic IDs.
Relative to semantic occurrences it is 75.4523%. Thus the strong-LoD workload
reduces but does not eliminate execution-side materialization opportunity.
Most far cameras retain nearly the entire fixed skybox prefix, while nearer
cameras provide the largest conservative-filter reduction. The duplicate itself
changes physical-versus-occurrence count by one row on seven cameras; it is not
the source of the broader materialization reduction.

## Performance

Each mode used three fresh processes. Each process rendered the first ten
cameras as warm-up, preserved reuse/cache state, restarted at camera zero, and
measured all 25 cameras. Timing columns are nested diagnostics and are not
summed. End-to-end is inclusive preparation plus renderer wall time.

| Mode | End-to-end ms | FPS | Selection ms / share | Filter ms | Preparation ms | Host gather ms | Cache ms | H2D ms | Assembly ms | Reindex ms | Renderer ms | Raster ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Released Evaluator (No Cache) | 17.679 | 56.57 | 0.442 / 2.50% | 0.000 | 11.468 | 0.103 | -- | 0.250 | 0.262 | 0.0003 | 6.182 | 5.940 |
| EXT Previous | 9.577 | 104.41 | 0.388 / 4.05% | 4.730 | 7.110 | 0.114 | 0.018 | 0.239 | 0.271 | 0.075 | 2.445 | 2.314 |
| EXT Bounded | 15.161 | 65.96 | 0.403 / 2.66% | 4.753 | 12.418 | 0.030 | 0.058 | 5.424 | 0.343 | 0.078 | 2.715 | 2.554 |

The measured means include deterministic long-latency frames that recur in all
three fresh processes at mode-specific camera positions. They are retained by
the prospective end-to-end protocol; no trimming or substitution was applied.
In particular, bounded H2D timing includes a repeatable transfer/allocation
stall despite very small steady byte traffic.

| Mode | Steady dynamic-property H2D | Cold-inclusive H2D | Reduction versus no-cache |
| --- | ---: | ---: | ---: |
| Released Evaluator (No Cache) | 14,953,128 B | 29,815,728 B | -- |
| EXT Previous | 13,706,160 B | 27,378,372 B | 8.34% steady / 8.17% cold |
| EXT Bounded | 34,224 B | 13,706,436 B | 99.77% steady / 54.03% cold |

Previous reuse therefore provides the largest mean frame-time benefit on this
ordered path, while the union-covering bounded cache nearly eliminates steady
dynamic-property transfer but pays a larger measured preparation cost. The
bounded compatibility overhead is not hidden: mean occurrence reindexing is
0.078 ms and is included in its 12.418 ms preparation time. Previous mode has
0.075 ms mean reindexing.

## Scientific answer and thesis implication

Yes. On this controlled synthetic hierarchy, ALoD-EXT retains a measurable
benefit when semantic LoD is deliberately strong. Previous reuse gives a
substantial 45.82% mean frame-time reduction. Bounded cache gives a smaller but
still measurable 14.24% reduction while removing 99.77% of steady measured
dynamic-property H2D. Unique materialization remains 24.55% below unique
semantic selection in pooled row accounting.

The result supports semantic LoD and execution-side materialization as
complementary on this workload, while also showing that strong semantic LoD
reduces the remaining opportunity unevenly along the trajectory. At the far
duplicate cameras, hierarchy selection is already only four occurrences and
three unique rows, so EXT cannot materially reduce those hierarchy rows; its
remaining work is dominated by the fixed semantic skybox prefix and
compatibility/filter overhead. This mechanism result must not be generalized
to natural hierarchies, other camera paths, or paper `Render Full`.

## Instrumentation amendment

The first nine-process timing attempt revealed that new multiset fields were
not copied from runtime metrics into `frames.jsonl`. That attempt is preserved
under `attempts/instrumentation_missing_001` and excluded from the final result.
An instrumentation-only amendment added the missing record fields without
changing selection, filter mathematics, cache policy, capacity, timing
boundaries, hierarchy, or cameras. A second complete nine-process execution
supplies the reported result. Package validation confirms 225 final measured
frames, required metric presence, exact code hashes, complete scheduling, and
zero bounded evictions.

## Canonical evidence

- prospective protocol:
  `thesis_notes/ALOD_SYNTH_STRONG_LOD_EXT_MULTISET_CLOSURE_PROTOCOL.md`;
- machine protocol and amendment: `protocol.json`,
  `protocol_amendment_001.json`, and `protocol_execution_002.json`;
- duplicate and full correctness:
  `checks/duplicate_cameras_correctness.json` and
  `checks/all_cameras_correctness.json`;
- natural regression: `checks/natural_regression_summary.json`;
- image agreement: `checks/image_agreement.json`;
- capacity: `checks/all_cameras_correctness.json`;
- fresh-process schedule: `checks/schedule.json`;
- aggregate and representative/per-camera tables: `tables/aggregate.json` and
  `tables/per_camera_summary.json`;
- raw final executions: `runs/ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE-001/`;
- final manifest and validation: `manifest.json` and
  `checks/package_validation.json`.

All package-relative paths above are under
`benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/`.
