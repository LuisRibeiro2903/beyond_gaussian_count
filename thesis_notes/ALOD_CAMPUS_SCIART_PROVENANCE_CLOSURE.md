# ALoD Campus/Sci-Art provenance closure

Audit date: 2026-10-17. Classification: project-specific `ALOD-EXT` evidence reconciliation, not a new rendering experiment or released ALoD `Render Full` reproduction.

## Executive outcome

**RESOLVED for final-state funnel provenance and publication reconciliation.**
The already executed post-multiset natural regression supplies the requested final-runtime funnel and correctness evidence. Its runtime hashes match the final natural timing state. Independent reanalysis also finds the same per-camera semantic, retained, and packed-survivor counts in every final timing replicate. No count changes, including single-row differences, were found.

**No retraining, hierarchy reconstruction, model modification, GPU execution, or timing rerun was needed.** This audit recalculated existing artifacts only. The two paper statements that funnel evidence exists *only* in the earlier runtime are obsolete and can be replaced. Timing equivalence between the two campaigns must still **not** be claimed.

The discrepancy is a change in the measured implementation, including multiset compatibility and instrumentation work inside preparation, under the same recorded inputs/configuration and timing definition. The filtered modes directly record substantial unique-ID extraction time. The reload mode performs three additional, separately untimed unique-count operations inside preparation. These establish concrete causes of additional work; the records do not isolate the causal contribution of every changed operation or explain every millisecond.

## Evidence key and reproducibility

All paths below are relative to the workspace root. These aliases identify canonical, preserved packages:

- **H** — [historical association package](../benchmark_artifacts/alod_ext_prevalence_association_001/manifest.json), `ALOD-EXT-PREVALENCE-ASSOCIATION-001`.
- **R** — [post-multiset regression package](../benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/manifest.json), `ALOD-SYNTH-STRONG-LOD-EXT-MULTISET-CLOSURE-001`.
- **F** — [final natural timing package](../benchmark_artifacts/alod_ext_natural_previous_completion_001/manifest.json), `ALOD-EXT-NATURAL-PREVIOUS-COMPLETION-001`.
- **A** — [this audit's machine-readable findings](data/alod_campus_sciart_provenance_audit.json), including exact counts, means, input hashes, configurations, source paths, execution Git bases, and code hashes.
- Reproducible [CPU-only audit script](data/audit_alod_campus_sciart_provenance.py). It imports NumPy, not Torch; it does not execute a renderer. With no output argument it only reads. `--output` exclusively creates a new file and refuses to overwrite one.

From the workspace root:

```bash
/home/luis/miniconda3/envs/nerfstudio2/bin/python \
  thesis_notes/data/audit_alod_campus_sciart_provenance.py
```

The audit verified every listed artifact's SHA-256 and byte size: H has 100 entries, F 95, and R 44. It rehashed the actual two hierarchy assets, camera files, oracle manifests, parent configuration, regression authority, and all seven final protocol source files. All checks passed. These checks verify preserved contents, not an independently attested execution history. A records this audit; H/R/F remain the original numerical authorities.

## 1. Experiment provenance matrix

| Evidence | Execution/revision identity | Exact authority | Scope and aggregation |
| --- | --- | --- | --- |
| Historical funnel | H, September 23, 2026 UTC execution period; runtime identities in H/`protocol_amendment_001.json`; timing processes report dirty base `bc4246a3f9309e59c49b805f32024952278f31cc` | H/`checks/{campus,sciart}_correctness.json`, `traces/{campus,sciart}_trace.npz`, `tables/association_summary.json` | One complete canonical camera sequence per correctness gate; row counts are per-camera means; percentages divide pooled counts. All-camera correctness passed. Gate JSON itself does not record a separate Git HEAD. |
| Historical timings | H; all 18 process summaries report the same dirty base `bc4246a3f9309e59c49b805f32024952278f31cc`; amended execution file hashes take precedence | H/`runs/ALOD-EXT-PREVALENCE-{CAMPUS,SCIART}-001/{nocache,previous,bounded}/replicate-{01,02,03}/{frames.jsonl,warmup_frames.jsonl,summary.json,resolved_config.json}` | Three fresh processes per policy; 10 warmups; full sequence once per process; pooled arithmetic frame mean. Previous already exists in H although the dissertation table shows only Reload/Bounded. |
| Final-state funnel and correctness | R/`protocol.json` and `protocol_execution_002.json` share the final host/materialization/residency/selection/type hashes; recorder changes between those protocols are diagnostic field exposure | R/`checks/{campus,sciart}_regression.json`, R/`traces/{campus,sciart}_regression_trace.npz`, R/`checks/natural_regression_summary.json` | The natural GPU correctness executions were already rerun after the multiset change. Their JSON and NPZ are byte-identical to H. Reused here, not newly executed. |
| Final paper timings | F; all 18 processes report dirty base `fd34041a8d45965d392f0bcc2030cc073f9a3de6`; F/`protocol.json` plus its administrative hash amendment | F/`runs/{campus,sciart}/{nocache,previous,bounded}/replicate-{01,02,03}/`, F/`tables/completion_summary.json`, F/`checks/package_validation.json` | Same final implementation across the three policies; balanced cyclic order; three processes each; 72 measured frames per Campus policy and 63 per Sci-Art policy. |
| Independent final funnel cross-check | This CPU-only audit of F raw frames, using F's recorded revision/configuration | A/`datasets` and audit assertions over all F/`frames.jsonl` | Every final Previous/Bounded frame matches R's selected, materialized, and survivor counts; Reload matches selected and survivors and materializes the full selected population. No GPU regeneration required. |

H process summary timestamps span 2026-09-23 23:05:37–23:09:30 UTC; F spans 2026-09-26 18:16:14–18:22:21 UTC. These are summary-record times, not inferred training dates. H's result/runner/config were subsequently packaged in implementation commit `4a0c3af81c61894a87bd8cd5ac4ceff54c3cd128`; F's result/postprocessor in `d9cdd17c0e6a7cc3222a1ac044cccd51f5db1d22`. **Neither packaging commit is a clean, complete snapshot of the runtime used at execution.** The runtime files remain untracked/modified in the implementation repository; use the sealed file hashes, not a clean-checkout assumption.

Root writing-history commit `0381f6f` records natural completion evidence. Later paper edits include `da0bee8` (timing evidence clarification), `f8241c1` (execution study clarification), and current audit-entry root HEAD `1c750fe`. These are writing provenance, not experimental revisions.

### Frozen inputs and configuration

The common parent is [ALOD-EXT-PREVALENCE-ASSOCIATION-001.json](../gs_framework/configs/alod/ALOD-EXT-PREVALENCE-ASSOCIATION-001.json), SHA-256 `7ea720ae072812cbde2ffbc03ed32dc140af795e1649eae58f1859f862c6fb5e`. Exact asset/camera/oracle paths are in H/`protocol_amendment_001.json` and A/`datasets/*/inputs`.

| Input | Campus | Sci-Art |
| --- | --- | --- |
| Dataset | `ALOD-PREV-H3DGS-CAMPUS-CHUNK-01` | `ALOD-PREV-US3D-SCIART-01` |
| Hierarchy | Dataset's `TRAIN-001/output/*TRAIN-001.dhier` | Dataset's `TRAIN-001/output/*TRAIN-001.dhier` |
| Hierarchy SHA-256 | `8743b7d34f4c61b638f5e6528539b8dca021b9752ddd88d076a66608641815c4` | `acc49a863e022a6871391a224dfa01e2de24133a168c3e4498c2c8b83afa3010` |
| Camera SHA-256 | `96e363f578e09d9861473fa6587509ae3f858ef1c60dda0d3e4e63372d5b5aa9` | `b43b3d018a926497e73e0634775bcad7835a54cc0a4ea3e4b18fb790da509def` |
| Oracle manifest SHA-256 | `171bb1c12b48402ebd78465fe466def2d1b929580f6ecf5b06e262feba2801d6` | `5bf84d3ee1a6a1f88210ed6f0e260fcf9617bccc37b0eae9660d16c77ee55cac` |
| Trajectory suffix | `EVAL-24-PUBLIC-HELDOUT-V1` | `EVAL-21-PUBLIC-VAL-V1` |
| Cameras / actual rendered size | 24 / 1436 × 1075 | 21 / 1216 × 912 |
| Camera resolution scale | 1.0 | 0.25 |
| HSPT volume / minimum SPT size | 9600 / 100 | 1228800 / 100 |
| Target granularity | 0.0027701877932439297 | 0.0005407377082833597 |
| Fixed capacity / retained hierarchy union | 621377 / 588300 rows | 2666887 / 2524927 rows |

Camera IDs and exact order are retained in A/`datasets/*/camera_ids`; every measured sequence and its first-ten-camera warmup agrees. The runner forwards the oracle's exact selector camera values. H's prospective camera-precision amendment did not change the frozen camera identities or selection.

Every corresponding H/F resolved configuration is identical, across all replicates. Both use packed gsplat, pinhole, classic rasterization, RGB, SH1, `eps2d=0.3`, `radius_clip=0`, near/far 0.01/1e10, distance multiplier 1, and a 100000-row skybox prefix. Reload uses no materialization filter; Previous/Bounded use `conservative_pinhole_gpu`. Host pinning and mapping/property validation are off during timing. There is no asynchronous transfer, prefetch, or optimized selector. Correctness separately enables attribute validation.

Both campaigns report RTX 5080, Python 3.12.12, Torch `2.11.0.dev20260128+cu129`, CUDA 12.9, gsplat 1.5.3. This is version/environment agreement; it is not proof of identical clocks, scheduling, thermals, or all unsealed dependency binaries.

**Metadata trap:** `resolved_config.camera_path.resolution` retains a generic 1920 × 1080 fallback. Use `configuration.width/height` and the raw frame/image dimensions above; those agree across campaigns. This metadata inconsistency does not indicate a resolution change.

## 2. Code revision comparison

Full execution hashes are in H's amended protocol, F's protocol/amendment, and A/`historical_implementation` and `final_implementation`. The relevant comparison is:

| File under gs_framework | H execution SHA-256 | F execution SHA-256 |
| --- | --- | --- |
| `scene_runtime/alod/selection.py` | `f893713fd0f51d8e4defe962052bf5b03682df09b8c527dc763b7b3bc122d55e` | Same |
| `scene_runtime/alod/materialization.py` | `351aa70af87efa43773bf4a8351b8ab1774b47830f0ec78dbbe5ea350a069330` | `b3b40be33c5c02cc3140e87417a8dbd46f201d3c90e283373156d7b4fdfabbb2` |
| `scene_runtime/alod/host_scene.py` | `894ebc4cf44d817d67786100c2aa9322973be4ee0132feff131f00bc33ee7d6d` | `b1fe33752ec72870e0f11411dfadca69332c98d84b491a08146a7f517bb093fa` |
| `scene_runtime/alod/residency.py` | `66e2fa2ac5d8329e906a25a90ddb68f31ebcbe9964ca2a74f7a4df126a708f79` | `297d026cc121c5be2b6e7f491e500c56bd1be126bb557b3c576d0fe6c3c7a473` |
| `benchmarks/benchmark_real_scene.py` | `f83670e2c3f97b5932009af8412766b12675dfdc509947379c39ca38f9190873` | `c789d854ec06ed629582d00fb46173b59102d885aa6d9611aa072d069c9c22c0` |
| `scripts/alod/alod_ext_prevalence_association.py` | `5e2fed852d3d3f1043759de1f28d43de9911dd4554f1cc5aeef8b74e9f7fa350` | `9c132e919c486d040e1de32dffa75fa693e9a0d78445e2245365e6dd828652cf` |

F's `scene_runtime/types.py` hash is `821db12c83e329ca3081197f1270172be4ea0cd62f0aaf105e67488ebde38f8d`. F's original protocol mistyped the residency hash; [protocol_amendment_001.json](../benchmark_artifacts/alod_ext_natural_previous_completion_001/protocol_amendment_001.json) supplies the correct value above. It matches R and the actual file. No runtime/result correction was involved.

Specific changed work, established by the sealed current source and contemporaneous [multiset closure memo](../gs_framework/docs/results/alod_synth_strong_lod_ext_multiset_closure.md):

1. `materialization.py:187–205,628–739`: stable first-occurrence unique IDs and inverse mapping, filtering unique rows, propagation of the retain mask back to ordered occurrences, and mapping validation. This wraps the existing predicate rather than changing its geometric rule.
2. `host_scene.py:334–418`: filtered paths call the multiset wrapper, residency addresses unique hierarchy IDs, assembly restores occurrence multiplicity if needed. Natural inputs are duplicate-free, so the extra occurrence attribute gather is skipped; unique extraction and propagation still run.
3. `materialization.py:173–184,216–232` and `host_scene.py:450–456`: identity materialization leaves unique arrays unset. Reload's new metrics consequently call `np.unique` on semantic IDs twice and materialization IDs once. These execute inside `prepare_frame`, before its timer ends at line 558; they have no dedicated duration field. A zero Reload filter timer does **not** mean no new CPU bookkeeping.
4. Residency preserves the same policies: uncached occurrence gathering, previous-row reuse, bounded row-LRU. The unique-input validation contract now permits duplicate occurrences in the uncached path (`residency.py:123–133`). Natural H2D requests, hits, misses, evictions, and bytes match H/F **per frame**, including warmup; no policy change is inferred from the timing difference.
5. The recorder exposes unique/multiset diagnostic fields. R's instrumentation amendment explicitly records the recorder transition from `f82b618e…` to `c789d854…`. H's earlier exact-camera-forwarding changes were later moved into the runner; H's memo records restoration of the shared helper to `f82b618e…` and runner packaging at `9c132e91…`. That refactor did not regenerate H measurements.

There is no tracked historical `materialization.py` commit in the implementation Git history, and no sealed old source snapshot was found in the investigated packages. Thus a complete literal old/new runtime diff cannot be supplied. The precise execution fingerprints and verified final call sites above are stronger evidence than attributing all changes to `d9cdd17`, which only commits the final memo/postprocessor. HSPT/filter historical byte-for-byte source identity is not claimed where the protocol did not seal that source; behavior is supported by the oracle and regression gates.

## 3. Timing discrepancy and comparability

The original and final means independently recompute from each campaign's raw measured frames (A/`timings`; raw paths and revisions in section 1):

| Dataset | State | Reload ms | Previous ms | Fixed/Bounded ms |
| --- | --- | ---: | ---: | ---: |
| Campus | Historical H | 23.334592 | 19.067133 | 17.291735 |
| Campus | Final F | 90.249640 | 66.812379 | 65.667085 |
| Sci-Art | Historical H | 129.622887 | 74.008581 | 62.625081 |
| Sci-Art | Final F | 521.086160 | 378.908688 | 366.419835 |

The dissertation's 23.335/17.292 and 129.623/62.625 are correctly rounded H values. The paper's values are correctly rounded F values. Neither set is an arithmetic error.

**Same measurement definition, different measured work.** Both define end-to-end as synchronized preparation plus renderer wall time. Every raw row satisfies `end_to_end_frame_time_ms = preparation_time_ms + frame_time_ms`. The final code synchronizes before/after `scene.prepare_frame` (`benchmark_real_scene.py:717–730`) and adds renderer wall time (line 793). It is not one timer over the entire benchmark loop: initial frame-camera construction, between-stage memory probes, result checking/serialization, and startup are outside this reported boundary. Existing H protocols and raw algebra support the same definition; unavailable old source prevents a fresh line-by-line synchronization comparison. There is no evidence that H omitted semantic selection, filtering, gathering, transfer, assembly, or rasterization and later added those stages.

Both start each fresh process cold, run the first ten cameras as warmup, retain cache state, then restart at camera zero for the full measured path. Means pool all frames across three processes, without trimming. Initialization and initial persistent skybox upload are outside measured frames. Skybox filtering/assembly/rendering remain inside frame work. Dynamic-property traffic counts 92 bytes per hierarchy row and excludes persistent skybox, selector metadata, ID/index traffic and device-to-device work. This distinction is identical in H/F.

### Directly observed added filtered-path cost

| Dataset / policy | H filter ms | F filter ms | F unique extraction ms | F occurrence propagation ms |
| --- | ---: | ---: | ---: | ---: |
| Campus / Previous | 2.443231 | 48.026074 | 42.918514 | 1.990007 |
| Campus / Bounded | 2.469878 | 49.850305 | 44.362753 | 2.017627 |
| Sci-Art / Previous | 4.992126 | 301.052962 | 281.485211 | 10.142622 |
| Sci-Art / Bounded | 5.310729 | 300.790990 | 280.725820 | 10.132468 |

Source: A/`timings/{historical,final}/{campus,sciart}/{previous,bounded}/means_ms`, independently recomputed from H/F, with the revisions/configurations above. These timers are nested; do not add filter time to preparation or add all stage columns together. Stable uniqueness and propagation explain most of the filtered paths' extra measured preparation work without any hypothesis based only on timing ratios.

Reload's mean preparation grows from 19.884693 to 86.177187 ms (Campus) and 124.036819 to 509.460194 ms (Sci-Art). Selection, host gather, transfer, and rendering alone do not account for that increase. The three unique-count operations are a verified additional code path inside this boundary, but their isolated historical cost is **not recorded**. It is therefore supportable to identify instrumentation/multiset bookkeeping as a concrete source of overhead, but not to assign the entire Reload difference to those operations as a measured causal result.

H/F have identical dynamic transfer/cache-count traces, not merely equal aggregate totals. Larger assets, changed resolution/order, cache capacity/initialization changes, increased requested rows, or extra transferred property volume are not supported explanations. Host gather, transfer, and renderer latencies nevertheless differ; equal bytes do not imply equal latency. Machine load, clocks, scheduling, allocator state, and unsealed dependency differences remain unquantified. No new profiling or performance tuning was done.

**Comparison classification:** F's three rows are direct, same-state comparisons within each dataset. H's rows are direct within their earlier state. H-to-F is a historical implementation comparison, not a controlled performance ablation or evidence of timing equivalence. The final values describe these instrumented implementations; they are not instrumentation-free algorithm costs. In particular, the metric bookkeeping has different paths for Reload and filtered policies.

## 4. Final-state funnel validation

The supervisor's preferred route was checked first: reproduce final-state counts using existing correctness infrastructure and frozen assets. R already contains that execution. Its shared runtime hashes equal F's; current files also match. R's original and final execution protocols differ in recorder field exposure, not the natural correctness runner or the host/filter/residency/selector implementation. F explicitly validates and reuses R. Therefore another rendering run would duplicate available evidence.

| Dataset | Pooled selected | Pooled materialized | Pooled packed survivors | Mean selected | Mean materialized | Mean survivors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Campus | 12723995 | 3810896 | 3519197 | 530166.458333 | 158787.333333 | 146633.208333 |
| Sci-Art | 53522320 | 16109479 | 15377098 | 2548681.904762 | 767118.047619 | 732242.761905 |

| Dataset | Materialized / selected | Survivors / selected | Largest H/R per-camera count difference | Difference in either pooled percentage |
| --- | ---: | ---: | ---: | ---: |
| Campus | 29.950467600781% | 27.657956483007% | 0 rows | 0 percentage points |
| Sci-Art | 30.098618669744% | 28.730253098147% | 0 rows | 0 percentage points |

Authority: R/`checks/campus_regression.json` (SHA-256 `8aee50f44585dfb101de410f1a9e0a38af98f3d0312b4fc5ca42087e91c08566`) and R/`checks/sciart_regression.json` (`5f5e48120180892c4736211f1fd43584d9188cfaed1c20d94c3f6aef027b4f9e`); exact sums and raw F cross-check in A. These are the final-runtime hashes in section 2, using the common parent configuration. H correctness JSON has exactly the same hashes. All NPZ byte identities were independently rechecked.

Counting conventions:

- A count in the publication table is the arithmetic mean over the canonical 24 or 21 cameras, rounded to the nearest row.
- A percentage is `100 × sum(numerator) / sum(semantic_rows)`, **not** the mean of per-camera percentages and not a ratio of rounded displayed means.
- Semantic rows include all 100000 skybox occurrences per camera. Retained rows include surviving skybox rows; packed survivors include surviving skybox occurrences. Dynamic-property H2D/capacity counts instead exclude the persistent skybox.
- These two natural workloads contain no duplicate semantic hierarchy IDs; unique and occurrence populations coincide. The audit also verifies no duplicate packed-survivor IDs in their NPZ traces. This does not authorize deduplication for synthetic or arbitrary workloads.
- F's Reload path intentionally materializes all semantic rows. The funnel's reduced materialization column corresponds to the conservative-filter extension (Previous/Bounded), not all three policies.
- Every F replicate agrees per camera: selected and packed counts for all three policies, retained/assembled counts for both filtered policies. Every final frame reports zero duplicate semantic occurrences. No averaging concealed small differences.

The published rounded percentages 29.9505%, 27.6580%, 30.0986%, and 28.7303% remain supported unchanged.

## 5. Correctness: scope of reused and fresh checks

**Reused GPU correctness:** `alod_ext_prevalence_association.py:279–430` instantiates Reload and Previous-filtered scenes, forwards exact oracle camera values, and compares both ordered semantic ID arrays with the official oracle. It compares CPU and CUDA retained IDs and kept indices, then compares retained means, quaternions, scales, opacities, colors/SH and global IDs bitwise against the selected Reload attributes. It requires exact ordered packed-survivor IDs, zero false negatives, and bitwise RGB/alpha. R reran this gate on all natural cameras after the multiset change. The JSON retains H's campaign ID because the same runner emits it; its R package path, protocol and manifest establish the later execution provenance.

**Multiplicity:** natural trace arrays and F metrics independently show no duplicates. The existing synthetic R/`checks/all_cameras_correctness.json` and `checks/duplicate_cameras_correctness.json` cover ordered occurrence reconstruction and duplicate survivor multiplicity. That synthetic gate is preserved, not rerun here. Natural equality alone is not claimed to test duplicate behavior.

**Policy scope:** the natural image/attribute gate directly exercises Reload versus Previous, not a new Bounded RGB comparison on every frame. Existing bounded-cache validation remains in force; F checks the same semantic/materialized/survivor populations, configured capacity, and zero evictions. This audit does not relabel count equality as a fresh pixel-level test of each timed Bounded frame.

**Fresh checks in this task:** artifact/hash validation; actual asset identity; NPZ sizes/duplicate checks and byte identity; exact H/F config equality; camera order and resolution; all measured funnel fields; pooled timing recomputation; timing-boundary algebra; per-frame H2D/cache-count equivalence including warmup; balanced schedule; source hash agreement. The script returned `PASS-ALOD-CAMPUS-SCIART-PROVENANCE-CLOSURE`.

Semantic equivalence, execution-funnel equality, and timing comparability are thus separate conclusions: the first two pass for these assets/cameras; cross-state timing equivalence is neither established nor needed.

## 6. Canonical evidence recommendation

Use **F for the main final Campus/Sci-Art three-policy timing comparison** in both publications. It contains all three policies measured in the multiset-capable runtime and the final recorder, under the same configuration. Use **R plus the F raw cross-check for their unchanged funnel and correctness claims**.

Retain H's original figures as a clearly dated historical pre-multiset implementation result, in a short historical paragraph or appendix. It is valid evidence of the earlier extension and its association study, not an erroneous or incomplete benchmark. Do not mix H Reload with F Previous/Bounded or retain H's derived speedups beside F's times.

For final main-table derived values, calculate from unrounded F means (A):

| Dataset | Reload / Previous / Fixed ms | Fixed versus Reload reduction / speedup | Previous versus Reload reduction / speedup |
| --- | --- | --- | --- |
| Campus | 90.250 / 66.812 / 65.667 | 27.24% / 1.374× | 25.97% / 1.351× |
| Sci-Art | 521.086 / 378.909 / 366.420 | 29.68% / 1.422× | 27.28% / 1.375× |

Final FPS from mean time: Campus 11.08 / 14.97 / 15.23; Sci-Art 1.92 / 2.64 / 2.73. H/F dynamic-property transfer savings are unchanged because the per-frame byte counts match.

Preserve MatrixCity Run009 and synthetic-control findings and their own revision qualifications. Their rows are contextual workload results, not a pooled cross-dataset speed ranking or a common-runtime ablation.

## 7. Exact proposed publication edits (not applied)

No manuscript or manuscript PDF was edited or compiled in this task. Line numbers below refer to the audit-entry sources.

### Paper caveat 1: paper.tex, lines 413–420

Replace the whole paragraph with:

> For each workload, the three execution policies in the timing table were measured with one implementation state and one timing protocol. For Campus and Sci-Art, the post-multiset correctness regression uses the same runtime as the final policy comparison and reproduces the earlier funnel counts exactly. The final timing traces independently confirm the per-camera selected, materialized, and packed-survivor populations. MatrixCity timing comes from its own direct three-policy experiment.

Add source comments pointing to R/`checks/{campus,sciart}_regression.json`, R/`protocol_execution_002.json`, F/`protocol.json` and its amendment, F/`runs/`, and this closure note. “Same runtime” refers to the sealed runtime components, not a claim that the regression itself collected timing.

### Paper caveat 2: paper.tex, lines 643–645

Keep the preceding limitations about independently retrained natural hierarchies. Replace the last two sentences' timing/provenance clause with:

> Their three-policy timings are compared within each dataset and final implementation. The final-runtime correctness regression and policy traces support the reported funnel counts.

Both obsolete *earlier-only funnel provenance* caveats can therefore be removed. Do not remove the empirical scope, retraining/prevalence limitation, or cross-workload comparability qualifications.

### Additional timing clarification worth retaining in both publications

Add a concise methods sentence where end-to-end timing is defined:

> Reported frame time includes synchronized preparation and renderer wall time, including multiset bookkeeping and runtime metric computation.

If space permits, add:

> These values characterize the measured instrumented implementations; they do not isolate cache-policy cost or establish timing equivalence with earlier implementations.

This matters because Reload's unique-count metrics execute inside preparation while filtered paths reuse the unique arrays they already construct.

### Dissertation: targeted updates

Target [ALoDReproductionAndEvaluation.tex](../thesis_latex/chapters/ALoDReproductionAndEvaluation.tex), not a broad chapter rewrite:

1. **Funnel table `tab:alod-ext-cross-dataset-funnel` (around 2574–2599):** keep every number and counting convention. Update its evidence comments to R and this audit; state that the final-runtime regression reproduces the historical counts.
2. **Performance table `tab:alod-ext-cross-dataset-performance` (around 2611–2637):** make F's three-policy values above the main comparison. Add Previous explicitly. Use final FPS, reductions and speedups; preserve the existing H table/numbers in an explicitly labeled historical record rather than silently replacing them.
3. **Replacement interpretation for 2639–2645:**

   > In the final implementation, mean frame times for Reload, Previous, and Fixed cache are 90.250, 66.812, and 65.667 ms on Campus, and 521.086, 378.909, and 366.420 ms on Sci-Art. Fixed cache reduces mean frame time relative to Reload by 27.24% and 29.68%, respectively. These are direct comparisons within each dataset and implementation state; they do not establish performance equivalence with the earlier association campaign.

4. **Historical reconciliation paragraph, adjacent to the preserved H table or its appendix reference:**

   > The earlier association campaign measured Reload/Bounded frame times of 23.335/17.292 ms on Campus and 129.623/62.625 ms on Sci-Art. It used the same trained assets, ordered cameras, render settings, warmup schedule, and synchronized preparation-plus-renderer timing definition. The later campaign includes ordered-multiset support and additional metric computation within preparation. Its filtered paths directly record substantial unique-ID extraction cost, while the Reload path computes unique counts without a separate timer. The two campaigns therefore describe different implementations; the available records do not isolate every component of the timing increase. Their execution-funnel counts and dynamic-property transfer traces remain unchanged.

5. **Bottleneck discussion around 2781–2800:** qualify the existing 37.11%/44.46% selection shares as H only. For F Bounded, selection is 10.29% of Campus and 7.83% of Sci-Art frame time; the inclusive materialization/filter stage is 75.91% and 82.09%. Source: A's raw-derived means at the F revision. Suggested replacement:

   > Semantic selection occupied a larger fraction of the earlier filtered implementation. In the final natural runs, multiset/filter bookkeeping dominates instead: the inclusive materialization-filter stage accounts for 75.91% of Campus and 82.09% of Sci-Art bounded frame time, while semantic selection accounts for 10.29% and 7.83%. These implementation-dependent stage profiles support measuring the complete preparation path rather than inferring its bottleneck from Gaussian counts alone.

   Scope any following claim that selection is “the dominant stage” explicitly to Run009 or H. Do not alter Run009's measured selection cost.
6. **Chapter synthesis around 3050–3059:** replace the unqualified Campus 25.90% historical reduction with “27.24% in the final same-state comparison,” or explicitly label 25.90% as the earlier campaign. Replace the blanket natural selection-bottleneck sentence with the implementation-dependent interpretation above. Keep synthetic and Run009 conclusions intact.
7. Search subsequent conclusions/table references for the historical 25.90%, 51.69%, 1.349×, 2.070× and natural selection-dominance statements; qualify any retained H claims rather than presenting them as F. Update evidence comments to the correct campaign.

## 8. Remaining limitations and experiment decision

- The old runtime's complete source snapshot is unavailable in the inspected Git history/packages. Historical file hashes and result artifacts identify it, but do not reconstruct a complete executable checkout or full literal diff.
- Historical synchronization is documented and supported by raw timing algebra; it cannot be freshly verified line-by-line against an unavailable old helper snapshot. No evidence points to a shortened H timing boundary.
- The natural regression's original recorder hash differs from the later diagnostic recorder; all correctness-relevant sealed runtime hashes match. F directly supplies the final-recorder count check.
- Some dependency/HSPT/renderer source identities are not sealed by these natural protocols. Same versions, exact oracle/gate behavior, and raw-frame equality support the stated empirical scope, not universal source identity.
- The exact cost of Reload's unique-count metrics and the residual host/GPU latency changes remains unmeasured. Do not subtract guessed overhead, invent “corrected” frame times, or claim a controlled causal performance decomposition.
- Correctness is empirical for the frozen assets, cameras and packed backend. No new image test or training repeat was performed.
- The planned independently retrained prevalence repeat remains a separate research limitation. It is not necessary to close this provenance issue.

**No further experimental work is necessary for the requested provenance closure or the qualified final tables.** A separate profiling/ablation task would be required only to claim a complete causal decomposition or instrumentation-free timings; neither is needed or authorized here. Publication finalization requires the targeted textual/table reconciliation above.

Evidence update: this closure report and the ALoD topic index; registry unchanged; reproduction matrix unchanged; classification `ALOD-EXT` evidence reconciliation; canonical numerical artifacts H/R/F preserved, with CPU-only audit A.
