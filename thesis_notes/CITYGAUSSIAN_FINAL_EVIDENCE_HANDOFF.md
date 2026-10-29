# CityGaussian V1 Final Thesis-Evidence Handoff

This is the consolidated Layer-2 entry point for the completed CityGaussian V1
investigation. It is a writing-preparation record, not dissertation prose and
not a replacement for the canonical machine-readable artifacts. Before using
an exact number in the dissertation, re-open the linked artifact table or raw
record and follow the authority rules in
[`THESIS_WRITING_CONTEXT.md`](../THESIS_WRITING_CONTEXT.md).

The main protection this handoff provides is the chronological supersession of
performance interpretations. In particular, the July renderer-call throughput,
the historical official 19.04 FPS observation, the early 4--5x backend
attribution, and the sparse-replay contamination inference are not the final
performance conclusions.

## 1. Method and reproduction scope

Classification:

> Faithful reproduction of inspected released-code runtime selection semantics,
> with gsplat as a deliberate renderer-backend deviation.

This work reproduces the inspected CityGaussian V1 viewer/runtime behavior:

- official MatrixCity runtime-cell construction;
- projected-corner cell visibility;
- nearest-cell visibility fallback;
- released-code Nyquist adaptive LoD;
- one LoD level per visible cell;
- the author-provided three-level PLY representation;
- rendering through the project gsplat backend;
- separate parity execution through the released GraphDECO backend.

It does not reproduce:

- CityGaussian training;
- block training and fusion;
- LightGaussian pruning, distillation, or LoD generation;
- VQ/extreme-saving execution;
- the paper's fixed-distance LoD policy;
- the paper's cameras, hardware, quality evaluation, or headline benchmark
  protocol.

Do not describe this work as a complete CityGaussian paper reproduction.

## 2. Chronological implementation and evidence history

### 2.1 Paper, source, and asset audit

The paper and released source were audited separately. This identified the
released runtime's 6x6x1 MatrixCity cell organization, in-memory
`BlockedGaussian` construction, projected-corner visibility, nearest-cell
fallback, released Nyquist selector, paper/source LoD-policy divergence, and
whole-scene residency of all LoD models.

### 2.2 Component-wise reproduction

The implementation followed:

```text
paper/source/asset audit
-> independent oracle
-> project implementation
-> exact selected-set validation
-> short CUDA validation
-> controlled parent-child experiment
-> formal benchmark
```

Source-derived validation covered cell assignment and population, MAD-style
cell bounds, visibility-selected cell IDs, nearest-cell fallback, selected
level per visible cell, selected counts per cell and level, and total selected
workload.

The 12 frozen validation cameras were:

```text
[0, 131, 262, 392, 523, 654, 785, 916, 1047, 1177, 1308, 1439]
```

The project reproduction matched the source-derived oracles exactly. Later
official executable parity matched camera transforms and runtime selections
exactly.

Correctness principle:

> Semantic and selected-set equivalence is the primary correctness oracle.
> Matching FPS is not required for implementation correctness.

### 2.3 Initial formal benchmark and timing correction

The July campaign used `mean_frame_time_ms`, which measured the high-level
`GsplatRenderer.render(...)` call rather than complete frame processing.
Consequently, values such as approximately `127 FPS` for `CG-LOD-RELEASE` were
renderer-call throughput equivalents, not viewer or end-to-end FPS.

### 2.4 Canonical repeated gsplat campaign

The corrected campaign introduced:

```text
preparation_time_ms
render_call_time_ms
cuda_rasterization_time_ms
end_to_end_frame_time_ms
```

Each configuration used three fresh processes, one excluded frame-0 warmup,
and 1,440 measured frames.

### 2.5 Official executable parity

The released direct-PLY runtime was executed through the pinned GraphDECO
backend. Camera matrices, visible cells, LoD assignments, and selected
workloads matched the project reproduction exactly. Cross-backend images were
nearly identical.

### 2.6 Backend isolation

The initial high-level official-versus-gsplat gap suggested that the rasterizer
might dominate. Direct same-input isolation disproved that explanation: the raw
GraphDECO disadvantage was modest and camera-dependent, not 4--5x.

### 2.7 Twelve-camera attribution

Direct backend measurement was extended to all 12 oracle cameras. Sparse
replay showed that historical stalls at cameras 785 and 1047 disappeared when
those cameras were rendered independently or in the sparse sequence. This
established that static camera input and ordinary first-shape behavior were
insufficient explanations.

### 2.8 Final clean full-trajectory campaign

A final clean three-process, 1,440-frame official-runtime campaign retained the
complete sequential camera history. The large stalls returned at strongly
aligned trajectory regions in every process. This superseded the inference
that the earlier stalls were merely historical external contamination.

## 3. Canonical artifact register

| Artifact | Role | Dissertation authority |
| --- | --- | --- |
| [`citygs_canonical_20260824_135615`](../benchmark_artifacts/citygs_canonical_20260824_135615/README.md) | Canonical corrected gsplat campaign | Primary project-reproduction performance, controlled transitions, workload, and memory evidence |
| [`citygs_official_parity_20260824_191215`](../benchmark_artifacts/citygs_official_parity_20260824_191215/README.md) | Released executable parity | Primary official camera/selection parity and cross-backend image-similarity evidence |
| [`citygs_backend_microbench_20260824_213420`](../benchmark_artifacts/citygs_backend_microbench_20260824_213420/README.md) | Camera-392 backend isolation | Supersedes strong rasterizer-dominance attribution |
| [`citygs_12camera_attribution_20260824_222048`](../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/README.md) | Twelve-camera backend and sparse-runtime attribution | Primary camera-dependent isolated backend evidence; sparse replay is not full-trajectory FPS |
| [`citygs_official_clean_full_20260825_163319`](../benchmark_artifacts/citygs_official_clean_full_20260825_163319/README.md) | Final clean full official runtime | Representative official sequential-runtime performance and stall evidence |
| [`CITYGS-FORMAL-20260730`](../benchmark_artifacts/citygs_formal_bench_20260730_121148/manifest.json) | July precursor | Historical only; generic frame time was not end-to-end |
| `/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_132000` | Failed first corrected-campaign attempt | Preserve as failed evidence; no canonical aggregate |

## 4. Pinned source, assets, and hashes

### Source identities

```text
CityGaussian:
  path: /home/luis/tese/CityGaussian
  branch: V1-original
  commit: e16c526f78ab01d377b13b3e6862dcc93ddbe953

GraphDECO diff-gaussian-rasterization:
  commit: 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d

Canonical gs_framework campaign:
  commit: cfc2136459dd63cc47dc3acfde55a3d72a4d8bba
```

Reported compiled GraphDECO binary:

```text
/home/luis/tese/CityGaussian_official_parity/.official_extensions/
diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so

SHA-256:
cc2ada2c5875c272771ebe4624208c0a5f4004fc150cc9914d553274d02fffd8
```

The build contained native `sm_120` cubins, no PTX, and no CUDA debug build.
Exact simple-knn and GLM revisions should be read from the official parity/build
manifests before citation.

### Assets

| Asset | Population | SH | SHA-256 |
| --- | ---: | ---: | --- |
| `/home/luis/tese/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply` | 23,389,621 | 3 | `06e76f13e2805396713f587da05e991048be9ca019f63121e11772fc2a1f8357` |
| `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_50_vq/point_cloud.ply` | 11,694,810 | 2 | `70e0401c629f98f13bb55f8f6dc8579a4ba5dffe686c638b090565aabcd974bb` |
| `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_66_vq/point_cloud.ply` | 7,952,471 | 2 | `57904c2602a077f5593d236a0c72203b89456cf700060774c1184d307163c77a` |
| `/home/luis/tese/mc_aerial_c36/LoD/mc_aerial_c36_light_75_vq/point_cloud.ply` | 5,847,405 | 2 | `7142e855e4367b352269a1de9e6b7f3a1ab98909099f6ad2ba4b272489fd0c0d` |

Although the directory names contain `_vq`, these experiments loaded the PLY
files directly; no VQ decoding was exercised.

Frozen trajectory:

```text
/home/luis/tese/gs_framework/benchmarks/camera_paths/
citygaussian_23m_navigation_v1.json

SHA-256:
c20ea29cfacd8331a808eec08840cf814d908de7e7e2f8715763b89d5e1605b9
```

## 5. Released runtime behavior

For MatrixCity:

```text
block_dim = [6, 6, 1]
logical cells = 36
AABB = [-3.5, -4, -10, 4.5, 2, 10]
```

The inspected released runtime:

- loads whole-scene LoD models;
- contracts coordinates into the official normalized domain;
- groups Gaussians into uniform cells;
- constructs `BlockedGaussian` cells in memory;
- computes MAD-style robust bounds and an average activated-scale statistic;
- does not consume persistent runtime chunks or `lod-meta.json`.

Visibility:

- operates at cell granularity;
- projects eight MAD-bound corners;
- applies the released `z > 0.2` handling;
- tests the resulting screen-space rectangle;
- forces the nearest cell visible as fallback.

Released LoD:

- orders levels finest, middle, coarsest;
- uses coarsest-level bounds for visibility;
- derives focal length from width and FoVx;
- uses minimum camera-space Euclidean distance to the eight corners;
- selects the first level satisfying the strict Nyquist comparison;
- falls back to coarsest;
- selects one level per visible cell;
- keeps all three models resident.

## 6. Paper/source divergence

The paper describes MatrixCity distance intervals:

```text
0--200
200--400
400+
```

The inspected V1 source uses a Nyquist-style selector instead. The canonical
reproduction implements the released-source policy. The paper policy remains
unresolved under the current scene and camera scale and was not formally
evaluated. This is a substantive reproducibility finding.

## 7. Controlled experiment ladder

Finest SH2 branch:

```text
CG-FINEST-FLAT
-> CG-FIXED-FINEST-ALL
-> CG-FIXED-FINEST-VISIBILITY
-> CG-LOD-RELEASE
```

- Flat to all cells adds official in-memory cell organization and per-frame
  active-set assembly without changing the asset.
- All cells to visibility adds released visibility selection on the same
  finest asset and cell structure.
- Visibility to LoD adds the middle/coarsest representations and released
  per-cell Nyquist selection.

Fused SH3 branch:

```text
FLAT-CITYGS-FULL
-> CG-FUSED-CELLS-ALL
-> CG-FUSED-VISIBILITY
```

- Flat to all cells uses the same fused asset and adds cell organization and
  assembly.
- All cells to visibility keeps the same asset/cells and adds visibility.

Never treat `FLAT-CITYGS-FULL -> CG-LOD-RELEASE` as an isolated transition.
Representation, SH degree, Gaussian count, visibility, LoD, and residency all
change.

## 8. Final validated gsplat findings

Mean end-to-end time plus or minus sample standard deviation of run means:

| Configuration | End-to-end | FPS |
| --- | ---: | ---: |
| `CG-FINEST-FLAT` | 11.360 +/- 0.622 ms | 88.03 |
| `CG-FIXED-FINEST-ALL` | 21.733 +/- 1.266 ms | 46.01 |
| `CG-FIXED-FINEST-VISIBILITY` | 19.049 +/- 1.227 ms | 52.50 |
| `CG-LOD-RELEASE` | 15.673 +/- 0.738 ms | 63.80 |
| `FLAT-CITYGS-FULL` | 18.502 +/- 0.816 ms | 54.05 |
| `CG-FUSED-CELLS-ALL` | 45.726 +/- 2.021 ms | 21.87 |
| `CG-FUSED-VISIBILITY` | 62.928 +/- 5.722 ms | 15.89 |

### Controlled findings

Cell organization:

- Finest flat to cells: end-to-end `+91.32%`; render call only `+0.39%`.
- Fused flat to cells: end-to-end `+147.15%`; render call only `+0.08%`.

Supported conclusion:

> Same-asset cell organization has negligible renderer-call overhead, but
> dynamic active-set assembly introduces substantial preparation cost in the
> current gsplat integration.

Visibility on finest SH2:

- selected workload: `-49.22%`;
- end-to-end time: `-12.35%`;
- render call: `-6.41%`.

Released LoD relative to finest visibility:

- selected workload: `-29.04%`;
- end-to-end: `-17.72%`;
- render call: `-29.22%`;
- CUDA: `-29.51%`;
- render-peak allocated memory: `+0.893 GiB`;
- post-transfer allocated memory: `+2.161 GiB`.

Supported conclusion:

> Released Nyquist LoD reduces active workload, render time, and end-to-end
> time within the structured branch, while increasing residency because all
> levels remain loaded.

Fused visibility negative result:

- selected workload: `-49.41%`;
- end-to-end time: `+37.62%`;
- render call: `+102.12%`;
- radii-surviving population remains almost unchanged.

No definitive SH, ordering, tiling, or sorting explanation is established.

### Workload and memory terminology

Keep distinct:

- resident Gaussians;
- selected Gaussians;
- active Gaussians;
- radii-derived visible or surviving Gaussians.

For `CG-LOD-RELEASE`:

```text
resident: 25,494,686
mean selected/active: 4,214,145.394
```

This runtime is not streaming or out-of-core.

## 9. Official executable parity

Parity findings:

- direct three-PLY runtime executed successfully;
- camera matrices matched exactly;
- all 12 selected-set checks passed;
- visible cells matched exactly;
- LoD assignments matched exactly;
- per-cell, per-level, and total selected counts matched;
- resident and mean selected populations matched.

Cross-backend image similarity:

```text
mean RGB absolute difference approximately 1.93e-5
mean PSNR approximately 75.30 dB
mean SSIM approximately 0.9999968
mean coverage Jaccard approximately 0.999999959
```

These are implementation-similarity metrics, not ground-truth reconstruction
metrics. The official comparison strengthens the reproduction claim:
performance disagreement does not imply semantic or rendering disagreement.

## 10. Backend and sequential-runtime attribution

### Direct camera-392 backend isolation

Representative LoD input:

```text
GraphDECO: 8.989 +/- 0.229 ms
gsplat:    6.652 +/- 0.021 ms
ratio:     1.351x
```

Finest-flat input:

```text
GraphDECO: 10.769 +/- 0.052 ms
gsplat:     9.738 +/- 0.267 ms
ratio:      1.106x
```

Inputs and outputs matched. This supersedes the claim that the 4--5x
high-level gap was primarily raw rasterizer throughput.

### Twelve-camera direct attribution

GraphDECO/gsplat direct GPU-time ratio:

```text
mean:    1.401x
median:  1.380x
minimum: 1.088x
maximum: 2.039x
```

Ten of twelve cameras were at or below 1.5x. Camera 1047 reached 2.039x.

Supported conclusion:

> gsplat has a real camera-dependent isolated backend advantage, but it is
> insufficient to explain the full sequential-runtime gap.

Sparse official replay:

```text
official: 25.717 ms, 38.89 s^-1 equivalent
gsplat:   15.597 ms, 64.11 s^-1 equivalent
ratio:    1.649x
```

These are sparse-camera throughput equivalents, not full-trajectory FPS.

### Wrapper and first-shape probes

- screen-space tensor initialization: `<0.4 ms`;
- activated attribute views: approximately `0.005--0.007 ms`;
- rasterizer argument/object construction: `<0.2 ms`;
- first target-shaped calls: approximately `26--32 ms`;
- warmed calls: approximately `12--16 ms`.

These costs do not explain the historical 350--380 ms stalls.

### Final clean official full trajectory

Protocol:

- three fresh processes;
- no avoidable video, game, or competing compute workload;
- one excluded frame-0 warmup;
- synchronization and memory-peak reset;
- frames 0--1439 in unchanged order;
- 1920x1080;
- direct PLY runtime;
- no optimization.

Result:

```text
end-to-end:                  74.778 +/- 15.459 ms
FPS:                         13.37
selection:                    1.487 +/- 0.035 ms
active-set assembly:          9.179 +/- 0.849 ms
render call:                 64.111 +/- 14.646 ms
CUDA rasterization boundary: 63.938 +/- 14.617 ms
between-run CV:              20.67%
run means:                   78.689, 87.905, 57.738 ms
```

Stall evidence:

- 95 frames exceeded 250 ms in all three processes;
- pairwise frame-time correlations were approximately 0.967--0.978;
- recurring ranges were:

```text
783--801
809--815
836--859
895
975--977
979--1008
1045--1055
```

Cameras 785 and 1047 had several-hundred-millisecond render/CUDA boundaries in
all three processes.

Established:

- stalls are inside the synchronized GraphDECO rasterization boundary;
- they recur at nearly identical trajectory positions;
- selected count has little direct linear association;
- first workload signatures explain only a minority;
- sparse rendering does not reproduce them;
- clean external conditions do not eliminate them;
- selection is too small to explain them;
- assembly contributes but does not explain the largest events;
- reservation transitions and near-capacity usage are associated with the
  regions.

Not established:

- allocator causality;
- memory fragmentation;
- workspace-growth causality;
- cache or sorting-state causality;
- the precise low-level state that produces sequence dependence.

Required wording:

> Large trajectory-state-dependent stalls occur inside the official
> synchronized rasterization boundary; their low-level cause remains
> unresolved.

### Final official-versus-gsplat comparison

| Metric | Official clean full | Canonical gsplat | Ratio |
| --- | ---: | ---: | ---: |
| End-to-end | 74.778 +/- 15.459 ms | 15.673 +/- 0.738 ms | 4.771x |
| FPS | 13.37 | 63.80 | 4.771x gsplat |
| Selection+assembly / preparation | 10.667 ms | 8.126 ms | 1.313x |
| Render call | 64.111 ms | 7.547 ms | 8.495x |
| CUDA boundary | 63.938 ms | 7.480 ms | 8.547x |

Both use:

```text
resident Gaussians: 25,494,686
mean selected Gaussians: 4,214,145.394
```

Final attribution:

> The sequential gap cannot be reduced to a static rasterizer-speed
> difference. Direct isolated measurements show a camera-dependent 1.09--2.04x
> gsplat advantage, while the released full runtime additionally develops
> reproducible sequence-dependent stalls within its synchronized rasterization
> boundary.

## 11. Superseded, failed, and historical evidence

### Supersession chain

1. July approximately 127 FPS:
   - renderer-call throughput equivalent only;
   - never use as end-to-end FPS;
   - superseded by corrected canonical timing.
2. Initial canonical attempt:
   - `/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_132000`;
   - completed 21 raw runs but failed aggregation because a flat-scene selected
     count was non-finite;
   - preserve as failed evidence only.
3. Corrected canonical gsplat campaign:
   - `63.80 FPS` for `CG-LOD-RELEASE`;
   - current representative project-reproduction result.
4. Historical official parity result:
   - `52.516 +/- 3.337 ms`, `19.04 FPS`;
   - retain as a historical observation and semantic-parity artifact;
   - do not use as representative official performance.
5. Initial high-level attribution:
   - suggested a dominant GraphDECO backend gap;
   - superseded by same-input direct backend isolation.
6. Sparse clean replay:
   - large camera-785/1047 stalls did not reproduce;
   - valid only for sparse replay;
   - its contamination inference is superseded by the full sequential campaign.
7. Final official performance:
   - `74.778 +/- 15.459 ms`, `13.37 FPS`;
   - representative official sequential-runtime result under the project
     protocol;
   - historical `19.04 FPS` is superseded only for representative performance,
     not semantic parity.

## 12. Reproducibility assessment

### Implementation reproducibility: strong

- source identity is pinned;
- paper and source policies are distinguished;
- components have independent oracles;
- selected-set parity is exact;
- official executable parity passes;
- cross-backend images are nearly identical.

### Experimental reproducibility: strong for project campaigns, partial versus paper

Strengths:

- frozen trajectory and resolution;
- asset/config hashes;
- clean source commit for the canonical campaign;
- fresh-process repetitions;
- explicit warmup and timing boundaries;
- isolated official environment;
- recorded clean-system full trajectory.

Threats:

- different PyTorch/runtime environment from the authors;
- different renderer in the primary reproduction;
- RTX 5080 rather than author hardware;
- WSL memory-accounting caveats;
- project cameras rather than paper cameras;
- direct PLY rather than VQ execution.

### Result reproducibility: partial

Reproduced:

- runtime cell and visibility semantics;
- released Nyquist LoD decisions;
- selected populations;
- camera conversion;
- appearance under equivalent cameras.

Not reproduced:

- paper FPS;
- paper hardware;
- paper cameras;
- paper fixed-distance LoD;
- paper PSNR/SSIM/LPIPS;
- training and model-generation results.

Recommended method-level matrix status:

- implementation reproducibility: strong;
- experimental reproducibility: partial;
- result reproducibility: partial.

Update `REPRODUCTION_MATRIX.md` only if its current method-level status differs.

## 13. Limitations and threats to validity

- The primary reproduction deliberately uses gsplat instead of GraphDECO.
- The official comparison uses the project camera path and protocol.
- Paper fixed-distance LoD was not evaluated.
- No ground-truth MatrixCity quality campaign exists.
- Cross-backend PSNR/SSIM is implementation similarity only.
- Positive-radii counts are not definitionally comparable across rasterizers.
- Memory counters are not representation-neutral.
- PyTorch reserved bytes can exceed physical VRAM under WSL/custom-extension
  accounting.
- Whole-device after-run memory is not a continuously sampled peak.
- All three LoD models remain resident.
- No VQ or LightGaussian runtime path was evaluated.
- The exact official sequential-stall mechanism remains unresolved.
- No formal interactive event-loop, temporal-popping, or moving-preview quality
  campaign was performed.

## 14. Claims safe to write

After tracing exact numbers to their artifacts:

- The inspected released CityGaussian V1 runtime behavior was reconstructed and
  independently validated.
- Runtime-cell construction, visibility, and released Nyquist LoD match
  source-derived oracles.
- The project implementation matches official executable selected sets exactly.
- Cross-backend outputs are nearly identical under the frozen cameras.
- Dynamic cell assembly introduces substantial preparation cost in the current
  gsplat integration.
- Same-asset cell organization has negligible renderer-call overhead, not
  negligible end-to-end overhead.
- Visibility reduces selected workload, but its end-to-end benefit is smaller
  than its population reduction.
- Released adaptive LoD improves render and end-to-end performance relative to
  fixed-finest visibility.
- Released LoD increases residency because all three levels remain loaded.
- The fused visibility result is a valid negative result.
- The isolated backend difference is camera-dependent and much smaller than the
  sequential gap.
- The official full sequential runtime develops reproducible
  trajectory-state-dependent stalls.
- The largest stalls occur inside its synchronized rasterization boundary.
- Their exact low-level cause remains unresolved.
- Equivalent semantics and nearly identical images do not guarantee equivalent
  performance.

## 15. Claims requiring qualification

- `13.37 FPS` applies only to the frozen 1,440-frame, 1920x1080, RTX 5080,
  direct-PLY project protocol.
- `63.80 FPS` is the project gsplat reproduction under the project protocol,
  not a paper result.
- Cross-backend PSNR and SSIM measure implementation similarity, not
  ground-truth quality.
- Memory comparisons span different implementations and accounting boundaries.
- The paper and released source use different LoD policies.
- The official-versus-gsplat difference is multi-factor and not attributable to
  one proven cause.
- Clean-system evidence excludes observed competing processes and avoidable
  media workloads; it does not prove the absence of every OS/driver influence.
- Reservation-transition association is not allocator causality.

## 16. Claims prohibited by the evidence

Do not write:

- `CG-LOD-RELEASE runs at 127 FPS end-to-end.`
- `Cell organization has negligible overall runtime overhead.`
- `GraphDECO is 4--5x slower than gsplat.`
- `The official viewer is slow because of SH3.`
- `Memory fragmentation causes the official stalls.`
- `Twitch caused the historical stalls.`
- `The stalls are generic high-level wrapper overhead.`
- `GraphDECO and gsplat positive-radii counts are directly comparable.`
- `The CityGaussian paper's FPS was reproduced.`
- `The project reproduced the full CityGaussian training pipeline.`
- `The reproduced CityGaussian runtime is streaming or out-of-core.`
- `Parent/child or cross-backend PSNR is ground-truth reconstruction quality.`
- `Sparse oracle-camera throughput is full-trajectory FPS.`
- `Performance disagreement proves a reproduction error.`

## 17. Optional remaining experiments

None is required before writing the CityGaussian reproduction and evaluation
chapter.

Optional extensions:

- fixed ground-truth MatrixCity PSNR/SSIM/LPIPS;
- temporal LoD transition and popping analysis;
- paper fixed-distance LoD after resolving scale semantics;
- VQ runtime and storage evaluation;
- engineering optimization of active-set assembly;
- deeper low-level investigation of the official sequential stalls;
- another SOTA hierarchy or residency method;
- formal interactive-viewer behavior evaluation.

## 18. Recommended thesis narrative

1. Introduce CityGaussian as the first large-scale runtime case study.
2. Separate the paper audit from the released-source audit.
3. Decompose runtime cells, visibility, and LoD into independently verifiable
   components.
4. Establish source-derived oracles before implementation.
5. Reproduce the released selector exactly through the project gsplat backend.
6. Evaluate same-asset cell organization and identify active-set assembly as an
   integration cost.
7. Show that visibility reduces workload without proportionate end-to-end
   improvement.
8. Show that released Nyquist LoD gives the strongest structured-branch
   improvement while increasing residency.
9. Preserve the fused visibility regression as a negative result.
10. Use the released executable as an external semantic oracle and demonstrate
    exact selection parity.
11. Explain that nearly identical semantics and images coexist with very
    different performance.
12. Use direct backend isolation to reject the initial 4--5x raw-rasterizer
    explanation.
13. Use the 12-camera study to establish camera-dependent backend variation and
    the insufficiency of sparse replay.
14. Use the final clean full trajectory to demonstrate reproducible
    sequence-dependent official-runtime stalls.
15. Report the stall mechanism as unresolved rather than speculating about
    allocator, sorting, or cache causality.
16. Frame the contribution as a component-wise reproduction and systems
    evaluation methodology separating semantic fidelity, integration cost,
    renderer behavior, residency, and result reproducibility.

## 19. Repository-aware writing checklist

1. Read [`THESIS_WRITING_CONTEXT.md`](../THESIS_WRITING_CONTEXT.md).
2. Open [`topics/citygaussian.md`](topics/citygaussian.md).
3. Verify the experiment rows in
   [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md).
4. Check [`REPRODUCTION_MATRIX.md`](REPRODUCTION_MATRIX.md).
5. Re-open the relevant artifact's manifest, validation gate, aggregate table,
   and raw frames for every exact numerical claim.
6. Verify commits, asset hashes, trajectory hash, and source cleanliness.
7. Preserve the supersession chain in Section 11.
8. Do not update the reproduction matrix unless its method-level status differs
   from strong/partial/partial.
9. Do not copy complete metric tables into other Layer-2 notes when canonical
   tables can be linked.
10. Do not edit dissertation prose until each exact number traces to its
    canonical evidence.
