# ALoD-EXT natural-hierarchy association result

Campaign: `ALOD-EXT-PREVALENCE-ASSOCIATION-001`

Status: `PASS-ALOD-EXT-PREVALENCE-ASSOCIATION`

Classification: project-specific `ALOD-EXT`; not released ALoD and not the
paper's `Render Full`.

## Protocol status

The protocol and bounded-capacity rule were frozen before any new ALoD-EXT
result. The separate Stage-B repeat choice was also frozen prospectively as
Sci-Art TRAIN/ORACLE-002 for the weak repeat and Campus TRAIN/ORACLE-002 for
the nontrivial repeat. The randomness audit and both repeats remain pending;
no Stage-B training or synthetic control was started.

The new immutable executions are:

- `ALOD-EXT-PREVALENCE-CAMPUS-001`: all 24 frozen Campus held-out cameras,
  `VALID-NONTRIVIAL-LOD`;
- `ALOD-EXT-PREVALENCE-SCIART-001`: all 21 frozen Sci-Art held-out cameras,
  `VALID-WEAK-LOD`.

Each mode ran in three fresh processes. Each process rendered the first ten
canonical cameras as ordered warmup, retained reuse/cache state, then restarted
the full canonical sequence at camera zero for measurement. End-to-end time is
CUDA-synchronized around selection, filtering, preparation/residency,
assembly, and rasterization. The balanced cyclic mode schedule is recorded in
the package.

The frozen modes were Released Evaluator (No Cache), ALoD-EXT CUDA Previous
Reuse, and ALoD-EXT CUDA Bounded Cache. The capacity rule was
`min(H, ceil(U * 3700000 / 3503048))`, transferring Run009's exact practical
union headroom without FPS tuning. This yielded 621,377 rows for the 588,300-row
Campus filtered union and 2,666,887 rows for the 2,524,927-row Sci-Art union.

The initial generic Sci-Art camera path exposed a selector-camera precision
incompatibility before timing: it reconstructed values from the public camera
record rather than forwarding the exact Stage-A `camera_center_float32` and
`full_projection_transform`. The prospective amendment forwards those already
frozen values through existing runtime fields for both datasets. It changes no
pose, intrinsics, resolution, camera order, selector, filter, cache, or
workload. A first Campus launch also encountered the known system CUDA 12.0
compiler incompatibility before a result; the established Conda CUDA 12.9
toolchain resolved it. Neither event is a scientific failure.

Execution hashes sealed before the successful rerun were:

| Item | SHA-256 |
| --- | --- |
| Frozen config | `7ea720ae072812cbde2ffbc03ed32dc140af795e1649eae58f1859f862c6fb5e` |
| Benchmark runner at execution | `5e2fed852d3d3f1043759de1f28d43de9911dd4554f1cc5aeef8b74e9f7fa350` |
| Shared real-scene benchmark at execution | `f83670e2c3f97b5932009af8412766b12675dfdc509947379c39ca38f9190873` |
| Prospective protocol | `37c05d8f6a785381959b67ed588d9048fa81a58a1465254dd14a3de762e29595` |
| Camera-compatibility amendment | `4e40f31e28caffac8ef8e2f4f75fbb9edc0d9163d7d7a4c82ce360968909b72c` |

After execution, the exact-camera forwarding was made self-contained in the
association runner and the shared benchmark helper was restored byte-for-byte
to its pre-study SHA-256
`f82b618ec13c2003e8ca8143fa60c929351278dc78cd919c9dc57c430638678d`.
The final reusable runner SHA-256 is
`9c132e919c486d040e1de32dffa75fa693e9a0d78445e2245365e6dd828652cf`.
This packaging refactor is functionally identical to the sealed execution
path; it did not regenerate measured frames.

## Correctness

| Dataset | Cameras | Official IDs/order | False negatives | CPU/CUDA filtered IDs/order | Properties/order | Packed survivors | RGB/alpha |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| Campus | 24/24 | exact | 0 | exact | bitwise exact | exact | bitwise exact |
| Sci-Art | 21/21 | exact | 0 | exact | bitwise exact | exact | bitwise exact |

Both datasets have status `PASS-ALOD-EXT-PREVALENCE-CORRECTNESS`. Property
parity covers means, scales, quaternions, opacities, colors, global node IDs,
and retained order. No historical Run009 correctness barrier was weakened.

## Workload funnel

Counts are canonical per-camera means. Ratios are pooled sums over the complete
trajectory and include the semantic raster rows supplied to the renderer,
including surviving skybox rows where applicable.

| Dataset / Stage-A label | Semantic selected | Materialized | Packed survivors | Materialized / semantic | Survivors / semantic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Campus / `VALID-NONTRIVIAL-LOD` | 530,166.46 | 158,787.33 | 146,633.21 | 29.9505% | 27.6580% |
| Sci-Art / `VALID-WEAK-LOD` | 2,548,681.90 | 767,118.05 | 732,242.76 | 30.0986% | 28.7303% |
| Run009 / historical `VALID-WEAK-LOD` | 3,532,140.96 | 525,049.24 | 496,181.51 | 14.8649% | 14.0476% |

Thus the tested nontrivial Campus hierarchy does not have a meaningfully
smaller semantic-to-materialization ratio than weak Sci-Art: it is lower by
only 0.148 percentage points. Its backend-survivor ratio is 1.072 percentage
points lower. Both new natural workloads have substantially larger ratios than
historical weak Run009. These three descriptive points do not support a
systematic or causal weak-versus-nontrivial relationship.

## Performance

Means pool all measured cameras and three fresh processes. `H2D steady / cold`
is dynamic Gaussian-property traffic in bytes, pooled across those processes;
cold includes the recorded ten-frame warmup. It excludes static skybox upload,
compact selector metadata, selected-ID transfers, retained-index D2H, and
device-to-device copies, so it is not whole-bus traffic. Selection share is
selection time divided by synchronized end-to-end time.

| Dataset | Mode | Total ms | FPS | Select ms / share | Filter ms | Prepare ms | Host gather ms | H2D ms | Raster ms | H2D steady / cold bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Campus | Released no-cache | 23.335 | 42.85 | 6.434 / 27.57% | 0.000 | 19.885 | 8.101 | 2.794 | 3.333 | 2,849,422,620 / 4,035,599,964 |
| Campus | EXT previous | 19.067 | 52.45 | 6.444 / 33.80% | 2.443 | 15.635 | 2.273 | 1.386 | 3.318 | 703,375,512 / 1,009,831,560 |
| Campus | EXT bounded | 17.292 | 57.83 | 6.417 / 37.11% | 2.470 | 14.063 | 0.214 | 0.303 | 3.120 | 22,038,876 / 162,370,800 |
| Sci-Art | Released no-cache | 129.623 | 7.71 | 28.143 / 21.71% | 0.000 | 124.037 | 61.653 | 13.692 | 5.426 | 14,192,560,320 / 20,921,084,832 |
| Sci-Art | EXT previous | 74.009 | 13.51 | 27.916 / 37.72% | 4.992 | 69.345 | 13.615 | 5.356 | 4.524 | 2,741,495,856 / 3,867,272,256 |
| Sci-Art | EXT bounded | 62.625 | 15.97 | 27.840 / 44.46% | 5.311 | 57.485 | 0.451 | 0.432 | 5.000 | 47,909,184 / 696,879,852 |

Relative to Released Evaluator (No Cache):

- Campus previous reuse reduces mean frame time by 18.29% (1.224x) and
  bounded cache by 25.90% (1.349x). Bounded steady dynamic-property H2D falls
  99.23%, and cold-inclusive traffic falls 95.98%.
- Sci-Art previous reuse reduces mean frame time by 42.90% (1.751x) and
  bounded cache by 51.69% (2.070x). Bounded steady dynamic-property H2D falls
  99.66%, and cold-inclusive traffic falls 96.67%.
- The union-covering bounded capacities produce zero evictions, as intended by
  the prospectively frozen capacity rule. They evaluate reuse under the
  canonical trajectory, not cache pressure or an FPS-optimal capacity.
- Once property movement is reduced, semantic selection becomes a larger
  residual share: 37.11% on Campus and 44.46% on Sci-Art in bounded mode.

Historical Run009 used the same packed backend, ten warmups, three fresh
processes, and synchronized end-to-end boundary, but a different asset and
71-camera workload. Its Released Evaluator, previous, and bounded means were
186.774, 71.657, and 73.480 ms. The 60.66% bounded reduction and 14.8649%
materialization ratio are context, not pooled evidence.

## Association interpretation

ALoD-EXT remains materially beneficial on the naturally nontrivial Campus
hierarchy: the frozen practical bounded path cuts mean frame time by 25.90%,
raises throughput from 42.85 to 57.83 FPS, and removes 99.23% of measured
steady dynamic-property H2D traffic. The benefit is smaller than on weak
Sci-Art, where frame time falls 51.69%, but the semantic-to-materialization
opportunity is almost identical (29.95% versus 30.10%). Historical weak
Run009 has a much larger gap (14.86%). On these exact hierarchies, the frozen
Stage-A weak/nontrivial label is therefore not sufficient to predict the
materialization ratio or runtime gain.

This is descriptive evidence about three trained hierarchies and their frozen
cameras. It does not show that weak LoD causes greater speedup, that nontrivial
LoD causes smaller speedup, that ALoD-EXT changes semantic quality, or that the
result applies to released/paper `Render Full`.

## Supervisor-ready summary

- Stage A remains unchanged: Campus is the sole natural
  `VALID-NONTRIVIAL-LOD` hierarchy; Sci-Art and Run009 are `VALID-WEAK-LOD`.
- Campus provides the requested natural contrast because it passed all three
  prospective cut-variation predicates, without implying monotonic scale-depth
  behavior.
- Every Campus and Sci-Art camera passed exact semantic, filter, property,
  packed-survivor, and bitwise image correctness before timing.
- ALoD-EXT still helps Campus: bounded execution is 25.90% faster than the
  released no-cache evaluator and reaches 57.83 FPS.
- The benefit is larger on Sci-Art (51.69%, 15.97 versus 7.71 FPS), but this is
  a descriptive dataset difference, not a causal LoD-class effect.
- Campus and Sci-Art have nearly equal materialized/semantic ratios (29.95%
  and 30.10%); historical Run009 is much lower at 14.86%.
- Bounded reuse removes more than 99% of steady dynamic-property H2D traffic
  on both new workloads; selection then occupies 37.11% of Campus and 44.46%
  of Sci-Art end-to-end time.
- Stage-B randomness audit and Sci-Art/Campus repeats remain required; this
  execution study does not replace reproducibility evidence.
- A synthetic strong-LoD control is no longer necessary for this primary
  natural descriptive contrast. It remains potentially useful later for a
  controlled mechanism/counterfactual study, after Stage B, but is lower
  priority.

## Canonical evidence

- Prospective protocol: `thesis_notes/ALOD_EXT_PREVALENCE_ASSOCIATION_PROTOCOL.md`.
- Stage-B freeze: `thesis_notes/ALOD_LOD_PREVALENCE_STAGE_B_REPEAT_FREEZE.md`.
- Machine manifest and validation: `benchmark_artifacts/alod_ext_prevalence_association_001/manifest.json` and
  `validation/package_validation.json`.
- Machine protocol and amendment:
  `benchmark_artifacts/alod_ext_prevalence_association_001/protocol.json` and
  `protocol_amendment_001.json`.
- Correctness and capacity:
  `checks/campus_correctness.json` and `checks/sciart_correctness.json`.
- Fresh-process schedules: `checks/campus_schedule.json` and
  `checks/sciart_schedule.json`.
- Canonical aggregate: `tables/association_summary.json`.
- Per-camera correctness traces: `traces/campus_trace.npz` and
  `traces/sciart_trace.npz`.
- Raw benchmark records:
  `runs/ALOD-EXT-PREVALENCE-{CAMPUS,SCIART}-001/{nocache,previous,bounded}/replicate-*`.
