# ALoD visual evidence package

Package ID: `ALOD-VISUAL-EVIDENCE-001`

Status: complete (`PASS-ALOD-VISUAL-EVIDENCE`)

## Scope

This package contains explanatory visualizations derived from the already-frozen
Campus and Sci-Art Stage-A experiments and the
`ALOD-EXT-PREVALENCE-ASSOCIATION-001` correctness campaign. It is not a new
scientific outcome, a new camera-selection study, a performance rerun, or a
change to selection, hierarchy, threshold, cache, or materialization semantics.

The complete machine-readable record is
[`manifest.json`](../../benchmark_artifacts/alod_visual_evidence_001/manifest.json).
The package contains 47 PNG files: individual RGB, alpha, difference, and depth
diagnostics; one summary panel per dataset; a within-project RGB parity panel;
a direct official-source comparison panel; and a maximum-depth-TVD comparison.

## Deterministic camera selection

Representative cameras were chosen from each frozen held-out workload using
only the Stage-A selected hierarchy-row counts:

1. minimum selected count;
2. camera whose count has minimum absolute distance to the frozen numerical
   median;
3. maximum selected count.

All ties use the lowest canonical evaluation-sequence index. Campus has 24
cameras, and its frozen median count (434,585) is midway between cameras 1 and
2; both are 3,177 rows away, so the tie-break selects camera 1. Visual
appearance was not used.

| Dataset / role | Canonical camera ID | Hierarchy rows | Semantic raster rows | Materialized rows | Packed survivors | Materialized / semantic | Mean / median / max depth |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Campus / minimum | `eval-0011-img-012919-cam-0001` | 350,849 | 450,849 | 26,202 | 18,617 | 5.8117% | 19.7181 / 19 / 44 |
| Campus / median | `eval-0001-img-002148-cam-0001` | 437,762 | 537,762 | 98,280 | 90,605 | 18.2757% | 20.2469 / 20 / 44 |
| Campus / maximum | `eval-0017-img-019381-cam-0001` | 460,527 | 560,527 | 134,338 | 120,981 | 23.9664% | 20.1548 / 19 / 44 |
| Sci-Art / minimum | `eval-0004-img-000009-cam-0001` | 2,413,401 | 2,513,401 | 167,599 | 153,692 | 6.6682% | 21.8577 / 22 / 48 |
| Sci-Art / median | `eval-0011-img-000012-cam-0001` | 2,454,985 | 2,554,985 | 338,985 | 296,337 | 13.2676% | 21.8726 / 22 / 48 |
| Sci-Art / maximum | `eval-0019-img-000020-cam-0001` | 2,472,825 | 2,572,825 | 1,506,034 | 1,470,416 | 58.5362% | 21.8779 / 22 / 48 |

“Hierarchy rows” is the frozen Stage-A semantic cut, excluding the 100,000
skybox rows. “Semantic raster rows” includes those skybox rows and is the
denominator used for the materialized/semantic ratio, matching the association
campaign. “Packed survivors” means packed-projection survivors, not final pixel
contributors.

## Assets, cameras, and renderer

Campus uses hierarchy asset SHA-256
`8743b7d34f4c61b638f5e6528539b8dca021b9752ddd88d076a66608641815c4`,
the 24-camera frozen public held-out workload, and 1436x1075 rendering. Sci-Art
uses hierarchy asset SHA-256
`acc49a863e022a6871391a224dfa01e2de24133a168c3e4498c2c8b83afa3010`,
the 21-camera frozen public validation workload, and 1216x912 rendering. Full
asset paths, frozen camera-manifest hashes, semantic-manifest hashes, Stage-A
response hashes, and correctness-record hashes are in the package manifest.

Both outputs use `GsplatRenderer` and the association campaign's exact official
float32 selector-camera forwarding. Released images use
`ALOD-REPRO-HOST-NOCACHE`. Extension images use
`ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL` with the frozen union-covering
bounded capacities: 621,377 Campus hierarchy rows and 2,666,887 Sci-Art
hierarchy rows. Each export replays the campaign's first ten canonical warmup
cameras and then restarts the canonical sequence at camera zero without
resetting residency state. No timing result was collected for this package.

## RGB and alpha equality

For all six selected cameras, Released Evaluator and bounded ALoD-EXT native
float RGB tensors were bitwise equal: maximum absolute difference 0 and
nonzero element count 0. Alpha tensors were also bitwise equal with the same
zero results. Equality was checked before conversion to lossless 8-bit PNG;
the conversion rule is recorded in the manifest. Each absolute-difference PNG
is therefore an unamplified black image annotated
`bitwise identical — max |Δ| = 0`.

The complete all-camera correctness claim remains that of
`ALOD-EXT-PREVALENCE-ASSOCIATION-001`: 24/24 Campus and 21/21 Sci-Art cameras
passed exact RGB/alpha parity. This visualization package rechecks only its six
deterministically selected exports and does not replace that campaign.

## Direct pinned-official comparison

The frozen official oracle already contains the native float32 RGB tensor
emitted inside the pinned official evaluator for every held-out camera. The
exporter validates both the payload-file SHA-256 and raw C-order tensor SHA-256,
then compares that tensor directly with bounded ALoD-EXT before either image is
converted to PNG. This comparison required no official rerun.

The cross-execution tensors are not bitwise equal. Their mean absolute errors
are around `10^-7`, while maximum errors reach approximately `0.0021` to
`0.0035`:

| Dataset / role | Camera index | Maximum absolute difference | Mean absolute difference | Nonzero RGB elements |
| --- | ---: | ---: | ---: | ---: |
| Campus / minimum | 11 | 0.0031881034 | 4.4784851e-08 | 2,453,744 |
| Campus / median | 1 | 0.0020769238 | 1.0499578e-07 | 3,121,828 |
| Campus / maximum | 17 | 0.0028305054 | 5.6770188e-08 | 2,713,765 |
| Sci-Art / minimum | 4 | 0.0033637285 | 6.7297724e-08 | 1,934,022 |
| Sci-Art / median | 11 | 0.0029847026 | 8.8316182e-08 | 2,641,926 |
| Sci-Art / maximum | 19 | 0.0035114288 | 1.8256085e-07 | 2,539,456 |

The difference panels show the true, unamplified absolute differences and are
therefore visually black at ordinary display scale. This is strong numerical
agreement across the pinned official execution and the project ALoD-EXT
execution, but it is not described as bitwise parity. The separate
Released-Evaluator-versus-ALoD-EXT comparison remains the bitwise proof that
the extension's materialization and residency changes preserve the project
renderer output exactly.

## Depth-distribution comparison

The maximum pairwise TVD pairs were recomputed directly from the frozen
Stage-A depth histograms and checked against the stored criteria values:

- Campus: cameras `eval-0001-img-002148-cam-0001` and
  `eval-0011-img-012919-cam-0001`, TVD 0.07197548966251403;
- Sci-Art: cameras `eval-0006-img-000008-cam-0001` and
  `eval-0019-img-000020-cam-0001`, TVD 0.013039350956579489.

The normalized comparison figure preserves the discrete distributions without
smoothing. It visually supports the frozen result that Campus crosses the
depth-materiality TVD threshold while Sci-Art changes less strongly; it does
not introduce a new criterion or label.

## Figures

- [Campus summary](../../benchmark_artifacts/alod_visual_evidence_001/campus/campus_summary.png)
- [Sci-Art summary](../../benchmark_artifacts/alod_visual_evidence_001/sciart/sciart_summary.png)
- [Released versus bounded RGB parity](../../benchmark_artifacts/alod_visual_evidence_001/comparison/alod_ext_rgb_parity.png)
- [Pinned official evaluator versus bounded ALoD-EXT](../../benchmark_artifacts/alod_visual_evidence_001/comparison/official_source_vs_alod_ext.png)
- [Maximum-depth-TVD comparison](../../benchmark_artifacts/alod_visual_evidence_001/comparison/depth_tvd_comparison.png)

Individual camera images and diagnostics are indexed with byte sizes, SHA-256
hashes, and pixel dimensions in the package manifest.
