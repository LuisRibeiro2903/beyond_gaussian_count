# CityGaussian V1 Released-Code Canonical Benchmark Memo

Date: 2026-08-24

Campaign ID: `citygs_canonical_20260824_135615`

Canonical artifact: `/home/luis/tese/benchmark_artifacts/citygs_canonical_20260824_135615`

Formal source commit: `cfc2136459dd63cc47dc3acfde55a3d72a4d8bba` (`git_dirty=false`)

This memo supersedes the July 30 memo for performance claims. The older package's
generic frame time measured `GsplatRenderer.render(...)`, not the complete
per-frame benchmark pipeline. The canonical primary metric is now synchronized
end-to-end time:

```text
end_to_end_frame_time_ms = preparation_time_ms + render_call_time_ms
```

Preparation is the synchronized `scene.prepare_frame(...)` interval. Render call
is the synchronized `GsplatRenderer.render(...)` interval; CUDA-event
rasterization is retained as a component within it. Loading, the excluded
warmup, post-render finite checks, serialization, and image writing are outside
the boundary. FPS is `1000 / mean(end_to_end_frame_time_ms)` and is never derived
from the renderer-only value.

## Protocol and scope

All seven configurations replayed the same 1,440-frame, 1920x1080
`citygaussian_23m_navigation_v1` trajectory in three independent processes.
Each process first rendered trajectory frame 0 once as a deterministic warmup;
that frame was excluded and memory peaks were reset afterward. Configuration
order was rotated across repetitions while camera order remained 0..1439.

The environment was an NVIDIA GeForce RTX 5080 (17,094,475,776 reported bytes),
Python 3.12.12, PyTorch `2.11.0.dev20260128+cu129`, CUDA 12.9, and gsplat 1.5.3.
This reproduces inspected released-code runtime behavior through the project
gsplat integration. It does not reproduce the author GraphDECO renderer,
training, or paper evaluation protocol.

## Corrected performance

Values are mean of the three independent run means. The `±` term is the sample
standard deviation of those three end-to-end means, not a pooled-frame error.

| Configuration | End-to-end ms ± run SD | End-to-end FPS | Preparation ms | Render call ms | CUDA raster ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CG-FINEST-FLAT` | 11.360 ± 0.622 | 88.03 | 0.010 | 11.350 | 11.289 |
| `CG-FIXED-FINEST-ALL` | 21.733 ± 1.266 | 46.01 | 10.339 | 11.394 | 11.352 |
| `CG-FIXED-FINEST-VISIBILITY` | 19.049 ± 1.227 | 52.50 | 8.386 | 10.663 | 10.611 |
| `CG-LOD-RELEASE` | 15.673 ± 0.738 | 63.80 | 8.126 | 7.547 | 7.480 |
| `FLAT-CITYGS-FULL` | 18.502 ± 0.816 | 54.05 | 0.011 | 18.491 | 18.434 |
| `CG-FUSED-CELLS-ALL` | 45.726 ± 2.021 | 21.87 | 27.221 | 18.506 | 18.415 |
| `CG-FUSED-VISIBILITY` | 62.928 ± 5.722 | 15.89 | 25.524 | 37.404 | 37.345 |

Complete within-run mean, median, p95, p99, minimum, and maximum statistics for
all four timing stages, plus across-run minimum, maximum, standard deviation,
and coefficient of variation, are in `tables/config_summary.json`.
End-to-end coefficients of variation range from 4.41% to 9.09%; the fused
visibility configuration is both the slowest and least stable (9.09%).

## Controlled transitions

| Transition | End-to-end time | End-to-end FPS | Preparation | Render call | CUDA raster | Selected Gaussians | Render peak allocated | Post-transfer allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| finest flat → cells-all | +91.32% | -47.73% | +106150.20% | +0.39% | +0.55% | 0.00% | +1.831 GiB | +0.176 GiB |
| finest cells-all → visibility | -12.35% | +14.09% | -18.89% | -6.41% | -6.52% | -49.22% | +0.000 GiB | +0.000 GiB |
| finest visibility → released LoD | -17.72% | +21.54% | -3.09% | -29.22% | -29.51% | -29.04% | +0.893 GiB | +2.161 GiB |
| fused flat → cells-all | +147.15% | -59.54% | +254196.77% | +0.08% | -0.10% | 0.00% | +5.489 GiB | +0.349 GiB |
| fused cells-all → visibility | +37.62% | -27.33% | -6.23% | +102.12% | +102.80% | -49.41% | -0.000 GiB | +0.000 GiB |

The very large preparation percentages use near-zero flat preparation as their
denominator; the absolute costs (10.339 ms finest and 27.221 ms fused) are the
more useful interpretation.

The results support a precise distinction: cell organization has negligible
renderer-call overhead for the same asset (+0.39% finest, +0.08% fused), but the
current gsplat integration incurs substantial active-set preparation and
assembly overhead. This nearly doubles finest end-to-end time and increases
fused end-to-end time by 147%.

Visibility is branch-dependent. On finest SH2 it halves the selected set,
reduces render-call time 6.41%, and reduces end-to-end time 12.35%. On fused SH3
it also halves the selected set, but render-call/CUDA time roughly doubles and
end-to-end time increases 37.62%. A reduced selected count is therefore not, by
itself, evidence of improved raster performance in this unpacked gsplat path.

Released adaptive LoD improves both renderer-call and end-to-end time relative
to finest visibility: -29.22% render call and -17.72% end-to-end, yielding
63.80 FPS. This is a controlled transition within the released LoD family, but
it adds fully resident middle and coarsest representations.

## Workload and memory

| Configuration | Resident Gaussians | Mean selected/active | Mean radii-visible | Post-transfer allocated | Render peak allocated | Raw render peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CG-FINEST-FLAT` | 11.695M | 11.695M | 3.018M | 1.656 GiB | 3.045 GiB | 3.268 GiB |
| `CG-FIXED-FINEST-ALL` | 11.695M | 11.695M | 3.018M | 1.831 GiB | 4.876 GiB | 6.051 GiB |
| `CG-FIXED-FINEST-VISIBILITY` | 11.695M | 5.938M | 3.017M | 1.831 GiB | 4.876 GiB | 16.021 GiB |
| `CG-LOD-RELEASE` | 25.495M | 4.214M | 1.975M | 3.992 GiB | 5.769 GiB | 5.979 GiB |
| `FLAT-CITYGS-FULL` | 23.390M | 23.390M | 5.273M | 5.142 GiB | 7.605 GiB | 9.506 GiB |
| `CG-FUSED-CELLS-ALL` | 23.390M | 23.390M | 5.273M | 5.491 GiB | 13.095 GiB | 13.693 GiB |
| `CG-FUSED-VISIBILITY` | 23.390M | 11.834M | 5.272M | 5.491 GiB | 13.094 GiB | 28.539 GiB |

`CG-LOD-RELEASE` keeps 25.495M Gaussians resident but selects 4.214M on average
(range 1.958M–6.705M). Its average selected cells per frame are 8.37 finest,
1.19 middle, and 7.47 coarsest. Relative to finest visibility, fully resident
multi-level LoD adds 2.161 GiB post-transfer allocated and 0.893 GiB render-peak
allocated memory.

The raw PyTorch reserved counter reaches 16.021 GiB and 28.539 GiB in the two
visibility configurations, including a value above the physical device total.
Those reserved values are preserved but are not treated as physically resident
VRAM; this nightly PyTorch/WSL allocator counter requires further validation.
PyTorch allocated stages are the primary campaign memory comparison. NVML
process accounting is also not used because it remains unreliable under WSL.

## Validation and provenance

The persistent `validation/` directory contains:

- the complete 118-test suite log (`OK`);
- the five-Gaussian CUDA smoke log;
- passing 12-camera finest and fused visibility oracles;
- the passing 12-camera released-LoD oracle;
- 12-camera finest and fused flat/all-cell equivalence outputs;
- package validation for seven configurations, three repetitions, and 30,240
  measured frames, with ambiguous legacy timing fields absent.

Same-asset equivalence remains exact for fused RGB/alpha. Finest cell reordering
has zero alpha difference and only tiny RGB drift (maximum absolute difference
`7.6532e-05`, minimum PSNR 146.13 dB).

The manifest records a clean formal worktree and SHA-256 identities for the
trajectory, all four external assets, and all seven tracked configuration
files. The external assets and trajectory were not modified.

## Answers to the evaluation questions

1. Cell organization does not materially change same-asset renderer-call cost.
2. It introduces 10.339 ms finest and 27.221 ms fused preparation cost.
3. Visibility reduces renderer cost on finest SH2, but increases it on fused SH3.
4. Visibility reduces end-to-end cost on finest, but increases it on fused.
5. The SH2 and SH3 branches therefore have opposite visibility outcomes even
   though both reduce selected Gaussians by about 49%.
6. Released adaptive LoD improves renderer-call time by 29.22% versus finest
   visibility.
7. It improves end-to-end time by 17.72% and FPS by 21.54%.
8. Fully resident LoD adds 2.161 GiB post-transfer allocated memory relative to
   finest visibility.
9. End-to-end run-mean CVs are 4.41%–9.09%; directional conclusions are stable
   for the large effects, while small renderer-only cell deltas are appropriately
   interpreted as negligible rather than speedups or regressions.

## Limitations

- This is released-source runtime-behavior reproduction through gsplat, not the
  official GraphDECO renderer or a reproduction of paper FPS/quality numbers.
- `CG-LOD-RELEASE` reproduces the inspected released Nyquist selector, not the
  paper's described fixed-distance MatrixCity policy.
- All LoD assets remain resident; this is not streaming, host residency, or a
  bounded GPU cache.
- No ground-truth PSNR/SSIM/LPIPS or temporal LoD-popping study was performed.
- The fused SH3 and released LoD SH2 representations differ; the cross-family
  `FLAT-CITYGS-FULL → CG-LOD-RELEASE` comparison is not an isolated-method
  transition and is intentionally omitted from controlled claims.
- Reserved-memory and NVML process counters require WSL-specific caution as
  described above.

No new method, runtime optimization, VQ reconstruction, LightGaussian
generation, paper fixed-distance LoD, FlashGS, streaming, host residency,
training, dependency installation/update, push, or asset modification occurred
in this campaign.
