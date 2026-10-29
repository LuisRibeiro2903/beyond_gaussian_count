# ALoD Faithful Rendering Summary

Three fresh processes per configuration, each with 10 ordered warmups and all 71 frozen measured cameras. Times are pooled CUDA-synchronized measurements.

| Configuration | End-to-end mean / median / p95 (ms) | FPS from mean | Prep mean (ms) | Raster mean (ms) | Selected hierarchy mean | H2D mean | Peak torch allocated | Whole-device VRAM max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ALOD-REPRO-HOST-NOCACHE` | 186.77 / 184.87 / 207.52 | 5.35 | 182.34 | 4.15 | 3.432M | 3.432M rows | 1.28 GiB | 3.96 GiB |
| `ALOD-REPRO-NOFRUSTUM` | 188.02 / 186.16 / 210.13 | 5.32 | 183.63 | 4.07 | 3.443M | 3.443M rows | 1.28 GiB | 4.00 GiB |
| `ALOD-REPRO-PREVIOUS-REUSE` | 134.21 / 131.58 / 160.44 | 7.45 | 129.80 | 4.03 | 3.456M | 0.088M rows | 1.28 GiB | 4.55 GiB |

`frame_time_ms` in the raw logs is raster-only. The table uses `end_to_end_frame_time_ms` for total performance and FPS.
