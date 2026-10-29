# ALoD Frustum Ablation

Positive avoided/saved values mean frustum-on performed less work than frustum-off. The 213 samples are camera-paired across three fresh-process replicates.

| Measure | Mean | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| semantic selected hierarchy rows avoided | 10781.8592 | 733.0000 | 1746.0000 | 704075.0000 |
| h2d rows avoided | 10781.8592 | 733.0000 | 1746.0000 | 704075.0000 |
| h2d bytes avoided | 991931.0423 | 67436.0000 | 160632.0000 | 64774900.0000 |
| packed projection survivors avoided | 0.1549 | 0.0000 | 1.0000 | 2.0000 |
| candidate spts avoided | 5.0282 | 5.0000 | 10.0000 | 10.0000 |
| host gather time saved ms | 0.3564 | -0.4317 | 20.2771 | 29.2596 |
| h2d time saved ms | 0.0019 | -0.0308 | 4.4315 | 29.4241 |
| pre raster time saved ms | -0.0255 | -0.0014 | 0.6881 | 28.3944 |
| raster time saved ms | -0.0821 | 0.0347 | 1.1430 | 9.2073 |
| end to end time saved ms | 1.2477 | 0.7869 | 31.6633 | 70.3013 |
| frustum traversal increment ms on minus off | 0.3719 | 0.3367 | 0.6827 | 1.5220 |

Mean selected-hierarchy reduction: 0.3132%.
Mean end-to-end change, frustum-on versus off: -0.664%.
Image check: 64/71 bit-identical; maximum RGB absolute difference 0.00035188.

Frustum-on avoided a small semantic/H2D workload on most cameras, but the total frame-time effect is near measurement variability and does not establish the paper's cached-renderer frustum timing trend.
