# Canonical CityGaussian V1 benchmark tables

| Configuration | End-to-end ms | Between-run SD | FPS | Preparation ms | Render call ms | CUDA raster ms |
|---|---:|---:|---:|---:|---:|---:|
| `CG-FINEST-FLAT` | 11.360 | 0.622 | 88.03 | 0.010 | 11.350 | 11.289 |
| `CG-FIXED-FINEST-ALL` | 21.733 | 1.266 | 46.01 | 10.339 | 11.394 | 11.352 |
| `CG-FIXED-FINEST-VISIBILITY` | 19.049 | 1.227 | 52.50 | 8.386 | 10.663 | 10.611 |
| `CG-LOD-RELEASE` | 15.673 | 0.738 | 63.80 | 8.126 | 7.547 | 7.480 |
| `FLAT-CITYGS-FULL` | 18.502 | 0.816 | 54.05 | 0.011 | 18.491 | 18.434 |
| `CG-FUSED-CELLS-ALL` | 45.726 | 2.021 | 21.87 | 27.221 | 18.506 | 18.415 |
| `CG-FUSED-VISIBILITY` | 62.928 | 5.722 | 15.89 | 25.524 | 37.404 | 37.345 |

Controlled transitions are in `controlled_transitions.json`.
