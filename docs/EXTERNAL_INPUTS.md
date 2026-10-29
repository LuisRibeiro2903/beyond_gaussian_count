# External inputs

The project distributes code and numerical evidence, not complete scenes.

| Input | Source and identity |
| --- | --- |
| CityGaussian V1 source | https://github.com/Linketic/CityGaussian — `e16c526f78ab01d377b13b3e6862dcc93ddbe953` |
| ALoD source | https://github.com/FelixWindisch/LoDOfGaussians — `73f547341547ae021ff036ad74fdf73a1240b450` |
| gsplat | https://github.com/nerfstudio-project/gsplat — use the recorded campaign version/build |
| MatrixCity | https://city-super.github.io/matrixcity/ — original CC BY-NC terms apply |
| Campus standalone chunk | https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/ — acquire from the provider |
| UrbanScene3D Sci-Art | https://github.com/Linxius/UrbanScene3D — acquire under its noncommercial/non-redistribution terms |

The pinned full record contains exact model/camera hashes, dataset manifests,
preparation scripts, dependency/build records and protocol amendments. The
topology and compatibility patches remain under `gs_framework/patches/alod/`.
The MatrixCity formal evaluation cameras needed by the provided benchmark
preset are retained with their applicable dataset terms.

Images, trained hierarchies, released PLY models and binary oracle arrays are
external. Campus/Sci-Art raw pose/calibration payloads are excluded from both
the reader view and the audited full record. Their local reconstruction scripts
are in that full record. Repeating training does not promise byte-identical
models. A hash identifies required data; it does not make missing data available.
The original paper assets and protocols are not universally reconstructible.
