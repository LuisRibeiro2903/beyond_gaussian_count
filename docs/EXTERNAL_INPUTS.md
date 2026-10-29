# External inputs and source revisions

| Input | Source / identity | Release treatment |
| --- | --- | --- |
| CityGaussian V1 | https://github.com/Linketic/CityGaussian — `e16c526f78ab01d377b13b3e6862dcc93ddbe953` | Obtain upstream; inspect CC BY-NC-SA and submodule terms |
| A LoD of Gaussians | https://github.com/FelixWindisch/LoDOfGaussians — `73f547341547ae021ff036ad74fdf73a1240b450` | Obtain upstream; research/noncommercial and inherited upstream terms apply |
| gsplat | https://github.com/nerfstudio-project/gsplat | Use versions/revisions in each campaign, not an assumed latest version |
| MatrixCity | https://city-super.github.io/matrixcity/ | Download from original providers; preserve dataset and upstream model terms |
| Campus standalone chunk | https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/ | Imagery redistribution permission not established in the local manifest |
| UrbanScene3D Sci-Art | See `gs_framework/datasets/alod/ALOD-PREV-US3D-SCIART-01/manifest.json` | Original download locations and noncommercial research restrictions recorded there |

CityGaussian asset hashes, environment records and camera identity are in
`benchmark_artifacts/citygs_canonical_20260824_135615/manifest.json` and the
official parity/comparison packages. The complete released LoD PLY models
are external inputs; a directory name containing `vq` is not proof of VQ decoding.

ALoD hierarchy/camera identities are preserved in the selected dataset manifests,
training records, oracle semantic manifests and benchmark protocols. The
Campus/Sci-Art audit is under `thesis_notes/data/`. Missing arrays and models
are listed in `release/omitted_artifacts.json`; they are **not** silently included
in the numerical release. Their archive location remains unassigned.

The topology correction and instrumentation patches are under
`gs_framework/patches/alod/`. Apply only to the pinned source revision. Inspect
the patch and original licence before use; a patch does not exempt its upstream
context from licence conditions. CUDA compatibility and native build details
are preserved in the original build manifest and official comparison packages.
