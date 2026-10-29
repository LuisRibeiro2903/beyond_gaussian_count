# ALoD Official-Source Smoke Environment

Generated: 2026-08-05

Final gate: **PASS-ALOD-SMOKE-ENVIRONMENT**

## Result

Pinned official source commit `73f547341547ae021ff036ad74fdf73a1240b450` now imports and runs its locally compiled CUDA extensions on the RTX 5080 from an isolated Python 3.10 environment. The canonical source checkout and `nerfstudio2` remained unchanged. No training, hierarchy generation, gsplat rendering, dataset operation, evaluation campaign, or benchmark ran.

Paths:

- environment: `/home/luis/miniconda3/envs/alod_official`;
- detached disposable worktree: `/home/luis/tese/alod_official_worktree_73f5473`;
- retained external builds and wheels: `/home/luis/tese/alod_official_build`;
- canonical clean reference: `/home/luis/tese/LoDOfGaussians`.

Resolved core versions:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Conda CUDA Toolkit / nvcc | 12.8.2 / 12.8.93 |
| gsplat | 1.5.3 |
| GCC / G++ | 13.4.0 |
| CMake | 4.4.2 |
| fused-SSIM source | `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` |

The detected device is `NVIDIA GeForce RTX 5080`, driver `591.74`, 16,303 MiB VRAM, compute capability `12.0`. PyTorch advertises `sm_120`, and the selected `nvcc` advertises `compute_120`.

## Official-command deviation

The README requests CUDA Toolkit 12.6 and an unpinned cu126 PyTorch wheel. CUDA 12.6 cannot emit native SM 120 code, so this environment uses the first native Blackwell-capable toolkit line, CUDA 12.8, plus the official PyTorch cu128 wheel. The detected capability, compiler target list, embedded cubins, and actual kernel execution were all checked; compilation alone was not treated as compatibility evidence.

The upstream CMake file hard-codes architectures `70;75;86`. The disposable worktree changes only that line to `120`; the standalone patch and SHA-256 are:

```text
patches/alod/cmake-cuda-architectures-sm120.patch
d57e1f7e28b2c779f5a07a8de5807b953f22979f83fd53d0c0853da01cd5d7af
```

CMake 4.4 also requires `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for the upstream `cmake_minimum_required(VERSION 3.0)`. This is a command-line compatibility flag, not a source change.

## Native components

| Component | Build | Local result | Runtime check |
|---|---|---|---|
| `simple_knn._C` | setuptools CUDAExtension | CPython 3.10 `.so`, SM 120 | `distCUDA2` returned finite nonnegative distances |
| `gaussian_hierarchy._C` | setuptools CUDAExtension | CPython 3.10 `.so`, SM 120 | `compute_relocation` matched analytic values |
| `fused_ssim_cuda` | setuptools CUDAExtension | CPython 3.10 `.so`, SM 120 | identical images returned SSIM `1.0` |
| `libGaussianHierarchy.a` | CMake Release | static library containing SM 120 cubins | `cuobjdump` verified embedded target |
| `GaussianHierarchyCreator` | CMake Release | native executable | source-defined no-argument diagnostic reached |
| `GaussianHierarchyMerger` | CMake Release | native executable | source-defined no-argument diagnostic reached |

The pinned checkout tracks an older `gaussian_hierarchy` CPython-3.10 binary with SHA-256 `17b2d377…`. It was removed only from the disposable worktree before rebuilding. Imported extension paths resolve under `alod_official`, and their hashes differ from the tracked artifact. Exact paths, sizes, wheel hashes, binary hashes, flags, and source locations are in `build-manifest.json`.

`gsplat 1.5.3` and the official `gaussian_renderer` imported successfully. Its native backend is lazy and was not invoked because this task explicitly prohibited inventing a rendering smoke workload.

## Smoke coverage

Successful imports include `torch`, `gsplat`, `simple_knn._C`, `gaussian_hierarchy._C`, `fused_ssim`, `fused_ssim_cuda`, `gaussian_renderer`, `scene.gaussian_model`, `train_hierarchy`, `eval_hierarchy`, and `hierarchy_viewer`.

The following metadata paths completed without starting work:

- `train.py --help`;
- `eval_hierarchy.py --help`;
- `hierarchy_viewer.py --help`;
- creator and merger startup through their source-defined missing-argument checks.

The bundled dummy COLMAP files are not a standalone upstream smoke operation: viewer/evaluation still require a hierarchy. They were therefore not used. Imports plus isolated official native CUDA calls are the smallest source-supported path available without training or creating a `.dhier`.

Machine-readable results are in `smoke-results.json`. Failed or intentionally non-zero attempts and exact classifications are in `attempt-log.md`.

## Preservation checks

Four independent `nerfstudio2` fingerprints—Conda explicit lock, pip freeze, Conda history, and top-level site-packages manifest—match before and after exactly. The canonical ALoD checkout remains at the pinned commit with an empty status and empty tracked diff.

The pre-existing `gs_framework` tracked diff hash and dirty-file content-manifest hash are unchanged. The task added only this ALoD environment documentation, locks, scripts, manifest, results, attempt log, and patch. No CityGaussian, shared runtime, test, benchmark, `CG-*`, or `FLAT-CITYGS-FULL` content changed.

## Recreation

The retained environment can be rechecked with:

```bash
/home/luis/tese/gs_framework/environments/alod-official/smoke-commands.sh
```

For a clean recreation, first perform the cleanup below, then run:

```bash
/home/luis/tese/gs_framework/environments/alod-official/build-commands.sh
```

The script uses `conda-explicit.txt`, installs the exact cu128 wheels, checks out both exact source commits, applies the build-only patch, rebuilds all native artifacts, installs the frozen Python inventory, builds the CMake tools, and reruns the smoke gate. GPU access and network access are required. The scripts set CUDA/compiler variables only in their own process.

## Cleanup

Cleanup is documented but was not run, because the isolated environment and build products are retained for the next bounded task:

```bash
/home/luis/miniconda3/bin/conda remove -y -p /home/luis/miniconda3/envs/alod_official --all
git -C /home/luis/tese/LoDOfGaussians worktree remove --force /home/luis/tese/alod_official_worktree_73f5473
rm -rf /home/luis/tese/alod_official_build
```

The exact next bounded task is to select and freeze a small legal COLMAP scene, define conservative Gaussian/cache/resource limits, run the first capped official-source training, and produce the first final `.dhier` oracle and provenance manifest.
