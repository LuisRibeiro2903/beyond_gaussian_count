# Reproduce analyses and use the implementation

## Supported workflows

| Workflow | Inputs | Validation and scope |
| --- | --- | --- |
| Inspect tables and verify release hashes | Python standard library | Supported without GPU or external data |
| Regenerate five empirical figures | NumPy/Matplotlib; included numerical evidence | Independent numerical checks of 126 saved inputs; no renderer |
| Run synthetic parser/selection tests | NumPy/PyTorch | Supported with CPU-only wheels; no datasets |
| Render a synthetic scene | CUDA/PyTorch/gsplat | `tools/smoke_render.py`; analytic splat and empty background, packed/unpacked |
| Rerun real-data GPU experiments | Licensed external scenes/hierarchies, reference sources and campaign configuration | Conditional on obtaining the recorded input bytes; exact original-paper performance is not promised |

## Installation for evidence and core tests

Python 3.12 on Linux was tested in a new isolated environment. Start from a
clean clone; the default workflow never imports CUDA or downloads a model:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evidence.txt
python tools/verify_release.py
python tools/verify_natural_results.py
python tools/check_publication.py
python tools/results.py
python reproduction/figures/generate_final_empirical_figures.py --output local/figures
```

The figure generator reads compressed traces directly. It checks the saved
statistics against original numerical frame records before plotting, and writes
a provenance record. It substitutes bundled DejaVu Sans for the excluded thesis
font; figure data are unchanged, but the PDFs differ typographically.

For the synthetic format and reference-selection suite:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
cd gs_framework
python -m unittest unit_tests.test_alod_dhier_format unit_tests.test_alod_selection_oracle unit_tests.test_alod_reference_frustum unit_tests.test_alod_reference_hspt unit_tests.test_alod_reference_selection
```

The broader `python -m unittest discover -s unit_tests` suite additionally needs
the runtime dependencies below. Integration tests require external assets and
are not equivalent to the synthetic unit tests. Original validation reports
are scientific evidence, not a claim that every gate has been rerun at release.

Legacy aggregators expect uncompressed files. From the repository root,
`python tools/restore_traces.py` restores exact bytes to ignored paths and
refuses to overwrite changed files. No scientific record is silently rewritten.

## CUDA rendering

Install a CUDA-enabled PyTorch build appropriate to your machine, then
`python -m pip install -r requirements-runtime.txt`. The tested historical GPU
stack is PyTorch `2.11.0.dev20260128+cu129`, gsplat `1.5.3`, CUDA toolkit
`12.9.86`, and an RTX 5080 (`sm_120`). This records a validated stack, rather
than promising a historical nightly wheel remains downloadable forever.

gsplat can build its native extension on first use. Set the toolkit explicitly
before starting Python. An older `/usr/bin/nvcc` can fail with
`Unsupported gpu architecture 'compute_120'` even when PyTorch sees the GPU:

```bash
export CUDA_HOME=/path/to/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
nvcc --version
```

For a Conda CUDA toolkit whose headers/libraries live under `targets/`, also
set these before running the smoke test (this layout was used in validation):

```bash
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

Then run `python tools/smoke_render.py` from the repository root.

`CUDA_HOME` must name the actual compatible toolkit on your machine. The
architecture shown is for the RTX 5080; use your own GPU's architecture.
The recorded original build/environment inventories remain under
`gs_framework/environments/alod-official/` and the campaign packages.

With the recorded CityGaussian LoD PLY inputs obtained separately, from
`gs_framework/`:

```bash
python -m benchmarks.benchmark_real_scene --config-id CG-LOD-RELEASE \
  --lod-finest-asset /path/to/light_50_vq.ply \
  --lod-middle-asset /path/to/light_66_vq.ply \
  --lod-coarsest-asset /path/to/light_75_vq.ply \
  --output-dir ../local/citygaussian-smoke --max-frames 3 --num-warmup 1
```

That is a smoke run, not the formal timing protocol. For scientific comparisons,
use the original campaign's cameras, input hashes, warmup, timing boundaries
and fresh-process repetition policy. A different GPU or retrained hierarchy
produces a new experiment. See `REPRODUCIBILITY_LIMITS.md` before making claims.

## External camera reconstruction

Campus and Sci-Art original calibration/pose payloads are excluded. After
obtaining the original archives under their providers' terms, use the existing
preparation scripts and manifests. For example, regenerate Campus cameras:

```bash
python gs_framework/scripts/alod/freeze_campus_prevalence_cameras.py \
  --dataset /path/to/campus \
  --manifest gs_framework/datasets/alod/ALOD-PREV-H3DGS-CAMPUS-CHUNK-01/manifest.json \
  --output local/campus-cameras.json
```

For Sci-Art, `freeze_sciart_prevalence_cameras.py` accepts `--train-dataset`,
`--oracle-dataset`, `--manifest` and `--output`. The original input identities
are in the dataset manifests. Locally generated manifests can differ in bytes
because paths are embedded; verify numerical camera content and source hashes,
and record the changed configuration. Full oracle replay also needs the
external trained hierarchy and binary oracle payloads, not just a camera JSON.
