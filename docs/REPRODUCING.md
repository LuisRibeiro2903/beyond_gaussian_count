# Reproducing the reader companion

## Evidence and figures

Use a normal Git clone so the pinned v1.0.0 evidence is available. A shallow
clone can fetch it with `git fetch origin tag v1.0.0`. Alternatively, set
`BEYOND_EVIDENCE_DIR` to an extracted v1.0.0 source archive. No script silently
downloads scene data or runs a renderer to inspect the results.

Python 3.12 on Linux is the validated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evidence.txt
python tools/verify_release.py
python tools/verify_natural_results.py
python reproduction/figures/generate_final_empirical_figures.py --output local/figures
```

The figures check 126 saved input records, and the natural-table verifier checks
876 statistics from 18 processes and 405 measured frames. These are checks of
preserved numerical evidence, not new GPU performance measurements. The figures
use bundled DejaVu Sans instead of the dissertation's excluded font.

## Focused runtime tests

Install PyTorch separately for your platform. For CPU-only tests:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-runtime.txt
python tools/fetch_citygaussian_reference.py
export GS_FRAMEWORK_CITYGAUSSIAN_REFERENCE_REPO="$PWD/local/reference-citygaussian"
cd gs_framework
python -m unittest discover -s unit_tests
```

Six optional source-oracle tests use one separately acquired, hash-checked file
from pinned CityGaussian source. They skip when the explicit reference setting
is absent. The selected tests cover the retained runtime and correctness tools;
tests of archived auxiliary workflows remain in v1.0.0. This is not a claim of
full oracle replay without the external assets.

## GPU smoke test and real scenes

Use a CUDA-enabled PyTorch build and install `requirements-runtime.txt`. The
validated GPU stack used PyTorch `2.11.0.dev20260128+cu129`, gsplat `1.5.3`,
CUDA toolkit `12.9.86` and an RTX 5080. Set the actual toolkit before Python;
an older system `nvcc` cannot compile `sm_120`:

```bash
export CUDA_HOME=/path/to/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
```

For a Conda toolkit whose headers/libraries live under `targets/`, also set:

```bash
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

Then run `python tools/smoke_render.py`. It checks an analytic Gaussian and an
empty background in packed and unpacked modes. Use your own GPU architecture
when it differs from the example.

With the separately acquired CityGaussian LoD inputs, from `gs_framework/`:

```bash
python -m benchmarks.benchmark_real_scene --config-id CG-LOD-RELEASE \
  --lod-finest-asset /path/to/light_50_vq.ply \
  --lod-middle-asset /path/to/light_66_vq.ply \
  --lod-coarsest-asset /path/to/light_75_vq.ply \
  --output-dir ../local/smoke --max-frames 3 --num-warmup 1
```

That command is a smoke run. Use the archived campaign's complete protocol,
input identities, warmup, repetition and timing policies for formal experiments.
Different inputs, hardware or retraining define a new experiment. See
[EXTERNAL_INPUTS.md](EXTERNAL_INPUTS.md) for acquisition and licensing boundaries.

## Optional complete working record

To inspect auxiliary code, training/preparation tools, original protocols or
all raw numerical records in a separate directory:

```bash
git worktree add --detach local/full-record v1.0.0
```

The original paths are preserved there, including the source/execution hash
records, licensing review and reproducibility limits. Do not copy restricted
datasets into the repository. The old record is deliberately retained and is
not needed as thousands of loose files in the default reader view.
