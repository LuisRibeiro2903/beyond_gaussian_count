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

The generator writes five PDFs, a PNG preview of the CityGaussian timing
figure, and `provenance.json` with input and output hashes under `local/figures/`.
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
cd ..
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
  --output-dir ../local/citygaussian-sample --sample-frames 3 --log-gpu-events
```

That command validates three evenly spaced cameras without warmup and writes
`record_kind=camera_sample_validation` in `summary.json`. It is a smoke run.
Use the archived campaign's complete protocol,
input identities, warmup, repetition and timing policies for formal experiments.
Different inputs, hardware or retraining define a new experiment. See
[EXTERNAL_INPUTS.md](EXTERNAL_INPUTS.md) for acquisition and licensing boundaries.

## ALoD-Exec example

This example uses the formal MatrixCity Run009 hierarchy with the tracked
71-camera evaluation path. It compares the source-faithful evaluator-style
no-cache preset with the project-specific CUDA materialization filter and
bounded row cache. The extension is not part of released ALoD.

The trained `.dhier` file is external. The required file identity is
`asset.sha256` in [the formal result record](../results/alod_exec/matrixcity_formal.json):
`38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b`.
Check your acquired file with `sha256sum /path/to/Run009.dhier` before running.
The [archived Run009 training record](https://github.com/LuisRibeiro2903/beyound_gaussian_count/blob/v1.0.0/gs_framework/docs/results/alod_formal_mixed_training_run009.md)
documents how the original hierarchy was produced; repeating training does not
guarantee the same bytes. These presets fix the hierarchy layout, scene scale,
selection settings and cache capacity. An arbitrary `.dhier` or a camera-path
override does not make them appropriate for a different scene.

After installing the CUDA runtime dependencies described above, run from
`gs_framework/`, using the same absolute asset path for both commands:

```bash
python -m benchmarks.benchmark_real_scene \
  --config-id ALOD-REPRO-HOST-NOCACHE \
  --asset /path/to/Run009.dhier \
  --output-dir ../local/alod-nocache-sample \
  --sample-frames 3 --log-gpu-events

python -m benchmarks.benchmark_real_scene \
  --config-id ALOD-EXT-PREMATERIALIZATION-CUDA-PRACTICAL \
  --asset /path/to/Run009.dhier \
  --output-dir ../local/alod-exec-sample \
  --sample-frames 3 --log-gpu-events
```

Both commands use the same three evenly spaced evaluation cameras. Each writes
`resolved_config.json`, `frames.jsonl`, `summary.json` and
`camera_sample_summary.json` to its own output directory. A successful sample
has `record_kind=camera_sample_validation`, `frame_count=3`, and
`finite_image_all=true` and `finite_alpha_all=true` in `summary.json`. No image
files are saved by these presets. The first frame includes initialization;
these samples have no warmup and do not establish a performance improvement.

In `frames.jsonl`, inspect `alod_semantic_selected_rows`,
`alod_materialization_rows`, `alod_unique_materialization_rows`,
`backend_projection_survivor_count`, `alod_h2d_bytes`,
`resident_gaussian_count` and `end_to_end_frame_time_ms` using the
[metric glossary](METRICS.md). Cache statistics refer to dynamic hierarchy
properties; persistent skybox and compact metadata have separate accounting.

For the formal comparison, follow the archived campaign protocol linked
through `companion.json`, including correctness gates, ten warmup frames,
all 71 cameras and three fresh processes per configuration. `--num-repeat`
repeats a camera sequence inside one process; it does not create independent
process replicates. The `--formal` label alone does not validate a protocol.
The sample commands above exercise the runtime without reproducing that
campaign or its timing results.

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
