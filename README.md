# Beyond Gaussian Count

**Runtime Analysis and Materialization-Aware Execution in Large-Scale 3D Gaussian Splatting**

Code and experimental evidence for Luis Ribeiro's Master's dissertation on
reproducing, instrumenting, and evaluating large-scale Gaussian rendering on
consumer hardware.

A research companion with source code, numerical evidence and reproducible
analyses. Original project contributions use MIT; upstream-derived portions
retain their research/noncommercial, share-alike or Apache terms. See
[licence scope](LICENSE.md) before reuse.

The [repository](https://github.com/LuisRibeiro2903/beyound_gaussian_count) is
prepared for public companion use within the documented scope. Its visibility
is controlled by the author. No DOI or public archive is claimed.

## Start here

Inspect the evidence with Python 3.10 or newer; no GPU or downloaded scene is
needed:

```bash
python tools/verify_release.py
python tools/verify_natural_results.py
python tools/check_publication.py
python tools/results.py
python tools/results.py --table benchmark_artifacts/citygs_canonical_20260824_135615/tables/config_summary.json
```

Read the [evidence guide](docs/EVIDENCE.md),
[reproduction matrix](evidence/REPRODUCTION_MATRIX.md), and
[comparison rules](evidence/COMPARISON_VALIDITY.md) before interpreting results.
ALoD reproduction, ALoD-Exec extensions (`ALOD-EXT` artifact identifiers), and
synthetic controls answer different questions. The experiments do not establish
a cross-method ranking or reproduce the original papers' headline FPS.

## Contents

| Location | Purpose |
| --- | --- |
| `gs_framework/` | Runtime, independent/reference implementations, benchmark tools, tests and experiment configurations |
| `benchmark_artifacts/` | Twelve main numerical campaigns plus selected supporting records, with original protocols and validation records |
| `runs/alod/` | Selected training, negative-preflight and oracle metadata; no trained hierarchies or bulk oracle arrays |
| `evidence/` | Experiment index, comparison boundaries and reproducibility assessment |
| `reproduction/figures/` | Portable numerical figure generator with recorded typography change |
| `release/` | File hashes, source-state record, thesis-reference map and omitted-artifact inventory |

Numerical frame traces are compressed losslessly. To restore their original
paths and bytes for existing analysis scripts:

```bash
python tools/restore_traces.py
```

## Reproduce analyses and run software

[REPRODUCING.md](docs/REPRODUCING.md) separates offline evidence inspection,
CPU tests, a synthetic GPU smoke test, and numerical figure generation from experiments requiring CUDA,
external reference source and scene assets. Download locations, pinned source
revisions and input identity records are in [EXTERNAL_INPUTS.md](docs/EXTERNAL_INPUTS.md).

The source snapshot includes relevant working files that were untracked in the
development repository. Historical numerical records retain their original
commit and per-file identities; this packaging commit is a different identity.
Read the [reproducibility limits](docs/REPRODUCIBILITY_LIMITS.md) and [source coverage](release/historical_source_coverage.json).
Some historical launch scripts retain machine-specific paths and are evidence
of the executed procedure, rather than portable launchers.

## Attribution and reuse

CityGaussian and A LoD of Gaussians are methods by their respective authors.
The project contributes reproduction, validation, instrumentation, controlled
evaluation, and explicitly labelled extensions. Full upstream repositories,
datasets, checkpoints, imagery and thesis fonts are obtained separately.
Campus/Sci-Art raw camera payloads are also excluded; local reconstruction scripts remain available.

See [LICENSE.md](LICENSE.md) and [the licensing review](docs/LICENSE_REVIEW.md).
The release does **not** apply one blanket licence to all third-party or
source-derived material. Citation metadata are in [CITATION.cff](CITATION.cff).
