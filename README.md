# Beyond Gaussian Count

**Runtime Analysis and Materialization-Aware Execution in Large-Scale 3D Gaussian Splatting**

Reader companion to Luis Ribeiro's Master's dissertation. This view contains
the runtime implementation, focused correctness tests, thirteen key result
records, and commands to verify the numerical analyses.

## Read the results

Start with [results/README.md](results/README.md). It maps the thesis topics to
the relevant records and code. The JSON result files are unchanged copies of
the original campaign tables; their source identities are in
[companion.json](companion.json).

```bash
python tools/verify_release.py
python tools/verify_natural_results.py
```

These commands need Python and Git, with no GPU or downloaded scene.

## Reproduce the analyses

```bash
python -m pip install -r requirements-evidence.txt
python reproduction/figures/generate_final_empirical_figures.py --output local/figures
```

This regenerates five empirical figures from the saved numerical records.
[REPRODUCING.md](docs/REPRODUCING.md) covers installation, focused tests,
synthetic GPU rendering, and external inputs for real-scene experiments.

## Detailed evidence

The [complete audited record at v1.0.0](https://github.com/LuisRibeiro2903/beyound_gaussian_count/tree/v1.0.0)
preserves the full numerical traces, protocols, validation reports, historical
source snapshots, auxiliary tools and development documentation. The analysis
commands read those records from the pinned Git revision without expanding
thousands of files into this working tree. Standard Git clones include it.

For a shallow clone, fetch it once with `git fetch origin tag v1.0.0`.
For a ZIP download, also obtain the [full-record ZIP](https://github.com/LuisRibeiro2903/beyound_gaussian_count/archive/refs/tags/v1.0.0.zip)
and set `BEYOND_EVIDENCE_DIR` to its extracted root. The expected revision is
`42af4e9ec5fb9bb4c75ef3229b6bbb517fb28a6c`.

## Scope and reuse

CityGaussian and ALoD algorithms belong to their original authors. This work
provides reproduction, instrumentation, controlled evaluation and the explicitly
identified ALoD-Exec extension. Results are compared within their stated
protocols, not used as a cross-method ranking or a reproduction of paper FPS.

Images, scene models, trained hierarchies and restricted raw camera payloads
must be obtained separately. Negative findings remain visible in the result
guide. Five older natural-association source versions were not recovered;
their exact historical replay is not claimed. The final natural campaign's
sealed files and recovered canonical CityGaussian sources are documented in
the full record. Numerical evidence can still be inspected and checked.

Original contributions use MIT. Upstream-derived portions retain their
noncommercial/research, share-alike or Apache terms; see [LICENSE.md](LICENSE.md)
and [NOTICE](NOTICE). Citation metadata are in [CITATION.cff](CITATION.cff).
No DOI is claimed. Repository visibility remains the author's choice.
