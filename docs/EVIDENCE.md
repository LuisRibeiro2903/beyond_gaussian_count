# Evidence guide

These packages are referenced by the dissertation. Read each original protocol and validation status before using a result. Reproduction, project extensions, synthetic controls and historical comparisons remain distinct.

- [`alod_ext_formal_run009_001`](../benchmark_artifacts/alod_ext_formal_run009_001/): original metadata and numerical records.
- [`alod_ext_natural_previous_completion_001`](../benchmark_artifacts/alod_ext_natural_previous_completion_001/): original metadata and numerical records.
- [`alod_ext_prevalence_association_001`](../benchmark_artifacts/alod_ext_prevalence_association_001/): original metadata and numerical records.
- [`alod_ext_run009_closure_001`](../benchmark_artifacts/alod_ext_run009_closure_001/): original metadata and numerical records.
- [`alod_faithful_reproduction_001`](../benchmark_artifacts/alod_faithful_reproduction_001/): original metadata and numerical records.
- [`alod_synth_strong_lod_control_001`](../benchmark_artifacts/alod_synth_strong_lod_control_001/): original metadata and numerical records.
- [`alod_synth_strong_lod_ext_multiset_closure_001`](../benchmark_artifacts/alod_synth_strong_lod_ext_multiset_closure_001/): original metadata and numerical records.
- [`citygs_12camera_attribution_20260824_222048`](../benchmark_artifacts/citygs_12camera_attribution_20260824_222048/): original metadata and numerical records.
- [`citygs_backend_microbench_20260824_213420`](../benchmark_artifacts/citygs_backend_microbench_20260824_213420/): original metadata and numerical records.
- [`citygs_canonical_20260824_135615`](../benchmark_artifacts/citygs_canonical_20260824_135615/): original metadata and numerical records.
- [`citygs_official_clean_full_20260825_163319`](../benchmark_artifacts/citygs_official_clean_full_20260825_163319/): original metadata and numerical records.
- [`citygs_official_parity_20260824_191215`](../benchmark_artifacts/citygs_official_parity_20260824_191215/): original metadata and numerical records.

The exact chapter/line mapping and availability of each literal evidence reference are in [`thesis_evidence.json`](../release/thesis_evidence.json). The chapter hashes identify the dissertation sources inspected; the dissertation itself is not redistributed here. References marked `external_or_unresolved` are not claimed available.

Frames are stored as deterministic gzip files; restore them with `python tools/restore_traces.py` before running historical aggregators or figure scripts. Original hashes, timing boundaries, rejected-run labels, and protocol amendments are preserved.

## Availability boundaries

Directory subsets contain numerical/provenance records, not every payload named
by their original manifests. Excluded raw Campus/Sci-Art camera files and binary
oracle/model payloads are external. Exact omissions are indexed in
`release/public_exclusions.json` and `release/omitted_artifacts.json`. Original
references are preserved for identity, not rewritten to imply full availability.
See `REPRODUCIBILITY_LIMITS.md` for supported workflows.
