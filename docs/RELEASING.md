# Releases and dissertation citation

The prepared reader companion update is version **1.1.2**. It corrects public
repository links and documents the historical-path audit; publication remains
a separate step. The complete numerical record remains pinned to `v1.0.0`;
`v1.1.0` and `v1.1.1` preserve earlier reader releases. Keep all existing tags,
commit identities and archived evidence hashes intact.

## Repository address

The GitHub rename is complete. The public repository is
[`LuisRibeiro2903/beyond_gaussian_count`](https://github.com/LuisRibeiro2903/beyond_gaussian_count),
and this clone's origin already uses that address. Current documentation and
citation metadata use it, including links to the original `v1.0.0` paths.
Changing the repository address does not change a historical release identity.

## Publishing a reviewed version

Run the release verifier, numerical verifier, figure generator and publication
check documented in [REPRODUCING.md](REPRODUCING.md):

```bash
python tools/verify_release.py
python tools/verify_natural_results.py
python tools/check_publication.py --history
python reproduction/figures/generate_final_empirical_figures.py --output local/figures
```

Changes to packaged files must also update their SHA-256 entries in
`companion.json`; new files need a licence classification there. Preserve
existing classifications and result-file hashes. The manifest's own hash
stays null because Git records its identity. Its validation record distinguishes
current checks from the previous release's preserved validation.

Use a new immutable release tag for the reviewed commit; `v1.1.2` is the
proposed tag for this update. Publishing the commit and tag requires the
author's explicit authorization. Never overwrite `v1.0.0`, `v1.1.0` or
`v1.1.1`. Verify public repository, release and evidence accessibility before
submitting the dissertation. Local checks do not establish that GitHub CI has
passed or that a proposed tag has been published.

Historical execution paths are provenance, not reader installation requirements.
Keep sealed records unchanged and use portable paths in new documentation and
examples. See [the path guidance](REPRODUCING.md#recorded-execution-paths) and
[the 1.1.2 audit](PUBLICATION_AUDIT_1.1.2.md). Removing a path from a current
file does not remove it from reachable Git history. Any history rewrite needs
separate authorization and an assessment of release and citation impact.

## Citing the companion

Use the author, software title, release version and date from `CITATION.cff`.
Include the corrected repository URL and full commit hash, obtained with
`git rev-parse HEAD` for the reviewed release. Link to that release or commit,
rather than only to the moving `main` branch. Cite the compact reader release
and the complete `v1.0.0` evidence record separately.

After publication finalizes the new reader identity, update the dissertation's
`ReproducibilityRecord.tex` reader version and full commit, and its companion
bibliography entry. Until then retain the existing published citation; do not
insert a speculative hash. The [audit report](PUBLICATION_AUDIT_1.1.2.md) gives
the exact source locations observed during this review.

A suitable availability statement is:

> The accompanying repository provides the project implementation, validation
> and benchmark tools, and preserved numerical evidence supporting the reported
> analyses. It records upstream revisions, applied patches, input identities
> and reproduction procedures. Third-party datasets, scene models and
> restricted raw camera payloads must be acquired or reconstructed separately;
> the repository documents the limits of exact historical replay.

Add the actual URL, version and commit to that statement. Cite CityGaussian,
ALoD and other original methods separately where they are discussed. Add a
dissertation URL or archive identifier only after it exists; no DOI is
assigned by this repository.
