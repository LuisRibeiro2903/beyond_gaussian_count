# Repository URL and historical-path audit — proposed v1.1.2

Review date: 29 October 2026. Base: published reader release `v1.1.1`, commit
`be099fae09c89a78178d1e0df210899f8e6f9a5b`. This patch corrects presentation
and explains retained provenance. It changes no implementation or measurement.

## Current URL and release metadata

The public repository is
[`LuisRibeiro2903/beyond_gaussian_count`](https://github.com/LuisRibeiro2903/beyond_gaussian_count).
Seven stale spelling occurrences were corrected in six current files:

| File | Correction |
| --- | --- |
| `README.md` | Complete-record tag link and full-record ZIP link. |
| `CITATION.cff` | `repository-code`; proposed version updated to `1.1.2`. |
| `companion.json` | Evidence URL; proposed version and current validation metadata updated. |
| `LICENSE.md` | Archived redistribution-review link; licence substance unchanged. |
| `docs/RELEASING.md` | Replaced the obsolete rename procedure with the completed rename and current address. |
| `docs/REPRODUCING.md` | Archived Run009 training-record link. |

All tracked current files, including workflows, notices and release metadata,
were searched. No stale spelling remains in current URLs. No workflow or
`NOTICE` edit was needed. New path guidance distinguishes recorded execution
paths from reproduction inputs. `companion.json` updates only hashes for changed
reader files and adds this report as MIT documentation; result identities,
evidence commit and evidence-manifest SHA-256 remain unchanged. No DOI or
archive identifier was invented. The citation date is 2026-10-29, the date of
this prepared patch; no `v1.1.2` tag or GitHub release has been published.

Anonymous HTTP checks returned 200 for the repository, `v1.0.0`, `v1.1.1`,
the archived licence review, the archived Run009 training record and the
full-record ZIP (which resolves to GitHub's codeload service). GitHub metadata
also identifies the renamed repository as public. No dependency on the old
address's redirect remains in current reader links.

## History scope and path findings

`git ls-remote --heads --tags` found one published branch (`main`) and three
annotated tags. Every advertised object and peeled commit matched the local
refs. `git rev-list --all` at the audit base covered six reachable commits;
all 2,493 distinct reachable file blobs were inspected, including decompression
of 415 gzip blobs (293,181,427 bytes after decompression). The single binary
blob is the published PNG preview. Blob text scans do not inspect image pixels.
Commit/tag metadata were also checked for unexpected identity or network fields;
the author's public attribution is intentional. Unreachable objects and private
server data are outside this public-ref audit.

Counts below are textual home-path occurrences in distinct blob versions,
counted once per blob regardless of how many branches, tags or filenames reuse
it. They are not counts of users or distinct path strings. A changed file
version counts separately. The classification separates generated/provenance
records from prose, source literals and configuration portability candidates.

| Category | Occurrences | Repository paths represented | Assessment |
| --- | ---: | ---: | --- |
| A: low-risk historical provenance | 5,722 | 913 | Generated manifests, resolved configurations, logs, numerical summaries, camera/oracle metadata and environment records. Retain original bytes. |
| B: unnecessary personal path details / portability candidates | 239 | 75 | Historical prose, source defaults and configuration examples. Low privacy risk; no such home paths occur in current reader documentation or current benchmark configuration examples. Existing source defaults remain outside this documentation patch. |
| C: confirmed sensitive information | 0 | 0 | No confirmed credentials, authenticated URLs, private network endpoints, unrelated personal files or private machine identifiers were found by these checks. |

Of the 5,961 combined occurrences, 5,960 are Linux/macOS-style home paths and
one is a Windows user-directory path expressed through a WSL mount. The latter
is in `gs_framework/specs/alod-first-capped-training-plan.md:49` at `v1.0.0`,
where it explains a historical multiprocessing failure on a Windows-backed
temporary directory. Both classifications expose ordinary directory structure,
not authentication material.

Representative A records at `v1.0.0` are
`benchmark_artifacts/citygs_canonical_20260824_135615/manifest.json` and
`benchmark_artifacts/alod_ext_natural_previous_completion_001/runs/` records
named `resolved_config.json`. Other examples are the retained MatrixCity camera
fixtures and `results/alod_exec/natural_completion.json`. Representative B
locations are the historical training-plan note above and source defaults in
`gs_framework/citygaussian_assets.py` and the two CityGaussian oracle extraction
scripts. No raw path inventory or credential values are reproduced here.

There are also 498 Python/Conda/CUDA environment-path matches, overlapping the
home-path counts, and 207 temporary/mount/workspace-path matches, including the
Windows match. These are additional pattern counts, not additional A/B totals.
They describe environments, build locations and local outputs. Four apparent
private IPv4 matches were manually rejected: all were the public CUDA package
version `10.3.9.90` in package/environment lists. There were zero candidates for
private hostname suffixes, UNC shares, authenticated URLs, URL credential
parameters, credential assignments or recorded hostname fields. Public dataset,
source and package-service URLs do not identify private network services.

The current reader tree retains 159 home-path occurrences: 117 in ten
camera/oracle fixture files, 36 in five unchanged result files, and six in four
source files. The source literals are optional reference/oracle path defaults;
the documented workflow supplies its own reference and input paths. Altering
those defaults would be a separate portability change, not necessary for this
publication patch. The audit diagnostics are kept locally outside the release;
no extra public archive or third-party asset was added.

## Treatment and privacy assessment

| Record type | Role and protection | Treatment |
| --- | --- | --- |
| Current reader documentation | Installation/acquisition guidance; reader-file hashes are updateable for a new patch. | Already uses `/path/to/...`, repository-relative paths or environment variables. Zero personal home paths required replacement. Added explicit historical-path guidance. |
| Current runnable configuration examples | Portable inputs for a new experiment; tracked in the reader inventory. | No personal home paths found; no changes. |
| Generated historical manifests/configurations | Record the original input locations, execution and environment. Many files are sealed by archived SHA-256 inventories and campaign manifests. | Retained unchanged; readers supply their own paths and verify input hashes. |
| Archived numerical evidence and reader copies | Exact saved results; reader copies must match the pinned archive byte for byte. | All result hashes and bytes retained, including path strings. |
| Immutable tagged history and source snapshots | Released identities cited by the dissertation; snapshots may be sealed by executed-source hashes. | No tag, public commit, snapshot or historical record altered. |

The privacy exposure is limited to the author's local username, directory
organization, experiment/model filenames, environment names and build paths.
A username in a filesystem path is not a credential. Generated paths help trace
which files and environments a measurement used, although input hashes establish
identity and the original directory layout is not required for reproduction.
Historical prose paths have little installation value today, but editing their
old tags would change public commit identities and dissertation references.

Retaining these ordinary paths is proportionate. Rewriting evidence would risk
manifest verification and misrepresent the execution record without removing
values from earlier Git objects. Removing the remaining strings from all public
history would require a separately authorized history rewrite and replacement of
immutable release identities. No such rewrite or credential revocation is
justified by the findings. Pattern scans cannot prove the absence of every
possible sensitive string. Any later confirmed secret requires private review
and revocation where applicable before considering a separately authorized
history response.

Existing docs already described preserved paths, but did not explicitly explain
their privacy and portability implications. The new
[recorded-execution-path guidance](REPRODUCING.md#recorded-execution-paths)
provides that distinction. Licensing classifications, `NOTICE`, excluded
Campus/Sci-Art camera payloads, historical source-coverage declarations, dataset
and source hashes, external-asset boundaries and replay limitations are intact.

## Validation

Checks used Python 3.12.12 from the existing local environment (NumPy 1.26.4,
Matplotlib 3.10.8). `python` below denotes that interpreter on `PATH`.

| Command / check | Result |
| --- | --- |
| `python tools/verify_release.py` | PASS: 173 classified reader files, 13 unchanged original results and pinned archive manifest. |
| `python tools/verify_natural_results.py` | PASS: 876 statistics, 18 fresh-process records, 405 measured frames. |
| `python tools/check_publication.py --history` | PASS: 173 classified files, six pre-patch reachable commits, 183 excluded camera payloads. |
| `python reproduction/figures/generate_final_empirical_figures.py --output local/release-1.1.2-figures` | PASS: 126 evidence inputs verified; five PDFs and one PNG generated, all byte-identical to the previous generation. |
| Full `EvidenceStore.read` hash sweep | PASS: all 3,063 file identities in the unchanged pinned evidence inventory. |
| Gitleaks 8.30.1, `git . --log-opts='--all --full-history' --max-archive-depth=2 --ignore-gitleaks-allow --redact=100` | PASS: all six pre-patch commits, about 295 MB scanned, zero findings with the default independent ruleset. Gzip archives included. |
| Current Markdown links, metadata and tree comparisons | PASS: 46 local file links resolve; proposed version and corrected citation URL agree; no stale current spelling; all existing licence classifications unchanged. Runtime, config, fixture, result, source-snapshot and workflow trees unchanged from `v1.1.1`. |
| Public refs and anonymous HTTP checks | PASS: advertised branch/tag objects match local refs; all six corrected public resources return HTTP 200. |
| `git diff --check` | PASS. |

Gitleaks was downloaded from its [official 8.30.1 release](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1)
to a temporary location and verified against the release checksum. The Linux
x64 archive SHA-256 is
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`.
Its JSON report has no findings. The broader path/network checks manually
classified the package-version false positives described above; no credential
values were printed. Commit and annotated-tag metadata were reviewed as well.
Detailed redacted diagnostics and the local audit/check scripts are retained in
ignored `local/publication-audit-1.1.2/`, outside the packaged release.

The figure generator emitted unwritable Fontconfig-cache warnings but completed
successfully with byte-identical outputs. No GPU experiment or runtime test
suite was rerun for this documentation-only patch. Earlier runtime validation
remains identified through the previous release's `companion.json`; it is not
claimed as a fresh check here. GitHub CI for this unpublished patch is untested.
The release, numerical, publication and independent history checks are repeated
after the local commit; that adds the patch as a seventh reachable commit.

## Release identity and dissertation coordination

The proposed release is `v1.1.2` on the normal local branch
`codex/publication-url-privacy-v1.1.2`. Its exact proposed commit is the commit
containing this report (`git rev-parse HEAD` when checking out that branch),
reported to the author after commit creation. It is deliberately not embedded
in its own hashed manifest or report. Existing identities are:

| Tag | Annotated tag object | Peeled release commit |
| --- | --- | --- |
| `v1.0.0` | `0246800f303a3d5834d8a732f84dab1789be2500` | `42af4e9ec5fb9bb4c75ef3229b6bbb517fb28a6c` |
| `v1.1.0` | `659c341d937bfa11450e392a8a2f40ea0caa744b` | `7ebd4e8034a68d99161491932ce78ef2a0b36e8e` |
| `v1.1.1` | `4c4b44db1c81ddfe5c7130abe5a26a53f33d7bd1` | `be099fae09c89a78178d1e0df210899f8e6f9a5b` |

The dissertation was inspected read-only. After the author publishes the final
reader release, update these source locations (line numbers at audit time):

- `thesis_latex/appendices/ReproducibilityRecord.tex:22`: reader release
  `v1.1.1` becomes the published `v1.1.2`.
- `thesis_latex/appendices/ReproducibilityRecord.tex:23`: replace the reader
  commit with the finalized full release commit. Keep lines 27–28's separate
  `v1.0.0` evidence identity unchanged; line 21's repository URL is already correct.
- `thesis_latex/dissertation.bib:130`: update the companion version, release
  date and full reader commit in `ribeiro2026companion`.
- `thesis_latex/dissertation.bib:131`: change the reader tag URL to the
  published `v1.1.2` URL. Retain the separate complete-evidence reference.

No dissertation source was modified or committed. Author authorization is
required to push the reviewed commit and create/publish a new tag or release.
No push, tag publication or history rewrite is part of this task. There is no
outstanding privacy action requiring approval; retained historical paths are
intentional provenance.
