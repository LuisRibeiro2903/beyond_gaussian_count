# Release, rename and dissertation citation

The reader documentation update is version **1.1.1**. The complete numerical
record remains pinned to `v1.0.0`; `v1.1.0` preserves the previous reader
release. Keep those existing tags and their evidence identities intact.

## Correcting the GitHub repository name

The intended spelling is **`beyond_gaussian_count`**. The current repository
is `LuisRibeiro2903/beyound_gaussian_count`. Until the GitHub rename is
completed, links and `CITATION.cff` retain that existing address.

With repository admin access, open the repository on GitHub, go to
**Settings → General → Repository name**, enter `beyond_gaussian_count`,
and click **Rename**. This changes its address to
`https://github.com/LuisRibeiro2903/beyond_gaussian_count`.

After GitHub confirms the rename, update this clone from its repository root:

```bash
git remote set-url origin git@github.com:LuisRibeiro2903/beyond_gaussian_count.git
```

The local folder can keep its existing name; no move is required. GitHub
redirects existing repository web links and Git operations, so the addresses
preserved in earlier releases continue to work. Do not create a different
repository at the old address, which would remove that redirect. See
[GitHub's rename instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository).

## Publishing a reviewed version

Run the release verifier, numerical verifier, figure generator and publication
check documented in [REPRODUCING.md](REPRODUCING.md). Changes to packaged files
must also update their SHA-256 entries and licence classifications in
`companion.json`. Its validation record distinguishes the current checks from
the previous release's preserved validation.

Use an immutable release tag for the reviewed commit. Push the commit and its
tag when ready to publish, and keep the `v1.0.0` evidence tag available to
readers. Verify that the repository, release and evidence tag are accessible
without signing in before submitting the dissertation. Local checks do not
establish that GitHub CI has passed or that the repository is public.

## Citing the companion

Use the author, software title, release version and date from `CITATION.cff`.
Include the repository URL and full commit hash, obtained with
`git rev-parse HEAD` for the reviewed release. Link to that release or commit,
rather than only to the moving `main` branch. Once the rename is complete,
use the corrected URL in new dissertation references.

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
