# Redistribution review and decisions

The release uses explicit per-file terms in `release/licensing.json`, together
with `LICENSE`, `LICENSE.md`, `NOTICE` and the original upstream licence texts.

The author authorized MIT for original contributions. Source-sensitive ALoD
HSPT, selection, frustum and camera modules, CityGaussian cell/visibility/LoD
modules, upstream patches and official runtime harnesses are conservatively
treated as containing upstream-derived expression. Their upstream terms are
preserved; no claim of licence independence is needed to redistribute these
research components under the permissions their original licences provide.
Original MIT code that calls these components does not grant broader rights
to the inherited portions. Copied viewer/helper notices and Apache text are
retained, and the COLMAP utility retains its BSD notice.

The ALoD licence expressly permits distribution of the work and derivatives
subject to its research/noncommercial limits, licence copy and notice retention.
CityGaussian's CC BY-NC-SA permits noncommercial sharing/adaptation with credit,
change disclosure and share-alike conditions. These conditions are recorded
rather than describing the whole companion as unrestricted open source.

Dataset boundaries were reviewed separately. MatrixCity-derived camera records
retain their original noncommercial terms. The UrbanScene3D copyright section
prohibits disseminating the dataset or altered variations; the Campus download
page supplies no explicit redistribution permission. Consequently, 183 files
with source-derived Campus/Sci-Art camera poses/calibrations are excluded from
the public tree and history. Their paths and original hashes are listed in
`release/public_exclusions.json`. The scripts to reconstruct them from locally
acquired data remain available. Project numerical performance, count, validation
and statistical outputs remain as evidence, without redistributing the source
imagery, trained models or copied raw camera payloads.

The review includes substantial-function comparison, source-sensitive module
inspection, notice/permission inspection, per-file classification and raw-camera
payload searches. The AST comparison alone is not used as proof of independent
authorship. External libraries obtained by the reader retain their own terms.

Primary sources checked:

- [Pinned ALoD licence](https://github.com/FelixWindisch/LoDOfGaussians/blob/73f547341547ae021ff036ad74fdf73a1240b450/LICENSE.md)
- [CityGaussian licence](https://github.com/Linketic/CityGaussian)
- [CC BY-NC-SA legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
- [CC BY-NC legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
- [UrbanScene3D copyright conditions](https://github.com/Linxius/UrbanScene3D#copyright)
- [Campus source and downloads](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/)

This documents release engineering decisions and their source permissions; it
does not assert commercial clearance or legal advice for downstream uses.
