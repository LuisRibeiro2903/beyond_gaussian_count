# ALoD natural-dataset prevalence Stage-A result

Study ID: `ALOD-LOD-PREVALENCE-STAGE-A-001`

Protocol: `ALOD-LOD-PREVALENCE-DESIGN-001`

Status: complete

## Outcome

All three prospectively listed natural hierarchies completed training,
artifact validation, deterministic official selection, and independent
selection parity. There were no resource-infeasible datasets, implementation
failures, unresolved external failures, or exact-degenerate outcomes.

The exact Stage-A tally is:

- `VALID-NONTRIVIAL-LOD`: 1/3 (Campus);
- `VALID-WEAK-LOD`: 2/3 (MatrixCity Run009 and Sci-Art);
- `VALID-DEGENERATE-LOD`: 0/3;
- no scientific label: 0/3.

This is a three-dataset result, not a population prevalence estimate.

## Frozen outcome table

| Dataset/run | Admission | Peak RSS / VRAM (GiB) | Total rows / max depth | Count relative range | Min Jaccard | Max depth TVD | Nearest-scale rho | Scene-centre rho | Label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MatrixCity mixed Run009 | pass | 7.163 / 6.501 | 7,072,627 / 55 | 0.044096 | 0.911268 | 0.018369 | not frozen for original Run009 | not frozen for original Run009 | `VALID-WEAK-LOD` |
| H3DGS Campus standalone | pass | 5.871 / 2.548 | 1,143,221 / 44 | 0.2523741040 | 0.4486992016 | 0.0719754897 | -0.071304 | 0.338261 | `VALID-NONTRIVIAL-LOD` |
| UrbanScene3D Sci-Art | pass | 6.944 / 5.024 | 5,074,185 / 48 | 0.0242054432 | 0.9388327853 | 0.0130393510 | 0.100000 | 0.833766 | `VALID-WEAK-LOD` |

Run009 predates the prevalence design's natural-camera proxy manifest, so this
table does not invent post hoc continuous proxy values for it. Its already
frozen selection metrics and weak classification are retained unchanged.

## Interpretation

The source-faithful official selector can produce a materially varying natural
hierarchy: Campus passes all three prospective predicates. That result rules
out the claim that the implementation is universally static or degenerate.
However, nontrivial behavior is not prevalent across this small sample:
MatrixCity Run009 and Sci-Art both change selected identities without crossing
the frozen count and depth-distribution materiality thresholds.

Scene extent or hierarchy depth alone does not guarantee a nontrivial result.
Sci-Art is the largest real scene in the new pair and shows a strong
scene-centre proxy correlation, yet its selected-count range is only 2.42% and
its maximum pairwise depth TVD is 0.0130. Campus, with fewer hierarchy rows,
shows a 25.24% count range and 0.0720 TVD. The defensible conclusion is about
these exact trained hierarchies and frozen cameras, not all natural datasets or
the paper's full configurations.

## Validity and boundaries

Every canonical oracle used the pinned official source at
`73f547341547ae021ff036ad74fdf73a1240b450`, fresh-process cold evaluation,
two exact repetitions per held-out camera, integrity validation, and
independent CPU parity. ALoD-EXT was excluded from classification. Resource
numbers are sampled watchdog maxima; VRAM is whole-device usage.

External interruptions and observation-tool corrections remained visible in
the per-dataset chronologies and did not become weak outcomes. No dataset,
training method, camera set, or threshold was changed after inspecting an LoD
outcome.

No PSNR/SSIM/LPIPS comparison, FPS claim, paper-number reproduction,
population-rate claim, Stage-B repeat, ALoD-EXT association result, or
synthetic-control result is implied.

## Stage-B consequence

The frozen protocol now requires the randomness audit before Stage B, followed
by one repeat of a Stage-A weak dataset and one repeat of the naturally
nontrivial Campus dataset. The exact nominal-seed repeat must measure execution
nondeterminism before any distinct-seed plumbing is introduced. Dataset choice
within the two weak outcomes must be justified and frozen before launching the
repeat; Stage-A results must not be used to tune training or classification
thresholds.

## Supporting records

- MatrixCity Run009:
  `docs/results/alod_formal_mixed_selection_oracle.md`.
- Campus:
  `docs/results/alod_prevalence_campus_stage_a.md`.
- Sci-Art:
  `docs/results/alod_prevalence_sciart_stage_a.md`.
- Frozen study design:
  `thesis_notes/ALOD_LOD_PREVALENCE_STUDY_DESIGN.md`.
