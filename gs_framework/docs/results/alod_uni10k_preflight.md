# Uni10K supplementary ALoD external-validity preflight

Date: 2026-09-18

Status: `BLOCKED-ALOD-UNI10K-FINE-HEADROOM`

This was a public-source ALoD preflight for the optional external-validity
question: whether the materialization/residency findings from formal Run009
would persist on a hierarchy with materially stronger natural view-dependent
LoD. It does not replace Run009 and did not alter the ALoD selector,
ALoD-EXT semantics, optimization objective, hierarchy algorithm, densification
semantics, or rasterizer.

## Candidate and completed stages

The candidate is public Uni10K at source commit
`811af1acf8d274ac5d59b498a89ba9dd8c523f86`. The validated derived dataset,
`ALOD-REPRO-UNI10K-01`, has 10,063 registered aerial images from four camera
IDs, an official `llff_hold=8` split of 8,805 training and 1,258 evaluation
images, and 6,170,655 sparse points. It has no ground/street images; it was
selected for its much larger camera set and multi-flight aerial scale diversity,
not as a mixed aerial/ground dataset. The manifest preserves camera, pose,
name, sparse point, colour, and error semantics while removing only unused
COLMAP observations/tracks.

RUN005 completed the official 30,000-iteration coarse stage and hierarchy
creator with exit status zero. The coarse PLY has 6,270,655 finite rows and
SHA-256 `a30180db1d8a98b3c9f1982a17242e593aa1075c22fea1219288f48f61f5bbbe`.
The creator produced a 3,208,740,352-byte initial `.dhier` with 12,341,309
rows and SHA-256
`bb0b58966b34e0855767eef69c2a9a97a118c72ec75508a91343de8a830b2429`.
Independent inspection found SH degree 3, maximum depth 23, finite attributes,
valid parent/depth links, and all 12,341,309 rows reachable from one root.

Runs002--004 did not reveal a training defect. They reached iterations 24,750,
28,290, and 27,070 before a monitor bug falsely fired the 1,800-second
no-progress abort: the parser recognized legacy progress messages but not the
official trainer's carriage-return `tqdm` lines. The resulting
`KeyboardInterrupt` was injected by the monitor. The parser was corrected and
regression-checked against all three frozen logs before RUN005.

## Structural HSPT result

A camera-, rendering-, quality-, FPS-, and LoD-outcome-independent structural
sweep used the existing source-faithful calibration rules. The largest tested
value inside a stable passing neighborhood is `SPT_root_volume=50`, with
passing neighbors 75 and 25. At 50 the initial hierarchy gives:

| Field | Value |
| --- | ---: |
| Upper hierarchy rows | 1,997 |
| SPT count | 56 |
| Flattened SPT rows | 12,339,368 |
| SPT coverage | 99.984272% |
| Minimum / median / maximum SPT rows | 375 / 18,077 / 1,542,663 |
| Conservative complete working set | 1,744,660 rows |
| Reduced cache target | 6,400,000 rows |
| Cache headroom | 4,655,340 rows |

The pinned official loader independently rebuilt exactly 56 SPTs, 1,997 upper
rows, and a 1,542,663-row maximum SPT, so the structural prediction and actual
source construction agree. Peak process-group RSS for that check was
5,924,225,024 bytes and peak whole-device VRAM was 2,475,687,936 bytes.

## Stop decision

Fine loading would prepend the 100,000-row skybox, producing 12,441,309 initial
rows under the frozen 15,000,000-row cap. This leaves 2,558,691 rows, or
17.05794% of the cap. The absolute 2,500,000-row minimum passes by 58,691 rows,
but the frozen 20% minimum headroom ratio fails.

The candidate is structurally valid and its HSPT is non-degenerate and
cache-feasible, but its permitted fine-training growth is not robust under the
predeclared resource envelope. The original protocol requires stopping before
expensive fine training when preflight is infeasible. Raising the cap,
weakening the gate, reducing the candidate, or changing densification would be
a new resource/method decision after observing the failed gate. No such change
was made; no fine hierarchy, selection oracle, strong-LoD classification, or
ALoD-EXT benchmark exists for Uni10K.

This is therefore a resource-feasibility negative result, not a `WEAK-LOD`
result. The strong-LoD question remains untested on this candidate because the
experiment correctly stopped before the final hierarchy existed. A further
training attempt is not recommended for the current thesis unless the resource
envelope is deliberately redesigned and justified in advance.

## Canonical evidence

- Dataset: `datasets/alod/ALOD-REPRO-UNI10K-01/manifest.json`.
- Resource policy: `environments/alod-official/uni10k-preflight-resource-policy.json`.
- Completed raw run: `runs/alod/ALOD-REPRO-UNI10K-01/ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT-RUN005`.
- Closure: `runs/alod/ALOD-REPRO-UNI10K-01/ALOD-REPRO-UNI10K-TRAIN-001-PREFLIGHT-RUN005-CLOSURE/closure.json`.
- Independent validation, structural sweep, and official HSPT rebuild are in
  the closure directory's `validation/` subdirectory.

Historical failed/interrupted attempts remain immutable. The closure directory
contains post-run analysis only and does not modify RUN005.
