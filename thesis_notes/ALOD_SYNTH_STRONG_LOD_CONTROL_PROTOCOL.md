# ALoD synthetic strong-LoD positive-control protocol

Protocol and experiment ID: `ALOD-SYNTH-STRONG-LOD-CONTROL-001`

Freeze date: 2026-09-25

Status: `FROZEN-AFTER-SMOKE-CORRECTION-BEFORE-SCIENTIFIC-CUTS`

Classification: controlled synthetic positive-mechanism experiment and
project-specific ALoD-EXT evaluation. It is excluded from the natural Stage-A
denominator, is not a training method, and does not replace the pending Stage-B
repeats.

## Question and invariants

The primary question is whether the already validated ALoD-EXT execution paths
still reduce materialization, transfer, and frame time when the unchanged
released ALoD selector is supplied with a hierarchy deliberately constructed
to permit a strong ordered camera-scale response.

Only the analytic hierarchy and camera geometry are artificial. The official
two-pixel selector predicate, `distance_multiplier=1`, HSPT construction,
four-plane frustum convention, conservative pre-materialization filter,
previous-working-set reuse, and bounded row-LRU implementation are unchanged.
No camera is mapped to a hierarchy level, no selected ID is injected or
replaced, and no threshold in this protocol may be changed after inspecting a
scientific cut. All outcomes, including failure of the strong-LoD gate or loss
of ALoD-EXT opportunity, are accepted.

## Deterministic analytic hierarchy

The non-skybox leaves form a row-major `1024 x 512` rectangular lattice in the
plane `z=0`, spanning `[-32,32] x [-16,16]`. Leaf centres are cell centres.
Each leaf is an axis-aligned Gaussian with scale `(0.028, 0.028, 0.014)`,
activated opacity `0.82`, identity quaternion `(1,0,0,0)`, and degree-1 SH.
The DC colour is a deterministic combination of normalized position, a
checker term, and a radial term; the remaining SH coefficients are zero. The
formula is fixed in the construction script and is not optimized against
renders.

The hierarchy is a strict balanced binary spatial tree. At each interior node,
the current leaf rectangle is split in half along its longer cell-count axis;
ties split x. Children preserve lower-coordinate then upper-coordinate order.
The scientific tree therefore has `524,288` leaves, `1,048,575` hierarchy
nodes, and maximum depth 19. A prefix of exactly 100,000 transparent,
finite-property skybox rows uses the official `-99` topology sentinel, making
the serialized scientific asset contain 1,148,575 rows with hierarchy root
global ID 100,000. Transparent prefix rows exist solely because the released
evaluator fixes this prefix size.

The serialization is the official little-endian, field-major degree-1
`.dhier` layout. A separate smoke artifact uses the same equations and topology
algorithm with a `64 x 32` lattice (4,095 hierarchy nodes plus the fixed
prefix). It is for serialization, loader, selector, and renderer debugging
only and cannot supply a scientific outcome.

### Parent aggregation

For child `i`, define

```text
A_i = s_ix*s_iy + s_ix*s_iz + s_iy*s_iz
w_i = alpha_i * A_i
W   = sum_i w_i
```

and use `w_i/W` for all mixture moments. With child mean `mu_i` and covariance
`Sigma_i = R_i diag(s_i^2) R_i^T`, the parent is

```text
mu_p    = (1/W) sum_i w_i mu_i
Sigma_p = (1/W) sum_i w_i [Sigma_i +
                            (mu_i-mu_p)(mu_i-mu_p)^T].
```

The symmetric covariance is eigendecomposed in float64; eigenvalues are
clamped only to `1e-12`, eigenvectors are made right-handed, scales are the
square roots of eigenvalues, and the rotation is converted deterministically
to the official scalar-first quaternion convention. SH coefficients are the
same weighted mean. Opacity uses the fixed projected-mass rule

```text
alpha_p = clip(W / A_p, 1e-4, 0.999),
```

where `A_p` is computed from parent scales. This follows the released creator's
area-weighted aggregation intent while keeping every serialized value valid.
No aggregation equation or clamp may be changed after a cut is inspected.

## Frozen selector and HSPT configuration

- source authority: official commit
  `73f547341547ae021ff036ad74fdf73a1240b450`;
- selector predicate:
  `min_distance_squared > squared_camera_distance * distance_multiplier`;
- target granularity: `2 / 512 = 0.00390625` world-to-pixel angular proxy;
- `distance_multiplier=1.0`;
- `SPT_root_volume=1.5`;
- evaluator-effective `min_SPT_size=100`;
- scale-derived bounds, bounding-sphere propagation disabled;
- released four-plane frustum test enabled;
- root global ID and skybox count: 100,000;
- SH degree 1, packed gsplat renderer, black background.

These values were selected from the analytic scale range and corrected once on the non-scientific smoke hierarchy before any performance-scale cut. The original value `0.002` produced many small SPTs; at far smoke cameras the released upper traversal stopped at ordinary upper interior nodes that the released evaluator does not materialize. The frozen value `1.5` places the two depth-1 subtrees in SPTs, so every valid camera-dependent coarse cut is handled by the unchanged released SPT predicate. This is a smoke-stage HSPT representation correction, not an LoD threshold, camera, hierarchy, or acceptance-rule change. They are not selected from
observed IDs, timing, or speedup.

## Frozen camera trajectory

There are 25 cameras. Camera `j` has centre `(0,0,r_j)` with

```text
r_j = 40 * 2^(j/3),  j=0,...,24,
```

so the path covers eight distance octaves from 40 to 10,240. Every camera
looks at the origin with fixed world-up `+y`, identical orientation, and no
jitter. Resolution is `1024 x 1024`; `fx=fy=512`, `cx=cy=511.5`. The path is
stored and evaluated in near-to-far order. The geometry fits the horizontal
and vertical field of view at the nearest camera. The scientific camera
centres, matrices, resolution, focal length, count, and order may not be
changed after cuts are observed.

For response analysis, projected scale is
`x_j = log2((fx/r_j) / median(fx/r))`; larger `x` is nearer. Selected depth is
computed only from non-skybox semantic hierarchy IDs.

## Prospective strong-LoD gate

The label `PASS-SYNTH-STRONG-LOD-CONTROL` is assigned if and only if all of the
following pass after topology, property, official-loader, renderer, and exact
official/reference selector-parity barriers:

1. every existing `criteria_v1` predicate passes: count relative range at
   least 0.05, minimum pairwise Jaccard at most 0.95, and maximum depth TVD at
   least 0.05 or a changing selected maximum depth;
2. Spearman correlation between projected scale `x` and normalized mean
   selected depth `mu/D` is at least 0.95;
3. the Theil-Sen slope of normalized mean selected depth per projected-scale
   octave is at least 0.06;
4. median normalized mean depth in the nearest camera quintile minus that in
   the farthest quintile is at least 0.40;
5. the complete trajectory span
   `(max(mu)-min(mu))/D` is at least 0.40;
6. median normalized mean depth is non-increasing from near to far in five
   consecutive equal-count distance regimes (ties allowed); and
7. the near, middle, and far representative cameras occupy at least three
   distinct rounded mean-depth bands and the near-to-far rounded mean-depth
   difference is at least `ceil(0.35*D)`.

The correlation and slope demand an ordered response; the quintile and full
span conditions prevent tiny changes around one depth band from being called
strong. The five-regime condition is robust to row-level step changes. These
are mechanism-positive thresholds, not natural-dataset quality criteria, and
do not redefine `VALID-NONTRIVIAL-LOD`.

`criteria_v1`, continuous response metrics, the ten-bin depth/count response
curve, and bootstrap diagnostics use the frozen prevalence-study definitions,
including seed 20260921 where a bootstrap is reported.

## Correctness barriers and decision-margin instrumentation

Before classification, the final artifact must pass the independent `.dhier`
parser; strict finite/property, count, binary topology, depth, reciprocal-link,
reachability, and prefix checks; the official native loader; and a smoke and
scientific renderer invocation. For all 25 cameras, the official selector is
run twice in fresh processes and must be deterministic. The independent CPU
reference must reproduce the exact ordered semantic hierarchy and raster IDs.

For every visited upper candidate and every tested SPT row, record global ID,
depth, spatial-scale term, camera-distance term, threshold term, signed margin
`lhs-rhs`, normalized margin `(lhs-rhs)/max(abs(lhs),abs(rhs),eps)`, and the
descend/select result. Reports include distributions by depth, the fraction
within normalized absolute margin 0.05, the fraction of node decisions that
change over the trajectory, parent/child extent ratios, and selected depth
versus camera scale. Instrumentation observes but never alters the predicate.

## Visual evidence

After correctness and classification, render cameras 0, 12, and 24. A compact
panel contains RGB, selected-depth histogram, semantic selected-row count,
mean depth, and median depth for each. This panel is explanatory only and is
not part of the acceptance gate.

## Frozen ALoD-EXT evaluation

ALoD-EXT is executed only if the strong-LoD gate passes. The modes are:

1. Released Evaluator (No Cache);
2. ALoD-EXT CUDA Previous Reuse;
3. ALoD-EXT CUDA Bounded Cache.

The established conservative CUDA pre-materialization filter is used without
retuning. The bounded capacity is fixed structurally before timing as

```text
min(hierarchy_rows, ceil(filtered_union_rows * 3,700,000 / 3,503,048)),
```

the same 5.62% union-headroom transfer rule as
`ALOD-EXT-PREVALENCE-ASSOCIATION-001`. It is not selected from synthetic FPS.

Each mode uses three fresh processes in a balanced cyclic schedule. Each
process renders the first ten trajectory cameras as ordered warmup, retains
reuse/cache state, then restarts all 25 cameras for measurement. CUDA is
synchronized at the established boundaries. Required per-camera barriers are
exact official selected IDs/order, zero conservative-filter false negatives,
exact CPU/CUDA filtered IDs/order, bitwise retained properties/order, exact
packed-projection survivor IDs, and numerical RGB/alpha agreement. Image
agreement is reported with mean and maximum absolute error and output-to-output
PSNR; bitwise equality is diagnostic, not required. Before the extension
correctness run, numerical agreement is frozen as mean absolute RGB and alpha
error at most `1e-6` and maximum absolute RGB and alpha error at most `0.01`.
These bounds are looser than the previously observed real-data errors but tight
enough to reject a material image change.

Report the semantic-selected to materialized to packed-survivor funnel;
materialized/semantic and packed/semantic ratios; selection, filtering,
inclusive preparation, host gather, H2D, renderer wall, nested rasterization,
and end-to-end time; FPS; and measured dynamic-property H2D bytes. Nested
columns are never summed. End-to-end is preparation wall plus renderer wall;
rasterization remains nested within renderer wall.

## Natural comparison and quality closure

After the synthetic result is frozen, the same structure/margin analysis is
applied without modification to Campus, Sci-Art, and Run009. The comparison is
descriptive and cannot establish causality from three natural hierarchies.
Synthetic results remain separate from the natural prevalence numerator and
denominator.

The Campus/Sci-Art held-out quality closure is a separate evidence package. It
uses the already frozen held-out images and cameras and reports
PSNR/SSIM/LPIPS against ground truth for Released Evaluator and bounded
ALoD-EXT, plus mean/max absolute RGB error and optional output PSNR between
the two executions. It does not delay, tune, or amend this protocol.
