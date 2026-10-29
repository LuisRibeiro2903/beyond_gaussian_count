# Formal mixed-scale ALoD SPT calibration

Date: 2026-08-20

Status: `PASS-ALOD-FORMAL-SPT-CALIBRATION`

Dataset: `ALOD-REPRO-MC-MIXED-01`

Method: dataset-specific source-faithful hyperparameter calibration

## Result

`SPT_root_volume=4.75` is frozen for the next formal fine-training attempt.
It replaces `150` and is the only methodological change in
`configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target-v2.json`. The output name
changes administratively from Run002 to Run003. Every other optimization,
HSPT, cache, and resource field is identical.

The exact pinned-source HSPT construction at `4.75` produces 22 SPTs, 889
upper rows, and 4,633,388 flattened SPT rows. Coverage is 99.9813% of the
4,634,255-row initial hierarchy. SPT sizes are minimum 281, median 4,525,
p95 1,158,563, and maximum 1,158,563 rows. The largest SPT is 25.0000% of
the hierarchy.

The frozen structural cache gate is:

```text
100,000 skybox
+     889 complete upper hierarchy
+ 1,158,563 largest complete SPT
+   100,000 static reserve
= 1,359,452 required reduced-cache rows
< 1,600,000 frozen reduced cache
= 240,548 rows headroom (15.03425%)
```

No fine training, rendering, camera-specific selection, image inspection,
quality metric, FPS measurement, or diagnostic-camera use occurred.

## Immutable inputs and provenance

| Input | Identity |
|---|---|
| Official source | `/home/luis/tese/LoDOfGaussians` at `73f547341547ae021ff036ad74fdf73a1240b450`, clean before and after calibration |
| Dataset manifest | `eaf3cb228234efd92441d28ea56482dcf4676c5a318a6d6beef1020ee27c5f7a` |
| Run002 coarse PLY | 2,417,128 rows; `e4b04b79024e751efa8b89cf381156a6d74b14fc9c2ae750c1a33634f0c51e1d` |
| Run002 initial hierarchy | 4,634,255 rows; SH degree 3; max depth 22; `c1487170a0b1e53314b70df36bc5cc8aa0da4304c5a3d21b67e3ffc3dcecadf3` |
| Run002 config | `57490a019fca1db105263ebcd2f99db9890a152cdc4ad5f40ee8bed63a76d105` |
| v2 config | `5f1709427b7d93c692669311eb709f36a29820021ef15a397831f9ac134bfc75` |

Run001, Run002, Run005, the Run005 oracle, the dataset, split, diagnostic
cameras, canonical official source, CityGaussian, extension experiments, and
historical benchmarks were not modified.

## Official-method evidence

The evidence is pinned by commit and file hash, not inferred from a later
version:

- `Docs/Hyperparameters.md:2-7` classifies `SPT_root_volume` under “Essential
  Hyperparameters,” says it must be set for the specific dataset, states that
  larger values create fewer SPTs with more Gaussians and vice versa, and asks
  users to inspect both SPT count and contained-Gaussian percentage. It gives
  at least 95% as the coverage guideline and cites about 500 SPTs for one
  H-3DGS chunk, 25k for Campus, and 60k for full MatrixCity. The pinned file
  SHA-256 is `cec03b731ee57de55b44344d768c291930f8fca37f340811d522e39f6004215a`.
- `Docs/Hyperparameters.md:17-21` defines `cache_size`, recommends a roughly
  20%-lower `cache_size_after_reduction`, and explains that `min_SPT_size`
  prevents tiny SPTs whose overhead is not worthwhile.
- `scene/gaussian_model.py:204-209` implements the strict partition predicate
  `prod(exp(log_scales)) > SPT_root_volume`; `:224-305` builds each cut-root
  subtree and admits it only when `len(SPT) > min_SPT_size`; `:307-340`
  constructs the upper hierarchy. The pinned file SHA-256 is
  `f92239232c38b30c77d0c8dfcb462177ba4187eb78548411f9c60dab8ccbcae6`.
- `configs/MatrixCity.json:38-43` contains the literal official example
  `SPT_root_volume=150`, `min_SPT_size=256`, and 25M/20M cache sizes. Its
  SHA-256 is `c3db41104e1859baf675cb2d29e5bf81cf0e60ccde25bd1d5682a63b899388d5`.

This is an officially documented dataset-specific parameter, not a project
algorithm modification.

## Why Run002 failed

The initial hierarchy root has activated scale-volume 53.108448. With the
strict official predicate, `53.108448 > 150` is false, so traversal stops at
the root and the entire 4,634,255-row hierarchy becomes one SPT. The upper
hierarchy contains only that root. The frozen Run002 gate therefore required
4,834,256 rows and exceeded the reduced cache by 3,234,256 rows.

This was a deterministic structural preflight failure before fine training,
not a convergence, training, CUDA-memory, rendering, or quality failure.

## Scene-coordinate diagnosis

The hierarchy-position bounding box is:

```text
minimum: [-9.5296297, -4.3441062, -35.1209641]
maximum: [ 2.2478590,  7.5374188,   4.5959964]
sides:   [11.7774887, 11.8815250,  39.7169604]
volume:  5,557.7740715
diagonal: 43.0965987
component range: -35.1209641 .. 7.5374188
```

Because the source compares `SPT_root_volume` directly with products of
activated scene-coordinate scales, the number is coordinate-system dependent.
The official MatrixCity JSON establishes the authors' literal value but does
not publish a root scale-volume or hierarchy bounding box from which an exact
paper/proxy spatial-scale ratio can be recovered. No unavailable paper scale
is invented here. Literal numeric portability from the paper preprocessing to
this public mixed-scale proxy is therefore not justified.

## Cache semantics and formal feasibility function

The exact trainer behavior is more dynamic than one static equation:

1. `train_hierarchy.py:180-206` begins with the skybox in GPU tensors.
2. `:252-305` derives a camera-dependent upper cut and current SPT IDs.
3. `:348-398` asks the native SPT cutter for distance-dependent SPT prefixes
   and combines new SPT rows with rendered upper rows.
4. `:415-443` starts eviction only when the *previous* `gaussian_indices`
   length exceeds `cache_size` (or at a clear interval), targets
   `cache_size_after_reduction`, and can evict only retained old cache SPTs.
   Current loaded/reused SPTs are protected.
5. `:450-452` then assembles skybox, new upper/SPT rows, reused rows, and
   surviving old-cache rows.

Consequently, the pinned trainer does not enforce a pre-load static inequality
and `reserve_rows` is a formal resource-policy term, not an upstream field.
An exact per-frame total additionally needs camera-selected SPT IDs and native
distance cuts, which are deliberately forbidden for this calibration.

The predeclared camera-independent formal gate is therefore the same
conservative *single complete largest-SPT* requirement for every candidate:

```text
R(v) = skybox_rows
     + complete_upper_hierarchy_rows(v)
     + max_complete_SPT_rows(v)
     + static_reserve_rows

accept cache iff R(v) < cache_size_after_reduction
```

It deliberately counts the complete upper hierarchy and complete largest SPT,
even though a real frame normally emits cuts of both. It is a structural
necessary/robustness gate for one complete requested SPT, not a claim that it
bounds arbitrary simultaneous camera selections or retained-cache overshoot.
No camera was used to turn it into such a claim.

## Predeclared acceptance and selection rules

Before selecting a value, the sweep encoded these deterministic criteria:

- coverage: at least 95% of unique hierarchy rows lie in admitted SPTs;
- strict cache feasibility: `R(v) < 1,600,000`;
- comfortable headroom: at least 10% of reduced cache (160,000 rows);
- non-degenerate count: at least 16 admitted SPTs;
- no dominant SPT: largest SPT at most 25% of hierarchy rows;
- no count over-partition: at most
  `floor(4,634,255 / (4 * 256)) = 4,525` SPTs;
- no size over-partition: median admitted SPT at least `4 * 256 = 1,024`
  rows;
- stability: the selected value and its immediate tested neighbors must pass
  all preceding criteria, while adjacent SPT-count and maximum-size ratios
  remain within `[0.5, 2.0]`.

The deterministic preference is the largest value inside such a stable
regime. A value on the first sharp passing transition is not “inside”: it must
have passing tested neighbors on both sides.

`min_SPT_size=256` uses the strict source rule, so admitted SPTs have at least
257 rows. The table's “rejected” column counts cut-root subtrees at or below
256 that the source expands into the upper hierarchy; they are not formed
SPTs.

The upstream status message at `train_hierarchy.py:160` prints
`len(SPT_starts)`, which includes the terminal prefix-sum sentinel; for the
selected structure it would therefore display 23 although there are 22 SPT
intervals and 22 roots. Its printed coverage also divides flattened SPT rows
by flattened-plus-upper rows, double-counting each SPT root. The selected
official-print value is 99.9808%; the requested unique-row coverage used for
acceptance is 99.9813%. Both are retained in the JSON evidence.

## Full deterministic sweep

Coverage is unique SPT rows divided by 4,634,255 hierarchy rows. Percentiles
use NumPy's linear definition. `required` and `headroom` use the formula above.
The timing column is isolated structural HSPT partition time after source-
faithful scale/subtree precomputation. Peak RSS includes the fresh process,
memory-mapped hierarchy, and precomputation. The complete machine-readable
records retain every field and candidate in
`data/alod_formal_spt_calibration_coarse.json`,
`data/alod_formal_spt_calibration_local.json`, and
`data/alod_formal_spt_calibration_tail.json`.

| volume | upper | SPTs | cov % | min | p10 | median | mean | p90 | p95 | p99 | max | max % | upper % | rejected | cache | required | headroom | ms | peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 150 | 1 | 1 | 100.0000 | 4,634,255 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255 | 100.0000 | 0.0000 | 0 | no | 4,834,256 | -3,234,256 | 0.131 | 821.5 |
| 100 | 1 | 1 | 100.0000 | 4,634,255 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255 | 100.0000 | 0.0000 | 0 | no | 4,834,256 | -3,234,256 | 0.112 | 821.7 |
| 75 | 1 | 1 | 100.0000 | 4,634,255 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255.0 | 4,634,255 | 100.0000 | 0.0000 | 0 | no | 4,834,256 | -3,234,256 | 0.112 | 821.7 |
| 50 | 15 | 8 | 99.9998 | 36,203 | 36,204.4 | 217,230.0 | 579,281.0 | 1,506,132.2 | 1,911,629.6 | 2,236,027.5 | 2,317,127 | 50.0000 | 0.0003 | 0 | no | 2,517,142 | -917,142 | 0.191 | 821.1 |
| 25 | 309 | 14 | 99.9936 | 281 | 734.8 | 27,153.0 | 330,997.1 | 984,778.4 | 1,564,060.4 | 2,166,513.7 | 2,317,127 | 50.0000 | 0.0067 | 5 | no | 2,517,436 | -917,436 | 0.312 | 821.6 |
| 10 | 311 | 15 | 99.9936 | 281 | 565.0 | 18,101.0 | 308,930.6 | 926,850.2 | 1,506,132.2 | 2,154,928.0 | 2,317,127 | 50.0000 | 0.0067 | 5 | no | 2,517,438 | -917,438 | 0.318 | 821.8 |
| 9 | 313 | 16 | 99.9936 | 281 | 565.0 | 11,313.0 | 289,622.4 | 868,922.0 | 1,448,204.0 | 2,143,342.4 | 2,317,127 | 50.0000 | 0.0068 | 6 | no | 2,517,440 | -917,440 | 0.300 | 821.5 |
| 8 | 319 | 19 | 99.9935 | 281 | 282.6 | 4,525.0 | 243,892.4 | 695,137.4 | 1,274,419.4 | 2,108,585.5 | 2,317,127 | 50.0000 | 0.0069 | 7 | no | 2,517,446 | -917,446 | 0.303 | 821.4 |
| 7 | 599 | 18 | 99.9875 | 281 | 480.4 | 4,525.0 | 257,426.3 | 753,065.6 | 1,332,347.6 | 2,120,171.1 | 2,317,127 | 50.0000 | 0.0129 | 11 | no | 2,517,726 | -917,726 | 0.314 | 821.7 |
| 6.5 | 601 | 19 | 99.9874 | 281 | 508.6 | 4,525.0 | 243,877.5 | 695,137.4 | 1,274,419.4 | 2,108,585.5 | 2,317,127 | 50.0000 | 0.0130 | 11 | no | 2,517,728 | -917,728 | 0.307 | 821.7 |
| 6 | 601 | 19 | 99.9874 | 281 | 508.6 | 4,525.0 | 243,877.5 | 695,137.4 | 1,274,419.4 | 2,108,585.5 | 2,317,127 | 50.0000 | 0.0130 | 12 | no | 2,517,728 | -917,728 | 0.414 | 821.8 |
| 5.75 | 601 | 19 | 99.9874 | 281 | 508.6 | 4,525.0 | 243,877.5 | 695,137.4 | 1,274,419.4 | 2,108,585.5 | 2,317,127 | 50.0000 | 0.0130 | 12 | no | 2,517,728 | -917,728 | 0.317 | 821.8 |
| 5.5 | 603 | 20 | 99.9874 | 281 | 282.8 | 4,525.0 | 231,683.6 | 637,209.2 | 1,216,491.2 | 2,096,999.8 | 2,317,127 | 50.0000 | 0.0130 | 12 | no | 2,517,730 | -917,730 | 0.321 | 821.9 |
| 5.25 | 605 | 21 | 99.9874 | 281 | 283.0 | 4,525.0 | 220,651.0 | 579,281.0 | 1,158,563.0 | 2,085,414.2 | 2,317,127 | 50.0000 | 0.0131 | 12 | no | 2,517,732 | -917,732 | 0.309 | 821.5 |
| 5 | 607 | 22 | 99.9874 | 281 | 283.0 | 4,525.0 | 210,621.4 | 1,100,634.8 | 1,158,563.0 | 1,158,563.0 | 1,158,563 | 25.0000 | 0.0131 | 13 | yes | 1,359,170 | 240,830 | 0.362 | 821.8 |
| **4.75** | **889** | **22** | **99.9813** | **281** | **311.2** | **4,525.0** | **210,608.5** | **1,100,634.8** | **1,158,563.0** | **1,158,563.0** | **1,158,563** | **25.0000** | **0.0192** | **15** | **yes** | **1,359,452** | **240,548** | **0.315** | **821.8** |
| 4.5 | 1,173 | 22 | 99.9752 | 281 | 565.0 | 3,394.0 | 210,595.6 | 1,100,634.8 | 1,158,563.0 | 1,158,563.0 | 1,158,563 | 25.0000 | 0.0253 | 17 | yes | 1,359,736 | 240,264 | 0.325 | 821.9 |
| 4 | 1,179 | 25 | 99.9751 | 281 | 283.0 | 2,263.0 | 185,324.0 | 926,850.2 | 1,158,563.0 | 1,158,563.0 | 1,158,563 | 25.0000 | 0.0254 | 19 | yes | 1,359,742 | 240,258 | 0.431 | 821.8 |
| 3.5 | 1,183 | 27 | 99.9751 | 281 | 283.0 | 2,263.0 | 171,596.3 | 810,993.8 | 1,158,563.0 | 1,158,563.0 | 1,158,563 | 25.0000 | 0.0255 | 19 | yes | 1,359,746 | 240,254 | 0.325 | 821.7 |
| 2 | 2,049 | 36 | 99.9566 | 281 | 282.0 | 1,696.0 | 128,673.4 | 579,281.0 | 724,101.5 | 1,158,563.0 | 1,158,563 | 25.0000 | 0.0442 | 35 | yes | 1,360,612 | 239,388 | 0.335 | 821.6 |
| 1 | 3,499 | 53 | 99.9256 | 281 | 281.0 | 1,131.0 | 87,373.8 | 289,641.0 | 579,281.0 | 857,336.4 | 1,158,563 | 25.0000 | 0.0755 | 66 | yes | 1,362,062 | 237,938 | 0.346 | 821.8 |
| 0.5 | 6,943 | 81 | 99.8519 | 281 | 281.0 | 1,129.0 | 57,128.3 | 289,639.0 | 289,641.0 | 347,569.0 | 579,281 | 12.5000 | 0.1498 | 138 | yes | 786,224 | 813,776 | 0.365 | 821.6 |
| 0.25 | 13,537 | 125 | 99.7106 | 281 | 281.0 | 1,131.0 | 36,966.7 | 144,819.0 | 144,821.0 | 289,640.5 | 579,281 | 12.5000 | 0.2921 | 300 | yes | 792,818 | 807,182 | 0.396 | 821.3 |
| 0.1 | 27,271 | 199 | 99.4158 | 281 | 281.0 | 2,263.0 | 23,151.7 | 72,409.0 | 72,409.0 | 144,819.0 | 289,639 | 6.2500 | 0.5885 | 782 | yes | 516,910 | 1,083,090 | 0.449 | 821.8 |
| 0.05 | 43,657 | 329 | 99.0650 | 281 | 281.0 | 4,523.0 | 13,954.2 | 36,205.0 | 72,409.0 | 72,410.4 | 144,819 | 3.1250 | 0.9421 | 1,482 | yes | 388,476 | 1,211,524 | 0.504 | 822.0 |
| 0.025 | 66,963 | 527 | 98.5664 | 281 | 281.0 | 2,263.0 | 8,667.6 | 25,343.0 | 36,204.4 | 72,409.0 | 144,819 | 3.1250 | 1.4450 | 2,388 | yes | 411,782 | 1,188,218 | 0.761 | 821.7 |
| 0.01 | 108,857 | 961 | 97.6718 | 281 | 281.0 | 2,261.0 | 4,710.1 | 9,051.0 | 18,101.0 | 36,203.8 | 72,411 | 1.5625 | 2.3490 | 4,214 | yes | 381,268 | 1,218,732 | 0.901 | 821.6 |
| 0.005 | 154,925 | 1,493 | 96.6892 | 281 | 281.0 | 1,131.0 | 3,001.2 | 9,049.0 | 9,051.0 | 18,101.0 | 36,205 | 0.7812 | 3.3430 | 5,541 | yes | 391,130 | 1,208,870 | 1.135 | 821.8 |
| 0.002 | 253,591 | 2,561 | 94.5832 | 281 | 281.0 | 1,129.0 | 1,711.5 | 4,525.0 | 4,525.0 | 9,051.0 | 18,103 | 0.3906 | 5.4721 | 7,232 | yes | 471,694 | 1,128,306 | 1.416 | 821.6 |
| 0.001 | 390,749 | 3,540 | 91.6446 | 281 | 281.0 | 565.0 | 1,199.7 | 2,263.0 | 4,525.0 | 9,049.0 | 18,101 | 0.3906 | 8.4318 | 9,583 | yes | 608,850 | 991,150 | 1.907 | 821.9 |

The high-value failure boundary is cache/dominance: 5.25 still has a
2,317,127-row maximum. The low-value boundary is official coverage: 0.002
drops to 94.5832%; 0.001 also drops the median below 1,024 rows. Inconvenient
and failing candidates are retained.

## Selection and sensitivity

`5.0` is the first feasible point, but it lies directly on the sharp
5.25→5.0 transition where the largest SPT halves. Under the predeclared
stability rule it is a boundary, not the interior choice. `4.75` has passing
neighbors on both sides and minimizes unnecessary further partitioning inside
that stable neighborhood.

| value | SPTs | coverage | median | p95 | max | upper | required | headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 (larger) | 22 | 99.9874% | 4,525 | 1,158,563 | 1,158,563 | 607 | 1,359,170 | 240,830 |
| **4.75 (selected)** | **22** | **99.9813%** | **4,525** | **1,158,563** | **1,158,563** | **889** | **1,359,452** | **240,548** |
| 4.5 (smaller) | 22 | 99.9752% | 3,394 | 1,158,563 | 1,158,563 | 1,173 | 1,359,736 | 240,264 |

Exact pinned-source builds, using the real Run002 hierarchy, real scaffold,
frozen two-pixel target (`2 / 2317.6449482429634`), and
`min_SPT_size=256`, match every count and size above:

| value | official HSPT wall | process peak RSS | PyTorch peak reserved |
|---:|---:|---:|---:|
| 5.0 | 0.498 s | 5,550,051,328 B | 182,452,224 B |
| **4.75** | **0.488 s** | **5,554,540,544 B** | **182,452,224 B** |
| 4.5 | 0.442 s | 5,549,928,448 B | 182,452,224 B |

These runs called the official `GaussianModel.build_hierarchical_SPT` and did
not enter selection, training, or rendering. Reports are
`data/alod_formal_spt_official_5.json`,
`data/alod_formal_spt_official_4_75.json`, and
`data/alod_formal_spt_official_4_5.json`.

The selected count is intentionally not forced to 60k. This proxy has only a
4.63M-row initial hierarchy, a different public camera mixture and
preprocessing, and a compact coordinate extent. Even the authors describe
count as proportional rather than fixed. The sweep does include 527 SPTs at
0.025, comparable in order to the one-chunk example, but selecting it merely
to match that example would violate the predeclared largest-stable-value rule
and add 66,074 upper rows relative to 4.75 without a cache need.

## Resource/cache preflight

The authorized Run002 resource policy remains byte-for-byte unchanged
(`db807e03ef0f9a793ca012de10e8856868487e2faab2a471a4bcac9eaeff3fac`).
The selected HSPT passes the 1.6M reduced-cache gate and also remains below the
2.0M cache threshold.

Planning checks remain within all frozen hard limits:

| Constraint | Unchanged limit | Evidence/plan | Result |
|---|---:|---:|---|
| Process RSS | 24 GiB | prior fine plan 16.8–20.1 GiB; 3.9 GiB upper-plan margin; exact selected HSPT peak 5.173 GiB | pass-plan |
| Available RAM | 5 GiB floor | 30.46 GiB inventory minus 20.1 GiB upper process estimate = 10.36 GiB before other-system variation | pass-plan; monitor authoritative |
| Swap growth | 256 MiB abort | unchanged monitor and termination policy; no new persistent allocation class | pass-policy |
| Whole-device VRAM | 13.5 GiB | prior fine plan 6–10 GiB; 3.5 GiB upper-plan margin; HSPT PyTorch reserve 174 MiB | pass-plan |
| Output | 5 GiB | Run002 output plus maximum degree-1 12M-row final `.dhier` = 2,913,748,080 B (2.714 GiB) | pass-plan |

Fine-stage RSS/VRAM values are still planning estimates because this task is
forbidden from starting fine training. The unchanged formal monitor remains
the execution authority for the 24 GiB RSS, 5 GiB available-RAM, 256 MiB swap,
13.5 GiB whole-device VRAM, and 5 GiB output aborts. Calibration neither
relaxes nor bypasses them.

## Reuse boundary

Reuse of the validated Run002 coarse PLY and initial `.dhier` is
source-faithful:

- `train.py:54-63` loads one config namespace.
- `train.py:70-83` passes it to coarse training, but `train_scaffold.py`
  contains no `SPT_root_volume` reference.
- `train.py:102-113` invokes `GaussianHierarchyCreator` only with artifact and
  directory paths; the creator has no SPT-volume input.
- `train.py:64-68` and `:99-101` explicitly support reusing an existing coarse
  iteration and hierarchy through `--skip_if_exists`.
- The first consumption of `SPT_root_volume` is fine-stage HSPT construction
  at `train_hierarchy.py:159`; it is consumed again on later HSPT rebuilds at
  `:800`.

Therefore changing only `SPT_root_volume` cannot change the already completed
coarse optimization or initial hierarchy creation. The next formal run should
treat the two frozen Run002 hashes above as immutable inputs and execute the
fine stage only; rerunning 30k coarse iterations would add variation without
methodological benefit.

## Limitations

- This is pre-training calibration on the validated initial hierarchy. Fine
  densification changes the hierarchy, and the source rebuilds HSPT metadata.
  The initial distribution does not prove every later distribution or every
  camera-dependent retained-cache total.
- At `train_hierarchy.py:800`, the later rebuild omits the configured
  `min_SPT_size`; the Python default in `gaussian_model.py:204` is 100. The
  initial build correctly uses frozen 256. This pinned-source behavior is
  documented, not patched or hidden. It must be watched during formal fine
  training.
- The formal cache formula is intentionally camera-independent. It proves
  space for the complete upper hierarchy, one complete largest SPT, skybox,
  and reserve. It does not use or predict the sixteen protected diagnostic
  cameras and does not claim to bound arbitrary simultaneous selected SPTs.
- Exact paper MatrixCity spatial statistics and preprocessing scale were not
  available. This remains a public-source faithful ALoD reproduction on a
  mixed-scale MatrixCity proxy with explicit dataset-specific calibration,
  not an exact or bit-identical paper MatrixCity reproduction.

## Frozen artifacts and next task

The complete freeze record is
`data/alod_formal_spt_calibration_freeze.json`. The new formal target is
`configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target-v2.json`.

The next bounded task is to execute formal fine hierarchical training using
`SPT_root_volume=4.75` and the frozen Run002 coarse/hierarchy inputs, then
validate the final `.dhier` before any rendering or oracle generation.
