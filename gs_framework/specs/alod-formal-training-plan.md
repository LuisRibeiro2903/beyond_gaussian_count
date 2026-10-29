# Formal ALoD training envelope

Frozen: 2026-08-20  
Dataset: `ALOD-REPRO-MC-MIXED-01`  
Future run: `ALOD-FORMAL-TRAIN-001`  
Training state: **not started**

## Chosen outcome

The first formal reproduction run is frozen at resolution factor 2, 30,000
coarse iterations, 100,000 fine iterations, and a 12,000,000-row fine cap.
This is the largest defensible target rather than the mathematical maximum:
it provides 7,265,745 rows of worst-case growth headroom while keeping
Run005-calibrated host and VRAM estimates below unchanged machine abort limits.

Canonical files:

```text
configs/alod/ALOD-REPRO-MC-MIXED-01-formal-target.json
environments/alod-official/formal-mixed-resource-policy.json
environments/alod-official/formal-mixed-training-command.sh
environments/alod-official/formal-mixed-provenance-template.json
```

The policy remains `execution_authorized=false`. This task does not train or
create a hierarchy.

## Source-faithful invariants and correction boundary

At pinned official commit `73f547341547ae021ff036ad74fdf73a1240b450`,
`load_from_hier` prepends skybox rows and hierarchy traversal still defaults to
global root row 100,000. The formal run therefore retains exactly 100,000
skybox rows and root row 100,000. `ALOD-DYNAMIC-SKYBOX-ROOT` is excluded.

Classic scheduled densification calls `relocate_gs`, so the Run005 relocation
bug remains reachable. The frozen
`patches/alod/ALOD-RELOCATE-TOPOLOGY-FIX.patch` is required in a disposable
execution worktree. It recomputes reciprocal parent/depth metadata beneath
forward-topology roots modified by relocation and fails on invalid child or
sibling chains. It does **not** alter camera selection, HSPT/SPT construction,
rendering, losses, optimizer updates, Gaussian attributes, densification
thresholds, spawning math, cache policy, or serialization format. The final
classification is therefore `correctness-patched local faithful public-source
reproduction`, not bit-identical unpatched-source output.

Only launcher-level build/environment compatibility, instrumentation, and this
known correctness repair are allowed. Prematerialization culling, CUDA
materialization filters, 160k/525k project caches, project selectors, and all
`ALOD-EXT-*` optimizations are excluded.

## Count and file-size model

The frozen sparse model has 2,317,128 points. Coarse prepends 100,000 skybox
rows, so its fixed population is 2,417,128. With at most `2L-1` creator nodes,
the non-skybox initial hierarchy is at most 4,634,255 rows and fine begins at
at most 4,734,255 rows after skybox prepend.

| Quantity | Target value |
|---|---:|
| Sparse / coarse total | 2,317,128 / 2,417,128 |
| Initial non-skybox / fine total maximum | 4,634,255 / 4,734,255 |
| Fine cap | 12,000,000 |
| Worst-case growth headroom | 7,265,745 rows (60.55% of cap) |
| Coarse PLY payload at 104 B/row | ~251 MB |
| Maximum degree-3 initial `.dhier` at 260 B/row | ~1.205 GB |
| Cap degree-1 final `.dhier` at 116 B/row | ~1.392 GB |

The combined worst-case core artifacts exceed the policy's 2 GiB output
warning but remain below its unchanged 5 GiB hard limit before ancillary GDF,
logs, and reports. The warning is expected and must be recorded, not bypassed.

## Host-memory model

Source arithmetic is a lower bound. Fine compact storage at SH degree 1 is 69
float32 values (276 B/row), plus 24 B/node and 4 B/classic criterion = 304
B/cap row. The target cap therefore reserves at least 3,648,000,000 bytes
(3.40 GiB) before hierarchy-rebuild temporaries, Python, cameras, renderer,
allocator fragmentation, and process overhead. The coarse lower-bound model
is 376 B/row, or 908,840,128 bytes for 2,417,128 rows.

Run005 calibration is preferred to treating those values as RSS. Run005 at a
1.1M cap measured 13,859,332,096 B peak process-group RSS, zero swap growth,
and 3,473,932,288 B peak whole-device VRAM. Its fine cap lower bound was only
334.4 MB, demonstrating the large fixed/framework and temporary component.

| Stage/envelope | Tensor/source lower bound | Planning process-RSS range | Hard safety relation |
|---|---:|---:|---|
| Coarse target | 0.85 GiB | 10–13 GiB | below 20 GiB warning |
| Conservative fine, 8M | 2.27 GiB cap rows | 15.6–18.5 GiB | below 20/24 GiB warning/hard |
| **Target fine, 12M** | **3.40 GiB cap rows** | **16.8–20.1 GiB** | warning may be approached; below 24 GiB hard estimate |
| Upper feasible, 16M | 4.53 GiB cap rows | 19.0–22.5 GiB | narrow margin; conditional only |

The ranges include the larger initial hierarchy, HSPT/SPT metadata, rebuild
temporaries, eight-worker on-demand image ingestion, allocator uncertainty,
and an OS margin. They remain planning estimates until measured. The 20M
provisional ceiling is rejected: it reserves 6.08 GB for compact rows alone
and can plausibly cross the 24 GiB process hard limit during rebuilds.

## VRAM model

The full properties and nodes remain on CPU; VRAM holds the skybox/upper cut,
active selection, current cache, gradients/optimizer state for resident rows,
image/render buffers, and PyTorch reserve. Relative to Run005's 300k-row cache,
the target 2M cache adds about 469 MB of property payload alone, plus optimizer
and selection/render temporaries.

| Envelope | Cache / reduced rows | Planning whole-device VRAM | Relation to unchanged policy |
|---|---:|---:|---|
| Conservative | 1.0M / 0.8M | 5–8 GiB | comfortable |
| **Target** | **2.0M / 1.6M** | **6–10 GiB** | leaves 2–6 GiB before 12 GiB warning |
| Upper feasible | 3.0M / 2.4M | 8–12.5 GiB | warning likely; hard margin narrow |

These are explicitly unmeasured estimates. A live GPU cache preflight is a
mandatory subsequent gate. Cache values are complete current GPU rows under
the official semantics, not a new project cache method.

## Duration model

Run005 measured about 37.8 it/s for 5k coarse iterations and 6.97 it/s for 12k
fine iterations on 741 aerial-only cameras at factor 2. The formal set has
mixed tensor sizes, about 4.8x the worst-case initial hierarchy, many more
densification events, and much larger caches. The following intentionally
broad ranges avoid false precision:

| Envelope | Coarse | Creator/cache gate | Fine | Pipeline planning range |
|---|---:|---:|---:|---:|
| Conservative | 20k; 0.3–1 h | 0.2–1.5 h | 30k; 2–5 h | 3–8 h |
| **Target** | **30k; 0.4–1.2 h** | **0.3–2 h** | **100k; 6–16 h** | **7–20 h** |
| Upper feasible | 60k; 0.8–2.5 h | 0.5–3 h | 100k full-resolution; 12–30 h | 14–36 h |

The target policy warns at 14 h fine/18 h pipeline and aborts at 20 h
fine/24 h pipeline. A timeout is a preserved partial run, not permission to
raise a machine-safety limit.

## Candidate envelopes

All use official SH1, classic densification, CPU storage, frustum culling,
100k skybox, `SPT_root_volume=150`, target granularity 2 pixels,
`min_SPT_size=256`, and `llff_hold=10`.

| Field | Conservative | **Target (frozen)** | Upper feasible |
|---|---:|---:|---:|
| Resolution factor | 4 | **2** | 1 |
| Coarse / fine iterations | 20k / 30k | **30k / 100k** | 60k / 100k |
| `cap_max` | 8M | **12M** | 16M |
| Cache / reduced | 1M / 0.8M | **2M / 1.6M** | 3M / 2.4M |
| Densify window | 3k–25.5k | **10k–85k** | 10k–85k |
| Densification interval | 750 | **1500** | 1500 |
| `densify_percent` | 1.028 | **1.028** | 1.028 |
| Vary distance multiplier | true | **true** | true |

The conservative envelope is safe but risks underdeveloping the hierarchy.
The upper envelope adds full-resolution raster cost and narrow resource margin
without evidence that additional row capacity improves the reproduction. The
target preserves the shipped small-config 30k/100k training scale and source
hierarchy parameters, adopts the MatrixCity distance variation and 0.5 LR
multiplier, provides substantial growth beyond the worst-case initial tree,
and retains practical rasterization/OS headroom. It is methodologically
preferable to maximizing headline row count.

## Frozen target configuration

Important effective fields:

```text
resolution=2 (CLI)
skybox_num=100000 (CLI)
coarse_iterations=30000
iterations=100000
cap_max=12000000
cache_size=2000000
cache_size_after_reduction=1600000
SPT_root_volume=150
target_granularity_pixels=2
min_SPT_size=256
densification=classic
densify_from_iter=10000
densify_until_iter=85000
densification_interval=1500
densify_grad_threshold=7.5e-6
densify_percent=1.028
```

The machine-readable JSON is authoritative. Unknown optimization keys must be
rejected before execution because the official loader otherwise ignores them.

## Resource policy

Policy ID: `ALOD-REPRO-MC-MIXED-01-FORMAL-RESOURCE-POLICY`.

The five-second monitor and all machine-safety thresholds are unchanged from
Run005: process RSS warns/hard-aborts at 20/24 GiB; available RAM at 8/5 GiB;
swap growth at 128/256 MiB plus the six-sample monotonic rule; whole-device
VRAM at 12/13.5 GiB; attributed process VRAM at 11.5/13 GiB; output at 2/5
GiB; and no-progress at 900/1800 seconds. Node thresholds scale to 11.4M/12M.
Wall-time budgets alone are explicitly extended for the formal schedule.

The machine inventory is 30.46 GiB RAM, 16 GiB swap, 710 GiB available disk,
and the previously verified RTX 5080 with 16,303 MiB VRAM. Live GPU access was
unavailable during this CPU-only freeze, so the execution task must remeasure
GPU state before authorization.

## Future run and command

External run root:

```text
/home/luis/tese/runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-001
```

Expected artifacts:

```text
output/scaffold/point_cloud/iteration_30000/point_cloud.ply
output/scaffold/point_cloud/iteration_30000/pc_info.txt
output/scaffold/point_cloud/iteration_30000/hierarchy.dhier
output/ALOD-REPRO-MC-MIXED-01-FORMAL-TRAIN-001.dhier
metrics/{coarse,creator,cache-preflight,fine}.jsonl
logs/events.jsonl
provenance.json
validation/{coarse,initial-hierarchy,cache-preflight,final-hierarchy}.json
```

Current safe command:

```bash
/home/luis/tese/gs_framework/environments/alod-official/formal-mixed-training-command.sh --validate-only
```

After a separate task performs the live GPU/cache preflight, creates the exact
pinned-plus-topology-fix disposable worktree, and deliberately changes
`execution_authorized` in a newly frozen policy:

```bash
/home/luis/tese/gs_framework/environments/alod-official/formal-mixed-training-command.sh --execute
```

The launcher refuses reuse of a run directory, refuses a dirty canonical
official checkout, rejects dataset/config/policy hash drift, and refuses
execution while authorization is false.

## Future acceptance gates

1. `PASS-ALOD-FORMAL-DATASET`: already satisfied by official ingestion.
2. `PASS-ALOD-FORMAL-TRAINING-PREFLIGHT`: live inventory, exact correction
   worktree, strict config, cache preflight, initial hierarchy, headroom, and
   resource-policy checks all pass.
3. `PASS-ALOD-FORMAL-DHIER`: final source-format size/count checks, finite
   properties, one non-skybox root, complete reachability, acyclicity, forward
   child/sibling consistency, reciprocal parent links, correct traversal
   depths, artifact hashes, complete provenance, and resource compliance.

The exact next bounded task is: **execute the frozen formal ALoD training run
under the approved resource policy, validate the resulting final `.dhier`, and
stop before rendering/benchmark integration**.
