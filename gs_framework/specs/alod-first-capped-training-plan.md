# A LoD of Gaussians — First Capped Official Training Plan

Generated: 2026-08-06

Current decision: **PASS-ALOD-FIRST-DHIER** for `ALOD-SMOKE-TRAIN-005`.
Run004 remains frozen as `FAILED-ALOD-FINAL-BARRIER`; its historical evidence
below is intentionally preserved. Run005 reused the accepted Run004 coarse PLY
and initial degree-3 hierarchy, applied the project-owned
`ALOD-RELOCATE-TOPOLOGY-FIX` in a disposable worktree, completed all scheduled
fine mutations, and produced a structurally valid 1,099,999-row degree-1 final
hierarchy. Its SHA-256 is
`6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c`.
The fidelity classification is **correctness-patched local reproduction; not
bit-identical pinned official-source reproduction**.

2026-08-07 inspector addendum: the standalone project parser independently
validated exact size arithmetic, numeric fields, skybox sentinels, all 999,998
forward edges, reverse parents, depths, ownership, cycles, and reachability.
It also rediscovered the rejected Run004 final's 4,355 depth mismatches, one
reverse-parent mismatch, and corrupted skybox row zero. Raw native-loader and
Python model-load parity both pass exactly for the degree-3 initial artifact
and Run005 final artifact.

## 1. Scope

This document records the frozen plan and outcome of the first official-source, resource-adapted A LoD of Gaussians smoke training at commit `73f547341547ae021ff036ad74fdf73a1240b450`. It attempted to prove complete coarse training, creator serialization, capped fine training, and final `.dhier` validity on the local RTX 5080 / 30 GiB host. It is not a paper-scale reproduction, performance baseline, formal-quality reproduction, city-scale claim, or method comparison.

The first authorized attempt, `ALOD-SMOKE-TRAIN-001`, failed before optimization when DataLoader IPC selected Windows/9p temporary storage. The second, `ALOD-SMOKE-TRAIN-002`, cleared that fault but failed at gsplat's first lazy JIT because it selected `/usr/bin/nvcc` 12.0 for `compute_120`. The third, `ALOD-SMOKE-TRAIN-003`, cleared both issues and completed the first render forward/backward, then failed before its first optimizer update because pinned `OurAdam` calls a PyTorch method renamed in 2.11. Run 004 cleared all three compatibility faults and completed training, but Barrier 3 found inconsistent hierarchy metadata. All four runs are retained immutably. Diagnostic work did not alter the canonical official source, dataset, optimization configuration, resource policy, or either Conda environment.

## 2. Dataset and contamination restriction

The frozen asset is:

```text
ID: ALOD-ORACLE-MC-AERIAL-TEST-01
Path: /home/luis/tese/data/alod/ALOD-ORACLE-MC-AERIAL-TEST-01
Images: 741 RGBA PNG, 1920 x 1080
Sparse points: 443,620
ALoD local oracle split: 726 train / 15 evaluation
Camera: one PINHOLE, fixed focal length
```

> **Formal-evaluation contamination warning:** This asset is derived from MatrixCity's original aerial test reconstruction. Because 726 of its cameras are repurposed for ALoD training, it must not be used as an untouched MatrixCity test set or compared directly with formal results that preserve the original MatrixCity train/test protocol, including the completed CityGaussian results.

The loader forces `eval=True`; the resolved `llff_hold=50` uses zero-based stable image-name order, and `train_test_exp=false` excludes every 50th camera from training. The image-resolution factor is 2, producing 960 x 540 tensors. `CameraDataset` forces CPU image storage and loads samples on demand through eight persistent workers.

## 2.1 DataLoader interoperability remediation

The root cause is environmental: the launching shell inherited `TEMP` and `TMP` as `/mnt/c/Users/LUISRI~1/AppData/Local/Temp`, a Windows-backed 9p mount where WSL returns `errno 95` when Python multiprocessing tries to bind its AF_UNIX resource-sharer listener. Direct probes reproduced the same failure there and passed on native ext4 `/tmp`.

Resolution A is selected and the fallback ladder stops there:

- create or reuse `/tmp/alod-multiprocessing-1000`, require uid 1000 ownership and mode `0700`, and reject 9p/DrvFS;
- export `TMPDIR`, `TEMP`, and `TMP` to that directory before validation or training;
- bind/connect/accept/clean up a unique AF_UNIX socket before launch;
- retain the official `num_workers=8`, `prefetch_factor=1`, `persistent_workers=True`, `fork` start method, and PyTorch `file_descriptor` sharing strategy.

The exact official CameraDataset/DataLoader path passed a CPU-only three-batch parity smoke against `num_workers=0`, including camera identity and RGB/alpha checksums, with CUDA uninitialized and all workers cleaned up. Therefore Resolution B (switch sharing strategy) and Resolution C (`num_workers=0`) were not attempted. Evidence is frozen at `/home/luis/tese/runs/alod/diagnostics/ALOD-DATALOADER-DIAG-001/diagnosis.json` (SHA-256 `d5e0090e3364b707fb26a418a97138922be9ea44560d571c0ddeefcec3f350bc`).

## 2.2 CUDA toolchain selection remediation

Directly invoking the environment's Python does not activate Conda's compiler variables. Run 002 therefore let `torch.utils.cpp_extension` detect `CUDA_HOME=/usr`; gsplat 1.5.3 invoked `/usr/bin/nvcc` 12.0.140 and failed its lazy CUDA build with `Unsupported gpu architecture 'compute_120'`.

The launcher now selects the same already-installed toolchain used for the frozen environment build:

- `CUDA_HOME=/home/luis/miniconda3/envs/alod_official` and its `nvcc` 12.8.93;
- the existing Conda C/C++ compilers, include paths, and runtime library paths;
- `TORCH_CUDA_ARCH_LIST=12.0` and `MAX_JOBS=2`, matching the environment build;
- private native-ext4 `TORCH_EXTENSIONS_DIR=/tmp/alod-torch-extensions-1000`, uid-owned and mode `0700`.

A four-Gaussian packed gsplat smoke compiled the extension for the RTX 5080, completed forward and backward, and produced finite renders, alpha, and all tested gradients. The cached `gsplat_cuda.so` hash is `e35f7fc8680ea40725521ef76aec2630bb5317a20f392c7b8a99942894e11f57`. This is launcher-only toolchain selection: no source or package was changed. Evidence is frozen at `/home/luis/tese/runs/alod/diagnostics/ALOD-GSPLAT-JIT-DIAG-001/diagnosis.json` (SHA-256 `c170baef22b1b1ee3410219606ef36ce078d87f52f42791a18d66bb10604d26f`).

## 2.3 Pinned OurAdam / PyTorch 2.11 compatibility

Pinned `scene.OurAdam.Adam.step` calls the inherited `_cuda_graph_capture_health_check()`. Frozen PyTorch 2.11 generalized that method to `_accelerator_graph_capture_health_check()` and upstream `torch.optim.Adam.step` now calls the new name. The launcher asserts that the old name is absent and the new name exists, then installs the old name as a direct alias to the new inherited method before running either official entrypoint.

This does not replace `OurAdam`, its parameter filtering, state, update function, or mathematics. A bounded CUDA smoke reproduced the unaliased `AttributeError`, then passed both a full custom-Adam step and a selected-row step with finite changed parameters. No source or package was modified. Evidence is frozen at `/home/luis/tese/runs/alod/diagnostics/ALOD-OURADAM-COMPAT-DIAG-001/diagnosis.json` (SHA-256 `22faa37fabbceeae517c986c9394443c0e49b47a5ba191539f5185b1da789988`).

## 2.4 Run 004 final-barrier diagnosis

Run 004 completed all three official stages. Coarse produced 543,620 finite rows, the creator produced a valid 887,239-node degree-3 hierarchy, cache preflight passed, and fine training serialized 1,099,999 degree-1 Gaussian/node rows. Peak process-group RSS was 13,561,638,912 bytes, observed swap growth was zero, and no resource-policy threshold fired.

The monitor first stopped because its skybox predicate incorrectly required every metadata column to remain `-99`; official loading immediately zeroes column 5. An independent CPU-only parser then applied a source-faithful skybox exclusion and traversed the full non-skybox hierarchy. Forward child/sibling topology was complete and all 999,999 non-skybox rows were reachable, but reverse/depth metadata was not valid: 4,355 child depths disagreed with traversal depth and child 1,099,901 stored parent 916,150 instead of traversed parent 959,013. The frozen Barrier 3 explicitly requires parent/depth consistency, so the validator was not relaxed.

Evidence is frozen at `/home/luis/tese/runs/alod/diagnostics/ALOD-FINAL-BARRIER-DIAG-001/diagnosis.json` and the run-local `failure/final-barrier-failure.json`. Repairing official relocation logic, post-processing the serialized hierarchy, or accepting topology-only validity would change a protected hierarchy-semantic boundary and requires a new explicit decision.

## 3. Official pipeline and stage barriers

```text
frozen COLMAP scene
  -> Stage 1: official train_scaffold.py (5,000 iterations)
  -> coarse degree-1 GraphDECO PLY + pc_info.txt
  -> Barrier 1: layout/count/finiteness/hash
  -> Stage 2: external validated GaussianHierarchyCreator
  -> initial non-skybox degree-3 hierarchy.dhier
  -> Barrier 2: byte/count/tree/finiteness/headroom/hash
  -> Stage 2.5: exact official HSPT rebuild and cache preflight
  -> Stage 3: seeded runpy -> official train.py --skip_if_exists
  -> final degree-1 final.dhier
  -> Barrier 3: byte/count/tree/finiteness/resource/hash
```

Official output locations under the frozen Run 004 directory:

| Artifact | Path relative to run directory |
|---|---|
| Official output root | `output/` |
| Coarse diagnostic save | `output/scaffold/point_cloud/iteration_2500/` |
| Coarse creator input | `output/scaffold/point_cloud/iteration_5000/point_cloud.ply` |
| Skybox metadata | `output/scaffold/point_cloud/iteration_5000/pc_info.txt` |
| Initial hierarchy | `output/scaffold/point_cloud/iteration_5000/hierarchy.dhier` |
| Creator diagnostic graph | `output/scaffold/point_cloud/iteration_5000/hierarchy.gdf` |
| Final hierarchy | `output/final.dhier` |

The creator filters non-skybox PLY rows with infinite/NaN opacity, scale components above 10, NaN scale/rotation/position/SH, or exactly zero opacity. It then builds a point-based kd-tree, merges parents, aligns rotations, and writes a dynamic hierarchy. The creator excludes the first `pc_info.txt` rows and hard-codes degree 3 for the initial `.dhier`. Fine training prepends the saved skybox and serializes its final hierarchy at configured degree 1.

## 4. Configuration precedence and strict validation

`train.py` constructs and parses `ModelParams` and `PipelineParams` first. It opens `configs/{args.config}`, constructs `OptimizationParams`, and extracts only keys present in that namespace. Consequently:

1. model, dataset, pipeline, and orchestration values come from CLI, source defaults, or later source assignments;
2. the JSON applies only to `OptimizationParams`;
3. optimization CLI overrides are unavailable through `train.py` because those arguments are not registered at its initial `parse_args` call;
4. omitted JSON keys retain constructor defaults;
5. unknown JSON keys are silently ignored by official extraction; and
6. `train.py` overwrites source/model/images/hierarchy/scaffold/output paths at runtime.

The project validator therefore parses `OptimizationParams.__init__` with Python AST, rejects duplicate and unknown JSON keys, checks every accepted field is represented in the resolved manifest, compares JSON values with resolved values, and rejects intentionally omitted unused/overwritten keys if they appear. JSON parse success alone is not an acceptance signal.

The command runs from `/home/luis/tese/alod_official_worktree_73f5473`. The official parser receives:

```text
configs/../../gs_framework/configs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01-capped.json
```

The command resolves that path with `realpath`, requires equality with the tracked thesis config, and verifies SHA-256 before launch. It does not add a file or symlink to the official checkout.

## 5. Configuration audit

Fidelity labels:

- `OFFICIAL-SMALL`: shipped `small.json` value preserved;
- `SOURCE-DEFAULT`: omitted and inherited from the constructor;
- `RESOURCE-ADAPTED`: deliberately reduced/changed for this machine;
- `PIPELINE-REQUIRED`: required for the frozen asset or three-stage orchestration;
- `DISABLED`: intentionally false;
- `PARSED-UNUSED` or `RUNTIME-OVERRIDDEN`: not controlled as its name suggests.

The canonical machine-readable account, including origin and source reference for every effective field, is `configs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01-resolved.json`.

### 5.1 Model, dataset, and pipeline fields

| Field | Default | Chosen/effective | Source and stage | Effect / fidelity |
|---|---:|---:|---|---|
| `source_path` | `""` | frozen dataset | `train.py` assignment; both | CLI then runtime override; pipeline-required |
| `model_path` | `""` | `output/scaffold` | `train.py`; both | output placement; runtime-overridden |
| `images` | `images` | absolute frozen link | `train.py`; both | prevents ambiguous path; pipeline-required |
| `alpha_masks` | `masks` | empty | `ModelParams`; both | use embedded RGBA alpha only; pipeline-required |
| `depths` | `depths` | empty | `ModelParams`; both | no depth supervision; pipeline-required |
| `resolution` | -1 | 2 | `loadCam`; both | 960 x 540; resource-adapted |
| `white_background` | false | false | model default; both | coarse still replaces background with random RGB; source-default |
| `train_test_exp` | false | false | model default/split | preserves held-out set; disabled |
| `data_device` | `cuda` | `cpu` | `CameraDataset`; both | forced runtime override, on-demand CPU images |
| `eval` | false | true | CLI then reader force; both | enables local split; pipeline-required/runtime-overridden |
| `skybox_num` | 100,000 | 100,000 | model CLI; coarse | pinned root invariant; pipeline-required |
| `skybox_locked` | false | false | model default; coarse | gradients are nevertheless explicitly frozen for skybox; source-default |
| `pretrained`, `bounds_file` | empty | empty | model defaults; coarse | inactive |
| `hierarchy`, `scaffold_file` | empty | initial artifact paths | `train.py`; fine | runtime-overridden |
| `anti_aliasing` | false | false | model default; fine render | disabled; omitted from optimization JSON |
| model `sh_degree` | 1 | 1 | model default | coarse model is explicitly constructed at degree 1 |
| `convert_SHs_python` | false | false | pipeline default | source-default |
| `compute_cov3D_python` | false | false | pipeline default | source-default |
| `debug` | false | false | pipeline default | disabled |

### 5.2 Optimization fields

| Key | Default / `small.json` | Capped value | Stage and semantics | Fidelity |
|---|---|---:|---|---|
| `iterations` | 100k / 100k | 12,000 | fine loop and final save | resource-adapted |
| `coarse_iterations` | 60k / 30k | 5,000 | coarse loop/final save | resource-adapted |
| `SH_degree` | 1 / 1 | 1 | fine properties/final serialization | official-small |
| `SH_increase_after_train_percent` | 0.1 / omitted | 0.1 | no consumer found | parsed-unused, omitted JSON |
| `lr_multiplier` | 1 / 1 | 1 | fine parameter LRs | official-small |
| `position_lr_init` | 2e-5 / 2e-5 | 2e-5 | coarse/fine xyz schedule; coarse xyz later frozen | official-small |
| `position_lr_final` | 2e-7 / 2e-7 | 2e-7 | position schedule | official-small |
| `position_lr_delay_mult` | .01 / .01 | .01 | position schedule | official-small |
| `position_lr_max_steps` | 100k / 125k | 125k | schedule horizon | official-small |
| `feature_lr` | .0025 / .0025 | .0025 | DC/rest features | official-small |
| `opacity_lr` | .05 / .05 | .05 | opacity | official-small |
| `scaling_lr` | .005 / .005 | .005 | scale | official-small |
| `rotation_lr` | .001 / .001 | .001 | rotation | official-small |
| exposure LR fields | source = small | source defaults | coarse constructs scheduler; fine inactive because optimization disabled | omitted JSON |
| `percent_dense` | .01 / .02 | .01 | assigned, but relevant split predicates are commented | parsed-effectively-unused, omitted JSON |
| `lambda_dssim` | .2 / .2 | .2 | L1/SSIM mixture | official-small |
| `densification_interval` | 300 / 1500 | 2,000 | strict scheduled test | resource-adapted |
| `densification` | MCMC / classic | classic | criterion, spawning, relocation | official-small |
| `opacity_reset_interval` | 1e11 / 1e11 | source default | no training-loop consumer | parsed-unused, omitted JSON |
| `densify_from_iter` | 100 / 10k | 1,000 | strict lower window | resource-adapted |
| `densify_until_iter` | default dependent / 220k | 10,000 | strict upper window | resource-adapted |
| `densify_grad_threshold` | .0015 / 7.5e-6 | 7.5e-6 | classic spawn threshold | official-small |
| depth weight fields | 1/.01 | hard-coded 1/.01 | fine builds hard-coded schedule; no depths present | runtime-overridden, omitted JSON |
| `llff_hold` | -1 / huge | 50 | zero-based name-sorted evaluation split | pipeline-required |
| `vary_distance_multiplier` | false / false | false | disables random distance inflation | official-small/disabled |
| `output_file_name` | `result.dhier` / `_small.dhier` | `final.dhier` | final output | pipeline-required |
| `noise_lr` | constructor branch anomaly / 50k | 50k | inactive for classic path | official-small |
| `lambda_scaling` | constructor branch / .1 | .1 | fine regularizer | official-small |
| `lambda_opacity` | constructor branch / .1 | .1 | fine regularizer | official-small |
| `densify_percent` | 1.02 / 1.028 | 1.028 | per-event growth request | official-small |
| `cap_max` | 12M / 15M | 1.1M | total CPU allocation and source spawning cap, including skybox | resource-adapted |
| `graph_view_select` | false / false | false | ordinary shuffled cameras | disabled |
| `view_graph_k` | 100 / 100 | source default | graph path inactive | omitted JSON |
| `use_bounding_spheres` | false / false | false | scale-derived bounds used | official-small/disabled |
| `use_frustum_culling` | true / true | true | selection cull | official-small |
| `use_occlusion_culling` | false / omitted | false | training uses module global instead of field | parsed-unused, omitted JSON |
| `prune_unused` | false / omitted | false | no contribution bitmap/prune | explicitly disabled |
| `dampen_scale_grad` | false / omitted | false | no scale-gradient dampening | explicitly disabled |
| `storage_device` | CPU / CPU | CPU | full property/node arrays in host RAM | official-small |
| `SPT_root_volume` | 10 / 150 | 150 | HSPT upper cut scale-volume threshold | official-small |
| `target_granularity_pixels` | 2 / 2 | 2 | divided by original COLMAP focal | official-small |
| `min_SPT_size` | 256 / 256 | 256 | an SPT is retained only when total size is strictly greater than 256 | official-small |
| `use_GPU_caching` | true / true | true | reuse/write-back cache | official-small |
| `cache_size` | 15M / 25M | 300,000 rows | reduction trigger counts complete current GPU rows | resource-adapted |
| `cache_size_after_reduction` | 12M / 20M | 240,000 rows | reduction target | resource-adapted |
| `clear_cache_interval` | 1,000 / 1,000 | 1,000 | periodic retained-SPT eviction | official-small |
| `optimize_exposure` | false / omitted | false | exposure not used by fine renderer | explicitly disabled |

Coarse training does not call densification, pruning, opacity reset, or a Gaussian cap. Its population remains the initialized sparse points plus skybox. It always constructs `GaussianModel(1)`, freezes xyz learning rate to zero, uses a random background every iteration, and saves the creator-compatible degree-1 PLY.

## 6. Pinned source invariant

```text
PINNED SOURCE INVARIANT

At official commit 73f547341547ae021ff036ad74fdf73a1240b450,
the hierarchy cut begins at global row 100000.

Because skybox rows are prepended to hierarchy rows, the source-faithful
configuration must use exactly 100000 skybox rows.

Using another skybox count without patching the hierarchy-root logic would
make traversal begin at an unrelated hierarchy node and invalidate selection
and training behavior.
```

Source proof:

- `scene/gaussian_model.py:create_from_hier`, lines 1117–1138, prepends skybox attributes and node sentinels and offsets tree links;
- `scene/gaussian_model.py:cut_hierarchy_on_condition`, line 398, defaults `root_node=100000`; and
- `scene/gaussian_model.py:build_hierarchical_SPT`, line 209, invokes that function without overriding the root.

Deferred issue `ALOD-DYNAMIC-SKYBOX-ROOT`: replace the hard-coded root with `self.skybox_points`. Classification: project correction/generalization, not part of initial source-faithful reproduction. It is not implemented here.

## 7. Count model

The creator receives 543,620 PLY rows but skips the first 100,000 using `pc_info.txt`. Filtering can reduce, never increase, the 443,620 candidate leaves. For a full binary tree, at most `2L-1` non-skybox nodes are written.

| Quantity | Planning value |
|---|---:|
| Sparse/coarse scene points | 443,620 |
| Skybox rows | 100,000 |
| Coarse total | 543,620 |
| Creator leaves | at most 443,620 |
| Initial non-skybox hierarchy | at most 887,239 |
| Fine initial total after skybox prepend | at most 987,239 |
| Fine `cap_max` | 1,100,000 |
| Nominal minimum growth headroom | 112,761 |
| Nominal headroom / cap | about 10.25% |

Barrier 2 reads the actual initial file and requires: `G==N`, total after adding 100k skybox no greater than 1,050,000, total below cap, at least 50,000 growth rows, at least 5% headroom, finite attributes, one root, in-range links, correct child chains/depths, no cycle, and full reachability. It reports `(cap-initial_total)/cap`. The cap is never raised automatically.

The 1.05M warning is preflight-enforced before fine and log-derived diagnostic during fine. The external monitor cannot reliably hard-stop on current node count because the official progress line is not a machine-readable contract. The 1.1M `cap_max` is the configuration-enforced hard safeguard in allocation and spawning.

## 8. Memory and storage model

All figures below are planning arithmetic, not observed peaks.

### 8.1 Source-format arithmetic

- Coarse degree-1 parameter row: 23 float32 = 92 bytes.
- Coarse PLY vertex row: 26 float32 including normals = 104 bytes.
- Initial creator `.dhier` degree 3: 260 bytes/node when `G==N`, plus 12 header/count bytes.
- Final fine `.dhier` degree 1: 116 bytes/node when `G==N`, plus 12 bytes.

Derived storage:

| Artifact | Arithmetic |
|---|---:|
| Coarse PLY vertex payload | 56,536,480 B plus small PLY header |
| Initial degree-3 `.dhier`, maximum | 230,682,152 B |
| Final degree-1 `.dhier`, minimum initial count | 114,519,736 B |
| Final degree-1 `.dhier`, cap | 127,600,012 B |

The 2,500 diagnostic PLY may add another approximately 54 MiB. Input/camera copies, the creator GDF, logs, reports, and allocator/checkpoint overhead are additional. Output warning/hard limits remain 2/5 GiB.

### 8.2 Source-derived allocation lower bounds

Coarse, 543,620 rows:

- parameters: 50,013,040 B;
- gradients: 50,013,040 B;
- Adam first/second moments: 100,026,080 B;
- xyz gradient/denominator accumulators: 4,348,960 B;
- subtotal: 204,401,120 B, excluding exposures, images, rasterizer temporaries, CUDA context, allocator fragmentation, Python, and workers.

Fine at cap 1.1M uses `compact_gaussians` with degree 1 and classic densification:

- host `properties`: 69 float32/row = 303,600,000 B. This comprises 101,200,000 B parameters and 202,400,000 B embedded Adam states;
- nodes: 24 B/row = 26,400,000 B;
- classic criterion: 4 B/row = 4,400,000 B;
- source-derived host lower bound: 334,400,000 B.

The 100k permanently live skybox rows require at least 27,600,000 B for parameters plus moments, 9,200,000 B for gradients during backward, and about 800,000 B for row indices plus classic criterion: approximately 37,600,000 B before upper/SPT metadata, selected/cached rows, cameras, render intermediates, and framework overhead.

### 8.3 Explicit estimate labels

```text
Host RSS 4–8 GiB: PROJECT SAFETY ESTIMATE — UNMEASURED
VRAM 4–10 GiB:    PROJECT SAFETY ESTIMATE — UNMEASURED
```

These are project safety estimates, not expected measurements. No paper-derived peak is asserted for this adapted asset. Runtime overhead and rasterizer peak are unknown. The execution task must replace them with measured RSS, swap, whole-device VRAM, process VRAM, and stage peaks.

## 9. Cache feasibility: conditional gate

`cache_size` measures the complete current GPU row list (`gaussian_indices`), not only retained cache entries. Every such list begins with all 100,000 skybox rows. Selected upper rows, newly loaded SPT cuts, reused SPT rows, and retained SPT rows are then concatenated. Reduction is attempted only after the current selection/reuse calculation, so temporary overshoot remains possible.

The HSPT upper size depends on creator output scales and cannot be established before the initial hierarchy exists. The envelope is therefore **CONDITIONAL-EXECUTION-PREFLIGHT**, not statically proven safe.

The exact official HSPT is rebuilt after Barrier 2, under the resource monitor and before fine training. The conservative acceptance rule is:

```text
100000 skybox rows
+ rebuilt upper-hierarchy rows
+ 257 rows (smallest qualifying SPT because len(SPT) > 256)
+ 20000 temporary/reuse reserve
< 240000 reduced cache target
```

Thus rebuilt upper rows must be at most 119,742. This deliberately counts all rebuilt upper rows even though a camera normally emits only a cut, providing conservative structural headroom. The preflight records actual initial rows, upper rows, SPT count/minimum, conservative total, and result. Failure returns `BLOCKED-ALOD-RESOURCE-BUDGET`; the script stops before Stage 3 and does not increase either cache value.

## 10. Fine schedule

Source condition:

```python
opt.densify_from_iter < iteration < opt.densify_until_iter \
and iteration % opt.densification_interval == 0
```

Resolved schedule:

```text
1,000–10,000: configured strict densification window
2,000, 4,000, 6,000, 8,000: actual scheduled events
10,000: excluded by strict upper inequality
10,000–12,000: post-densification stabilization
12,000: final degree-1 hierarchy save
```

The source's scheduled saves at 200k/250k/300k do not occur in this run.

## 11. Resource policy and monitoring

The external monitor samples every five seconds and launches each stage with `start_new_session=True`. It records timestamp, stage, leader PID, PGID, all observed group PIDs, process-group RSS, available RAM, swap and baseline growth, whole-device/process VRAM when `nvidia-smi` is reliable, output bytes, latest log-derived iteration, and diagnostic node/memory text. PyTorch allocated/reserved values remain diagnostic-only unless the official process emits them.

| Resource | Warning | Hard / classification |
|---|---:|---:|
| Process-group RSS | 20 GiB for 30 s | 24 GiB for 15 s, hard-enforced |
| Available system RAM | below 8 GiB | below 5 GiB for 15 s, hard-enforced |
| Swap growth from stage-launch baseline | 128 MiB for 15 s | 256 MiB for 15 s, hard-enforced |
| Whole-device VRAM | 12 GiB for 10 s | 13.5 GiB for 10 s, hard-enforced |
| Training-process VRAM | 11.5 GiB for 10 s when attribution is reliable | 13 GiB for 10 s when attribution is reliable, hard-enforced |
| PyTorch allocated/reserved | `null`, diagnostic-if-emitted | `null`, never an abort trigger |
| Output directory | 2 GiB | 5 GiB, hard-enforced |
| Node count | 1.05M diagnostic/preflight | 1.1M configuration-enforced |
| No log progress | 15 minutes | 30 minutes, hard-enforced |

All sampled threshold durations use consecutive violating samples. The first violating sample counts as sample one and the required count is `ceil(duration / 5 seconds)`: 15 seconds is three samples and 10 seconds is two samples. A non-violating sample resets that metric and level. Process VRAM is recorded as `null` when reliable PID attribution is unavailable; whole-device VRAM is never substituted. In addition to the numeric swap limit, six strictly increasing swap-growth samples with total growth of at least 128 MiB cause a hard abort; the first increase from the stage baseline counts as sample one.

Wall-time warning/hard limits: coarse 90 min/3 h; creator 20 min/1 h; cache preflight 20 min/1 h; fine 4 h/10 h; whole pipeline 12 h hard. On abort, the monitor signals the process group with SIGINT, waits 30 seconds, SIGTERM, waits 30 seconds, then SIGKILL. It records stage, leader, PGID, children, timestamps, exit status, and reason. Partial artifacts and logs are preserved but cannot cross a barrier.

Hard failures include OOM, CUDA illegal access, non-finite loss, traceback/nonzero process exit, persistent swap, hard resource/output/time/no-progress limits, corrupt barrier artifacts, invalid hierarchy counts/links, and any observed `GaussianHierarchyCreator` process during Stage 3.

Planning history: the prior execution request stopped at `BLOCKED-ALOD-PREFLIGHT` because its thresholds disagreed with the then-frozen policy. This refreeze preserves that event as a planning inconsistency corrected before execution; it does not claim that training started.

## 12. Three-stage orchestration proof

At `train.py` lines 57–82, `--skip_if_exists` checks for a non-empty `output/scaffold/point_cloud`, parses `iteration_*` directory suffixes, and selects the maximum. The frozen layout contains `iteration_2500` and `iteration_5000`, so the selected iteration must be 5000 and coarse training is skipped.

At lines 96–115, the same flag checks specifically for:

```text
output/scaffold/point_cloud/iteration_5000/hierarchy.dhier
```

Its existence skips the in-tree creator. Lines 119–133 then set `model_params.hierarchy`, `scaffold_file`, and `output_path` to the validated artifacts and call `train_hierarchy.training`. The command asserts the maximum directory, file existence, saved hashes, config real path, creator binary hash, and both skip conditions immediately before Stage 3. The monitor also treats any creator child process during fine as an orchestration failure. No external binary is copied into official source.

## 13. `safe_state()` runpy launcher

Unlike the standalone `train_scaffold.py` and `train_hierarchy.py` mains, `train.py` never calls `safe_state`. The narrow Stage-3 launcher:

1. imports the official `utils.general_utils.safe_state`;
2. removes only the official script path from `sys.argv`;
3. calls `safe_state(False)` before `train.py` argument parsing/module execution;
4. runs the official file with `runpy.run_path(..., run_name="__main__")`.

`safe_state(False)` seeds Python, NumPy, and Torch with zero and selects CUDA device 0. It does not enable deterministic algorithms. `False` preserves timestamped stdout; `True` would silence it. The wrapper does not patch functions, change configuration, catch official exceptions, or alter CUDA behavior beyond the official helper. `validate-launcher` exercises the exact pop/runpy argument-forwarding pattern on a temporary inert script without importing `safe_state`, initializing CUDA, or starting training.

## 14. Executed command and frozen run

The launcher used for Run 004 remains available for provenance validation:

```bash
bash /home/luis/tese/gs_framework/environments/alod-official/first-capped-training-command.sh --validate-only
```

The frozen shell file contains the exact argv arrays for Stage 1, the external creator, cache preflight, and seeded `train.py --skip_if_exists`. It uses `set -euo pipefail`, deterministic absolute paths, a new run directory, stage-specific process groups, barriers, hashing, and factual provenance finalization.

The frozen run directory is `/home/luis/tese/runs/alod/ALOD-ORACLE-MC-AERIAL-TEST-01/ALOD-SMOKE-TRAIN-004`. Runs 001 through 004 are evidence only and must never be resumed, overwritten, or reused. The launcher still names Run 004 and therefore must not be executed again. No Run 005 launcher has been prepared or authorized.

## 15. Recovery and restart

The planned 2,500 coarse recovery point is not a usable checkpoint at the pinned commit. `GaussianModel.capture()` reads `max_radii2D`, but its initialization is commented out in both construction paths. Requesting `--checkpoint_iterations 2500` risks terminating coarse training. The frozen command instead writes `iteration_2500/point_cloud.ply` and `pc_info.txt` as diagnostic/intermediate artifacts. They contain no optimizer or loop state and cannot resume coarse training. A failed coarse stage restarts from iteration zero in a new run or explicitly reviewed attempt.

The creator is independently rerunnable from a Barrier-1-validated coarse PLY; its prior partial outputs must not be accepted without Barrier 2. Stage 3 can start independently from validated coarse and initial hierarchy hashes.

> The fine stage has no usable checkpoint/restart path at the pinned commit. If it aborts, it must restart from the validated initial hierarchy.

The initial hierarchy, logs, policy, command, and hashes must be retained for that restart. No partial fine hierarchy is treated as resumable state.

## 16. Acceptance and abort criteria

The run could succeed only if:

1. official coarse training reaches 5,000;
2. both coarse PLY barriers pass and `pc_info.txt` is exactly 100,000;
3. the external creator completes and the initial degree-3 hierarchy passes all checks;
4. actual fine-initial count/headroom and the conditional cache preflight pass;
5. `train.py --skip_if_exists` takes both expected skip branches;
6. fine training reaches 12,000 and writes `final.dhier`;
7. final `G==N`, degree is 1, all attributes are finite, and the non-skybox tree is valid/reachable;
8. no resource, swap, CUDA, process-tree, output, or wall-time policy is violated;
9. all hashes, metrics, stage durations, failures/restarts, and provenance are completed; and
10. canonical ALoD, both Conda environments, the frozen dataset, CityGaussian, runtime, tests, and benchmarks remain unchanged.

Image quality is not a success criterion. Any failed barrier, cache inequality, insufficient headroom, invalid/truncated artifact, non-finite value, creator launch in Stage 3, hard threshold, abnormal process exit, or protected-state modification makes the run unsuccessful and stops subsequent stages.

## 17. Limitations and decision

Limitations: resource-adapted cap and schedule; aerial-only imagery; fixed focal length; limited height/distance range; repurposed original test reconstruction; invalid for formal MatrixCity/CityGaussian comparison; no street/aerial transition; and non-trivial LoD variation remains unverified. A second formal or mixed-scale dataset will still be required.

The compatibility remediations and resource envelope were validated. Run 004 nevertheless failed the frozen hierarchy-consistency criterion, and the output must not be accepted as the first official `.dhier` oracle.

```text
FAILED-ALOD-FINAL-BARRIER
```

Decision required before further execution: authorize either a bounded disposable-worktree repair to official hierarchy relocation with a synthetic structural reproducer, a post-serialization metadata repair (less source-faithful), or a weaker topology-only acceptance rule (not recommended). Any approved retraining must use a new immutable Run 005 directory.
