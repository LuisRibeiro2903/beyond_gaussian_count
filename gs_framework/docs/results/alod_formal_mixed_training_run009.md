# Formal mixed-scale ALoD fine training: Run009

Status: `PASS-ALOD-FORMAL-DHIER`

Run009 completed the 100,000-iteration fine-only training stage for
`ALOD-REPRO-MC-MIXED-01` and produced a strictly validated final hierarchy:

```text
/home/luis/tese/runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-009/output/ALOD-REPRO-MC-MIXED-01-FORMAL-TRAIN-009.dhier
```

Final asset:

| Field | Value |
|---|---:|
| SHA-256 | `38f4abdc0d47f2d20316de33f4fff87086e038a57daac0f2feaf98fa9178eb7b` |
| File bytes | 820,424,744 |
| Total rows | 7,072,627 |
| Hierarchy rows | 6,972,627 |
| Skybox rows | 100,000 |
| SH degree | 1 |
| Maximum depth | 55 |

## Fidelity classification

This result is a **public-source faithful ALoD reproduction on the frozen
mixed-scale MatrixCity proxy, with source-documented dataset/resource-specific
HSPT and training-cache calibration**. It is not an exact paper configuration
or a bit-identical paper reproduction.

Frozen method semantics include the dataset and split, resolution, coarse and
fine schedules, 12M cap, densification start/end/interval and behavior, target
granularity, minimum SPT size, 100,000-row skybox, SH degree, objective,
sampling behavior, hierarchy convention, and machine safety limits. Run009
reused the exact Run002 coarse PLY (`e4b04b...c51e1d`) and initial hierarchy
(`c14871...ecadf3`); coarse training and hierarchy creation were not rerun.

Dataset/resource-specific calibration is limited to
`SPT_root_volume=4.75`, `cache_size=6,250,000`, and
`cache_size_after_reduction=5,000,000`. The 0.8 cache ratio matches the released
source style. The cache-only change preserves hierarchy construction, LoD
semantics, training objective, densification, and render representation.

The execution tree retains the known relocation-topology correctness patch and
validation instrumentation over canonical commit
`73f547341547ae021ff036ad74fdf73a1240b450`. The canonical checkout remained
clean. The relocation diff and instrumented trainer were hash-pinned; topology,
ownership, depth, parent/child, relocation, sentinel, and skybox checks were not
weakened.

## Chronological calibration and attempts

1. **v1:** `SPT_root_volume=150` produced a degenerate one-SPT decomposition.
2. **v2 / Run006:** `SPT_root_volume=4.75`, 2.0M cache, and 1.6M reduced cache
   produced a structurally valid 22-SPT HSPT but exhausted the dynamic cache at
   iteration 13,500 (`1,601,132 > 1,600,000`). All resource and topology gates
   were healthy.
3. **v3 / Run007:** retained `SPT_root_volume=4.75` and scaled only cache capacity
   to 6.25M/5.0M. The choice used all Run006 barriers, a full-12M-cap growth
   projection, 35%-50% largest-SPT sensitivity, a measured/theoretical memory
   model, and the source-style 0.8 ratio. Run007 passed every barrier through
   iteration 48,000 with 3,120,988 rows of reduced-cache headroom, then an
   external WSL restart terminated it at iteration 49,160.
4. **Run008:** used identical methodological settings and restarted from frozen
   Run002. It was externally stopped at iteration 310 when the non-lingering
   user service manager exited with the ChatGPT login session. WSL stayed
   running; this was not a method, cache, topology, or resource failure.
5. **Run009:** used the same v3 settings and frozen inputs, completed training,
   serialized the final hierarchy, and passed all artifact gates. No checkpoint
   resume was used in any retry.

The v3 cache model uses 284 persistent bytes and 92 gradient bytes per cache row.
At the 6.25M threshold it projected about 6.05 GiB whole-device VRAM normally
and 8.86 GiB under a conservative 1 KiB/row copy-churn envelope, retaining
4.64 GiB to the 13.5 GiB hard limit. Run009's measured peak was lower.

## Dynamic correctness and resources

Run009 recorded 51 HSPT/cache barriers from iteration 0 through the last
densification rebuild at iteration 84,000. All passed. The minimum reduced-cache
headroom was 2,924,624 rows. All 152 topology events passed with unchanged
skybox nodes.

| Resource | Observed | Hard condition |
|---|---:|---:|
| Peak process-tree RSS | 7.163 GiB | <= 24 GiB |
| Minimum available system RAM | 21.842 GiB | >= 5 GiB |
| Peak whole-device VRAM | 6.501 GiB | <= 13.5 GiB |
| Maximum swap growth | 0 MiB | <= 256 MiB |
| Final output | 0.764 GiB | <= 5 GiB |

## Final validation

The monitor inspector passed with every property finite, valid links, and valid
skybox sentinels. The independent strict parser reported one root, zero cycles,
zero unreachable rows, zero parent/depth mismatches, valid binary child counts,
and all 6,972,627 hierarchy rows reachable. The official loader passed loaded,
compacted, and rebuilt-HSPT topology checks; active/HSPT nodes were unchanged,
and the serialized node metadata round trip was equal.

The launcher's first protected-state final check initially exited nonzero due a
bookkeeping collision: both smoke Run005 and formal Run005 used the filename
`run005-before.json`, so the second snapshot overwrote the first. The correct
start-time smoke hash remained in `launcher.log`. A no-training/no-mutation
recovery reran all ten protected-tree checks with unique labels; every current
hash matched its start hash. The recovery then repeated frozen-input, source,
artifact, dynamic, and resource assertions before sealing provenance. This did
not alter the `.dhier`.

No diagnostic-camera rendering, selection oracle, FPS benchmark, quality
evaluation, paper ablation, CityGaussian comparison, or `ALOD-EXT` optimization
was executed after training.

The next bounded task is to generate the formal mixed-scale selection oracle on
the 16 frozen cameras and validate official selection behavior against this
Run009 hierarchy. That task is identified here but was not executed.
