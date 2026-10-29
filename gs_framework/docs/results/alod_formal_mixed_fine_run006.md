# Formal mixed-scale ALoD fine training: Run006

Status: `FAILED-ALOD-FORMAL-DYNAMIC-CACHE-FEASIBILITY`

Run006 reused the exact immutable Run002 coarse PLY and initial hierarchy. Both
input hashes matched their freeze record. The pinned official-source HSPT
re-freeze also reproduced the calibrated `SPT_root_volume=4.75` structure:
22 SPTs, 889 upper rows, 99.9813% coverage, maximum SPT 1,158,563 rows, and
1,359,452 required reduced-cache rows.

The value 4.75 was selected before fine training by a 30-candidate structural
sweep using official HSPT behavior, at least 95% SPT coverage, non-degenerate
partitioning, cache feasibility, and stability/sensitivity. The calibration
used no rendering, FPS, quality result, or diagnostic-camera LoD behavior.

Fine training passed the initial gate and two dynamic rebuild gates. The third
densification rebuild exceeded the frozen reduced cache by 1,132 rows:

| Iteration | Total rows | Maximum SPT | Required cache | Headroom |
|---:|---:|---:|---:|---:|
| 0 | 4,734,255 | 1,158,563 | 1,359,452 | 240,548 |
| 10,500 | 5,066,985 | 1,273,115 | 1,474,012 | 125,988 |
| 12,000 | 5,287,977 | 1,349,533 | 1,550,438 | 49,562 |
| 13,500 | 5,437,163 | 1,400,225 | 1,601,132 | -1,132 |

The iteration-13,500 densification added 149,186 rows and relocated 6,771.
The runtime barrier stopped training immediately after rebuilding the HSPT.
No final `.dhier` was serialized.

All 11 mutation/topology validations passed, including global reachability,
parent/child and depth consistency, affected-subtree checks, and unchanged
skybox node metadata. Fine-stage wall time was 6,588.22 seconds. Peak
process-tree RSS was 7,248,592,896 bytes, minimum available system RAM was
23,796,523,008 bytes, peak whole-device VRAM was 4,746,903,552 bytes, swap
growth was zero, and peak output size was 13,066,990 bytes. No machine hard
resource limit was reached.

Runs003 and Run004 were preserved resource-policy failures caused respectively
by fresh gsplat JIT compilation and forked DataLoader RSS accounting. Run005
was preserved as a provenance-writer infrastructure failure before training.
Run006 used the exact precompiled extension and ordered synchronous loading;
these operational corrections did not alter the frozen optimization envelope.

No selection oracle, diagnostic-camera evaluation, quality metric, FPS run,
ablation, or benchmark was executed. A new methodological freeze is required
before another fine-training attempt; this run does not authorize changing the
SPT threshold, cache sizes, minimum SPT size, densification schedule, cap, or
other training semantics.
