# Fast ALoD Pre-Materialization Culling and Reduced Cache

Status: `PASS-ALOD-FAST-PREMATERIALIZATION-CULLING`

Artifact root:
`/home/luis/tese/benchmark_artifacts/alod_fast_materialization_culling_001`

Asset: Run005 `final.dhier`, SHA-256
`6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c`

## Scope and frozen semantics

This task changes only the post-selection mapping from ordered semantic IDs to
materialization IDs. The semantic selector, HSPT construction, upper traversal,
candidate SPT generation, SPT cut, LoD predicate, four-plane frustum, selected
IDs, and selected order are unchanged. `scene_runtime/alod/selection.py` still
has SHA-256 `f893713fd0f51d8e4defe962052bf5b03682df09b8c527dc763b7b3bc122d55e`,
matching the prior formal campaign manifest.

The prior CPU implementation remains available as `conservative_pinhole` and
is the executable correctness reference. The historical
`ALOD-HOST-BOUNDED-CACHE` file also retains SHA-256
`2b9a17f0481ff4a6b7a916d3dc9a148442fc5476df1239c44f3e0446772ebdb0`.

## Exact filter semantics

Persistent per-Gaussian inputs are only:

- serialized float32 mean `(x, y, z)`;
- padded float32 maximum activated scale.

The maximum scale is constructed exactly as before: take the float32 maximum
of the three log scales, cast it to float64, apply float64 `exp`, multiply by
`1.0001`, then cast the result to contiguous float32. The filter does not use
SH, opacity, quaternion, or the complete scale vector.

For each selected mean, the existing explicit float32 multiply/add order
transforms its center by `w2c`. Depth retention is finite and inclusive:

```text
z >= near_plane - 1e-4
z <= far_plane + 1e-4
```

Projection additionally requires positive depth. The calculation then retains
the existing gsplat-aligned clamp limits, pinhole Jacobian norms, `eps2d`,
extent `3.33`, `ceil`, and 2-pixel radius padding. Image overlap comparisons
remain strict:

```text
mean_x + radius_x > 0
mean_x - radius_x < width
mean_y + radius_y > 0
mean_y - radius_y < height
```

Non-pinhole, UT, and eval3d modes retain the identity materialization set.

## Fast architecture

`conservative_pinhole_gpu` is a separate opt-in mode:

```text
unchanged CPU semantic selection
  -> ordered int64 semantic IDs
  -> ID H2D
  -> persistent compact-metadata gather
  -> eager PyTorch CUDA projection and conservative mask
  -> stable GPU compaction
  -> compacted semantic-position D2H for CPU host/cache lookup
  -> full-property gather/transfer only for retained IDs
  -> reuse compacted GPU global IDs for raster identity
```

There is one required device/host dependency because the existing residency
maps and host property store are CPU-side. The path transfers retained semantic
positions rather than sending retained global IDs back and then retransferring
them for rendering. No custom CUDA, selector migration, `.item()` loop, or
full-property prefilter transfer was added. CUDA events measure GPU stages;
the compacted-position copy provides the required synchronization for host
lookup.

## Compact metadata memory

| Quantity | Bytes | MiB |
|---|---:|---:|
| Bytes per Gaussian | 16 | N/A |
| Logical CPU compact table | 17,599,984 | 16.78 |
| Incremental CPU bytes beyond existing means | 4,399,996 | 4.20 |
| Persistent GPU compact table | 17,599,984 | 16.78 |

CPU and GPU means/max-scales are bitwise equal. The GPU table contains no SH,
opacity, rotation, or full scale vector.

The mean semantic-ID H2D payload is 4,658,016 bytes/frame (4.44 MiB), with a
4,792,240-byte maximum. These totals include the filtered skybox prefix; the
mean hierarchy-only semantic count remains 482,252 rows. The mean retained
semantic-position D2H payload is only 150,735 bytes/frame (0.144 MiB).

## Correctness gate

`checks/fast_filter_validation.json` validates the ten frozen cameras against
the CPU filter, the unfiltered resident production runtime, the host bounded
cache, and packed gsplat projection IDs.

- fast ordered materialization IDs equal CPU reference IDs on all cameras;
- semantic selected IDs and order are unchanged;
- surviving means, quaternions, scales, opacities, and SH rows are bitwise
  equal to the corresponding resident rows;
- zero false negatives remain across 168,566 backend projection-survivor
  instances;
- packed backend survivor IDs are exact;
- RGB and alpha are bitwise identical.

The materialization counts remain the validated
`[2018, 6353, 33008, 51074, 9970, 19041, 19414, 21252, 22378, 3911]`.
Thus the full-property reduction is unchanged.

## Filter performance

The dedicated timing campaign used two complete warmup repetitions followed by
100 measured calls over repeated unchanged camera/semantic-ID pairs.

| Stage | Mean ms | Median ms | P95 ms | Maximum ms |
|---|---:|---:|---:|---:|
| CPU reference filter | 17.164 | 17.146 | 20.714 | 24.162 |
| Fast total filter | 2.468 | 2.326 | 3.067 | 4.039 |
| CPU ID preparation | 0.116 | 0.112 | 0.148 | 0.193 |
| Selected-ID H2D | 0.351 | 0.308 | 0.791 | 0.814 |
| Compact metadata gather | 0.204 | 0.190 | 0.254 | 0.806 |
| Projection/culling | 1.163 | 1.066 | 1.677 | 2.711 |
| Mask compaction | 0.304 | 0.291 | 0.363 | 0.895 |
| Retained-position D2H | 0.165 | 0.168 | 0.216 | 0.241 |
| CPU materialization-ID production | 0.023 | 0.023 | 0.047 | 0.070 |

Mean filter speedup is 6.95x. A separate ten-warmup/ten-frame integrated run
measured 2.259 ms fast filter time versus 19.083 ms for the CPU-filter/525k
control. Mean end-to-end time fell from 29.965 ms to 11.812 ms. The semantic
selector remains unchanged at about 6.4 ms and is again the largest stage.

## Runtime memory characterization

The fast/reduced path preserves the same mean frame-local full-property bytes
as the CPU-filter control: 3,466,910 bytes, with a 9,397,616-byte maximum.
The change is execution location plus persistent cache capacity, not a change
to materialization semantics.

| Persistent component | CPU filter / 525k | Fast filter / 160k |
|---|---:|---:|
| Dynamic full-property cache | 48,300,000 B | 14,720,000 B |
| Compact GPU metadata | 0 B | 17,599,984 B |
| Net of these two components | 48,300,000 B | 32,319,984 B |

The reduced cache saves 33,580,000 property bytes and adds 17,599,984 compact
bytes, for a net persistent saving of 15,980,016 bytes (15.24 MiB) relative to
the CPU-filter/525k path. Relative to the earlier unfiltered path, the prior
217.88 MiB whole-device reduction would be 201.10 MiB after adding the compact
table at the same cache size; the reduced cache adds a further 32.02 MiB of
persistent capacity reduction.

In the fresh-process integrated comparison, maximum PyTorch allocation fell
from 108,075,008 to 95,535,616 bytes. Maximum reserved memory instead rose from
123,731,968 to 140,509,184 bytes because the allocator retained temporary GPU
culling buffers. Whole-device NVML was 26.25 MiB higher in that two-process
sample, so it is not used to claim a net whole-device reduction. Persistent
component accounting and allocated/reserved peaks are reported separately.

## Deterministic request trace

`cache/request_trace_ids.npz` stores the exact ordered dynamic-property IDs for
each of the ten frames. `cache/request_trace_and_simulation.json` stores
per-frame counts, overlap, new/discarded rows, cumulative union, and ordered
SHA-256 hashes. The combined ordered trace SHA-256 is
`191d899c0eb5a2ce2d24d6747d6ffa10deddd91d7d80876e79f31c72d1d136bf`.

Recomputed bounds:

- maximum one-frame dynamic request: 50,936 rows;
- absolute minimum feasible cache without chunking: 50,936 rows;
- ten-camera dynamic union: 151,621 rows.

## Cache-capacity replay

The table replays 100 frames (ten exact trajectory repetitions) with the
unchanged production LRU and 92 full-property bytes per row.

| Capacity rows | Capacity MiB | Hit rate | Miss rate | Loads | Evictions | H2D MiB | Zero-transfer frames |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50,936 | 4.47 | 18.12% | 81.88% | 1,540,935 | 1,489,999 | 135.20 | 0 |
| 64,000 | 5.62 | 21.14% | 78.86% | 1,484,059 | 1,420,059 | 130.21 | 0 |
| 96,000 | 8.42 | 23.17% | 76.83% | 1,445,959 | 1,349,959 | 126.87 | 0 |
| 128,000 | 11.23 | 32.95% | 67.05% | 1,261,805 | 1,133,805 | 110.71 | 0 |
| 151,621 | 13.30 | 91.94% | 8.06% | 151,621 | 0 | 13.30 | 90 |
| 160,000 | 14.04 | 91.94% | 8.06% | 151,621 | 0 | 13.30 | 90 |
| 525,000 | 46.06 | 91.94% | 8.06% | 151,621 | 0 | 13.30 | 90 |

The absolute minimum is feasible but churn-heavy. Capacities through 128k
still transfer more than 1.26 million rows. The 160k candidate is the first
evaluated round capacity at the trace-union knee: it reproduces 525k loads and
zero-eviction behavior while retaining 8,379 rows (5.53%) of measured-union
headroom. It is therefore selected for the next formal benchmark rather than
using either the hard minimum or exactly the observed union.

`ALOD-HOST-BOUNDED-CACHE-REDUCED` records this separately. It does not overwrite
the historical 525k configuration.

## Previous-working-set comparison

On the same 100-frame trace, previous-working-set reuse achieves a 12.54% hit
rate, loads 1,645,905 rows, transfers 151,423,260 bytes (144.41 MiB), and has no
zero-transfer frame. The 160k cache loads 151,621 rows and transfers 13,949,132
bytes (13.30 MiB), a 90.79% H2D-byte reduction. In this filtered regime,
explicit cache policy is materially different from immediate-previous reuse.

## Validation and limitations

- 209 full default unit tests passed (two environment skips);
- 14 opt-in ALoD external integration tests passed on Run005/CUDA;
- 63 CityGaussian regression tests passed;
- `git diff --check` passed;
- no source was staged, committed, or pushed.

The benchmark is intentionally bounded and is not the next formal campaign.
The test process used the existing compiled gsplat 1.5.3 extension because the
system CUDA 12.0 compiler cannot JIT `sm_120`; no gsplat source or version was
changed. Allocator reservation remains a memory caveat even though persistent
capacity and peak allocation fall.

## Next bounded task

Run the formal materialization-aware residency benchmark using the fast
conservative filter and the selected 160k cache capacity, then decide whether
CPU HSPT/SPT selection should be the next optimization before scaling to a
larger mixed-scale asset.
