# ALoD `.dhier` Inspector Validation

Date: 2026-08-07

Status: `PASS-ALOD-DHIER-INSPECTOR`

## Scope

This result covers binary parsing, numeric inspection, structural validation,
official-loader parity, and deterministic reporting only. It implements no
HSPT construction, hierarchy selection, residency, cache, rendering, viewer,
or training behavior.

## Assets

| Role | Rows | SH degree | SHA-256 | Result |
|---|---:|---:|---|---|
| Run004 initial creator hierarchy | 887,239 | 3 | `a7862de91cd3620ab7be1296a50b9030aa1c2cca2b4fe15d79abaf55f4664d17` | Valid with explicit `creator-initial` profile |
| Run005 correctness-patched final | 1,099,999 | 1 | `6c37ee8eb88b3b126ec8a6bf35e5bb754a2d1983cd53d3b9b6b39358cc6c923c` | Strictly valid |
| Run004 rejected final | 1,099,999 | 1 | `b763f008c466170628e817d818b5f41d53c018153feaab8a725f8305b1aa3c0f` | Rejected |

Run005 contains a 100,000-row skybox prefix. Its hierarchy root is stable
`global_node_id=100000`; all 999,999 non-skybox nodes and 999,998 forward edges
are reachable exactly once, with maximum depth 23.

The rejected Run004 final independently reports 4,355 depth mismatches, one
reverse-parent mismatch, and one corrupted skybox topology row. These counts
are discovered from the asset and are not embedded in parser logic.

## Format and platform

The native upstream format has no magic, schema version, checksum, endianness
marker, or width metadata. The project parser uses explicit little-endian
32-bit integers/floats under the **PINNED-ASSET PLATFORM ASSUMPTION** that these
assets were produced and consumed on validated x86-64 Linux. Cross-endian
portability is unsupported.

Degree-1 rows use 92 attribute bytes plus 24 hierarchy bytes; degree-3 rows use
236 plus 24. The common header/count overhead is 12 bytes. Both accepted
assets match the resulting expected byte length exactly.

## Official parity

Layer A compares the raw project decode with the official native dynamic
loader. Positions, rotations, log scales, serialized opacity, SH coefficients,
and all six node columns match by exact full-array SHA-256 and sampled raw row
bytes for both assets.

Layer B runs the official Python model with `scaffold_file=""` to isolate
serialized-row transformations. The project independently reproduces SH
DC/rest splitting, opacity clamp and inverse-sigmoid, node-column-five zeroing,
and the exp/normalized-quaternion/sigmoid render-ready views. Every resulting
full-array hash and sampled row matches exactly.

The degree-3 creator artifact uses two explicit compatibility rules derived
from pinned source. Its 443,620 leaves carry unused positive `first_child`
values that are reported but never traversed. Its merge equation produces
92,926 finite, positive interior opacity proxies above one; the official Python
model clamps them before inverse-sigmoid. Neither exception applies to strict
Run005 validation.

## Engineering measurement

On the local machine, Run005 parsing/memory-map setup took approximately
0.0007 seconds. Complete validation, hashing, and report generation took
approximately 2.99 seconds, with process peak RSS approximately 315 MB. These
are engineering observations, not formal thesis benchmark results.

## Next bounded task

Instrument the official ALoD runtime and export the first
hierarchy/HSPT/selection oracle fixtures from Run005.
