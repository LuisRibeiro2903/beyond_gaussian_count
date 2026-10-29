# ALoD Released Viewer Previous-Working-Set Reuse

| Measure | Mean | Median | p95 |
| --- | ---: | ---: | ---: |
| Retained rows | 3368738.86 | 3482442.00 | 3484077.00 |
| Retention ratio | 0.9747 | 0.9999 | 1.0000 |
| Newly loaded rows | 87569.76 | 293.00 | 776473.00 |
| H2D rows saved vs no-cache | 3344571.20 | 3450968.00 | 3478223.00 |
| End-to-end time saved (ms) | 52.57 | 52.11 | 75.08 |

Mean end-to-end change versus no-cache: -28.14%.

exact released viewer preceding-working-set reuse semantics: strict 0.3 < previous/current camera-distance ratio < 1.5 per SPT; whole prior SPT cuts retained; newly required properties loaded; no LRU or bounded cache.
