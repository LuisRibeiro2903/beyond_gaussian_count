# Formal mixed-scale ALoD training result

Date: 2026-08-20  
Dataset: `ALOD-REPRO-MC-MIXED-01`  
Run: `ALOD-FORMAL-TRAIN-002`  
Status: `FAILED-ALOD-FORMAL-RESOURCE-POLICY`

## Result

The frozen formal run completed 30,000-iteration coarse training, official
hierarchy creation, and independent initial hierarchy validation. It stopped
at the mandatory live cache/HSPT gate before fine training. No final `.dhier`
was produced, and `PASS-ALOD-FORMAL-DHIER` is not available.

The coarse artifact contains exactly 2,417,128 finite rows and has SHA-256
`e4b04b79024e751efa8b89cf381156a6d74b14fc9c2ae750c1a33634f0c51e1d`.
The official creator produced a degree-3 hierarchy with 4,634,255 rows,
2,317,128 leaves, 2,317,127 interiors, maximum depth 22, and SHA-256
`c1487170a0b1e53314b70df36bc5cc8aa0da4304c5a3d21b67e3ffc3dcecadf3`.
The project-owned creator-profile inspector found complete reachability and
no topology mismatches.

## Failure gate

With frozen `SPT_root_volume=150`, the official HSPT construction creates one
SPT containing all 4,634,255 non-skybox rows. The conservative complete cache
requirement is therefore:

```text
100,000 skybox + 1 upper row + 4,634,255 SPT rows + 100,000 reserve
= 4,834,256 rows
```

This cannot satisfy the frozen 1,600,000-row reduced-cache target. The strict
cache inequality fails, so changing cache sizes or SPT parameters would cross
the frozen methodological boundary. Fine training was not started.

The official code first exposed a PyTorch 2.11 scalar-indexing compatibility
error because the upper HSPT contains exactly one node. A diagnostic-only
shape wrapper retained the identical scalar value as a one-element tensor and
allowed the cache inequality to be measured. No source patch was applied for
training after this diagnosis.

## Resources

Coarse training took 901.11 seconds. Peak process-tree RSS was 10,460,053,504
bytes and peak whole-device VRAM was 2,947,547,136 bytes. Minimum available
RAM was 26,962,919,424 bytes; swap growth was zero. No machine hard limit was
reached. The gate is a frozen cache-feasibility failure, not an OOM.

Formal evidence is under
`/home/luis/tese/runs/alod/ALOD-REPRO-MC-MIXED-01/ALOD-FORMAL-TRAIN-002`.
No rendering, oracle generation, quality evaluation, or benchmark integration
was performed.
