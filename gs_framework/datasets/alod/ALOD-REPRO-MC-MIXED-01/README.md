# ALOD-REPRO-MC-MIXED-01

Frozen status: `PASS-ALOD-FORMAL-DATASET` and
`PASS-ALOD-MIXED-SCALE-CAMERA-COVERAGE`.

This is a **mixed-scale MatrixCity proxy for faithful public-source
reproduction; it is not the unreleased exact ALoD paper dataset**.

The external payload is at
`/home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01`. Images remain outside Git
and are immutable symlinks to preserved MatrixCity source payload. The
project-side JSON files freeze construction provenance, every image hash and
pose, the official split, camera statistics, diagnostic identities, and the
official-loader validation report.

Core identity:

| Item | Frozen value |
|---|---:|
| Cameras | 710 |
| Aerial / street | 335 / 375 |
| Train / evaluation | 639 / 71 |
| Diagnostic evaluation cameras | 16 |
| Sparse points | 2,317,128 |
| Official split | `llff_hold=10` |
| Manifest SHA-256 | `eaf3cb228234efd92441d28ea56482dcf4676c5a318a6d6beef1020ee27c5f7a` |

Rebuild with the `construction_command` in `manifest.json`. Validate with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/luis/miniconda3/envs/alod_official/bin/python \
  /home/luis/tese/gs_framework/scripts/alod/validate_formal_mixed_dataset.py \
  --dataset-root /home/luis/tese/data/alod/ALOD-REPRO-MC-MIXED-01 \
  --metadata-root /home/luis/tese/gs_framework/datasets/alod/ALOD-REPRO-MC-MIXED-01 \
  --official-source /home/luis/tese/LoDOfGaussians \
  --official-commit 73f547341547ae021ff036ad74fdf73a1240b450
```

Do not alter the filename ordering: evaluation membership is deliberately
encoded at indices divisible by ten for the pinned loader.
