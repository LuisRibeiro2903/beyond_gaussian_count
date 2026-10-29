# Sci-Art Stage-A input view

`prepare_sciart_prevalence_dataset.py` verifies the frozen public COLMAP
train/val source hashes, prepares the official UrbanScene3D photographs with
the Mega-NeRF/CityGaussian undistortion metadata, and seals `manifest.json` and
`image-manifest.json`. The pinned ALoD loader applies `--resolution 4`, yielding
1216×912 training/evaluation images.

The public train-only model is the training input (`llff_hold=0`). The pinned
loader has no reachable named test-list branch, so the oracle uses a 42-camera
view with public val names at sorted indices 0, 2, ..., 40 (`llff_hold=2`).
The other 21 cameras are source train cameras used only for the evaluator's
normalization/focal setup. Their source image IDs collide with the val export,
so only these normalization-camera IDs are deterministically remapped by
adding 1,000,000. Public validation IDs, camera poses, calibrations, and pixels
remain unchanged. The mapping is in `manifest.json`.

`freeze_sciart_prevalence_cameras.py` writes the all-21-camera oracle manifest
and coverage diagnostic before coarse training. The source ZIP and photos are
local research inputs and are not committed or redistributed.
