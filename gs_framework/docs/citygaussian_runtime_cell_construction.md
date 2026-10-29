# CityGaussian Runtime Cell Construction

This note traces the official CityGaussian V1 runtime cell construction path in the read-only reference repository at `/home/luis/tese/CityGaussian`.

The runtime cells are produced during `render_large_lod.py::render_sets(...)`, before per-frame rendering. The per-frame `render_lod(...)` call consumes already-built `BlockedGaussian` cells; it does not construct them.

## Complete Call Chain

```text
render_large_lod.py::render_sets
-> render_large_lod.py::load_gaussians
-> scene.LargeScene.__init__
-> GaussianModelLOD.load_vq(...) or GaussianModelLOD.load_ply(...)
-> render_sets
-> BlockedGaussian(...)
-> BlockedGaussian.__init__
-> BlockedGaussian.cell_divider
-> utils.large_utils.block_filtering
-> utils.large_utils.contract_to_unisphere
```

Relevant source files:

- `/home/luis/tese/CityGaussian/render_large_lod.py`
- `/home/luis/tese/CityGaussian/scene/__init__.py`
- `/home/luis/tese/CityGaussian/scene/gaussian_model.py`
- `/home/luis/tese/CityGaussian/utils/large_utils.py`

## 1. `render_sets` Iterates LoD Configs

```python
for i in range(len(dataset.lod_configs)):
    dataset.model_path = dataset.lod_configs[i]
    lod_gs, scene = load_gaussians(dataset, iteration, load_vq)
    print(f"Init LoD {len(dataset.lod_configs)-i} with {lod_gs.get_xyz.shape[0]} points from {dataset.model_path}")
    lod_gs = BlockedGaussian(lod_gs, dataset, compute_cov3D_python=pp.compute_cov3D_python)
    lod_gs_list.append(lod_gs)
```

Transformation:

- For each YAML LoD path, `dataset.model_path` is set to that LoD model directory.
- The whole LoD model is loaded into a `GaussianModelLOD`.
- That complete model is immediately converted into a `BlockedGaussian`.
- One `BlockedGaussian` is produced per LoD level.

## 2. `load_gaussians` Constructs The Gaussian Model

```python
gaussians = getattr(modules, lp.model_config['name'])(lp.sh_degree, device=device, **lp.model_config['kwargs'])
scene = LargeScene(lp, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)
return gaussians, scene
```

Transformation:

- `lp.model_config['name']` is `GaussianModelLOD`.
- `lp.sh_degree` comes from YAML, `2` for MatrixCity LoD.
- `LargeScene` fills that Gaussian object from disk.

## 3. `LargeScene` Loads VQ Or PLY Data

```python
if self.load_vq:
    self.gaussians.load_vq(self.model_path)
elif self.loaded_iter:
    self.gaussians.load_ply(os.path.join(self.model_path,
                                        "point_cloud",
                                        "iteration_" + str(self.loaded_iter),
                                        "point_cloud.ply"))
```

Transformation:

- With `--load_vq`, official LoD rendering loads `extreme_saving`.
- Without `--load_vq`, it loads PLY.
- Either way, the `GaussianModelLOD` ends with tensors for xyz, opacity, SH features, scales, and rotations.

## 4. `BlockedGaussian.__init__` Stores Grid Metadata

```python
self.block_dim = lp.block_dim
self.num_cell = lp.block_dim[0] * lp.block_dim[1] * lp.block_dim[2]
self.aabb = lp.aabb
self.scale = scale
self.range = range

self.cell_divider(gaussians)
self.cell_corners = torch.stack(self.cell_corners, dim=0)
```

Transformation:

- MatrixCity uses `block_dim=[6,6,1]`, so `num_cell=36`.
- AABB is the official MatrixCity AABB from YAML.
- `cell_divider(...)` is where cells are actually created.

## 5. `cell_divider` Packs Per-Gaussian Features

```python
if self.compute_cov3D_python:
    geometry = gaussians.get_covariance(self.scale).to(self.device)
else:
    geometry = torch.cat([gaussians.get_scaling,
                          gaussians.get_rotation], dim=1)
self.feats = torch.cat([gaussians.get_xyz,
                        gaussians.get_opacity,
                        gaussians.get_features.reshape(geometry.shape[0], -1),
                        geometry], dim=1)
```

Transformation:

Each Gaussian becomes one row in `self.feats`:

```text
[x, y, z,
 opacity,
 flattened SH features,
 scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3]
```

For SH degree 2, flattened SH is `(2+1)^2 * 3 = 27` values.

## 6. `cell_divider` Loops Over 36 Cells

```python
for cell_idx in range(self.num_cell):
    cell_mask = block_filtering(cell_idx, self.feats[:, :3], self.aabb, self.block_dim, self.scale)
    self.cell_idxs.append(self.cell_idxs[-1] + cell_mask.sum())
    feat_list.append(self.feats[cell_mask])
```

Transformation:

- For each logical cell ID `0..35`, it computes a boolean mask over all Gaussian positions.
- Matching Gaussian rows are appended to `feat_list`.
- `cell_idxs` stores cumulative offsets so each cell can later be sliced out of concatenated `self.feats`.

## 7. `block_filtering` Converts Cell ID To 3D Grid Coordinate

```python
block_id_z = block_id // (block_dim[0] * block_dim[1]);
block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0];
block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0];
```

Transformation:

- Linear cell ID is decoded as x-fastest, then y, then z.
- For `[6,6,1]`, all cells are in `z=0`; x/y form the 6x6 grid.

## 8. `contract_to_unisphere` Maps World Positions To `[0,1]^3`

```python
aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
x = (x - aabb_min) / (aabb_max - aabb_min)
x = x * 2 - 1  # aabb is at [-1, 1]
mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
mask = mag.squeeze(-1) > 1

x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
return x
```

Transformation:

- World xyz is normalized by the official AABB.
- AABB interior maps to `[-1,1]`.
- Points outside the AABB are contracted with infinity norm in this call.
- Contracted coordinates are remapped to `[0,1]`.

## 9. `block_filtering` Builds The Cell Bounds In Contracted Space

```python
xyz = contract_to_unisphere(xyz_tensor, aabb, ord=torch.inf)
min_x, max_x = float(block_id_x) / block_dim[0], float(block_id_x + 1) / block_dim[0]
min_y, max_y = float(block_id_y) / block_dim[1], float(block_id_y + 1) / block_dim[1]
min_z, max_z = float(block_id_z) / block_dim[2], float(block_id_z + 1) / block_dim[2]
```

Transformation:

- Cell bounds are uniform intervals in contracted `[0,1]^3`.
- For x/y with `block_dim=6`, each cell spans `1/6`.
- For z with `block_dim=1`, the single z interval spans all `[0,1]`.

## 10. `block_filtering` Returns The Cell Mask

```python
block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)

if mask_only:
    return block_mask
```

Transformation:

- A Gaussian belongs to a runtime cell if its contracted position lies inside that cell interval.
- This is not reading pre-generated chunks; it is assigning whole-scene Gaussians to cells in memory.

## 11. `cell_divider` Computes Runtime Bounds And Scale Stats

```python
xyz_median = torch.median(xyz[cell_mask], dim=0)[0]
delta_median = torch.median(torch.abs(xyz[cell_mask] - xyz_median), dim=0)[0]
xyz_min = xyz_median - n * delta_median
xyz_min = torch.max(xyz_min, torch.min(xyz[cell_mask], dim=0)[0])
xyz_max = xyz_median + n * delta_median
xyz_max = torch.min(xyz_max, torch.max(xyz[cell_mask], dim=0)[0])
```

Transformation:

- For each cell, it computes a robust MAD-like world-space bound.
- `n=4` by default.
- Bounds are clamped to the true min/max of the cell's Gaussians.

Then corners are built:

```python
corners = torch.tensor([[xyz_min[0], xyz_min[1], xyz_min[2]],
                       ...
                       [xyz_max[0], xyz_max[1], xyz_max[2]]], device=xyz.device)
self.cell_corners.append(corners)
self.avg_scalings.append(torch.mean(scaling[cell_mask], dim=0))
```

Transformation:

- Each cell gets 8 world-space corners.
- Each cell also stores mean Gaussian scale vector.

Finally:

```python
self.feats = torch.cat(feat_list, dim=0)
self.avg_scalings = torch.max(torch.stack(self.avg_scalings, dim=0), dim=-1).values
```

Transformation:

- Cell feature tensors are concatenated into one packed tensor.
- `cell_idxs` provides slice boundaries.
- `avg_scalings` becomes one scalar per cell: max component of the mean scale vector.

## Produced Runtime Cell Structure

```text
BlockedGaussian
|-- feats: all Gaussian rows sorted/grouped by cell
|-- cell_idxs: cumulative offsets into feats
|-- cell_corners: [num_cell, 8, 3] robust world-space bounds
|-- avg_scalings: [num_cell] scalar scale statistic
|-- block_dim: [6,6,1]
|-- aabb: official MatrixCity AABB
`-- num_cell: 36
```

The key point: official CityGaussian V1 runtime cell construction is in-memory, per loaded LoD model, through `BlockedGaussian.cell_divider`; it does not require chunk directories or `lod-meta.json`.
