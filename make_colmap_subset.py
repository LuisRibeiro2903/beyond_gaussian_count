from pathlib import Path
import copy
import json
import shutil
import sys

sys.path.append(".")
from read_write_model import read_model, write_model

STRIDE = 2
SRC = Path("/home/luis/tese/CityGaussian/data/matrix_city/aerial/train/block_all")
DST = Path(f"/home/luis/tese/gsplat_data/aerial_subset_{STRIDE}")

src_images_dir = SRC / "input"
src_sparse_dir = SRC / "sparse" / "0"
src_transforms_path = SRC / "transforms.json"

dst_images_dir = DST / "images"
dst_sparse_dir = DST / "sparse" / "0"
dst_transforms_path = DST / "transforms.json"
dst_images_dir.mkdir(parents=True, exist_ok=True)
dst_sparse_dir.mkdir(parents=True, exist_ok=True)


def write_subset_transforms(src_json, src_image_dir, dst_json, image_names):
    if not src_json.exists():
        print(f"missing transforms.json, skipping NeRF-style output: {src_json}")
        return 0

    with open(src_json, "r") as f:
        meta = json.load(f)

    frames = meta.get("frames", [])
    src_image_names = sorted(p.name for p in src_image_dir.iterdir() if p.is_file())

    if len(src_image_names) != len(frames):
        raise ValueError(
            "Cannot map flattened images to transforms.json frames: "
            f"{len(src_image_names)} images vs {len(frames)} frames"
        )

    frame_by_image_name = dict(zip(src_image_names, frames))
    subset_frames = []
    missing_frames = 0

    for image_name in image_names:
        frame = frame_by_image_name.get(image_name)
        if frame is None:
            missing_frames += 1
            continue

        new_frame = copy.deepcopy(frame)
        new_frame["file_path"] = f"images/{image_name}"
        subset_frames.append(new_frame)

    subset_meta = copy.deepcopy(meta)
    subset_meta["frames"] = subset_frames

    with open(dst_json, "w") as f:
        json.dump(subset_meta, f, indent=2)
        f.write("\n")

    if missing_frames > 0:
        print(f"Warning: {missing_frames} selected images had no transform frame.")

    return len(subset_frames)

cameras, images, points3D = read_model(str(src_sparse_dir), ext=".bin")

# Select every STRIDE-th registered image by filename order
sorted_images = sorted(images.items(), key=lambda kv: kv[1].name)
keep_image_ids = {
    image_id for i, (image_id, img) in enumerate(sorted_images)
    if i % STRIDE == 0
}

new_images = {}
used_camera_ids = set()

for image_id in keep_image_ids:
    img = images[image_id]
    src_img = src_images_dir / img.name
    dst_img = dst_images_dir / img.name

    if not src_img.exists():
        print(f"missing image, skipping: {src_img}")
        continue

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if not dst_img.exists():
        dst_img.symlink_to(src_img)

    new_images[image_id] = img
    used_camera_ids.add(img.camera_id)

new_cameras = {
    cam_id: cam for cam_id, cam in cameras.items()
    if cam_id in used_camera_ids
}

# Keep only 3D points observed by kept images
new_points3D = {}

for point_id, p in points3D.items():
    keep_obs = [i for i, img_id in enumerate(p.image_ids) if img_id in new_images]

    if len(keep_obs) < 2:
        continue

    new_points3D[point_id] = p._replace(
        image_ids=p.image_ids[keep_obs],
        point2D_idxs=p.point2D_idxs[keep_obs],
    )

write_model(new_cameras, new_images, new_points3D, str(dst_sparse_dir), ext=".bin")
num_transform_frames = write_subset_transforms(
    src_transforms_path,
    src_images_dir,
    dst_transforms_path,
    [img.name for image_id, img in sorted_images if image_id in new_images],
)

print("Done")
print("Images:", len(new_images))
print("Cameras:", len(new_cameras))
print("Points3D:", len(new_points3D))
print("Transform frames:", num_transform_frames)
print("Output:", DST)
