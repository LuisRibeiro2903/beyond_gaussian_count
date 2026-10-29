import struct

from scripts.alod.build_uni10k_dataset import (
    image_semantic_digest,
    point_semantic_digest,
    strip_image_observations,
    strip_point_tracks_and_write_ply,
)


def test_streaming_correspondence_stripping_preserves_semantics(tmp_path):
    source_images = tmp_path / "source-images.bin"
    target_images = tmp_path / "target-images.bin"
    image_records = [
        (1, "aerial/a.jpg", 4, (1.0, 2.0, 7)),
        (2, "ground/b.jpg", 9, (3.0, 4.0, 8)),
    ]
    with source_images.open("wb") as handle:
        handle.write(struct.pack("<Q", len(image_records)))
        for image_id, name, camera_id, observation in image_records:
            handle.write(
                struct.pack(
                    "<idddddddi", image_id, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, camera_id
                )
            )
            handle.write(name.encode() + b"\0")
            handle.write(struct.pack("<Qddq", 1, *observation))

    result = strip_image_observations(source_images, target_images)
    source_count, source_digest = image_semantic_digest(source_images)
    target_count, target_digest = image_semantic_digest(target_images)
    assert source_count == target_count == result["registered_images"] == 2
    assert source_digest == target_digest == result["registered_image_semantic_sha256"]
    assert result["source_observations_removed"] == 2
    assert result["aerial_images"] == result["ground_images"] == 1

    source_points = tmp_path / "source-points3D.bin"
    target_points = tmp_path / "target-points3D.bin"
    target_ply = tmp_path / "target-points3D.ply"
    points = [
        (10, (1.0, 2.0, 3.0), (4, 5, 6), 0.1, [(1, 2)]),
        (11, (-1.0, -2.0, -3.0), (7, 8, 9), 0.2, [(2, 3), (4, 5)]),
    ]
    with source_points.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point_id, xyz, rgb, error, track in points:
            handle.write(struct.pack("<QdddBBBd", point_id, *xyz, *rgb, error))
            handle.write(struct.pack("<Q", len(track)))
            for image_id, point2d_index in track:
                handle.write(struct.pack("<II", image_id, point2d_index))

    result = strip_point_tracks_and_write_ply(
        source_points, target_points, target_ply
    )
    source_count, source_digest = point_semantic_digest(source_points)
    target_count, target_digest = point_semantic_digest(target_points)
    assert source_count == target_count == result["sparse_points"] == 2
    assert source_digest == target_digest == result["sparse_point_semantic_sha256"]
    assert result["source_track_entries_removed"] == 3
    assert target_ply.is_file()
