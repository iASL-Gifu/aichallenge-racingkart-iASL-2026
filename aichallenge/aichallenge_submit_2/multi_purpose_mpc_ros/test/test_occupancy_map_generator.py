import csv
import math
from pathlib import Path

from PIL import Image
import pytest
import yaml

from multi_purpose_mpc_ros.tools.occupancy_map_generator import (
    generate_occupancy_grid,
    load_boundary_csv,
    write_map_files,
)


def write_ring_csv(path: Path, point_count: int = 32) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("left_x", "left_y", "right_x", "right_y")
        )
        writer.writeheader()
        for index in range(point_count):
            angle = 2.0 * math.pi * index / point_count
            writer.writerow(
                {
                    "left_x": 5.0 * math.cos(angle),
                    "left_y": 5.0 * math.sin(angle),
                    "right_x": 3.0 * math.cos(angle),
                    "right_y": 3.0 * math.sin(angle),
                }
            )


def test_generates_closed_free_track_and_ros_yaml(tmp_path):
    csv_path = tmp_path / "boundary_true.csv"
    write_ring_csv(csv_path)

    boundary = load_boundary_csv(csv_path, x_offset=0.0, y_offset=0.0)
    generated = generate_occupancy_grid(
        boundary, resolution=0.1, padding=1.0, max_step=1.1
    )
    pgm_path, yaml_path, preview_path = write_map_files(
        generated, tmp_path / "map"
    )

    assert pgm_path.exists()
    assert preview_path.exists()
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert metadata["image"] == "occupancy_grid_map.pgm"
    assert metadata["resolution"] == pytest.approx(0.1)
    assert metadata["origin"] == pytest.approx([-6.0, -6.0, 0.0])

    image = Image.open(pgm_path)
    center = image.width // 2, image.height // 2
    track = center[0] + 40, center[1]
    assert image.getpixel(center) == 0
    assert image.getpixel(track) == 255
    assert image.getpixel((0, 0)) == 0


def test_refuses_to_overwrite_existing_map(tmp_path):
    csv_path = tmp_path / "boundary_true.csv"
    write_ring_csv(csv_path)
    generated = generate_occupancy_grid(
        load_boundary_csv(csv_path, 0.0, 0.0),
        resolution=0.1,
        padding=1.0,
        max_step=1.1,
    )
    output_dir = tmp_path / "map"
    write_map_files(generated, output_dir)

    with pytest.raises(FileExistsError):
        write_map_files(generated, output_dir)


def test_rejects_broken_closed_loop(tmp_path):
    csv_path = tmp_path / "boundary_true.csv"
    write_ring_csv(csv_path)
    boundary = load_boundary_csv(csv_path, 0.0, 0.0)

    with pytest.raises(ValueError, match="boundary discontinuity"):
        generate_occupancy_grid(
            boundary, resolution=0.1, padding=1.0, max_step=0.1
        )
