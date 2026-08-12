"""Generate a ROS occupancy map from paired left/right track boundaries.

The input CSV is expected to contain ``left_x``, ``left_y``, ``right_x`` and
``right_y`` columns.  Consecutive boundary pairs, including the final pair and
the first pair, are rasterized as quadrilaterals.  White pixels are drivable
and black pixels are occupied, matching :mod:`multi_purpose_mpc_ros.core.map`.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import yaml


DEFAULT_X_OFFSET = 5.332886
DEFAULT_Y_OFFSET = -75.727413
REQUIRED_COLUMNS = ("left_x", "left_y", "right_x", "right_y")


@dataclass(frozen=True)
class BoundaryData:
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class GeneratedMap:
    image: Image.Image
    origin: tuple[float, float, float]
    resolution: float
    track_widths: np.ndarray
    adjacent_steps: np.ndarray


def load_boundary_csv(
    csv_path: Path, x_offset: float = DEFAULT_X_OFFSET,
    y_offset: float = DEFAULT_Y_OFFSET,
) -> BoundaryData:
    """Load and validate paired boundary coordinates from *csv_path*."""
    left = []
    right = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{csv_path}: required CSV columns are missing: {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in REQUIRED_COLUMNS]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path}:{line_number}: boundary coordinate is not numeric"
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{csv_path}:{line_number}: boundary coordinate is not finite"
                )
            left.append((values[0] + x_offset, values[1] + y_offset))
            right.append((values[2] + x_offset, values[3] + y_offset))

    if len(left) < 3:
        raise ValueError(f"{csv_path}: at least three boundary pairs are required")

    return BoundaryData(
        left=np.asarray(left, dtype=np.float64),
        right=np.asarray(right, dtype=np.float64),
    )


def _world_to_pixel(
    point: np.ndarray, origin_x: float, origin_y: float,
    resolution: float, image_height: int,
) -> tuple[int, int]:
    x = int((float(point[0]) - origin_x) / resolution + 0.5)
    y = int((image_height - 1) - (float(point[1]) - origin_y) / resolution + 0.5)
    return x, y


def generate_occupancy_grid(
    boundary: BoundaryData, resolution: float = 0.1, padding: float = 2.0,
    max_step: float = 1.0,
) -> GeneratedMap:
    """Rasterize a closed track ribbon into an 8-bit occupancy image."""
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if padding < 0.0:
        raise ValueError("padding must not be negative")
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    if boundary.left.shape != boundary.right.shape or boundary.left.shape[1:] != (2,):
        raise ValueError("left and right boundaries must be equally sized Nx2 arrays")

    track_widths = np.linalg.norm(boundary.left - boundary.right, axis=1)
    if np.any(track_widths <= resolution):
        index = int(np.argmin(track_widths))
        raise ValueError(
            f"boundary pair {index} is only {track_widths[index]:.3f} m wide "
            f"at {resolution:.3f} m/px"
        )

    left_steps = np.linalg.norm(boundary.left - np.roll(boundary.left, -1, axis=0), axis=1)
    right_steps = np.linalg.norm(boundary.right - np.roll(boundary.right, -1, axis=0), axis=1)
    adjacent_steps = np.maximum(left_steps, right_steps)
    largest_step = float(np.max(adjacent_steps))
    if largest_step > max_step:
        index = int(np.argmax(adjacent_steps))
        raise ValueError(
            f"boundary discontinuity between rows {index} and "
            f"{(index + 1) % len(boundary.left)} is {largest_step:.3f} m "
            f"(limit {max_step:.3f} m); increase --max-step only if intentional"
        )

    all_points = np.vstack((boundary.left, boundary.right))
    origin_x = float(np.min(all_points[:, 0]) - padding)
    origin_y = float(np.min(all_points[:, 1]) - padding)
    max_x = float(np.max(all_points[:, 0]) + padding)
    max_y = float(np.max(all_points[:, 1]) + padding)
    image_width = int(math.ceil((max_x - origin_x) / resolution)) + 1
    image_height = int(math.ceil((max_y - origin_y) / resolution)) + 1

    image = Image.new("L", (image_width, image_height), color=0)
    draw = ImageDraw.Draw(image)
    left_pixels = [
        _world_to_pixel(point, origin_x, origin_y, resolution, image_height)
        for point in boundary.left
    ]
    right_pixels = [
        _world_to_pixel(point, origin_x, origin_y, resolution, image_height)
        for point in boundary.right
    ]

    point_count = len(left_pixels)
    for index in range(point_count):
        next_index = (index + 1) % point_count
        draw.polygon(
            (
                left_pixels[index],
                left_pixels[next_index],
                right_pixels[next_index],
                right_pixels[index],
            ),
            fill=255,
        )

    # Seal sub-pixel rounding gaps along both walls without shrinking the track.
    draw.line(left_pixels + [left_pixels[0]], fill=255, width=1)
    draw.line(right_pixels + [right_pixels[0]], fill=255, width=1)

    image_array = np.asarray(image)
    if not np.any(image_array == 255):
        raise RuntimeError("generated occupancy map contains no drivable cells")

    # Every paired-boundary midpoint must be inside the generated free region.
    midpoints = (boundary.left + boundary.right) * 0.5
    invalid_midpoints = []
    for index, midpoint in enumerate(midpoints):
        pixel_x, pixel_y = _world_to_pixel(
            midpoint, origin_x, origin_y, resolution, image_height
        )
        if image_array[pixel_y, pixel_x] != 255:
            invalid_midpoints.append(index)
    if invalid_midpoints:
        preview = ", ".join(str(index) for index in invalid_midpoints[:10])
        raise RuntimeError(
            f"{len(invalid_midpoints)} center samples are occupied after rasterization "
            f"(first indices: {preview})"
        )

    return GeneratedMap(
        image=image,
        origin=(origin_x, origin_y, 0.0),
        resolution=resolution,
        track_widths=track_widths,
        adjacent_steps=adjacent_steps,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_map_files(
    generated: GeneratedMap, output_dir: Path,
    name: str = "occupancy_grid_map", overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Write PGM, ROS YAML and PNG preview files atomically."""
    if not name or Path(name).name != name:
        raise ValueError("name must be a non-empty file basename")

    pgm_path = output_dir / f"{name}.pgm"
    yaml_path = output_dir / f"{name}.yaml"
    preview_path = output_dir / f"{name}_preview.png"
    output_paths = (pgm_path, yaml_path, preview_path)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output file already exists; use --overwrite to replace it: "
            + ", ".join(existing)
        )

    pgm_buffer = io.BytesIO()
    generated.image.save(pgm_buffer, format="PPM")
    preview_buffer = io.BytesIO()
    generated.image.save(preview_buffer, format="PNG")
    yaml_data = {
        "image": pgm_path.name,
        "resolution": float(generated.resolution),
        "origin": [float(value) for value in generated.origin],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    yaml_bytes = yaml.safe_dump(
        yaml_data, sort_keys=False, default_flow_style=False
    ).encode("utf-8")

    _atomic_write(pgm_path, pgm_buffer.getvalue())
    _atomic_write(yaml_path, yaml_bytes)
    _atomic_write(preview_path, preview_buffer.getvalue())
    return output_paths


def _build_parser() -> argparse.ArgumentParser:
    source_default = Path(__file__).resolve().parents[1] / "boundary_true.csv"
    parser = argparse.ArgumentParser(
        description="Generate a closed-track PGM/YAML occupancy map from boundary_true.csv."
    )
    parser.add_argument(
        "input_csv", nargs="?", type=Path, default=source_default,
        help=f"input boundary CSV (default: {source_default})",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="directory in which to write the PGM, YAML and preview PNG",
    )
    parser.add_argument("--name", default="occupancy_grid_map", help="output basename")
    parser.add_argument("--resolution", type=float, default=0.1, help="metres per pixel")
    parser.add_argument("--padding", type=float, default=2.0, help="occupied padding in metres")
    parser.add_argument(
        "--x-offset", type=float, default=DEFAULT_X_OFFSET,
        help="x offset applied to every CSV coordinate",
    )
    parser.add_argument(
        "--y-offset", type=float, default=DEFAULT_Y_OFFSET,
        help="y offset applied to every CSV coordinate",
    )
    parser.add_argument(
        "--max-step", type=float, default=1.0,
        help="maximum allowed adjacent boundary-point gap in metres",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    boundary = load_boundary_csv(args.input_csv, args.x_offset, args.y_offset)
    generated = generate_occupancy_grid(
        boundary,
        resolution=args.resolution,
        padding=args.padding,
        max_step=args.max_step,
    )
    pgm_path, yaml_path, preview_path = write_map_files(
        generated, args.output_dir, args.name, args.overwrite
    )

    points = np.vstack((boundary.left, boundary.right))
    print(f"Loaded boundary pairs : {len(boundary.left)}")
    print(
        "World extent          : "
        f"x=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
        f"y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]"
    )
    print(
        "Paired track width    : "
        f"min={generated.track_widths.min():.3f} m, "
        f"mean={generated.track_widths.mean():.3f} m, "
        f"max={generated.track_widths.max():.3f} m"
    )
    print(
        f"Largest closed-loop gap: {generated.adjacent_steps.max():.3f} m"
    )
    print(
        f"Raster                : {generated.image.width}x{generated.image.height} px "
        f"at {generated.resolution:.3f} m/px"
    )
    print(f"PGM                   : {pgm_path}")
    print(f"YAML                  : {yaml_path}")
    print(f"Preview               : {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
