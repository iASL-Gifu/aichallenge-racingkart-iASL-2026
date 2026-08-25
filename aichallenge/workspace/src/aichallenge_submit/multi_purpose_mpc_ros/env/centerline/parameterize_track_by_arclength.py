#!/usr/bin/env python3
"""Convert track.csv to a validated closed-loop arc-length representation.

This is intentionally only the first stage of the centerline generation
pipeline.  It does not resample or smooth the input geometry.  The output can
subsequently be used as the canonical input for periodic spline processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("x_m", "y_m", "w_tr_right_m", "w_tr_left_m")
DEFAULT_X_OFFSET_M = 5.332886
DEFAULT_Y_OFFSET_M = -75.727413


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Validate track.csv, remove its duplicated closing point, apply "
            "the MPC coordinate offset, and calculate closed-loop arc length."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir.parent / "track.csv",
        help="Input track CSV (default: ../track.csv relative to this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "track_arclength.csv",
        help="Output CSV (default: track_arclength.csv beside this script)",
    )
    parser.add_argument("--x-offset", type=float, default=DEFAULT_X_OFFSET_M)
    parser.add_argument("--y-offset", type=float, default=DEFAULT_Y_OFFSET_M)
    parser.add_argument(
        "--duplicate-tolerance",
        type=float,
        default=1.0e-6,
        help="Distance [m] at or below which two coordinates are duplicates",
    )
    return parser.parse_args()


def build_arc_length_table(
    source: pd.DataFrame,
    *,
    x_offset_m: float,
    y_offset_m: float,
    duplicate_tolerance_m: float,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(source) < 3:
        raise ValueError("At least three track points are required")
    if duplicate_tolerance_m < 0.0:
        raise ValueError("duplicate tolerance must be non-negative")

    values = source.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Input contains NaN or infinite values")
    if (values[["w_tr_right_m", "w_tr_left_m"]] <= 0.0).any().any():
        raise ValueError("Track widths must be positive")

    input_rows = len(values)
    closing_distance = float(
        np.hypot(
            values.iloc[-1]["x_m"] - values.iloc[0]["x_m"],
            values.iloc[-1]["y_m"] - values.iloc[0]["y_m"],
        )
    )
    removed_duplicate_endpoint = closing_distance <= duplicate_tolerance_m
    if removed_duplicate_endpoint:
        values = values.iloc[:-1].copy()
    else:
        values = values.copy()

    x = values["x_m"].to_numpy(dtype=float) + x_offset_m
    y = values["y_m"].to_numpy(dtype=float) + y_offset_m
    segment_length = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    duplicate_segments = np.flatnonzero(segment_length <= duplicate_tolerance_m)
    if duplicate_segments.size:
        indices = ", ".join(map(str, duplicate_segments[:10]))
        raise ValueError(
            "Zero-length or duplicate closed-loop segments found after endpoint "
            f"processing at row indices: {indices}"
        )

    s_m = np.empty(len(values), dtype=float)
    s_m[0] = 0.0
    s_m[1:] = np.cumsum(segment_length[:-1])
    if not np.all(np.diff(s_m) > 0.0):
        raise ValueError("Calculated arc length is not strictly increasing")

    result = pd.DataFrame(
        {
            "source_index": values.index.to_numpy(dtype=int),
            "s_m": s_m,
            "x_m": x,
            "y_m": y,
            "w_tr_right_m": values["w_tr_right_m"].to_numpy(dtype=float),
            "w_tr_left_m": values["w_tr_left_m"].to_numpy(dtype=float),
            "segment_length_to_next_m": segment_length,
        }
    )
    diagnostics: dict[str, float | int | bool] = {
        "input_rows": input_rows,
        "output_rows": len(result),
        "removed_duplicate_endpoint": removed_duplicate_endpoint,
        "input_closing_distance_m": closing_distance,
        "course_length_m": float(segment_length.sum()),
        "segment_min_m": float(segment_length.min()),
        "segment_mean_m": float(segment_length.mean()),
        "segment_max_m": float(segment_length.max()),
    }
    return result, diagnostics


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    result, diagnostics = build_arc_length_table(
        source,
        x_offset_m=args.x_offset,
        y_offset_m=args.y_offset,
        duplicate_tolerance_m=args.duplicate_tolerance,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.10f")

    print(f"Input:  {args.input.resolve()}")
    print(f"Output: {args.output.resolve()}")
    print(
        f"Rows: {diagnostics['input_rows']} -> {diagnostics['output_rows']} "
        f"(duplicate endpoint removed: {diagnostics['removed_duplicate_endpoint']})"
    )
    print(
        "Offsets: "
        f"x={args.x_offset:+.6f} m, y={args.y_offset:+.6f} m"
    )
    print(f"Closed-loop course length: {diagnostics['course_length_m']:.6f} m")
    print(
        "Segment length [m]: "
        f"min={diagnostics['segment_min_m']:.6f}, "
        f"mean={diagnostics['segment_mean_m']:.6f}, "
        f"max={diagnostics['segment_max_m']:.6f}"
    )


if __name__ == "__main__":
    main()
