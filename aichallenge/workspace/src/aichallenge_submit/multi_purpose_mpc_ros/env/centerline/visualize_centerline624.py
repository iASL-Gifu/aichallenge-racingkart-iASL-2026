#!/usr/bin/env python3
"""Visualize the generated 624-point center trajectory and its boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=script_dir / "traj_center624_periodic_spline.csv",
    )
    parser.add_argument(
        "--bounds",
        type=Path,
        default=script_dir / "waypoint_bounds_center624_periodic_spline.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "centerline624_trajectory_and_bounds.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = pd.read_csv(args.trajectory)
    bounds = pd.read_csv(args.bounds)
    if len(trajectory) != len(bounds):
        raise ValueError("Trajectory and bounds row counts do not match")

    x = trajectory["x_m"].to_numpy(dtype=float)
    y = trajectory["y_m"].to_numpy(dtype=float)
    s = trajectory["s_m"].to_numpy(dtype=float)
    kappa = trajectory["kappa_radpm"].to_numpy(dtype=float)
    left_x = bounds["left_x"].to_numpy(dtype=float)
    left_y = bounds["left_y"].to_numpy(dtype=float)
    right_x = bounds["right_x"].to_numpy(dtype=float)
    right_y = bounds["right_y"].to_numpy(dtype=float)
    left_width = bounds["ub"].to_numpy(dtype=float)
    right_width = -bounds["lb"].to_numpy(dtype=float)

    closed = lambda values: np.r_[values, values[0]]
    min_left_index = int(np.argmin(left_width))
    min_right_index = int(np.argmin(right_width))

    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.55, 1.0))
    axis_xy = figure.add_subplot(grid[:, 0])
    axis_width = figure.add_subplot(grid[0, 1])
    axis_kappa = figure.add_subplot(grid[1, 1])

    polygon_x = np.r_[left_x, right_x[::-1]]
    polygon_y = np.r_[left_y, right_y[::-1]]
    axis_xy.fill(polygon_x, polygon_y, color="#d9d9d9", alpha=0.65, label="Corridor")
    axis_xy.plot(closed(left_x), closed(left_y), color="#d62728", linewidth=1.5, label="Left boundary")
    axis_xy.plot(closed(right_x), closed(right_y), color="#1f77b4", linewidth=1.5, label="Right boundary")
    axis_xy.plot(closed(x), closed(y), color="black", linewidth=1.2, label="Center trajectory")
    axis_xy.scatter(x[0], y[0], s=65, color="#2ca02c", marker="o", zorder=5, label="Start")
    axis_xy.scatter(
        right_x[min_right_index], right_y[min_right_index],
        s=75, color="#ff7f0e", marker="x", linewidths=2.2, zorder=6,
        label=f"Min right width: {right_width[min_right_index]:.3f} m",
    )
    for index in range(0, len(x), 48):
        axis_xy.plot(
            [left_x[index], right_x[index]],
            [left_y[index], right_y[index]],
            color="#777777", linewidth=0.45, alpha=0.55,
        )
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.set_xlabel("X [m]")
    axis_xy.set_ylabel("Y [m]")
    axis_xy.set_title("624-point steering-constrained Centerline")
    axis_xy.grid(True, alpha=0.25)
    axis_xy.legend(loc="best")

    axis_width.plot(s, left_width, color="#d62728", label="Left width (ub)")
    axis_width.plot(s, right_width, color="#1f77b4", label="Right width (-lb)")
    axis_width.scatter(s[min_left_index], left_width[min_left_index], color="#d62728", zorder=4)
    axis_width.scatter(s[min_right_index], right_width[min_right_index], color="#ff7f0e", zorder=4)
    axis_width.axhline(0.8, color="#555555", linestyle="--", linewidth=1.0, label="Vehicle half-width 0.8 m")
    axis_width.set_ylabel("Width [m]")
    axis_width.set_title("Boundary widths")
    axis_width.grid(True, alpha=0.25)
    axis_width.legend(loc="best")

    axis_kappa.plot(s, kappa, color="#9467bd", linewidth=1.1, label="kappa")
    axis_kappa.axhline(0.251622, color="#555555", linestyle="--", linewidth=0.9)
    axis_kappa.axhline(-0.251622, color="#555555", linestyle="--", linewidth=0.9, label="Steering-angle curvature limit")
    axis_kappa.set_xlabel("Arc length s [m]")
    axis_kappa.set_ylabel("Curvature [rad/m]")
    axis_kappa.set_title("Constrained curvature")
    axis_kappa.grid(True, alpha=0.25)
    axis_kappa.legend(loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Saved: {args.output.resolve()}")
    print(f"Rows: trajectory={len(trajectory)}, bounds={len(bounds)}")
    print(f"Minimum left width:  idx={min_left_index}, s={s[min_left_index]:.3f} m, width={left_width[min_left_index]:.6f} m")
    print(f"Minimum right width: idx={min_right_index}, s={s[min_right_index]:.3f} m, width={right_width[min_right_index]:.6f} m")


if __name__ == "__main__":
    main()
