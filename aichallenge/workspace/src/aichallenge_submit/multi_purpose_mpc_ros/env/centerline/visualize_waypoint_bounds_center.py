#!/usr/bin/env python3
"""Plot the Center reference path, its physical bounds, and waypoint IDs."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path,
                        default=HERE / "traj_center_mincurv_capped.csv")
    parser.add_argument("--bounds", type=Path,
                        default=HERE / "waypoint_bounds_center_mincurv_capped.csv")
    parser.add_argument("--output", type=Path,
                        default=HERE / "centerline_bounds_waypoints.png")
    parser.add_argument("--label-step", type=int, default=10,
                        help="Label every Nth waypoint; all WP positions are drawn.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def boundary_coordinates(traj, bounds):
    # Production bounds contain exact map-frame coordinates. Prefer these to
    # reconstructing boundaries from the reference heading and lateral bounds.
    columns = {"left_x", "left_y", "right_x", "right_y"}
    if columns.issubset(bounds.columns):
        return tuple(bounds[name].to_numpy() for name in
                     ("left_x", "left_y", "right_x", "right_y"))

    psi = traj["psi_rad"].to_numpy()
    normal_x = -np.sin(psi)
    normal_y = np.cos(psi)
    x = traj["x_m"].to_numpy()
    y = traj["y_m"].to_numpy()
    upper = bounds["ub"].to_numpy()
    lower = bounds["lb"].to_numpy()
    return (x + normal_x * upper, y + normal_y * upper,
            x + normal_x * lower, y + normal_y * lower)


def close_loop(values):
    return np.append(values, values[0])


def main():
    args = parse_args()
    if args.label_step < 1:
        raise ValueError("--label-step must be at least 1")

    traj = pd.read_csv(args.trajectory)
    bounds = pd.read_csv(args.bounds)
    if len(traj) != len(bounds):
        raise ValueError(
            f"trajectory/bounds row mismatch: {len(traj)} != {len(bounds)}")

    x = traj["x_m"].to_numpy()
    y = traj["y_m"].to_numpy()
    left_x, left_y, right_x, right_y = boundary_coordinates(traj, bounds)

    # Offset the large map coordinates to keep axis labels readable.
    origin_x = float(np.floor(np.min(np.r_[left_x, right_x]) / 10.0) * 10.0)
    origin_y = float(np.floor(np.min(np.r_[left_y, right_y]) / 10.0) * 10.0)
    local = lambda values, origin: np.asarray(values) - origin

    fig, ax = plt.subplots(figsize=(15, 11), constrained_layout=True)
    ax.fill(
        np.r_[local(left_x, origin_x), local(right_x[::-1], origin_x)],
        np.r_[local(left_y, origin_y), local(right_y[::-1], origin_y)],
        color="#d9e8f5", alpha=0.42, label="Drivable width", zorder=0)
    ax.plot(local(close_loop(left_x), origin_x),
            local(close_loop(left_y), origin_y), color="#e53935",
            linewidth=1.5, label="Left boundary (ub)")
    ax.plot(local(close_loop(right_x), origin_x),
            local(close_loop(right_y), origin_y), color="#1976d2",
            linewidth=1.5, label="Right boundary (lb)")
    ax.plot(local(close_loop(x), origin_x), local(close_loop(y), origin_y),
            color="black", linewidth=1.2, label="Center reference")
    ax.scatter(local(x, origin_x), local(y, origin_y), s=11,
               color="#ff9800", edgecolors="black", linewidths=0.25,
               label=f"Waypoints ({len(x)})", zorder=3)

    for waypoint_id in range(0, len(x), args.label_step):
        ax.annotate(str(waypoint_id),
                    (x[waypoint_id] - origin_x, y[waypoint_id] - origin_y),
                    xytext=(3, 3), textcoords="offset points", fontsize=7,
                    color="#5d4037", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.55)
    ax.set_title("Current Center Reference, Physical Bounds, and Waypoints\n"
                 f"labels: every {args.label_step} WP")
    ax.set_xlabel(f"map X - {origin_x:.0f} [m]")
    ax.set_ylabel(f"map Y - {origin_y:.0f} [m]")
    ax.legend(loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output} ({len(x)} waypoints)")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
