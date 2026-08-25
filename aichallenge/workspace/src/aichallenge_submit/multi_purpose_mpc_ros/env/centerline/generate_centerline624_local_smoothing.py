#!/usr/bin/env python3
"""Generate a 624-point Centerline constrained by the MPC's discrete geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


X_OFFSET = 5.332886
Y_OFFSET = -75.727413


def arguments() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=here / "track_arclength.csv")
    parser.add_argument("--boundary", type=Path, default=here / "boundary.csv")
    parser.add_argument("--points", type=int, default=624)
    parser.add_argument("--base-window", type=int, default=25)
    parser.add_argument("--local-window", type=int, default=21)
    parser.add_argument("--derivative-window", type=int, default=11)
    parser.add_argument("--mpc-curvature-window", type=int, default=7)
    parser.add_argument("--mpc-stride", type=int, default=2)
    parser.add_argument("--rate-trigger", type=float, default=0.58)
    parser.add_argument("--rate-limit", type=float, default=0.60)
    parser.add_argument("--kappa-trigger", type=float, default=0.24)
    parser.add_argument("--padding-points", type=int, default=5)
    parser.add_argument("--transition-sigma", type=float, default=5.0)
    parser.add_argument("--local-passes", type=int, default=150)
    parser.add_argument("--speed", type=float, default=9.694)
    parser.add_argument("--wheelbase", type=float, default=1.087)
    parser.add_argument("--understeer-coeff", type=float, default=0.002)
    parser.add_argument("--vx-mps", type=float, default=9.694)
    parser.add_argument("--trajectory-name", default="traj_center624_local_smooth.csv")
    parser.add_argument("--bounds-name", default="waypoint_bounds_center624_local_smooth.csv")
    parser.add_argument("--report-name", default="center624_local_smoothing_report.csv")
    parser.add_argument("--plot-name", default="center624_local_smoothing.png")
    return parser.parse_args()


def geometry(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, ...]:
    dx = savgol_filter(x, window, 3, deriv=1, mode="wrap")
    dy = savgol_filter(y, window, 3, deriv=1, mode="wrap")
    ddx = savgol_filter(x, window, 3, deriv=2, mode="wrap")
    ddy = savgol_filter(y, window, 3, deriv=2, mode="wrap")
    norm_sq = dx * dx + dy * dy
    if np.any(norm_sq < 1.0e-10):
        raise RuntimeError("degenerate local derivative")
    psi = np.unwrap(np.arctan2(dy, dx))
    kappa = (dx * ddy - dy * ddx) / np.power(norm_sq, 1.5)
    return psi, kappa


def mpc_discrete_geometry(
        x: np.ndarray, y: np.ndarray,
        curvature_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ReferencePath's heading, curvature and circular smoothing."""
    dx_ahead = np.roll(x, -1) - x
    dy_ahead = np.roll(y, -1) - y
    segment = np.hypot(dx_ahead, dy_ahead)
    if np.any(segment < 1.0e-10):
        raise RuntimeError("degenerate MPC waypoint segment")
    heading = np.arctan2(dy_ahead, dx_ahead)
    heading_behind = np.arctan2(y - np.roll(y, 1), x - np.roll(x, 1))
    heading_change = np.angle(np.exp(1j * (heading - heading_behind)))
    kappa = heading_change / segment
    kappa = savgol_filter(
        kappa, curvature_window, 3, mode="wrap")
    return heading, kappa


def requested_rate(
        x: np.ndarray, y: np.ndarray, kappa: np.ndarray,
        args: argparse.Namespace) -> np.ndarray:
    """Evaluate the steering objective at the same waypoint stride as MPC."""
    gain = 1.0 / (1.0 + args.understeer_coeff * args.speed**2)
    delta = np.arctan(args.wheelbase * kappa / gain)
    stride = args.mpc_stride
    horizon_distance = np.hypot(
        np.roll(x, -stride) - x,
        np.roll(y, -stride) - y)
    return (
        args.speed
        * np.abs(np.roll(delta, -stride) - delta)
        / horizon_distance
    )


def resample_base(source: pd.DataFrame, count: int) -> tuple[np.ndarray, ...]:
    order = np.r_[0, np.arange(len(source) - 1, 0, -1)]
    source = source.iloc[order].reset_index(drop=True)
    x = source.x_m.to_numpy(); y = source.y_m.to_numpy()
    segment = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    s = np.r_[0.0, np.cumsum(segment[:-1])]; length = float(segment.sum())
    target_s = np.linspace(0.0, length, count, endpoint=False)
    sx = CubicSpline(np.r_[s, length], np.r_[x, x[0]], bc_type="periodic")
    sy = CubicSpline(np.r_[s, length], np.r_[y, y[0]], bc_type="periodic")
    return target_s, sx(target_s), sy(target_s), length


def wall_intersection(point: np.ndarray, normal: np.ndarray, wall: np.ndarray) -> tuple[np.ndarray, float, int]:
    start = wall; edge = np.roll(wall, -1, axis=0) - start; rel = start - point
    denominator = normal[0] * edge[:, 1] - normal[1] * edge[:, 0]
    usable = np.abs(denominator) > 1.0e-10
    distance = np.full(len(wall), np.nan); fraction = np.full(len(wall), np.nan)
    distance[usable] = (rel[usable, 0] * edge[usable, 1] - rel[usable, 1] * edge[usable, 0]) / denominator[usable]
    fraction[usable] = (rel[usable, 0] * normal[1] - rel[usable, 1] * normal[0]) / denominator[usable]
    valid = usable & (fraction >= 0.0) & (fraction <= 1.0) & (distance > 0.0) & (distance <= 20.0)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise RuntimeError("no forward normal/boundary intersection")
    index = int(indices[np.argmin(distance[indices])]); value = float(distance[index])
    return point + value * normal, value, index


def build_bounds(boundary: pd.DataFrame, x: np.ndarray, y: np.ndarray, psi: np.ndarray) -> pd.DataFrame:
    left_wall = boundary[["left_x", "left_y"]].to_numpy(float) + [X_OFFSET, Y_OFFSET]
    right_wall = boundary[["right_x", "right_y"]].to_numpy(float) + [X_OFFSET, Y_OFFSET]
    rows = []
    for index, (px, py, heading) in enumerate(zip(x, y, psi)):
        point = np.array([px, py]); left_normal = np.array([-np.sin(heading), np.cos(heading)])
        left, ub, li = wall_intersection(point, left_normal, left_wall)
        right, right_distance, ri = wall_intersection(point, -left_normal, right_wall)
        rows.append((index, ub, -right_distance, li, ri, *left, *right))
    return pd.DataFrame(rows, columns=("idx", "ub", "lb", "left_boundary_idx", "right_boundary_idx", "left_x", "left_y", "right_x", "right_y"))


def main() -> None:
    args = arguments(); source = pd.read_csv(args.input); boundary = pd.read_csv(args.boundary)
    s, raw_x, raw_y, length = resample_base(source, args.points)

    # Small whole-loop filtering removes measurement-scale noise only.
    base_x = savgol_filter(raw_x, args.base_window, 3, mode="wrap")
    base_y = savgol_filter(raw_y, args.base_window, 3, mode="wrap")
    base_psi, base_kappa = mpc_discrete_geometry(
        base_x, base_y, args.mpc_curvature_window)
    base_rate = requested_rate(base_x, base_y, base_kappa, args)

    x = base_x.copy(); y = base_y.copy()
    local_mask = np.zeros(args.points, dtype=bool)
    blend = np.zeros(args.points, dtype=float)
    for _ in range(args.local_passes):
        _, pass_kappa = mpc_discrete_geometry(
            x, y, args.mpc_curvature_window)
        pass_rate = requested_rate(x, y, pass_kappa, args)
        pass_mask = (pass_rate > args.rate_trigger) | (np.abs(pass_kappa) > args.kappa_trigger)
        for _ in range(args.padding_points):
            pass_mask |= np.roll(pass_mask, 1) | np.roll(pass_mask, -1)
        local_mask |= pass_mask
        pass_blend = np.maximum(
            pass_mask.astype(float),
            gaussian_filter1d(pass_mask.astype(float), args.transition_sigma, mode="wrap"),
        )
        blend = np.maximum(blend, pass_blend)
        local_x = savgol_filter(x, args.local_window, 3, mode="wrap")
        local_y = savgol_filter(y, args.local_window, 3, mode="wrap")
        x = x + pass_blend * (local_x - x)
        y = y + pass_blend * (local_y - y)
        _, constrained_kappa = mpc_discrete_geometry(
            x, y, args.mpc_curvature_window)
        constrained_rate = requested_rate(x, y, constrained_kappa, args)
        if constrained_rate.max() <= args.rate_limit:
            break
    heading, kappa = mpc_discrete_geometry(
        x, y, args.mpc_curvature_window)
    rate = requested_rate(x, y, kappa, args)
    if rate.max() > args.rate_limit + 1.0e-9:
        raise RuntimeError(
            "MPC discrete steering-rate constraint was not achieved: "
            f"{rate.max():.6f} > {args.rate_limit:.6f} rad/s")

    actual_segment = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    s = np.r_[0.0, np.cumsum(actual_segment[:-1])]

    # ReferencePath reconstructs the driving heading from x/y.  The psi_rad
    # column supplied to it is retained as Waypoint.normal_angle and RViz
    # converts bounds with (-cos(psi_rad), -sin(psi_rad)).  Preserve that
    # established CSV convention: heading - pi/2 makes that vector the left
    # normal (-sin(heading), cos(heading)).
    boundary_normal_angle = np.unwrap(heading - np.pi / 2.0)

    trajectory = pd.DataFrame({"s_m": s, "x_m": x, "y_m": y, "psi_rad": boundary_normal_angle, "kappa_radpm": kappa, "vx_mps": args.vx_mps, "ax_mps3": 0.0})
    bounds = build_bounds(boundary, x, y, heading)
    here = Path(__file__).resolve().parent
    trajectory.to_csv(here / args.trajectory_name, index=False, float_format="%.10f")
    bounds.to_csv(here / args.bounds_name, index=False, float_format="%.10f")
    report = pd.DataFrame({"idx": np.arange(args.points), "s_m": s, "local_smoothing": local_mask, "blend": blend, "displacement_from_arclength_base_m": np.hypot(x-raw_x, y-raw_y), "kappa_radpm": kappa, "requested_steer_rate_radps": rate, "ub": bounds.ub, "lb": bounds.lb})
    report.to_csv(here / args.report_name, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=170)
    axes[0].plot(bounds.left_x, bounds.left_y, "r", lw=1, label="Left boundary"); axes[0].plot(bounds.right_x, bounds.right_y, "b", lw=1, label="Right boundary")
    axes[0].plot(raw_x, raw_y, color="0.7", lw=1, label="Arc-length base"); axes[0].plot(x, y, "k", lw=1.3, label="Local smooth")
    axes[0].scatter(x[local_mask], y[local_mask], s=5, c="#ff9800", label="Locally processed"); axes[0].axis("equal"); axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)
    axes[1].plot(s, base_rate, color="0.7", label="Before local"); axes[1].plot(s, rate, color="#1565c0", label="After local"); axes[1].fill_between(s, 0, 1, where=local_mask, transform=axes[1].get_xaxis_transform(), color="#ff9800", alpha=.15, label="Processed sections")
    axes[1].axhline(args.rate_limit, color="r", ls="--", label=f"{args.rate_limit:.2f} rad/s"); axes[1].set_xlabel("s [m]"); axes[1].set_ylabel("Requested steer rate [rad/s]"); axes[1].grid(alpha=.25); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(here / args.plot_name); plt.close(fig)
    print(f"Generated {args.points} points from closed-loop arc length: {length:.6f} m")
    print(f"Local smoothing points: {local_mask.sum()}/{args.points}")
    print(f"Max displacement: {report.displacement_from_arclength_base_m.max():.6f} m")
    print(f"MPC discrete kappa: {kappa.min():.6f} .. {kappa.max():.6f} rad/m")
    print(f"MPC stride-{args.mpc_stride} requested steer rate: {base_rate.max():.6f} -> {rate.max():.6f} rad/s")
    print(f"Bounds: left min={bounds.ub.min():.6f} m, right min={(-bounds.lb).min():.6f} m")


if __name__ == "__main__":
    main()
