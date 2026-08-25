#!/usr/bin/env python3
"""Generate an MPC center trajectory and bounds using periodic splines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, make_splprep
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial import cKDTree


REQUIRED_COLUMNS = (
    "s_m",
    "x_m",
    "y_m",
    "w_tr_right_m",
    "w_tr_left_m",
    "segment_length_to_next_m",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Resample a closed track with periodic cubic splines and generate "
            "MPC trajectory/bounds CSV files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "track_arclength.csv",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=script_dir / "traj_center624_periodic_spline.csv",
    )
    parser.add_argument(
        "--bounds-output",
        type=Path,
        default=script_dir / "waypoint_bounds_center624_periodic_spline.csv",
    )
    parser.add_argument("--num-points", type=int, default=624)
    parser.add_argument("--vx-mps", type=float, default=9.694)
    parser.add_argument("--ax-mps3", type=float, default=0.0)
    parser.add_argument("--wheelbase-m", type=float, default=1.087)
    parser.add_argument("--understeer-coeff", type=float, default=0.002)
    parser.add_argument("--delta-max-deg", type=float, default=18.0)
    parser.add_argument("--steer-rate-max-radps", type=float, default=0.60)
    parser.add_argument(
        "--smoothing-rms-m",
        type=float,
        default=0.125,
        help=(
            "Target RMS coordinate smoothing scale [m]. The spline smoothing "
            "budget is source_points * smoothing_rms_m^2 (default: 0.125)."
        ),
    )
    parser.add_argument(
        "--keep-input-direction",
        action="store_true",
        help=(
            "Keep track.csv direction. By default it is reversed around the "
            "first point to match the existing Center trajectory direction."
        ),
    )
    return parser.parse_args()


def _validate(source: pd.DataFrame, num_points: int) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if num_points < 4:
        raise ValueError("num-points must be at least 4")
    values = source.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinite values")
    if len(source) < 4:
        raise ValueError("At least four source points are required")
    if not np.all(np.diff(source["s_m"].to_numpy(dtype=float)) > 0.0):
        raise ValueError("Input s_m must be strictly increasing")
    if (source[["w_tr_right_m", "w_tr_left_m"]] <= 0.0).any().any():
        raise ValueError("Track widths must be positive")
    if (source["segment_length_to_next_m"] <= 0.0).any():
        raise ValueError("Closed-loop segment lengths must be positive")


def _reverse_closed_loop(values: pd.DataFrame) -> pd.DataFrame:
    """Reverse traversal while retaining original row zero as the anchor."""
    reverse_indices = np.r_[0, np.arange(len(values) - 1, 0, -1)]
    return values.iloc[reverse_indices].reset_index(drop=True)


def _closed_loop_parameter(values: pd.DataFrame) -> tuple[np.ndarray, float]:
    x = values["x_m"].to_numpy(dtype=float)
    y = values["y_m"].to_numpy(dtype=float)
    segments = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    if np.any(segments <= 0.0):
        raise ValueError("Reordered source contains a zero-length segment")
    s = np.empty(len(values), dtype=float)
    s[0] = 0.0
    s[1:] = np.cumsum(segments[:-1])
    return s, float(segments.sum())


def _periodic_spline(s: np.ndarray, values: np.ndarray, length: float) -> CubicSpline:
    return CubicSpline(
        np.r_[s, length],
        np.r_[values, values[0]],
        bc_type="periodic",
    )


def _project_steering_constraints(
    *,
    x_anchor: float,
    y_anchor: float,
    psi_anchor: float,
    desired_kappa: np.ndarray,
    ds_m: float,
    evaluation_speed_mps: float,
    wheelbase_m: float,
    understeer_coeff: float,
    delta_max_rad: float,
    steer_rate_max_radps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project path steering into limits and integrate an exactly closed loop."""
    if evaluation_speed_mps <= 0.0:
        raise ValueError("vx-mps must be positive for steering-rate evaluation")
    if wheelbase_m <= 0.0 or delta_max_rad <= 0.0 or steer_rate_max_radps <= 0.0:
        raise ValueError("wheelbase and steering limits must be positive")

    count = len(desired_kappa)
    curvature_gain = 1.0 / (
        1.0 + max(float(understeer_coeff), 0.0) * evaluation_speed_mps**2
    )
    desired_delta = np.arctan(wheelbase_m * desired_kappa / curvature_gain)
    max_delta_step = steer_rate_max_radps * ds_m / evaluation_speed_mps

    # Cyclic first difference: row i is delta[i] - delta[i-1].
    difference = np.eye(count) - np.roll(np.eye(count), 1, axis=1)
    initial_delta = np.clip(desired_delta, -delta_max_rad, delta_max_rad)

    # Produce a rate-feasible starting point before SLSQP enforces closure.
    for _ in range(2_000):
        changed = False
        for index in range(count):
            next_index = (index + 1) % count
            delta_step = initial_delta[next_index] - initial_delta[index]
            if delta_step > max_delta_step:
                midpoint = (
                    initial_delta[index] + initial_delta[next_index] - max_delta_step
                ) / 2.0
                initial_delta[index] = midpoint
                initial_delta[next_index] = midpoint + max_delta_step
                changed = True
            elif delta_step < -max_delta_step:
                midpoint = (
                    initial_delta[index] + initial_delta[next_index] + max_delta_step
                ) / 2.0
                initial_delta[index] = midpoint
                initial_delta[next_index] = midpoint - max_delta_step
                changed = True
        if not changed:
            break

    turn_sign = float(np.sign(np.sum(desired_kappa) * ds_m))
    if turn_sign == 0.0:
        raise ValueError("Cannot determine the closed track turning direction")
    required_turn = turn_sign * 2.0 * np.pi

    def reconstruct(delta: np.ndarray) -> tuple[np.ndarray, ...]:
        kappa = curvature_gain * np.tan(delta) / wheelbase_m
        psi = psi_anchor + np.r_[
            0.0,
            np.cumsum(0.5 * (kappa[:-1] + kappa[1:]) * ds_m),
        ]
        segment_heading = psi + 0.5 * kappa * ds_m
        x = x_anchor + np.r_[0.0, np.cumsum(np.cos(segment_heading[:-1]) * ds_m)]
        y = y_anchor + np.r_[0.0, np.cumsum(np.sin(segment_heading[:-1]) * ds_m)]
        return kappa, psi, segment_heading, x, y

    def closure_equalities(delta: np.ndarray) -> np.ndarray:
        kappa, _, segment_heading, _, _ = reconstruct(delta)
        return np.array(
            [
                np.sum(kappa) * ds_m - required_turn,
                np.sum(np.cos(segment_heading)) * ds_m,
                np.sum(np.sin(segment_heading)) * ds_m,
            ]
        )

    result = minimize(
        fun=lambda delta: 0.5 * np.sum(np.square(delta - desired_delta)),
        x0=initial_delta,
        jac=lambda delta: delta - desired_delta,
        method="SLSQP",
        bounds=Bounds(-delta_max_rad, delta_max_rad),
        constraints=[
            LinearConstraint(difference, -max_delta_step, max_delta_step),
            {"type": "eq", "fun": closure_equalities},
        ],
        options={"maxiter": 200, "ftol": 1.0e-10, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Steering-constrained geometry optimization failed: {result.message}")
    closure_error = np.max(np.abs(closure_equalities(result.x)))
    if closure_error > 1.0e-6:
        raise RuntimeError(f"Constrained geometry closure error is too large: {closure_error}")

    kappa, psi, _, x, y = reconstruct(result.x)
    return x, y, psi, kappa, result.x


def generate(
    source: pd.DataFrame,
    *,
    num_points: int,
    vx_mps: float,
    ax_mps3: float,
    smoothing_rms_m: float,
    wheelbase_m: float,
    understeer_coeff: float,
    delta_max_deg: float,
    steer_rate_max_radps: float,
    reverse_direction: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | bool | int]]:
    _validate(source, num_points)
    if smoothing_rms_m < 0.0:
        raise ValueError("smoothing-rms-m must be non-negative")
    values = source.copy()
    if reverse_direction:
        values = _reverse_closed_loop(values)

    source_s, source_length = _closed_loop_parameter(values)
    source_u = source_s / source_length
    smoothing_budget = len(values) * smoothing_rms_m**2
    geometry_points = np.vstack(
        [
            values["x_m"].to_numpy(dtype=float),
            values["y_m"].to_numpy(dtype=float),
        ]
    )
    geometry_spline, _ = make_splprep(
        np.column_stack([geometry_points, geometry_points[:, 0]]),
        u=np.r_[source_u, 1.0],
        k=3,
        s=smoothing_budget,
        bc_type="periodic",
    )

    # Reparameterize the smoothed curve approximately by its own arc length.
    dense_count = max(20_000, num_points * 32)
    dense_u = np.linspace(0.0, 1.0, dense_count + 1)
    dense_xy = np.asarray(geometry_spline(dense_u))
    dense_segments = np.hypot(np.diff(dense_xy[0]), np.diff(dense_xy[1]))
    dense_s = np.r_[0.0, np.cumsum(dense_segments)]
    smoothed_length = float(dense_s[-1])
    target_s = np.linspace(0.0, smoothed_length, num_points, endpoint=False)
    target_u = np.interp(target_s, dense_s, dense_u)

    # Widths remain tied to their source arc position and are periodic too.
    spline_wr = _periodic_spline(
        source_u, values["w_tr_right_m"].to_numpy(), 1.0
    )
    spline_wl = _periodic_spline(
        source_u, values["w_tr_left_m"].to_numpy(), 1.0
    )

    xy = np.asarray(geometry_spline(target_u))
    derivative_1 = np.asarray(geometry_spline(target_u, nu=1))
    derivative_2 = np.asarray(geometry_spline(target_u, nu=2))
    x, y = xy
    dx, dy = derivative_1
    ddx, ddy = derivative_2
    speed_s = np.hypot(dx, dy)
    if np.any(speed_s <= 1.0e-9):
        raise ValueError("Periodic spline has a degenerate tangent")

    # Heading is derived from the same smoothed coordinates as curvature.
    psi = np.unwrap(np.arctan2(dy, dx))
    kappa = (dx * ddy - dy * ddx) / np.power(speed_s, 3)
    w_right = spline_wr(target_u)
    w_left = spline_wl(target_u)
    if np.any(w_right <= 0.0) or np.any(w_left <= 0.0):
        raise ValueError("Width spline produced a non-positive boundary width")

    unconstrained_x = x.copy()
    unconstrained_y = y.copy()
    unconstrained_psi = psi.copy()
    ds_m = smoothed_length / num_points
    x, y, psi, kappa, steering_angle = _project_steering_constraints(
        x_anchor=float(x[0]),
        y_anchor=float(y[0]),
        psi_anchor=float(psi[0]),
        desired_kappa=kappa,
        ds_m=ds_m,
        evaluation_speed_mps=vx_mps,
        wheelbase_m=wheelbase_m,
        understeer_coeff=understeer_coeff,
        delta_max_rad=np.deg2rad(delta_max_deg),
        steer_rate_max_radps=steer_rate_max_radps,
    )
    geometry_displacement = np.hypot(x - unconstrained_x, y - unconstrained_y)
    unconstrained_normal_x = -np.sin(unconstrained_psi)
    unconstrained_normal_y = np.cos(unconstrained_psi)
    lateral_displacement = (
        (x - unconstrained_x) * unconstrained_normal_x
        + (y - unconstrained_y) * unconstrained_normal_y
    )
    # Preserve the original physical boundary approximately while moving the
    # reference center: moving left consumes left width and adds right width.
    w_left = w_left - lateral_displacement
    w_right = w_right + lateral_displacement
    if np.any(w_left <= 0.0) or np.any(w_right <= 0.0):
        raise RuntimeError(
            "Steering-constrained centerline crosses a reconstructed track boundary"
        )
    steering_rate = (
        vx_mps
        * np.abs(steering_angle - np.roll(steering_angle, 1))
        / ds_m
    )
    source_center = values[["x_m", "y_m"]].to_numpy(dtype=float)
    nearest_distance, nearest_index = cKDTree(source_center).query(
        np.column_stack([x, y])
    )
    conservative_source_half_width = np.minimum(
        values["w_tr_left_m"].to_numpy(dtype=float)[nearest_index],
        values["w_tr_right_m"].to_numpy(dtype=float)[nearest_index],
    )
    conservative_center_clearance = conservative_source_half_width - nearest_distance
    if np.any(conservative_center_clearance <= 0.0):
        raise RuntimeError(
            "Steering-constrained centerline leaves the source track-width envelope"
        )

    trajectory = pd.DataFrame(
        {
            "s_m": target_s,
            "x_m": x,
            "y_m": y,
            "psi_rad": psi,
            "kappa_radpm": kappa,
            "vx_mps": np.full(num_points, vx_mps),
            "ax_mps3": np.full(num_points, ax_mps3),
        }
    )

    # Geometric left normal for heading psi is (-sin(psi), cos(psi)).
    normal_x = -np.sin(psi)
    normal_y = np.cos(psi)
    left_x = x + normal_x * w_left
    left_y = y + normal_y * w_left
    right_x = x - normal_x * w_right
    right_y = y - normal_y * w_right
    bounds = pd.DataFrame(
        {
            "idx": np.arange(num_points, dtype=int),
            "ub": w_left,
            "lb": -w_right,
            # These are resampled boundary indices, not source track indices.
            "left_boundary_idx": np.arange(num_points, dtype=int),
            "right_boundary_idx": np.arange(num_points, dtype=int),
            "left_x": left_x,
            "left_y": left_y,
            "right_x": right_x,
            "right_y": right_y,
        }
    )

    output_segments = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    seam_heading_jump = float(
        abs(np.arctan2(np.sin(psi[0] - psi[-1]), np.cos(psi[0] - psi[-1])))
    )
    diagnostics: dict[str, float | bool | int] = {
        "source_points": len(source),
        "output_points": num_points,
        "reversed": reverse_direction,
        "source_length_m": source_length,
        "smoothed_spline_length_m": smoothed_length,
        "smoothing_rms_target_m": smoothing_rms_m,
        "output_polyline_length_m": float(output_segments.sum()),
        "spacing_min_m": float(output_segments.min()),
        "spacing_mean_m": float(output_segments.mean()),
        "spacing_max_m": float(output_segments.max()),
        "psi_min_rad": float(psi.min()),
        "psi_max_rad": float(psi.max()),
        "seam_heading_step_rad": seam_heading_jump,
        "kappa_min_radpm": float(kappa.min()),
        "kappa_max_radpm": float(kappa.max()),
        "kappa_abs_max_radpm": float(np.abs(kappa).max()),
        "steering_angle_abs_max_deg": float(np.degrees(np.abs(steering_angle).max())),
        "steering_rate_abs_max_radps": float(steering_rate.max()),
        "geometry_displacement_rms_m": float(
            np.sqrt(np.mean(np.square(geometry_displacement)))
        ),
        "geometry_displacement_max_m": float(geometry_displacement.max()),
        "source_envelope_clearance_min_m": float(
            conservative_center_clearance.min()
        ),
        "left_width_min_m": float(w_left.min()),
        "left_width_max_m": float(w_left.max()),
        "right_width_min_m": float(w_right.min()),
        "right_width_max_m": float(w_right.max()),
    }
    return trajectory, bounds, diagnostics


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    trajectory, bounds, diagnostics = generate(
        source,
        num_points=args.num_points,
        vx_mps=args.vx_mps,
        ax_mps3=args.ax_mps3,
        smoothing_rms_m=args.smoothing_rms_m,
        wheelbase_m=args.wheelbase_m,
        understeer_coeff=args.understeer_coeff,
        delta_max_deg=args.delta_max_deg,
        steer_rate_max_radps=args.steer_rate_max_radps,
        reverse_direction=not args.keep_input_direction,
    )

    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.bounds_output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(args.trajectory_output, index=False, float_format="%.10f")
    bounds.to_csv(args.bounds_output, index=False, float_format="%.10f")

    print(f"Input:      {args.input.resolve()}")
    print(f"Trajectory: {args.trajectory_output.resolve()}")
    print(f"Bounds:     {args.bounds_output.resolve()}")
    print(
        f"Points: {diagnostics['source_points']} -> {diagnostics['output_points']}, "
        f"direction_reversed={diagnostics['reversed']}"
    )
    print(
        f"Length: source={diagnostics['source_length_m']:.6f} m, "
        f"smoothed_spline={diagnostics['smoothed_spline_length_m']:.6f} m, "
        f"output_polyline={diagnostics['output_polyline_length_m']:.6f} m"
    )
    print(
        "Coordinate smoothing RMS target: "
        f"{diagnostics['smoothing_rms_target_m']:.6f} m"
    )
    print(
        "Output spacing [m]: "
        f"min={diagnostics['spacing_min_m']:.6f}, "
        f"mean={diagnostics['spacing_mean_m']:.6f}, "
        f"max={diagnostics['spacing_max_m']:.6f}"
    )
    print(
        "psi [rad]: "
        f"min={diagnostics['psi_min_rad']:.6f}, "
        f"max={diagnostics['psi_max_rad']:.6f}, "
        f"closed-seam step={diagnostics['seam_heading_step_rad']:.6f}"
    )
    print(
        "kappa [rad/m]: "
        f"min={diagnostics['kappa_min_radpm']:.6f}, "
        f"max={diagnostics['kappa_max_radpm']:.6f}, "
        f"abs_max={diagnostics['kappa_abs_max_radpm']:.6f}"
    )
    print(
        "Steering constraints: "
        f"angle_abs_max={diagnostics['steering_angle_abs_max_deg']:.6f} deg, "
        f"rate_abs_max={diagnostics['steering_rate_abs_max_radps']:.6f} rad/s"
    )
    print(
        "Constrained geometry displacement from initial spline [m]: "
        f"RMS={diagnostics['geometry_displacement_rms_m']:.6f}, "
        f"max={diagnostics['geometry_displacement_max_m']:.6f}"
    )
    print(
        "Conservative center clearance inside source width envelope: "
        f"min={diagnostics['source_envelope_clearance_min_m']:.6f} m"
    )
    print(
        "Widths [m]: "
        f"left={diagnostics['left_width_min_m']:.6f}.."
        f"{diagnostics['left_width_max_m']:.6f}, "
        f"right={diagnostics['right_width_min_m']:.6f}.."
        f"{diagnostics['right_width_max_m']:.6f}"
    )


if __name__ == "__main__":
    main()
