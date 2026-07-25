"""Per-vehicle finite-difference velocity tracker for V2X positions.

This module is intentionally pure Python with no rclpy dependency: it
operates on duck-typed messages whose attributes match
``v2x_msgs/V2XVehiclePositionArray``. That keeps it cheap to unit-test
and reusable from non-ROS contexts (e.g. offline replay of rosbag CSVs).
"""

import math
from collections import deque
from typing import Deque, Dict, List, Tuple


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class V2XVehicleTracker:
    """Tracks the latest two samples per ``vehicle_id`` and exposes
    constant-velocity predictions over a caller-provided time grid."""

    def __init__(self, v_max_safety: float, position_jump_threshold: float, warn_callback=None):
        self._v_max_safety = float(v_max_safety)
        self._jump_thresh = float(position_jump_threshold)
        self._warn = warn_callback if warn_callback is not None else (lambda _msg: None)
        self._samples: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self._velocities: Dict[str, Tuple[float, float]] = {}
        self._velocity_valid: Dict[str, bool] = {}
        self._active: List[str] = []

    def update(self, msg) -> None:
        active: List[str] = []
        for v in msg.vehicles:
            vid = v.vehicle_id
            t = _stamp_to_seconds(v.header.stamp)
            x = float(v.position.x)
            y = float(v.position.y)
            buf = self._samples.setdefault(vid, deque(maxlen=2))

            # Detect a position jump against the previous sample (if any).
            jumped = False
            if buf:
                _t_prev, x_prev, y_prev = buf[-1]
                if math.hypot(x - x_prev, y - y_prev) > self._jump_thresh:
                    buf.clear()
                    jumped = True
                    self._warn(
                        f"V2X: position jump for vehicle '{vid}' "
                        f"(>{self._jump_thresh} m) — velocity reset")

            buf.append((t, x, y))

            if jumped or len(buf) < 2:
                self._velocities[vid] = (0.0, 0.0)
                self._velocity_valid[vid] = False
            else:
                t0, x0, y0 = buf[0]
                t1, x1, y1 = buf[1]
                dt = t1 - t0
                if dt > 0.0:
                    vx = (x1 - x0) / dt
                    vy = (y1 - y0) / dt
                    if math.hypot(vx, vy) > self._v_max_safety:
                        self._velocities[vid] = (0.0, 0.0)
                        self._velocity_valid[vid] = False
                        self._warn(
                            f"V2X: velocity for vehicle '{vid}' exceeds "
                            f"{self._v_max_safety} m/s — clamped to zero")
                    else:
                        self._velocities[vid] = (vx, vy)
                        self._velocity_valid[vid] = True
                else:
                    self._velocities[vid] = (0.0, 0.0)
                    self._velocity_valid[vid] = False
            active.append(vid)
        self._active = active

    def velocity(self, vehicle_id: str) -> Tuple[float, float]:
        return self._velocities.get(vehicle_id, (0.0, 0.0))

    def has_velocity_estimate(self, vehicle_id: str) -> bool:
        return self._velocity_valid.get(vehicle_id, False)

    def predict_positions(
        self, vehicle_id: str, t_samples
    ) -> List[Tuple[float, float]]:
        buf = self._samples.get(vehicle_id)
        if not buf:
            return []
        _t_last, x_last, y_last = buf[-1]
        vx, vy = self._velocities.get(vehicle_id, (0.0, 0.0))
        return [(x_last + vx * t, y_last + vy * t) for t in t_samples]

    def active_vehicle_ids(self) -> List[str]:
        return list(self._active)

    def predict_all(self, t_samples) -> Dict[str, List[Tuple[float, float]]]:
        return {vid: self.predict_positions(vid, t_samples) for vid in self._active}


def predictions_to_obstacles(predictions, vehicle_radius: float, obstacle_cls=None):
    """Flatten a ``{vehicle_id: [(x, y), ...]}`` mapping into a list of
    circular obstacles consumable by ``multi_purpose_mpc_ros.core.map``.

    ``obstacle_cls`` is injectable for testability; production callers
    leave it as ``None`` to use ``core.map.Obstacle``. The deferred
    import keeps this module's load time fast and lets the unit tests
    on hosts without ``scikit-image`` exercise the helper with a stub
    dataclass.
    """
    if obstacle_cls is None:
        from multi_purpose_mpc_ros.core.map import Obstacle as obstacle_cls
    out = []
    for _vid, points in predictions.items():
        for x, y in points:
            out.append(obstacle_cls(cx=x, cy=y, radius=vehicle_radius))
    return out


def relative_longitudinal_distance(dx: float, dy: float, heading: float) -> float:
    """Project a relative position onto the ego vehicle's forward axis."""
    return dx * math.cos(heading) + dy * math.sin(heading)


def is_follow_target_ahead(longitudinal) -> bool:
    """Return whether a target is strictly ahead and valid for ACC follow."""
    return bool(
        longitudinal is not None
        and math.isfinite(float(longitudinal))
        and float(longitudinal) > 0.0
    )


def should_start_prepass_recovery_from_safety(
    *,
    recovery_requested: bool,
    fallback_enabled: bool,
    applied_lane_idx,
    latched_vehicle_id,
    opponent_ahead: bool,
    fallback_recovery_active: bool,
    fallback_follow_active: bool,
) -> bool:
    """Return whether an outer-lane MPC failure should enter Prepass recovery.

    The applied lane is used instead of the requested lane so a failure during
    the full-width transition window is not incorrectly attributed to L0/L2.
    """
    return bool(
        recovery_requested
        and fallback_enabled
        and applied_lane_idx in (0, 2)
        and latched_vehicle_id is not None
        and opponent_ahead
        and not fallback_recovery_active
        and not fallback_follow_active
    )


def should_start_l1_recovery_from_safety(
    *,
    recovery_requested: bool,
    applied_lane_idx,
    l1_recovery_pending: bool,
) -> bool:
    """Return whether a SafetyRecovery belongs to an applied L1 constraint.

    Use the lane that was actually applied to MPC.  A requested L1 that is
    still inside the full-width transition window must not be blamed for an
    unrelated full-width failure.
    """
    return bool(
        recovery_requested
        and applied_lane_idx == 1
        and not l1_recovery_pending
    )


def is_prepass_fallback_lane_change(
    *,
    fallback_lane_idx,
    requested_lane_idx,
) -> bool:
    """Return whether the requested lane was selected by Prepass fallback."""
    return bool(
        fallback_lane_idx in (0, 1, 2)
        and requested_lane_idx == fallback_lane_idx
    )


def prepass_recovery_owns_lane_selection(
    recovery_active: bool,
    fallback_commit_pending: bool = False,
) -> bool:
    """Return whether Prepass must suppress ordinary lane selection."""
    return bool(recovery_active or fallback_commit_pending)


def should_reevaluate_follow_overtake(
    follow_active: bool,
    now_sec: float,
    last_evaluation_sec,
    retry_interval_sec: float,
) -> bool:
    """Return whether a terminal-looking follow state should retry passing."""
    if not follow_active:
        return False
    if retry_interval_sec <= 0.0 or last_evaluation_sec is None:
        return True
    return now_sec - float(last_evaluation_sec) >= retry_interval_sec


def is_follow_retry_within_distance(distance, max_distance: float) -> bool:
    """Require a finite non-negative distance strictly below the retry gate."""
    return bool(
        distance is not None
        and math.isfinite(float(distance))
        and 0.0 <= float(distance) < float(max_distance)
    )


def should_release_prepass_distance_gate(
    *, recovery_active: bool, distance, max_distance: float
) -> bool:
    """Release active Prepass immediately on a confirmed distance gate exit.

    A missing/non-finite V2X sample is not treated as a confirmed gate exit;
    the existing target-loss hold remains responsible for that case.
    """
    return bool(
        recovery_active
        and distance is not None
        and math.isfinite(float(distance))
        and float(distance) >= max(float(max_distance), 0.0)
    )


def update_continuous_condition_since(
    current_since,
    *,
    now_sec: float,
    condition: bool,
):
    """Track the start of a continuous condition, resetting on any gap."""
    if not condition:
        return None
    if current_since is None:
        return float(now_sec)
    return float(current_since)


def continuous_condition_confirmed(
    since,
    *,
    now_sec: float,
    confirm_sec: float,
) -> bool:
    """Return true once a continuously tracked condition is confirmed."""
    return bool(
        since is not None
        and float(now_sec) - float(since) >= max(float(confirm_sec), 0.0)
    )


def classify_prepass_timeout_reasons(
    *,
    mpc_stable: bool,
    heading_stable: bool,
    dynamics_stable: bool,
    physical_passage_available: bool,
    traffic_clear: bool,
    distance_within_gate: bool,
    target_behind: bool,
):
    """Return machine-readable reasons why Prepass recovery timed out."""
    reasons = []
    if target_behind:
        reasons.append("target_behind")
    if not distance_within_gate:
        reasons.append("distance_gate")
    if not mpc_stable:
        reasons.append("mpc_unstable")
    if not heading_stable:
        reasons.append("heading_unstable")
    if not dynamics_stable:
        reasons.append("vehicle_dynamics_unstable")
    if not physical_passage_available:
        reasons.append("physical_passage_blocked")
    elif not traffic_clear:
        reasons.append("traffic_blocked")
    if not reasons:
        reasons.append("stability_window_not_confirmed")
    return tuple(reasons)


def circular_forward_progress(start_wp, current_wp, n_waypoints: int) -> int:
    """Return forward waypoint progress on a circular reference path."""
    if start_wp is None or current_wp is None or int(n_waypoints) <= 0:
        return 0
    return (int(current_wp) - int(start_wp)) % int(n_waypoints)


def should_exit_l1_probe_backoff(
    *,
    elapsed_sec: float,
    cooldown_sec: float,
    waypoint_progress: int,
    minimum_waypoint_progress: int,
    full_width_success_sec: float,
    required_full_width_success_sec: float,
) -> bool:
    """Require time, spatial progress, and fresh full-width MPC recovery."""
    return bool(
        float(elapsed_sec) >= max(float(cooldown_sec), 0.0)
        and int(waypoint_progress) >= max(int(minimum_waypoint_progress), 0)
        and float(full_width_success_sec)
            >= max(float(required_full_width_success_sec), 0.0)
    )


def update_fallback_commit_success_since(
    current_success_since,
    *,
    now_sec: float,
    lane_applied: bool,
    feasible_solution: bool,
):
    """Track when continuous fallback feasibility began; reset on any gap."""
    if not lane_applied or not feasible_solution:
        return None
    if current_success_since is None:
        return float(now_sec)
    return float(current_success_since)


def ordered_outer_lane_candidates(preferred_lane_idx, excluded_lane_idx=None):
    """Return opposite-first outer lanes, optionally excluding a failed lane."""
    preferred = preferred_lane_idx if preferred_lane_idx in (0, 2) else 0
    ordered = (2 if preferred == 0 else 0, preferred)
    return tuple(lane for lane in ordered if lane != excluded_lane_idx)


def ordered_prepass_fallback_candidates(
    failed_lane_idx, attempted_outer_lanes=()
):
    """Return opposite outer, failed outer re-probe, then center lane."""
    if failed_lane_idx == 0:
        ordered = (2, 0, 1)
    elif failed_lane_idx == 2:
        ordered = (0, 2, 1)
    else:
        ordered = (0, 2, 1)
    attempted = set(attempted_outer_lanes)
    return tuple(
        lane_idx for lane_idx in ordered
        if lane_idx == 1 or lane_idx not in attempted
    )


def absolute_heading_difference(first: float, second: float) -> float:
    """Return the wrapped absolute heading difference in radians."""
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def classify_lane_conflicts(
    candidate_lane_idx,
    relative_vehicles,
    *,
    front_distance,
    side_distance,
    rear_distance,
):
    """Classify vehicles that make a candidate lane unsafe.

    ``relative_vehicles`` contains ``(vehicle_id, lane_idx, longitudinal_m)``
    tuples in the ego frame. Repeated IDs (for predicted positions) are
    de-duplicated in each conflict group.
    """
    conflicts = {"front": [], "side": [], "rear": []}
    for vehicle_id, lane_idx, longitudinal in relative_vehicles:
        if lane_idx != candidate_lane_idx:
            continue
        if abs(longitudinal) <= side_distance:
            group = "side"
        elif side_distance < longitudinal <= front_distance:
            group = "front"
        elif -rear_distance <= longitudinal < -side_distance:
            group = "rear"
        else:
            continue
        if vehicle_id not in conflicts[group]:
            conflicts[group].append(vehicle_id)
    return conflicts


def lane_conflicts_are_clear(conflicts) -> bool:
    return not any(conflicts.get(group) for group in ("front", "side", "rear"))


def should_recover_from_mpc_stall(
    *,
    safety_recovery_active,
    actual_speed,
    stall_speed_threshold,
    gnss_is_stuck,
    infeasibility_counter,
    has_fresh_valid_prediction,
):
    """Allow reverse only while MPC still has no fresh feasible solution."""
    return (
        safety_recovery_active
        and abs(actual_speed) <= stall_speed_threshold
        and gnss_is_stuck
        and (
            infeasibility_counter > 0
            or not has_fresh_valid_prediction
        )
    )


def should_count_mpc_recovery_success(
    *,
    stuck_recovery_active,
    gear_is_drive,
    infeasibility_counter,
    has_current_prediction,
    used_prediction_fallback,
):
    """Count full-width recovery only after reverse/shift has fully ended."""
    return (
        not stuck_recovery_active
        and gear_is_drive
        and infeasibility_counter == 0
        and has_current_prediction
        and not used_prediction_fallback
    )


def prediction_clears_moving_vehicle(
    prediction_x,
    prediction_y,
    prediction_times,
    *,
    vehicle_x,
    vehicle_y,
    vehicle_vx,
    vehicle_vy,
    minimum_clearance,
):
    """Return true only when every prediction point clears a moving vehicle."""
    if (
        not prediction_x
        or len(prediction_x) != len(prediction_y)
        or len(prediction_x) != len(prediction_times)
    ):
        return False
    return all(
        math.hypot(
            ego_x - (vehicle_x + vehicle_vx * prediction_time),
            ego_y - (vehicle_y + vehicle_vy * prediction_time),
        ) >= minimum_clearance
        for ego_x, ego_y, prediction_time in zip(
            prediction_x, prediction_y, prediction_times)
    )


def is_parallel_vehicle(
    *,
    ego_lane_idx,
    other_lane_idx,
    lateral_distance,
    longitudinal_distance,
    minimum_lateral_distance,
    maximum_lateral_distance,
    maximum_longitudinal_distance,
):
    """Classify true side-by-side traffic, excluding same-lane following."""
    return (
        ego_lane_idx is not None
        and other_lane_idx is not None
        and ego_lane_idx != other_lane_idx
        and minimum_lateral_distance <= lateral_distance
            <= maximum_lateral_distance
        and abs(longitudinal_distance) <= maximum_longitudinal_distance
    )


def select_latched_overtake_lane(
    overtake_active,
    candidate_vehicle_id,
    candidate_lane_idx,
    latched_vehicle_id,
    latched_lane_idx,
):
    """Latch one passing side until the high-level overtake mode finishes."""
    if not overtake_active:
        return None, None, None, False
    if latched_lane_idx in (0, 2):
        return (
            latched_lane_idx,
            latched_vehicle_id,
            latched_lane_idx,
            False,
        )
    if candidate_vehicle_id is not None and candidate_lane_idx in (0, 2):
        return (
            candidate_lane_idx,
            candidate_vehicle_id,
            candidate_lane_idx,
            True,
        )
    return candidate_lane_idx, latched_vehicle_id, latched_lane_idx, False


def should_release_latched_overtake_lane(
    *,
    overtake_active,
    latched_vehicle_id,
    latched_lane_idx,
    target_longitudinal_distance,
    infeasibility_counter,
    behind_distance,
    infeasible_cycles,
):
    """Release only an outer-lane constraint after a completed pass stalls MPC."""
    return (
        overtake_active
        and latched_vehicle_id is not None
        and latched_lane_idx in (0, 2)
        and target_longitudinal_distance is not None
        and target_longitudinal_distance <= -behind_distance
        and infeasibility_counter >= infeasible_cycles
    )


def evaluate_stopped_lead_overtake(
    *,
    tracked_stopped_lead,
    distance,
    left_is_free,
    right_is_free,
    target_lane_idx,
    infeasibility_counter,
    reverse_distance,
    infeasible_cycles,
):
    """Choose forced overtaking or a stopped-vehicle reverse request."""
    passing_side_available = left_is_free or right_is_free
    outer_lane_accepted = target_lane_idx in (0, 2)
    reverse_requested = (
        tracked_stopped_lead
        and distance <= reverse_distance
        and (
            not passing_side_available
            or not outer_lane_accepted
            or infeasibility_counter >= infeasible_cycles
        )
    )
    forced_overtake_active = (
        tracked_stopped_lead
        and passing_side_available
        and outer_lane_accepted
        and not reverse_requested
    )
    return forced_overtake_active, reverse_requested
