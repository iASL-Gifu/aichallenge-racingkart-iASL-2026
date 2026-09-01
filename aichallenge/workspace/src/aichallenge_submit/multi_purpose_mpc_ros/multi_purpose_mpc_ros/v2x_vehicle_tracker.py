"""Per-vehicle finite-difference velocity tracker for V2X positions.

This module is intentionally pure Python with no rclpy dependency: it
operates on duck-typed messages whose attributes match
``v2x_msgs/V2XVehiclePositionArray``. That keeps it cheap to unit-test
and reusable from non-ROS contexts (e.g. offline replay of rosbag CSVs).
"""

import math
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Tuple


def build_closed_path_arc_lengths(points):
    """Build cumulative segment-start distances for a closed reference path."""
    xy = [(float(x), float(y)) for x, y in points]
    if len(xy) < 2:
        return xy, [], 0.0

    cumulative = [0.0]
    for index in range(len(xy)):
        x0, y0 = xy[index]
        x1, y1 = xy[(index + 1) % len(xy)]
        cumulative.append(
            cumulative[-1] + math.hypot(x1 - x0, y1 - y0))
    return xy, cumulative, cumulative[-1]


def project_to_closed_path_arc(x, y, points, cumulative, total_length):
    """Project a world position onto a closed polyline and return arc length."""
    if len(points) < 2 or total_length <= 0.0:
        return None

    px = float(x)
    py = float(y)
    best_distance_sq = float("inf")
    best_s = None
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        vx = x1 - x0
        vy = y1 - y0
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-12:
            continue
        ratio = ((px - x0) * vx + (py - y0) * vy) / length_sq
        ratio = min(max(ratio, 0.0), 1.0)
        projected_x = x0 + ratio * vx
        projected_y = y0 + ratio * vy
        distance_sq = (
            (px - projected_x) ** 2 + (py - projected_y) ** 2)
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            segment_length = math.sqrt(length_sq)
            best_s = cumulative[index] + ratio * segment_length

    return None if best_s is None else best_s % total_length


def project_to_closed_path_frenet(x, y, points, cumulative, total_length):
    """Project onto a closed path and return ``(s, signed lateral offset)``.

    Positive lateral offset is to the left of the path direction.  Computing
    each vehicle at its own closest Center segment avoids mixing longitudinal
    separation into the lateral distance on curves.
    """
    if len(points) < 2 or total_length <= 0.0:
        return None

    px = float(x)
    py = float(y)
    best_distance_sq = float("inf")
    best_frenet = None
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        vx = x1 - x0
        vy = y1 - y0
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-12:
            continue
        ratio = ((px - x0) * vx + (py - y0) * vy) / length_sq
        ratio = min(max(ratio, 0.0), 1.0)
        projected_x = x0 + ratio * vx
        projected_y = y0 + ratio * vy
        offset_x = px - projected_x
        offset_y = py - projected_y
        distance_sq = offset_x * offset_x + offset_y * offset_y
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            segment_length = math.sqrt(length_sq)
            lateral_offset = (
                offset_x * (-vy / segment_length)
                + offset_y * (vx / segment_length)
            )
            best_frenet = (
                (cumulative[index] + ratio * segment_length) % total_length,
                lateral_offset,
            )

    return best_frenet


def signed_closed_path_arc_distance(ego_s, other_s, total_length):
    """Return other-minus-ego arc distance wrapped to half a lap."""
    if ego_s is None or other_s is None or total_length <= 0.0:
        return None
    return (
        (float(other_s) - float(ego_s) + total_length / 2.0)
        % total_length
        - total_length / 2.0
    )


def resolve_applied_corridor(
    *, requested_lane, l0_prohibited: bool,
    full_width_recovery: bool, transition_active: bool,
):
    """Resolve the one corridor that owns the next MPC solve.

    Priority is geographic safety, solver/reverse recovery, ordinary transition,
    then the requested lane.  Keeping this decision in one function prevents
    several independent flags from overwriting the applied constraint.
    """
    if l0_prohibited:
        lane = 2 if requested_lane == 2 else 1
        return lane, "l0_prohibited"
    if full_width_recovery:
        return None, "full_width_recovery"
    if transition_active:
        return None, "lane_transition"
    return requested_lane, "requested_lane"


def update_motion_latch(
    has_moved_once: bool,
    actual_speed: float,
    movement_threshold: float = 1.0,
) -> bool:
    """Latch vehicle motion until an explicit simulator reset clears it."""
    return bool(has_moved_once or abs(actual_speed) > movement_threshold)


def should_reset_motion_latch(awsim_state) -> bool:
    """Allow motion-latch reset only in AWSIM's pre-start states."""
    return awsim_state in ("Grounded", "Ready")


def should_suppress_overtake_before_grounded_snapshot(
    awsim_state, snapshot_completed: bool
) -> bool:
    """Hold lateral manoeuvre state until the initial grid is captured.

    ``None`` is included because the MPC node may start before the first
    ``/awsim/state`` sample.  ``Start`` deliberately fails open: if this node
    missed Grounded/Ready entirely, normal driving must not remain disabled
    for the rest of the run.
    """
    return bool(not snapshot_completed and awsim_state != "Start")


def startup_follow_restart_gap(
    *, startup_waiting: bool, normal_min_gap: float, startup_min_gap: float
) -> float:
    """Use the shorter, separately configured restart gap only at launch."""
    return float(startup_min_gap if startup_waiting else normal_min_gap)


def startup_same_lane_lead_key(
    *, vehicle_id, vehicle_lane_idx, ego_lane_idx,
    longitudinal: float, distance: float,
):
    """Return a sortable key only for valid same-lane forward grid cars."""
    if (
        ego_lane_idx is None
        or vehicle_lane_idx != ego_lane_idx
        or not math.isfinite(float(longitudinal))
        or float(longitudinal) <= 0.0
        or not math.isfinite(float(distance))
    ):
        return None
    return (float(longitudinal), float(distance), str(vehicle_id))


def reverse_path_has_vehicle_conflict(
    *,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    reverse_distance: float,
    corridor_half_width: float,
    reverse_path_lanes,
    vehicle_x: float,
    vehicle_y: float,
    vehicle_lane,
) -> bool:
    """Return whether a vehicle occupies the predicted reverse corridor.

    Vehicles assigned to an adjacent lane do not block recovery merely by
    being inside a broad longitudinal rear window. Unknown lane positions are
    handled conservatively using the swept-path geometry.
    """
    path_lanes = {
        int(lane) for lane in (reverse_path_lanes or ()) if lane is not None
    }
    if (
        vehicle_lane is not None
        and path_lanes
        and int(vehicle_lane) not in path_lanes
    ):
        return False

    dx = float(vehicle_x) - float(ego_x)
    dy = float(vehicle_y) - float(ego_y)
    heading = float(ego_heading)
    longitudinal = dx * math.cos(heading) + dy * math.sin(heading)
    lateral = -dx * math.sin(heading) + dy * math.cos(heading)
    return bool(
        -max(float(reverse_distance), 0.0) <= longitudinal < 0.0
        and abs(lateral) <= max(float(corridor_half_width), 0.0)
    )


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
        self._generation = 0
        self._lock = threading.RLock()

    def update(self, msg) -> None:
        with self._lock:
            self._update_locked(msg)
            self._generation += 1

    def _update_locked(self, msg) -> None:
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

    def snapshot(self):
        """Return a detached, cycle-consistent tracker snapshot.

        ROS callbacks update the live tracker while the controller runs in a
        separate thread.  Copy every related collection under one lock so a
        control cycle cannot mix positions, velocities, validity and IDs from
        different V2X messages.
        """
        with self._lock:
            result = V2XVehicleTracker(
                self._v_max_safety, self._jump_thresh, self._warn)
            result._samples = {
                vehicle_id: deque(samples, maxlen=2)
                for vehicle_id, samples in self._samples.items()
            }
            result._velocities = dict(self._velocities)
            result._velocity_valid = dict(self._velocity_valid)
            result._active = list(self._active)
            result._generation = self._generation
            return result

    @property
    def generation(self) -> int:
        return self._generation

    def clear_active(self) -> None:
        """Publish an empty active set without exposing internal collections."""
        with self._lock:
            self._active = []
            self._generation += 1

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


def should_reset_overtake_latch_for_target_change(
    *,
    active_target_id,
    candidate_target_id,
    candidate_is_relevant,
    active_target_distance=None,
    candidate_target_distance=None,
    switch_margin_m: float = 0.0,
    target_locked: bool = False,
) -> bool:
    """Return whether a newly selected relevant lead replaces an old target.

    Passing-side latches are deliberately sticky during one manoeuvre.  A
    different nearby/stopped vehicle is a new manoeuvre, however, and must not
    inherit the previous target's L0/L2 decision.
    """
    # Once an overtake owns a target, a different nearby vehicle must remain
    # a safety input only. ParallelSafety and EmergencyBrake still inspect all
    # vehicles; they do not need to replace the manoeuvre target.
    if target_locked:
        return False
    if not bool(
        candidate_is_relevant
        and candidate_target_id is not None
        and active_target_id is not None
        and candidate_target_id != active_target_id
    ):
        return False
    if (
        active_target_distance is None
        or candidate_target_distance is None
        or not math.isfinite(float(active_target_distance))
        or not math.isfinite(float(candidate_target_distance))
    ):
        return True
    return float(candidate_target_distance) <= (
        float(active_target_distance) - max(float(switch_margin_m), 0.0)
    )


def should_hold_follow_escape_exclusive(
    *, target_longitudinal, target_distance, safe_distance
) -> bool:
    """Keep deadlock escape ownership while its target is ahead and close."""
    return bool(
        is_follow_target_ahead(target_longitudinal)
        and target_distance is not None
        and math.isfinite(float(target_distance))
        and float(target_distance) < float(safe_distance)
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


def should_release_active_overtake_distance_gate(
    *, outer_lane_active: bool, target_longitudinal, distance,
    max_distance: float
) -> bool:
    """Release an ordinary outer-lane constraint once its lead exits the gate.

    The behind-target state machine remains responsible after the target is
    passed.  This gate specifically prevents a still-ahead but already distant
    target from retaining L0/L2 until that constraint eventually becomes
    infeasible.
    """
    return bool(
        outer_lane_active
        and is_follow_target_ahead(target_longitudinal)
        and distance is not None
        and math.isfinite(float(distance))
        and float(distance) >= float(max_distance)
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


def should_defer_l0_prohibited_l1_hard(
    *,
    l0_prohibited,
    physical_lane_idx,
    l1_conflicts,
    heading_error=0.0,
    max_heading_error=math.inf,
    lateral_speed=0.0,
    max_lateral_speed=math.inf,
    yaw_rate=0.0,
    max_yaw_rate=math.inf,
    l1_lateral_error=0.0,
    max_l1_lateral_error=math.inf,
    prediction_available=True,
    prediction_fit=1.0,
    minimum_prediction_fit=0.0,
    l1_hard_already_owned=False,
) -> bool:
    """Defer a new L1 hard corridor until existing entry gates are ready."""
    return bool(l0_prohibited_l1_hard_defer_reasons(
        l0_prohibited=l0_prohibited,
        physical_lane_idx=physical_lane_idx,
        l1_conflicts=l1_conflicts,
        heading_error=heading_error,
        max_heading_error=max_heading_error,
        lateral_speed=lateral_speed,
        max_lateral_speed=max_lateral_speed,
        yaw_rate=yaw_rate,
        max_yaw_rate=max_yaw_rate,
        l1_lateral_error=l1_lateral_error,
        max_l1_lateral_error=max_l1_lateral_error,
        prediction_available=prediction_available,
        prediction_fit=prediction_fit,
        minimum_prediction_fit=minimum_prediction_fit,
        l1_hard_already_owned=l1_hard_already_owned,
    ))


def l0_prohibited_l1_hard_defer_reasons(
    *,
    l0_prohibited,
    physical_lane_idx,
    l1_conflicts,
    heading_error,
    max_heading_error,
    lateral_speed,
    max_lateral_speed,
    yaw_rate,
    max_yaw_rate,
    l1_lateral_error,
    max_l1_lateral_error,
    prediction_available,
    prediction_fit,
    minimum_prediction_fit,
    l1_hard_already_owned=False,
):
    """List existing readiness gates that reject immediate hard L1 entry."""
    if (
        not l0_prohibited
        or physical_lane_idx not in (1, 2)
        or l1_hard_already_owned
    ):
        return ()
    reasons = []
    if not lane_conflicts_are_clear(l1_conflicts):
        reasons.append("predicted_v2x_conflict")
    if float(heading_error) > float(max_heading_error):
        reasons.append("heading_unstable")
    if float(lateral_speed) > float(max_lateral_speed):
        reasons.append("lateral_speed_unstable")
    if float(yaw_rate) > float(max_yaw_rate):
        reasons.append("yaw_rate_unstable")
    if float(l1_lateral_error) > float(max_l1_lateral_error):
        reasons.append("l1_lateral_error")
    if not prediction_available:
        reasons.append("prediction_unavailable")
    elif float(prediction_fit) < float(minimum_prediction_fit):
        reasons.append("prediction_fit_low")
    return tuple(reasons)


def should_defer_l0_prohibited_physical_l2_l1_hard(
    *, l0_prohibited, physical_lane_idx, requested_lane_idx,
    failed_readiness_conditions,
) -> bool:
    """Defer any new physical-L2 to L1-hard entry that is not ready."""
    return bool(
        l0_prohibited
        and physical_lane_idx == 2
        and requested_lane_idx != 2
        and failed_readiness_conditions
    )


def select_safe_outer_lane(
    preferred_lane_idx,
    physical_passage,
    conflicts_by_lane,
):
    """Choose a physically passable, traffic-clear L0/L2 candidate."""
    preferred = preferred_lane_idx if preferred_lane_idx in (0, 2) else 0
    for lane_idx in (preferred, 2 if preferred == 0 else 0):
        if (
            physical_passage.get(lane_idx, False)
            and lane_conflicts_are_clear(conflicts_by_lane.get(lane_idx, {}))
        ):
            return lane_idx
    return None


def exclude_short_failed_normal_commit_candidates(
    binary_safe_outer_lanes, short_failed_outer_lanes
):
    """Remove attempt-level short failures from ordinary new commits only."""
    short_failed = set(short_failed_outer_lanes)
    return [
        lane_idx for lane_idx in binary_safe_outer_lanes
        if lane_idx in (0, 2) and lane_idx not in short_failed
    ]


def minimum_predicted_vehicle_margin(
    prediction_x,
    prediction_y,
    prediction_times,
    moving_vehicles,
    *,
    minimum_clearance,
):
    """Return the worst predicted clearance margin across all V2X traffic."""
    if (
        not prediction_x
        or len(prediction_x) != len(prediction_y)
        or len(prediction_x) != len(prediction_times)
    ):
        return -math.inf, None
    minimum_margin = math.inf
    limiting_vehicle_id = None
    for vehicle_id, vehicle_x, vehicle_y, vehicle_vx, vehicle_vy in (
        moving_vehicles
    ):
        for ego_x, ego_y, prediction_time in zip(
            prediction_x, prediction_y, prediction_times
        ):
            margin = math.hypot(
                float(ego_x)
                - (float(vehicle_x) + float(vehicle_vx) * prediction_time),
                float(ego_y)
                - (float(vehicle_y) + float(vehicle_vy) * prediction_time),
            ) - float(minimum_clearance)
            if margin < minimum_margin:
                minimum_margin = margin
                limiting_vehicle_id = vehicle_id
    return minimum_margin, limiting_vehicle_id


def select_ranked_safe_outer_lane(
    preferred_lane_idx,
    physical_passage,
    conflicts_by_lane,
    quality_by_lane,
    *,
    tie_margin,
):
    """Rank two binary-safe outer lanes without rescuing an unsafe lane."""
    safe_lanes = [
        lane_idx for lane_idx in (0, 2)
        if physical_passage.get(lane_idx, False)
        and lane_conflicts_are_clear(conflicts_by_lane.get(lane_idx, {}))
    ]
    if not safe_lanes:
        return None, "no_safe_lane"
    if len(safe_lanes) == 1:
        return safe_lanes[0], "only_safe_lane"

    preferred = preferred_lane_idx if preferred_lane_idx in (0, 2) else 0
    tie_margin = max(float(tie_margin), 0.0)
    l0_v2x, l0_wall = quality_by_lane[0]
    l2_v2x, l2_wall = quality_by_lane[2]
    if abs(float(l0_v2x) - float(l2_v2x)) > tie_margin:
        return (
            (0, "v2x_margin")
            if l0_v2x > l2_v2x else (2, "v2x_margin")
        )
    if abs(float(l0_wall) - float(l2_wall)) > tie_margin:
        return (
            (0, "wall_margin")
            if l0_wall > l2_wall else (2, "wall_margin")
        )
    return preferred, "preferred_tiebreak"


def follow_stop_deadlock_conditions_met(
    *,
    follow_active,
    ego_speed,
    lead_speed,
    gnss_moved_distance,
    forward_command,
    ego_speed_threshold,
    lead_speed_threshold,
    gnss_distance_threshold,
    forward_command_threshold,
):
    """Return whether a stopped follow pair needs an escape evaluation."""
    return (
        follow_active
        and abs(ego_speed) < ego_speed_threshold
        and lead_speed < lead_speed_threshold
        and gnss_moved_distance < gnss_distance_threshold
        and forward_command < forward_command_threshold
    )


def update_follow_escape_probe_success_cycles(
    current_cycles,
    *,
    lane_applied,
    feasible_solution,
    executable_forward_prediction,
    prediction_clear,
    emergency_brake_active,
):
    """Count only consecutive, executable and collision-free lane probes."""
    if (
        lane_applied
        and feasible_solution
        and executable_forward_prediction
        and prediction_clear
        and not emergency_brake_active
    ):
        return current_cycles + 1
    return 0


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
    minimum_prediction_time=0.0,
    maximum_prediction_time=None,
):
    """Return true when every selected prediction point clears a vehicle."""
    if (
        not prediction_x
        or len(prediction_x) != len(prediction_y)
        or len(prediction_x) != len(prediction_times)
    ):
        return False
    selected = [
        (ego_x, ego_y, prediction_time)
        for ego_x, ego_y, prediction_time in zip(
            prediction_x, prediction_y, prediction_times)
        if prediction_time >= minimum_prediction_time
        and (
            maximum_prediction_time is None
            or prediction_time <= maximum_prediction_time
        )
    ]
    if not selected:
        return False
    return all(
        math.hypot(
            ego_x - (vehicle_x + vehicle_vx * prediction_time),
            ego_y - (vehicle_y + vehicle_vy * prediction_time),
        ) >= minimum_clearance
        for ego_x, ego_y, prediction_time in selected
    )


def minimum_predicted_envelope_conflict(
    prediction_x, prediction_y, prediction_times, moving_vehicles, *,
    project_frenet, arc_total_length, ego_width, other_half_width,
    ego_half_length, other_half_length, timing=None,
    candidate_frenet_cache_enabled=True,
    external_candidate_frenet_cache=None,
):
    """Return the worst predicted rectangular-envelope overlap, if any."""
    total_start = time.perf_counter() if timing is not None else None
    if timing is not None:
        timing["candidate_sample_count"] = len(prediction_x)
        timing["moving_vehicle_count"] = len(moving_vehicles)
        timing["pair_check_count"] = 0
        timing["center_frenet_call_count"] = 0
        timing["candidate_frenet_cache_hits"] = 0
        timing["candidate_frenet_cache_misses"] = 0
        timing["external_candidate_frenet_cache_hits"] = 0
        timing["external_candidate_frenet_cache_misses"] = 0
        timing["frenet_projection_ms"] = 0.0
        timing["clearance_evaluation_ms"] = 0.0
        timing["conflict_found"] = False
        timing["early_returned"] = False
    if (
        not prediction_x
        or len(prediction_x) != len(prediction_y)
        or len(prediction_x) != len(prediction_times)
    ):
        if timing is not None:
            timing["early_returned"] = True
            timing["total_ms"] = (
                time.perf_counter() - total_start) * 1000.0
        return None
    worst_conflict = None
    candidate_frenet_by_index = {}
    for vehicle_id, vehicle_x, vehicle_y, vehicle_vx, vehicle_vy in moving_vehicles:
        for index, (ego_x, ego_y, prediction_time) in enumerate(zip(
            prediction_x, prediction_y, prediction_times
        )):
            if timing is not None:
                timing["pair_check_count"] += 1
            frenet_start = (
                time.perf_counter() if timing is not None else None)
            candidate_cache_hit = bool(
                candidate_frenet_cache_enabled
                and index in candidate_frenet_by_index
            )
            external_candidate_cache_hit = False
            candidate_projection_called = False
            if candidate_cache_hit:
                ego_frenet = candidate_frenet_by_index[index]
            else:
                external_cache_key = (float(ego_x), float(ego_y))
                external_candidate_cache_hit = bool(
                    external_candidate_frenet_cache is not None
                    and external_cache_key in external_candidate_frenet_cache
                )
                if external_candidate_cache_hit:
                    ego_frenet = external_candidate_frenet_cache[
                        external_cache_key]
                else:
                    ego_frenet = project_frenet(ego_x, ego_y)
                    candidate_projection_called = True
                    if external_candidate_frenet_cache is not None:
                        external_candidate_frenet_cache[
                            external_cache_key] = ego_frenet
                if candidate_frenet_cache_enabled:
                    candidate_frenet_by_index[index] = ego_frenet
            vehicle_frenet = project_frenet(
                vehicle_x + vehicle_vx * prediction_time,
                vehicle_y + vehicle_vy * prediction_time,
            )
            if timing is not None:
                if candidate_cache_hit:
                    timing["candidate_frenet_cache_hits"] += 1
                elif candidate_projection_called:
                    timing["candidate_frenet_cache_misses"] += 1
                if external_candidate_cache_hit:
                    timing["external_candidate_frenet_cache_hits"] += 1
                elif (
                    not candidate_cache_hit
                    and external_candidate_frenet_cache is not None
                ):
                    timing["external_candidate_frenet_cache_misses"] += 1
                timing["center_frenet_call_count"] += (
                    1 + int(candidate_projection_called))
                timing["frenet_projection_ms"] += (
                    time.perf_counter() - frenet_start) * 1000.0
            if ego_frenet is None or vehicle_frenet is None:
                continue
            clearance_start = (
                time.perf_counter() if timing is not None else None)
            longitudinal = signed_closed_path_arc_distance(
                ego_frenet[0], vehicle_frenet[0], arc_total_length)
            if longitudinal is None:
                if timing is not None:
                    timing["clearance_evaluation_ms"] += (
                        time.perf_counter() - clearance_start) * 1000.0
                continue
            lateral_clearance = lateral_vehicle_clearance(
                vehicle_frenet[1] - ego_frenet[1],
                ego_width, other_half_width)
            longitudinal_clearance = longitudinal_vehicle_clearance(
                longitudinal, ego_half_length, other_half_length)
            if (
                lateral_clearance <= 0.0
                and longitudinal_clearance <= 0.0
                and (
                    worst_conflict is None
                    or lateral_clearance
                        < worst_conflict["predicted_lateral_clearance"]
                )
            ):
                worst_conflict = {
                    "vehicle_id": vehicle_id,
                    "predicted_lateral_clearance": lateral_clearance,
                    "predicted_longitudinal_clearance": longitudinal_clearance,
                    "prediction_time": float(prediction_time),
                    "prediction_step": index,
                }
            if timing is not None:
                timing["clearance_evaluation_ms"] += (
                    time.perf_counter() - clearance_start) * 1000.0
    if timing is not None:
        timing["conflict_found"] = worst_conflict is not None
        timing["total_ms"] = (
            time.perf_counter() - total_start) * 1000.0
    return worst_conflict


def exclude_envelope_conflicting_lanes(candidate_lanes, conflicts_by_lane):
    """Keep candidate lanes whose continuous envelope prediction is clear."""
    return [
        lane_idx for lane_idx in candidate_lanes
        if conflicts_by_lane.get(lane_idx) is None
    ]


def is_parallel_vehicle(
    *,
    ego_lane_idx,
    other_lane_idx,
    lateral_clearance,
    longitudinal_clearance,
    longitudinal_distance,
    maximum_lateral_clearance,
    maximum_longitudinal_clearance,
    minimum_longitudinal_distance,
    maximum_longitudinal_distance,
):
    """Classify side-by-side traffic using envelope-to-envelope clearance."""
    return (
        ego_lane_idx is not None
        and other_lane_idx is not None
        and ego_lane_idx != other_lane_idx
        # Negative clearance means the configured envelopes overlap and must
        # remain in the most critical class rather than falling through a
        # minimum center-distance gate.
        and lateral_clearance <= maximum_lateral_clearance
        # A small lateral gap alone is insufficient: vehicles separated along
        # the Center arc must also have overlapping or nearby length envelopes.
        and longitudinal_clearance <= maximum_longitudinal_clearance
        and minimum_longitudinal_distance <= longitudinal_distance
            <= maximum_longitudinal_distance
    )


def lateral_vehicle_clearance(
    lateral_center_distance,
    ego_width,
    other_half_width,
):
    """Return signed lateral free space between two vehicle envelopes."""
    return (
        abs(float(lateral_center_distance))
        - 0.5 * max(float(ego_width), 0.0)
        - max(float(other_half_width), 0.0)
    )


def longitudinal_vehicle_clearance(
    longitudinal_center_distance,
    ego_half_length,
    other_half_length,
):
    """Return signed longitudinal free space between vehicle envelopes."""
    return (
        abs(float(longitudinal_center_distance))
        - max(float(ego_half_length), 0.0)
        - max(float(other_half_length), 0.0)
    )


def select_parallel_abort_lane(ego_lane_idx, other_lane_idx):
    """Keep an outer lane when yielding to a vehicle occupying L1."""
    if other_lane_idx == 1 and ego_lane_idx in (0, 2):
        return int(ego_lane_idx)
    return 1


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


def should_hold_generic_reverse_for_minimum_distance(
    *,
    front_vehicle_origin,
    travelled_distance,
    minimum_distance,
    localization_consistent,
    boundary_has_remaining_clearance,
):
    """Block an MPC-only early exit until a generic reverse actually moves.

    Front-vehicle recovery owns separate release conditions.  A localization
    failure or insufficient static-map clearance must never force continued
    reverse travel merely to satisfy the nominal minimum.
    """
    return bool(
        not front_vehicle_origin
        and float(travelled_distance) + 1e-6 < float(minimum_distance)
        and localization_consistent
        and boundary_has_remaining_clearance
    )
