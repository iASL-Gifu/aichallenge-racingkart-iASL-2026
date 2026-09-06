"""Per-vehicle finite-difference velocity tracker for V2X positions.

This module is intentionally pure Python with no rclpy dependency: it
operates on duck-typed messages whose attributes match
``v2x_msgs/V2XVehiclePositionArray``. That keeps it cheap to unit-test
and reusable from non-ROS contexts (e.g. offline replay of rosbag CSVs).
"""

from .collision_geometry import body_pose
from dataclasses import replace

import math
import threading
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

    Priority is solver/reverse recovery, geographic safety, ordinary transition,
    then the requested lane.  Keeping this decision in one function prevents
    several independent flags from overwriting the applied constraint.
    """
    if full_width_recovery:
        return None, "full_width_recovery"
    if l0_prohibited:
        lane = 2 if requested_lane == 2 else 1
        return lane, "l0_prohibited"
    if transition_active:
        return None, "lane_transition"
    return requested_lane, "requested_lane"


def prepass_recovery_timeout_expired(
    *, elapsed: float, recovery_timeout: float,
    soft_guidance_timeout: float, full_width_prediction_safe: bool,
) -> tuple[bool, bool]:
    """Return ordinary/watchdog expiry without allowing an infinite soft hold."""
    ordinary_expired = bool(
        recovery_timeout > 0.0
        and elapsed >= recovery_timeout
        and not full_width_prediction_safe
    )
    watchdog_expired = bool(
        soft_guidance_timeout > 0.0
        and elapsed >= soft_guidance_timeout
    )
    return ordinary_expired, watchdog_expired


def outer_lane_problem_slow_override_active(
    *, latched_target_id, opponent_vehicle_id, velocity_valid: bool,
) -> bool:
    """Require a real, velocity-qualified target for the local exception."""
    return bool(
        opponent_vehicle_id is not None
        and velocity_valid
        and latched_target_id is not None
        and latched_target_id == opponent_vehicle_id
    )


def follow_emergency_reacquire_blocked(
    *, vehicle_id, released_vehicle_id, released_at,
    now_sec: float, hysteresis_sec: float,
) -> bool:
    """Prevent release/re-latch oscillation for the same emergency blocker."""
    return bool(
        vehicle_id is not None
        and vehicle_id == released_vehicle_id
        and released_at is not None
        and now_sec - released_at < hysteresis_sec
    )


def overtake_shadow_solution_acceptable(
    *, accurate: bool, used_prediction_fallback: bool,
    time_budget_exceeded: bool, recovery_requested: bool,
    infeasibility_counter: int, has_prediction: bool,
    constraint_collapsed: bool, lane_relaxation: float,
    max_lane_relaxation: float, forward_width_valid: bool,
) -> bool:
    """Apply every mandatory gate for a new outer-lane commitment."""
    return bool(
        accurate
        and not used_prediction_fallback
        and not time_budget_exceeded
        and not recovery_requested
        and int(infeasibility_counter) == 0
        and has_prediction
        and not constraint_collapsed
        and float(lane_relaxation) <= float(max_lane_relaxation) + 1e-9
        and forward_width_valid
    )


def outer_prediction_bypass_target_matches(
    *, vehicle_id, latched_target_id, handoff_target_id,
    handoff_lane_idx, verified_outer_lane,
) -> bool:
    """Accept the active target or a same-lane consecutive-handoff target.

    A handoff target is intentionally accepted before it becomes the latched
    target.  The caller must still require a fresh collision-free MPC
    prediction, separated current envelopes, and a safe dynamic gap.
    """
    if vehicle_id is None:
        return False
    if vehicle_id == latched_target_id:
        return True
    return bool(
        vehicle_id == handoff_target_id
        and handoff_lane_idx in (0, 2)
        and verified_outer_lane == handoff_lane_idx
    )


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

    def __init__(
        self, v_max_safety: float, position_jump_threshold: float,
        warn_callback=None, zero_velocity_hold_sec: float = 0.20,
        zero_velocity_threshold: float = 0.10,
    ):
        self._v_max_safety = float(v_max_safety)
        self._jump_thresh = float(position_jump_threshold)
        self._warn = warn_callback if warn_callback is not None else (lambda _msg: None)
        self._zero_velocity_hold_sec = max(float(zero_velocity_hold_sec), 0.0)
        self._zero_velocity_threshold = max(float(zero_velocity_threshold), 0.0)
        self._samples: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self._velocities: Dict[str, Tuple[float, float]] = {}
        self._velocity_valid: Dict[str, bool] = {}
        self._last_moving_velocities: Dict[str, Tuple[float, float]] = {}
        self._last_moving_velocity_times: Dict[str, float] = {}
        self._body_headings = {}
        self._body_frames = {}
        self._measured_bodies = {}
        self._body_position_uncertainty = {}
        self._active: List[str] = []
        self._generation = 0
        self._lock = threading.RLock()

    def update(self, msg) -> None:
        with self._lock:
            self._update_locked(msg)
            self._generation += 1

    def _update_locked(self, msg) -> None:
        active: List[str] = []
        incoming = {v.vehicle_id for v in msg.vehicles}
        for vid in set(self._samples) - incoming:
            self._samples.pop(vid, None)
            self._velocities.pop(vid, None)
            self._velocity_valid.pop(vid, None)
            self._body_headings.pop(vid, None)
            self._body_frames.pop(vid, None)
            self._measured_bodies.pop(vid, None)
            self._last_moving_velocities.pop(vid, None)
            self._last_moving_velocity_times.pop(vid, None)
            self._body_position_uncertainty.pop(vid, None)
        for v in msg.vehicles:
            vid = v.vehicle_id
            t = _stamp_to_seconds(v.header.stamp)
            x = float(v.position.x)
            y = float(v.position.y)
            buf = self._samples.setdefault(vid, deque(maxlen=2))
            previous = buf[-1] if buf else None
            frame = getattr(v.header, 'frame_id', 'map')
            old_frame = self._body_frames.get(vid, frame)
            self._body_frames[vid] = frame
            covariance = getattr(v, 'covariance', None)
            sigma = max(abs(float(getattr(covariance, 'x', 0.))), abs(float(getattr(covariance, 'y', 0.))))
            self._body_position_uncertainty[vid] = 2.0 * sigma if math.isfinite(sigma) else math.inf
            discontinuity = (previous is not None and (
                t <= previous[0] or t-previous[0] > 1.0
                or math.hypot(x-previous[1], y-previous[2]) > self._jump_thresh
                or frame != old_frame))
            if previous == (t, x, y) and frame == old_frame:
                discontinuity = False
            if discontinuity or not all(math.isfinite(value) for value in (t,x,y)):
                self._body_headings.pop(vid, None)
                self._measured_bodies.pop(vid, None)
            if not all(math.isfinite(value) for value in (t, x, y)):
                buf.clear()
                self._velocities[vid] = (0.0, 0.0)
                self._velocity_valid[vid] = False
                self._last_moving_velocities.pop(vid, None)
                self._last_moving_velocity_times.pop(vid, None)
                active.append(vid)
                continue

            # A retransmission is not a new motion sample. In particular it
            # must not replace a valid velocity with zero or renew its hold.
            # Equal timestamps with changed coordinates are still invalid.
            if buf and (t, x, y) == buf[-1]:
                active.append(vid)
                continue

            # Detect a position jump against the previous sample (if any).
            jumped = False
            if buf:
                _t_prev, x_prev, y_prev = buf[-1]
                if math.hypot(x - x_prev, y - y_prev) > self._jump_thresh:
                    buf.clear()
                    jumped = True
                    self._last_moving_velocities.pop(vid, None)
                    self._last_moving_velocity_times.pop(vid, None)
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
                        self._last_moving_velocities.pop(vid, None)
                        self._last_moving_velocity_times.pop(vid, None)
                        self._warn(
                            f"V2X: velocity for vehicle '{vid}' exceeds "
                            f"{self._v_max_safety} m/s — clamped to zero")
                    else:
                        measured_speed = math.hypot(vx, vy)
                        last_moving_time = self._last_moving_velocity_times.get(vid)
                        hold_previous = bool(
                            measured_speed <= self._zero_velocity_threshold
                            and last_moving_time is not None
                            and t1 >= last_moving_time
                            and t1 - last_moving_time
                                <= self._zero_velocity_hold_sec
                            and vid in self._last_moving_velocities
                        )
                        if hold_previous:
                            self._velocities[vid] = self._last_moving_velocities[vid]
                        else:
                            self._velocities[vid] = (vx, vy)
                        if measured_speed > self._zero_velocity_threshold:
                            self._last_moving_velocities[vid] = (vx, vy)
                            self._last_moving_velocity_times[vid] = t1
                        self._velocity_valid[vid] = True
                else:
                    self._velocities[vid] = (0.0, 0.0)
                    self._velocity_valid[vid] = False
                    self._last_moving_velocities.pop(vid, None)
                    self._last_moving_velocity_times.pop(vid, None)
            if (previous is not None and not discontinuity and self._velocity_valid.get(vid)
                    and t > previous[0]):
                vx,vy = (x-previous[1])/(t-previous[0]), (y-previous[2])/(t-previous[0])
                if math.hypot(vx,vy) > 0.15:
                    yaw = math.atan2(vy,vx)
                    old = self._body_headings.get(vid)
                    directional = bool(old and old[2])
                    if old and math.cos(yaw-old[0]) < 0.0:
                        yaw = math.atan2(math.sin(yaw+math.pi),math.cos(yaw+math.pi))
                    self._body_headings[vid] = (yaw,'motion_axis',directional,t)
            if not self._velocity_valid.get(vid) and previous is not None:
                self._body_headings.pop(vid, None)
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
                self._v_max_safety, self._jump_thresh, self._warn,
                self._zero_velocity_hold_sec, self._zero_velocity_threshold)
            result._samples = {
                vehicle_id: deque(samples, maxlen=2)
                for vehicle_id, samples in self._samples.items()
            }
            result._velocities = dict(self._velocities)
            result._velocity_valid = dict(self._velocity_valid)
            result._last_moving_velocities = dict(self._last_moving_velocities)
            result._last_moving_velocity_times = dict(
                self._last_moving_velocity_times)
            result._body_headings = dict(self._body_headings)
            result._body_frames = dict(self._body_frames)
            result._measured_bodies = dict(self._measured_bodies)
            result._body_position_uncertainty = dict(self._body_position_uncertainty)
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
            self._samples.clear()
            self._velocities.clear()
            self._velocity_valid.clear()
            self._body_frames.clear()
            self._body_position_uncertainty.clear()
            self._body_headings.clear()
            self._measured_bodies.clear()
            self._last_moving_velocities.clear()
            self._last_moving_velocity_times.clear()
            self._generation += 1

    def set_measured_body_pose(self, vehicle_id, x, y, yaw, stamp, frame='map'):
        """Optional synchronized map/body-center PoseStamped; no wire-format change."""
        with self._lock:
            if frame != 'map' or not all(math.isfinite(v) for v in (x,y,yaw,stamp)):
                self._measured_bodies.pop(vehicle_id, None)
                return
            samples = self._samples.get(vehicle_id)
            if samples and stamp < samples[-1][0] - 0.2:
                return
            previous = self._measured_bodies.get(vehicle_id)
            if previous and stamp <= previous.stamp:
                return
            self._measured_bodies[vehicle_id] = body_pose(
                x,y,yaw,stamp,frame=frame,source='measured',direction_valid=True)
            self._body_headings[vehicle_id] = (yaw,'measured',True,stamp)
            self._generation += 1

    def collision_body(self, vehicle_id, now, *, origin='unconfirmed', offset=0.522, max_age=0.5, origin_lateral_margin=None):
        with self._lock:
            if vehicle_id not in self._active or not self._samples.get(vehicle_id):
                return None
            stamp,x,y = self._samples[vehicle_id][-1]
            measured = self._measured_bodies.get(vehicle_id)
            if measured and 0.0 <= now-measured.stamp <= max_age and abs(stamp-measured.stamp) <= 0.2:
                return measured
            heading = self._body_headings.get(vehicle_id)
            fresh = 0.0 <= now-stamp <= max_age
            body = body_pose(x,y,heading[0] if heading else None,stamp,
                frame=self._body_frames.get(vehicle_id,'map'),
                source=(heading[1]+'_held' if heading else 'unknown'),
                direction_valid=bool(heading and heading[2]),origin=origin,offset=offset,
                uncertainty=self._body_position_uncertainty.get(vehicle_id,0.),
                origin_lateral_margin=origin_lateral_margin)
            if heading:
                body = replace(body,yaw_stamp=heading[3])
            if not fresh:
                body = replace(body, position_valid=False, yaw_source='stale')
            return body

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
    failed_lane_idx, attempted_outer_lanes=(), reconsider_attempted=False
):
    """Return opposite outer, failed outer re-probe, then center lane."""
    if failed_lane_idx == 0:
        ordered = (2, 0, 1)
    elif failed_lane_idx == 2:
        ordered = (0, 2, 1)
    else:
        ordered = (0, 2, 1)
    attempted = set() if reconsider_attempted else set(attempted_outer_lanes)
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


def strict_shadow_slow_commit_creep_allowed(
    *,
    target_matches: bool,
    shadow_verified: bool,
    committed_outer_lane: bool,
    target_is_slow: bool,
    current_envelopes_separated: bool,
    candidate_passable: bool,
    candidate_conflicts,
) -> bool:
    """Allow forward creep only for a still-safe verified slow-car pass.

    This is deliberately stricter than ordinary overtake acquisition.  It is
    only a deadlock breaker after the exact target/lane has passed strict
    Shadow MPC verification; any current body overlap, physical-width loss or
    live front/side traffic conflict restores the normal full-stop behaviour.
    Rear traffic is intentionally left to the existing merge/parallel safety
    layers and does not prevent the low-speed forward motion itself.
    """
    return bool(
        target_matches
        and shadow_verified
        and committed_outer_lane
        and target_is_slow
        and current_envelopes_separated
        and candidate_passable
        and not candidate_conflicts.get("front")
        and not candidate_conflicts.get("side")
    )


def l0_restricted_follow_can_ignore_passage(
    *, restriction_active, lane_has_vehicle_width, non_target_conflicts
) -> bool:
    """Keep L0 when only the followed target invalidates passing clearance.

    Rear-only traffic is intentionally ignored by the L0-priority policy, but
    any unrelated front/side vehicle or an actual lane-width loss still blocks
    the override.
    """
    return bool(
        restriction_active
        and lane_has_vehicle_width
        and not non_target_conflicts.get("front")
        and not non_target_conflicts.get("side")
    )


def select_l2_restricted_zone_lane(
    candidate_lane_idx,
    *,
    restriction_active,
    slow_lead_override,
    l0_physically_passable,
    l0_conflicts,
):
    """Apply the local L0 -> L1 policy without yielding to rear-only traffic.

    A front-only L0 conflict is a follow target, not a reason to select L2.
    Side traffic still blocks L0. A physical-passage failure is accepted only
    with a front target because following it does not require passing width.
    """
    if not restriction_active or slow_lead_override:
        return candidate_lane_idx
    l0_front_occupied = bool(l0_conflicts.get("front"))
    l0_side_clear = not bool(l0_conflicts.get("side"))
    if l0_side_clear and (l0_physically_passable or l0_front_occupied):
        return 0
    return 1


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


def evaluate_lane_width_samples(
    widths, *, required_width, tolerance, max_consecutive_tolerated
):
    """Evaluate a lane horizon while tolerating only short, minor CSV noise.

    A deficit larger than ``tolerance`` is always blocking.  Smaller deficits
    are accepted only while their consecutive run does not exceed the
    configured point count.  The returned details are intended for passage
    diagnostics as well as tests.
    """
    required = max(float(required_width), 0.0)
    allowed_deficit = max(float(tolerance), 0.0)
    allowed_run = max(int(max_consecutive_tolerated), 0)
    minimum_width = float("inf")
    current_minor_run = 0
    longest_minor_run = 0
    first_failed_index = None
    failure_reason = None

    for index, raw_width in enumerate(widths):
        width = float(raw_width)
        minimum_width = min(minimum_width, width)
        deficit = required - width
        if deficit <= 0.0:
            current_minor_run = 0
            continue
        if deficit > allowed_deficit:
            first_failed_index = index
            failure_reason = "lane_width_deficit_exceeds_tolerance"
            break
        current_minor_run += 1
        longest_minor_run = max(longest_minor_run, current_minor_run)
        if current_minor_run > allowed_run:
            first_failed_index = index
            failure_reason = "lane_width_minor_deficit_run_too_long"
            break

    return {
        "passable": first_failed_index is None,
        "minimum_width": minimum_width,
        "longest_minor_run": longest_minor_run,
        "first_failed_index": first_failed_index,
        "failure_reason": failure_reason,
    }


def evaluate_overtake_commit_gate(
    *, target_lane, target_distance, preview_curvatures,
    minimum_distance, maximum_distance, outside_curvature_threshold,
):
    """Gate only the curve-outside direction of a new L0/L2 commitment."""
    lane = int(target_lane) if target_lane in (0, 2) else None
    distance = float(target_distance)
    curvatures = [float(value) for value in preview_curvatures]
    # Positive curvature is a left turn: L0 (right) is outside. Negative
    # curvature is a right turn: L2 (left) is outside.
    outside_curvature = (
        max([value for value in curvatures if value > 0.0], default=0.0)
        if lane == 0
        else max([-value for value in curvatures if value < 0.0], default=0.0)
        if lane == 2
        else float("inf")
    )
    distance_ready = bool(
        float(minimum_distance) <= distance <= float(maximum_distance))
    curvature_ready = bool(
        outside_curvature <= float(outside_curvature_threshold))
    return {
        "allowed": bool(lane is not None and distance_ready and curvature_ready),
        "distance_ready": distance_ready,
        "curvature_ready": curvature_ready,
        "outside_curvature": outside_curvature,
    }


def slow_lead_commit_distance(
    *, lead_is_stationary: bool, lead_speed: float,
    ultra_slow_speed_threshold: float, early_commit_distance: float,
    ordinary_slow_commit_distance: float,
) -> float:
    """Give only stopped/ultra-slow leads the longer commit distance."""
    use_early_commit = bool(
        lead_is_stationary
        or (
            math.isfinite(float(lead_speed))
            and float(lead_speed) <= max(
                float(ultra_slow_speed_threshold), 0.0)
        )
    )
    return float(
        early_commit_distance
        if use_early_commit else ordinary_slow_commit_distance
    )


def hybrid_lateral_escape_creep_allowed(
    *,
    target_matches: bool,
    transition_active: bool,
    target_is_slow: bool,
    rectangles_overlap: bool,
    lateral_body_gap: float,
    minimum_lateral_body_gap: float,
    candidate_passable: bool,
    candidate_conflicts,
) -> bool:
    """Break the zero-speed/spatial-transition interlock safely.

    Unlike the strict Shadow creep, this is usable immediately after reverse
    recovery, when a new Shadow proof may not yet exist.  It therefore requires
    the exact active Hybrid target, positive corner-aware lateral separation,
    a physically passable candidate lane, and no live front/side traffic.
    """
    return bool(
        target_matches
        and transition_active
        and target_is_slow
        and not rectangles_overlap
        and float(lateral_body_gap) >= float(minimum_lateral_body_gap)
        and candidate_passable
        and not candidate_conflicts.get("front")
        and not candidate_conflicts.get("side")
    )


def rolling_precommit_speed_margin(
    *, base_margin: float, far_bonus: float, target_distance: float,
    commit_distance: float, prepare_distance: float,
) -> float:
    """Preserve momentum while an early Shadow overtake is being verified.

    The extra closing margin is largest at the far prepare gate and fades to
    zero at the configured close-range gate. A candidate that is still
    unverified near the target therefore returns to the conservative base
    margin automatically without sacrificing momentum at Shadow start.
    """
    span = max(float(prepare_distance) - float(commit_distance), 1e-6)
    alpha = min(max(
        (float(target_distance) - float(commit_distance)) / span,
        0.0,
    ), 1.0)
    return max(float(base_margin), 0.0) + max(float(far_bonus), 0.0) * alpha


def l1_rejoin_preemption_target_relevant(
    *, opponent_ahead_detected: bool, lead_is_stationary: bool,
    lead_is_special_slow: bool, opponent_vehicle_id,
    opponent_arc_distance: float, maximum_distance: float,
) -> bool:
    """Use metric slow-lead detection to preempt L1 rejoin after recovery.

    Waypoint-difference detection can be false for a physically forward target
    near a curve or lap seam. A valid stopped/slow target must still get the
    chance to re-acquire a verified outer lane instead of deadlocking in L1.
    """
    return bool(
        opponent_vehicle_id is not None
        and (opponent_ahead_detected
             or lead_is_stationary
             or lead_is_special_slow)
        and is_follow_retry_within_distance(
            opponent_arc_distance, maximum_distance)
    )
