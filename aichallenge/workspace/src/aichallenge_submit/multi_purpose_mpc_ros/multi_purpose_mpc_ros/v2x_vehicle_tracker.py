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
