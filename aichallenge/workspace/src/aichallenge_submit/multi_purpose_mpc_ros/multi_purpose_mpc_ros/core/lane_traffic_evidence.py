"""Use a recent exact-lane prediction to resolve unknown lane labels.

This is traffic evidence only, never permission to skip final live MPC/wall
admission. Stale or missing body/path evidence retains the unknown verdict.
"""
from types import SimpleNamespace
import math
import numpy as np
from .control_continuity import fresh_solution, timed_mpc_path
from .path_check_work import prepare_bodies, traffic_clear
from .. import collision_geometry as collision


def remember(c, mpc, lane, target, *, accepted=True):
    now = float(getattr(c, '_collision_now', math.nan))
    if lane not in (0, 2) or target is None or not math.isfinite(now):
        return
    records = getattr(c, '_lane_traffic_paths', None)
    if records is None:
        c._lane_traffic_paths = records = {}
    key = (id(c._reference_path), target, lane)
    if not accepted or not fresh_solution(mpc) or not getattr(mpc, 'last_solution_accurate', False):
        records.pop(key, None)
        return
    points = np.asarray(getattr(mpc, 'current_recovery_prediction', None), dtype=float)
    times = np.asarray(getattr(mpc, 'current_prediction_times', None), dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or times.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) < 0.)):
        records.pop(key, None)
        return
    if len(records) >= 4:
        records.clear()
    delay = float(getattr(c, '_steering_command_delay', 0.)) if getattr(c, '_delay_prediction_enabled', False) else 0.
    records[key] = (now, points.copy(), times.copy()+delay)


def unknown_lane_conflict(c, pose, lane, target_id, vehicle_id):
    key = (id(getattr(c, '_reference_path', None)), target_id, lane)
    record = getattr(c, '_lane_traffic_paths', {}).get(key)
    if record is None:
        return None
    stamp, points, times = record
    now = float(c._collision_now)
    age = now-stamp
    if not math.isfinite(age) or not 0. <= age <= .25:
        return None
    target = collision.target_body(c, vehicle_id)
    tracker = c._v2x_tracker
    if (target is None or not target.position_valid or not target.yaw_valid
            or not 0. <= now-target.stamp <= getattr(c, '_collision_max_age', .5)
            or not tracker.has_velocity_estimate(vehicle_id)):
        return None
    velocity = tracker.velocity(vehicle_id)
    if not all(math.isfinite(v) for v in velocity):
        return None
    # Remove elapsed prediction. Connect the actual current pose to remaining
    # points and keep their original arrival times, rather than restarting time.
    keep = times > age
    remaining = points[keep]
    remaining_times = times[keep]-age
    if (not len(remaining) or remaining_times[-1] < max(
            float(getattr(c, '_prepass_lane_fallback_prediction_sec', 1.)), 1.)
            or math.hypot(remaining[0,0]-pose.x, remaining[0,1]-pose.y) > 2.):
        return None
    prediction = SimpleNamespace(
        current_recovery_prediction=np.vstack(([pose.x,pose.y,pose.theta], remaining)),
        current_prediction_times=np.concatenate(([0.],remaining_times)))
    timed = timed_mpc_path(prediction, pose)
    if timed is None:
        return None
    path, stamps = timed
    bodies = prepare_bodies(c, path)
    if not all(b.position_valid and b.yaw_valid for b in bodies):
        return None
    observation_times = collision.prediction_times_from_observation(target.stamp, now, stamps)
    return not traffic_clear(c, bodies, observation_times, target, velocity, collision.geometry(c))
