"""Private full-width continuation of a short recovery motion.

Endpoint predictions are never adopted at the current pose. Only their lateral
intent can be proposed to ordinary transactional admission, with a fresh solve.
"""
import math
from types import SimpleNamespace
from .control_continuity import fork_solver, fresh_solution
from .overtake_reference import prepare_entry_reference
from .path_check_work import prepare_mpc_path
from .runtime_diagnostics import detail_scope


def requested_lane(c):
    session = getattr(c, '_overtake', None)
    lane = (getattr(c, '_prepass_soft_candidate_lane_idx', None)
            if getattr(c, '_prepass_fallback_recovery_active', False) else None)
    if lane not in (0, 2):
        lane = getattr(session, 'requested_lane', None)
    if lane not in (0, 2):
        lane = getattr(c, '_prepass_soft_candidate_lane_idx', None)
    if (lane not in (0, 2) or getattr(session, 'target_id', None) is None
            or getattr(c, '_reference_path', None) is not getattr(c, '_reference_pathN_center', None)
            or getattr(c, '_manual_control_override', False)
            or getattr(c, '_parallel_abort_active', False)
            or getattr(c, '_intentional_follow_stop_active', False)):
        return None
    return lane


def connection_goal(c, wp):
    lane = requested_lane(c)
    if lane is None:
        return None
    path = c._reference_path
    def point(index):
        waypoint = path.get_waypoint(index)
        lateral = c._mpc._compute_lane_center(index, lane)
        return (waypoint.x-lateral*math.sin(waypoint.psi),
                waypoint.y+lateral*math.cos(waypoint.psi))
    following = (wp+1) % path.n_waypoints if path.circular else min(wp+1,path.n_waypoints-1)
    a, b = point(wp), point(following)
    heading = math.atan2(b[1]-a[1],b[0]-a[0]) if following != wp else path.get_waypoint(wp).psi
    if not all(math.isfinite(v) for v in (*a,heading)):
        return None
    return a, heading


def set_guidance(c, mpc, lane, speed):
    model = mpc.model
    model.get_current_waypoint()
    model.spatial_state = model.t2s(reference_state=model.temporal_state, reference_waypoint=model.current_waypoint)
    model.reference_path.target_lane_idx = None
    model.reference_path.is_overtaking = False
    mpc.set_full_width_l1_offset_limits()
    mpc.set_full_width_l0_offset_limits()
    mpc.set_target_lane_lateral_offsets()
    prepare_entry_reference(c, mpc, lane, speed)
    # Reference only: no artificial outer lane bounds, including transition weights.
    mpc.set_lane_transition_weights()


def connection_checker(c, now, estimated_steering):
    lane = requested_lane(c)
    if lane is None:
        return None
    target = c._overtake.target_id
    # Evidence expires unless this selection produces a new checked connection.
    c._recovery_connection_intent = None
    attempts = 0
    def check(motion):
        nonlocal attempts
        if attempts >= 3:
            return False, 'connection_probe_budget'
        attempts += 1
        speed = min(1., c._straight_reentry_speed, motion.speed_limit)
        if not math.isfinite(speed) or speed <= .01:
            return False, 'connection_speed'
        end = SimpleNamespace(x=motion.poses[-1][0], y=motion.poses[-1][1], theta=motion.poses[-1][2])
        measured = abs(c._velocity_report.longitudinal_velocity)
        stopped = measured <= .1
        duration = sum(math.hypot(b[0]-a[0],b[1]-a[1])
                       for a,b in zip(motion.poses,motion.poses[1:]))/max(speed,measured)
        change = c._mpc.max_steering_rate*duration
        endpoint_steering = (motion.steering if stopped else estimated_steering
            + max(-change,min(change,motion.steering-estimated_steering)))
        delay = (max(0., float(getattr(c, '_steering_command_delay', 0.)))
                 if getattr(c, '_delay_prediction_enabled', False) else 0.)
        from .boundary_recovery import rollout
        delayed = rollout((end.x,end.y,end.theta),1,endpoint_steering,
                          speed*delay,float(getattr(getattr(getattr(c,'_cfg',None),
                          'bicycle_model',None),'length',1.087)))[-1] if delay else (end.x,end.y,end.theta)
        with detail_scope(c, 'recovery_connection.endpoint_mpc'):
            probe = fork_solver(c._mpc)
            probe.model.update_states(*delayed)
            probe.previous_steering = endpoint_steering
            probe.update_wp_id_offset(0)
            probe.update_v_max(speed)
            set_guidance(c, probe, lane, speed)
            command, _ = probe.get_control()
        if not fresh_solution(probe) or not probe.last_solution_accurate or not math.isfinite(float(command[0])) or command[0] <= .01:
            return False, 'endpoint_mpc_unavailable'
        continuation = prepare_mpc_path(c, probe, end, delay)
        if continuation is None:
            return False, 'endpoint_times'
        # Include steering preparation while stationary in the traffic forecast.
        remaining = abs(motion.steering-estimated_steering)
        wait = max(0., float(getattr(c, '_reentry_steering_ready_at', now))-now) if stopped else 0.
        if stopped and remaining > .01:
            wait = max(wait, remaining/c._mpc.max_steering_rate+.25)
        gear_ready = getattr(c, '_current_gear_is_drive', None)
        if gear_ready is not None and not gear_ready():
            wait = max(wait, float(getattr(c, '_stuck_gear_shift_delay', 1.)))
        path = [motion.poses[0], motion.poses[0]]
        times = [0., wait]
        for a, b in zip(motion.poses, motion.poses[1:]):
            path.append(b)
            times.append(times[-1]+math.hypot(b[0]-a[0], b[1]-a[1])/speed)
        offset = times[-1]
        path.extend(continuation[0][1:])
        times.extend(offset+t for t in continuation[1][1:])
        with detail_scope(c, 'recovery_connection.combined_check'):
            failure_history = getattr(c, '_recovery_attempts', None)
            safe, reason = failure_history.wall_path_is_clear(path) if failure_history else (True, 'clear')
            if safe:
                safe, reason = c._reentry_path_is_clear(path, times=times)
        if safe:
            c._recovery_connection_intent = (target, lane, id(c._reference_path), now)
        return safe, reason
    return check


def apply_guidance(c, mpc, speed):
    intent = getattr(c, '_recovery_connection_intent', None)
    if intent is None:
        return False
    target, lane, path_id, stamp = intent
    now = float(getattr(c, '_collision_now', math.inf))
    if (requested_lane(c) != lane or c._overtake.target_id != target
            or id(c._reference_path) != path_id or not 0. <= now-stamp <= .5
            or not getattr(c, '_straight_reentry_active', False)):
        c._recovery_connection_intent = None
        return False
    # Runs before CorridorState.capture: final admission checks the exact new
    # full-width proposal. No future-pose control is copied into the live MPC.
    set_guidance(c, mpc, lane, min(1., max(0., speed)))
    return True
