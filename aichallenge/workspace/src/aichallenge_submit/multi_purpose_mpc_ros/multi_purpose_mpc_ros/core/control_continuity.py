"""Transactional corridor application and bounded reuse of a spatial MPC plan.

No ROS or traffic decisions live here. Callers supply current geometry checks.
"""
import copy
import math
from dataclasses import dataclass

import numpy as np


REFERENCE_FIELDS = (
    'soft_target_lane_idx', 'soft_target_start_e_y', 'soft_target_alpha',
    'soft_target_lateral_offset', 'soft_lateral_targets',
    'lane_transition_weights', 'target_lane_lateral_offsets',
    'full_width_l1_offset_limits', 'full_width_l0_offset_limits',
)
HORIZON_FIELDS = REFERENCE_FIELDS[4:]
# These are application progress, not target/side selection or traffic proof.
PROGRESS_FIELDS = (
    '_lane_decision', '_applied_corridor_mode', '_constraint_transition_until',
    '_l1_entry_waiting',
    '_last_lane_change_time', '_hybrid_reference_key', '_l1_probe_constraint_applied',
    '_l1_soft_rejoin_started_at', '_l1_soft_rejoin_start_e_y',
    '_l1_soft_rejoin_effective_ramp_sec', '_l1_soft_rejoin_full_strength_logged',
    '_overtake_soft_transition_lane_idx', '_overtake_soft_transition_start_e_y',
    '_initial_start_soft_l0_started_at', '_initial_start_soft_l0_start_e_y',
    '_initial_start_soft_l0_effective_ramp_sec', '_initial_start_soft_l0_full_strength_logged',
    '_initial_start_soft_l0_curvature_shift', '_initial_start_soft_l0_last_update_sec',
    '_prepass_soft_guidance_started_at', '_prepass_soft_guidance_start_e_y',
    '_prepass_soft_guidance_ramp_sec', '_prepass_soft_guidance_key',
    '_prepass_soft_guidance_paused_at',
    '_race_rejoin_handoff_started_at', '_race_rejoin_handoff_start_e_y',
    '_race_rejoin_handoff_effective_ramp_sec', '_race_rejoin_handoff_guidance_ready',
)


@dataclass
class CorridorState:
    path: object
    lane: object
    overtaking: bool
    reference: dict
    wp: int
    progress: dict
    hybrid: object
    accepted_key: tuple
    target: object
    stamp: float

    @classmethod
    def capture(cls, controller):
        from multi_purpose_mpc_ros.core.runtime_diagnostics import detail_scope
        with detail_scope(controller, 'state_copy.corridor_capture'):
            m = controller._mpc
            return cls(
                m.model.reference_path, m.model.reference_path.target_lane_idx,
                m.model.reference_path.is_overtaking,
                {k: copy.deepcopy(getattr(m, k)) for k in REFERENCE_FIELDS},
                int(m.model.wp_id),
                {k: copy.deepcopy(getattr(controller, k)) for k in PROGRESS_FIELDS
                 if hasattr(controller, k)},
                copy.deepcopy(controller._overtake.hybrid),
                controller._overtake.accepted_key, controller._overtake.target_id,
                float(getattr(controller, '_collision_now', 0.)))

    def apply(self, mpc):
        # Rebase horizon-indexed objectives to the current waypoint. Never
        # rewind the measured pose, traffic, speed limits, or map observations.
        from multi_purpose_mpc_ros.core.runtime_diagnostics import detail_scope
        with detail_scope(mpc, 'state_copy.corridor_apply'):
            advance = (int(mpc.model.wp_id) - self.wp) % self.path.n_waypoints
            if advance > self.path.n_waypoints // 2:
                advance -= self.path.n_waypoints
            mpc.model.reference_path.target_lane_idx = self.lane
            mpc.model.reference_path.is_overtaking = self.overtaking
            for key, value in self.reference.items():
                value = copy.deepcopy(value)
                if key in HORIZON_FIELDS and value is not None and len(value):
                    indexes = np.clip(np.arange(len(value)) + advance, 0, len(value)-1)
                    value = np.asarray(value)[indexes].copy()
                setattr(mpc, key, value)

    def restore_progress(self, controller, *, pause=False, restore_manoeuvre=True):
        from multi_purpose_mpc_ros.core.runtime_diagnostics import detail_scope
        with detail_scope(controller, 'state_copy.progress_restore'):
            elapsed = max(float(getattr(controller, '_collision_now', self.stamp))-self.stamp, 0.) if pause else 0.
            for key, value in self.progress.items():
                if value is not None and (key.endswith('_started_at')
                        or key in ('_constraint_transition_until', '_last_lane_change_time',
                                   '_initial_start_soft_l0_last_update_sec',
                                   '_prepass_soft_guidance_paused_at')):
                    value += elapsed
                setattr(controller, key, copy.deepcopy(value))
            # Never resurrect another target's pass or its clearance evidence.
            if restore_manoeuvre and controller._overtake.target_id == self.target:
                controller._overtake.hybrid = copy.deepcopy(self.hybrid)
                controller._overtake.accepted_key = self.accepted_key

    def equivalent(self, other):
        return (self.path is other.path and self.lane == other.lane
                and self.overtaking == other.overtaking
                and self.target == other.target and self.accepted_key == other.accepted_key
                and all(getattr(self.hybrid, k) == getattr(other.hybrid, k)
                        for k in ('vehicle_id', 'lane_idx', 'start_wp', 'started_at', 'start_e_y', 'length'))
                and self.progress.get('_applied_corridor_mode') == other.progress.get('_applied_corridor_mode')
                and self.progress.get('_lane_decision') == other.progress.get('_lane_decision')
                and all(np.array_equal(self.reference[k], other.reference[k])
                        for k in REFERENCE_FIELDS))

    def continues_transition(self, other):
        """A new horizon of the already admitted spatial manoeuvre, not a switch.

        Only the running hybrid trajectory may change its sampled objective and
        weights here. New targets, anchors, constraint modes and lateral goals
        still require private admission. A fresh solve/check remains mandatory.
        """
        if (self.path is not other.path or self.lane != other.lane
                or self.overtaking != other.overtaking
                or self.target != other.target or self.accepted_key != other.accepted_key
                or self.progress.get('_lane_decision') != other.progress.get('_lane_decision')
                or self.progress.get('_applied_corridor_mode') != 'hybrid_lane_transition'
                or other.progress.get('_applied_corridor_mode') != 'hybrid_lane_transition'):
            return False
        a, b = self.hybrid, other.hybrid
        anchors = ('vehicle_id', 'lane_idx', 'start_wp', 'started_at', 'start_e_y', 'length')
        if (a.start_wp is None or a.paused or b.paused or a.completed
                or any(getattr(a, k) != getattr(b, k) for k in anchors)
                or not math.isfinite(b.travelled) or b.travelled < a.travelled):
            return False
        evolving = {'soft_lateral_targets', 'lane_transition_weights'}
        for k in REFERENCE_FIELDS:
            x, y = self.reference[k], other.reference[k]
            if k not in evolving:
                if not np.array_equal(x, y):
                    return False
            elif ((x is None) != (y is None)
                  or (y is not None and (np.shape(x) != np.shape(y)
                      or not np.isfinite(y).all()))):
                return False
        return True


def postpass_alternative(controller, candidate):
    """Scope an outer continuation to the completed pass and ordinary rejoin.

    Safety owners and a new traffic/policy manoeuvre always take precedence.
    The caller must solve and collision-check this geometry before using it.
    """
    state = getattr(controller, '_postpass_outer_corridor', None)
    if not isinstance(state, CorridorState):
        return None
    if (state.path is not candidate.path or state.target != candidate.target
            or state.target != getattr(controller, '_overtake_completed_target_id', None)
            or candidate.lane not in (None, 1, state.lane)):
        return None
    if (getattr(controller, '_stuck_recovery_until', None) is not None
            or getattr(controller, '_straight_reentry_active', False)):
        return None
    if any(getattr(controller, key, False) for key in (
            '_postpass_rejoin_suppressed', '_manual_recovery_reset_pending',
            '_follow_only', '_follow_escape_active', '_prepass_retry_after_reverse',
            '_manual_control_override', '_mpc_safety_recovery_active',
            '_post_reverse_full_width_recovery_active', '_parallel_abort_active',
            '_prepass_fallback_recovery_active', '_prepass_fallback_follow_active',
            '_l1_safety_recovery_active', '_l1_rejoin_backoff_active')):
        return None
    # A fixed outer request belongs to geographic policy or the next pass;
    # regular admission already handles that, so end the rejoin alternative.
    if candidate.lane in (0, 2):
        return None
    return state


def timed_mpc_path(mpc, pose, delay=0.):
    """Densify the accepted spatial solution with its own time state.

    The spatial model resets t to zero at the delayed initial pose. Preserve
    that origin, interpolate time with position/yaw, and include current pose.
    Never invent an arrival time by dividing the whole path by one speed.
    """
    points = np.asarray(getattr(mpc, 'current_recovery_prediction', None), dtype=float)
    times = np.asarray(getattr(mpc, 'current_prediction_times', None), dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or times.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(times).all() or not math.isfinite(delay) or delay < 0.
            or abs(times[0]) > .001 or np.any(np.diff(times) < -1e-6)):
        return None
    times = np.maximum.accumulate(np.maximum(times, 0.)) + delay
    path, stamps = [tuple(map(float, (pose.x, pose.y, pose.theta)))], [0.]
    for point, end_time in zip(points, times):
        x, y, yaw = path[-1]
        start_time = stamps[-1]
        dx, dy = point[0]-x, point[1]-y
        angle = math.atan2(math.sin(point[2]-yaw), math.cos(point[2]-yaw))
        steps = max(1, math.ceil(math.hypot(dx, dy)/.05), math.ceil(abs(angle)/.05))
        for i in range(1, steps+1):
            ratio = i/steps
            path.append((x+ratio*dx, y+ratio*dy, yaw+ratio*angle))
            stamps.append(start_time+ratio*(end_time-start_time))
    return path, stamps


def fork_solver(mpc):
    """Private mutable solver/model/boundaries; immutable map is shared."""
    from multi_purpose_mpc_ros.core.runtime_diagnostics import detail_scope
    with detail_scope(mpc, 'state_copy.solver_fork'):
        result = copy.copy(mpc)
        # Timing wrappers close over the original bound method. Copying them would
        # run a candidate build on the live MPC and corrupt its solver state.
        for key, value in vars(mpc).items():
            if hasattr(value, '_timing_original'):
                result.__dict__.pop(key, None)
        result._runtime_role = lambda: 'corridor_probe_mpc'
        for key, value in vars(mpc).items():
            if isinstance(value, (np.ndarray, dict, list)):
                setattr(result, key, copy.deepcopy(value))
        result.model = copy.copy(mpc.model)
        result.model.temporal_state = copy.deepcopy(mpc.model.temporal_state)
        result.model.spatial_state = copy.deepcopy(mpc.model.spatial_state)
        path = copy.copy(mpc.model.reference_path)
        path.waypoints = [copy.copy(wp) for wp in path.waypoints]
        path.border_cells = copy.deepcopy(path.border_cells)
        path.unsafe_static_fallback_wp_ids = []
        result.model.reference_path = path
        result.model.current_waypoint = path.get_waypoint(result.model.wp_id)
        result.optimizer = type(mpc.optimizer)()
        result.osqp_initialized = False
        result._continuity_warm_start = copy.deepcopy(getattr(mpc, 'last_solution_primal', None))
        return result


def adopt_solver(mpc, candidate):
    """Keep public model/path identities and adopt the exact checked solution."""
    from multi_purpose_mpc_ros.core.runtime_diagnostics import detail_scope
    with detail_scope(mpc, 'state_copy.solver_adopt'):
        model, path = mpc.model, mpc.model.reference_path
        role = getattr(mpc, '_runtime_role', None)
        checked_path = candidate.model.reference_path
        for key in ('border_cells', 'last_constraint_bounds', 'unsafe_static_fallback_wp_ids',
                    'rect_points', 'upper_cols', 'lower_cols', 'free_segs', 'select_free_segs',
                    'modified_ub', 'modified_lb'):
            if hasattr(checked_path, key):
                setattr(path, key, getattr(checked_path, key))
        for original, checked in zip(getattr(path, 'waypoints', ()), getattr(checked_path, 'waypoints', ())):
            for key in ('ub_sm', 'lb_sm', 'dynamic_border_cells'):
                setattr(original, key, getattr(checked, key, None))
        model.__dict__.update(candidate.model.__dict__)
        model.reference_path = path
        model.current_waypoint = path.get_waypoint(model.wp_id)
        mpc.__dict__.update(candidate.__dict__)
        mpc.model = model
        if role is not None:
            mpc._runtime_role = role


def fresh_solution(mpc):
    return (mpc.infeasibility_counter == 0 and mpc.current_prediction is not None
            and not mpc.used_prediction_fallback and not mpc.recovery_requested
            and not mpc.time_budget_exceeded)


@dataclass
class RemainingPlan:
    stamp: float
    path_id: object
    lane: object
    points: np.ndarray
    controls: np.ndarray
    command: np.ndarray

    @classmethod
    def capture(cls, mpc, stamp, command):
        points = np.asarray(mpc.current_recovery_prediction, dtype=float)
        controls = np.asarray(mpc.current_control, dtype=float).reshape(-1, 2)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or len(controls) < len(points) or not np.isfinite(points).all()
                or not np.isfinite(controls).all()):
            return None
        return cls(float(stamp), mpc.model.reference_path,
                   mpc.model.reference_path.target_lane_idx,
                   points.copy(), controls.copy(), np.asarray(command).copy())

    def rollout(self, pose, *, now, speed, previous_command, steering,
                wheelbase, rate, delay, deceleration, period, understeer=0.,
                alignment_pose=None, steering_history=()):
        """Align by measured progress; follow stored steering through a stop.

        Controls are spatial samples, not one sample per control-loop tick.
        The returned timed trajectory includes command delay and braking.
        """
        aligned = pose if alignment_pose is None else alignment_pose
        age = float(now) - self.stamp
        if not 0. <= age <= 3.*period + 1e-6:
            return None, 'expired'
        values = (*pose, *aligned, speed, *previous_command, steering, wheelbase, rate,
                  delay, deceleration, period, understeer)
        if not all(math.isfinite(float(x)) for x in values):
            return None, 'nonfinite'
        if speed <= .05 or previous_command[0] <= .05:
            return None, 'not_moving_forward'
        if min(wheelbase, rate, deceleration, period) <= 0. or delay < 0.:
            return None, 'invalid_dynamics'
        lengths = np.linalg.norm(np.diff(self.points[:, :2], axis=0), axis=1)
        arc = np.r_[0., np.cumsum(lengths)]
        best = None
        for i, length in enumerate(lengths):
            if length < 1e-6:
                continue
            a, b = self.points[i], self.points[i+1]
            ratio = float(np.clip(np.dot(np.asarray(aligned[:2])-a[:2], b[:2]-a[:2]) / length**2, 0., 1.))
            point = a[:2] + ratio*(b[:2]-a[:2])
            error = float(np.linalg.norm(np.asarray(aligned[:2])-point))
            progress = float(arc[i]+ratio*length)
            if progress > speed*(age+delay)+.5:
                continue
            heading = a[2]+ratio*math.atan2(math.sin(b[2]-a[2]), math.cos(b[2]-a[2]))
            heading_error = abs(math.atan2(math.sin(aligned[2]-heading), math.cos(aligned[2]-heading)))
            if best is None or error < best[0]:
                best = error, heading_error, progress, i
        if best is None or best[0] > .35 or best[1] > .25:
            return None, 'disconnected'
        _, _, progress, index = best
        command_speed = min(speed, previous_command[0], self.command[0], self.controls[index, 0])
        if command_speed <= .05:
            return None, 'no_forward_command'
        hold = delay + period
        stop_distance = speed*hold + speed**2/(2.*deceleration)
        if progress+stop_distance+.1 > arc[-1]:
            return None, 'insufficient_stopping_distance'
        x, y, yaw = map(float, pose)
        delta = float(steering)
        target = float(self.controls[index, 1])
        # Bound the command itself as well as the physical rollout.
        command_delta = float(np.clip(target, previous_command[1]-rate*period,
                                      previous_command[1]+rate*period))
        path, times = [(x, y, yaw)], [0.]
        t, distance = 0., 0.
        dt = min(.02, .05/max(speed, .1))
        while t < hold+speed/deceleration-1e-9:
            step = min(dt, hold+speed/deceleration-t)
            mid = t+step/2.
            velocity = max(0., speed-deceleration*max(0., mid-hold))
            i = min(int(np.searchsorted(arc, progress+max(distance-speed*delay, 0.), side='right')-1), len(self.controls)-1)
            desired = command_delta
            if t < delay:
                desired = steering
                for stamp, angle in reversed(steering_history):
                    if stamp <= now+t-delay:
                        desired = angle
                        break
            elif t >= hold:
                desired = self.controls[max(i, 0), 1]
            delta += float(np.clip(desired-delta, -rate*step, rate*step))
            turn = velocity*math.tan(delta)/(wheelbase*(1.+understeer*velocity**2))
            x += velocity*math.cos(yaw+turn*step/2.)*step
            y += velocity*math.sin(yaw+turn*step/2.)*step
            yaw += turn*step
            distance += velocity*step
            t += step
            path.append((x, y, yaw))
            times.append(t)
        return (np.array([command_speed, command_delta]), path, times), 'checked_rollout'


def l1_entry_confirmed(controller, now, ready, duration):
    """Continuous measured entry readiness, independent of request timers."""
    since = getattr(controller, '_l1_application_stable_since', None)
    if not ready or not math.isfinite(now):
        controller._l1_application_stable_since = None
        return False
    if since is None or now < since:
        since = now
    controller._l1_application_stable_since = since
    return now - since >= max(duration, 0.) - 1e-9


def prepass_return_alternative(controller, candidate):
    """Keep geometric guidance after target expiry, never its traffic proof."""
    state = getattr(controller, '_prepass_return_corridor', None)
    if not isinstance(state, CorridorState) or state.path is not candidate.path:
        return None
    if getattr(controller, '_stuck_recovery_until', None) is not None:
        return None
    if any(getattr(controller, key, False) for key in (
            '_postpass_rejoin_suppressed', '_manual_control_override',
            '_manual_recovery_reset_pending', '_straight_reentry_active',
            '_mpc_safety_recovery_active', '_post_reverse_full_width_recovery_active',
            '_parallel_abort_active', '_follow_escape_active',
            '_prepass_retry_after_reverse', '_prepass_fallback_follow_active',
            '_prepass_fallback_blocked', '_prepass_fallback_recovery_active',
            '_l1_safety_recovery_active', '_l1_rejoin_backoff_active')):
        return None
    return state
