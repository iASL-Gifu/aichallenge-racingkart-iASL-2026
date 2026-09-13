"""Read-only recovery snapshots; no solving or collision checks here."""
import json
import math
import time
import numpy as np


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def snapshot(c, event, now, **extra):
    m = c._mpc
    path = m.model.reference_path
    from .control_continuity import REFERENCE_FIELDS
    lower = np.asarray(getattr(m, '_prediction_lower_bounds', []), dtype=float).reshape(-1)
    upper = np.asarray(getattr(m, '_prediction_upper_bounds', []), dtype=float).reshape(-1)
    wp = np.asarray(getattr(m, '_constraint_wp_ids', []), dtype=int).reshape(-1)
    physical = np.asarray(getattr(m, '_constraint_physical_free_widths', []), dtype=float).reshape(-1)
    count = min(len(lower), len(upper), len(wp))
    indexes = sorted(range(count), key=lambda i: upper[i]-lower[i])[:3]
    traffic = []
    tracker = getattr(c, '_v2x_tracker', None)
    if tracker is not None:
        for vid in tracker.active_vehicle_ids():
            samples = tracker._samples.get(vid)
            if not samples:
                continue
            stamp, x, y = samples[-1]
            traffic.append(dict(id=vid, position=[x,y], age=now-stamp,
                velocity=tracker.velocity(vid), velocity_valid=tracker.has_velocity_estimate(vid)))
        pose = m.model.temporal_state
        traffic.sort(key=lambda t: (t['position'][0]-pose.x)**2+(t['position'][1]-pose.y)**2)
    return clean(dict(event=event, sim_time=now, wp=m.model.wp_id,
        pose=[m.model.temporal_state.x, m.model.temporal_state.y, m.model.temporal_state.psi],
        lane=path.target_lane_idx, mode=getattr(c, '_applied_corridor_mode', None),
        requested_lane=getattr(c._overtake, 'requested_lane', None), target=c._overtake.target_id,
        recovery=getattr(c,'_mpc_safety_recovery_active',False),
        fallback=getattr(c,'_steering_fallback_armed',False),
        accurate=getattr(m,'last_solution_accurate',False), failure=m.failure_reason,
        handoff_reason=getattr(c,'_mpc_handoff_reason',None),
        constraint_lane=getattr(m,'_constraint_target_lane',None),
        relaxation=getattr(m,'_constraint_lane_relaxation',None),
        reference={k:getattr(m,k,None) for k in REFERENCE_FIELDS},
        bounds=[dict(wp=int(wp[i]), reference_lb=lower[i],reference_ub=upper[i],
                     physical_width=physical[i] if i<len(physical) else None) for i in indexes],
        traffic=traffic[:6], extra=extra))


def diagnostic_due(c, key, state, interval=1.):
    """Gate expensive collection, emitting changed states and clock rollback."""
    clock = getattr(c, 'get_clock', None)
    now = float(clock().now().nanoseconds)/1e9 if clock else time.monotonic()
    records = getattr(c, '_diagnostic_gates', None)
    if records is None:
        records = c._diagnostic_gates = {}
    previous = records.get(key)
    if previous is not None and previous[1] == state and 0 <= now-previous[0] < interval:
        return False
    records[key] = (now, state)
    return True


def log_event(c, event, extra_factory=None, **extra):
    if bool(getattr(getattr(getattr(c,"_cfg",None),"mpc",None),"minimal_logging",False)):
        return
    try:
        now = float(c.get_clock().now().nanoseconds)/1e9
        last = getattr(c, '_recovery_diagnostic_release', None)
        if event in ('path_rejected', 'speed_limit'):
            if not (getattr(c,'_mpc_safety_recovery_active',False)
                    or (last is not None and 0 <= now-last['sim_time'] <= 3.)):
                return
            stamp_field = '_recovery_diagnostic_' + event + '_at'
            previous = getattr(c, stamp_field, None)
            if previous is not None and 0 <= now-previous < 1.:
                return
            setattr(c, stamp_field, now)
        # Caller-side lists and sampled paths are constructed only for output.
        if extra_factory is not None:
            extra.update(extra_factory())
        m = c._mpc
        state = (getattr(c, '_mpc_safety_recovery_active', False),
                 getattr(c, '_steering_fallback_armed', False),
                 getattr(c, '_applied_corridor_mode', None),
                 m.model.reference_path.target_lane_idx, c._overtake.target_id,
                 m.failure_reason, extra.get('reason'))
        field = '_recovery_diagnostic_' + event + '_state'
        repeated = (event in ('path_rejected', 'speed_limit')
                    and getattr(c, field, None) == state)
        if repeated:
            data = clean(dict(event=event, sim_time=now, wp=m.model.wp_id,
                              summary=True, extra=extra))
        else:
            data = snapshot(c,event,now,**extra)
        setattr(c, field, state)
        if event == 'enter' and last is not None:
            data['previous_release'] = last
            data['since_release_sec'] = clean(now-last['sim_time'])
        if event == 'release':
            c._recovery_diagnostic_release = data
        c.get_logger().info('[RecoveryTransitionDiagnostic] '+json.dumps(data,ensure_ascii=False,allow_nan=False))
    except Exception as error:
        c.get_logger().warn('[RecoveryTransitionDiagnostic] collection failed: '+str(error),
                            throttle_duration_sec=5.)


def routine_log_due(c, key, state=None):
    minimal=bool(getattr(getattr(getattr(c,'_cfg',None),'mpc',None),'minimal_logging',False))
    return diagnostic_due(c,key,state,interval=(30. if state is None else 5.) if minimal else 0.)
