"""Prepare a voluntary return without prematurely ending the applied pass."""
import copy
from .control_continuity import CorridorState, fresh_solution


def prepare(c, destination):
    state = getattr(c, '_committed_corridor', None)
    if (not isinstance(state, CorridorState) or state.lane not in (0, 2)
            or state.path is not c._reference_path
            or state.target is None or state.target != c._overtake.target_id):
        return True
    now = c._collision_now
    key = (id(state.path), state.target, state.lane, destination)
    last = getattr(c, '_early_rejoin_retry', None)
    if last and last[0] == key and 0. <= now-last[1] < .25:
        return False
    c._early_rejoin_trial = (key, now, copy.deepcopy(c._overtake))
    return True


def finish(c, accepted):
    trial = getattr(c, '_early_rejoin_trial', None)
    if trial is None:
        return
    key, now, session = trial
    if accepted:
        c._early_rejoin_retry = None
    else:
        # Geometry was restored by the corridor transaction. Restore its pass
        # ownership/proof as well, before speed arbitration sees the result.
        c._overtake.__dict__.update(session.__dict__)
        c._target_lane_idx = session.requested_lane
        c._early_rejoin_retry = (key, now)
    c._early_rejoin_trial = None


def speed_handoff_clear(c, target, pose, command):
    context = getattr(c, '_checked_return_target', None)
    if context is not None and context != (id(c._reference_path), c._overtake.target_id):
        c._checked_return_target = None
        return False
    if (context is None or context != (id(c._reference_path), target)
            or target is None or c._overtake.target_id != target
            or not fresh_solution(c._mpc)
            or any(getattr(c, k, False) for k in (
                '_mpc_safety_recovery_active', '_post_reverse_full_width_recovery_active',
                '_parallel_abort_active', '_steering_fallback_armed',
                '_prepass_fallback_recovery_active', '_manual_control_override'))):
        return False
    # Only the pre-commit speed-matching label changes. All ordinary braking
    # and dynamic vehicle checks still run downstream.
    return (c._return_target_is_separated(pose, target)
            and c._mpc_prediction_path_is_clear(pose, command))
