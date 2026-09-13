"""Local L1 policy. Requests only: ordinary corridor and body checks still apply."""
import math
from .l0_preparation import approaching_zone


def active(c, wp=None):
    zones = getattr(c, '_normal_follow_zones', ())
    if not zones:
        return False
    if wp is None:
        wp = c._carN_center.wp_id
    wp = int(wp)
    if any((a <= wp <= b) if a <= b else (wp >= a or wp <= b) for a,b in zones):
        return True
    return approaching_zone(wp, zones, c._reference_pathN_center.segment_lengths,
                            getattr(c, '_normal_follow_prepare_distance', 0.))


def exception(c, target):
    if target is None or not c._v2x_tracker.has_velocity_estimate(target):
        return False
    vx, vy = c._v2x_tracker.velocity(target)
    speed = math.hypot(vx, vy)
    # Hysteresis only for an already committed pass on this same target.
    session = c._overtake
    continuing = (session.target_id == target and session.committed
                  and session.requested_lane in (0, 2))
    threshold = c._ultra_slow_early_commit_speed + (.3 if continuing else 0.)
    return math.isfinite(speed) and speed <= threshold


def blocked(c, target, wp=None):
    return active(c, wp) and not exception(c, target)


def final_request(c, lane, pose, speed):
    """Preserve safe applied motion if merging is blocked; never force L1."""
    if not blocked(c, c._overtake.target_id):
        return lane, False
    if any(getattr(c,k,False) for k in (
            '_mpc_safety_recovery_active','_post_reverse_full_width_recovery_active',
            '_prepass_fallback_recovery_active','_l1_safety_recovery_active',
            '_l1_rejoin_backoff_active','_straight_reentry_active',
            '_parallel_abort_active','_follow_escape_active')):
        return lane, False
    state = getattr(c, '_committed_corridor', None)
    applied = getattr(state, 'lane', None)
    if applied in (0, 2):
        clear, _ = c._l1_rejoin_traffic_is_clear(pose, speed)
        if not clear:
            return applied, False
        from .early_rejoin import prepare
        if not prepare(c, 1):
            return applied, False
    return 1, True
