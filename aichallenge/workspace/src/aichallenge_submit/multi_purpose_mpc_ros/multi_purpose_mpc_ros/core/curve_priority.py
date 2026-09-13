"""One Center-WP lane preference shared by acquisition and final ownership."""
import math
from .l0_preparation import l2_exception


def priority_at(wp, zones, lengths, preview, prepare_distances=None):
    count = len(lengths)
    if not count:
        return None
    wp = int(wp) % count
    # The current curve wins over preparation for the following curve.
    for start, end, lane in zones:
        inside = start <= wp <= end if start <= end else wp >= start or wp <= end
        if inside:
            return lane
    prepare_distances = prepare_distances or {}
    distances = {zone: float(prepare_distances.get(zone, preview)) for zone in zones}
    if any(not math.isfinite(value) or value < 0. for value in distances.values()):
        return None
    maximum = max(distances.values(), default=0.)
    if maximum <= 0.:
        return None
    distance = 0.
    index = wp
    for _ in range(count):
        length = float(lengths[index])
        if not math.isfinite(length) or length <= 0:
            return None
        distance += length
        if distance > maximum:
            return None
        index = (index+1) % count
        for start, end, lane in zones:
            if index == start:
                # Do not prepare a farther curve across the nearer one.
                return lane if distance <= distances[(start,end,lane)] else None
    return None


def controller_priority(c, wp):
    zones = getattr(c, '_curve_lane_priority_zones', ())
    if not zones:
        return None
    return priority_at(wp, zones, c._reference_pathN_center.segment_lengths,
                       c._curve_lane_priority_prepare_distance,
                       getattr(c, '_curve_lane_priority_prepare_distances', None))


def select(c, preferred, candidate, target, passage, conflicts):
    active = transition_owner(c)
    if active is not None:
        return active
    if candidate in (0, 2) and candidate == recovery_lane_owner(c):
        return candidate
    opposite = 2-preferred
    if l2_exception(target, c._l2_restricted_slow_override(target),
                    passage.get(opposite, False), conflicts.get(opposite, {})):
        return candidate
    traffic = conflicts.get(preferred, {})
    # Rank rear-only traffic as in the existing end-of-lap policy. This is
    # only a request: Shadow/final swept-body validation remains mandatory.
    if traffic.get('side'):
        return 1
    if (passage.get(preferred, False)
            or (target is not None and c._lane_horizon_has_vehicle_width(preferred))):
        return preferred
    return 1


def recovery_lane_owner(c):
    """Preference must not overwrite Prepass; safety/release retain authority.

    Reuse Prepass's lifetime instead of creating a second lane latch. Its lane
    remains recorded after successful commit and is cleared by existing exit,
    failure and target-release handling.
    """
    if getattr(c, '_prepass_fallback_commit_pending', False):
        lane = getattr(c, '_prepass_fallback_commit_lane_idx', None)
    else:
        lane = getattr(c, '_prepass_fallback_lane_idx', None)
    return lane if lane in (0, 2) else None


def retained_priority(c):
    """Suppress ordinary rejoin only for the currently applied inner lane.

    No geometry verdict is manufactured here. Live MPC/body checks and all
    recovery owners retain authority to discard an unsafe applied corridor.
    """
    from .follow_zone import active
    if active(c):
        return None
    path = getattr(c, '_reference_path', None)
    center = getattr(c, '_reference_pathN_center', None)
    if path is None or center is None or path is not center:
        return None
    if any(getattr(c, name, False) for name in (
            '_mpc_safety_recovery_active', '_post_reverse_full_width_recovery_active',
            '_prepass_fallback_recovery_active', '_prepass_fallback_commit_pending',
            '_prepass_fallback_follow_active', '_prepass_fallback_blocked',
            '_close_obstacle_reverse_requested', '_center_lane_rejoin_constraint_released',
            '_follow_escape_active', '_follow_only',
            '_l1_safety_recovery_active', '_l1_safety_reprobe_pending',
            '_l1_rejoin_backoff_active', '_parallel_abort_active',
            '_straight_reentry_active')):
        return None
    car = getattr(c, '_carN_center', None)
    if car is None:
        return None
    wp = int(car.wp_id)
    # Preparation may rank/probe an upcoming inner lane, but only membership
    # in the actual zone suppresses ordinary rejoin and progress timeouts.
    preferred = None
    for start, end, lane in getattr(c, '_curve_lane_priority_zones', ()):
        if (start <= wp <= end) if start <= end else (wp >= start or wp <= end):
            preferred = lane
            break
    if preferred is None:
        for start, end in getattr(c, '_l2_entry_restricted_zones', ()):
            if (start <= wp <= end) if start <= end else (wp >= start or wp <= end):
                preferred = 0
                break
    if preferred is None or getattr(path, 'target_lane_idx', None) != preferred:
        return None
    return preferred


def cancel_ordinary_rejoin(c):
    """Cancel only voluntary rejoin state; never clear recovery/fallback probes."""
    c._trajectory_clear_since = None
    c._center_lane_rejoin_stable_since = None
    c._center_lane_rejoin_active = False
    if getattr(c, '_race_rejoin_handoff_active', False):
        c._reset_race_rejoin_handoff()
    if getattr(c, '_l1_probe_context', None) == 'rejoin':
        c._l1_probe_active = False
        c._l1_probe_context = None
        c._l1_probe_success_cycles = 0
        c._l1_probe_constraint_applied = False


def transition_owner(c):
    """A geographic preference cannot reverse an admitted unfinished movement.

    This only guards preference selection. Passage/traffic checks, transition
    timeout and recovery still release an unsafe or unsuccessful manoeuvre.
    """
    session = getattr(c, '_overtake', None)
    hybrid = getattr(session, 'hybrid', None)
    state = getattr(c, '_committed_corridor', None)
    if hybrid is None or state is None:
        return None
    lane = getattr(hybrid, 'lane_idx', None)
    if (lane not in (0, 2) or getattr(hybrid, 'completed', False)
            or not getattr(hybrid, 'length', 0.)
            or getattr(hybrid, 'vehicle_id', None) != getattr(session, 'target_id', None)
            or getattr(state, 'lane', None) != lane
            or getattr(state, 'path', None) is not getattr(c, '_reference_path', None)
            or any(getattr(c, k, False) for k in (
                '_mpc_safety_recovery_active', '_prepass_fallback_recovery_active',
                '_post_reverse_full_width_recovery_active', '_parallel_abort_active',
                '_follow_escape_active', '_straight_reentry_active'))):
        return None
    return lane
