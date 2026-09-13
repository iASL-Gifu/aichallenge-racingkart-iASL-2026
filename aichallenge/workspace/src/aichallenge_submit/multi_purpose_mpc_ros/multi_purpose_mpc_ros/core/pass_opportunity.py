"""Progress-based admission of a new pass; physical/traffic admission stays separate."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Opportunity:
    allowed: bool
    reason: str
    seconds: float = math.inf
    gain: float = 0.
    ego_speed: float = 0.


def evaluate(distance, lead_speed, available_speed, *, speed_valid=True,
             ultra_slow_speed=1.5, max_seconds=8., min_gain=1.,
             speed_reserve=.5, clearance=3., available_distance=80., current_speed=None, closing_speed=None, acceleration=1.):
    if not speed_valid or not all(math.isfinite(v) for v in
            (distance, lead_speed, available_speed, available_distance)):
        return Opportunity(False, 'unknown_speed_or_distance')
    if lead_speed < 0. or distance <= 0.:
        return Opportunity(False, 'invalid_target')
    if lead_speed <= ultra_slow_speed:
        return Opportunity(True, 'stopped_or_ultra_slow')
    ego = max(0., available_speed-speed_reserve)
    gain = ego-lead_speed
    if gain < min_gain:
        return Opportunity(False, 'insufficient_speed_advantage', gain=gain, ego_speed=ego)
    initial = ego if current_speed is None else current_speed
    if not math.isfinite(initial) or not math.isfinite(acceleration) or acceleration <= 0.:
        return Opportunity(False, 'invalid_motion')
    initial = max(0., min(initial, ego))
    # A measured arc-distance trend can make the estimate more conservative,
    # but never manufacture an advantage beyond the observed lead velocity.
    lead = lead_speed
    if closing_speed is not None:
        if not math.isfinite(closing_speed):
            return Opportunity(False, 'invalid_closing_speed')
        lead = max(lead, initial-closing_speed)
    gain = ego-lead
    if gain < min_gain:
        return Opportunity(False, 'insufficient_speed_advantage', gain=gain, ego_speed=ego)
    ramp = (ego-initial)/acceleration
    ramp_gain = (initial-lead)*ramp + .5*acceleration*ramp*ramp
    required = distance+clearance
    if required <= ramp_gain:
        seconds = (-(initial-lead)+math.sqrt((initial-lead)**2+2*acceleration*required))/acceleration
    else:
        seconds = ramp + (required-ramp_gain)/gain
    accelerating = min(seconds, ramp)
    travel = initial*accelerating + .5*acceleration*accelerating**2 + ego*max(0.,seconds-ramp)
    allowed = seconds <= max_seconds and travel <= available_distance
    return Opportunity(allowed, 'pass_within_window' if allowed else 'pass_too_far',
                       seconds, gain, ego)


def controller_opportunity(c, wp, distance, lead_speed, valid, *, current_speed=None, closing_speed=None):
    # Reference speeds, not the ACC-limited command or matched measured speed.
    # Bound work to one local circuit and the configured assessment distance.
    path = c._reference_pathN_center
    limit = c._mpcN_center.input_constraints['umax'][0]
    span = 0.
    for n in range(path.n_waypoints):
        index = int(wp)+n
        if not path.circular and index >= path.n_waypoints:
            break
        point = path.get_waypoint(index)
        value = point.v_ref
        if value is None or not math.isfinite(float(value)):
            return Opportunity(False, 'missing_reference_speed')
        limit = min(limit, max(0., float(value)))
        if not path.circular and index + 1 >= path.n_waypoints:
            break
        # segment_lengths stores the *incoming* segment and starts with zero.
        # Measure the outgoing segment explicitly, including the circuit seam.
        next_point = path.get_waypoint(index + 1)
        segment = math.hypot(next_point.x-point.x, next_point.y-point.y)
        if not math.isfinite(segment):
            return Opportunity(False, 'invalid_reference_distance')
        span += segment
        if span >= c._pass_opportunity_distance:
            break
    return evaluate(float(distance), float(lead_speed), float(limit), speed_valid=valid,
        ultra_slow_speed=c._ultra_slow_early_commit_speed,
        max_seconds=c._pass_opportunity_max_seconds,
        min_gain=c._pass_opportunity_min_gain, speed_reserve=c._pass_opportunity_speed_reserve,
        clearance=c._pass_opportunity_clearance,
        available_distance=min(span,c._pass_opportunity_distance),
        current_speed=current_speed, closing_speed=closing_speed,
        acceleration=getattr(c, '_pass_opportunity_acceleration', 1.))


def observe_closing(c, target, now, distance, valid):
    """Bounded one-target arc-distance history; called even while Shadow waits."""
    from collections import deque
    history = getattr(c, '_pass_closing_history', None)
    previous_target = getattr(c, '_pass_closing_target', None)
    if (not valid or target is None or not math.isfinite(now)
            or not math.isfinite(distance) or distance <= 0.):
        c._pass_closing_history = None
        c._pass_closing_target = None
        return None
    if (history is None or previous_target != target or
            (history and (now < history[-1][0] or now-history[-1][0] > .5))):
        history = deque(maxlen=80)
    if history and now == history[-1][0]:
        return None
    if history and abs(distance-history[-1][1]) > 5.:
        history.clear()
    history.append((now, distance))
    while len(history) > 1 and now-history[0][0] > 1.:
        history.popleft()
    c._pass_closing_history, c._pass_closing_target = history, target
    elapsed = now-history[0][0]
    return (history[0][1]-distance)/elapsed if elapsed >= .3 else None
