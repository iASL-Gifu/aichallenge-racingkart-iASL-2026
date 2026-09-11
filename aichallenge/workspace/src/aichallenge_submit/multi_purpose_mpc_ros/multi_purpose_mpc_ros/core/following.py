"""Shared bumper-gap following model; all distances are body clearances in metres."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FollowingState:
    body_gap: float
    lead_speed: float
    required_gap: float
    speed_limit: float
    target_speed: float


def following_state(*, body_gap, lead_speed, ego_speed, minimum_gap,
                    desired_gap, reaction_sec, deceleration, spacing_kp,
                    maximum_acceleration=2.5):
    """Bound following by reaction travel and an immediately braking lead.

    No future lead acceleration is credited; both vehicles use the configured
    braking deceleration, while ego may keep accelerating during reaction time.
    Below the desired spacing, match
    the moving lead only as far as the braking envelope permits. This avoids
    treating an opening launch gap as a stationary obstacle. The minimum gap
    is a hard reserve, distinct from the comfortable desired gap.
    """
    values = (body_gap, lead_speed, ego_speed, minimum_gap, desired_gap,
              reaction_sec, deceleration, spacing_kp, maximum_acceleration)
    if (not all(math.isfinite(x) for x in values) or minimum_gap < 0
            or desired_gap < minimum_gap or reaction_sec < 0
            or deceleration <= 0 or spacing_kp < 0 or lead_speed < 0
            or maximum_acceleration < 0):
        return FollowingState(body_gap, 0.0, math.inf, 0.0, 0.0)
    reaction_gain = maximum_acceleration * reaction_sec
    reaction_distance = (max(0.0, ego_speed) * reaction_sec
                         + .5 * maximum_acceleration * reaction_sec ** 2)
    required_gap = minimum_gap + max(
        0.0, reaction_distance + ((max(0.0, ego_speed) + reaction_gain) ** 2 - lead_speed ** 2)
        / (2 * deceleration))
    if body_gap <= minimum_gap:
        return FollowingState(body_gap, lead_speed, required_gap, 0.0, 0.0)
    available = body_gap - minimum_gap + lead_speed ** 2 / (2 * deceleration)
    # Charge reaction travel at both the measured and candidate speed. In
    # particular, a fast ego cannot acquire a generous cap by requesting slow.
    measured_cap = max(0.0, math.sqrt(
        2 * deceleration * max(0.0, available - reaction_distance)) - reaction_gain)
    candidate_cap = max(0.0, math.sqrt(
                              deceleration * (deceleration + maximum_acceleration) * reaction_sec ** 2
                              + 2 * deceleration * available)
                     - (deceleration + maximum_acceleration) * reaction_sec)
    speed_limit = min(measured_cap, candidate_cap)
    target = lead_speed + spacing_kp * max(0.0, body_gap - desired_gap)
    return FollowingState(body_gap, lead_speed, required_gap, speed_limit, min(target, speed_limit))
