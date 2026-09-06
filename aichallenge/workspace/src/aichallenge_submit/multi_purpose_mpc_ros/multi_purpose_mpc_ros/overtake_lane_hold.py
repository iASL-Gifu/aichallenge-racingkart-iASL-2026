"""Continuation of an already selected passing lane, never entry permission."""
from dataclasses import dataclass
import math


def dynamic_longitudinal_conflict_unsafe(
    *, longitudinal, ego_speed, other_speed, body_length_sum,
    desired_gap, reaction_sec, available_deceleration,
):
    """Check signed front/rear gap and closing speed along the course.

    Unknown inputs cannot establish safety. Other speed may be negative for
    oncoming traffic; callers must not turn an invalid velocity into zero.
    """
    values = (longitudinal, ego_speed, other_speed, body_length_sum,
              desired_gap, reaction_sec, available_deceleration)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return True
    body_gap = abs(longitudinal) - max(body_length_sum, 0.0)
    if body_gap <= 0.0:
        return True
    closing_speed = max(ego_speed - other_speed if longitudinal > 0.0
                        else other_speed - ego_speed, 0.0)
    if closing_speed <= 0.0:
        return False
    required = (max(desired_gap, 0.0) + max(reaction_sec, 0.0) * closing_speed
                + closing_speed ** 2 / (2.0 * max(available_deceleration, 0.1)))
    return body_gap <= required


@dataclass(frozen=True)
class LaneHoldDecision:
    unsafe: bool
    physical_passage_lost: bool
    waiting: bool
    locked: bool
    reasons: tuple
    motion_blocked: bool = False


@dataclass
class PassageHold:
    key: object = None
    since: object = None
    unknown_since: object = None

    def reset(self):
        self.key = self.since = self.unknown_since = None

    def evaluate(self, *, key, now_sec, lane_width_valid, target_geometry_known,
                 target_clearance_lost, body_overlap, hybrid_matches, hybrid_progress,
                 alongside, unrelated_unsafe, confirm_sec, lock_ratio,
                 legacy_target_hold=False):
        if self.key != key:
            self.reset()
            self.key = key
        # Missing evidence pauses longitudinal motion before changing sides.
        # A real overlap, width loss or unrelated traffic still acts immediately.
        geometry_missing = not target_geometry_known or body_overlap is None
        if (geometry_missing and lane_width_valid and not unrelated_unsafe
                and body_overlap is not True and math.isfinite(now_sec)):
            if self.unknown_since is None or now_sec < self.unknown_since:
                self.unknown_since = now_sec
            if now_sec-self.unknown_since < max(confirm_sec,0.0)-1e-9:
                return LaneHoldDecision(False,False,True,False,
                                        ('target_geometry_pending',),True)
        else:
            self.unknown_since = None
        reasons = []
        if not lane_width_valid:
            reasons.append('lane_width_lost')
        if not target_geometry_known or body_overlap is None:
            reasons.append('target_geometry_unknown')
        elif body_overlap:
            reasons.append('body_overlap')
        if unrelated_unsafe:
            reasons.append('unrelated_traffic_unsafe_or_unknown')
        if not math.isfinite(now_sec):
            reasons.append('invalid_time')
        if reasons:
            self.since = None
            return LaneHoldDecision(True, not lane_width_valid or not target_geometry_known,
                                    False, False, tuple(reasons))
        if not target_clearance_lost:
            self.since = None
            return LaneHoldDecision(False, False, False, False, ())
        if self.since is None or now_sec < self.since:
            self.since = now_sec
        locked = bool(alongside or legacy_target_hold or (
            hybrid_matches and math.isfinite(hybrid_progress)
            and hybrid_progress >= lock_ratio))
        confirmed = now_sec - self.since >= max(confirm_sec, 0.0) - 1e-9
        lost = bool(confirmed and not locked)
        return LaneHoldDecision(lost, lost, not confirmed and not locked, locked,
                                ('target_clearance_lost',) if lost else ())
