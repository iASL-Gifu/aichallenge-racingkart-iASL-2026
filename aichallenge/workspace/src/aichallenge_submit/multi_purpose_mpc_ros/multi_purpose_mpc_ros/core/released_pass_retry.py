"""Re-arm a released pass on changed observations, without authorizing a lane."""
from dataclasses import dataclass, field
import math


@dataclass
class ReleasedPassRetry:
    target: object = None
    released_at: float = 0.
    peak_speed: float = 0.
    blocked_lanes: set = field(default_factory=set)
    pending: object = None
    since: float = 0.

    def observe(self, *, target, now, speed, clear_lanes, candidate_lane, ready,
                slow_speed, cooldown=2., confirm=.5, speed_drop=1.):
        if not all(math.isfinite(v) for v in (now, speed, slow_speed)) or speed < 0.:
            self.pending = None
            return None
        clear = set(clear_lanes) & {0, 2}
        if self.target != target or now < self.released_at:
            self.target, self.released_at = target, now
            self.peak_speed = speed
            self.blocked_lanes = {0, 2}-clear
            self.pending = None
            return None
        self.peak_speed = max(self.peak_speed, speed)
        self.blocked_lanes.update({0, 2}-clear)
        slower = speed <= slow_speed and self.peak_speed-speed >= speed_drop
        newly_clear = candidate_lane in self.blocked_lanes and candidate_lane in clear
        reason = 'lead_slowed' if slower else 'passage_improved' if newly_clear else None
        if (not ready or candidate_lane not in clear or reason is None
                or now-self.released_at < max(cooldown, 0.)):
            self.pending = None
            return None
        key = reason, candidate_lane
        if self.pending != key or now < self.since:
            self.pending, self.since = key, now
        return reason if now-self.since >= max(confirm, 0.)-1e-9 else None

