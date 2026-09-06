"""Advisory lane ranking only: never a live corridor or motion permission."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LaneObstruction:
    vehicle_id: str
    distance: float
    blocked_lanes: tuple


def propose_lane(preferred, passage, conflicts, obstructions=(), active_lane=None):
    """Keep an admissible active lane; otherwise prefer the longest clear prefix.

    Primary passage and near-traffic checks remain admission gates. Other
    vehicles farther ahead rank alternatives, but cannot erase all candidates.
    A prefix is a ranking metric, NOT proof that it is safe to drive to its end.
    """
    candidates = [lane for lane in (0, 2) if passage.get(lane, False)
                  and lane in conflicts and not any(
                      conflicts[lane].get(kind, ()) for kind in ('front', 'side', 'rear'))]
    if active_lane in candidates:
        return active_lane
    def score(lane):
        prefix = min((max(0., row.distance) for row in obstructions
                      if lane in row.blocked_lanes), default=float('inf'))
        return prefix, lane == preferred, lane == 0
    return max(candidates, key=score) if candidates else None
