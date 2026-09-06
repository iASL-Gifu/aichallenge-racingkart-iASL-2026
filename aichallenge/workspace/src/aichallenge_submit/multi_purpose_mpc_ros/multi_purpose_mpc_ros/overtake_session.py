"""Overtake data and the single final corridor decision, independent of ROS."""

from dataclasses import dataclass, field
from typing import Optional

from .v2x_vehicle_tracker import resolve_applied_corridor
from .overtake_lane_hold import PassageHold


@dataclass
class HybridTransition:
    vehicle_id: Optional[str] = None
    lane_idx: Optional[int] = None
    start_wp: Optional[int] = None
    started_at: Optional[float] = None
    start_e_y: Optional[float] = None
    length: Optional[float] = None
    last_wp: Optional[int] = None
    travelled: float = 0.0
    paused: bool = False
    completed: bool = False
    # Historical admission of this spatial transition, NOT live safety proof.
    verified_start: bool = False


@dataclass
class ShadowProbe:
    vehicle_id: Optional[str] = None
    lane_idx: Optional[int] = None
    success_cycles: int = 0
    confirmed: bool = False
    confirmed_at: Optional[float] = None


@dataclass
class ShadowVerification:
    vehicle_id: Optional[str] = None
    lane_idx: Optional[int] = None


@dataclass(frozen=True)
class LaneDecision:
    target_id: Optional[str]
    requested_lane: Optional[int]
    applied_lane: Optional[int]
    mode: str


@dataclass
class OvertakeSession:
    # Evaluators may change the request. Only apply() changes the accepted key.
    target_id: Optional[str] = None
    requested_lane: Optional[int] = None
    committed: bool = False
    hybrid: HybridTransition = field(default_factory=HybridTransition)
    probe: ShadowProbe = field(default_factory=ShadowProbe)
    verification: ShadowVerification = field(default_factory=ShadowVerification)
    accepted_key: tuple = (None, None)
    passage_hold: PassageHold = field(default_factory=PassageHold)
    traffic_key: tuple = (None, None)
    traffic_relevant_ids: set = field(default_factory=set)

    def clear_proof(self):
        """Discard executable evidence without losing the spatial manoeuvre."""
        self.clear_probe()
        self.clear_verification()
        self.passage_hold.reset()
        self.traffic_key = (None, None)
        self.traffic_relevant_ids.clear()

    def release_target(self):
        """Target change/loss: no evidence or spatial anchor may be inherited."""
        self.target_id = self.requested_lane = None
        self.committed = False
        self.clear_hybrid()
        self.clear_proof()
        self.accepted_key = (None, None)

    def complete_pass(self):
        """Retain target/side and geometry for the existing return-to-Race gates."""
        self.committed = False
        self.hybrid.verified_start = False
        self.hybrid.completed = True
        self.clear_proof()

    def recommit_lane(self, lane):
        """Same target, new side: restart geometry and verification together."""
        if lane not in (0, 2):
            raise ValueError("an opposite-side recommit requires an outer lane")
        self.clear_hybrid()
        self.clear_proof()
        self.requested_lane = lane

    def clear_hybrid(self):
        self.hybrid = HybridTransition()
        self.passage_hold.reset()
        self.traffic_key = (None, None)
        self.traffic_relevant_ids.clear()

    def start_hybrid(self, *, vehicle_id, lane_idx, start_wp, started_at,
                     start_e_y, length):
        self.hybrid = HybridTransition(
            vehicle_id=vehicle_id, lane_idx=lane_idx, start_wp=start_wp,
            started_at=started_at, start_e_y=start_e_y, length=length,
            last_wp=start_wp,
            verified_start=bool(
                vehicle_id is not None and lane_idx in (0, 2)
                and (vehicle_id, lane_idx) == (self.target_id, self.requested_lane)
                and (self.committed or
                     (self.verification.vehicle_id, self.verification.lane_idx)
                     == (vehicle_id, lane_idx))),
        )

    def clear_probe(self):
        self.probe = ShadowProbe()

    def clear_verification(self):
        self.verification = ShadowVerification()

    def can_resume_hybrid(self, lane):
        return bool(
            (self.committed or self.hybrid.verified_start)
            and self.target_id is not None
            and lane in (0, 2) and lane == self.requested_lane
            and (self.hybrid.vehicle_id, self.hybrid.lane_idx)
                == (self.target_id, lane)
            and self.hybrid.start_wp is not None
            and not self.hybrid.completed)

    def apply(self, decision: LaneDecision, *, preserve_manoeuvre=False) -> bool:
        """Atomically accept identity and invalidate data for another pass.

        Full-width solver recovery pauses the same request; it does not create
        a different pass. A different target or requested side does.
        """
        temporary_pause = bool(
            preserve_manoeuvre
            and decision.requested_lane in (None, self.requested_lane)
            and decision.applied_lane is None
            and decision.target_id == self.target_id
            and self.can_resume_hybrid(self.requested_lane))
        if temporary_pause or decision.mode == "full_width_recovery":
            # Keep only geometry/identity, never old executable clearance proof.
            self.clear_proof()
        lane = (self.requested_lane if temporary_pause
                else decision.requested_lane)
        key = (decision.target_id, lane)
        changed = key != self.accepted_key
        if changed:
            if (self.hybrid.vehicle_id, self.hybrid.lane_idx) != key:
                self.clear_hybrid()
            if (self.verification.vehicle_id, self.verification.lane_idx) != key:
                self.clear_verification()
            if (self.probe.vehicle_id, self.probe.lane_idx) == self.accepted_key:
                self.clear_probe()
        if decision.applied_lane in (0, 2) and decision.target_id is not None:
            self.requested_lane = decision.applied_lane
        self.accepted_key = key
        return changed


def decide_lane(*, target_id, requested_lane, l0_prohibited,
                full_width_recovery, transition_active, hybrid_active):
    """Resolve all corridor overrides before any MPC boundary is mutated."""
    lane, mode = resolve_applied_corridor(
        requested_lane=requested_lane, l0_prohibited=l0_prohibited,
        full_width_recovery=full_width_recovery,
        transition_active=transition_active,
    )
    # Hybrid can replace an ordinary transition, never a geographic/recovery
    # decision. This used to run after the resolver and could override L0 bans.
    if (hybrid_active and requested_lane in (0, 2)
            and mode in ("lane_transition", "requested_lane")):
        lane, mode = requested_lane, "hybrid_lane_transition"
    if mode == "l0_prohibited":
        requested_lane = lane
    return LaneDecision(target_id, requested_lane, lane, mode)
