"""Bounded advisory scheduling; cached proposals never authorize motion."""


def passage_key(controller, target_id, pose):
    """Exact inputs, including mutable Center bounds, for one-cycle reuse."""
    path = controller._reference_pathN_center
    return (
        target_id, tuple(controller._v2x_tracker._samples.get(target_id, ())),
        tuple(controller._v2x_tracker.velocity(target_id)), pose.x, pose.y,
        controller._carN_center.wp_id, controller._mpcN_center.N,
        controller._cfg.bicycle_model.width, controller._v2x_vehicle_radius,
        controller._passage_clearance, controller._prepass_lane_fallback_prediction_sec,
        tuple(controller._v2x_t_samples), controller._passage_lane_width_tolerance,
        controller._passage_lane_width_tolerance_points,
        id(path), getattr(path, 'n_lanes', 2), getattr(path, 'inner_lane_width', .5),
        tuple((wp.x, wp.y, wp.psi, wp.lb, wp.ub) for wp in path.waypoints),
    )


class TrafficWork:
    def __init__(self):
        self.passages = {}
        self.relative_samples = {}
        self.relative_hits = 0
        self.proposal = None
        self.proposal_key = None
        self.proposal_at = None
        self.probe_cycle = set()
        self.probe_attempts = {}
        self.hits = 0
        self.proposal_hits = 0
        self.probe_skips = 0

    def begin_cycle(self):
        self.passages.clear()
        self.relative_samples.clear()
        self.probe_cycle.clear()

    def held_proposal(self, key, now, interval=.1):
        if (self.proposal_key == key and self.proposal_at is not None
                and 0 <= now - self.proposal_at < interval):
            self.proposal_hits += 1
            return True, self.proposal
        return False, None

    def remember_proposal(self, key, now, value):
        self.proposal_key, self.proposal_at, self.proposal = key, now, value

    def claim_probe(self, name):
        # One solve per purpose per control cycle; a skipped solve cannot add
        # a confirmation cycle or extend proof freshness.
        if name in self.probe_cycle:
            self.probe_skips += 1
            return False
        self.probe_cycle.add(name)
        return True

    def probe_due(self, key, now):
        previous = self.probe_attempts.get(key)
        if previous is not None:
            stamp, success = previous
            if not success and 0 <= now - stamp < .1:
                self.probe_skips += 1
                return False
        return self.claim_probe(key)

    def reset_probe(self, purpose):
        self.probe_attempts = {k: v for k, v in self.probe_attempts.items()
                               if k[0] != purpose}

    def record_probe(self, key, now, success):
        # Keep only the current candidate for each purpose. Returning to a
        # previous candidate must not inherit its retry suppression.
        purpose = key[0]
        self.probe_attempts = {k: v for k, v in self.probe_attempts.items()
                               if k[0] != purpose}
        self.probe_attempts[key] = (now, bool(success))


def relative_samples_key(controller, pose, ego_speed):
    path = controller._reference_path
    tracker = controller._v2x_tracker
    return (
        pose.x, pose.y, pose.theta, ego_speed,
        controller._prepass_lane_fallback_prediction_sec,
        id(controller._center_arc_points), id(controller._center_arc_cumulative),
        controller._center_arc_total_length,
        id(path), id(controller._car),
        getattr(path, 'n_lanes', 2), getattr(path, 'inner_lane_width', .5),
        tuple((wp.x, wp.y, wp.psi, wp.lb, wp.ub) for wp in path.waypoints),
        tuple((vid, tuple(tracker._samples.get(vid, ())),
               tracker.has_velocity_estimate(vid), tuple(tracker.velocity(vid)))
              for vid in tracker.active_vehicle_ids()),
    )
