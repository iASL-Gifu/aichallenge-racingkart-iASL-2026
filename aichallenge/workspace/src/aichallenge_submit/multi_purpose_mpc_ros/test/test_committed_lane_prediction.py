"""Tests for predicted V2X envelopes on a committed outer lane."""

import unittest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    exclude_envelope_conflicting_lanes,
    minimum_predicted_envelope_conflict,
)


def _conflict(vehicles, prediction_y=2.0):
    return minimum_predicted_envelope_conflict(
        [0.0, 1.0, 2.0], [prediction_y] * 3, [0.0, 1.0, 2.0], vehicles,
        project_frenet=lambda x, y: (float(x), float(y)),
        arc_total_length=1000.0, ego_width=1.6, other_half_width=0.725,
        ego_half_length=1.0, other_half_length=1.0)


class CommittedLanePredictionTest(unittest.TestCase):
    def _envelope_with_candidate_cache(self, vehicles, cache_enabled):
        timing = {}
        projection_calls = []

        def project_frenet(x, y):
            projection_calls.append((float(x), float(y)))
            return float(x), float(y)

        result = minimum_predicted_envelope_conflict(
            [0.0, 1.0, 2.0], [2.0, 2.0, 2.0], [0.0, 1.0, 2.0],
            vehicles,
            project_frenet=project_frenet,
            arc_total_length=1000.0,
            ego_width=1.6,
            other_half_width=0.725,
            ego_half_length=1.0,
            other_half_length=1.0,
            timing=timing,
            candidate_frenet_cache_enabled=cache_enabled,
        )
        return result, timing, projection_calls

    def test_candidate_frenet_cache_preserves_conflict_result(self):
        vehicles = [
            ("clear", 30.0, -3.0, 0.0, 0.0),
            ("blocking", 1.0, 2.0, 0.0, 0.0),
        ]

        cached, timing, cached_calls = self._envelope_with_candidate_cache(
            vehicles, True)
        uncached, _, uncached_calls = self._envelope_with_candidate_cache(
            vehicles, False)

        self.assertEqual(cached, uncached)
        self.assertEqual(cached["vehicle_id"], "blocking")
        self.assertEqual(timing["candidate_frenet_cache_misses"], 3)
        self.assertEqual(timing["candidate_frenet_cache_hits"], 3)
        self.assertEqual(timing["center_frenet_call_count"], 9)
        self.assertEqual(len(cached_calls), 9)
        self.assertEqual(len(uncached_calls), 12)

    def test_candidate_frenet_cache_preserves_clear_result(self):
        vehicles = [
            ("clear_a", 30.0, -3.0, 0.0, 0.0),
            ("clear_b", 40.0, 4.0, 0.0, 0.0),
        ]

        cached, timing, _ = self._envelope_with_candidate_cache(
            vehicles, True)
        uncached, _, _ = self._envelope_with_candidate_cache(
            vehicles, False)

        self.assertIsNone(cached)
        self.assertEqual(cached, uncached)
        self.assertEqual(timing["candidate_frenet_cache_misses"], 3)
        self.assertEqual(timing["candidate_frenet_cache_hits"], 3)

    def test_external_candidate_frenet_cache_preserves_all_results(self):
        cases = (
            [
                ("clear", 30.0, -3.0, 0.0, 0.0),
                ("blocking", 1.0, 2.0, 0.0, 0.0),
            ],
            [
                ("clear_a", 30.0, -3.0, 0.0, 0.0),
                ("clear_b", 40.0, 4.0, 0.0, 0.0),
            ],
        )
        for vehicles in cases:
            with self.subTest(vehicles=vehicles):
                uncached, _, _ = self._envelope_with_candidate_cache(
                    vehicles, True)
                timing = {}
                projection_calls = []

                def project_frenet(x, y):
                    projection_calls.append((float(x), float(y)))
                    return float(x), float(y)

                external_cache = {
                    (0.0, 2.0): (0.0, 2.0),
                    (1.0, 2.0): (1.0, 2.0),
                    (2.0, 2.0): (2.0, 2.0),
                }
                cached = minimum_predicted_envelope_conflict(
                    [0.0, 1.0, 2.0], [2.0, 2.0, 2.0],
                    [0.0, 1.0, 2.0], vehicles,
                    project_frenet=project_frenet,
                    arc_total_length=1000.0,
                    ego_width=1.6,
                    other_half_width=0.725,
                    ego_half_length=1.0,
                    other_half_length=1.0,
                    timing=timing,
                    external_candidate_frenet_cache=external_cache,
                )

                self.assertEqual(cached, uncached)
                self.assertEqual(
                    timing["external_candidate_frenet_cache_hits"], 3)
                self.assertEqual(
                    timing["external_candidate_frenet_cache_misses"], 0)
                self.assertEqual(timing["candidate_frenet_cache_hits"], 3)
                self.assertEqual(timing["candidate_frenet_cache_misses"], 0)
                self.assertEqual(timing["center_frenet_call_count"], 6)
                self.assertEqual(len(projection_calls), 6)

    def test_candidate_gate_excludes_only_conflicting_outer_lane(self):
        blocker = {"vehicle_id": "d2"}
        self.assertEqual(
            exclude_envelope_conflicting_lanes(
                [0, 2], {0: blocker, 2: None}),
            [2],
        )
        self.assertEqual(
            exclude_envelope_conflicting_lanes(
                [0, 2], {0: None, 2: blocker}),
            [0],
        )

    def test_candidate_gate_keeps_safe_and_rejects_both_unsafe(self):
        blocker = {"vehicle_id": "other"}
        self.assertEqual(
            exclude_envelope_conflicting_lanes(
                [0, 2], {0: None, 2: None}),
            [0, 2],
        )
        self.assertEqual(
            exclude_envelope_conflicting_lanes(
                [0, 2], {0: blocker, 2: blocker}),
            [],
        )

    def test_current_envelope_overlap_is_conflict_at_initial_sample(self):
        conflict = _conflict([("other", 0.0, 2.0, 0.0, 0.0)])
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["vehicle_id"], "other")
        self.assertEqual(conflict["prediction_step"], 0)

    def test_longitudinal_overlap_with_lateral_separation_is_safe(self):
        self.assertIsNone(_conflict([
            ("d3", 0.0, -2.0, 0.0, 0.0),
        ]))

    def test_distant_adjacent_vehicle_keeps_committed_lane(self):
        self.assertIsNone(_conflict([("d3", 2.0, -3.0, 0.0, 0.0)]))

    def test_currently_clear_lateral_mover_conflicts_before_overlap(self):
        conflict = _conflict([("d3", 2.0, 0.0, 0.0, 1.0)])
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["vehicle_id"], "d3")
        self.assertLessEqual(conflict["predicted_lateral_clearance"], 0.0)
        self.assertLessEqual(
            conflict["predicted_longitudinal_clearance"], 0.0)

    def test_non_target_active_vehicle_is_checked(self):
        conflict = _conflict([
            ("target", 20.0, -3.0, 0.0, 0.0),
            ("other", 2.0, 0.0, 0.0, 1.0),
        ])
        self.assertEqual(conflict["vehicle_id"], "other")

    def test_l0_is_symmetric(self):
        conflict = _conflict(
            [("d3", 2.0, 0.0, 0.0, -1.0)], prediction_y=-2.0)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["vehicle_id"], "d3")

    def test_longitudinally_separated_vehicle_does_not_block(self):
        self.assertIsNone(_conflict([
            ("d3", 20.0, 0.0, 0.0, 1.0),
        ]))


if __name__ == "__main__":
    unittest.main()
