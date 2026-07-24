"""Tests for stopped-lead overtake and reverse-wait decisions."""

import unittest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    absolute_heading_difference,
    classify_lane_conflicts,
    evaluate_stopped_lead_overtake,
    lane_conflicts_are_clear,
    is_parallel_vehicle,
    ordered_outer_lane_candidates,
    prediction_clears_moving_vehicle,
    relative_longitudinal_distance,
    select_latched_overtake_lane,
    should_release_latched_overtake_lane,
    should_start_prepass_recovery_from_safety,
)


class StoppedVehicleSafetyTest(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "tracked_stopped_lead": True,
            "distance": 5.0,
            "left_is_free": True,
            "right_is_free": False,
            "target_lane_idx": 2,
            "infeasibility_counter": 0,
            "reverse_distance": 3.0,
            "infeasible_cycles": 5,
        }
        values.update(overrides)
        return evaluate_stopped_lead_overtake(**values)

    def test_outer_lane_accepted_enables_forced_overtake(self):
        self.assertEqual(self.evaluate(), (True, False))

    def test_close_lead_without_passing_space_requests_reverse(self):
        self.assertEqual(self.evaluate(
            distance=2.5,
            left_is_free=False,
            target_lane_idx=1,
        ), (False, True))

    def test_close_lead_without_accepted_outer_lane_requests_reverse(self):
        self.assertEqual(self.evaluate(
            distance=2.5,
            target_lane_idx=1,
        ), (False, True))

    def test_repeated_infeasibility_requests_reverse(self):
        self.assertEqual(self.evaluate(
            distance=2.5,
            infeasibility_counter=5,
        ), (False, True))

    def test_moving_or_untracked_vehicle_gets_no_exemption(self):
        self.assertEqual(self.evaluate(
            tracked_stopped_lead=False,
            distance=2.5,
        ), (False, False))


class PrepassSafetyRecoveryTest(unittest.TestCase):
    def should_start(self, **overrides):
        values = {
            "recovery_requested": True,
            "fallback_enabled": True,
            "applied_lane_idx": 0,
            "latched_vehicle_id": "vehicle-a",
            "opponent_ahead": True,
            "fallback_recovery_active": False,
            "fallback_follow_active": False,
        }
        values.update(overrides)
        return should_start_prepass_recovery_from_safety(**values)

    def test_outer_lane_failure_starts_prepass_recovery(self):
        self.assertTrue(self.should_start())
        self.assertTrue(self.should_start(applied_lane_idx=2))

    def test_full_width_or_l1_failure_is_not_attributed_to_outer_lane(self):
        self.assertFalse(self.should_start(applied_lane_idx=None))
        self.assertFalse(self.should_start(applied_lane_idx=1))

    def test_missing_or_cleared_target_does_not_start_prepass_recovery(self):
        self.assertFalse(self.should_start(latched_vehicle_id=None))
        self.assertFalse(self.should_start(opponent_ahead=False))

    def test_existing_fallback_state_is_not_restarted(self):
        self.assertFalse(self.should_start(fallback_recovery_active=True))
        self.assertFalse(self.should_start(fallback_follow_active=True))

    def test_failed_lane_is_excluded_from_retry_candidates(self):
        self.assertEqual(ordered_outer_lane_candidates(0, 0), (2,))
        self.assertEqual(ordered_outer_lane_candidates(2, 2), (0,))

    def test_normal_retry_still_checks_opposite_lane_first(self):
        self.assertEqual(ordered_outer_lane_candidates(0), (2, 0))
        self.assertEqual(ordered_outer_lane_candidates(2), (0, 2))


class OvertakeGeometryTest(unittest.TestCase):
    def parallel(self, **overrides):
        values = {
            "ego_lane_idx": 0,
            "other_lane_idx": 1,
            "lateral_distance": 2.2,
            "longitudinal_distance": 1.0,
            "minimum_lateral_distance": 2.0,
            "maximum_lateral_distance": 3.5,
            "maximum_longitudinal_distance": 4.5,
        }
        values.update(overrides)
        return is_parallel_vehicle(**values)

    def test_parallel_requires_different_lanes(self):
        self.assertFalse(self.parallel(other_lane_idx=0))

    def test_parallel_requires_at_least_two_metres_lateral_gap(self):
        self.assertFalse(self.parallel(lateral_distance=1.99))
        self.assertTrue(self.parallel(lateral_distance=2.0))

    def test_parallel_requires_small_longitudinal_gap(self):
        self.assertFalse(self.parallel(longitudinal_distance=4.51))

    def test_parallel_rejects_unknown_lane(self):
        self.assertFalse(self.parallel(ego_lane_idx=None))

    def test_prediction_must_clear_moving_target_at_every_step(self):
        self.assertFalse(prediction_clears_moving_vehicle(
            [0.0, 2.0, 4.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            vehicle_x=2.0,
            vehicle_y=0.5,
            vehicle_vx=0.0,
            vehicle_vy=0.0,
            minimum_clearance=1.3,
        ))

    def test_prediction_can_bypass_brake_when_all_steps_are_clear(self):
        self.assertTrue(prediction_clears_moving_vehicle(
            [0.0, 2.0, 4.0],
            [2.0, 2.0, 2.0],
            [0.0, 1.0, 2.0],
            vehicle_x=2.0,
            vehicle_y=0.0,
            vehicle_vx=0.0,
            vehicle_vy=0.0,
            minimum_clearance=1.3,
        ))

    def test_empty_prediction_never_bypasses_brake(self):
        self.assertFalse(prediction_clears_moving_vehicle(
            [], [], [],
            vehicle_x=0.0,
            vehicle_y=0.0,
            vehicle_vx=0.0,
            vehicle_vy=0.0,
            minimum_clearance=1.3,
        ))

    def test_lane_conflicts_cover_front_side_and_rear(self):
        conflicts = classify_lane_conflicts(
            2,
            [("front", 2, 6.0), ("side", 2, 1.0), ("rear", 2, -4.0),
             ("other-lane", 0, 0.0)],
            front_distance=8.0,
            side_distance=2.0,
            rear_distance=5.0,
        )
        self.assertEqual(conflicts, {
            "front": ["front"], "side": ["side"], "rear": ["rear"]})
        self.assertFalse(lane_conflicts_are_clear(conflicts))

    def test_lane_is_clear_when_only_other_lanes_are_occupied(self):
        conflicts = classify_lane_conflicts(
            1,
            [("left", 2, 1.0), ("right", 0, -1.0)],
            front_distance=8.0,
            side_distance=2.0,
            rear_distance=5.0,
        )
        self.assertTrue(lane_conflicts_are_clear(conflicts))

    def test_vehicle_ahead_has_positive_longitudinal_distance(self):
        self.assertGreater(relative_longitudinal_distance(5.0, 0.0, 0.0), 0.0)

    def test_vehicle_behind_has_negative_longitudinal_distance(self):
        self.assertLess(relative_longitudinal_distance(-5.0, 0.0, 0.0), 0.0)

    def test_longitudinal_projection_respects_ego_heading(self):
        self.assertLess(
            relative_longitudinal_distance(0.0, -5.0, 1.5707963267948966),
            0.0,
        )

    def test_heading_difference_wraps_at_pi(self):
        difference = absolute_heading_difference(
            3.12413936106985,
            -3.12413936106985,
        )
        self.assertAlmostEqual(difference, 0.034906585039886195)

    def test_latched_passing_side_ignores_opposite_candidate(self):
        decision = select_latched_overtake_lane(
            True,
            "vehicle-b",
            0,
            "vehicle-a",
            2,
        )
        self.assertEqual(decision, (2, "vehicle-a", 2, False))

    def test_new_passing_side_is_latched_with_vehicle_id(self):
        decision = select_latched_overtake_lane(
            True,
            "vehicle-a",
            0,
            None,
            None,
        )
        self.assertEqual(decision, (0, "vehicle-a", 0, True))

    def test_latch_is_released_when_overtake_finishes(self):
        decision = select_latched_overtake_lane(
            False,
            "vehicle-b",
            0,
            "vehicle-a",
            2,
        )
        self.assertEqual(decision, (None, None, None, False))

    def test_outer_lane_releases_after_target_is_behind_and_mpc_stalls(self):
        self.assertTrue(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=0,
            target_longitudinal_distance=-1.5,
            infeasibility_counter=8,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))

    def test_outer_lane_does_not_release_while_target_is_ahead(self):
        self.assertFalse(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=2,
            target_longitudinal_distance=0.5,
            infeasibility_counter=20,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))

    def test_outer_lane_does_not_release_for_transient_infeasibility(self):
        self.assertFalse(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=0,
            target_longitudinal_distance=-2.0,
            infeasibility_counter=7,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))

    def test_center_lane_is_never_released_by_outer_lane_escape(self):
        self.assertFalse(should_release_latched_overtake_lane(
            overtake_active=True,
            latched_vehicle_id="vehicle-a",
            latched_lane_idx=1,
            target_longitudinal_distance=-2.0,
            infeasibility_counter=8,
            behind_distance=1.0,
            infeasible_cycles=8,
        ))


if __name__ == "__main__":
    unittest.main()
