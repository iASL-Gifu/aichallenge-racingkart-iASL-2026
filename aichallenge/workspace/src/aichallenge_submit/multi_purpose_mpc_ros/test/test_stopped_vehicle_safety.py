"""Tests for stopped-lead overtake and reverse-wait decisions."""

import unittest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    evaluate_stopped_lead_overtake,
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


if __name__ == "__main__":
    unittest.main()
