import unittest

import numpy as np

from multi_purpose_mpc_ros.core.MPC import (
    build_arc_length_steering_reservation,
)


class SteeringReservationTest(unittest.TestCase):
    def test_ramp_starts_before_corner(self):
        distances = np.arange(0.0, 13.0, 1.0)
        references = np.zeros_like(distances)
        references[8:] = 0.30
        speeds = np.full_like(distances, 10.0)
        reserved = build_arc_length_steering_reservation(
            distances, references, speeds, 0.15, 0.60, 0.0, 0.025)
        self.assertGreater(reserved[3], 0.0)
        self.assertLessEqual(np.max(np.abs(np.diff(reserved))), 0.061)

    def test_first_command_obeys_physical_rate(self):
        distances = np.arange(0.0, 6.0, 1.0)
        references = np.full_like(distances, 0.30)
        speeds = np.full_like(distances, 10.0)
        reserved = build_arc_length_steering_reservation(
            distances, references, speeds, 0.15, 0.60, 0.0, 0.025)
        self.assertLessEqual(abs(reserved[0]), 0.60 * 0.025 + 1e-12)

    def test_delay_advances_future_demand(self):
        distances = np.arange(0.0, 11.0, 1.0)
        references = np.zeros_like(distances)
        references[6:] = 0.30
        speeds = np.full_like(distances, 10.0)
        without_delay = build_arc_length_steering_reservation(
            distances, references, speeds, 0.0, 0.60, 0.0, 0.025)
        with_delay = build_arc_length_steering_reservation(
            distances, references, speeds, 0.15, 0.60, 0.0, 0.025)
        self.assertGreaterEqual(with_delay[3], without_delay[3])


if __name__ == '__main__':
    unittest.main()
