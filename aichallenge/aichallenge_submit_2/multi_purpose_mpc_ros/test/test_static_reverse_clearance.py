"""Tests for static-map swept-footprint reverse clearance."""

import unittest

import numpy as np

from multi_purpose_mpc_ros.core.map import Map


def make_map(width=30, height=20, resolution=1.0):
    occupancy_map = Map.__new__(Map)
    occupancy_map.width = width
    occupancy_map.height = height
    occupancy_map.resolution = resolution
    occupancy_map.origin = [0.0, 0.0, 0.0]
    occupancy_map.data_backup = np.ones((height, width), dtype=np.int8)
    occupancy_map.data = occupancy_map.data_backup.copy()
    return occupancy_map


class StaticReverseClearanceTest(unittest.TestCase):
    def test_disk_outside_map_is_occupied(self):
        occupancy_map = make_map()
        self.assertFalse(occupancy_map.static_disk_is_free(0.1, 10.0, 1.0))

    def test_reverse_clearance_stops_before_static_wall(self):
        occupancy_map = make_map()
        # World y=10 maps to row 9. A vertical wall at world x=5 is behind
        # a vehicle at x=10 with heading zero.
        occupancy_map.data_backup[:, 5] = 0
        clearance = occupancy_map.static_straight_path_clearance(
            x=10.0,
            y=10.0,
            heading=0.0,
            max_distance=8.0,
            footprint_radius=1.0,
            step=0.25,
        )
        self.assertGreaterEqual(clearance, 3.5)
        self.assertLessEqual(clearance, 4.25)

    def test_reverse_clearance_returns_requested_distance_when_free(self):
        occupancy_map = make_map()
        clearance = occupancy_map.static_straight_path_clearance(
            x=15.0,
            y=10.0,
            heading=0.0,
            max_distance=4.5,
            footprint_radius=1.0,
            step=0.25,
        )
        self.assertAlmostEqual(clearance, 4.5)

    def test_current_static_collision_returns_zero(self):
        occupancy_map = make_map()
        occupancy_map.data_backup[9, 10] = 0
        clearance = occupancy_map.static_straight_path_clearance(
            x=10.0,
            y=10.0,
            heading=0.0,
            max_distance=4.0,
            footprint_radius=1.0,
        )
        self.assertEqual(clearance, 0.0)


if __name__ == "__main__":
    unittest.main()
