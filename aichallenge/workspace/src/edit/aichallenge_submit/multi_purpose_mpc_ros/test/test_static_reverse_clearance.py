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

    def test_oriented_box_accepts_parallel_wall_clearance_rejected_by_disk(self):
        occupancy_map = make_map(width=200, height=200, resolution=0.1)
        # Wall at world x=5.0. The kart is parallel to it and has 0.2 m real
        # lateral clearance. A diagonal enclosing disk is over-conservative.
        occupancy_map.data_backup[:, 50] = 0
        self.assertFalse(occupancy_map.static_disk_is_free(
            x=6.0, y=10.0, radius=1.10))
        self.assertTrue(occupancy_map.static_oriented_box_is_free(
            x=6.0, y=10.0, heading=np.pi / 2.0,
            half_length=1.05, half_width=0.80))

    def test_oriented_box_rejects_actual_wall_overlap(self):
        occupancy_map = make_map(width=200, height=200, resolution=0.1)
        occupancy_map.data_backup[:, 50] = 0
        self.assertFalse(occupancy_map.static_oriented_box_is_free(
            x=5.7, y=10.0, heading=np.pi / 2.0,
            half_length=1.05, half_width=0.80))

    def test_straight_reverse_allows_monotonic_escape_from_contact(self):
        occupancy_map = make_map(width=200, height=200, resolution=0.1)
        # Wall behind the kart overlaps its rear at the initial pose. Straight
        # reverse in this heading moves away from the wall and becomes free.
        occupancy_map.data_backup[:, :101] = 0
        clearance, initial, reached_free = (
            occupancy_map.static_straight_reverse_box_clearance(
                x=10.8, y=10.0, heading=np.pi,
                max_distance=3.0, half_length=1.0, half_width=0.7,
                escape_max_distance=0.5, step=0.1))
        self.assertGreater(initial, 0.0)
        self.assertTrue(reached_free)
        self.assertAlmostEqual(clearance, 3.0)

    def test_straight_reverse_rejects_increasing_contact(self):
        occupancy_map = make_map(width=200, height=200, resolution=0.1)
        occupancy_map.data_backup[:, :101] = 0
        clearance, initial, reached_free = (
            occupancy_map.static_straight_reverse_box_clearance(
                x=10.8, y=10.0, heading=0.0,
                max_distance=3.0, half_length=1.0, half_width=0.7,
                escape_max_distance=0.5, step=0.1))
        self.assertGreater(initial, 0.0)
        self.assertFalse(reached_free)
        self.assertEqual(clearance, 0.0)

    def test_straight_reverse_rejects_contact_not_cleared_in_time(self):
        occupancy_map = make_map(width=200, height=200, resolution=0.1)
        occupancy_map.data_backup[:, :101] = 0
        clearance, initial, reached_free = (
            occupancy_map.static_straight_reverse_box_clearance(
                x=10.8, y=10.0, heading=np.pi,
                max_distance=3.0, half_length=1.0, half_width=0.7,
                escape_max_distance=0.1, step=0.1))
        self.assertGreater(initial, 0.0)
        self.assertFalse(reached_free)
        self.assertEqual(clearance, 0.0)
        diagnostic = occupancy_map.last_static_reverse_diagnostic
        self.assertEqual(diagnostic["reason"], "escape_distance_exceeded")
        self.assertGreater(diagnostic["safe_prefix_clearance"], 0.0)
        self.assertGreater(diagnostic["total_improvement"], 0.0)

    def test_straight_reverse_tolerates_single_quantization_increase(self):
        occupancy_map = make_map()
        values = iter([0.0067, 0.0040, 0.0054, 0.0030, 0.0, 0.0])
        occupancy_map.static_oriented_box_interference = (
            lambda *args, **kwargs: next(values, 0.0))
        clearance, initial, reached_free = (
            occupancy_map.static_straight_reverse_box_clearance(
                x=10.0, y=10.0, heading=0.0,
                max_distance=0.5, half_length=1.0, half_width=0.7,
                escape_max_distance=0.5, step=0.1,
                interference_tolerance=0.005, trend_window=4,
                total_improvement_threshold=0.001))
        self.assertAlmostEqual(initial, 0.0067)
        self.assertTrue(reached_free)
        self.assertAlmostEqual(clearance, 0.5)


if __name__ == "__main__":
    unittest.main()
