"""Focused regressions for spatial-step moving-V2X occupancy."""

from types import SimpleNamespace

import numpy as np

from multi_purpose_mpc_ros.core.reference_path import ReferencePath


class _GridMap:
    width = 7
    height = 7
    resolution = 1.0

    def __init__(self):
        self.data = np.ones((self.height, self.width), dtype=np.int8)

    @staticmethod
    def m2w(x, y):
        return float(x), float(y)

    @staticmethod
    def w2m(x, y):
        return int(round(x)), int(round(y))


def _reference_path():
    path = ReferencePath.__new__(ReferencePath)
    path.map = _GridMap()
    return path


def test_step_specific_moving_obstacle_does_not_change_static_occupancy():
    path = _reference_path()
    static = path.map.data.copy()
    static[1, 1] = 0
    moving = [SimpleNamespace(cx=4.0, cy=4.0, radius=0.6)]

    assert path._is_obstacle_occupied(
        1, 1, occupancy_data=static, dynamic_obstacles=moving)
    assert path._is_obstacle_occupied(
        4, 4, occupancy_data=static, dynamic_obstacles=moving)
    assert not path._is_obstacle_occupied(
        0, 6, occupancy_data=static, dynamic_obstacles=moving)
    assert np.array_equal(path.map.data, np.ones((7, 7), dtype=np.int8))


def test_without_step_specific_obstacles_existing_map_semantics_remain():
    path = _reference_path()
    path.map.data[3, 3] = 0

    assert path._is_obstacle_occupied(3, 3)
    assert not path._is_obstacle_occupied(0, 0)
