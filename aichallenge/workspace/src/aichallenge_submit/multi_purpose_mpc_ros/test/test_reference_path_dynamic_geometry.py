"""Regression tests for per-call dynamic-obstacle raster geometry reuse."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.reference_path import ReferencePath, line_aa


class _IdentityMap:
    resolution = 1.0
    width = 20
    height = 20

    def __init__(self, data):
        self.data = data

    @staticmethod
    def w2m(x, y):
        return int(x), int(y)

    @staticmethod
    def m2w(x, y):
        return float(x), float(y)


def _path_and_waypoint(static_occupied_x=None):
    data = np.ones((20, 20), dtype=np.int8)
    if static_occupied_x is not None:
        data[10, static_occupied_x] = 0
    path = ReferencePath.__new__(ReferencePath)
    path.map = _IdentityMap(data)
    waypoint = SimpleNamespace(
        static_border_cells=((2.0, 10.0), (17.0, 10.0)))
    return path, waypoint


def _obstacle(cx, cy, radius):
    return SimpleNamespace(cx=float(cx), cy=float(cy), radius=float(radius))


def _legacy_compute_free_segments(
    path, waypoint, min_width, *, dynamic_obstacles=None,
):
    """Reproduce the pre-optimization cell loop for exact comparison."""
    free_segments = []
    ub_p = path.map.w2m(*waypoint.static_border_cells[0])
    lb_p = path.map.w2m(*waypoint.static_border_cells[1])
    x_list, y_list, _ = line_aa(ub_p[0], ub_p[1], lb_p[0], lb_p[1])
    ub_o, lb_o = ub_p, ub_p
    free_cells = False
    map_data = path.map.data

    def occupied(x, y):
        if map_data[y, x] == 0:
            return True
        if not dynamic_obstacles:
            return False
        return any(
            path._dynamic_obstacle_occupies_cell(x, y, obstacle)
            for obstacle in dynamic_obstacles
        )

    all_segments = []
    for x, y in zip(x_list[1:], y_list[1:]):
        cell_value = 0 if occupied(x, y) else 1
        if cell_value == 1:
            free_cells = True
            lb_o = (x, y)
        if (cell_value == 0 or (x, y) == lb_p) and free_cells:
            lb_o = (x, y)
            ub_w = path.map.m2w(ub_o[0], ub_o[1])
            lb_w = path.map.m2w(lb_o[0], lb_o[1])
            segment_width_sq = (
                (ub_w[0] - lb_w[0]) ** 2
                + (ub_w[1] - lb_w[1]) ** 2
            )
            all_segments.append(((ub_w, lb_w), segment_width_sq))
            if segment_width_sq > min_width**2:
                free_segments.append((ub_w, lb_w))
            ub_o = (x, y)
            free_cells = False
        elif cell_value == 0 and not free_cells:
            ub_o = (x, y)
            lb_o = (x, y)

    if not free_segments and all_segments:
        all_segments.sort(key=lambda segment: segment[1], reverse=True)
        widest_segment, widest_width_sq = all_segments[0]
        if widest_width_sq >= min_width**2:
            free_segments.append(widest_segment)
    return free_segments


@pytest.mark.parametrize(
    ("dynamic_obstacles", "static_occupied_x"),
    [
        (None, None),
        ([_obstacle(8, 10, 2)], None),
        ([_obstacle(8, 15, 1)], None),
        ([_obstacle(8, 10, 2), _obstacle(13, 10, 2)], None),
        ([_obstacle(8, 10, 2), _obstacle(13, 10, 2)], 5),
    ],
)
def test_precomputed_dynamic_geometry_preserves_free_segments(
    dynamic_obstacles, static_occupied_x,
):
    legacy_path, waypoint = _path_and_waypoint(static_occupied_x)
    legacy_segments = legacy_path._compute_free_segments(
        waypoint,
        0.5,
        dynamic_obstacles=dynamic_obstacles,
        precompute_dynamic_obstacle_geometry=False,
    )

    precomputed_path, waypoint = _path_and_waypoint(static_occupied_x)
    precomputed_segments = precomputed_path._compute_free_segments(
        waypoint,
        0.5,
        dynamic_obstacles=dynamic_obstacles,
        precompute_dynamic_obstacle_geometry=True,
    )

    assert precomputed_segments == legacy_segments
    assert len(precomputed_segments) == len(legacy_segments)
    for precomputed_segment, legacy_segment in zip(
        precomputed_segments, legacy_segments,
    ):
        np.testing.assert_array_equal(
            np.asarray(precomputed_segment), np.asarray(legacy_segment))


@pytest.mark.parametrize(
    ("dynamic_obstacles", "static_occupied_x", "min_width"),
    [
        (None, None, 0.5),
        (None, 5, 0.5),
        ([_obstacle(8, 10, 2)], None, 0.5),
        ([_obstacle(8, 15, 1)], None, 0.5),
        ([_obstacle(8, 10, 2), _obstacle(13, 10, 2)], None, 0.5),
        ([_obstacle(8, 10, 2), _obstacle(13, 10, 2)], 5, 0.5),
        ([_obstacle(8, 10, 6)], None, 20.0),
    ],
)
def test_reused_cell_coordinate_preserves_legacy_free_segments(
    dynamic_obstacles, static_occupied_x, min_width,
):
    legacy_path, waypoint = _path_and_waypoint(static_occupied_x)
    legacy_segments = _legacy_compute_free_segments(
        legacy_path,
        waypoint,
        min_width,
        dynamic_obstacles=dynamic_obstacles,
    )

    optimized_path, waypoint = _path_and_waypoint(static_occupied_x)
    optimized_segments = optimized_path._compute_free_segments(
        waypoint,
        min_width,
        dynamic_obstacles=dynamic_obstacles,
    )

    assert optimized_segments == legacy_segments
    assert len(optimized_segments) == len(legacy_segments)
    for optimized_segment, legacy_segment in zip(
        optimized_segments, legacy_segments,
    ):
        np.testing.assert_array_equal(
            np.asarray(optimized_segment), np.asarray(legacy_segment))


def test_precomputed_raster_formula_matches_existing_cell_occupancy():
    path, _ = _path_and_waypoint()
    obstacles = [_obstacle(8, 10, 2), _obstacle(13, 10, 3)]

    for obstacle in obstacles:
        center_x, center_y = path.map.w2m(obstacle.cx, obstacle.cy)
        radius_px = int(np.ceil(
            float(obstacle.radius) / path.map.resolution))
        for cell_x in range(2, 18):
            for cell_y in range(7, 14):
                delta_x = int(cell_x) - int(center_x)
                delta_y = int(cell_y) - int(center_y)
                precomputed_result = bool(
                    -radius_px <= delta_x < radius_px
                    and -radius_px <= delta_y < radius_px
                    and delta_x ** 2 + delta_y ** 2 <= radius_px ** 2
                )
                assert precomputed_result == (
                    path._dynamic_obstacle_occupies_cell(
                        cell_x, cell_y, obstacle))


def test_static_occupancy_still_short_circuits_dynamic_checks():
    path, waypoint = _path_and_waypoint(static_occupied_x=5)
    obstacle = _obstacle(8, 15, 1)
    original = path._dynamic_obstacle_occupies_cell
    checked_cells = []

    def record_dynamic_check(self, cell_x, cell_y, checked_obstacle):
        checked_cells.append((cell_x, cell_y))
        return original(cell_x, cell_y, checked_obstacle)

    path._dynamic_obstacle_occupies_cell = MethodType(
        record_dynamic_check, path)
    path._compute_free_segments(
        waypoint,
        0.5,
        dynamic_obstacles=[obstacle],
        precompute_dynamic_obstacle_geometry=False,
    )

    assert (5, 10) not in checked_cells
    assert checked_cells


@pytest.mark.parametrize("static_occupied_x", [None, 5])
def test_precomputed_geometry_preserves_three_by_three_occupancy(
    static_occupied_x,
):
    path, _ = _path_and_waypoint(static_occupied_x)
    obstacles = [_obstacle(8, 10, 2), _obstacle(13, 10, 2)]
    geometry = [
        (
            obstacle,
            *path.map.w2m(obstacle.cx, obstacle.cy),
            int(np.ceil(float(obstacle.radius) / path.map.resolution)),
        )
        for obstacle in obstacles
    ]

    for cell_x in range(3, 17):
        legacy = path._is_obstacle_occupied(
            cell_x, 10, dynamic_obstacles=obstacles)
        precomputed = path._is_obstacle_occupied(
            cell_x,
            10,
            dynamic_obstacles=obstacles,
            dynamic_obstacle_raster_geometry=geometry,
        )
        assert precomputed == legacy
