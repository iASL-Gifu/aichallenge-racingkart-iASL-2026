"""Projection optimization must preserve arc/lateral geometry and tie order."""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.closed_path_projector import ClosedPathProjector
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    build_closed_path_arc_lengths, project_to_closed_path_frenet,
)
from .test_overtake_session import controller_method


def course_points():
    path = Path(__file__).parents[1] / 'env/centerline/traj_center_mincurv_capped.csv'
    with path.open() as stream:
        return [(float(row['x_m']), float(row['y_m'])) for row in csv.DictReader(stream)]


def test_real_course_matches_scalar_projection_at_vertices_and_near_walls():
    geometry = build_closed_path_arc_lengths(course_points())
    projector = ClosedPathProjector(*geometry)
    points = np.array(geometry[0])
    rng = np.random.default_rng(260)
    queries = np.vstack((points, points + rng.uniform(-5, 5, points.shape),
                         points[230:281] + [0.1, -0.1]))
    for x, y in queries:
        expected = project_to_closed_path_frenet(x, y, *geometry)
        assert projector.project(x, y) == pytest.approx(expected, abs=1e-10, rel=0)
        assert projector.project(x, y) == pytest.approx(expected, abs=1e-10, rel=0)
    assert projector.hits >= len(queries)


@pytest.mark.parametrize('points', [[], [(0, 0)], [(0, 0), (0, 0)],
    [(0, 0), (2, 0), (2, 0), (2, 2), (0, 2)],
    [(0, 0), (2, 2), (0, 2), (2, 0)]])
def test_degenerate_segments_wrap_and_first_segment_ties(points):
    geometry = build_closed_path_arc_lengths(points)
    projector = ClosedPathProjector(*geometry)
    for x, y in [(0, 0), (1, 1), (0, -1), (2, 2), (1, 0)]:
        expected = project_to_closed_path_frenet(x, y, *geometry)
        assert projector.project(x, y) == (None if expected is None else pytest.approx(expected))
    for x in [float('nan'), float('inf'), -float('inf')]:
        assert projector.project(x, 0) is None


def test_exact_keys_lru_and_geometry_ownership():
    points, cumulative, total = build_closed_path_arc_lengths([(0, 0), (2, 0), (2, 2)])
    projector = ClosedPathProjector(points, cumulative, total, cache_size=2)
    original = projector.project(1., 0.1)
    points[0] = (100, 100)
    cumulative[0] = 100
    assert projector.project(1., 0.1) == original
    assert projector.hits == 1
    assert projector.project(1., 0.10000001) != original
    projector.project(1., 0.2)
    assert len(projector._cache) == 2
    assert projector.project(1., 0.1) == original
    assert projector.misses == 4
    replacement = ClosedPathProjector(points, cumulative, total)
    assert replacement.project(1., 0.1) != original


def test_controller_reuses_only_coordinate_projection_across_target_changes():
    projector = ClosedPathProjector(*build_closed_path_arc_lengths([(0, 0), (2, 0), (2, 2)]))
    controller = SimpleNamespace(_center_projector=projector, target_id='d2')
    project = controller_method('_center_frenet')
    initial = project(controller, 1., .1)
    controller.target_id = 'd3'
    assert project(controller, 1., .1) == initial
    assert projector.hits == 1
    assert project(controller, 1., .2) != initial
