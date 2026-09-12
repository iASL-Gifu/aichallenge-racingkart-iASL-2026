"""Obstacle clearance constrains the reference point after fitting the body."""
import math

import pytest

from multi_purpose_mpc_ros.core.map import Obstacle
from multi_purpose_mpc_ros.core.reference_path import obstacle_center_bounds
from .probe_support import configured_mpc


@pytest.mark.parametrize('lane', [0, 1, 2])
def test_only_occupied_edges_need_an_extra_half_width(lane):
    assert obstacle_center_bounds(
        -4., -1., 1.6, lane, .8, lower_occupied=False,
        upper_occupied=True) == pytest.approx((-4., -1.8))
    assert obstacle_center_bounds(
        1., 4., 1.6, lane, .8, lower_occupied=True,
        upper_occupied=False) == pytest.approx((1.8, 4.))
    assert obstacle_center_bounds(
        -3.95, 3.95, 1.6, lane, .8, lower_occupied=False,
        upper_occupied=False) == pytest.approx((-3.95, 3.95))


@pytest.mark.parametrize('margin,expected', [(.3, (-1.2, 1.2)),
                                             (.8, (-1.2, 1.2)),
                                             (1., (-1., 1.))])
def test_full_width_uses_larger_margin_without_adding_half_width_twice(margin, expected):
    assert obstacle_center_bounds(
        -2., 2., 1.6, None, margin, lower_occupied=True,
        upper_occupied=True) == pytest.approx(expected)


@pytest.fixture(scope='module')
def model():
    return configured_mpc()


@pytest.fixture
def path(model):
    reference = model.model.reference_path
    reference.map.reset_map()
    reference.reset_dynamic_constraints()
    reference.unsafe_static_fallback_wp_ids = []
    reference.unsafe_static_fallback_on_narrow = False
    return reference


def occupied_segment(path, lower, upper):
    wp = path.get_waypoint(62)
    cells = tuple((wp.x - math.sin(wp.psi) * lateral,
                   wp.y + math.cos(wp.psi) * lateral)
                  for lateral in (upper, lower))
    path.map.add_obstacles([Obstacle(x, y, .15) for x, y in cells])
    return cells


def constraints(path):
    wp = path.get_waypoint(62)
    upper, lower, _ = path.update_path_constraints(
        62, (wp.x, wp.y, wp.psi), 1, 1.087, 1.6, .8,
        connect_lane_from_current_pose=False)
    return lower[0], upper[0]


@pytest.mark.parametrize('lane', [None, 1])
@pytest.mark.parametrize('unsafe_fallback', [False, True])
def test_two_metre_passage_leaves_point_four_metres_for_reference_point(
        path, monkeypatch, lane, unsafe_fallback):
    path.target_lane_idx = lane
    path.unsafe_static_fallback_on_narrow = unsafe_fallback
    center = sum(path.get_lane_bounds(62)[1]) / 2.
    cells = occupied_segment(path, center - 1., center + 1.)
    monkeypatch.setattr(path, '_compute_free_segments', lambda *a, **k: [cells])

    lower, upper = constraints(path)

    assert (lower, upper) == pytest.approx((center - .2, center + .2))
    assert path.unsafe_static_fallback_wp_ids == []


@pytest.mark.parametrize('lane', [None, 1])
def test_free_segment_narrower_than_body_remains_blocked(path, monkeypatch, lane):
    path.target_lane_idx = lane
    center = sum(path.get_lane_bounds(62)[1]) / 2.
    cells = occupied_segment(path, center - .75, center + .75)
    monkeypatch.setattr(path, '_compute_free_segments', lambda *a, **k: [cells])
    assert constraints(path) == pytest.approx((0., 0.))


def test_raw_lane_overlap_is_rejected_when_body_center_cannot_fit(path, monkeypatch):
    path.target_lane_idx = 1
    _, lane_lower = path.get_lane_bounds(62)[1]
    # The raw segment overlaps L1 by .2m, but its upper obstacle requires .8m.
    cells = occupied_segment(path, lane_lower - 1.8, lane_lower + .2)
    monkeypatch.setattr(path, '_compute_free_segments', lambda *a, **k: [cells])
    assert constraints(path) == pytest.approx((0., 0.))
    assert path.free_segs == []
