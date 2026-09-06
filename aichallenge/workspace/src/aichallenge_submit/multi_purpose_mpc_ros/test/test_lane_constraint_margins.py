import pytest

from multi_purpose_mpc_ros.core.reference_path import (
    OUTER_COURSE_MARGIN,
    collapsed_constraint_snapshot,
    lane_constraint_margins,
    lane_minimum_free_segment_width,
    lane_relaxation_profile,
    relaxed_lane_bounds,
    intersect_constraint_bounds,
    retain_first_collapsed_constraint,
)


def test_retry_relaxation_tapers_from_near_vehicle_to_zero():
    assert lane_relaxation_profile(4, 0.6) == pytest.approx(
        [0.6, 0.4, 0.2, 0.0])


def test_retry_relaxation_can_keep_terminal_residual():
    assert lane_relaxation_profile(
        4, 1.2, terminal_ratio=0.35
    ) == pytest.approx([1.2, 0.94, 0.68, 0.42])


@pytest.mark.parametrize(
    "lane,expected",
    [(0, (-2.0, -0.4)), (1, (-0.8, 0.8)), (2, (-0.1, 2.0))],
)
def test_retry_relaxes_artificial_lane_toward_center(lane, expected):
    lane_lb = -2.0 if lane == 0 else (-0.5 if lane == 1 else 0.5)
    lane_ub = -1.0 if lane == 0 else (0.5 if lane == 1 else 2.0)
    assert relaxed_lane_bounds(
        lane, lane_lb, lane_ub, 0.6) == pytest.approx(expected)


def test_lane_relaxation_is_clipped_by_hard_obstacle_bounds():
    lb, ub = intersect_constraint_bounds(-0.4, 0.7, -1.0, 1.0)
    assert (lb, ub) == pytest.approx((-0.4, 0.7))


@pytest.mark.parametrize("lane", [0, 1, 2])
def test_selected_lane_has_no_additional_dynamic_margin(lane):
    assert lane_constraint_margins(lane, 1.2) == (0.0, 0.0)


def test_full_width_keeps_model_safety_margin():
    assert lane_constraint_margins(None, 1.2) == (1.2, 1.2)


def test_negative_margin_is_clamped():
    assert lane_constraint_margins(None, -0.3) == (0.0, 0.0)


def test_outer_course_margin_remains_point_six_metres():
    assert OUTER_COURSE_MARGIN == pytest.approx(0.6)


def test_l1_uses_center_corridor_width_instead_of_full_vehicle_width():
    assert lane_minimum_free_segment_width(1, 1.6, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("lane", [None, 0, 2])
def test_non_l1_corridors_keep_full_vehicle_width(lane):
    assert lane_minimum_free_segment_width(lane, 1.6, 0.5) == pytest.approx(1.6)


def test_outer_lane_zero_width_snapshot_is_collapsed():
    detail = collapsed_constraint_snapshot(
        [2.0, 0.0, 1.8], [0.5, 0.0, 0.6], [312, 313, 314])

    assert detail == {
        "index": 1,
        "wp": 313,
        "width": pytest.approx(0.0),
        "lower": pytest.approx(0.0),
        "upper": pytest.approx(0.0),
    }


def test_outer_lane_positive_width_snapshot_is_not_collapsed():
    assert collapsed_constraint_snapshot(
        [-0.2, -0.1], [-1.8, -1.7], [220, 221]) is None


def test_first_constraint_collapse_survives_later_retry():
    first = retain_first_collapsed_constraint(
        None, [1.0, 0.0], [-1.0, 0.0], [10, 11])
    retained = retain_first_collapsed_constraint(
        first, [1.0, 1.0], [-1.0, -1.0], [10, 11])
    assert retained == first
    assert retained["wp"] == 11
    assert retained["width"] == pytest.approx(0.0)
