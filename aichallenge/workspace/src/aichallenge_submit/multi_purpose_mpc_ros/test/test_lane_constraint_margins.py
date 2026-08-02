import pytest

from multi_purpose_mpc_ros.core.reference_path import (
    OUTER_COURSE_MARGIN,
    lane_constraint_margins,
    lane_minimum_free_segment_width,
)


@pytest.mark.parametrize("lane", [0, 1, 2])
def test_selected_lane_has_no_additional_dynamic_margin(lane):
    assert lane_constraint_margins(lane, 1.2) == (0.0, 0.0)


def test_full_width_keeps_model_safety_margin():
    assert lane_constraint_margins(None, 1.2) == (1.2, 1.2)


def test_negative_margin_is_clamped():
    assert lane_constraint_margins(None, -0.3) == (0.0, 0.0)


def test_outer_course_margin_is_point_three_metres():
    assert OUTER_COURSE_MARGIN == pytest.approx(0.3)


def test_l1_uses_center_corridor_width_instead_of_full_vehicle_width():
    assert lane_minimum_free_segment_width(1, 1.6, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("lane", [None, 0, 2])
def test_non_l1_corridors_keep_full_vehicle_width(lane):
    assert lane_minimum_free_segment_width(lane, 1.6, 0.5) == pytest.approx(1.6)
