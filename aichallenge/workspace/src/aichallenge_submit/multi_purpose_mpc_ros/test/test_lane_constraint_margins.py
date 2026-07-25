import pytest

from multi_purpose_mpc_ros.core.reference_path import (
    OUTER_COURSE_MARGIN,
    lane_constraint_margins,
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
