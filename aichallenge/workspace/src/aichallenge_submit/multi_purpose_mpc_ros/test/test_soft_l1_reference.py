import pytest

from multi_purpose_mpc_ros.core.MPC import (
    MPC,
    blend_lateral_reference,
    curvature_lateral_shift,
    lateral_reference_ramp_duration,
)


def test_soft_reference_starts_at_current_lateral_position():
    assert blend_lateral_reference(-0.8, 0.2, 0.0) == pytest.approx(-0.8)


def test_soft_reference_reaches_lane_center():
    assert blend_lateral_reference(-0.8, 0.2, 1.0) == pytest.approx(0.2)


def test_soft_reference_interpolates_and_clamps_alpha():
    assert blend_lateral_reference(-0.8, 0.2, 0.5) == pytest.approx(-0.3)
    assert blend_lateral_reference(-0.8, 0.2, -1.0) == pytest.approx(-0.8)
    assert blend_lateral_reference(-0.8, 0.2, 2.0) == pytest.approx(0.2)


def test_ramp_duration_keeps_minimum_for_nearby_target():
    assert lateral_reference_ramp_duration(
        -0.2, 0.0, 1.5, 0.5
    ) == pytest.approx(1.5)


def test_ramp_duration_extends_to_respect_speed_limit():
    assert lateral_reference_ramp_duration(
        -3.5, 0.0, 1.5, 0.5
    ) == pytest.approx(7.0)


def test_zero_speed_limit_falls_back_to_minimum_duration():
    assert lateral_reference_ramp_duration(
        -3.5, 0.0, 1.5, 0.0
    ) == pytest.approx(1.5)


def test_curvature_shift_is_zero_below_threshold():
    assert curvature_lateral_shift(0.03, 0.04, 6.0, 1.0) == pytest.approx(0.0)


def test_curvature_shift_scales_above_threshold():
    assert curvature_lateral_shift(0.12, 0.04, 6.0, 1.0) == pytest.approx(0.48)


def test_curvature_shift_is_capped():
    assert curvature_lateral_shift(0.30, 0.04, 6.0, 1.0) == pytest.approx(1.0)


def test_soft_lane_reference_accepts_l1_side_offset():
    mpc = MPC.__new__(MPC)

    mpc.set_soft_lateral_reference(
        lane_idx=0,
        start_e_y=-2.0,
        alpha=0.4,
        lateral_offset=0.3,
    )

    assert mpc.soft_target_lane_idx == 0
    assert mpc.soft_target_start_e_y == pytest.approx(-2.0)
    assert mpc.soft_target_alpha == pytest.approx(0.4)
    assert mpc.soft_target_lateral_offset == pytest.approx(0.3)


def test_clearing_soft_lane_reference_clears_offset():
    mpc = MPC.__new__(MPC)

    mpc.set_soft_lateral_reference(lane_idx=0, lateral_offset=0.3)
    mpc.set_soft_lateral_reference()

    assert mpc.soft_target_lane_idx is None
    assert mpc.soft_target_lateral_offset == pytest.approx(0.0)
