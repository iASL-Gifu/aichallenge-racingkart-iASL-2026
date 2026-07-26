import pytest

from multi_purpose_mpc_ros.core.MPC import (
    blend_lateral_reference,
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
