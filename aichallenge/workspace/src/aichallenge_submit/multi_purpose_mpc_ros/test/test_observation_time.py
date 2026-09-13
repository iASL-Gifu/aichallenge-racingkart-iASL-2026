import math
import pytest
from multi_purpose_mpc_ros.core.observation_time import body_observation_valid


def test_logged_clock_delivery_skew_is_valid_without_altering_stamps():
    assert body_observation_valid(43.539999026, 43.549999026, 43.564999026,
                                  previous_now=43.51)


@pytest.mark.parametrize('position,yaw', [(10.051, 10.), (10., 10.051),
                                         (9.499, 10.), (10., 9.499),
                                         (9.6, 9.9), (math.nan, 10.)])
def test_future_stale_misaligned_and_invalid_observations_stay_rejected(position, yaw):
    assert not body_observation_valid(10., position, yaw)


def test_rollback_is_not_hidden_by_future_tolerance_and_next_fresh_tick_recovers():
    assert not body_observation_valid(10., 10.01, 10.01, previous_now=10.02)
    assert body_observation_valid(10.025, 10.03, 10.03, previous_now=10.)
    assert not body_observation_valid(10., 20., 20., previous_now=10.)
