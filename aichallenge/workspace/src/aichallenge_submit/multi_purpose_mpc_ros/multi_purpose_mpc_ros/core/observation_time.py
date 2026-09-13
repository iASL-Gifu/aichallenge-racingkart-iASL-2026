"""Bounded tolerance for independently delivered ROS clock and sensor data."""
import math


def body_observation_valid(now, position_stamp, yaw_stamp, *, previous_now=None,
                           max_age=.5, future_tolerance=.05, max_stamp_gap=.2):
    if not all(math.isfinite(v) for v in (now, position_stamp, yaw_stamp)):
        return False
    if previous_now is not None and now < previous_now - 1e-9:
        return False
    return ( -future_tolerance-1e-9 <= now-position_stamp <= max_age
             and -future_tolerance-1e-9 <= now-yaw_stamp <= max_age
             and abs(position_stamp-yaw_stamp) <= max_stamp_gap)
