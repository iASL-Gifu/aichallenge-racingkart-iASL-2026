"""Enforce an already selected emergency limit without choosing lanes/hazards."""
import math


def enforce_emergency_limit(speed, acceleration, boost, *, limit, measured_speed,
                            kp, a_min, a_max, active):
    if not active or limit is None or speed < 0.0:
        return speed, acceleration, boost
    if not all(math.isfinite(float(x)) for x in
               (speed, acceleration, limit, measured_speed, kp, a_min, a_max)):
        return speed, acceleration, boost
    bounded_speed = min(speed, max(0.0, limit))
    bounded_acceleration = min(acceleration,
        min(a_max, max(a_min, kp * (bounded_speed - abs(measured_speed)))))
    return bounded_speed, bounded_acceleration, False
