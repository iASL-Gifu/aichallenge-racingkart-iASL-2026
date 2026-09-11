"""Physical wall constraints shared by MPC, fallback and recovery."""
import math


def longitudinal_extent(config=None):
    body = getattr(config, 'collision_geometry', None)
    # Include uncertain pose origin; do not confuse wheelbase with body length.
    return (float(getattr(body, 'length', 2.064)) / 2.
            + abs(float(getattr(body, 'rear_axle_to_center', .522))))


def wall_center_bounds(lower, upper, *, course_margin, half_width, guard,
                       heading_error=0., half_length=1.554):
    """Bounds already include course_margin. Count the body width only once.

    half_width + half_length*abs(yaw error) conservatively bounds the lateral
    projection of the rectangular body (|sin(e)| <= |e|, |cos(e)| <= 1).
    """
    if not all(math.isfinite(v) for v in (
            lower, upper, course_margin, half_width, guard, heading_error, half_length)):
        return math.inf, -math.inf
    inset = max(0., half_width - course_margin) + max(0., guard)
    inset += half_length * abs(heading_error)
    return lower + inset, upper - inset
