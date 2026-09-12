"""Physical wall constraints shared by MPC, fallback and recovery."""
import math


def longitudinal_extent(config=None):
    body = getattr(config, 'collision_geometry', None)
    if getattr(body, 'ego_position_origin', None) in ('rear_axle', 'center'):
        return float(getattr(body, 'length', 2.064))/2.
    # Compatibility for configurations with an unspecified origin.
    return (float(getattr(body, 'length', 2.064)) / 2.
            + abs(float(getattr(body, 'rear_axle_to_center', .522))))


def center_offset(config=None):
    body = getattr(config, 'collision_geometry', None)
    return (float(getattr(body, 'rear_axle_to_center', .522))
            if getattr(body, 'ego_position_origin', None) == 'rear_axle' else 0.)


def wall_half_width(config):
    body = getattr(config, 'collision_geometry', None)
    return (float(body.width)/2. if getattr(body, 'ego_position_origin', None) in ('rear_axle', 'center')
            else float(config.bicycle_model.width)/2.)


def wall_center_bounds(lower, upper, *, course_margin, half_width, guard,
                       heading_error=0., half_length=1.554, body_center_offset=0.):
    """Bounds already include course_margin. Count the body width only once.

    half_width + half_length*abs(yaw error) conservatively bounds the lateral
    projection of the rectangular body (|sin(e)| <= |e|, |cos(e)| <= 1).
    body_center_offset shifts the linearized center from the rear axle; the
    resulting corner coefficients are offset +/- half_length.
    """
    if not all(math.isfinite(v) for v in (
            lower, upper, course_margin, half_width, guard, heading_error, half_length, body_center_offset)):
        return math.inf, -math.inf
    inset = max(0., half_width - course_margin) + max(0., guard)
    inset += half_length * abs(heading_error)
    shift = body_center_offset * heading_error
    return lower + inset - shift, upper - inset - shift
