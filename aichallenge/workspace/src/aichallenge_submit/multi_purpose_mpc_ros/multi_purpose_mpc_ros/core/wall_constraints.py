"""Physical wall constraints shared by MPC, fallback and recovery."""
import math
from functools import lru_cache


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


@lru_cache(maxsize=4096)
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


def swept_sample_padding(bodies, geometry):
    """Cover continuous interpolation using its nearer endpoint (distance <= 1/2).

    Translation plus radius * yaw bounds displacement of every corner. Include
    uncertain extents and rear-axle-to-center rotation; never assume point cars.
    """
    if not all(math.isfinite(v) and v > 0. for v in (geometry.length, geometry.width)):
        return math.inf
    padding = 0.
    for body in bodies:
        values = (body.x, body.y, body.uncertainty, body.lateral_padding, body.center_offset)
        if (not body.yaw_valid or not body.position_valid
                or not all(math.isfinite(v) for v in values)
                or min(body.uncertainty, body.lateral_padding) < 0.):
            return math.inf
    for a, b in zip(bodies, bodies[1:]):
        angle = abs(math.atan2(math.sin(b.yaw-a.yaw), math.cos(b.yaw-a.yaw)))
        radius = math.hypot(geometry.length/2+max(a.uncertainty,b.uncertainty),
                            geometry.width/2+max(a.lateral_padding,b.lateral_padding))
        uncertainty_change = abs(a.uncertainty-b.uncertainty)+abs(a.lateral_padding-b.lateral_padding)
        center_arc = max(abs(a.center_offset), abs(b.center_offset))*angle*angle/8.
        padding = max(padding, .5*(math.hypot(b.x-a.x,b.y-a.y)+radius*angle)
                      + uncertainty_change + center_arc)
    return padding
