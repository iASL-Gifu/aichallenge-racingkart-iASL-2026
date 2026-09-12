"""Compare three forward steering choices toward a waypoint, then reverse."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Motion:
    direction: int
    steering: float
    poses: tuple
    improvement: float
    wall_reduction: float = 0.
    heading_improvement: float = 0.
    speed_limit: float = math.inf


def rollout(pose, direction, steering, distance, wheelbase, step=.05):
    x, y, yaw = pose
    count = max(1, math.ceil(distance / step))
    ds = direction * distance / count
    curvature = math.tan(steering) / wheelbase
    points = [(x, y, yaw)]
    for _ in range(count):
        mid = yaw + .5 * ds * curvature
        x += ds * math.cos(mid)
        y += ds * math.sin(mid)
        yaw += ds * curvature
        points.append((x, y, yaw))
    return tuple(points)


def recovery_speed_limit(path):
    """Reserve .5 s response, braking at 1 m/s² and .05 m clearance."""
    distance = sum(math.hypot(b[0]-a[0], b[1]-a[1])
                   for a, b in zip(path, path[1:]))
    return min(1., max(0., math.sqrt(.25+2.*max(distance-.05, 0.))-.5))


def evaluate(pose, *, target, distance, wheelbase, steering_limit, clear,
             previous_steering=0., steering_step=math.inf, overlap=None, min_reverse_distance=.1, reverse_clear=None):
    """Choose WP-directed forward motion, otherwise straight reverse.

    Neither corridor membership nor corridor improvement participates in this
    decision. The caller checks the full swept body against walls and traffic.
    """
    values = (*pose, *target, distance, wheelbase, steering_limit, previous_steering, min_reverse_distance)
    if (not all(math.isfinite(float(v)) for v in values)
            or distance <= 0 or wheelbase <= 0 or steering_limit < 0
            or steering_step < 0 or math.isnan(steering_step)):
        return None, 'invalid_motion_input'
    x, y, yaw = pose
    dx, dy = target[0]-x, target[1]-y
    length_sq = dx*dx+dy*dy
    if length_sq < 1e-6:
        return None, 'invalid_waypoint_target'
    start_distance = math.sqrt(length_sq)
    # Use one fixed WP bearing so translation does not change the heading goal.
    bearing = math.atan2(dy, dx)
    def heading_error(angle):
        return abs(math.atan2(math.sin(angle-bearing), math.cos(angle-bearing)))
    initial_overlap = float(overlap(pose)) if overlap else 0.
    if not math.isfinite(initial_overlap) or initial_overlap < 0.:
        return None, 'invalid_overlap_metric'
    candidates, reasons = [], []
    for requested in (steering_limit, 0., -steering_limit):
        steering = max(previous_steering-steering_step,
                       min(previous_steering+steering_step, requested))
        steering = max(-steering_limit, min(steering_limit, steering))
        path = rollout(pose, 1, steering, distance, wheelbase)
        end_x, end_y, _ = path[-1]
        improvement = start_distance - math.hypot(target[0]-end_x, target[1]-end_y)
        safe, reason = clear(path)
        if safe and (improvement > .01 or reason == 'wall_escape'):
            end_overlap = float(overlap(path[-1])) if overlap else 0.
            if not math.isfinite(end_overlap) or end_overlap < 0.:
                reasons.append(f'forward/{requested:+.3f}:invalid_overlap_metric')
                continue
            reduction = initial_overlap-end_overlap
            heading_gain = heading_error(yaw)-heading_error(path[-1][2])
            candidates.append(Motion(1, steering, path, improvement,
                                     reduction, heading_gain))
        else:
            reasons.append(f'forward/{requested:+.3f}:' + (reason if not safe else 'away_from_waypoint'))
    if candidates:
        # Physical clearance first; among equal clearance outcomes, restore
        # heading before optimizing WP distance. Safety checks above are hard gates.
        candidates.sort(key=lambda motion: (-motion.wall_reduction,
                                            -motion.heading_improvement,
                                            -motion.improvement, abs(motion.steering)))
        return candidates[0], 'forward_to_waypoint'
    path = rollout(pose, -1, 0., distance, wheelbase)
    check_reverse = reverse_clear or clear
    safe, reverse_reason = check_reverse(path)
    if safe:
        return Motion(-1, 0., path, 0.), 'reverse: ' + '; '.join(reasons)
    # Stop the validated prefix before worsening overlap OR a new wall.
    # Re-run static and traffic checks; never skip a failing sample.
    if reverse_reason.startswith(('wall_overlap_', 'new_wall_contact_at_step=')):
        for end in range(len(path)-1, 1, -1):
            prefix = path[:end]
            length = math.hypot(prefix[-1][0]-pose[0], prefix[-1][1]-pose[1])
            if length+1e-8 < max(.1, min_reverse_distance):
                break
            safe, reason = check_reverse(prefix)
            if safe and reason in ('wall_escape', 'clear'):
                return Motion(-1, 0., prefix, 0.,
                              speed_limit=recovery_speed_limit(prefix)), (
                    f'reverse_prefix={length:.2f}m; full_reverse={reverse_reason}')
    return None, '; '.join(reasons + [f'reverse={reverse_reason}'])
