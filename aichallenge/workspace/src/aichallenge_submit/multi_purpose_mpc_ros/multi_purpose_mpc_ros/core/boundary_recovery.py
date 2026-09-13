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


def rollout(pose, direction, steering, distance, wheelbase, step=.05, *,
            initial_steering=None, steering_rate=math.inf, speed=1.):
    x, y, yaw = pose
    count = max(1, math.ceil(distance / step))
    ds = direction * distance / count
    angle = steering if initial_steering is None else initial_steering
    points = [(x, y, yaw)]
    for _ in range(count):
        change = max(-steering_rate*abs(ds)/speed,
                     min(steering_rate*abs(ds)/speed, steering-angle))
        curvature = math.tan(angle+.5*change) / wheelbase
        angle += change
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
             previous_steering=0., steering_step=math.inf, overlap=None, min_reverse_distance=.1, reverse_clear=None, target_heading=None, preferred_steering=0., allow_forward=True, forward_turn_clear=None, wall_escape_clear=None, excluded=(),
             steering_rate=math.inf, motion_speed=1., measured_speed=0.,
             retained=None, retained_distance=None, only_motion=None, failure_path_clear=None, prepare_steering=False, compare_retained_turns=True, recompare=False, retry_clear=None, connection_check=None):
    """Choose WP-directed forward motion, otherwise straight reverse.

    Neither corridor membership nor corridor improvement participates in this
    decision. The caller checks the full swept body against walls and traffic.
    """
    if retained is not None:
        # Recompute from the CURRENT pose; never execute a cached path.
        options = dict(target=target, distance=(distance if retained_distance is None else retained_distance), wheelbase=wheelbase,
            steering_limit=steering_limit, clear=clear, previous_steering=previous_steering,
            steering_step=steering_step, overlap=overlap, min_reverse_distance=min_reverse_distance,
            reverse_clear=reverse_clear, target_heading=target_heading,
            preferred_steering=preferred_steering, allow_forward=allow_forward,
            forward_turn_clear=forward_turn_clear, wall_escape_clear=wall_escape_clear,
            excluded=excluded, steering_rate=steering_rate, motion_speed=motion_speed,
            measured_speed=measured_speed, only_motion=retained, failure_path_clear=failure_path_clear, prepare_steering=prepare_steering, retry_clear=retry_clear, connection_check=connection_check)
        held, why = evaluate(pose, **options)
        if held is not None:
            if recompare:
                options['only_motion'] = None
                options['distance'] = distance
                alternative, _ = evaluate(pose, **options)
                if (alternative is not None
                        and alternative.wall_reduction >= held.wall_reduction-1e-6
                        and (alternative.heading_improvement > held.heading_improvement+math.radians(5.)
                             or alternative.wall_reduction > held.wall_reduction+.01)):
                    return alternative, 'periodic_improvement'
            if not recompare and compare_retained_turns and held.direction == 1 and abs(held.steering) < .01:
                turns = []
                for steering in (steering_limit, -steering_limit):
                    options['only_motion'] = (1, steering)
                    turn, _ = evaluate(pose, **options)
                    if (turn is not None and turn.direction == 1
                            and abs(turn.steering) >= .01
                            and turn.heading_improvement > max(0., held.heading_improvement)+math.radians(5.)
                            and turn.wall_reduction >= held.wall_reduction-1e-6):
                        turns.append(turn)
                if turns:
                    return max(turns, key=lambda m: (m.wall_reduction, m.heading_improvement,
                                                    m.improvement)), 'turn_improves_held_straight'
            return held, 'retained: '+why
    values = (*pose, *target, distance, wheelbase, steering_limit, previous_steering, min_reverse_distance, motion_speed, measured_speed)
    if (not all(math.isfinite(float(v)) for v in values)
            or distance <= 0 or wheelbase <= 0 or steering_limit < 0
            or steering_step < 0 or math.isnan(steering_step)
            or steering_rate <= 0 or math.isnan(steering_rate) or motion_speed <= 0):
        return None, 'invalid_motion_input'
    x, y, yaw = pose
    dx, dy = target[0]-x, target[1]-y
    length_sq = dx*dx+dy*dy
    if length_sq < 1e-6:
        return None, 'invalid_waypoint_target'
    start_distance = math.sqrt(length_sq)
    # Use one fixed WP bearing so translation does not change the heading goal.
    bearing = math.atan2(dy, dx) if target_heading is None else float(target_heading)
    if not math.isfinite(bearing):
        return None, 'invalid_target_heading'
    def heading_error(angle):
        return abs(math.atan2(math.sin(angle-bearing), math.cos(angle-bearing)))
    initial_overlap = float(overlap(pose)) if overlap else 0.
    if not math.isfinite(initial_overlap) or initial_overlap < 0.:
        return None, 'invalid_overlap_metric'
    minimum_distance = max(.1, min_reverse_distance,
                           .05+.5*abs(measured_speed)+.5*measured_speed**2)
    if distance < minimum_distance:
        return None, 'insufficient_stopping_distance'
    def trajectory(direction, steering, length, speed_cap=math.inf):
        cap = min(motion_speed, speed_cap, math.sqrt(.25+2.*max(length-.05, 0.))-.5)
        return rollout(pose, direction, steering, length, wheelbase,
                       initial_steering=(previous_steering
                           if direction > 0 and not prepare_steering and math.isfinite(steering_rate) else None),
                       steering_rate=steering_rate, speed=max(.01, abs(measured_speed), cap))
    candidates, reasons, alternatives = [], [], []
    def key(direction, steering):
        return (direction, 0 if abs(steering) < .01 else (1 if steering > 0 else -1))
    for requested in ((only_motion[1],) if only_motion and only_motion[0] == 1
                      else (() if only_motion else (steering_limit, 0., -steering_limit))):
        steering = max(previous_steering-steering_step,
                       min(previous_steering+steering_step, requested))
        steering = max(-steering_limit, min(steering_limit, steering))
        path = trajectory(1, steering, distance)
        alternatives.append((1, steering, path))
        if key(1, steering) in excluded and retry_clear is None:
            reasons.append(f'forward/{requested:+.3f}:measured_no_progress')
            continue
        def checked_forward(candidate_path):
            if failure_path_clear is not None:
                safe, reason = failure_path_clear(candidate_path)
                if not safe:
                    return safe, reason
            safe, reason = clear(candidate_path)
            gain = heading_error(yaw)-heading_error(candidate_path[-1][2])
            if (not safe and reason.startswith('wall_overlap_')
                    and abs(steering) > 1e-6 and gain > math.radians(2.)
                    and forward_turn_clear is not None):
                safe, reason = forward_turn_clear(candidate_path)
                if safe:
                    reason = 'wall_escape'
            return safe, reason
        safe, reason = checked_forward(path)
        full_reason = reason
        # Shorten wall- or vehicle-limited paths. Every shorter trajectory is rebuilt
        # at its stopping-distance speed and checked against walls AND traffic.
        if not safe and reason.startswith(('wall_overlap_', 'new_wall_contact_at_step=',
                                            'static_collision_at_step=', 'vehicle_collision=')):
            lengths = sorted({distance*.75, distance*.5, distance*.25,
                              distance*.125, minimum_distance}, reverse=True)
            for length in lengths:
                if length >= distance or length+1e-8 < minimum_distance:
                    continue
                prefix = trajectory(1, steering, length)
                safe, reason = checked_forward(prefix)
                if safe:
                    path = prefix
                    break
        if safe and key(1, steering) in excluded:
            if retry_clear is None or not retry_clear(1, steering, path):
                reasons.append(f'forward/{requested:+.3f}:measured_no_progress')
                continue
        end_x, end_y, _ = path[-1]
        improvement = start_distance-math.hypot(target[0]-end_x, target[1]-end_y)
        heading_gain = heading_error(yaw)-heading_error(path[-1][2])
        turning_back = (target_heading is not None and heading_error(yaw) > math.pi/4
                        and heading_gain > math.radians(2.))
        if safe and (improvement > .01 or reason == 'wall_escape' or turning_back):
            end_overlap = float(overlap(path[-1])) if overlap else 0.
            if not math.isfinite(end_overlap) or end_overlap < 0.:
                reasons.append(f'forward/{requested:+.3f}:invalid_overlap_metric')
                continue
            candidates.append(Motion(1, steering, path, improvement,
                initial_overlap-end_overlap, heading_gain, recovery_speed_limit(path)))
        else:
            reasons.append(f'forward/{requested:+.3f}:' + (full_reason if not safe else 'away_from_waypoint'))
    if candidates and allow_forward:
        if target_heading is not None and heading_error(yaw) > math.pi/4:
            continuing = [m for m in candidates if m.heading_improvement > math.radians(2.)
                          and m.steering*preferred_steering > 0.]
            if continuing:
                candidates = continuing
        # Physical clearance first; among equal clearance outcomes, restore
        # heading before optimizing WP distance. Safety checks above are hard gates.
        candidates.sort(key=lambda motion: (-motion.wall_reduction,
                                            -motion.heading_improvement,
                                            -motion.improvement, abs(motion.steering)))
        if connection_check is not None:
            # Try the best safe prefixes first. The callback checks the whole
            # prefix + endpoint MPC with one traffic time origin.
            paired_candidates = []
            for candidate in candidates:
                # End the preparatory turn early when it can already connect;
                # keep stopping distance and steering ramp in every short trial.
                for length in (1., .5):
                    if length >= distance or length < minimum_distance:
                        continue
                    prefix = trajectory(1, candidate.steering, length)
                    if failure_path_clear is not None and not failure_path_clear(prefix)[0]:
                        continue
                    if not clear(prefix)[0]:
                        continue
                    if key(1, candidate.steering) in excluded and (retry_clear is None
                            or not retry_clear(1, candidate.steering, prefix)):
                        continue
                    end = prefix[-1]
                    paired_candidates.append(Motion(1, candidate.steering, prefix,
                        start_distance-math.hypot(target[0]-end[0],target[1]-end[1]),
                        initial_overlap-float(overlap(end)) if overlap else 0.,
                        heading_error(yaw)-heading_error(end[2]), recovery_speed_limit(prefix)))
                    break
                else:
                    paired_candidates.append(candidate)
            for candidate in paired_candidates:
                accepted, why = connection_check(candidate)
                if accepted:
                    return candidate, 'forward_mpc_connection'
                reasons.append('connection:' + why)
            # Preserve safe wall escape if no full connection exists. It must
            # not create a lane admission or a normal-MPC handoff certificate.
        return candidates[0], ('forward_heading_recovery' if candidates[0].improvement <= .01
                               and candidates[0].heading_improvement > 0. else 'forward_to_waypoint')
    path = rollout(pose, -1, 0., distance, wheelbase)
    check_reverse = reverse_clear or clear
    if only_motion is None or only_motion[0] == -1:
        alternatives.append((-1, 0., path))
    safe, reverse_reason = (check_reverse(path) if (-1, 0) not in excluded and (only_motion is None or only_motion[0] == -1)
                            else (False, 'measured_no_progress'))
    if safe:
        return Motion(-1, 0., path, 0., speed_limit=recovery_speed_limit(path)), 'reverse: ' + '; '.join(reasons)
    # Stop the validated prefix before worsening overlap OR a new wall.
    # Re-run static and traffic checks; never skip a failing sample.
    if reverse_reason.startswith(('wall_overlap_', 'new_wall_contact_at_step=')):
        for end in range(len(path)-1, 1, -1):
            prefix = path[:end]
            length = math.hypot(prefix[-1][0]-pose[0], prefix[-1][1]-pose[1])
            if length+1e-8 < minimum_distance:
                break
            safe, reason = check_reverse(prefix)
            if safe and reason in ('wall_escape', 'clear'):
                return Motion(-1, 0., prefix, 0.,
                              speed_limit=recovery_speed_limit(prefix)), (
                    f'reverse_prefix={length:.2f}m; full_reverse={reverse_reason}')
    # Wall-only last resort: compare prefixes of all four motions together.
    # Require actual initial contact, terminal improvement and bounded depth;
    # the callback retains invalid-map and traffic checks.
    best = []
    if wall_escape_clear is not None and initial_overlap > 0.:
        for direction, steering, full_path in alternatives:
            if key(direction, steering) in excluded:
                continue
            for length in (.5, 1., 1.5, distance):
                if length > distance or length < minimum_distance:
                    continue
                prefix = trajectory(direction, steering, length, speed_cap=.5)
                if direction > 0 and failure_path_clear is not None and not failure_path_clear(prefix)[0]:
                    continue
                safe, _ = wall_escape_clear(prefix)
                if not safe:
                    continue
                end = prefix[-1]
                reduction = initial_overlap-float(overlap(end))
                if not math.isfinite(reduction) or reduction <= 1e-6:
                    continue
                gain = heading_error(yaw)-heading_error(end[2])
                best.append(Motion(direction, steering, prefix,
                    start_distance-math.hypot(target[0]-end[0], target[1]-end[1]),
                    reduction, gain, min(.5, recovery_speed_limit(prefix))))
        if best:
            best.sort(key=lambda m: (-m.wall_reduction, -m.heading_improvement, -m.improvement))
            return best[0], 'wall_escape_best_improvement'
    return None, '; '.join(reasons + [f'reverse={reverse_reason}'])


def failure_pose_changed(position, old_heading, old_overlap, x, y, heading, overlap):
    """A new measured pose may retry; do not clear failures while scraping deeper."""
    if position is None or not all(math.isfinite(v) for v in (x, y, overlap, old_overlap)):
        return False
    turned = (old_heading is not None and heading is not None
              and math.isfinite(old_heading) and math.isfinite(heading)
              and abs(math.atan2(math.sin(heading-old_heading),
                                 math.cos(heading-old_heading))) >= math.radians(10.))
    return (overlap <= old_overlap+1e-6
            and (math.hypot(x-position[0], y-position[1]) >= .5 or turned))


class ForwardProgress:
    """Measured progress persists across solver success and recovery handoffs."""

    def __init__(self):
        self.anchor = None
        self.failed_pose = None
        self.failed_overlap = 0.
        self.failed_heading = None
        self.last_stamp = None
        self.reason = ''

    @property
    def blocked(self):
        return self.failed_pose is not None

    def update(self, stamp, x, y, overlap, *, valid, forward, reverse=False, heading=None):
        if not valid or not all(math.isfinite(v) for v in (stamp, x, y, overlap)):
            self.anchor = None
            return
        if self.last_stamp is not None:
            if stamp < self.last_stamp:
                self.anchor = None
                self.last_stamp = stamp
                return
            if stamp == self.last_stamp:
                return
            if stamp-self.last_stamp > .5:
                self.anchor = None
        self.last_stamp = stamp
        if self.blocked:
            # Release the failed-pose veto, not the path safety checks. Solver
            # success alone still cannot release it at the same measured pose.
            distance = math.hypot(x-self.failed_pose[0], y-self.failed_pose[1])
            wall_improved = self.failed_overlap-overlap >= max(.01, .05*self.failed_overlap)
            changed = failure_pose_changed(self.failed_pose, self.failed_heading,
                                           self.failed_overlap, x, y, heading, overlap)
            if ((reverse and distance >= .5)
                    or (forward and distance >= .15 and wall_improved)
                    or changed):
                self.failed_pose = None
                self.reason = ''
                self.anchor = None
            return
        if not forward:
            self.anchor = None
            return
        if self.anchor is None:
            self.anchor = (stamp, x, y, overlap)
            return
        start, ax, ay, initial_overlap = self.anchor
        moved = math.hypot(x-ax, y-ay) >= .15
        wall_improved = (overlap <= 1e-6 or
                         initial_overlap-overlap >= max(.01, .05*initial_overlap))
        if moved and (initial_overlap <= 1e-6 or wall_improved):
            self.anchor = (stamp, x, y, overlap)
        elif stamp-start >= 1.5:
            self.failed_pose = (x, y)
            self.failed_overlap = overlap
            self.failed_heading = heading
            self.reason = 'no_measured_motion' if not moved else 'wall_overlap_not_improving'
            self.anchor = None


class RecoveryAttempts:
    """Stop repeating a recovery motion that does not improve real contact."""
    def __init__(self):
        self.anchor = None
        self.excluded = set()
        self.failure_position = None
        self.failure_overlap = 0.
        self.wall_failures = []
        self.improving_until = -math.inf
        self.improving_key = None
        self.failed_conditions = {}
        self.execution_condition = None

    @staticmethod
    def condition(steering, path, initial_steering):
        length = sum(math.hypot(b[0]-a[0], b[1]-a[1])
                     for a, b in zip(path, path[1:]))
        return (length, abs(steering-initial_steering))

    def retry_is_improved(self, direction, steering, path, initial_steering):
        """Called only AFTER the new path passes physical and traffic checks.

        Every failed condition remains a veto unless the new executable length
        or steering preparation improves materially. Time alone never retries.
        """
        key = (direction, 0 if abs(steering) < .01 else (1 if steering > 0 else -1))
        records = self.failed_conditions.get(key, ())
        if direction != 1 or not records or len(records) >= 32:
            return False
        length, error = self.condition(steering, path, initial_steering)
        return (math.isfinite(length) and math.isfinite(error)
                and all(length >= old_length+.25 or error <= old_error-.10
                        for old_length, old_error in records))

    def executing(self, direction, steering, path, initial_steering):
        key = (direction, 0 if abs(steering) < .01 else (1 if steering > 0 else -1))
        condition = self.condition(steering, path, initial_steering)
        # Keep the conditions at the beginning of the measured attempt. The
        # remaining horizon may shorten as the vehicle advances.
        if self.anchor is None or self.anchor[0] != key or self.execution_condition is None:
            self.execution_condition = (key, condition)
        self.excluded.discard(key)

    def remember_wall_return(self, pose, steering):
        """Record failed spatial poses, not an entire steering direction."""
        if not all(math.isfinite(v) for v in pose):
            return
        if any(math.hypot(pose[0]-old[0], pose[1]-old[1]) < .3
               and abs(math.atan2(math.sin(pose[2]-old[2]), math.cos(pose[2]-old[2])))
                   < math.radians(15.) for old in self.wall_failures):
            return
        self.wall_failures.append(tuple(pose))
        self.wall_failures = self.wall_failures[-16:]

    def wall_path_is_clear(self, path):
        """Reject returning within 30 cm/15 deg of a failed pose, for any steer.

        A path starting inside that region may leave it. Physical wall checks
        remain mandatory; starting near an old failure must not prevent escape.
        """
        if not path or not all(math.isfinite(v) for p in path for v in p):
            return False, 'invalid_failure_path'
        for old in self.wall_failures:
            def matches(p):
                return (math.hypot(p[0]-old[0], p[1]-old[1]) <= .3
                        and abs(math.atan2(math.sin(p[2]-old[2]), math.cos(p[2]-old[2])))
                            < math.radians(15.))
            left_region = not matches(path[0])
            for i, p in enumerate(path[1:], 1):
                inside = matches(p)
                if inside and left_region:
                    return False, f'remembered_wall_return_at_step={i}'
                left_region = left_region or not inside
        return True, 'clear'

    def observe(self, now, pose, overlap, direction, steering, commanded, *, commanded_speed=1.):
        if not all(math.isfinite(v) for v in (now, *pose, overlap)):
            self.anchor = None
            return
        if self.failure_position and failure_pose_changed(
                self.failure_position, self.failure_position[2], self.failure_overlap,
                *pose, overlap):
            self.excluded.clear()
            self.failed_conditions.clear()
            self.failure_position = None
        key = (direction, 0 if abs(steering) < .01 else (1 if steering > 0 else -1))
        if not commanded or direction == 0:
            self.anchor = None
            return
        if self.anchor is None or self.anchor[0] != key or now < self.anchor[1]:
            self.anchor = (key, now, pose, overlap)
            return
        _, start, old_pose, old_overlap = self.anchor
        # A short validated path may permit less than .1 m/s. Require half
        # of its commanded 1.5 s travel, capped at the normal 15 cm threshold.
        # Keep a 3 cm noise floor and the independent contact-improvement gate.
        required_travel = min(.15, max(.03, .75*abs(commanded_speed)))
        moved = math.hypot(pose[0]-old_pose[0], pose[1]-old_pose[1]) >= required_travel
        improved = old_overlap <= 1e-6 or overlap <= old_overlap-max(.01, .05*old_overlap)
        if moved and improved:
            self.improving_until = now+1.5
            self.improving_key = key
            self.execution_condition = None
            self.anchor = (key, now, pose, overlap)
        elif now-start >= 1.5:
            self.excluded.add(key)
            if self.execution_condition is not None and self.execution_condition[0] == key:
                self.failed_conditions.setdefault(key, []).append(self.execution_condition[1])
                self.failed_conditions[key] = self.failed_conditions[key][:32]
            self.failure_position = pose
            self.failure_overlap = overlap
            self.anchor = None


class MotionStart:
    """Arm stall recovery only after measured travel, without simulator state."""
    def __init__(self):
        self.origin = None
        self.previous = None

    def update(self, stamp, x, y, speed, valid):
        if not valid or not all(math.isfinite(v) for v in (stamp, x, y, speed)):
            self.origin = self.previous = None
            return False
        if self.previous is not None:
            t, px, py = self.previous
            if stamp == t:
                return False
            if stamp < t or stamp-t > .5 or math.hypot(x-px, y-py) > 1.:
                self.origin = None
        self.previous = (stamp, x, y)
        if self.origin is None:
            self.origin = (x, y)
            return False
        return (abs(speed) > .15
                and math.hypot(x-self.origin[0], y-self.origin[1]) >= .3)


class SteeringEstimate:
    """Rate-limited estimate from elapsed time and the last issued tire command."""
    def __init__(self):
        self.angle = None
        self.stamp = None

    def update(self, now, command, rate):
        if not all(math.isfinite(v) for v in (now, command, rate)) or rate <= 0.:
            self.angle = self.stamp = None
            return None
        if self.stamp is None or now < self.stamp:
            self.angle = command
        else:
            step = rate*(now-self.stamp)
            self.angle += max(-step, min(step, command-self.angle))
        self.stamp = now
        return self.angle
