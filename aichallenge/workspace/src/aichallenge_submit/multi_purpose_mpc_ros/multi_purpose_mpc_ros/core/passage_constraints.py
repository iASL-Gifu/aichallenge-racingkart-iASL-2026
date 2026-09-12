"""Read-only candidate corridor checks using the live MPC boundary builder.

This is a necessary geometric check, not a replacement for Shadow MPC or
time-dependent body collision checks. Artificial bounds constrain the vehicle
reference point; do not subtract a second full vehicle width from them.
"""
import copy

import numpy as np

from .MPC import apply_outer_boundary_guard, zero_inverted_bounds
from .reference_path import OUTER_COURSE_MARGIN, collapsed_constraint_snapshot
from .wall_constraints import wall_center_bounds


def corridor_inputs(controller, pose):
    path = controller._reference_pathN_center
    car = controller._carN_center
    state = getattr(car, 'temporal_state', None)
    if getattr(controller, '_reference_path', path) is path and state is not None:
        # The active model has already advanced through the steering delay.
        return int(car.wp_id), (float(state.x), float(state.y), float(state.psi))
    start = (car.wp_id if getattr(controller, '_reference_path', path) is path
             else car.get_closest_waypoint(pose.x, pose.y))
    return int(start), (float(pose.x), float(pose.y), float(getattr(pose, 'theta', 0.)))


def corridor_key(controller, pose):
    path = controller._reference_pathN_center
    mpc = controller._mpcN_center
    model = controller._cfg.bicycle_model
    road = path.map
    start, predicted_pose = corridor_inputs(controller, pose)
    return (
        (start, predicted_pose), id(path), id(road.data),
        getattr(road, 'revision', 0),
        getattr(controller, '_v2x_applied_generation', None),
        int(mpc.N), float(model.length), float(model.width),
        float(getattr(mpc, 'prediction_outer_boundary_guard', 0.)),
        int(getattr(mpc, 'lane_constraint_connection_points', 10)),
        getattr(path, 'n_lanes', 2), getattr(path, 'inner_lane_width', .5),
        tuple((wp.x, wp.y, wp.psi, wp.lb, wp.ub,
               tuple(tuple(cell) for cell in wp.static_border_cells))
              for wp in (path.get_waypoint(start + n) for n in range(int(mpc.N) + 1))),
    )


def preview_lane_corridor(path, start_wp, pose, horizon, length, width, lane,
                          *, guard=0., connection_points=10):
    snapshot = copy.copy(path)
    snapshot.waypoints = list(path.waypoints)
    for index in {(start_wp + n) % path.n_waypoints for n in range(1, horizon + 1)}:
        snapshot.waypoints[index] = copy.copy(path.waypoints[index])
    snapshot.target_lane_idx = lane
    snapshot.is_overtaking = True
    # Never certify a passage created by erasing an obstacle restriction.
    snapshot.unsafe_static_fallback_on_narrow = False
    snapshot.unsafe_static_fallback_wp_ids = []
    upper, lower, _ = snapshot.update_path_constraints(
        start_wp + 1, pose, horizon, length, width, 0.,
        lane_relaxation=0., connect_lane_from_current_pose=True,
        lane_connection_points=connection_points, lane_transition_weights=None)
    lower, upper = apply_outer_boundary_guard(lower, upper, lane, guard)
    lower, upper = zero_inverted_bounds(lower, upper)
    wp_ids = [(start_wp + n) % path.n_waypoints for n in range(1, horizon + 1)]
    failure = collapsed_constraint_snapshot(upper, lower, wp_ids)
    if failure is not None:
        return {**failure, 'reason': 'mpc_corridor_collapsed'}
    # Even at zero heading error the reference point must fit inside the
    # independent body-wall rows. Shadow MPC checks the actual yaw dynamics.
    wall_bounds = [wall_center_bounds(
        path.get_waypoint(wp).lb, path.get_waypoint(wp).ub,
        course_margin=OUTER_COURSE_MARGIN, half_width=.5 * width, guard=guard)
        for wp in wp_ids]
    wall_lower, wall_upper = np.asarray(wall_bounds).T
    failure = collapsed_constraint_snapshot(
        np.minimum(upper, wall_upper), np.maximum(lower, wall_lower), wp_ids)
    return None if failure is None else {**failure, 'reason': 'mpc_body_wall_corridor_collapsed'}


def candidate_corridor_failure(controller, pose, lane, geometry_key=None):
    work = getattr(controller, '_traffic_work', None)
    key = (corridor_key(controller, pose) if geometry_key is None else geometry_key, lane)
    if work is not None and key in work.corridors:
        return work.corridors[key]
    start, predicted_pose = corridor_inputs(controller, pose)
    model = controller._cfg.bicycle_model
    mpc = controller._mpcN_center
    failure = preview_lane_corridor(
        controller._reference_pathN_center, start, predicted_pose,
        int(mpc.N), float(model.length), float(model.width), lane,
        guard=float(getattr(mpc, 'prediction_outer_boundary_guard', 0.)),
        connection_points=int(getattr(mpc, 'lane_constraint_connection_points', 10)))
    if work is not None:
        work.corridors[key] = failure
    return failure
