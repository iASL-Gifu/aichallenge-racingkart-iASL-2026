"""Exact, bounded collision work reuse, owned by one controller cycle.

Only immutable body/time/geometry values are keys. No rounded coordinates,
solver identities, or previous-cycle admission are used as safety evidence.
The controller refreshes this object with each frozen observation cycle.
"""
from types import SimpleNamespace
from .. import collision_geometry as collision


class PathCheckWork:
    def __init__(self, diagnostics=None):
        self.interpolated = {}
        self.bodies = {}
        self.body_points = {}
        self.wall = {}
        self.traffic = {}
        self.diagnostics = diagnostics

    def cached(self, cache, key, label, compute, limit=128):
        hit = key in cache
        diag = self.diagnostics
        if diag is not None and diag.recording_detail():
            diag.counts['path_check_' + label + ('_hits' if hit else '_misses')] += 1
        if hit:
            return cache[key]
        result = compute()
        if len(cache) >= limit:
            cache.clear()
        cache[key] = result
        return result


def wall_margin(controller):
    config = getattr(getattr(controller, "_cfg", None), "mpc", None)
    return float(getattr(config, "prediction_outer_boundary_guard", .05))


def wall_clear(controller, bodies, geometry, *, allowance=0., slide=False):
    grid = controller._map
    margin = wall_margin(controller)
    def compute():
        if allowance == 0. and not slide and margin == .05:
            return grid.static_recovery_path_is_clear(bodies, geometry)
        return grid.static_recovery_path_is_clear(
            bodies, geometry, temporary_depth_increase=allowance,
            recovery_contact_slide=slide, wall_margin=margin)
    work = getattr(controller, '_path_check_work', None)
    if work is None:
        return compute()
    # data_backup is the static map; its in-place edits must begin a new cycle.
    key = (id(grid), id(grid.data_backup), grid.resolution, tuple(grid.origin),
           grid.width, grid.height, tuple(bodies), geometry, allowance, slide, margin)
    return work.cached(work.wall, key, 'wall', compute)


def traffic_clear(controller, bodies, times, target, velocity, geometry, *, reverse=False):
    bodies, times, velocity = tuple(bodies), tuple(times), tuple(velocity)
    def compute():
        return (collision.swept_path_clear(bodies, times, target, velocity, geometry)
                or collision.separating_path_clear(
                    bodies, times, target, velocity, geometry, reverse=reverse))
    work = getattr(controller, '_path_check_work', None)
    if work is None:
        return compute()
    key = (bodies, times, target, velocity, geometry, reverse)
    return work.cached(work.traffic, key, 'traffic', compute)


def prepare_bodies(controller, path):
    path = tuple(tuple(p) for p in path)
    def compute():
        return tuple(collision.ego_body(controller, SimpleNamespace(x=x,y=y,theta=yaw))
                     for x,y,yaw in path)
    work = getattr(controller, '_path_check_work', None)
    if work is None:
        return compute()
    # Every input consulted by ego_body, including origin/alignment and validity.
    metadata = getattr(controller, '_collision_ego_metadata', None)
    alignment = getattr(controller, '_collision_ego_alignment', None)
    key = (path, getattr(controller, '_collision_now', 0.),
           tuple(metadata) if metadata is not None else None,
           tuple(alignment) if alignment is not None else None,
           getattr(controller, '_collision_ego_origin', 'unconfirmed'),
           getattr(controller, '_collision_center_offset', .522),
           getattr(controller, '_collision_origin_lateral_margin', None))
    def shared_compute():
        return tuple(work.cached(work.body_points, (point,key[1:]), 'body_point',
                     lambda point=point: collision.ego_body(controller, SimpleNamespace(
                         x=point[0],y=point[1],theta=point[2])), limit=4096) for point in path)
    return work.cached(work.bodies, key, 'body_prepare', shared_compute)


def prepare_mpc_path(controller, mpc, pose, delay=0.):
    """Reuse interpolation, not admission; mutated arrays invalidate by value."""
    import numpy as np
    from .control_continuity import timed_mpc_path
    from .runtime_diagnostics import detail_scope
    with detail_scope(controller, 'path_check.interpolation'):
        points = np.asarray(getattr(mpc, 'current_recovery_prediction', None), dtype=float)
        times = np.asarray(getattr(mpc, 'current_prediction_times', None), dtype=float)
        key = (points.shape, points.tobytes(), times.shape, times.tobytes(),
               float(pose.x), float(pose.y), float(pose.theta), float(delay))
        def compute():
            result = timed_mpc_path(mpc, pose, delay)
            return None if result is None else (tuple(result[0]), tuple(result[1]))
        work = getattr(controller, '_path_check_work', None)
        return compute() if work is None else work.cached(
            work.interpolated, key, 'interpolation', compute)
