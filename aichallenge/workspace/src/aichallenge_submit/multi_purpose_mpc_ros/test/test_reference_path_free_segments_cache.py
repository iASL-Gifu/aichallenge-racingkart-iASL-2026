"""Regression tests for prospective raw physical free-segment reuse."""

import copy
from types import MethodType, SimpleNamespace

import numpy as np

from multi_purpose_mpc_ros.core.reference_path import ReferencePath


def _path():
    path = ReferencePath.__new__(ReferencePath)
    path.n_waypoints = 8
    path.circular = True
    path.target_lane_idx = 2
    path.unsafe_static_fallback_on_narrow = False
    path.unsafe_static_fallback_wp_ids = []
    path.waypoints = [
        SimpleNamespace(
            x=float(index), y=0.0, psi=0.0,
            ub=2.0, lb=-2.0, ub_sm=2.0, lb_sm=-2.0,
            static_border_cells=(
                (float(index), 2.0), (float(index), -2.0)),
            dynamic_border_cells=None,
        )
        for index in range(path.n_waypoints)
    ]
    path.border_cells = SimpleNamespace()
    path.map = SimpleNamespace()

    path.get_waypoint = MethodType(
        lambda self, index: self.waypoints[index % self.n_waypoints], path)
    path.get_lane_bounds = MethodType(
        lambda self, _index: [
            (-0.5, -2.0), (0.5, -0.5), (2.0, 0.5)], path)

    def compute_free_segments(self, wp, _min_width, **_kwargs):
        return [((wp.x, 2.0), (wp.x, -2.0))]

    path._compute_free_segments = MethodType(compute_free_segments, path)
    return path


def _update(path, precomputed=None):
    return path.update_path_constraints(
        1,
        [0.0, 0.0, 0.0],
        3,
        2.0,
        1.0,
        0.0,
        lane_relaxation=0.2,
        precomputed_free_segments_hor=precomputed,
    )


def test_precomputed_raw_free_segments_produce_identical_bounds():
    uncached_path = _path()
    uncached_upper, uncached_lower, _ = _update(uncached_path)

    cached_path = _path()
    raw = [
        cached_path._compute_free_segments(
            cached_path.get_waypoint(1 + index),
            1.0,
            wp_idx=1 + index,
        )
        for index in range(3)
    ]
    raw_before = [list(segments) for segments in raw]
    cached_upper, cached_lower, _ = _update(cached_path, raw)

    np.testing.assert_array_equal(cached_upper, uncached_upper)
    np.testing.assert_array_equal(cached_lower, uncached_lower)
    assert raw == raw_before


def _multi_segment_path():
    path = _path()
    path.target_lane_idx = None
    path.map = SimpleNamespace(
        w2m=lambda x, y: (int(round(x)), int(round(y))),
        resolution=1.0,
    )
    path.collision_scan_count = 0

    def compute_free_segments(self, wp, _min_width, **_kwargs):
        return [
            ((wp.x, 2.0), (wp.x, 0.25)),
            ((wp.x, -0.25), (wp.x, -2.0)),
        ]

    def obstacle_occupied(self, *_args, **_kwargs):
        self.collision_scan_count += 1
        return False

    path._compute_free_segments = MethodType(compute_free_segments, path)
    path._is_obstacle_occupied = MethodType(obstacle_occupied, path)
    return path


def _dynamic_update(
    path,
    cache_enabled,
    precompute_geometry=True,
    shared_transition_cache=None,
    shared_transition_stats=None,
    shared_bound_cache=None,
    shared_bound_stats=None,
    precomputed_free_segments=None,
    lane_relaxation=0.0,
):
    dynamic_v2x_by_step = [
        [SimpleNamespace(cx=100.0, cy=100.0, radius=0.5)]
        for _ in range(3)
    ]
    return path.update_path_constraints(
        1,
        [0.0, 0.0, 0.0],
        3,
        2.0,
        1.0,
        0.0,
        dynamic_v2x_by_step=dynamic_v2x_by_step,
        static_occupancy_data=np.ones((8, 8), dtype=np.int8),
        lane_relaxation=lane_relaxation,
        combination_collision_cache_enabled=cache_enabled,
        precompute_combination_dynamic_geometry=precompute_geometry,
        precomputed_combination_collision_cache=shared_transition_cache,
        shared_combination_collision_cache_stats=shared_transition_stats,
        precomputed_segment_bound_cache=shared_bound_cache,
        shared_precomputed_segment_bound_cache_stats=shared_bound_stats,
        precomputed_free_segments_hor=precomputed_free_segments,
    )


def test_combination_collision_cache_preserves_dynamic_v2x_result():
    uncached_path = _multi_segment_path()
    uncached_upper, uncached_lower, uncached_cells = _dynamic_update(
        uncached_path, False)

    cached_path = _multi_segment_path()
    cached_upper, cached_lower, cached_cells = _dynamic_update(
        cached_path, True)

    np.testing.assert_array_equal(cached_upper, uncached_upper)
    np.testing.assert_array_equal(cached_lower, uncached_lower)
    np.testing.assert_array_equal(cached_cells, uncached_cells)
    for field in (
        "hard_lb", "hard_ub", "lane_lb", "lane_ub", "final_lb", "final_ub",
    ):
        np.testing.assert_array_equal(
            getattr(cached_path.last_constraint_bounds, field),
            getattr(uncached_path.last_constraint_bounds, field),
        )
    assert cached_path.select_free_segs == uncached_path.select_free_segs
    assert cached_path.collision_scan_count < uncached_path.collision_scan_count


def test_combination_dynamic_geometry_preserves_constraint_result():
    legacy_path = _multi_segment_path()
    legacy_upper, legacy_lower, legacy_cells = _dynamic_update(
        legacy_path, True, False)

    precomputed_path = _multi_segment_path()
    precomputed_upper, precomputed_lower, precomputed_cells = _dynamic_update(
        precomputed_path, True, True)

    np.testing.assert_array_equal(precomputed_upper, legacy_upper)
    np.testing.assert_array_equal(precomputed_lower, legacy_lower)
    np.testing.assert_array_equal(precomputed_cells, legacy_cells)
    for field in (
        "hard_lb", "hard_ub", "lane_lb", "lane_ub", "final_lb", "final_ub",
    ):
        np.testing.assert_array_equal(
            getattr(precomputed_path.last_constraint_bounds, field),
            getattr(legacy_path.last_constraint_bounds, field),
        )
    assert precomputed_path.select_free_segs == legacy_path.select_free_segs
    assert precomputed_path.upper_cols == legacy_path.upper_cols


def _prospective_profile_results(shared):
    path = _multi_segment_path()
    transition_cache = {} if shared else None
    transition_stats = {"hits": 0, "misses": 0}
    results = []
    for lane_relaxation in (0.0, 0.2, 0.4):
        upper, lower, border_cells = _dynamic_update(
            path,
            True,
            shared_transition_cache=transition_cache,
            shared_transition_stats=(transition_stats if shared else None),
            lane_relaxation=lane_relaxation,
        )
        results.append({
            "upper": upper.copy(),
            "lower": lower.copy(),
            "border_cells": border_cells.copy(),
            "selected_combination": list(path.select_free_segs),
            "free_segments": list(path.free_segs),
            "upper_cols": list(path.upper_cols),
            "last_constraint_bounds": {
                field: getattr(path.last_constraint_bounds, field).copy()
                for field in (
                    "hard_lb", "hard_ub", "lane_lb", "lane_ub",
                    "final_lb", "final_ub",
                )
            },
        })
    return path, results, transition_stats


def test_shared_profile_transition_cache_preserves_all_constraint_results():
    isolated_path, isolated_results, _ = _prospective_profile_results(False)
    shared_path, shared_results, shared_stats = (
        _prospective_profile_results(True)
    )

    assert len(shared_results) == len(isolated_results)
    for shared, isolated in zip(shared_results, isolated_results):
        np.testing.assert_array_equal(shared["upper"], isolated["upper"])
        np.testing.assert_array_equal(shared["lower"], isolated["lower"])
        np.testing.assert_array_equal(
            shared["border_cells"], isolated["border_cells"])
        assert (
            shared["selected_combination"]
            == isolated["selected_combination"]
        )
        assert shared["free_segments"] == isolated["free_segments"]
        assert shared["upper_cols"] == isolated["upper_cols"]
        for field in shared["last_constraint_bounds"]:
            np.testing.assert_array_equal(
                shared["last_constraint_bounds"][field],
                isolated["last_constraint_bounds"][field],
            )

    assert shared_stats["hits"] > 0
    assert shared_path.collision_scan_count < isolated_path.collision_scan_count


def _prospective_bound_cache_results(cache_enabled):
    path = _multi_segment_path()
    path.target_lane_idx = 2
    precomputed_free_segments = [
        path._compute_free_segments(
            path.get_waypoint(1 + index),
            1.0,
            wp_idx=1 + index,
        )
        for index in range(3)
    ]
    bound_cache = {} if cache_enabled else None
    bound_stats = {"hits": 0, "misses": 0}
    results = []
    for lane_relaxation in (0.0, 0.2, 0.4):
        upper, lower, border_cells = _dynamic_update(
            path,
            True,
            shared_bound_cache=bound_cache,
            shared_bound_stats=(bound_stats if cache_enabled else None),
            precomputed_free_segments=precomputed_free_segments,
            lane_relaxation=lane_relaxation,
        )
        results.append({
            "upper": upper.copy(),
            "lower": lower.copy(),
            "border_cells": border_cells.copy(),
            "selected_combination": list(path.select_free_segs),
            "free_segments": list(path.free_segs),
            "last_constraint_bounds": {
                field: getattr(path.last_constraint_bounds, field).copy()
                for field in (
                    "hard_lb", "hard_ub", "lane_lb", "lane_ub",
                    "final_lb", "final_ub",
                )
            },
        })
    return results, bound_stats


def test_shared_precomputed_bound_cache_preserves_profile_results():
    uncached_results, _ = _prospective_bound_cache_results(False)
    cached_results, cache_stats = _prospective_bound_cache_results(True)

    assert len(cached_results) == len(uncached_results)
    for cached, uncached in zip(cached_results, uncached_results):
        np.testing.assert_array_equal(cached["upper"], uncached["upper"])
        np.testing.assert_array_equal(cached["lower"], uncached["lower"])
        np.testing.assert_array_equal(
            cached["border_cells"], uncached["border_cells"])
        assert (
            cached["selected_combination"]
            == uncached["selected_combination"]
        )
        assert cached["free_segments"] == uncached["free_segments"]
        for field in cached["last_constraint_bounds"]:
            np.testing.assert_array_equal(
                cached["last_constraint_bounds"][field],
                uncached["last_constraint_bounds"][field],
            )

    assert cache_stats["misses"] == 12
    assert cache_stats["hits"] == 24


def _copy_equivalence_path():
    path = ReferencePath.__new__(ReferencePath)
    path.n_waypoints = 8
    path.circular = True
    path.n_lanes = 3
    path.inner_lane_width = 0.5
    path.target_lane_idx = 2
    path.unsafe_static_fallback_on_narrow = False
    path.unsafe_static_fallback_wp_ids = []
    path.waypoints = [
        SimpleNamespace(
            x=float(index), y=0.0, psi=0.0,
            ub=2.0, lb=-2.0, ub_sm=2.0, lb_sm=-2.0,
            static_border_cells=(
                (float(index), 2.0), (float(index), -2.0)),
            dynamic_border_cells=None,
        )
        for index in range(path.n_waypoints)
    ]
    path.border_cells = SimpleNamespace()
    path.map = SimpleNamespace(
        w2m=lambda x, y: (int(round(x)), int(round(y))),
        resolution=1.0,
    )
    path._is_obstacle_occupied = MethodType(
        lambda _self, *_args, **_kwargs: False,
        path,
    )
    return path


def _copied_profile_results(copy_all_waypoints):
    source_path = _copy_equivalence_path()
    first_wp_id = 6
    horizon = 7
    precomputed_free_segments = [
        [
            (
                (source_path.get_waypoint(first_wp_id + index).x, 2.0),
                (source_path.get_waypoint(first_wp_id + index).x, 0.25),
            ),
            (
                (source_path.get_waypoint(first_wp_id + index).x, -0.25),
                (source_path.get_waypoint(first_wp_id + index).x, -2.0),
            ),
        ]
        for index in range(horizon)
    ]
    dynamic_v2x_by_step = [[] for _ in range(horizon)]
    results = []
    final_diagnostic_path = None
    for lane_relaxation in (0.0, 0.2, 0.4):
        diagnostic_path = copy.copy(source_path)
        if copy_all_waypoints:
            diagnostic_path.waypoints = [
                copy.copy(waypoint)
                for waypoint in source_path.waypoints
            ]
        else:
            diagnostic_path.waypoints = list(source_path.waypoints)
            copied_indices = set()
            for horizon_index in range(horizon):
                waypoint_index = (
                    first_wp_id + horizon_index
                ) % source_path.n_waypoints
                if waypoint_index in copied_indices:
                    continue
                diagnostic_path.waypoints[waypoint_index] = copy.copy(
                    source_path.waypoints[waypoint_index])
                copied_indices.add(waypoint_index)
        diagnostic_path.border_cells = copy.deepcopy(
            source_path.border_cells)
        diagnostic_path.unsafe_static_fallback_wp_ids = list(
            source_path.unsafe_static_fallback_wp_ids)

        upper, lower, border_cells = (
            diagnostic_path.update_path_constraints(
                first_wp_id,
                [5.0, 0.0, 0.0],
                horizon,
                2.0,
                1.0,
                0.0,
                lane_relaxation=lane_relaxation,
                dynamic_v2x_by_step=dynamic_v2x_by_step,
                static_occupancy_data=np.ones((16, 16), dtype=np.int8),
                precomputed_free_segments_hor=precomputed_free_segments,
            )
        )
        results.append({
            "upper": upper.copy(),
            "lower": lower.copy(),
            "border_cells": border_cells.copy(),
            "selected_free_segments": list(
                diagnostic_path.select_free_segs),
            "free_segments": list(diagnostic_path.free_segs),
            "last_constraint_bounds": {
                field: getattr(
                    diagnostic_path.last_constraint_bounds, field).copy()
                for field in (
                    "hard_lb", "hard_ub", "lane_lb", "lane_ub",
                    "final_lb", "final_ub",
                )
            },
        })
        final_diagnostic_path = diagnostic_path
    return source_path, final_diagnostic_path, results


def test_horizon_only_waypoint_copy_matches_full_copy_across_wraparound():
    _, _, full_copy_results = _copied_profile_results(True)
    source_path, horizon_path, horizon_copy_results = (
        _copied_profile_results(False)
    )

    assert len(horizon_copy_results) == len(full_copy_results)
    for horizon_result, full_result in zip(
        horizon_copy_results, full_copy_results
    ):
        np.testing.assert_array_equal(
            horizon_result["upper"], full_result["upper"])
        np.testing.assert_array_equal(
            horizon_result["lower"], full_result["lower"])
        np.testing.assert_array_equal(
            horizon_result["border_cells"], full_result["border_cells"])
        assert (
            horizon_result["selected_free_segments"]
            == full_result["selected_free_segments"]
        )
        assert horizon_result["free_segments"] == full_result["free_segments"]
        for field in horizon_result["last_constraint_bounds"]:
            np.testing.assert_array_equal(
                horizon_result["last_constraint_bounds"][field],
                full_result["last_constraint_bounds"][field],
            )

    copied_indices = {6, 7, 0, 1, 2, 3, 4}
    for index in range(source_path.n_waypoints):
        if index in copied_indices:
            assert horizon_path.waypoints[index] is not source_path.waypoints[index]
        else:
            assert horizon_path.waypoints[index] is source_path.waypoints[index]
    assert all(
        waypoint.ub_sm == 2.0
        and waypoint.lb_sm == -2.0
        and waypoint.dynamic_border_cells is None
        for waypoint in source_path.waypoints
    )
