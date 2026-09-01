"""Regression tests for diagnostic-only prospective outer corridors."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from multi_purpose_mpc_ros.mpc_controller import MPCController


class _DiagnosticReferencePath:
    last_call_kwargs = None
    free_segment_calls = 0
    precomputed_ids = []
    transition_cache_ids = []
    bound_cache_ids = []
    copied_waypoint_index_sets = []
    def __init__(self, obstacle_map):
        self.map = obstacle_map
        self.n_waypoints = 40
        self.inner_lane_width = 0.5
        self.target_lane_idx = None
        self.circular = True
        self.waypoints = [
            SimpleNamespace(
                x=float(index), y=0.0, psi=0.0, kappa=0.0,
                lb=-3.0, ub=3.0, lb_sm=-3.0, ub_sm=3.0,
                dynamic_border_cells=None,
            )
            for index in range(self.n_waypoints)
        ]
        self.source_waypoints = tuple(self.waypoints)
        self.border_cells = SimpleNamespace()
        self.unsafe_static_fallback_wp_ids = []

    def get_lane_bounds(self, _index):
        return [(-0.5, -2.5), (0.5, -0.5), (2.5, 0.5)]

    def get_waypoint(self, index):
        return self.waypoints[int(index) % self.n_waypoints]

    def _compute_free_segments(self, _wp, _min_width, **_kwargs):
        type(self).free_segment_calls += 1
        return [((0.0, 1.0), (0.0, -1.0))]

    def update_path_constraints(
        self, _wp_id, _pose, horizon, _length, _width, _safety_margin,
        **_kwargs,
    ):
        type(self).copied_waypoint_index_sets.append({
            index for index, waypoint in enumerate(self.waypoints)
            if waypoint is not self.source_waypoints[index]
        })
        type(self).last_call_kwargs = _kwargs
        type(self).precomputed_ids.append(
            id(_kwargs.get("precomputed_free_segments_hor")))
        type(self).transition_cache_ids.append(
            id(_kwargs.get("precomputed_combination_collision_cache")))
        type(self).bound_cache_ids.append(
            id(_kwargs.get("precomputed_segment_bound_cache")))
        # The fake keeps the test focused on the controller's diagnostic
        # plumbing: the production ReferencePath owns obstacle intersection.
        if self.map.obstacles:
            lower = np.full(horizon, -0.2 if self.target_lane_idx == 0 else 0.1)
            upper = np.full(horizon, 0.1 if self.target_lane_idx == 0 else 0.4)
        elif self.target_lane_idx == 0:
            lower, upper = np.full(horizon, -2.5), np.full(horizon, -0.5)
        else:
            lower, upper = np.full(horizon, 0.5), np.full(horizon, 2.5)
        return upper, lower, None


def _controller(*, with_obstacle=False):
    _DiagnosticReferencePath.free_segment_calls = 0
    _DiagnosticReferencePath.precomputed_ids = []
    _DiagnosticReferencePath.transition_cache_ids = []
    _DiagnosticReferencePath.bound_cache_ids = []
    _DiagnosticReferencePath.copied_waypoint_index_sets = []
    controller = MPCController.__new__(MPCController)
    obstacle_map = SimpleNamespace(
        obstacles=(
            [SimpleNamespace(cx=3.0, cy=0.0, radius=0.7)]
            if with_obstacle else []
        ))
    reference_path = _DiagnosticReferencePath(obstacle_map)
    controller._map = obstacle_map
    controller._reference_pathN_center = reference_path
    controller._carN_center = SimpleNamespace(
        get_closest_waypoint=lambda x, y: int(round(x)),
        spatial_state=SimpleNamespace(e_y=0.0),
    )
    controller._mpcN_center = SimpleNamespace(
        N=4,
        model=SimpleNamespace(length=2.0, width=1.6, safety_margin=0.2),
        lane_constraint_retry_relaxation_m=(0.0, 0.2),
        lane_constraint_retry_relax_toward_center_only=True,
        lane_constraint_retry_taper_over_horizon=True,
        lane_constraint_retry_terminal_ratio=0.0,
        lane_constraint_connection_points=2,
        prediction_outer_boundary_guard=0.0,
    )
    controller._cfg = SimpleNamespace(
        bicycle_model=SimpleNamespace(width=1.6))
    controller._v2x_t_samples = [0.0, 0.1]
    controller._v2x_vehicle_radius = 0.7
    controller._v2x_tracker = SimpleNamespace(
        predict_all=lambda times: (
            {"d3": [(3.0, 0.0) for _ in times]} if with_obstacle else {}))
    controller._outer_candidate_guidance = None
    return controller, reference_path


def test_prospective_corridor_uses_candidate_constraint_grid_when_validated():
    controller, reference_path = _controller(with_obstacle=False)
    times = [0.1, 0.3, 0.7, 1.2]
    by_step = [
        [SimpleNamespace(
            cx=3.0 + time, cy=0.0, radius=0.7,
            vehicle_id="d3", prediction_time=time,
        )]
        for time in times
    ]
    controller._validated_candidate_used_time_grid = (
        lambda vehicle_id, lane_idx: times)
    controller._time_aligned_v2x_by_step = lambda grid: by_step
    controller._static_obstacle_map_data = np.ones((2, 2), dtype=np.int8)
    controller._static_obstacles = []

    result = controller._prospective_outer_corridor_feasibility(
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        lane_idx=2,
        vehicle_id="d3",
    )

    assert result["time_aligned"] is True
    assert result["constraint_times"] == times
    assert _DiagnosticReferencePath.last_call_kwargs[
        "dynamic_v2x_by_step"] is by_step
    assert _DiagnosticReferencePath.last_call_kwargs[
        "static_occupancy_data"] is controller._static_obstacle_map_data
    assert _DiagnosticReferencePath.free_segment_calls == controller._mpcN_center.N
    assert len(set(_DiagnosticReferencePath.precomputed_ids)) == 1
    assert len(set(_DiagnosticReferencePath.transition_cache_ids)) == 1
    assert len(set(_DiagnosticReferencePath.bound_cache_ids)) == 1


def test_prospective_corridor_copies_only_wrapped_horizon_waypoints():
    controller, reference_path = _controller(with_obstacle=False)
    controller._carN_center.get_closest_waypoint = lambda _x, _y: 38

    controller._prospective_outer_corridor_feasibility(
        pose=SimpleNamespace(x=38.0, y=0.0, theta=0.0),
        lane_idx=2,
    )

    assert _DiagnosticReferencePath.copied_waypoint_index_sets
    assert all(
        copied_indices == {39, 0, 1, 2}
        for copied_indices in
        _DiagnosticReferencePath.copied_waypoint_index_sets
    )
    assert all(
        waypoint is reference_path.source_waypoints[index]
        for index, waypoint in enumerate(reference_path.waypoints)
    )


@pytest.mark.parametrize("lane_idx", [0, 2])
def test_dynamic_obstacle_marks_both_outer_corridors_would_block(lane_idx):
    controller, _ = _controller(with_obstacle=True)

    result = controller._prospective_outer_corridor_feasibility(
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        lane_idx=lane_idx)

    assert result["min_width"] == pytest.approx(0.3)
    assert result["required_width"] == pytest.approx(1.6)
    assert result["below_required_count"] == 4
    assert result["would_block"] is True
    assert result["limiting_vehicle_id"] == "d3"


@pytest.mark.parametrize("lane_idx", [0, 2])
def test_clear_outer_corridor_uses_existing_width_and_does_not_block(lane_idx):
    controller, reference_path = _controller(with_obstacle=False)
    original_target = reference_path.target_lane_idx
    original_waypoint_bound = reference_path.waypoints[0].lb_sm

    result = controller._prospective_outer_corridor_feasibility(
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        lane_idx=lane_idx)

    assert result["base_width_min"] == pytest.approx(2.0)
    assert result["min_width"] == pytest.approx(2.0)
    assert result["required_width"] == pytest.approx(1.6)
    assert result["below_required_count"] == 0
    assert result["would_block"] is False
    # The diagnostic copy must not alter live lane ownership or path state.
    assert reference_path.target_lane_idx is original_target
    assert reference_path.waypoints[0].lb_sm == original_waypoint_bound


@pytest.mark.parametrize("lane_idx", [0, 2])
def test_would_block_promotes_existing_diagnostic_to_gate(lane_idx):
    controller, _ = _controller(with_obstacle=True)
    logger = SimpleNamespace(info=Mock(), warn=Mock())
    controller.get_logger = lambda: logger

    blocked = controller._outer_commit_corridor_would_block(
        source="normal",
        vehicle_id="d3",
        lane_idx=lane_idx,
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
    )

    assert blocked is True
    message = logger.warn.call_args.args[0]
    assert "[OuterCommitCorridorBlocked]" in message
    assert "action=exclude_new_outer_commit" in message


def test_clear_prospective_corridor_is_transparent_to_commit_gate():
    controller, _ = _controller(with_obstacle=False)
    logger = SimpleNamespace(info=Mock(), warn=Mock())
    controller.get_logger = lambda: logger

    assert controller._outer_commit_corridor_would_block(
        source="normal",
        vehicle_id="d3",
        lane_idx=0,
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
    ) is False
    logger.warn.assert_not_called()


def test_all_new_outer_commit_sources_apply_gate_without_committed_maintenance():
    control_source = inspect.getsource(MPCController._control)

    assert control_source.count("_outer_commit_corridor_would_block(") == 5
    assert 'source="normal"' in control_source
    assert 'source="preempt"' in control_source
    assert 'source="retry"' in control_source
    assert 'source="fallback"' in control_source
    committed_source = inspect.getsource(
        MPCController._committed_lane_predicted_envelope_conflict)
    assert "_outer_commit_corridor_would_block" not in committed_source


def test_normal_gate_keeps_opposite_candidate_and_no_candidate_when_both_block():
    passage = {0: True, 2: True}
    conflicts = {
        0: {"front": [], "side": [], "rear": []},
        2: {"front": [], "side": [], "rear": []},
    }

    # This is the same existing selector called after a blocked lane is set
    # false in _control: one blocked side exposes the safe opposite.
    passage[0] = False
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import select_safe_outer_lane
    assert select_safe_outer_lane(0, passage, conflicts) == 2

    passage[2] = False
    assert select_safe_outer_lane(2, passage, conflicts) is None
