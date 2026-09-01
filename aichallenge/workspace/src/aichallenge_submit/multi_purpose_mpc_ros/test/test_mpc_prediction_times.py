"""Tests for time alignment of spatial MPC world predictions."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.mpc_controller import MPCController


class _ReferencePath:
    @staticmethod
    def get_waypoint(index):
        return SimpleNamespace(x=float(index), y=0.0, psi=0.0)


class _Model:
    reference_path = _ReferencePath()
    wp_id = 10

    @staticmethod
    def s2t(waypoint, state):
        return SimpleNamespace(x=waypoint.x, y=float(state[0]))


def _actual_prediction_controller():
    controller = MPCController.__new__(MPCController)
    controller._mpc = SimpleNamespace(
        model=SimpleNamespace(reference_path=object()),
        current_prediction=([0.0, 1.0], [0.0, 0.0]),
        current_prediction_times=[0.25, 0.9],
        infeasibility_counter=0,
        used_prediction_fallback=False,
        recovery_requested=False,
        time_budget_exceeded=False,
        last_solution_accurate=True,
    )
    controller._v2x_tracker = SimpleNamespace(
        active_vehicle_ids=lambda: ["d2"],
        _samples={"d2": [(0.0, 20.0, 0.0)]},
        velocity=lambda vehicle_id: (1.0, 0.0),
        has_velocity_estimate=lambda vehicle_id: True,
    )
    controller._cfg = SimpleNamespace(
        bicycle_model=SimpleNamespace(width=1.0),
        mpc=SimpleNamespace(prediction_outer_boundary_guard=0.1),
    )
    controller._v2x_vehicle_radius = 0.5
    controller._moving_vehicle_brake_bypass_min_speed = 0.1
    controller._moving_vehicle_brake_bypass_confirm_sec = 0.1
    controller._moving_vehicle_brake_bypass_horizon_sec = 2.0
    controller._reference_path = SimpleNamespace(
        target_lane_idx=2,
        is_overtaking=True,
        get_waypoint=lambda index: SimpleNamespace(
            x=0.0, y=0.0, psi=0.0, lb=-2.0, ub=2.0),
    )
    controller._car = SimpleNamespace(get_closest_waypoint=lambda x, y: 0)
    controller._center_longitudinal_between = (
        lambda x0, y0, x1, y1: float(x1) - float(x0))
    controller._parallel_ego_half_length = 1.0
    controller._parallel_vehicle_half_length = 1.0
    controller._parallel_critical_clearance = 0.0
    controller.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    return controller


def test_world_prediction_and_times_use_identical_spatial_steps():
    mpc = MPC.__new__(MPC)
    mpc.model = _Model()
    states = np.array([
        [0.0, 0.0, 5.0],
        [0.1, 0.0, 5.1],
        [0.2, 0.0, 5.3],
        [0.3, 0.0, 5.7],
        [0.4, 0.0, 6.2],
    ])

    prediction, times = mpc.update_prediction_with_times(states, 5)

    assert prediction[0] == [12.0, 13.0, 14.0]
    assert prediction[1] == pytest.approx([0.2, 0.3, 0.4])
    assert times == pytest.approx([0.3, 0.7, 1.2])
    assert all(later >= earlier for earlier, later in zip(times, times[1:]))


def test_constraint_prediction_times_match_constraint_indices():
    states = np.zeros((23, 3), dtype=float)
    states[:, 2] = np.linspace(5.0, 8.3, 23)

    times = MPC.constraint_prediction_times(states, 22)

    assert len(times) == 22
    for index in (0, 5, 10, 15, 21):
        assert times[index] == pytest.approx(
            states[index + 1, 2] - states[0, 2])
    assert times[0] >= 0.0
    assert all(later >= earlier for earlier, later in zip(times, times[1:]))


def test_candidate_prediction_rebase_keeps_zero_age_unchanged():
    result = MPCController._rebase_candidate_prediction(
        [1.0, 2.0], [3.0, 4.0], [0.2, 0.8], 0.0)

    assert result == ([1.0, 2.0], [3.0, 4.0], [0.2, 0.8], 0)


def test_candidate_prediction_rebase_slices_xyz_time_at_same_indices():
    result = MPCController._rebase_candidate_prediction(
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.1, 0.4, 0.9], 0.3)

    assert result[0] == [2.0, 3.0]
    assert result[1] == [5.0, 6.0]
    assert result[2] == pytest.approx([0.1, 0.6])
    assert result[3] == 1
    assert all(value >= 0.0 for value in result[2])


def test_candidate_prediction_rebase_rejects_no_future_sample():
    assert MPCController._rebase_candidate_prediction(
        [1.0, 2.0], [3.0, 4.0], [0.1, 0.2], 0.3) is None


def test_constraint_grid_rebase_preserves_indices_and_rejects_expired_grid():
    assert MPCController._rebase_candidate_constraint_grid(
        [0.1, 0.4, 0.9], 0.3) == pytest.approx([0.0, 0.1, 0.6])
    assert MPCController._rebase_candidate_constraint_grid(
        [0.1, 0.2], 0.3) is None


def test_committed_envelope_uses_mpc_derived_prediction_times():
    controller = MPCController.__new__(MPCController)
    controller._mpc = SimpleNamespace(
        current_prediction=([1.0, 2.0], [0.0, 0.0]),
        current_prediction_times=[0.4, 1.1],
        infeasibility_counter=0,
        used_prediction_fallback=False,
        recovery_requested=False,
        time_budget_exceeded=False,
        last_solution_accurate=True,
    )
    controller._v2x_tracker = SimpleNamespace(
        active_vehicle_ids=lambda: [], _samples={})
    controller._center_frenet = lambda x, y: (x, y)
    controller._center_arc_total_length = 100.0
    controller._cfg = SimpleNamespace(
        bicycle_model=SimpleNamespace(width=1.0))
    controller._v2x_parallel_vehicle_half_width = 0.5
    controller._parallel_ego_half_length = 1.0
    controller._parallel_vehicle_half_length = 1.0

    with patch(
        "multi_purpose_mpc_ros.mpc_controller."
        "minimum_predicted_envelope_conflict",
        return_value=None,
    ) as conflict_check:
        result = controller._committed_lane_predicted_envelope_conflict(
            SimpleNamespace(x=0.0, y=0.0))

    assert result is None
    assert conflict_check.call_args.args[2] == [0.4, 1.1]


def test_missing_or_misaligned_times_rejects_committed_prediction():
    controller = MPCController.__new__(MPCController)
    controller._mpc = SimpleNamespace(
        current_prediction=([1.0, 2.0], [0.0, 0.0]),
        current_prediction_times=[0.4],
        infeasibility_counter=0,
        used_prediction_fallback=False,
        recovery_requested=False,
        time_budget_exceeded=False,
        last_solution_accurate=True,
    )
    controller._v2x_tracker = SimpleNamespace(active_vehicle_ids=lambda: [])

    assert controller._committed_lane_predicted_envelope_conflict(
        SimpleNamespace(x=0.0, y=0.0)) is None


@pytest.mark.parametrize("helper_name", [
    "_prediction_is_clear_of_vehicle",
    "_moving_vehicle_will_clear_after_brief_conflict",
    "_overtake_prediction_is_clear",
])
def test_actual_moving_vehicle_proofs_use_mpc_derived_times(helper_name):
    controller = _actual_prediction_controller()
    with patch(
        "multi_purpose_mpc_ros.mpc_controller."
        "prediction_clears_moving_vehicle",
        return_value=True,
    ) as clear_check:
        if helper_name == "_moving_vehicle_will_clear_after_brief_conflict":
            result = getattr(controller, helper_name)(
                "d2", SimpleNamespace(x=0.0, y=0.0, theta=0.0), 0.0)
        else:
            result = getattr(controller, helper_name)("d2")

    assert result is True
    assert clear_check.call_args.args[2] == [0.25, 0.9]


def test_wall_v2x_controlled_pass_uses_mpc_derived_times():
    controller = _actual_prediction_controller()
    with patch(
        "multi_purpose_mpc_ros.mpc_controller."
        "prediction_clears_moving_vehicle",
        return_value=True,
    ) as clear_check:
        controller._wall_v2x_controlled_pass_is_safe(
            SimpleNamespace(x=0.0, y=0.0),
            "d2",
            mpc_fresh_accurate=True,
        )

    assert clear_check.call_args.args[2] == [0.25, 0.9]


@pytest.mark.parametrize("invalid_update", [
    {"current_prediction_times": [0.25]},
    {"current_prediction_times": [0.9, 0.25]},
    {"used_prediction_fallback": True},
    {"recovery_requested": True},
    {"time_budget_exceeded": True},
    {"last_solution_accurate": False},
    {"infeasibility_counter": 1},
])
def test_actual_prediction_proof_rejects_invalid_or_nonfresh_state(
    invalid_update,
):
    controller = _actual_prediction_controller()
    for name, value in invalid_update.items():
        setattr(controller._mpc, name, value)

    assert controller._prediction_is_clear_of_vehicle("d2") is False


def test_stationary_vehicle_result_is_independent_of_prediction_time_spacing():
    controller = _actual_prediction_controller()
    controller._v2x_tracker.velocity = lambda vehicle_id: (0.0, 0.0)
    controller._v2x_tracker._samples["d2"] = [(0.0, 20.0, 0.0)]

    first = controller._prediction_is_clear_of_vehicle("d2")
    controller._mpc.current_prediction_times = [0.05, 2.0]
    second = controller._prediction_is_clear_of_vehicle("d2")

    assert first is True
    assert second is True


def test_candidate_prediction_uses_reachable_mpc_time_model():
    controller = _actual_prediction_controller()
    controller._mpc.current_prediction = ([0.2, 0.9], [0.0, 0.0])
    controller._mpc.current_prediction_times = [0.2, 1.5]
    controller._mpc.N = 4
    controller._mpc.current_constraint_prediction_times = [0.1, 0.2, 1.5, 1.7]
    controller._v2x_t_samples = [0.0, 0.1, 0.4, 1.0]
    controller._reference_pathN_center = SimpleNamespace(
        get_waypoint=lambda index: SimpleNamespace(
            x=float(index), y=0.0, psi=0.0),
        get_lane_bounds=lambda index: [(0.0, -1.0), (1.0, 0.0), (2.0, 1.0)],
    )
    controller._carN_center = SimpleNamespace(
        get_closest_waypoint=lambda x, y: int(round(x)))
    controller._center_frenet = lambda x, y: (x, y)
    controller._loop = 2
    controller._outer_candidate_guidance = {
        "vehicle_id": "d2",
        "lane_idx": 2,
        "source": "normal",
        "prediction_loop": 1,
        "candidate_state_time_sec": 10.0,
        "mpc_id": id(controller._mpc),
        "prediction_id": id(controller._mpc.current_prediction),
        "constraint_time_grid_used": (0.08, 0.18, 1.4, 1.6),
        "constraint_time_grid_result": (0.1, 0.2, 1.5, 1.7),
        "candidate_solve_time_aligned": True,
        "reference_path_id": id(controller._mpc.model.reference_path),
        "grid_source_loop": 0,
        "grid_source_mpc_id": id(controller._mpc),
        "grid_source_reference_path_id": id(
            controller._mpc.model.reference_path),
        "grid_source_prediction_id": 1,
    }

    _, _, times, samples = controller._candidate_lane_center_prediction(
        SimpleNamespace(x=0.0, y=0.0, theta=0.0), 1.0, 2, "d2")

    assert times == [0.2, 1.5]
    assert samples[0]["e_y"] == pytest.approx(0.0)
    assert samples[1]["e_y"] == pytest.approx(0.0)
    assert all("transition_progress" not in sample for sample in samples)


def _candidate_constraint_grid_controller():
    controller = _actual_prediction_controller()
    controller._loop = 11
    controller._mpc.N = 4
    controller._mpc.current_constraint_prediction_times = [0.1, 0.25, 0.9, 1.2]
    controller._outer_candidate_guidance = {
        "vehicle_id": "d2",
        "lane_idx": 2,
        "source": "preempt",
        "guidance_loop": 7,
        "prediction_loop": 10,
        "candidate_state_time_sec": 10.0,
        "mpc_id": id(controller._mpc),
        "prediction_id": id(controller._mpc.current_prediction),
        "constraint_time_grid_used": (0.08, 0.2, 0.8, 1.1),
        "constraint_time_grid_result": (0.1, 0.25, 0.9, 1.2),
        "candidate_solve_time_aligned": True,
        "reference_path_id": id(controller._mpc.model.reference_path),
        "grid_source_loop": 9,
        "grid_source_mpc_id": id(controller._mpc),
        "grid_source_reference_path_id": id(
            controller._mpc.model.reference_path),
        "grid_source_prediction_id": 1,
    }
    controller._v2x_tracker.active_vehicle_ids = lambda: ["d2", "d3"]
    controller._v2x_tracker._samples["d3"] = [(0.0, 5.0, 4.0)]
    velocities = {"d2": (2.0, 0.0), "d3": (0.0, 0.0)}
    controller._v2x_tracker.velocity = lambda vehicle_id: velocities[vehicle_id]
    controller._v2x_tracker.predict_positions = lambda vehicle_id, times: [
        (
            controller._v2x_tracker._samples[vehicle_id][-1][1]
            + velocities[vehicle_id][0] * time,
            controller._v2x_tracker._samples[vehicle_id][-1][2]
            + velocities[vehicle_id][1] * time,
        )
        for time in times
    ]
    controller._filter_obstacles_to_corridor = lambda obstacles: obstacles
    return controller


def test_time_aligned_v2x_uses_only_vehicle_position_at_each_step():
    controller = _candidate_constraint_grid_controller()
    times = controller._validated_candidate_constraint_time_grid("d2", 2)

    by_step = controller._time_aligned_v2x_by_step(times)

    assert len(by_step) == 4
    assert all({ob.vehicle_id for ob in step} == {"d2", "d3"}
               for step in by_step)
    d2_positions = [
        next(ob.cx for ob in step if ob.vehicle_id == "d2")
        for step in by_step
    ]
    assert d2_positions == pytest.approx([20.2, 20.5, 21.8, 22.4])
    assert all(
        len([ob for ob in step if ob.vehicle_id == "d2"]) == 1
        for step in by_step
    )
    d3_positions = [
        (next(ob.cx for ob in step if ob.vehicle_id == "d3"),
         next(ob.cy for ob in step if ob.vehicle_id == "d3"))
        for step in by_step
    ]
    assert d3_positions == [(5.0, 4.0)] * 4


def test_selected_constraint_indices_share_time_waypoint_and_v2x_position():
    controller = _candidate_constraint_grid_controller()
    times = [0.05 * (index + 1) for index in range(22)]
    controller._mpc.N = 22
    controller._mpc.current_prediction = (
        [float(index) for index in range(20)], [0.0] * 20)
    controller._mpc.current_prediction_times = times[1:-1]
    controller._outer_candidate_guidance.update(
        prediction_id=id(controller._mpc.current_prediction),
        constraint_time_grid_result=tuple(times),
    )

    validated = controller._validated_candidate_constraint_time_grid("d2", 2)
    by_step = controller._time_aligned_v2x_by_step(validated)

    base_waypoint = 100
    for index in (0, 5, 10, 15, 21):
        d2 = next(ob for ob in by_step[index] if ob.vehicle_id == "d2")
        assert d2.prediction_time == pytest.approx(times[index])
        assert d2.cx == pytest.approx(20.0 + 2.0 * times[index])
        assert base_waypoint + 1 + index == 101 + index


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: setattr(c, "_loop", 12),
        lambda c: c._outer_candidate_guidance.update(vehicle_id="d3"),
        lambda c: c._outer_candidate_guidance.update(lane_idx=0),
        lambda c: setattr(c._mpc, "used_prediction_fallback", True),
        lambda c: setattr(c._mpc, "recovery_requested", True),
        lambda c: setattr(c._mpc, "time_budget_exceeded", True),
        lambda c: setattr(c._mpc, "last_solution_accurate", False),
        lambda c: setattr(c._mpc, "infeasibility_counter", 1),
    ],
)
def test_candidate_constraint_grid_rejects_stale_or_invalid_ownership(mutation):
    controller = _candidate_constraint_grid_controller()
    mutation(controller)

    assert controller._validated_candidate_constraint_time_grid("d2", 2) is None


def _candidate_transaction_controller(source="normal"):
    controller = MPCController.__new__(MPCController)
    reference_path = object()
    prediction = ([1.0, 2.0], [0.0, 0.1])
    controller._loop = 10
    controller._reference_path = reference_path
    controller._reference_pathN_center = reference_path
    controller._mpc = SimpleNamespace(
        N=4,
        model=SimpleNamespace(reference_path=reference_path),
        current_prediction=prediction,
        current_prediction_times=[0.2, 0.7],
        current_constraint_prediction_times=[0.1, 0.2, 0.7, 1.1],
        infeasibility_counter=0,
        used_prediction_fallback=False,
        recovery_requested=False,
        time_budget_exceeded=False,
        last_solution_accurate=True,
        dynamic_v2x_by_step=None,
        static_occupancy_data=None,
    )
    controller._mpcN_center = controller._mpc
    controller._outer_candidate_guidance = {
        "vehicle_id": "d3",
        "lane_idx": 2,
        "source": source,
        "guidance_loop": 9,
        "last_guidance_alpha": 0.2,
        "ready_logged": True,
        "prediction_loop": None,
        "candidate_state_time_sec": None,
        "prediction_id": None,
        "mpc_id": None,
        "reference_path_id": None,
        "constraint_time_grid_used": None,
        "constraint_time_grid_result": None,
        "candidate_solve_time_aligned": False,
    }
    controller._latest_full_width_time_grid = {
        "control_loop": 9,
        "mpc_id": id(controller._mpc),
        "reference_path_id": id(reference_path),
        "controller_reference_path_id": id(reference_path),
        "prediction_id": id(prediction),
        "constraint_times": (0.1, 0.2, 0.7, 1.1),
    }
    controller._static_obstacle_map_data = np.ones((2, 2), dtype=np.int8)
    controller._v2x_vehicle_radius = 0.5
    controller._v2x_tracker = SimpleNamespace(
        active_vehicle_ids=lambda: ["d2", "d3"],
        predict_positions=lambda vehicle_id, times: [
            (float(index) + time, 0.0 if vehicle_id == "d2" else 2.0)
            for index, time in enumerate(times)
        ],
    )
    controller._filter_obstacles_to_corridor = lambda obstacles: obstacles
    controller._center_frenet = lambda x, y: (float(x), float(y))
    controller.get_logger = lambda: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    controller.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    return controller


@pytest.mark.parametrize("source", ["normal", "fallback", "preempt"])
def test_candidate_solve_uses_previous_fresh_grid_and_keeps_result_separate(
    source,
):
    controller = _candidate_transaction_controller(source)

    transaction = controller._activate_outer_candidate_time_aligned_v2x(
        guidance_applied=True, applied_lane_idx=None)

    assert transaction["grid_source_loop"] == 9
    assert transaction["constraint_times"] == pytest.approx(
        [0.1, 0.2, 0.7, 1.1])
    assert len(controller._mpc.dynamic_v2x_by_step) == 4
    assert all(len(step) == 2 for step in controller._mpc.dynamic_v2x_by_step)

    # Simulate the solve result. Its true trajectory times deliberately differ
    # from the prior grid actually used for dynamic bounds.
    controller._mpc.current_prediction = ([1.1, 2.1], [0.1, 0.2])
    controller._mpc.current_prediction_times = [0.25, 0.8]
    controller._mpc.current_constraint_prediction_times = [
        0.12, 0.25, 0.8, 1.25]
    controller._capture_outer_candidate_guided_prediction(
        guidance_applied=True,
        applied_lane_idx=None,
        time_aligned_transaction=transaction,
        candidate_state_time_sec=10.0,
    )

    record = controller._outer_candidate_guidance
    assert record["candidate_solve_time_aligned"] is True
    assert record["constraint_time_grid_used"] == pytest.approx(
        [0.1, 0.2, 0.7, 1.1])
    assert record["constraint_time_grid_result"] == pytest.approx(
        [0.12, 0.25, 0.8, 1.25])
    assert record["constraint_time_grid_used"] != record[
        "constraint_time_grid_result"]


def test_flattened_warmup_prediction_cannot_authorize_commit():
    controller = _candidate_transaction_controller()
    controller._latest_full_width_time_grid = None

    transaction = controller._activate_outer_candidate_time_aligned_v2x(
        guidance_applied=True, applied_lane_idx=None)
    assert transaction is None

    controller._capture_outer_candidate_guided_prediction(
        guidance_applied=True,
        applied_lane_idx=None,
        time_aligned_transaction=None,
        candidate_state_time_sec=10.0,
    )
    controller._capture_latest_full_width_time_grid(
        full_width_ownership=True)
    controller._loop = 11

    assert controller._outer_candidate_guidance[
        "candidate_solve_time_aligned"] is False
    assert controller._outer_candidate_guided_prediction_ready("d3", 2) is False
    next_transaction = controller._activate_outer_candidate_time_aligned_v2x(
        guidance_applied=True, applied_lane_idx=None)
    assert next_transaction is not None
    assert next_transaction["constraint_times"] == pytest.approx(
        [0.1, 0.2, 0.7, 1.1])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c._latest_full_width_time_grid.update(control_loop=8),
        lambda c: c._latest_full_width_time_grid.update(mpc_id=-1),
        lambda c: c._latest_full_width_time_grid.update(reference_path_id=-1),
        lambda c: c._latest_full_width_time_grid.update(
            controller_reference_path_id=-1),
        lambda c: c._latest_full_width_time_grid.update(prediction_id=-1),
        lambda c: setattr(c._mpc, "used_prediction_fallback", True),
        lambda c: setattr(c._mpc, "recovery_requested", True),
        lambda c: setattr(c._mpc, "time_budget_exceeded", True),
        lambda c: setattr(c._mpc, "last_solution_accurate", False),
        lambda c: setattr(c._mpc, "infeasibility_counter", 1),
    ],
)
def test_candidate_prior_grid_rejects_stale_or_invalid_ownership(mutation):
    controller = _candidate_transaction_controller()
    mutation(controller)

    assert controller._activate_outer_candidate_time_aligned_v2x(
        guidance_applied=True, applied_lane_idx=None) is None


def test_candidate_commit_transaction_reuses_used_not_result_grid():
    controller = _candidate_transaction_controller()
    transaction = controller._activate_outer_candidate_time_aligned_v2x(
        guidance_applied=True, applied_lane_idx=None)
    controller._mpc.current_prediction = ([1.1, 2.1], [0.1, 0.2])
    controller._mpc.current_prediction_times = [0.25, 0.8]
    controller._mpc.current_constraint_prediction_times = [
        0.12, 0.25, 0.8, 1.25]
    controller._capture_outer_candidate_guided_prediction(
        guidance_applied=True,
        applied_lane_idx=None,
        time_aligned_transaction=transaction,
        candidate_state_time_sec=10.0,
    )
    controller._loop = 11
    controller._mpcN_center.soft_target_lane_idx = 2
    controller._mpcN_center.soft_target_start_e_y = 0.0
    controller._mpcN_center.soft_target_alpha = 0.2
    controller._mpcN_center.soft_target_lateral_offset = 0.0
    controller._mpcN_center.soft_lateral_targets = None
    controller._outer_candidate_reference_handoff = None
    controller._outer_commit_time_aligned_v2x = None

    assert controller._capture_outer_candidate_reference_handoff(
        vehicle_id="d3", lane_idx=2, source="normal")
    assert controller._outer_commit_time_aligned_v2x[
        "constraint_times"] == pytest.approx([0.1, 0.2, 0.7, 1.1])
    assert controller._outer_commit_time_aligned_v2x[
        "grid_source"] == "candidate_used_grid"


@pytest.mark.parametrize("source", ["normal", "fallback", "preempt"])
def test_candidate_commit_sources_activate_same_time_aligned_grid(source):
    controller = _candidate_constraint_grid_controller()
    controller._target_lane_idx = 2
    controller._static_obstacle_map_data = np.ones((3, 3), dtype=np.int8)
    controller.get_logger = lambda: SimpleNamespace(
        info=lambda *args, **kwargs: None)
    controller._outer_commit_time_aligned_v2x = {
        "source": source,
        "vehicle_id": "d2",
        "lane_idx": 2,
        "guidance_loop": 7,
        "prediction_loop": 10,
        "candidate_state_time_sec": 10.0,
        "mpc_id": id(controller._mpc),
        "prediction_id": id(controller._mpc.current_prediction),
        "commit_loop": 11,
        "constraint_times": (0.1, 0.25, 0.9, 1.2),
    }

    assert controller._activate_outer_commit_time_aligned_v2x() is True
    assert len(controller._mpc.dynamic_v2x_by_step) == controller._mpc.N
    for index, expected_time in enumerate((0.1, 0.25, 0.9, 1.2)):
        assert all(
            obstacle.prediction_time == pytest.approx(expected_time)
            for obstacle in controller._mpc.dynamic_v2x_by_step[index]
        )
    assert controller._mpc.static_occupancy_data is (
        controller._static_obstacle_map_data)


def test_invalid_commit_grid_preserves_conservative_flatten_mode():
    controller = _candidate_constraint_grid_controller()
    controller._target_lane_idx = 2
    controller._outer_commit_time_aligned_v2x = None
    controller._mpc.dynamic_v2x_by_step = ["old"]
    controller._mpc.static_occupancy_data = object()

    assert controller._activate_outer_commit_time_aligned_v2x() is False
    assert controller._mpc.dynamic_v2x_by_step is None
    assert controller._mpc.static_occupancy_data is None
