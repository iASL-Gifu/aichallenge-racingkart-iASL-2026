"""Regression tests for solved pre-commit trajectory safety."""

import ast
from pathlib import Path

from types import SimpleNamespace

import pytest

from multi_purpose_mpc_ros.mpc_controller import MPCController
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    minimum_predicted_envelope_conflict,
)


def _controller_method_ast(method_name):
    source_path = (
        Path(__file__).resolve().parents[1]
        / "multi_purpose_mpc_ros"
        / "mpc_controller.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    controller_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MPCController"
    )
    return next(
        node for node in controller_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_control_time_is_assigned_before_every_candidate_guidance_use():
    control = _controller_method_ast("_control")
    assignment_lines = [
        node.lineno
        for node in ast.walk(control)
        if isinstance(node, ast.Name)
        and node.id == "current_time_sec"
        and isinstance(node.ctx, ast.Store)
    ]
    use_lines = [
        node.lineno
        for node in ast.walk(control)
        if isinstance(node, ast.Name)
        and node.id == "current_time_sec"
        and isinstance(node.ctx, ast.Load)
    ]

    assert len(assignment_lines) == 1
    assert use_lines
    assert assignment_lines[0] < min(use_lines)


def test_stop_reuses_control_with_early_time_initialization():
    stop = _controller_method_ast("stop")

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_control"
        for node in ast.walk(stop)
    )


def _controller(lane_idx, vehicle_y, vehicle_x=2.0):
    controller = MPCController.__new__(MPCController)
    controller._mpc = SimpleNamespace(
        N=5,
        model=SimpleNamespace(reference_path=object()),
        current_prediction=([2.0, 6.0, 15.0], [0.6, 0.6, 0.6]),
        current_prediction_times=[0.2, 0.6, 1.5],
        current_constraint_prediction_times=[0.1, 0.2, 0.6, 1.5, 2.0],
        infeasibility_counter=0,
        used_prediction_fallback=False,
        recovery_requested=False,
        time_budget_exceeded=False,
        last_solution_accurate=True,
        previous_steering=0.0,
    )
    controller._reference_pathN_center = SimpleNamespace(
        get_waypoint=lambda index: SimpleNamespace(
            x=float(index), y=0.0, psi=0.0),
        get_lane_bounds=lambda index: [
            (-1.39, -2.39), (0.5, -0.5), (2.39, 1.39)],
    )
    controller._carN_center = SimpleNamespace(
        get_closest_waypoint=lambda x, y: int(round(x)),
        spatial_state=SimpleNamespace(e_y=0.6),
        wp_id=0,
    )
    controller._mpcN_center = SimpleNamespace(
        N=2,
        _compute_lane_center=lambda wp, lane: -1.5 if lane == 0 else 1.5,
    )
    controller._l1_soft_rejoin_ramp_sec = 1.0
    controller._race_handoff_max_reference_speed = 1.0
    controller.get_logger = lambda: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    controller.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    controller._center_frenet = lambda x, y: (float(x), float(y))
    controller._center_arc_total_length = 1000.0
    controller._v2x_tracker = SimpleNamespace(
        active_vehicle_ids=lambda: ["d3"],
        _samples={"d3": [(0.0, vehicle_x, vehicle_y)]},
        velocity=lambda vehicle_id: (0.0, 0.0),
    )
    controller._cfg = SimpleNamespace(
        bicycle_model=SimpleNamespace(width=1.0, length=2.0))
    controller._v2x_parallel_vehicle_half_width = 0.5
    controller._parallel_ego_half_length = 1.0
    controller._parallel_vehicle_half_length = 1.0
    controller._v2x_t_samples = [0.0, 0.2, 0.6, 1.5]
    controller._slow_lead_speed_match_release_lateral_error = 0.45
    controller._candidate_envelope_diagnostic_cache = {}
    controller._loop = 1
    controller._outer_candidate_guidance = {
        "vehicle_id": "d3",
        "lane_idx": lane_idx,
        "source": "normal",
        "guidance_loop": 0,
        "prediction_loop": 0,
        "candidate_state_time_sec": 10.0,
        "mpc_id": id(controller._mpc),
        "prediction_id": id(controller._mpc.current_prediction),
        "constraint_time_grid_used": (0.08, 0.18, 0.55, 1.4, 1.9),
        "constraint_time_grid_result": (0.1, 0.2, 0.6, 1.5, 2.0),
        "candidate_solve_time_aligned": True,
        "reference_path_id": id(controller._mpc.model.reference_path),
        "grid_source_loop": -1,
        "grid_source_mpc_id": id(controller._mpc),
        "grid_source_reference_path_id": id(
            controller._mpc.model.reference_path),
        "grid_source_prediction_id": 1,
    }
    pose = SimpleNamespace(x=0.0, y=0.6, theta=0.0)
    return controller, pose, lane_idx


@pytest.mark.parametrize(
    "lane_idx,vehicle_y",
    [(2, 0.8), (0, 0.4)],
)
def test_continuous_transition_detects_conflict_hidden_by_lane_center_snap(
    lane_idx, vehicle_y,
):
    controller, pose, lane_idx = _controller(lane_idx, vehicle_y)

    conflict = controller._candidate_lane_predicted_envelope_conflict(
        pose, 10.0, lane_idx, "d3")

    assert conflict is not None
    assert conflict["vehicle_id"] == "d3"
    assert conflict["prediction_time"] == pytest.approx(0.2)
    assert conflict["predicted_lateral_clearance"] < 0.0
    assert conflict["predicted_longitudinal_clearance"] < 0.0


@pytest.mark.parametrize("lane_idx", [0, 2])
def test_clear_continuous_outer_transition_remains_safe(lane_idx):
    controller, pose, lane_idx = _controller(
        lane_idx, vehicle_y=10.0, vehicle_x=2.0)

    conflict = controller._candidate_lane_predicted_envelope_conflict(
        pose, 10.0, lane_idx, "d3")

    assert conflict is None


def test_candidate_envelope_uses_rebased_times_with_latest_v2x_snapshot():
    controller, pose, lane_idx = _controller(2, vehicle_y=10.0)
    controller._outer_candidate_guidance["candidate_state_time_sec"] = 9.7
    observed = {}

    def capture_conflict(_x, _y, times, moving_vehicles, **_kwargs):
        observed["times"] = list(times)
        observed["vehicles"] = list(moving_vehicles)
        return None

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "multi_purpose_mpc_ros.mpc_controller."
            "minimum_predicted_envelope_conflict",
            capture_conflict,
        )
        assert controller._candidate_lane_predicted_envelope_conflict(
            pose, 10.0, lane_idx, "d3") is None

    assert observed["times"] == pytest.approx([0.3, 1.2])
    assert observed["vehicles"] == [("d3", 2.0, 10.0, 0.0, 0.0)]


def test_candidate_transition_starts_at_pose_and_covers_existing_mpc_horizon():
    controller, pose, lane_idx = _controller(2, vehicle_y=10.0)

    candidate_x, candidate_y, times, samples = (
        controller._candidate_lane_center_prediction(
        pose, 10.0, lane_idx, "d3")
    )

    assert candidate_x == pytest.approx([2.0, 6.0, 15.0])
    assert candidate_y == pytest.approx([0.6, 0.6, 0.6])
    assert times == pytest.approx([0.2, 0.6, 1.5])
    assert samples[0]["x"] == pytest.approx(2.0)
    assert samples[0]["y"] == pytest.approx(0.6)
    assert samples[0]["e_y"] == pytest.approx(0.6)
    assert samples[-1]["time"] > 1.0
    assert all("transition_progress" not in sample for sample in samples)


def test_soft_transition_duration_does_not_move_solved_prediction_to_lane():
    controller, pose, lane_idx = _controller(2, vehicle_y=10.0)

    _, candidate_y, times, samples = (
        controller._candidate_lane_center_prediction(
            pose, 10.0, lane_idx, "d3")
    )

    assert 0.6 in times
    assert candidate_y[times.index(0.6)] == pytest.approx(0.6)
    assert samples[times.index(0.6)]["e_y"] == pytest.approx(0.6)


def test_missing_fresh_prediction_does_not_create_artificial_proof():
    controller, pose, lane_idx = _controller(2, vehicle_y=10.0)
    controller._mpc.used_prediction_fallback = True

    assert controller._candidate_lane_center_prediction(
        pose, 10.0, lane_idx, "d3") is None
    assert controller._candidate_lane_predicted_envelope_conflict(
        pose, 10.0, lane_idx, "d3") is None


@pytest.mark.parametrize(
    "lane_idx,terminal_y,expected_error,expected_fit",
    [
        # Latest-run d2/L2 and d3/L0 failures.
        (2, -0.56, 2.45, False),
        (0, 0.02, 1.91, False),
        # Successful d3/L2 control.
        (2, 1.62, 0.27, True),
        # L0 symmetry of the successful control.
        (0, -1.62, 0.27, True),
    ],
)
def test_candidate_terminal_fit_uses_outer_lane_error(
    lane_idx, terminal_y, expected_error, expected_fit,
):
    controller, pose, _ = _controller(lane_idx, vehicle_y=10.0)
    controller._mpc.current_prediction = (
        [2.0, 6.0, 15.0], [0.6, 0.2, terminal_y])
    controller._outer_candidate_guidance["prediction_id"] = id(
        controller._mpc.current_prediction)

    result = controller._candidate_guided_prediction_terminal_fit(
        pose, 10.0, lane_idx, "d3")

    assert result is not None
    assert result["terminal_lane_error"] == pytest.approx(expected_error)
    assert result["allowed_error"] == pytest.approx(0.45)
    assert result["fit"] is expected_fit


def test_candidate_terminal_fit_fails_closed_without_fresh_prediction():
    controller, pose, lane_idx = _controller(2, vehicle_y=10.0)
    controller._mpc.used_prediction_fallback = True

    assert controller._candidate_guided_prediction_terminal_fit(
        pose, 10.0, lane_idx, "d3") is None


@pytest.mark.parametrize(
    "vehicle_y,terminal_y",
    [(0.8, 0.6), (10.0, 1.62)],
)
def test_shared_geometry_frenet_cache_preserves_candidate_safety_and_fit(
    vehicle_y, terminal_y,
):
    def evaluate(cache_enabled):
        controller, pose, lane_idx = _controller(2, vehicle_y=vehicle_y)
        controller._mpc.current_prediction = (
            [2.0, 6.0, 15.0], [0.6, 0.6, terminal_y])
        controller._outer_candidate_guidance["prediction_id"] = id(
            controller._mpc.current_prediction)
        timing = {
            "prediction_prepare_ms": 0.0,
            "geometry_prepare_ms": 0.0,
            "dynamic_v2x_prepare_ms": 0.0,
            "envelope_collision_ms": 0.0,
            "terminal_fit_ms": 0.0,
        }
        geometry_cache = {} if cache_enabled else None
        collision_timing = {}
        conflict = controller._candidate_lane_predicted_envelope_conflict(
            pose, 10.0, lane_idx, "d3", timing=timing,
            collision_timing=collision_timing,
            geometry_frenet_cache=geometry_cache,
        )
        terminal_fit = controller._candidate_guided_prediction_terminal_fit(
            pose, 10.0, lane_idx, "d3", timing=timing,
            geometry_frenet_cache=geometry_cache,
        )
        return {
            "conflict": conflict,
            "terminal_fit": terminal_fit,
            "samples": controller._candidate_envelope_diagnostic_cache[
                lane_idx]["samples"],
            "allow_commit": bool(
                conflict is None
                and terminal_fit is not None
                and terminal_fit["fit"]
            ),
            "timing": timing,
            "collision_timing": collision_timing,
        }

    uncached = evaluate(False)
    cached = evaluate(True)

    assert cached["conflict"] == uncached["conflict"]
    assert cached["terminal_fit"] == uncached["terminal_fit"]
    assert cached["samples"] == uncached["samples"]
    assert cached["allow_commit"] is uncached["allow_commit"]
    assert cached["timing"]["geometry_frenet_cache_misses"] == 3
    assert cached["timing"]["geometry_frenet_cache_hits"] == 4
    assert cached["timing"]["geometry_center_frenet_call_count"] == 3
    assert cached["collision_timing"][
        "external_candidate_frenet_cache_hits"] == 3
    assert cached["collision_timing"][
        "external_candidate_frenet_cache_misses"] == 0
    assert cached["collision_timing"]["center_frenet_call_count"] == 3


def test_all_new_outer_commit_sources_use_terminal_fit_gate():
    control = _controller_method_ast("_control")
    fit_calls = [
        node for node in ast.walk(control)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_candidate_guided_prediction_terminal_fit"
    ]
    fit_checks = [
        node for node in ast.walk(control)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id.endswith("terminal_fit")
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "fit"
    ]
    logged_sources = {
        keyword.value.value
        for node in ast.walk(control)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_log_outer_candidate_trajectory_fit"
        for keyword in node.keywords
        if keyword.arg == "source"
        and isinstance(keyword.value, ast.Constant)
    }

    assert len(fit_calls) == 4
    assert len(fit_checks) == 4
    assert logged_sources == {"normal", "preempt", "retry", "fallback"}


def test_terminal_fit_helper_does_not_touch_failure_history():
    method = _controller_method_ast(
        "_candidate_guided_prediction_terminal_fit")
    mutated_attributes = {
        node.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
    }

    assert not {
        "_prepass_short_failed_outer_lanes",
        "_prepass_failed_lane_idx",
        "_prepass_attempted_outer_lanes",
    } & mutated_attributes


def test_terminal_fit_does_not_use_guidance_alpha_as_a_threshold():
    method = _controller_method_ast(
        "_candidate_guided_prediction_terminal_fit")

    assert all(
        "alpha" not in node.id
        for node in ast.walk(method)
        if isinstance(node, ast.Name)
    )


def test_reachable_path_catches_conflict_hidden_by_old_lane_interpolation():
    controller, pose, lane_idx = _controller(
        2, vehicle_y=0.6, vehicle_x=6.0)

    conflict = controller._candidate_lane_predicted_envelope_conflict(
        pose, 10.0, lane_idx, "d3")

    assert conflict is not None
    assert conflict["prediction_time"] == pytest.approx(0.6)

    # The removed implementation forced e_y to the L2 center at 0.6 s.
    # That unsolved path falsely clears the same stationary vehicle.
    old_interpolated_y = [1.03, 1.89, 1.89]
    old_result = minimum_predicted_envelope_conflict(
        [2.0, 6.0, 15.0], old_interpolated_y, [0.2, 0.6, 1.5],
        [("d3", 6.0, 0.6, 0.0, 0.0)],
        project_frenet=controller._center_frenet,
        arc_total_length=controller._center_arc_total_length,
        ego_width=1.0,
        other_half_width=0.5,
        ego_half_length=1.0,
        other_half_length=1.0,
    )
    assert old_result is None


def test_pre_guidance_prediction_cannot_prove_l2_candidate():
    controller, pose, _ = _controller(2, vehicle_y=10.0)
    controller._outer_candidate_guidance = None

    assert controller._candidate_lane_center_prediction(
        pose, 10.0, 2, "d3") is None


def test_guided_prediction_is_not_reused_for_opposite_candidate():
    controller, pose, _ = _controller(2, vehicle_y=10.0)

    assert controller._candidate_lane_center_prediction(
        pose, 10.0, 0, "d3") is None


def test_guided_prediction_is_not_reused_for_another_vehicle():
    controller, pose, _ = _controller(2, vehicle_y=10.0)

    assert controller._candidate_lane_center_prediction(
        pose, 10.0, 2, "d2") is None


def test_guided_prediction_is_valid_for_only_the_following_loop():
    controller, pose, _ = _controller(2, vehicle_y=10.0)
    controller._loop = 2

    assert controller._candidate_lane_center_prediction(
        pose, 10.0, 2, "d3") is None


def test_candidate_switch_invalidates_the_previous_guided_proof():
    controller, pose, _ = _controller(2, vehicle_y=10.0)

    controller._arm_outer_candidate_guidance(
        vehicle_id="d3", lane_idx=0, source="normal", now_sec=1.0)

    assert controller._outer_candidate_guidance["lane_idx"] == 0
    assert controller._outer_candidate_guidance["prediction_loop"] is None
    assert controller._candidate_lane_center_prediction(
        pose, 10.0, 2, "d3") is None
    assert controller._candidate_lane_center_prediction(
        pose, 10.0, 0, "d3") is None


def test_zero_strength_pre_guidance_solve_is_not_captured_as_proof():
    controller, _, _ = _controller(2, vehicle_y=10.0)
    record = controller._outer_candidate_guidance
    record["prediction_loop"] = None
    record["prediction_id"] = None
    record["last_guidance_alpha"] = 0.0
    record["ready_logged"] = False

    controller._capture_outer_candidate_guided_prediction(
        guidance_applied=True, applied_lane_idx=None,
        candidate_state_time_sec=10.0)

    assert record["prediction_loop"] is None


def _handoff_controller(lane_idx, alpha):
    controller = MPCController.__new__(MPCController)
    controller._loop = 10
    controller.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    controller._outer_candidate_guidance = {
        "vehicle_id": "d3", "lane_idx": lane_idx, "source": "normal",
        "guidance_loop": 8,
        "prediction_loop": 9,
        "candidate_state_time_sec": 10.0,
        "mpc_id": None,
        "prediction_id": None,
        "constraint_time_grid_used": (0.08, 0.18, 0.28),
        "constraint_time_grid_result": (0.1, 0.2, 0.3),
        "candidate_solve_time_aligned": True,
        "reference_path_id": None,
        "grid_source_loop": 8,
        "grid_source_mpc_id": None,
        "grid_source_reference_path_id": None,
        "grid_source_prediction_id": 1,
    }
    controller._outer_candidate_reference_handoff = None
    controller._overtake_soft_transition_lane_idx = None
    controller._overtake_soft_transition_start_e_y = None
    controller._overtake_soft_transition_start_alpha = 0.0
    controller._overtake_soft_transition_lateral_offset = 0.0
    controller._overtake_soft_transition_lateral_targets = None
    target = -1.5 if lane_idx == 0 else 1.5
    controller._mpcN_center = SimpleNamespace(
        soft_target_lane_idx=lane_idx,
        soft_target_start_e_y=0.0,
        soft_target_alpha=alpha,
        soft_target_lateral_offset=0.0,
        soft_lateral_targets=None,
        _compute_lane_center=lambda _wp, _lane: target,
        set_soft_lateral_reference=lambda **kwargs: setattr(
            controller, "applied_soft_reference", kwargs),
    )
    controller._carN_center = SimpleNamespace(
        wp_id=10, spatial_state=SimpleNamespace(e_y=0.1))
    controller._mpc = SimpleNamespace(
        N=3,
        model=SimpleNamespace(reference_path=object()),
        current_prediction=([1.0], [0.1]),
        current_prediction_times=[0.2],
        infeasibility_counter=0,
        used_prediction_fallback=False,
        recovery_requested=False,
        time_budget_exceeded=False,
        last_solution_accurate=True,
    )
    controller._outer_candidate_guidance["mpc_id"] = id(controller._mpc)
    controller._outer_candidate_guidance["grid_source_mpc_id"] = id(
        controller._mpc)
    controller._outer_candidate_guidance["reference_path_id"] = id(
        controller._mpc.model.reference_path)
    controller._outer_candidate_guidance[
        "grid_source_reference_path_id"] = id(
            controller._mpc.model.reference_path)
    controller._outer_candidate_guidance["prediction_id"] = id(
        controller._mpc.current_prediction)
    controller._outer_commit_time_aligned_v2x = None
    controller._outer_candidate_guided_prediction_ready = (
        lambda vehicle_id, candidate_lane: (
            vehicle_id == "d3" and candidate_lane == lane_idx))
    controller.get_logger = lambda: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    return controller


@pytest.mark.parametrize("lane_idx", [0, 2])
def test_candidate_reference_handoff_is_symmetric_and_does_not_reset(lane_idx):
    controller = _handoff_controller(lane_idx, alpha=1.0)

    assert controller._capture_outer_candidate_reference_handoff(
        vehicle_id="d3", lane_idx=lane_idx, source="normal")
    controller._discard_outer_candidate_guidance("test_commit")
    controller._update_overtake_transition_soft_reference(
        enabled=True,
        lane_idx=lane_idx,
        now_sec=10.0,
        transition_end_sec=10.6,
        transition_duration_sec=0.6,
    )

    assert controller.applied_soft_reference["lane_idx"] == lane_idx
    assert controller.applied_soft_reference["start_e_y"] == pytest.approx(0.0)
    assert controller.applied_soft_reference["alpha"] == pytest.approx(1.0)
    assert controller._outer_commit_time_aligned_v2x["source"] == "normal"
    assert controller._outer_commit_time_aligned_v2x["constraint_times"] == (
        0.08, 0.18, 0.28)


def test_partial_candidate_reference_continues_from_effective_alpha():
    controller = _handoff_controller(2, alpha=0.4)

    assert controller._capture_outer_candidate_reference_handoff(
        vehicle_id="d3", lane_idx=2, source="fallback")
    controller._discard_outer_candidate_guidance("test_commit")
    controller._update_overtake_transition_soft_reference(
        enabled=True,
        lane_idx=2,
        now_sec=20.0,
        transition_end_sec=20.6,
        transition_duration_sec=0.6,
    )
    assert controller.applied_soft_reference["alpha"] == pytest.approx(0.4)
    assert controller._outer_commit_time_aligned_v2x["source"] == "fallback"

    controller._update_overtake_transition_soft_reference(
        enabled=True,
        lane_idx=2,
        now_sec=20.3,
        transition_end_sec=20.6,
        transition_duration_sec=0.6,
    )
    assert controller.applied_soft_reference["alpha"] == pytest.approx(0.7)


def test_preempt_handoff_survives_common_latch_for_the_same_commit():
    controller = _handoff_controller(2, alpha=0.4)
    controller._outer_candidate_guidance["source"] = "preempt"

    captured = controller._capture_outer_candidate_reference_handoff(
        vehicle_id="d3", lane_idx=2, source="preempt")
    controller._discard_outer_candidate_guidance("preempt_outer_commit")
    preserve_preempt_handoff = bool(
        captured
        and controller._outer_candidate_reference_handoff_matches(
            source="preempt",
            vehicle_id="d3",
            lane_idx=2,
            control_loop=controller._loop,
        )
    )
    if not preserve_preempt_handoff:
        controller._capture_outer_candidate_reference_handoff(
            vehicle_id="d3", lane_idx=2, source="normal")

    assert preserve_preempt_handoff
    assert controller._outer_candidate_reference_handoff["source"] == "preempt"
    assert controller._outer_commit_time_aligned_v2x["source"] == "preempt"
    assert controller._outer_candidate_reference_handoff["alpha"] == pytest.approx(0.4)

    controller._update_overtake_transition_soft_reference(
        enabled=True,
        lane_idx=2,
        now_sec=20.0,
        transition_end_sec=20.6,
        transition_duration_sec=0.6,
    )
    assert controller.applied_soft_reference["start_e_y"] == pytest.approx(0.0)
    assert controller.applied_soft_reference["alpha"] == pytest.approx(0.4)


def test_common_latch_does_not_recapture_a_preserved_preempt_handoff():
    control = _controller_method_ast("_control")
    parents = {}
    for node in ast.walk(control):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    normal_capture_calls = []
    for node in ast.walk(control):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_capture_outer_candidate_reference_handoff"
        ):
            continue
        source = next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "source"
                and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
        if source == "normal":
            normal_capture_calls.append(node)

    assert len(normal_capture_calls) == 1
    capture_statement = parents[normal_capture_calls[0]]
    capture_guard = parents[capture_statement]
    assert isinstance(capture_guard, ast.If)
    assert isinstance(capture_guard.test, ast.UnaryOp)
    assert isinstance(capture_guard.test.op, ast.Not)
    assert isinstance(capture_guard.test.operand, ast.Name)
    assert capture_guard.test.operand.id == "preserve_preempt_handoff"


@pytest.mark.parametrize(
    ("source", "vehicle_id", "lane_idx", "control_loop"),
    [
        ("normal", "d3", 2, 10),
        ("preempt", "d2", 2, 10),
        ("preempt", "d3", 0, 10),
        ("preempt", "d3", 2, 11),
    ],
)
def test_preempt_handoff_is_not_reused_for_another_commit_identity(
    source, vehicle_id, lane_idx, control_loop,
):
    controller = _handoff_controller(2, alpha=0.4)
    controller._outer_candidate_guidance["source"] = "preempt"
    assert controller._capture_outer_candidate_reference_handoff(
        vehicle_id="d3", lane_idx=2, source="preempt")

    assert not controller._outer_candidate_reference_handoff_matches(
        source=source,
        vehicle_id=vehicle_id,
        lane_idx=lane_idx,
        control_loop=control_loop,
    )


def test_different_candidate_cannot_use_reference_handoff():
    controller = _handoff_controller(2, alpha=1.0)

    assert not controller._capture_outer_candidate_reference_handoff(
        vehicle_id="d3", lane_idx=0, source="normal")
    controller._update_overtake_transition_soft_reference(
        enabled=True,
        lane_idx=0,
        now_sec=30.0,
        transition_end_sec=30.6,
        transition_duration_sec=0.6,
    )

    assert controller.applied_soft_reference["start_e_y"] == pytest.approx(0.1)
    assert controller.applied_soft_reference["alpha"] == pytest.approx(0.0)


def test_commit_without_candidate_proof_keeps_existing_transition_start():
    controller = _handoff_controller(2, alpha=1.0)
    controller._outer_candidate_guidance = None

    assert not controller._capture_outer_candidate_reference_handoff(
        vehicle_id="d3", lane_idx=2, source="normal")
    controller._update_overtake_transition_soft_reference(
        enabled=True,
        lane_idx=2,
        now_sec=40.0,
        transition_end_sec=40.6,
        transition_duration_sec=0.6,
    )

    assert controller.applied_soft_reference["start_e_y"] == pytest.approx(0.1)
    assert controller.applied_soft_reference["alpha"] == pytest.approx(0.0)


def test_nonzero_candidate_guidance_captures_fresh_full_width_prediction():
    controller, _, _ = _controller(2, vehicle_y=10.0)
    record = controller._outer_candidate_guidance
    record["prediction_loop"] = None
    record["prediction_id"] = None
    record["last_guidance_alpha"] = 0.1
    record["ready_logged"] = False

    controller._capture_outer_candidate_guided_prediction(
        guidance_applied=True, applied_lane_idx=None,
        candidate_state_time_sec=10.0)

    assert record["prediction_loop"] == controller._loop
    assert record["prediction_id"] == id(controller._mpc.current_prediction)
