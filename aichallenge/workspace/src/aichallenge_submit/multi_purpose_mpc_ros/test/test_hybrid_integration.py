"""Execute controller handoffs without ROS; use the real MPC reference setters."""
import ast
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.overtake_session import OvertakeSession, ShadowVerification
from multi_purpose_mpc_ros.v2x_vehicle_tracker import is_prepass_fallback_lane_change
from .test_overtake_session import controller_method, decision


def controller():
    mpc = MPC.__new__(MPC)
    mpc.N = 4
    mpc.current_prediction = None
    mpc.last_solution_accurate = False
    mpc.used_prediction_fallback = False
    mpc.recovery_requested = False
    mpc.set_soft_lateral_reference()
    mpc.set_lane_transition_weights()
    mpc._compute_lane_center = lambda wp, lane: -2.0 if lane == 0 else 2.0
    path = SimpleNamespace(n_waypoints=100, segment_lengths=np.ones(100),
                           target_lane_idx=2, is_overtaking=True)
    path.get_waypoint = lambda wp: SimpleNamespace(x=wp % 100, y=0.0, psi=0.0)
    c = SimpleNamespace(
        _overtake=OvertakeSession(target_id='d2', requested_lane=2, committed=True),
        _lane_decision=None, _hybrid_reference_key=None,
        _hybrid_overtake_enabled=True, _hybrid_overtake_transition_timeout=8.0,
        _hybrid_overtake_base_length=6.0, _hybrid_overtake_offset_gain=2.5,
        _hybrid_overtake_speed_gain=0.8, _hybrid_overtake_min_length=8.0,
        _hybrid_overtake_max_length=20.0,
        _hybrid_overtake_low_speed_start_threshold=2.0,
        _hybrid_overtake_low_speed_max_length=8.0,
        _hybrid_overtake_continuity_weight=0.30,
        _hybrid_overtake_continuity_max_deviation=0.75,
        _mpcN_center=mpc, _reference_path=path, _reference_pathN=path,
        _reference_pathN_center=path,
        _carN_center=SimpleNamespace(wp_id=0, spatial_state=SimpleNamespace(e_y=0.0)),
        _l2_inward_offset_zones=[], _l1_probe_active=False,
        _last_lane_change_time=None, _reset_outer_lane_progress=Mock(),
        get_logger=Mock(return_value=Mock()))
    for name in ('_apply_lane_decision', '_update_overtake_transition_soft_reference',
                 '_closed_path_distance', '_hybrid_horizon_distances',
                 '_shifted_center_prediction_lateral'):
        setattr(c, name, MethodType(controller_method(name), c))
    return c


def apply(c, lane=2, now=10.0, **kwargs):
    args = dict(requested_lane=lane, now_sec=now, l0_prohibited=False,
                full_width_recovery=False)
    args.update(kwargs)
    return c._apply_lane_decision(**args)


def reference(c, enabled=True, lane=2, now=10.0, speed=1.0):
    # This is the reset performed by _update_l1_soft_rejoin_reference each cycle.
    c._mpcN_center.set_soft_lateral_reference()
    c._update_overtake_transition_soft_reference(
        enabled=enabled, lane_idx=lane, now_sec=now,
        ego_speed=speed, vehicle_id=c._overtake.target_id)


@pytest.mark.parametrize('speed,expected', [(0.0, 8.0), (2.0, 8.0), (5.0, 15.0), (20.0, 20.0)])
def test_length_uses_start_speed_and_lateral_distance(speed, expected):
    c = controller()
    apply(c)
    reference(c, speed=speed)
    assert c._overtake.hybrid.length == pytest.approx(expected)


def test_stationary_time_does_not_advance_reference_or_bounds():
    c = controller()
    apply(c)
    reference(c)
    targets = c._mpcN_center.soft_lateral_targets.copy()
    weights = c._mpcN_center.lane_transition_weights.copy()
    # Even after the temporal window expires, geometry cannot jump to the lane.
    apply(c, now=20.0)
    reference(c, now=20.0)
    np.testing.assert_allclose(c._mpcN_center.soft_lateral_targets, targets)
    np.testing.assert_allclose(c._mpcN_center.lane_transition_weights, weights)
    assert c._overtake.hybrid.travelled == 0.0
    assert not c._overtake.hybrid.completed


@pytest.mark.parametrize('recovery', [False, True])
def test_full_width_pause_preserves_progress_deadline_and_cooldown(recovery):
    c = controller()
    apply(c)
    reference(c)
    anchor = c._overtake.hybrid
    deadline = c._constraint_transition_until
    c._last_lane_change_time = 10.0
    for now, lane in [(11.0, None), (11.1, 2), (11.2, None), (11.3, 2)]:
        c._carN_center.wp_id += 1
        apply(c, lane=lane, now=now, preserve_manoeuvre=True,
              full_width_recovery=recovery and lane is None)
        reference(c, enabled=lane is not None, lane=lane, now=now)
        assert c._overtake.hybrid is anchor
        assert c._constraint_transition_until == deadline
        assert c._last_lane_change_time == 10.0
        assert c._reference_path.target_lane_idx == lane
        assert c._overtake.verification == ShadowVerification()
        if lane is None:
            assert c._mpcN_center.lane_transition_weights is None
            assert c._mpcN_center.soft_lateral_targets is None
    assert anchor.travelled == pytest.approx(4.0)
    assert c._mpcN_center.soft_lateral_targets[0] == pytest.approx(1.0)


def test_forward_lap_seam_and_backward_waypoints_are_signed():
    c = controller()
    c._carN_center.wp_id = 98
    apply(c)
    reference(c)
    for wp, distance in [(99, 1.0), (0, 2.0), (99, 1.0), (98, 0.0), (97, -1.0), (98, 0.0)]:
        c._carN_center.wp_id = wp
        reference(c)
        assert c._overtake.hybrid.travelled == pytest.approx(distance)
        assert not c._overtake.hybrid.completed


def test_completion_ends_hybrid_without_restarting_it():
    c = controller()
    apply(c)
    reference(c)
    c._carN_center.wp_id = 8
    reference(c, now=11.0)
    assert c._overtake.hybrid.completed
    assert apply(c, now=11.1) == (2, False, False)
    reference(c, enabled=False, now=11.1)
    assert c._mpcN_center.lane_transition_weights is None


@pytest.mark.parametrize('kind', ['target', 'side', 'l1', 'cancel', 'complete', 'prohibited'])
def test_real_manoeuvre_end_does_not_inherit_anchor(kind):
    c = controller()
    apply(c)
    reference(c)
    c._overtake.verification = ShadowVerification('d2', 2)
    if kind == 'target':
        c._overtake.target_id = 'd3'
        apply(c, now=12.0, preserve_manoeuvre=True)
    elif kind == 'side':
        apply(c, lane=0, now=12.0, preserve_manoeuvre=True)
    elif kind == 'l1':
        apply(c, lane=1, now=12.0, preserve_manoeuvre=True)
    elif kind == 'cancel':
        apply(c, lane=None, now=12.0)
    elif kind == 'complete':
        c._overtake.complete_pass()
        apply(c, lane=None, now=12.0, preserve_manoeuvre=True)
    else:
        apply(c, lane=0, now=12.0, l0_prohibited=True)
    assert c._overtake.hybrid.start_wp is None
    assert c._overtake.verification.vehicle_id is None


@pytest.mark.parametrize('kwargs,expected', [
    ({'full_width_recovery': True}, (None, False, False)),
    ({'l0_prohibited': True}, (1, False, False)),
    ({'allow_hybrid': False}, (0, False, False)),
])
def test_safety_geography_and_startup_win_over_hybrid(kwargs, expected):
    c = controller()
    assert apply(c, lane=0, **kwargs) == expected


def test_initial_l0_hold_is_immediate_without_overtake_target():
    c = controller()
    c._overtake.release_target()
    assert apply(c, lane=0) == (0, False, False)
    assert c._constraint_transition_until == pytest.approx(10.0)
    assert c._last_lane_change_time == pytest.approx(10.0)


def test_disabled_hybrid_retains_legacy_window_on_subsequent_lane_change():
    c = controller()
    c._hybrid_overtake_enabled = False
    apply(c, lane=1)
    assert apply(c, lane=2, now=11.0) == (None, False, True)
    assert c._constraint_transition_until == pytest.approx(11.6)


def test_same_lane_resume_after_recovery_keeps_deadline_without_ordinary_owner_flag():
    c = controller()
    apply(c)
    reference(c)
    anchor = c._overtake.hybrid
    apply(c, lane=None, now=11.0, preserve_manoeuvre=True, full_width_recovery=True)
    apply(c, lane=2, now=11.2)
    assert c._overtake.hybrid is anchor
    assert c._constraint_transition_until == 18.0
    assert c._last_lane_change_time == 10.0


@pytest.mark.parametrize('problem', ['wrong_key', 'fallback', 'recovery', 'inaccurate'])
def test_old_prediction_cannot_cross_identity_or_solver_failure(problem):
    c = controller()
    apply(c)
    reference(c)
    c._mpcN_center.current_prediction = ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0])
    c._mpcN_center.last_solution_accurate = True
    if problem == 'wrong_key':
        c._hybrid_reference_key = ('d3', 2)
    elif problem == 'fallback':
        c._mpcN_center.used_prediction_fallback = True
    elif problem == 'recovery':
        c._mpcN_center.recovery_requested = True
    else:
        c._mpcN_center.last_solution_accurate = False
    assert c._shifted_center_prediction_lateral(0) is None


@pytest.mark.parametrize('name', ['_reset_overtake_state_for_target_change', '_release_lost_follow_target', '_complete_overtake_target_behind'])
def test_target_end_clears_target_local_speed_and_proof(name):
    c = controller()
    apply(c)
    reference(c)
    fn = controller_method(name)
    tree = ast.parse((Path(__file__).parents[1] / 'multi_purpose_mpc_ros/mpc_controller.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    for node in ast.walk(method):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
            if not hasattr(c, node.attr):
                setattr(c, node.attr, Mock())
    c._follow_escape_active = False
    c._prepass_dynamic_conflict_speed_limit = 3.0
    c._overtake.verification = ShadowVerification('d2', 2)
    c.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10**9)))
    if name == '_reset_overtake_state_for_target_change':
        fn(c, 'd3', reason='test')
    elif name == '_release_lost_follow_target':
        fn(c)
    else:
        fn(c, 'd2', -5.0, source='test')
    assert c._prepass_dynamic_conflict_speed_limit is None
    assert c._overtake.verification == ShadowVerification()
    assert not c._overtake.can_resume_hybrid(2)


def control_tree():
    tree = ast.parse((Path(__file__).parents[1] / 'multi_purpose_mpc_ros/mpc_controller.py').read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_control')


def actual_assignment(name):
    return next(n for n in ast.walk(control_tree()) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


@pytest.mark.parametrize('owner', [None, '_l1_probe_active', '_center_lane_rejoin_active',
                                 '_center_lane_rejoin_constraint_released',
                                 '_l1_safety_recovery_active', '_l1_rejoin_backoff_active',
                                 '_parallel_abort_active', '_prepass_fallback_follow_active'])
def test_actual_continuation_gate_yields_to_other_owners(owner):
    c = controller()
    apply(c)
    reference(c)
    node = actual_assignment('preserve_hybrid_request')
    values = {n.id: False for n in ast.walk(node) if isinstance(n, ast.Name)}
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'self':
            if not hasattr(c, n.attr):
                setattr(c, n.attr, False)
    c._prepass_fallback_lane_idx = None
    c._overtake_completed_target_id = None
    values.update(self=c, new_target_lane_idx=None, bool=bool, any=any)
    if owner:
        setattr(c, owner, True)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-continuation>', 'exec'), values)
    assert values['preserve_hybrid_request'] is (owner is None)


@pytest.mark.parametrize('preserve,expected', [(False, None), (True, 2)])
def test_actual_cooldown_allows_same_hybrid_resume(preserve, expected):
    node = next(n for n in ast.walk(control_tree()) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'new_target_lane_idx != prev_lane_idx')
    c = controller()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'self':
            if not hasattr(c, n.attr):
                setattr(c, n.attr, False)
    c._target_lane_idx = None
    c._last_lane_change_time = 10.0
    c._lane_change_cooldown_sec = 2.0
    c._prepass_fallback_lane_idx = None
    values = {n.id: False for n in ast.walk(node) if isinstance(n, ast.Name)}
    values.update(self=c, new_target_lane_idx=2, prev_lane_idx=None,
                  current_time_sec=10.1, preserve_hybrid_request=preserve,
                  opponent_ahead_detected=True,
                  is_prepass_fallback_lane_change=is_prepass_fallback_lane_change)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-cooldown>', 'exec'), values)
    assert c._target_lane_idx == expected


@pytest.mark.parametrize('owner', ['_l1_probe_active', '_center_lane_rejoin_active',
                                 '_center_lane_rejoin_constraint_released',
                                 '_l1_safety_recovery_active', '_l1_rejoin_backoff_active'])
def test_actual_old_corridor_monitor_cannot_preempt_l1(owner):
    node = next(n for n in ast.walk(control_tree()) if isinstance(n, ast.If)
                and any(isinstance(b, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'latched_lane_idx'
                    for t in b.targets) for b in n.body))
    c = controller()
    for n in ast.walk(node.test):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'self':
            if not hasattr(c, n.attr):
                setattr(c, n.attr, False)
    values = dict(math=math, self=c, latched_target_id='d2', latched_target_longitudinal=10.0,
                  recovery_active=False)
    expression = compile(ast.Expression(node.test), '<actual-monitor>', 'eval')
    assert eval(expression, values)
    setattr(c, owner, True)
    assert not eval(expression, values)
