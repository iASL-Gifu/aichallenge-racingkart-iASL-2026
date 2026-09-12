"""A checked approximate solve can end fallback without bypassing braking."""
import ast
import copy
import math
from types import MethodType
from unittest.mock import Mock

import pytest

from .test_slow_pass_spacing_release import controller as traffic_controller, POSE
from .test_overtake_session import controller_method
from .test_hybrid_integration import control_tree
from .test_v2x_vehicle_tracker import _msg


def controller():
    c = traffic_controller()
    c._mpc.last_solution_status = 'solved inaccurate'
    c._mpc.last_solution_accurate = False
    c._mpc.collision_prediction_context = (c._mpc.current_prediction, c._mpc.current_prediction)
    c._pure_pursuit_feedback_is_safe = Mock(return_value=(True, 'ok'))
    c._mpc_prediction_path_is_clear = Mock(return_value=True)
    c._inaccurate_mpc_handoff_is_safe = MethodType(
        controller_method('_inaccurate_mpc_handoff_is_safe'), c)
    return c


def test_approximate_prediction_passing_two_stopped_cars_is_usable():
    c = controller()
    assert c._inaccurate_mpc_handoff_is_safe(POSE, 0., [2., .1])
    c._mpc_prediction_path_is_clear.assert_called_once_with(POSE, [2., .1])
    c._pure_pursuit_feedback_is_safe.assert_not_called()


@pytest.mark.parametrize('case', ['wall', 'collision', 'unknown_vehicle', 'stale', 'failure',
    'fallback', 'recovery', 'budget', 'zero', 'nan', 'disconnected', 'stationary', 'invalid_ego'])
def test_approximate_solution_needs_current_command_and_clear_connected_path(case):
    c = controller()
    m = c._mpc
    u = [2., .1]
    if case == 'wall': c._mpc_prediction_path_is_clear.return_value = False
    elif case == 'collision': m.current_prediction[1][:] = [0.] * len(m.current_prediction[0])
    elif case == 'unknown_vehicle': c._v2x_tracker.update(_msg(.1, [('new', 15., 0.)]))
    elif case == 'stale': m.collision_prediction_context = (object(), m.current_prediction)
    elif case == 'failure': m.infeasibility_counter = 1
    elif case == 'fallback': m.used_prediction_fallback = True
    elif case == 'recovery': m.recovery_requested = True
    elif case == 'budget': m.time_budget_exceeded = True
    elif case == 'zero': u[0] = 0.
    elif case == 'nan': m.current_prediction[0][3] = float('nan')
    elif case == 'disconnected': m.current_prediction[0][0] = 5.
    elif case == 'stationary':
        m.current_prediction[0][:] = [.1] * len(m.current_prediction[0])
        m.current_prediction[1][:] = [0.] * len(m.current_prediction[1])
    elif case == 'invalid_ego': c._collision_ego_metadata = (.1, 'map', False)
    assert not c._inaccurate_mpc_handoff_is_safe(POSE, 0., u)


def fallback_tick(c, accurate=False):
    # Execute the actual ownership block, including its eight-cycle latch and
    # PP override, with the real additional collision validation.
    node = next(n for n in ast.walk(control_tree()) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'self._steering_fallback_armed'
                and any(isinstance(a, ast.Name) and a.id == 'checked_approximate'
                        for a in ast.walk(n)))
    scope = dict(math=math, self=c, solution_valid=True, solution_accurate=accurate,
                 predicted_pose=POSE, v=0., u=[2., .1],
                 moving_target_prediction_clear=False,
                 wall_fallback_stop=False, pure_pursuit_safe_this_cycle=False)
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), '<handoff>', 'exec'), scope)
    return scope


def test_eight_checked_cycles_restore_mpc_steering_and_speed_after_pp_stop():
    c = controller()
    c._steering_fallback_armed = True
    c._steering_fallback_success_cycles = 0
    c._steering_fallback_success_required = 8
    c._steering_fallback_speed = 7.5
    c._active_path_pure_pursuit_feedback = lambda *a: (None, 1., 262, 'wall')
    c._legacy_active_path_feedback = lambda: -.314
    # MPC steering is clear; the latched fallback steering runs into the wall.
    c._pure_pursuit_feedback_is_safe.side_effect = lambda p,v,d: (d > 0., 'wall')
    c.get_logger = Mock(return_value=Mock())
    for i in range(7):
        result = fallback_tick(c)
        assert result['wall_fallback_stop']
        assert result['u'][0] == 0.
        assert c._steering_fallback_armed
    result = fallback_tick(c)
    assert not c._steering_fallback_armed
    assert not result['wall_fallback_stop']
    assert result['u'] == [2., .1]
    assert c._steering_fallback_success_cycles == 0


def test_unsafe_cycle_resets_handoff_confirmation():
    c = controller()
    c._steering_fallback_armed = True
    c._steering_fallback_success_cycles = 7
    c._steering_fallback_success_required = 8
    c._steering_fallback_speed = 7.5
    c._active_path_pure_pursuit_feedback = lambda *a: (None, 1., 262, 'wall')
    c._legacy_active_path_feedback = lambda: -.314
    c._mpc_prediction_path_is_clear.return_value = False
    c._pure_pursuit_feedback_is_safe.return_value = (False, 'wall')
    c.get_logger = Mock(return_value=Mock())
    assert fallback_tick(c)['u'][0] == 0.
    assert c._steering_fallback_armed
    assert c._steering_fallback_success_cycles == 0


@pytest.mark.parametrize('case,reason', [
    ('wall', 'mpc_path: wall'),
    ('zero', 'speed_too_low:'),
    ('collision', 'vehicle_collision:'),
    ('stale', 'stale_collision_prediction'),
])
def test_handoff_reports_rejection_reason(case, reason):
    c = controller()
    u = [2., .1]
    if case == 'wall':
        def reject(*args):
            c._mpc_handoff_reason='mpc_path: wall'
            return False
        c._mpc_prediction_path_is_clear.side_effect=reject
    elif case == 'zero':
        u[0] = 0.
    elif case == 'collision':
        c._mpc.current_prediction[1][:] = [0.] * len(c._mpc.current_prediction[0])
    elif case == 'stale':
        c._mpc.collision_prediction_context = (object(), c._mpc.current_prediction)
    assert not c._inaccurate_mpc_handoff_is_safe(POSE, 0., u)
    assert c._mpc_handoff_reason.startswith(reason)


@pytest.mark.parametrize('wall_stop,speed,immobile,expected', [
    (True, 0., True, True),
    (False, 0., True, False),
    (True, .5, True, False),
    (True, 0., False, False),
])
@pytest.mark.parametrize('safety_recovery', [False, True])
def test_valid_prediction_does_not_block_wall_stop_recovery(
        wall_stop, speed, immobile, expected, safety_recovery):
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import should_recover_from_mpc_stall
    assert should_recover_from_mpc_stall(
        safety_recovery_active=safety_recovery,
        actual_speed=speed, stall_speed_threshold=.4,
        gnss_is_stuck=immobile, infeasibility_counter=0,
        has_fresh_valid_prediction=True,
        wall_fallback_stop=wall_stop,
    ) is expected


@pytest.mark.parametrize('moved', [None, 0., .29, .30])
def test_brief_mpc_recovery_preserves_observation_until_actual_motion(moved):
    from types import SimpleNamespace as NS
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import should_recover_from_mpc_stall
    # Exercise the controller's early fresh-prediction exit and actual detector
    # call together: bypassing only the detector would still leave the deadlock.
    from pathlib import Path
    tree = ast.parse((Path(__file__).parents[1] / 'multi_purpose_mpc_ros'
                      / 'mpc_controller.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == '_apply_vehicle_stuck_recovery')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'has_fresh_valid_prediction')
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'recover_from_mpc_stall')
    wrapper = ast.parse('def check(): pass').body[0]
    wrapper.body = copy.deepcopy(fn.body[start:end+1]) + [
        ast.Return(value=ast.Name(id='recover_from_mpc_stall', ctx=ast.Load()))]
    c = NS(_mpc=NS(infeasibility_counter=0, current_prediction=object(),
                   used_prediction_fallback=False, recovery_requested=False),
           _mpc_safety_recovery_active=True, _mpc_stall_speed_threshold=.4,
           _stuck_since=1., _gnss_history=[(1., 0., 0.)],
           _stuck_gnss_distance_threshold=.30)
    scope = dict(self=c, actual_speed=0., gnss_is_stuck=True,
                 wall_fallback_stop=False, gnss_moved_dist=moved,
                 should_recover_from_mpc_stall=should_recover_from_mpc_stall)
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, '<stall>', 'exec'), scope)
    for _ in range(8):
        assert scope['check']() is False
    did_move = moved is not None and moved >= .30
    assert bool(c._gnss_history) is not did_move
    assert (c._stuck_since is None) is did_move
    # The next wall stop can still use the accumulated observation.
    scope['wall_fallback_stop'] = True
    scope['gnss_is_stuck'] = not did_move
    assert scope['check']() is not did_move


@pytest.mark.parametrize('wall_safe', [True, False])
def test_reverse_requires_forward_command_to_pass_wall_check(wall_safe):
    from types import SimpleNamespace as NS
    c = NS(_mpc=NS(infeasibility_counter=0, current_prediction=object(),
                    used_prediction_fallback=False, recovery_requested=False,
                    time_budget_exceeded=False, last_solution_accurate=True),
           _overtake=NS(target_id=None),
           _prediction_has_forward_progress=Mock(return_value=True),
           _pure_pursuit_feedback_is_safe=Mock(return_value=(wall_safe, 'wall_at_step=0')),
           get_logger=Mock(return_value=Mock()))
    result = controller_method('_reverse_forward_path_is_valid')(c, POSE, [1.5, .1])
    assert result is wall_safe
    c._pure_pursuit_feedback_is_safe.assert_called_once_with(POSE, 1.5, .1)
    assert c.get_logger().warn.called is not wall_safe


@pytest.mark.parametrize('wall_safe', [True, False])
def test_lead_advance_cannot_end_reverse_with_blocked_forward_command(wall_safe):
    from types import SimpleNamespace as NS
    c = NS(_latched_follow_target_state=lambda *a: dict(
                longitudinal=10., speed=2., velocity_valid=True),
           _stuck_reverse_start_target_longitudinal=5.,
           _stuck_forward_resume_gap=2., _stuck_forward_resume_lead_speed=1.,
           _stuck_forward_resume_min_command=.5,
           _mpc=NS(infeasibility_counter=0, current_prediction=object(),
                   used_prediction_fallback=False),
           _mpc_safety_recovery_active=False, _overtake=NS(requested_lane=2),
           _select_prepass_retry_lane=lambda *a: (2, None, None),
           _pure_pursuit_feedback_is_safe=Mock(return_value=(wall_safe, 'wall_at_step=0')))
    result, reason = controller_method('_reverse_forward_resume_ready')(
        c, POSE, 10., [1.5, .1], -2.)
    assert result is wall_safe
    c._pure_pursuit_feedback_is_safe.assert_called_once_with(POSE, 2., .1)
    if not wall_safe:
        assert reason == 'forward_command_wall: wall_at_step=0'


@pytest.mark.parametrize('clearance,traffic_clear,stopped', [
    (.1, True, True), (1., False, True), (1., True, False),
])
def test_generic_reverse_still_stops_for_rear_hazards(clearance, traffic_clear, stopped):
    from pathlib import Path
    from types import SimpleNamespace as NS
    tree = ast.parse((Path(__file__).parents[1] / 'multi_purpose_mpc_ros'
                      / 'mpc_controller.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and any(isinstance(child, ast.Assign)
                        and ast.unparse(child.targets[0]) == 'rear_probe'
                        for child in n.body))
    c = NS(_stuck_reverse_drive_active=True, _adaptive_reverse_active=False,
           _generic_reverse_min_distance=.5, _prepass_reverse_distance=0.,
           _adaptive_reverse_wall_margin=.2,
           _static_reverse_clearance=lambda *a: clearance,
           _reverse_rear_is_clear=Mock(return_value=traffic_clear),
           _begin_stuck_drive_transition=Mock())
    scope = dict(self=c, in_drive_transition=False, pose=POSE,
                 actual_speed=-.5, now_sec=10., u=[1.5, .1])
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<rear>', 'exec'), scope)
    assert scope['in_drive_transition'] is stopped
    assert c._begin_stuck_drive_transition.called is stopped
    if stopped:
        assert scope['u'] == [0., 0.]



@pytest.mark.parametrize('safe', [True, False])
def test_accurate_normal_handoff_requires_shared_mpc_path_check(safe):
    c=controller()
    c._mpc_prediction_path_is_clear.return_value=safe
    c._steering_fallback_armed=True
    c._steering_fallback_success_cycles=7
    c._steering_fallback_success_required=8
    c._steering_fallback_speed=7.5
    c._active_path_pure_pursuit_feedback=lambda *a:(None,1.,262,'wall')
    c._legacy_active_path_feedback=lambda:-.314
    c._pure_pursuit_feedback_is_safe.return_value=(False,'wall')
    c.get_logger=Mock(return_value=Mock())
    result=fallback_tick(c,accurate=True)
    assert c._steering_fallback_armed is not safe
    assert result['u'][0] == (2. if safe else 0.)
    c._mpc_prediction_path_is_clear.assert_called_once()
