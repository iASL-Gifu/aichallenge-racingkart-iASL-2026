"""Physical continuation and transactional admission regression tests."""
import copy
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.control_continuity import (
    CorridorState, RemainingPlan, REFERENCE_FIELDS, fork_solver, adopt_solver)
from multi_purpose_mpc_ros.overtake_session import OvertakeSession, LaneDecision
from .test_overtake_session import controller_method


def straight_plan():
    return RemainingPlan(10., object(), None,
        np.array([(i*.6, 0., 0.) for i in range(16)]),
        np.array([(3., .03*i) for i in range(16)]), np.array([2., 0.]))


def rollout(plan, **overrides):
    args = dict(pose=(.06, 0., 0.), now=10.025, speed=2.,
        previous_command=(2., 0.), steering=0., wheelbase=1.087, rate=1.2,
        delay=.15, deceleration=2.5, period=.025)
    args.update(overrides)
    return plan.rollout(**args)


def test_reuse_aligns_by_distance_not_one_spatial_sample_per_tick():
    result, _ = rollout(straight_plan())
    command, path, times = result
    assert command[1] == 0.  # still in segment zero, not stored second command
    assert command[0] <= 2.
    assert times[-1] == pytest.approx(.15+.025+2./2.5)
    assert max(np.linalg.norm(np.diff(np.array(path)[:, :2], axis=0), axis=1)) <= .051


@pytest.mark.parametrize('change,reason', [
    ({'now': 9.9}, 'expired'), ({'now': 10.076}, 'expired'),
    ({'pose': (.1, .6, 0.)}, 'disconnected'),
    ({'pose': (.1, 0., 1.)}, 'disconnected'),
    ({'pose': (5., 0., 0.)}, 'disconnected'),
    ({'speed': 0.}, 'not_moving_forward'),
    ({'steering': float('nan')}, 'nonfinite'),
    ({'speed': 9.}, 'insufficient_stopping_distance'),
])
def test_unusable_reuse_is_rejected(change, reason):
    result, actual = rollout(straight_plan(), **change)
    assert result is None and actual == reason


def test_steering_delay_and_rate_are_in_the_checked_rollout():
    plan = straight_plan()
    plan.controls[:, 1] = .3
    result, _ = rollout(plan)
    command, path, times = result
    assert command[1] == pytest.approx(.03)
    assert all(p[2] == 0. for p,t in zip(path,times) if t <= .15)
    assert path[-1][2] > 0.


def fake_controller():
    path = NS(target_lane_idx=1, is_overtaking=True, n_waypoints=354)
    model = NS(reference_path=path, wp_id=10)
    mpc = NS(model=model, current_prediction=object(), current_recovery_prediction=((0,0,0),(1,0,0)),
        current_control=np.array([1.,0.,1.,0.]), previous_steering=0.,
        infeasibility_counter=0, used_prediction_fallback=False,
        recovery_requested=False, time_budget_exceeded=False, failure_reason=None)
    for key in REFERENCE_FIELDS: setattr(mpc, key, None)
    mpc.soft_target_alpha = 0.
    c = NS(_mpc=mpc, _reference_path=path, _reference_pathN=path,
        _overtake=OvertakeSession(target_id='d2', requested_lane=1),
        _lane_decision=LaneDecision('d2',1,1,'requested_lane'),
        _applied_corridor_mode='requested_lane', _last_u=np.array([1.,0.]),
        _mpc_handoff_reason='clear', get_logger=Mock(return_value=Mock()))
    c._mpc_prediction_path_is_clear = Mock(return_value=True)
    c._solve_with_corridor_commit = MethodType(controller_method('_solve_with_corridor_commit'), c)
    return c


def prepare_change(c):
    c._committed_corridor = CorridorState.capture(c)
    c._mpc.model.reference_path.target_lane_idx = None
    c._mpc.model.reference_path.is_overtaking = False
    c._mpc.soft_target_alpha = .4
    c._lane_decision = LaneDecision('d2',1,None,'lane_transition')
    c._applied_corridor_mode = 'lane_transition'
    c._overtake.hybrid.travelled = .7


def test_prepass_pause_and_anchor_restore_together_without_double_pause():
    c = fake_controller()
    c._collision_now = 5.
    c._prepass_soft_guidance_key = ('d2', 2, id(c._reference_path))
    c._prepass_soft_guidance_started_at = 1.
    c._prepass_soft_guidance_paused_at = 3.
    saved = CorridorState.capture(c)
    c._collision_now = 7.
    c._prepass_soft_guidance_key = ('d2', 0, id(c._reference_path))
    saved.restore_progress(c, pause=True)
    assert c._prepass_soft_guidance_key[1] == 2
    assert c._prepass_soft_guidance_started_at == 3.
    assert c._prepass_soft_guidance_paused_at == 5.
    assert c._prepass_soft_guidance_paused_at - c._prepass_soft_guidance_started_at == 2.


def test_rejected_proposal_restores_reference_corridor_and_progress(monkeypatch):
    c = fake_controller()
    prepare_change(c)
    candidate = copy.copy(c._mpc)
    candidate.infeasibility_counter = 1
    candidate.failure_reason = 'primal infeasible'
    candidate.get_control = Mock(return_value=(np.array([0.,.3]), .3))
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.fork_solver', lambda m: candidate)
    c._mpc.get_control = Mock(return_value=(np.array([1.,0.]), .1))
    command, _ = c._solve_with_corridor_commit(NS(), 1.)
    assert tuple(command) == (1.,0.)
    assert c._reference_path.target_lane_idx == 1
    assert c._mpc.soft_target_alpha == 0.
    assert c._overtake.hybrid.travelled == 0.
    assert c._overtake.target_id == 'd2'  # selection wasn't undone
    assert c._lane_decision.applied_lane == 1
    assert not c._corridor_hold_failed


def test_unsafe_previous_corridor_requests_recovery(monkeypatch):
    c = fake_controller()
    saved = object()
    c._remaining_mpc_plan = saved
    prepare_change(c)
    candidate = copy.copy(c._mpc)
    candidate.infeasibility_counter = 1
    candidate.get_control = Mock(return_value=(np.array([0.,.3]), .3))
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.fork_solver', lambda m: candidate)
    c._mpc.get_control = Mock(return_value=(np.array([1.,0.]), .1))
    c._mpc_prediction_path_is_clear.return_value = False
    command, _ = c._solve_with_corridor_commit(NS(), 1.)
    assert command[0] == 0.
    assert c._mpc.recovery_requested and c._corridor_hold_failed
    assert c._remaining_mpc_plan is saved


def test_holding_horizon_rebases_by_waypoint_without_rewinding_pose():
    c = fake_controller()
    c._mpc.soft_lateral_targets = np.arange(4.)
    state = CorridorState.capture(c)
    c._mpc.model.wp_id = 12
    state.apply(c._mpc)
    np.testing.assert_equal(c._mpc.soft_lateral_targets, [2.,3.,3.,3.])
    assert c._mpc.model.wp_id == 12


def test_private_solver_does_not_modify_live_path_and_adopts_exact_solution():
    from .probe_support import configured_mpc
    m = configured_mpc()
    m.model.wp_id = 30
    wp = m.model.reference_path.get_waypoint(30)
    m.model.temporal_state.x, m.model.temporal_state.y = wp.x, wp.y
    m.model.temporal_state.psi = wp.psi
    m.model.reference_path.target_lane_idx = None
    m.model.reference_path.is_overtaking = False
    path, model = m.model.reference_path, m.model
    candidate = fork_solver(m)
    candidate.model.reference_path.target_lane_idx = 1
    candidate.soft_target_alpha = .8
    candidate.get_control()
    assert path.target_lane_idx is None
    assert m.soft_target_alpha == 0.
    assert m.optimizer is not candidate.optimizer
    adopt_solver(m, candidate)
    assert m.model is model and m.model.reference_path is path
    assert m.current_control is candidate.current_control
    assert m.optimizer is candidate.optimizer


def test_fast_motion_aligns_at_command_arrival_and_checks_current_pose_too():
    plan = straight_plan()
    plan.points[:, 0] += 1.2
    plan.points = np.r_[plan.points, [[12.,0.,0.],[15.,0.,0.],[20.,0.,0.]]]
    plan.controls = np.tile([5., 0.], (len(plan.points),1))
    result, _ = rollout(plan, speed=5., pose=(.1,0.,0.), alignment_pose=(1.3,0.,0.))
    assert result is not None
    assert result[1][0] == (.1,0.,0.)


def test_pending_ramp_does_not_advance_while_candidate_is_rejected():
    c = fake_controller()
    c._collision_now = 10.
    c._l1_soft_rejoin_started_at = 9.
    saved = CorridorState.capture(c)
    c._collision_now = 11.
    saved.restore_progress(c, pause=True)
    assert c._l1_soft_rejoin_started_at == 10.


def test_admitted_candidate_transfers_checked_command_without_resolving(monkeypatch):
    c = fake_controller()
    prepare_change(c)
    candidate = copy.copy(c._mpc)
    candidate.model = copy.copy(c._mpc.model)
    candidate.model.reference_path = copy.copy(c._reference_path)
    candidate.model.current_waypoint = None
    c._reference_path.get_waypoint = lambda i: NS()
    candidate.get_control = Mock(return_value=(np.array([1.2,.02]), .1))
    candidate.current_control = np.array([1.2,.02,1.3,.03])
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.fork_solver', lambda m: candidate)
    c._mpc.get_control = Mock(side_effect=AssertionError('must adopt, not solve again'))
    command, _ = c._solve_with_corridor_commit(NS(), 1.)
    np.testing.assert_equal(command, [1.2,.02])
    assert c._reference_path.target_lane_idx is None
    assert c._mpc.soft_target_alpha == .4
    assert c._mpc.current_control is candidate.current_control
    assert c._overtake.hybrid.travelled == .7


def continuation_controller():
    c = fake_controller()
    c._continue_checked_mpc = MethodType(controller_method('_continue_checked_mpc'), c)
    c._enable_control = True
    c._current_gear_is_drive = lambda: True
    c._mpc_safety_recovery_active = c._post_reverse_full_width_recovery_active = False
    c._steering_fallback_armed = False
    c._last_u = np.array([2.,0.])
    c._mpc_cfg = NS(a_min=-2.5)
    c._steering_command_delay = .15
    c._steering_command_history = [(9.8,0.)]
    c._predict_pose_after_steering_delay = lambda p,v,t: p
    c._continuation_path_is_clear = Mock(return_value=(True, 'clear'))
    c._mpc.model.Ts = .025
    c._mpc.model.length = 1.087
    c._mpc.max_steering_rate = 1.2
    c._mpc.understeer_coeff = .001
    c._mpc.infeasibility_counter = 1
    c._mpc.used_prediction_fallback = True
    c._remaining_mpc_plan = straight_plan()
    c._remaining_mpc_plan.path_id = c._reference_path
    c._remaining_mpc_plan.lane = c._reference_path.target_lane_idx
    return c


@pytest.mark.parametrize('reason', ['clear', 'wall', 'vehicle_collision=d2', 'unknown_vehicle=d3'])
def test_real_controller_rechecks_reused_path_before_owning_control(reason):
    c = continuation_controller()
    original = np.array([0.,.3])
    c._continuation_path_is_clear.return_value = reason == 'clear', reason
    result = c._continue_checked_mpc(NS(x=.06,y=0.,theta=0.), 2., 10.025, original, False)
    assert c._mpc_continuation_active == (reason == 'clear')
    assert c._mpc.infeasibility_counter == 1  # failure evidence is not erased
    assert c._remaining_mpc_plan.stamp == 10.  # reused plans never renew freshness
    c._continuation_path_is_clear.assert_called_once()
    assert (result[0] > 0.) == (reason == 'clear')


@pytest.mark.parametrize('case', ['path', 'lane', 'fourth_failure', 'manual', 'reverse', 'recovery'])
def test_reuse_cannot_cross_ownership_or_corridor_changes(case):
    c = continuation_controller()
    if case == 'path': c._remaining_mpc_plan.path_id = object()
    if case == 'lane': c._remaining_mpc_plan.lane = 2
    if case == 'fourth_failure': c._mpc.infeasibility_counter = 4
    if case == 'manual': c._enable_control = False
    if case == 'reverse': c._current_gear_is_drive = lambda: False
    original = np.array([0.,.3])
    result = c._continue_checked_mpc(NS(x=.06,y=0.,theta=0.), 2., 10.025, original, case == 'recovery')
    assert not c._mpc_continuation_active
    np.testing.assert_equal(result, original)
    c._continuation_path_is_clear.assert_not_called()


def test_instrumented_private_solver_never_calls_live_bound_wrappers():
    from .probe_support import configured_mpc
    from multi_purpose_mpc_ros.core.runtime_diagnostics import RuntimeDiagnostics
    m = configured_mpc()
    diagnostics = RuntimeDiagnostics(.025)
    diagnostics.instrument(m, '_init_problem', 'live_mpc._init_problem')
    diagnostics.instrument(m, 'update_prediction', 'live_mpc.update_prediction')
    m._runtime_role = lambda: 'live_mpc'
    before = m.debug_counter
    private = fork_solver(m)
    assert '_init_problem' not in vars(private)
    assert 'update_prediction' not in vars(private)
    private.get_control()
    assert m.debug_counter == before
    assert m.current_prediction is None
    adopt_solver(m, private)
    assert m._runtime_role() == 'live_mpc'
    assert m._init_problem._timing_original.__self__ is m


def test_fresh_solution_after_continuation_needs_no_feedback_confirmation():
    c = continuation_controller()
    p = NS(x=.06,y=0.,theta=0.)
    c._continue_checked_mpc(p, 2., 10.025, np.array([0.,.3]), False)
    assert c._mpc_continuation_active
    c._mpc.infeasibility_counter = 0
    c._mpc.used_prediction_fallback = False
    command = np.array([1.5,.02])
    result = c._continue_checked_mpc(p, 2., 10.05, command, False)
    assert result is command
    assert not c._mpc_continuation_active and not c._steering_fallback_armed


def test_candidate_rejection_preserves_selectors_latest_proof(monkeypatch):
    c = fake_controller()
    c._overtake.probe.vehicle_id = 'd2'
    c._overtake.probe.confirmed = True
    c._committed_corridor = CorridorState.capture(c)
    c._corridor_request_session = c._overtake
    original = c._overtake
    c._overtake = copy.deepcopy(original)
    c._overtake.clear_proof()  # staged full-width apply()
    c._reference_path.target_lane_idx = None
    c._mpc.soft_target_alpha = .4
    candidate = copy.copy(c._mpc)
    candidate.infeasibility_counter = 1
    candidate.get_control = Mock(return_value=(np.array([0.,0.]),0.))
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.fork_solver', lambda m: candidate)
    c._mpc.get_control = Mock(return_value=(np.array([1.,0.]),0.))
    c._solve_with_corridor_commit(NS(), 1.)
    assert c._overtake is original
    assert c._overtake.probe.confirmed
    assert c._overtake.probe.vehicle_id == 'd2'


def test_accepted_continuation_bypasses_only_entry_not_invalid_solution_evidence():
    import ast
    from .test_hybrid_integration import control_tree
    tree = control_tree()
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and 'not self._mpc_continuation_active' in ast.unparse(n.test))
    for continuation in (True,False):
        c = NS(_steering_fallback_armed=False, _mpc_continuation_active=continuation,
               _mpc=NS(failure_reason='primal infeasible'),
               _mpc_continuation_reason='clear', get_logger=Mock(return_value=Mock()))
        scope = dict(self=c, solution_valid=False)
        exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), '<entry>', 'exec'), scope)
        assert c._steering_fallback_armed == (not continuation)
        assert c._mpc.failure_reason == 'primal infeasible'


def test_new_solve_after_continuation_still_requires_current_collision_check():
    c = continuation_controller()
    c._mpc_continuation_active = True
    c._mpc.infeasibility_counter = 0
    c._mpc.used_prediction_fallback = False
    c._mpc_prediction_path_is_clear.return_value = False
    result = c._continue_checked_mpc(NS(x=.06,y=0.,theta=0.), 2.,10.05,np.array([2.,.02]),False)
    assert result[0] == 0.
    assert c._mpc.recovery_requested
    assert not c._mpc_continuation_active


def test_corridor_probe_warm_start_is_private_and_used_once(monkeypatch):
    from .probe_support import configured_mpc
    from multi_purpose_mpc_ros.core.control_continuity import fresh_solution
    m = configured_mpc()
    wp = m.model.reference_path.get_waypoint(30)
    m.model.update_states(wp.x, wp.y, wp.psi)
    m.get_control()
    assert fresh_solution(m)
    probe = fork_solver(m)
    assert probe._continuity_warm_start is not m.last_solution_primal
    hint = probe._continuity_warm_start.copy()
    calls = []
    original = type(probe.optimizer).warm_start
    def record(optimizer, **kwargs):
        calls.append(kwargs)
        return original(optimizer, **kwargs)
    monkeypatch.setattr(type(probe.optimizer), 'warm_start', record)
    probe.get_control()
    assert probe._continuity_warm_start is None
    assert fresh_solution(probe)
    assert len(calls) == 1
    np.testing.assert_equal(calls[0]['x'], hint)


def test_exception_during_private_candidate_does_not_leave_partial_application(monkeypatch):
    c = fake_controller()
    prepare_change(c)
    candidate = copy.copy(c._mpc)
    candidate.get_control = Mock(side_effect=ValueError('invalid matrix'))
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.fork_solver', lambda m: candidate)
    live = c._mpc
    live.get_control = Mock(return_value=(np.array([1.,0.]), .1))
    command, _ = c._solve_with_corridor_commit(NS(), 1.)
    assert c._mpc is live
    assert c._reference_path.target_lane_idx == 1
    assert c._mpc.soft_target_alpha == 0.
    assert tuple(command) == (1.,0.)


def hybrid_update_controller():
    from multi_purpose_mpc_ros.overtake_session import HybridTransition
    c = fake_controller()
    c._reference_path.target_lane_idx = 0
    c._lane_decision = LaneDecision('d2', 0, 0, 'hybrid_lane_transition')
    c._applied_corridor_mode = 'hybrid_lane_transition'
    c._overtake.accepted_key = ('d2', 0)
    c._overtake.hybrid = HybridTransition(vehicle_id='d2', lane_idx=0,
        start_wp=10, started_at=1., start_e_y=.2, length=10., travelled=1.)
    c._mpc.soft_lateral_targets = np.array([.2, .4, .6])
    c._mpc.lane_transition_weights = np.array([.1, .2, .3])
    c._committed_corridor = CorridorState.capture(c)
    c._overtake.hybrid.travelled = 1.6
    c._mpc.soft_lateral_targets += .1
    c._mpc.lane_transition_weights += .1
    c._mpc.get_control = Mock(return_value=(np.array([2., .1]), .2))
    return c


def test_admitted_hybrid_horizon_updates_solve_once_and_still_check_traffic(monkeypatch):
    c = hybrid_update_controller()
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.fork_solver',
                        Mock(side_effect=AssertionError('not a new manoeuvre')))
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 2.
    c._mpc.get_control.assert_called_once()
    c._mpc_prediction_path_is_clear.assert_called_once()
    assert c._committed_corridor.hybrid.travelled == 1.6
    np.testing.assert_allclose(c._committed_corridor.reference['soft_lateral_targets'], [.3,.5,.7])


@pytest.mark.parametrize('change', ['target', 'lane', 'mode', 'anchor', 'restart', 'goal', 'pause', 'backwards'])
def test_changed_manoeuvre_cannot_use_routine_update(change):
    c = hybrid_update_controller()
    if change == 'target': c._overtake.target_id = 'd3'
    if change == 'lane': c._reference_path.target_lane_idx = 2
    if change == 'mode': c._applied_corridor_mode = 'full_width_recovery'
    if change == 'anchor': c._overtake.hybrid.start_e_y = .5
    if change == 'restart': c._overtake.hybrid.started_at = 2.
    if change == 'goal': c._mpc.target_lane_lateral_offsets = [.5,.5,.5]
    if change == 'pause': c._overtake.hybrid.paused = True
    if change == 'backwards': c._overtake.hybrid.travelled = .8
    assert not c._committed_corridor.continues_transition(CorridorState.capture(c))


@pytest.mark.parametrize('failure', ['solver', 'traffic'])
def test_failed_routine_update_rolls_back_progress_without_second_solve(failure):
    c = hybrid_update_controller()
    if failure == 'solver':
        c._mpc.infeasibility_counter = 1
        c._mpc.failure_reason = 'primal infeasible'
    else:
        c._mpc_prediction_path_is_clear.return_value = False
        c._mpc_handoff_reason = 'vehicle_collision=d3'
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    c._mpc.get_control.assert_called_once()
    assert c._overtake.hybrid.travelled == 1.
    np.testing.assert_equal(c._mpc.soft_lateral_targets, [.2,.4,.6])
    if failure == 'traffic':
        assert command[0] == 0. and c._mpc.recovery_requested
    else:
        assert c._mpc.infeasibility_counter == 1


def test_prediction_time_survives_real_solver_fork_and_adoption():
    from .probe_support import configured_mpc
    from multi_purpose_mpc_ros.core.control_continuity import timed_mpc_path, fresh_solution
    m = configured_mpc()
    wp = m.model.reference_path.get_waypoint(30)
    m.model.update_states(wp.x, wp.y, wp.psi)
    m.get_control()
    assert fresh_solution(m)
    assert len(m.current_prediction_times) == len(m.current_recovery_prediction)
    np.testing.assert_allclose(m.current_prediction_times,
        m.last_solution_primal[:m.nx*(m.N+1)].reshape(-1,m.nx)[:m.N,2])
    candidate = fork_solver(m)
    candidate.get_control()
    adopt_solver(m, candidate)
    assert m.current_prediction_times == candidate.current_prediction_times
    assert timed_mpc_path(m, NS(x=wp.x, y=wp.y, theta=wp.psi), .15) is not None


def test_mpc_times_interpolate_nonuniform_arrival_and_command_delay():
    from multi_purpose_mpc_ros.core.control_continuity import timed_mpc_path
    m = NS(current_recovery_prediction=((.2,0.,0.),(1.,0.,0.),(2.,0.,0.)),
           current_prediction_times=(0.,.25,2.))
    path, times = timed_mpc_path(m, NS(x=0.,y=0.,theta=0.), .15)
    assert times[0] == 0.
    for x, expected in [(.2,.15),(1.,.4),(2.,2.15)]:
        index = next(i for i,p in enumerate(path) if abs(p[0]-x) < 1e-9)
        assert times[index] == pytest.approx(expected)


@pytest.mark.parametrize('times', [None, (0.,1.), (0.,float('nan'),2.), (0.,1.,.5), (1.,2.,3.)])
def test_invalid_timing_never_falls_back_to_constant_speed(times):
    from multi_purpose_mpc_ros.core.control_continuity import timed_mpc_path
    m = NS(current_recovery_prediction=((0.,0.,0.),(1.,0.,0.),(2.,0.,0.)),
           current_prediction_times=times)
    assert timed_mpc_path(m, NS(x=0.,y=0.,theta=0.)) is None


def test_actual_arrival_times_change_crossing_vehicle_verdict_without_disabling_collision():
    from multi_purpose_mpc_ros import collision_geometry as cg
    from multi_purpose_mpc_ros.core.control_continuity import timed_mpc_path
    geometry = cg.BodyGeometry(length=.1, width=.1)
    target = cg.BodyPose(x=2.,y=-2.,yaw=np.pi/2,stamp=0.,direction_valid=True)
    m = NS(current_recovery_prediction=((0.,0.,0.),(2.,0.,0.),(4.,0.,0.)),
           current_prediction_times=(0.,.25,2.))
    path, times = timed_mpc_path(m, NS(x=0.,y=0.,theta=0.))
    bodies = [cg.BodyPose(x=x,y=y,yaw=yaw,stamp=0.) for x,y,yaw in path]
    assert cg.swept_path_clear(bodies, times, target, (0.,2.), geometry)
    # One constant speed predicts a collision at x=2,t=1 that the solution
    # avoids. A genuinely conflicting MPC time sequence must still be refused.
    m.current_prediction_times = (0.,1.,2.)
    _, conflicting_times = timed_mpc_path(m, NS(x=0.,y=0.,theta=0.))
    assert not cg.swept_path_clear(bodies, conflicting_times, target, (0.,2.), geometry)


@pytest.mark.parametrize('reason', ['clear', 'wall', 'vehicle_collision=d2', 'unknown_vehicle=d3'])
def test_held_corridor_rejection_can_only_use_independently_checked_remainder(reason):
    c = continuation_controller()
    c._corridor_hold_failed = True
    c._mpc.infeasibility_counter = 0  # solve succeeded, additional check rejected
    c._mpc.recovery_requested = True
    c._mpc.failure_reason = 'held corridor is no longer executable'
    c._continuation_path_is_clear.return_value = reason == 'clear', reason
    result = c._continue_checked_mpc(NS(x=.06,y=0.,theta=0.), 2., 10.025, np.array([0.,.3]), False)
    assert c._mpc_continuation_active == (reason == 'clear')
    assert (result[0] > 0.) == (reason == 'clear')
    c._continuation_path_is_clear.assert_called_once()
    assert c._mpc.recovery_requested and c._corridor_hold_failed
    assert c._remaining_mpc_plan.stamp == 10.


def test_held_corridor_remainder_expires_even_when_solver_failure_counter_is_zero():
    c = continuation_controller()
    c._corridor_hold_failed = True
    c._mpc.infeasibility_counter = 0
    c._mpc.recovery_requested = True
    c._continue_checked_mpc(NS(x=.06,y=0.,theta=0.), 2., 10.1, np.array([0.,.3]), False)
    assert not c._mpc_continuation_active
    assert c._mpc_continuation_reason == 'expired'
    c._continuation_path_is_clear.assert_not_called()
