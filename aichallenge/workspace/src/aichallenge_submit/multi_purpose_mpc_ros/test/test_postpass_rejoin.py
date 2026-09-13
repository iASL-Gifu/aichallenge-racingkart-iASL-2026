"""Post-pass return admission must retain a freshly safe outer alternative."""
import copy
from unittest.mock import Mock
from types import SimpleNamespace as NS

import numpy as np
import pytest

from .test_control_continuity import fake_controller
from .test_overtake_session import controller_method
from multi_purpose_mpc_ros.core.control_continuity import CorridorState, postpass_alternative
from multi_purpose_mpc_ros.overtake_session import LaneDecision


def completed_pass():
    c = fake_controller()
    c._reference_path.target_lane_idx = 0
    c._overtake.requested_lane = 0
    c._lane_decision = LaneDecision('d2', 0, 0, 'requested_lane')
    c._committed_corridor = CorridorState.capture(c)
    c._prepass_attempted_outer_lanes = set()
    c._clear_prepass_soft_guidance = Mock()
    c._clear_committed_shadow_verification = c._overtake.clear_proof
    c._mpc.osqp_initialized = True
    controller_method('_complete_overtake_target_behind')(c, 'd2', -3., source='test')
    return c


def propose_return(c, monkeypatch, *, lane=None, safe=True, outer_safe=True, solved=True):
    c._reference_path.target_lane_idx = lane
    c._mpc.soft_target_alpha = .6
    c._lane_decision = LaneDecision('d2', 1, lane, 'lane_transition')
    c._applied_corridor_mode = 'lane_transition'
    candidate = copy.copy(c._mpc)
    candidate.get_control = Mock(return_value=(np.array([3., .12]), .3))
    candidate.infeasibility_counter = 0 if solved else 1
    candidate.failure_reason = None if solved else 'infeasible'
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.fork_solver', lambda m: candidate)
    monkeypatch.setattr('multi_purpose_mpc_ros.core.control_continuity.adopt_solver',
                        lambda live, checked: setattr(live, 'current_control', checked.current_control))
    c._mpc.get_control = Mock(return_value=(np.array([2., .04]), .3))
    c._mpc_prediction_path_is_clear = Mock(
        side_effect=lambda *a: safe if c._mpc is candidate else outer_safe)
    return candidate


def test_completion_keeps_applied_geometry_without_solver_reset_or_old_proof():
    c = completed_pass()
    assert c._postpass_outer_corridor.lane == 0
    assert c._reference_path.target_lane_idx == 0
    assert c._mpc.osqp_initialized
    assert not c._postpass_outer_corridor.hybrid.verified_start
    assert not c._overtake.committed
    assert c._overtake_completed_target_id == 'd2'


@pytest.mark.parametrize('solved', [False, True])
def test_failed_return_uses_fresh_safe_outer_mpc(monkeypatch, solved):
    c = completed_pass()
    propose_return(c, monkeypatch, safe=False, solved=solved)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert tuple(command) == (2., .04)
    assert c._reference_path.target_lane_idx == 0
    assert c._mpc.soft_target_alpha == 0.
    assert not c._corridor_hold_failed
    assert not c._overtake.committed
    assert c._mpc.get_control.call_count == 1


def test_return_is_adopted_without_an_unnecessary_outer_solve(monkeypatch):
    c = completed_pass()
    candidate = propose_return(c, monkeypatch)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert tuple(command) == (3., .12)
    assert c._reference_path.target_lane_idx is None
    assert c._mpc.soft_target_alpha == .6
    assert candidate.get_control.call_count == 1
    assert c._mpc.get_control.call_count == 0
    assert c._postpass_outer_corridor.lane == 0


def test_later_full_width_failure_can_still_compare_outer(monkeypatch):
    c = completed_pass()
    propose_return(c, monkeypatch)
    c._solve_with_corridor_commit(NS(), 2.)
    assert c._committed_corridor.lane is None
    propose_return(c, monkeypatch, safe=False)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 2.
    assert c._committed_corridor.lane == 0


def test_both_routes_unsafe_release_to_recovery(monkeypatch):
    c = completed_pass()
    propose_return(c, monkeypatch, safe=False, outer_safe=False)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 0.
    assert c._mpc.recovery_requested and c._corridor_hold_failed
    assert c._postpass_outer_corridor is None


def test_checked_l1_commit_finishes_postpass_comparison(monkeypatch):
    c = completed_pass()
    propose_return(c, monkeypatch, lane=1)
    c._solve_with_corridor_commit(NS(), 2.)
    assert c._postpass_outer_corridor is None


@pytest.mark.parametrize('owner,value', [
    ('_postpass_rejoin_suppressed', True), ('_manual_recovery_reset_pending', True),
    ('_follow_only', True), ('_follow_escape_active', True), ('_prepass_retry_after_reverse', True),
    ('_manual_control_override', True), ('_mpc_safety_recovery_active', True),
    ('_post_reverse_full_width_recovery_active', True), ('_parallel_abort_active', True),
    ('_prepass_fallback_recovery_active', True), ('_prepass_fallback_follow_active', True),
    ('_l1_safety_recovery_active', True), ('_l1_rejoin_backoff_active', True),
    ('_stuck_recovery_until', 0.), ('_straight_reentry_active', True)])
def test_safety_owners_cancel_outer_alternative(owner, value):
    c = completed_pass()
    c._reference_path.target_lane_idx = None
    setattr(c, owner, value)
    assert postpass_alternative(c, CorridorState.capture(c)) is None


@pytest.mark.parametrize('change', ['target', 'path', 'policy'])
def test_new_target_path_or_outer_policy_does_not_inherit_old_rejoin(change):
    c = completed_pass()
    c._reference_path.target_lane_idx = None
    candidate = CorridorState.capture(c)
    if change == 'target': candidate.target = 'd4'
    if change == 'path': candidate.path = object()
    if change == 'policy': candidate.lane = 2
    assert postpass_alternative(c, candidate) is None


def test_completion_after_unsafe_outer_release_does_not_resurrect_it():
    c = completed_pass()
    c._reference_path.target_lane_idx = None
    c._committed_corridor = CorridorState.capture(c)
    controller_method('_complete_overtake_target_behind')(c, 'd2', -3., source='prepass')
    assert c._postpass_outer_corridor is None
