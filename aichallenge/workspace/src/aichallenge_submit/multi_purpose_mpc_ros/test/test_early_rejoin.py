from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from .test_control_continuity import fake_controller
from .test_postpass_rejoin import propose_return
from multi_purpose_mpc_ros.core.control_continuity import CorridorState
from multi_purpose_mpc_ros.core.early_rejoin import prepare, speed_handoff_clear
from .test_overtake_session import controller_method


def passing():
    c = fake_controller()
    c._collision_now = 10.
    c._reference_path.target_lane_idx = 2
    c._overtake.requested_lane = 2
    c._committed_corridor = CorridorState.capture(c)
    c._return_target_is_separated = Mock(return_value=True)
    return c


@pytest.mark.parametrize('safe,separated', [(False, True), (True, False)])
def test_rejected_return_preserves_pass_and_throttles_only_trials(monkeypatch, safe, separated):
    c = passing()
    assert prepare(c, 1)
    c._overtake.requested_lane = None
    c._return_target_is_separated.return_value = separated
    propose_return(c, monkeypatch, safe=safe)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 2.
    assert c._reference_path.target_lane_idx == 2
    assert c._overtake.requested_lane == 2
    assert not prepare(c, 1)
    c._collision_now += .25
    assert prepare(c, 1)
    # Previous/current outer path was solved and checked despite trial failure.
    assert c._mpc.get_control.call_count == 1


def test_clear_return_commits_immediately_with_speed_phase(monkeypatch):
    c = passing()
    assert prepare(c, 1)
    c._overtake.requested_lane = None
    candidate = propose_return(c, monkeypatch)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 3.
    assert c._reference_path.target_lane_idx is None
    assert c._overtake.requested_lane is None
    assert candidate.get_control.call_count == 1
    assert c._checked_return_target == (id(c._reference_path), c._overtake.target_id)
    c._mpc_prediction_path_is_clear = Mock(return_value=True)
    assert speed_handoff_clear(c, c._overtake.target_id, NS(), command)
    c._return_target_is_separated.return_value = False
    assert not speed_handoff_clear(c, c._overtake.target_id, NS(), command)


def test_unsafe_outer_is_not_held_after_failed_return(monkeypatch):
    c = passing()
    prepare(c, 1)
    c._overtake.requested_lane = None
    propose_return(c, monkeypatch, safe=False, outer_safe=False)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 0.
    assert c._mpc.recovery_requested


@pytest.mark.parametrize('change', ['target', 'path', 'recovery', 'prediction'])
def test_return_speed_phase_cannot_bypass_new_target_or_safety(change):
    c = passing()
    c._checked_return_target = (id(c._reference_path), c._overtake.target_id)
    c._mpc_prediction_path_is_clear = Mock(return_value=True)
    target = c._overtake.target_id
    if change == 'target': c._overtake.target_id = 'new'
    if change == 'path': c._reference_path = NS()
    if change == 'recovery': c._mpc_safety_recovery_active = True
    if change == 'prediction': c._mpc_prediction_path_is_clear.return_value = False
    assert not speed_handoff_clear(c, target, NS(), [3., 0.])


@pytest.mark.parametrize('x,y,expected', [(-4.,0.,True), (4.,0.,True),
                                         (0.,3.,True), (1.,0.,False)])
def test_return_separation_handles_passed_car_behind(monkeypatch,x,y,expected):
    from multi_purpose_mpc_ros import collision_geometry as cg
    ego = cg.BodyPose(0.,0.,0.,10.,direction_valid=True)
    target = cg.BodyPose(x,y,0.,10.,direction_valid=True)
    monkeypatch.setattr(cg,'ego_body',lambda *a:ego)
    monkeypatch.setattr(cg,'target_body',lambda *a:target)
    assert controller_method('_return_target_is_separated')(
        NS(_parallel_critical_clearance=.3),NS(),'d2') is expected
