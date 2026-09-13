"""Connection ownership, combined traffic timing and endpoint admission."""
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from multi_purpose_mpc_ros.core import recovery_connection as rc
from multi_purpose_mpc_ros.core.boundary_recovery import Motion
from .test_boundary_recovery import choose


def test_short_prefix_with_solvable_endpoint_preferred_to_long_motion():
    seen=[]
    def connection(m):
        seen.append(m)
        return m.poses[-1][0] < 1.1, 'endpoint'
    motion, reason=choose(connection_check=connection)
    assert reason=='forward_mpc_connection'
    assert motion.poses[-1][0] == pytest.approx(1.)
    assert .05+.5*motion.speed_limit+.5*motion.speed_limit**2 <= 1.+1e-8


def test_failed_connection_preserves_safe_escape_without_claiming_admission():
    motion,reason=choose(connection_check=lambda m:(False,'traffic'))
    assert motion is not None and reason!='forward_mpc_connection'


def controller():
    path=object()
    return NS(_reference_path=path,_reference_pathN_center=path,
        _overtake=NS(target_id='car',requested_lane=2),
        _straight_reentry_speed=1.,_straight_reentry_active=True,
        _velocity_report=NS(longitudinal_velocity=0.),
        _mpc=NS(max_steering_rate=.6),_collision_now=5.,
        _reentry_path_is_clear=Mock(return_value=(True,'clear')))


def fake_probe():
    return NS(model=NS(update_states=Mock()), previous_steering=0.,
        update_wp_id_offset=Mock(), update_v_max=Mock(),
        get_control=Mock(return_value=(np.array([.5,.2]),0.)),
        infeasibility_counter=0,current_prediction=True,used_prediction_fallback=False,
        recovery_requested=False,time_budget_exceeded=False,last_solution_accurate=True)


def test_future_solver_is_not_adopted_and_traffic_times_include_prefix_and_preparation(monkeypatch):
    c=controller();probe=fake_probe()
    monkeypatch.setattr(rc,'fork_solver',lambda live:probe)
    monkeypatch.setattr(rc,'set_guidance',Mock())
    monkeypatch.setattr(rc,'prepare_mpc_path',lambda *a:(((1.,0.,0.),(2.,0.,0.)),(0.,2.)))
    motion=Motion(1,.3,((0.,0.,0.),(1.,0.,0.)),1.,speed_limit=.5)
    assert rc.connection_checker(c,5.,0.)(motion)[0]
    path=c._reentry_path_is_clear.call_args.args[0]
    times=c._reentry_path_is_clear.call_args.kwargs['times']
    assert times==pytest.approx([0.,.75,2.75,4.75])
    assert path[-1]==(2.,0.,0.)
    assert c._mpc is not probe
    assert c._recovery_connection_intent[:2]==('car',2)
    c._reentry_path_is_clear.return_value=False,'vehicle_collision=car'
    assert not rc.connection_checker(c,5.1,0.)(motion)[0]
    assert c._recovery_connection_intent is None


@pytest.mark.parametrize('case',['expired','target','lane','path','manual','inactive','rollback'])
def test_guidance_cannot_cross_owner_or_freshness(monkeypatch,case):
    c=controller();c._recovery_connection_intent=('car',2,id(c._reference_path),5.)
    if case=='expired':c._collision_now=5.51
    if case=='rollback':c._collision_now=4.99
    if case=='target':c._overtake.target_id='other'
    if case=='lane':c._overtake.requested_lane=0
    if case=='path':c._reference_path=object()
    if case=='manual':c._manual_control_override=True
    if case=='inactive':c._straight_reentry_active=False
    setter=Mock();monkeypatch.setattr(rc,'set_guidance',setter)
    assert not rc.apply_guidance(c,c._mpc,0.)
    setter.assert_not_called()


def test_guidance_proposed_before_admission_uses_no_future_solver(monkeypatch):
    c=controller();c._recovery_connection_intent=('car',2,id(c._reference_path),5.)
    setter=Mock();monkeypatch.setattr(rc,'set_guidance',setter)
    assert rc.apply_guidance(c,c._mpc,.2)
    setter.assert_called_once_with(c,c._mpc,2,.2)


def test_real_endpoint_solver_has_full_width_soft_goal_and_keeps_live_state():
    from .probe_support import configured_mpc
    from multi_purpose_mpc_ros.core.control_continuity import fork_solver
    m= configured_mpc();c=controller();c._mpc=m
    c._l2_inward_offset_zones=[]
    for key,value in dict(base_length=6.,offset_gain=2.5,speed_gain=.8,
        low_speed_start_threshold=2.,low_speed_max_length=8.,min_length=8.,max_length=20.,
        continuity_weight=.3,continuity_max_deviation=.75).items():
        setattr(c,'_hybrid_overtake_'+key,value)
    old_pose=(m.model.temporal_state.x,m.model.temporal_state.y,m.model.temporal_state.psi)
    probe=fork_solver(m)
    wp=probe.model.reference_path.get_waypoint(269)
    probe.model.update_states(wp.x,wp.y,wp.psi)
    rc.set_guidance(c,probe,2,1.)
    assert probe.model.reference_path.target_lane_idx is None
    assert probe.lane_transition_weights is None
    assert probe.soft_target_lane_idx==2
    assert np.isfinite(probe.soft_lateral_targets).all()
    probe.update_v_max(1.)
    probe.get_control()
    assert (m.model.temporal_state.x,m.model.temporal_state.y,m.model.temporal_state.psi)==old_pose
    assert m.current_prediction is None


def test_repeated_candidates_keep_integer_budget_and_failure_history(monkeypatch):
    c=controller();probe=fake_probe()
    c._recovery_attempts=NS(wall_path_is_clear=Mock(return_value=(True,'clear')))
    monkeypatch.setattr(rc,'fork_solver',lambda live:probe)
    monkeypatch.setattr(rc,'set_guidance',Mock())
    monkeypatch.setattr(rc,'prepare_mpc_path',lambda *a:(((1.,0.,0.),(2.,0.,0.)),(0.,2.)))
    c._reentry_path_is_clear.return_value=(False,'vehicle_collision=d4')
    check=rc.connection_checker(c,5.,0.)
    motion=Motion(1,.3,((0.,0.,0.),(1.,0.,0.)),1.,speed_limit=.5)
    for _ in range(3):
        assert check(motion)==(False,'vehicle_collision=d4')
    assert check(motion)==(False,'connection_probe_budget')
    assert c._recovery_attempts.wall_path_is_clear.call_count==3
    assert c._recovery_connection_intent is None


def test_prepass_connection_uses_l2_instead_of_failed_l0():
    c=controller();c._overtake.requested_lane=0
    c._prepass_fallback_recovery_active=True;c._prepass_soft_candidate_lane_idx=2
    assert rc.requested_lane(c)==2
    c._manual_control_override=True
    assert rc.requested_lane(c) is None
