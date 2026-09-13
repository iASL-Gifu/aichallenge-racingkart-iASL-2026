"""A defensive inner lane is not an unsuccessful pass; safety owners still win."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import ast
from pathlib import Path
import pytest
from multi_purpose_mpc_ros.core.curve_priority import retained_priority, cancel_ordinary_rejoin
from .test_overtake_session import controller_method


def controller(wp=40,lane=0):
    path=NS(target_lane_idx=lane,is_overtaking=True,segment_lengths=[1.]*354)
    return NS(_reference_path=path,_reference_pathN_center=path,
        _carN_center=NS(wp_id=wp),_curve_lane_priority_prepare_distance=30.,
        _curve_lane_priority_zones=[(30,60,0),(80,105,2),(185,215,0)],
        _overtake=NS(committed=True),_reset_outer_lane_progress=Mock(),
        _lane_lateral_error=lambda *a:0.,_slow_lead_speed_match_release_lateral_error=.45,
        _outer_lane_progress_vehicle_id='lead',_outer_lane_progress_best_longitudinal=10.,
        _outer_lane_last_progress_at=0.,_outer_lane_min_progress=.5,
        _outer_lane_progress_timeout=2.5,_l1_rejoin_traffic_is_clear=Mock(return_value=(True,{})),
        _clear_prepass_soft_guidance=Mock(),_reset_overtake_commit_probe=Mock(),
        _clear_committed_shadow_verification=Mock(),_mpc=NS(osqp_initialized=True),
        get_logger=lambda:Mock())


@pytest.mark.parametrize('wp,lane',[(40,0),(90,2),(190,0)])
def test_timeout_keeps_applied_inner_lane_and_zone_exit_restores_normal_release(wp,lane):
    c=controller(wp,lane)
    method=controller_method('_update_outer_lane_progress_state')
    args=dict(target_id='lead',longitudinal=10.,pose=NS(x=0.,y=0.),ego_speed=5.,now_sec=10.)
    assert not method(c,**args)
    assert c._overtake.committed
    c._l1_rejoin_traffic_is_clear.assert_not_called()
    c._clear_prepass_soft_guidance.assert_not_called()
    c._carN_center.wp_id=220
    assert method(c,**args)
    assert not c._overtake.committed
    c._clear_prepass_soft_guidance.assert_called_once()


@pytest.mark.parametrize('owner',[
    '_mpc_safety_recovery_active','_post_reverse_full_width_recovery_active',
    '_prepass_fallback_recovery_active','_prepass_fallback_commit_pending',
    '_prepass_fallback_follow_active','_prepass_fallback_blocked',
    '_follow_escape_active','_follow_only','_l1_safety_recovery_active',
    '_l1_safety_reprobe_pending','_l1_rejoin_backoff_active','_parallel_abort_active',
    '_straight_reentry_active','_close_obstacle_reverse_requested',
    '_center_lane_rejoin_constraint_released'])
def test_priority_never_overrides_recovery_or_safety_owner(owner):
    c=controller();setattr(c,owner,True)
    assert retained_priority(c) is None


def test_opposite_slow_pass_and_race_are_not_forced_into_retention():
    c=controller(90,0)
    assert retained_priority(c) is None
    c=controller()
    c._reference_path=NS(target_lane_idx=0)
    assert retained_priority(c) is None


@pytest.mark.parametrize('context',['rejoin','fallback','safety'])
def test_cancel_only_voluntary_rejoin_probes(context):
    c=controller()
    c._center_lane_rejoin_active=True
    c._race_rejoin_handoff_active=True
    c._reset_race_rejoin_handoff=Mock()
    c._l1_probe_context=context;c._l1_probe_active=True;c._l1_probe_success_cycles=8
    cancel_ordinary_rejoin(c)
    assert not c._center_lane_rejoin_active
    c._reset_race_rejoin_handoff.assert_called_once()
    assert c._l1_probe_active == (context!='rejoin')
    assert c._l1_probe_success_cycles == (0 if context=='rejoin' else 8)


def test_actual_trajectory_hold_branch_preserves_center_and_cancels_pending_return():
    # Execute the production branch rather than a duplicated state machine.
    source=(Path(__file__).parents[1]/'multi_purpose_mpc_ros/mpc_controller.py').read_text()
    tree=ast.parse(source)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
              and ast.unparse(n.test)=='priority_hold_lane is not None')
    body=compile(ast.Module(body=node.body,type_ignores=[]),'<actual priority hold>','exec')
    c=controller();c._center_lane_rejoin_active=True
    c._race_rejoin_handoff_active=True;c._reset_race_rejoin_handoff=Mock()
    c._l1_probe_active=True;c._l1_probe_context='rejoin';c._l1_probe_success_cycles=7
    ns=dict(self=c,priority_hold_lane=0,center_wp_temp=40,opponent_ahead_detected=False,
            cancel_ordinary_rejoin=cancel_ordinary_rejoin)
    exec(body,ns)
    assert ns['opponent_ahead_detected']
    assert c._trajectory_switch_reason=='curve_priority_hold'
    assert not c._center_lane_rejoin_active and not c._l1_probe_active


@pytest.mark.parametrize('wp,lane',[(20,0),(70,2),(180,0),(300,0)])
def test_preparation_does_not_suppress_return(wp,lane):
    c=controller(wp,lane)
    c._l2_entry_restricted_zones=[(325,6)]
    c._l0_priority_zone_active=lambda wp:True
    assert retained_priority(c) is None


@pytest.mark.parametrize('wp',[325,350,0,6])
def test_end_lap_actual_zone_retains_inner_lane(wp):
    c=controller(wp,0);c._l2_entry_restricted_zones=[(325,6)]
    assert retained_priority(c)==0


def test_unfinished_lateral_motion_cannot_reset_stalled_pass():
    c=controller(220,0)
    c._lane_lateral_error=lambda *a:.98
    c._constraint_transition_until=100.
    method=controller_method('_update_outer_lane_progress_state')
    args=dict(target_id='lead',longitudinal=10.,pose=NS(x=0.,y=0.),ego_speed=5.,now_sec=10.)
    assert method(c,**args)
    assert not c._overtake.committed


def test_unfinished_pass_keeps_waiting_when_merge_is_occupied():
    c=controller(220,0);c._lane_lateral_error=lambda *a:.98
    c._l1_rejoin_traffic_is_clear=Mock(return_value=(False,{'side':['other']}))
    method=controller_method('_update_outer_lane_progress_state')
    args=dict(target_id='lead',longitudinal=10.,pose=NS(x=0.,y=0.),ego_speed=5.,now_sec=10.)
    assert not method(c,**args)
    assert c._overtake.committed
    c._reset_outer_lane_progress.assert_not_called()
    c._clear_prepass_soft_guidance.assert_not_called()


def test_preparation_final_policy_does_not_overwrite_requested_return():
    source=(Path(__file__).parents[1]/'multi_purpose_mpc_ros/mpc_controller.py').read_text()
    tree=ast.parse(source)
    gate=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and 'restricted_outer_lane' in ast.unparse(n.test)
        and 'recovery_lane_owner(self)' in ast.unparse(n.test))
    guard=next(n for n in gate.test.values if isinstance(n,ast.UnaryOp)
        and 'new_target_lane_idx' in ast.unparse(n))
    compiled=compile(ast.Expression(guard),'<priority return guard>','eval')
    c=controller(300,0);c._l2_entry_restricted_zones=[(325,6)]
    for lane in (None,1):
        assert not eval(compiled,dict(self=c,new_target_lane_idx=lane,retained_priority=retained_priority))
    c._carN_center.wp_id=330
    assert eval(compiled,dict(self=c,new_target_lane_idx=1,retained_priority=retained_priority))
