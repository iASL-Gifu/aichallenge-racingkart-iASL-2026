"""Actual controller permission and speed ordering, independent of ROS."""
import ast
import copy
import math
from types import SimpleNamespace, MethodType

import pytest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import rolling_precommit_speed_margin
from .test_overtake_session import controller_method
from .test_hybrid_integration import control_tree
from .test_overtake_lane_hold import controller, POSE
from .test_v2x_vehicle_tracker import _msg


def escape_controller():
    c=controller()
    c._overtake.target_id='target'
    c._overtake.requested_lane=2
    h=c._overtake.hybrid
    h.vehicle_id='target'; h.lane_idx=2
    c._reference_path=SimpleNamespace(target_lane_idx=2,is_overtaking=True)
    c._follow_only=False
    c._strict_shadow_commit_creep_max_target_speed=1.5
    c._hybrid_escape_min_lateral_gap=.1
    c._hybrid_escape_creep_speed=.6
    c._prepass_lane_fallback_front_distance=10.
    c._prepass_lane_fallback_side_distance=3.
    c._prepass_lane_fallback_rear_distance=10.
    c._current_center_envelopes_are_separated=lambda *a: (False,dict(rectangles_overlap=False,lateral_gap=.1))
    c._latched_target_passage=lambda *a: ({2:True},None)
    c._relative_lane_vehicle_samples=lambda *a: []
    c._v2x_tracker.update(_msg(0.,[('target',3.,0.)]))
    c._v2x_tracker.update(_msg(.1,[('target',3.,0.)]))
    c._hybrid_escape_speed=MethodType(controller_method('_hybrid_escape_speed'),c)
    return c


def test_unverified_spatial_transition_with_lateral_separation_can_creep():
    c=escape_controller()
    assert c._overtake.verification.vehicle_id is None
    assert c._hybrid_escape_speed(POSE,0.)==.6


@pytest.mark.parametrize('condition',['paused','completed','target_change','lane_change','unknown_velocity',
    'overlap','small_gap','no_envelope','width','rear_unknown','front','follow_only'])
def test_escape_rejects_invalid_or_unsafe_state(condition):
    c=escape_controller()
    if condition in ('paused','completed'):setattr(c._overtake.hybrid,condition,True)
    elif condition=='target_change':c._overtake.target_id='other'
    elif condition=='lane_change':c._reference_path.target_lane_idx=0
    elif condition=='unknown_velocity':c._v2x_tracker._velocity_valid['target']=False
    elif condition=='overlap':c._current_center_envelopes_are_separated=lambda *a:(False,dict(rectangles_overlap=True,lateral_gap=.2))
    elif condition=='small_gap':c._current_center_envelopes_are_separated=lambda *a:(False,dict(rectangles_overlap=False,lateral_gap=.099))
    elif condition=='no_envelope':c._current_center_envelopes_are_separated=lambda *a:(False,None)
    elif condition=='width':c._lane_horizon_has_vehicle_width=lambda *a:False
    elif condition=='rear_unknown':c._v2x_tracker.update(_msg(.2,[('target',3.,0.),('rear',-5.,2.)]))
    elif condition=='front':c._relative_lane_vehicle_samples=lambda *a:[('target',2,5.)]
    elif condition=='follow_only':c._follow_only=True
    assert c._hybrid_escape_speed(POSE,0.)==0.


def actual_speed_nodes():
    # Execute the actual target-specific emergency cap and final min in order.
    for parent in ast.walk(control_tree()):
        for _,body in ast.iter_fields(parent):
            if not isinstance(body,list):continue
            for i,n in enumerate(body):
                if isinstance(n,ast.If) and ast.unparse(n.test)=='vid == hybrid_escape_target and hybrid_escape_speed > 0.0':
                    return copy.deepcopy(body[i:i+2])
    raise AssertionError('emergency creep clamp not found')


@pytest.mark.parametrize('order',[('other','target'),('target','other')])
@pytest.mark.parametrize('other_limit',[0.,.2])
def test_other_vehicle_cap_survives_emergency_iteration_order(order,other_limit):
    ns=dict(ref_vel_kmph=8.,hybrid_escape_target='target',hybrid_escape_speed=.6,strict_commit_creep=False)
    code=compile(ast.Module(body=actual_speed_nodes(),type_ignores=[]),'<actual-emergency-cap>','exec')
    for vid in order:
        ns.update(vid=vid,v_ref_emg=0. if vid=='target' else other_limit)
        exec(code,ns)
    assert ns['ref_vel_kmph']==other_limit


@pytest.mark.parametrize('cap,expected',[(0.,0.),(.2,.2),(5.,.6)])
def test_actual_creep_command_respects_all_preceding_reference_limits(cap,expected):
    node=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.If) and any(
        isinstance(b,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='creep_command' for t in b.targets)
        for b in n.body))
    c=escape_controller()
    for n in ast.walk(node.test):
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and not hasattr(c,n.attr):
            setattr(c,n.attr,False)
    c._stuck_recovery_until=None
    ns=dict(self=c,hybrid_escape_speed=.6,hybrid_escape_target='target',
        intentional_follow_stop_active=False,close_overtake_blocked=False,
        fallback_stop_requested=False,pure_pursuit_safe_this_cycle=True,
        ref_vel_kmph=cap,u=[0.])
    exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),'<actual-creep-command>','exec'),ns)
    assert ns['u'][0]==expected


@pytest.mark.parametrize('distance,expected',[(0.,.8),(10.,.8),(22.5,1.9),(35.,3.),(45.,3.)])
def test_old_distance_dependent_margin(distance,expected):
    assert rolling_precommit_speed_margin(base_margin=.8,far_bonus=2.2,
        target_distance=distance,commit_distance=10.,prepare_distance=35.)==pytest.approx(expected)


@pytest.mark.parametrize('lane,overtaking,expected',[(1,False,3.),(2,True,2.5),(0,True,2.5)])
def test_actual_controller_uses_configured_rolling_or_committed_margin(lane,overtaking,expected):
    from pathlib import Path
    import yaml
    cfg=yaml.safe_load((Path(__file__).parents[1]/'config/config.yaml').read_text())['trajectory_switch']
    c=SimpleNamespace(_reference_path=SimpleNamespace(target_lane_idx=lane,is_overtaking=overtaking))
    for name in ['slow_lead_overtake_speed_margin','slow_lead_overtake_far_speed_bonus',
                 'slow_lead_overtake_speed_margin_fade_distance','slow_lead_overtake_prepare_distance',
                 'slow_lead_overtake_committed_speed_margin']:
        setattr(c,'_'+name,cfg[name])
    node=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.Assign)
        and isinstance(n.value,ast.IfExp) and any(isinstance(t,ast.Name) and t.id=='slow_lead_active_speed_margin' for t in n.targets))
    ns=dict(self=c,opponent_distance=35.,rolling_precommit_speed_margin=rolling_precommit_speed_margin)
    exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),'<actual-margin>','exec'),ns)
    assert ns['slow_lead_active_speed_margin']==expected
