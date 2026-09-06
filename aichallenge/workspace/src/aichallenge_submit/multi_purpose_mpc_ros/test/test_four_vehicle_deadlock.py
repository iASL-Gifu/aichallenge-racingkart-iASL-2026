"""Four-vehicle recovery regressions using actual controller methods."""
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock
import pytest
from .test_overtake_session import controller_method as extracted_method
from multi_purpose_mpc_ros import v2x_vehicle_tracker
from .test_stationary_group_lifecycle import group_controller
from .test_overtake_lane_hold import POSE
from .test_v2x_vehicle_tracker import _msg


def method(name):
    fn=extracted_method(name)
    fn.__globals__.update(vars(v2x_vehicle_tracker))
    return fn


def lane_controller():
    c=group_controller()
    c._follow_escape_attempted_lanes=set()
    c._follow_escape_target_id='d2'
    c._latched_target_passage=lambda *a:({0:False,2:True},None)
    c._relative_lane_vehicle_samples=lambda *a:[('d2',1,4.),('d3',0,4.)]
    return c


def test_free_l2_is_selected_by_follow_escape():
    c=lane_controller()
    lane,conflicts=method('_select_follow_escape_lane')(c,POSE,0.)
    assert not any(conflicts[2].values())
    assert lane==2


@pytest.mark.parametrize('problem',['width','policy','unknown_rear','attempted'])
def test_l2_selection_retains_safety_and_retry_guards(problem):
    c=lane_controller()
    if problem=='width':c._lane_horizon_has_vehicle_width=lambda *a:False
    if problem=='policy':c._apply_l2_restricted_zone_policy=lambda *a,**kw:1
    if problem=='attempted':c._follow_escape_attempted_lanes={2}
    if problem=='unknown_rear':c._v2x_tracker.update(_msg(.2,[('d2',0.,0.),('rear',-4.,2.)]))
    assert method('_select_follow_escape_lane')(c,POSE,0.)[0] is None


def test_abort_without_forward_group_member_releases_without_motion_permission():
    c=group_controller()
    for t in (.2,.3):c._v2x_tracker.update(_msg(t,[('d2',0.,0.),('d3',-8.,0.),('d4',-15.,0.)]))
    assert c._stationary_lane_group(POSE,0.,2)=={'d2':0.}
    assert c._release_stationary_parallel_abort(POSE,0.)
    assert not c._parallel_abort_active
    assert c._overtake.verification.vehicle_id is None
    assert not c._overtake.committed
    assert not c._release_stationary_parallel_abort(POSE,0.)


def prefix_controller(distance=25.):
    c=group_controller()
    for t in (.2,.3):c._v2x_tracker.update(_msg(t,[('d2',0.,0.),('d3',4.,0.),('d4',distance,2.)]))
    c._vehicle_passage=lambda vid,*a:({0:True,2:vid!='d4'},None)
    return c


def test_distant_fourth_car_leaves_safe_prefix_but_gets_no_creep_permission():
    c=prefix_controller()
    c._overtake.target_id=c._overtake.hybrid.vehicle_id='d2'
    assert c._stationary_lane_group(POSE,0.,2)=={'d2':0.,'d3':4.}
    assert c._hybrid_escape_speed(POSE,0.,'d4')==0.
    assert 'd4' in c._v2x_tracker.active_vehicle_ids()


def test_prefix_shrinks_before_next_blockage_and_depends_on_stopping_distance():
    c=prefix_controller(10.)
    assert c._stationary_lane_group(POSE,0.,2)=={'d2':0.}
    assert c._stationary_lane_group(POSE,8.,2)=={}
    assert prefix_controller(3.)._stationary_lane_group(POSE,0.,2)=={}


def reverse_controller(rear=-4.):
    c=group_controller()
    for t in (.2,.3):c._v2x_tracker.update(_msg(t,[('d2',3.,2.),('rear',rear,2.)]))
    c._adaptive_reverse_enabled=True
    c._latched_follow_target_state=lambda *a:dict(expired=False,vehicle_id='d2',longitudinal=3.,velocity_valid=True,speed=0.)
    c._latched_target_passage=lambda *a:({2:True},None)
    c._latched_target_is_primary_v2x_blocker=lambda *a:True
    c._localization_consistent=True
    c._static_current_footprint_is_free=lambda *a:True
    c._follow_escape_target_prediction_blocked=True
    c._adaptive_reverse_max_distance=6.
    c._adaptive_reverse_wall_margin=.5
    c._adaptive_reverse_min_clear_distance=.3
    c._static_reverse_clearance=lambda *a:10.
    c._localization_position_error=0.
    c._stuck_forward_reverse_speed=1.
    c._adaptive_reverse_forward_success_cycles_required=3
    c._cfg=NS(bicycle_model=NS(width=1.5))
    c._v2x_vehicle_radius=.75
    c.get_logger=Mock(return_value=Mock())
    for name in ('_reverse_vehicle_clear_distance','_reverse_rear_is_clear','_reverse_rear_conflict_summary'):
        setattr(c,name,MethodType(method(name),c))
    return c


def plan(c):
    return method('_prepare_follow_deadlock_reverse')(c,pose=POSE,now_sec=10.,ego_speed=0.,target_id='d2')


def test_rear_vehicle_shortens_reverse_with_body_and_braking_margin():
    c=reverse_controller()
    assert plan(c)=='vehicle_bounded'
    assert c._stuck_reverse_target_distance==pytest.approx(1.0)
    assert c._adaptive_reverse_active
    assert c._reverse_rear_is_clear(POSE,0.,reverse_distance=c._stuck_reverse_target_distance)


@pytest.mark.parametrize('problem',['too_close','unknown_velocity','approaching','missing_position'])
def test_short_reverse_rejects_unsafe_or_unknown_rear(problem):
    c=reverse_controller(-2.8 if problem=='too_close' else -4.)
    if problem=='unknown_velocity':c._v2x_tracker._velocity_valid['rear']=False
    if problem=='approaching':c._v2x_tracker._velocities['rear']=(5.,0.)
    if problem=='missing_position':c._v2x_tracker._samples['rear'].clear()
    assert plan(c)=='blocked'
    assert not c._adaptive_reverse_active


def test_reverse_live_monitor_revokes_permission_when_rear_closes():
    c=reverse_controller()
    assert plan(c)=='vehicle_bounded'
    c._v2x_tracker._velocities['rear']=(5.,0.)
    assert not c._reverse_rear_is_clear(POSE,0.,reverse_distance=c._stuck_reverse_target_distance)


def test_static_wall_still_limits_reverse():
    c=reverse_controller(-20.)
    c._static_reverse_clearance=lambda *a:1.5
    assert plan(c)=='wall_bounded'
    assert c._stuck_reverse_target_distance==1.


def test_waiting_to_reverse_checks_planned_short_distance_not_default_corridor():
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[1] / 'multi_purpose_mpc_ros' / 'mpc_controller.py'
    tree=ast.parse(source.read_text())
    recovery=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_apply_stuck_recovery')
    check=next(n for n in ast.walk(recovery) if isinstance(n,ast.If) and
        '_prepass_retry_after_reverse' in ast.unparse(n.test) and
        '_stuck_recovery_until is None' in ast.unparse(n.test))
    c=reverse_controller()
    assert plan(c)=='vehicle_bounded'
    c._prepass_retry_after_reverse=True
    c._stuck_recovery_until=None
    c._switch_prepass_to_follow=Mock()
    fn=ast.FunctionDef(name='waiting_check',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ('self','pose','actual_speed')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[check],decorator_list=[])
    ns={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])), '<actual-reverse-wait>', 'exec'),ns)
    ns['waiting_check'](c,POSE,0.)
    c._switch_prepass_to_follow.assert_not_called()
    c._v2x_tracker._velocities['rear']=(5.,0.)
    ns['waiting_check'](c,POSE,0.)
    c._switch_prepass_to_follow.assert_called_once()


def test_rejected_l2_does_not_hide_safe_l0():
    c=lane_controller()
    c._latched_target_passage=lambda *a:({0:True,2:True},None)
    c._relative_lane_vehicle_samples=lambda *a:[('d2',1,4.)]
    c._apply_l2_restricted_zone_policy=lambda lane,**kw:1 if lane==2 else lane
    assert method('_select_follow_escape_lane')(c,POSE,0.)[0]==0
    c._waypoint_in_configured_zones=lambda *a:True
    assert method('_select_follow_escape_lane')(c,POSE,0.)[0] is None
