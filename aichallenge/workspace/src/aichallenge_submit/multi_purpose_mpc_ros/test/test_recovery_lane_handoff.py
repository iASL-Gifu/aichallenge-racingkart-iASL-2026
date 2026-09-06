"""An unapplied trial is not a failed lane; Prepass yields only solve ownership."""
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock
import ast, copy
import pytest
from .test_slow_pass_spacing_release import controller as slow_controller, POSE
from .test_overtake_session import controller_method
from .test_hybrid_integration import control_tree


def controller():
    c=slow_controller()
    c._follow_escape_active=True;c._follow_escape_probe_lane_idx=2
    c._follow_escape_target_id=c._overtake.target_id='d2'
    c._prepass_fallback_recovery_active=True
    c._post_reverse_full_width_recovery_active=False
    c._follow_escape_lane_traffic_is_clear=lambda *a:True
    c._committed_target_body_overlap=lambda *a:False
    c._clear_prepass_soft_guidance=Mock();c.get_logger=lambda:Mock()
    c.handoff=MethodType(controller_method('_handoff_prepass_to_follow_probe'),c)
    return c


def test_stopped_prepass_yields_to_selected_l2_trial_without_forward_permission():
    c=controller();c._follow_escape_forward_active=False
    assert c.handoff(POSE,0.,1.)
    assert not c._prepass_fallback_recovery_active
    assert c._overtake.requested_lane==2
    assert c._follow_escape_probe_started_at==1.
    assert c._follow_escape_probe_success_cycles==0
    assert not c._follow_escape_forward_active
    c._clear_prepass_soft_guidance.assert_called_once()


@pytest.mark.parametrize('reason',['moving','unknown_speed','overlap','width','traffic','solver','reverse','target','passage'])
def test_handoff_cannot_override_real_safety_or_change_target(reason):
    c=controller();speed=0.
    if reason=='moving':speed=1.
    elif reason=='unknown_speed':c._v2x_tracker._velocity_valid['d3']=False
    elif reason=='overlap':c._committed_target_body_overlap=lambda *a:True
    elif reason=='width':c._lane_horizon_has_vehicle_width=lambda *a:False
    elif reason=='traffic':c._follow_escape_lane_traffic_is_clear=lambda *a:False
    elif reason=='solver':c._mpc_safety_recovery_active=True
    elif reason=='reverse':c._stuck_recovery_until=5.
    elif reason=='target':c._follow_escape_target_id='d3'
    elif reason=='passage':c._vehicle_passage=lambda *a:({2:False},None)
    assert not c.handoff(POSE,speed,1.)
    assert c._prepass_fallback_recovery_active


def test_actual_deferred_trial_does_not_consume_failure_budget():
    # Pull the actual branch from the complete controller source.
    from pathlib import Path
    tree=ast.parse((Path(__file__).parents[1]/'multi_purpose_mpc_ros/mpc_controller.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_update_follow_deadlock_escape')
    branch=next(n for n in ast.walk(method) if isinstance(n,ast.If) and ast.unparse(n.test)=='not lane_applied')
    fn=ast.FunctionDef(name='run',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[copy.deepcopy(branch)],decorator_list=[])
    c=controller();c._follow_escape_probe_success_cycles=2
    c._follow_escape_attempted_lanes=set();c._handoff_prepass_to_follow_probe=Mock()
    ns=dict(self=c,lane_applied=False,applied_lane_idx=None,pose=POSE,ego_speed=0.,now_sec=20.)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<deferred>','exec'),ns)
    ns['run']()
    assert c._follow_escape_probe_started_at==20.
    assert c._follow_escape_attempted_lanes==set()
    assert c._follow_escape_probe_lane_idx==2


def test_full_collision_prediction_recovers_omitted_near_points_only_for_same_solve():
    c=slow_controller()
    full=c._mpc.current_prediction
    visual=(full[0][3:],full[1][3:])
    c._mpc.current_prediction=visual
    c._live_prediction_context=(c._mpc,visual,2,c._reference_path,c._v2x_tracker)
    assert c.release(POSE)==set()
    c._mpc.collision_prediction_context=(visual,full)
    assert c.release(POSE)=={'d2','d3'}
    c._mpc.collision_prediction_context=(([],[]),full)
    assert c.release(POSE)==set()


def test_probe_reference_can_be_nonzero_while_output_remains_zero():
    tree=control_tree()
    seed=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
              and 'self._follow_escape_forward_active' in ast.unparse(n.test)
              and any(isinstance(b,ast.Expr) and 'update_v_max' in ast.unparse(b) for b in n.body))
    c=controller();c._prepass_fallback_recovery_active=False
    c._follow_escape_forward_active=False;c._collision_evidence_hold=False
    c._follow_escape_creep_speed=.6;c._reference_path.waypoints=[0,1,2]
    c._reference_path.set_v_ref=Mock();c._mpc.update_v_max=Mock()
    ns=dict(self=c,recovery_active=False,u=[0.,0.])
    exec(compile(ast.Module(body=[copy.deepcopy(seed)],type_ignores=[]),'<seed>','exec'),ns)
    c._mpc.update_v_max.assert_called_once_with(.6)
    c._reference_path.set_v_ref.assert_called_once_with([.6,.6,.6])
    assert ns['u'][0]==0.


def test_unknown_geometry_hold_overrides_forward_creep_command():
    node=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.If)
              and ast.unparse(n.test)=='self._collision_evidence_hold')
    ns=dict(self=NS(_collision_evidence_hold=True),u=[.6,0.])
    exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),'<hold>','exec'),ns)
    assert ns['u'][0]==0.
