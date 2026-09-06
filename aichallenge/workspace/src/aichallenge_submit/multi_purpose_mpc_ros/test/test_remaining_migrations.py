"""Fresh live proof, metric L1 preemption and initialization-only bounds."""
import ast
import copy
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.reference_path import ReferencePath, OUTER_COURSE_MARGIN
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    l1_rejoin_preemption_target_relevant, is_follow_retry_within_distance)
from .test_overtake_session import controller_method
from .test_hybrid_integration import control_tree


def fresh_case():
    pred=([0.,1.],[0.,-.5])
    mpc=NS(current_prediction=pred,infeasibility_counter=0,used_prediction_fallback=False,
           recovery_requested=False,time_budget_exceeded=False,last_solution_accurate=True,
           last_solution_status='solved')
    c=NS(_mpc=mpc,_parallel_warning_clearance=.3,
        _reference_path=NS(is_overtaking=True,target_lane_idx=0),
        _steering_fallback_armed=False,_mpc_safety_recovery_active=False,
        _v2x_tracker=NS(has_velocity_estimate=Mock(return_value=True)),
        _prediction_is_clear_of_vehicle=Mock(return_value=True),
        _center_path_collision_prediction=Mock(return_value={'collision':False}))
    c._live_prediction_context=(mpc,pred,0,c._reference_path,c._v2x_tracker)
    return c,dict(rectangles_overlap=False,lateral_gap=.3)


def test_fresh_live_proof_checks_actual_blocker_independent_of_latched_target():
    c,e=fresh_case();c._overtake=NS(target_id='d3')
    assert controller_method('_fresh_outer_prediction_releases_center_stop')(c,'d2',e)
    c._prediction_is_clear_of_vehicle.assert_called_once_with('d2')
    c._center_path_collision_prediction.assert_called_once_with('d2')


@pytest.mark.parametrize('problem',['overlap','gap','nan','stationary','fallback','budget','inaccurate',
    'old_context','lane','path','tracker','velocity','center_collision','unknown_center',
    'straight_collision','steering_owner','safety_recovery','failed'])
def test_fresh_release_rejects_unsafe_stale_or_non_mpc_control(problem):
    c,e=fresh_case()
    if problem=='overlap':e['rectangles_overlap']=True
    elif problem=='gap':e['lateral_gap']=.299
    elif problem=='nan':e['lateral_gap']=math.nan
    elif problem=='stationary':c._mpc.current_prediction[0][-1]=0.;c._mpc.current_prediction[1][-1]=0.
    elif problem=='fallback':c._mpc.used_prediction_fallback=True
    elif problem=='budget':c._mpc.time_budget_exceeded=True
    elif problem=='inaccurate':c._mpc.last_solution_status='solved inaccurate'
    elif problem=='failed':c._mpc.infeasibility_counter=1
    elif problem=='old_context':c._live_prediction_context=None
    elif problem=='lane':c._reference_path.target_lane_idx=2
    elif problem=='path':c._reference_path=NS(is_overtaking=True,target_lane_idx=0)
    elif problem=='tracker':c._v2x_tracker=NS(has_velocity_estimate=lambda *a:True)
    elif problem=='velocity':c._v2x_tracker.has_velocity_estimate.return_value=False
    elif problem=='center_collision':c._center_path_collision_prediction.return_value={'collision':True}
    elif problem=='unknown_center':c._center_path_collision_prediction.return_value=None
    elif problem=='straight_collision':c._prediction_is_clear_of_vehicle.return_value=False
    elif problem=='steering_owner':c._steering_fallback_armed=True
    elif problem=='safety_recovery':c._mpc_safety_recovery_active=True
    assert not controller_method('_fresh_outer_prediction_releases_center_stop')(c,'d2',e)


def preempt_guard():
    return next(n.test for n in ast.walk(control_tree()) if isinstance(n,ast.If)
        and any(isinstance(k,ast.Call) and isinstance(k.func,ast.Name)
                and k.func.id=='l1_rejoin_preemption_target_relevant' for k in ast.walk(n.test)))


@pytest.mark.parametrize('valid,stationary,slow,arc,expected',[(True,True,False,20.,True),
    (True,False,True,20.,True),(False,True,False,20.,False),
    (True,True,False,-1.,False),(True,True,False,35.,False),
    (True,True,False,math.nan,False),(True,False,False,20.,False)])
def test_actual_preempt_guard_handles_metric_detection_and_invalid_speed(valid,stationary,slow,arc,expected):
    c=NS(_follow_only=False,_parallel_abort_active=False,_overtake_latch_max_distance=35.,
         _mpc=NS(infeasibility_counter=0,current_prediction=object()),_mpc_safety_recovery_active=False)
    ns=dict(self=c,exclusive_l1_rejoin=True,outer_lane_mpc_problem_zone=False,
        outer_lane_problem_slow_override=False,startup_overtake_suppressed=False,
        opponent_ahead_detected=False,opponent_velocity_valid=valid,
        lead_is_stationary=stationary,lead_is_special_slow=slow,opponent_vehicle_id='d2',
        opponent_arc_distance=arc,l1_rejoin_preemption_target_relevant=l1_rejoin_preemption_target_relevant)
    code=compile(ast.Expression(preempt_guard()),'<actual-preemption>','eval')
    assert eval(code,ns)==expected
    c._mpc_safety_recovery_active=True
    assert not eval(code,ns)


def test_metric_target_reaches_actual_latch_writer_after_shadow_candidate_gate():
    call=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='select_latched_overtake_lane')
    ns=dict(self=NS(_parallel_abort_active=False),opponent_ahead_detected=False,
        opponent_velocity_valid=True,lead_is_stationary=True,lead_is_special_slow=False,
        opponent_arc_distance=20.,new_latch_distance=35.,
        is_follow_retry_within_distance=is_follow_retry_within_distance)
    assert eval(compile(ast.Expression(call.args[0]),'<actual-latch>','eval'),ns)
    ns['opponent_velocity_valid']=False
    assert not eval(compile(ast.Expression(call.args[0]),'<actual-latch>','eval'),ns)


def bare_path(ub,lb):
    p=ReferencePath.__new__(ReferencePath)
    p._outer_course_margin_applied=False
    p.unsafe_static_fallback_wp_ids=[]
    p.waypoints=[NS(x=10.,y=20.,psi=0.,ub=ub,lb=lb,static_border_cells=None,
                    dynamic_border_cells=None,ub_sm=None,lb_sm=None)]
    return p


@pytest.mark.parametrize('ub,lb',[(3.,-3.),(.8,-.8),(.3,-.3)])
def test_margin_once_preserves_narrow_geometry_and_does_not_expand(ub,lb):
    p=bare_path(ub,lb)
    assert p._apply_outer_course_margin_once()
    wp=p.waypoints[0]
    assert (wp.ub,wp.lb)==pytest.approx((ub-OUTER_COURSE_MARGIN,lb+OUTER_COURSE_MARGIN))
    assert wp.static_border_cells[0][1]==pytest.approx(20.+wp.ub)
    assert wp.static_border_cells[1][1]==pytest.approx(20.+wp.lb)
    wp.dynamic_border_cells=('live',)
    wp.ub_sm=.123
    before=copy.deepcopy(wp.__dict__)
    assert not p._apply_outer_course_margin_once()
    p.update_boundaries_from_markers([[100.,100.]],[[200.,200.]])
    assert wp.__dict__==before


@pytest.mark.parametrize('kind',['race','center'])
def test_configured_paths_have_csv_margin_before_marker_and_do_not_mutate_live_constraints(kind):
    from .probe_support import configured_mpc
    from .test_probe_mpc_integration import path_state
    mpc=configured_mpc(kind)
    p=mpc.model.reference_path
    raw=p.bounds
    n=len(p.waypoints)
    x=np.linspace(0.,1.,len(raw))
    expected_ub=np.interp(np.linspace(0.,1.,n),x,raw[:,1])-OUTER_COURSE_MARGIN
    expected_lb=np.interp(np.linspace(0.,1.,n),x,raw[:,2])+OUTER_COURSE_MARGIN
    assert np.array([w.ub for w in p.waypoints])==pytest.approx(expected_ub)
    assert np.array([w.lb for w in p.waypoints])==pytest.approx(expected_lb)
    p.waypoints[0].ub_sm=.1
    p.waypoints[0].dynamic_border_cells=((1.,2.),(3.,4.))
    before=path_state(p)
    p.update_boundaries_from_markers([[1.,2.]],[[3.,4.]])
    assert path_state(p)==before


def test_actual_emergency_release_removes_only_proven_vehicle_hazard():
    node=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.If)
        and any(isinstance(k,ast.Call) and isinstance(k.func,ast.Attribute)
            and k.func.attr=='_fresh_outer_prediction_releases_center_stop' for k in ast.walk(n.test)))
    c=NS(_fresh_outer_prediction_releases_center_stop=lambda vid,e:vid=='clear',
        _center_path_collision_hazard_until={'clear':1.,'blocked':1.},
        _overtake=NS(target_id='other'),_reference_path=NS(target_lane_idx=0),
        get_logger=lambda:Mock())
    # `continue` skips only this vehicle; the second one still reaches braking.
    loop=ast.For(target=ast.Name(id='vid',ctx=ast.Store()),
        iter=ast.List(elts=[ast.Constant('clear'),ast.Constant('blocked')],ctx=ast.Load()),
        body=[copy.deepcopy(node),ast.parse('braking.append(vid)').body[0]],orelse=[])
    module=ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[]))
    ns=dict(self=c,center_path_hazard_active=True,current_envelope_state={},braking=[])
    exec(compile(module,'<actual-per-vehicle-release>','exec'),ns)
    assert ns['braking']==['blocked']
    assert c._center_path_collision_hazard_until=={'blocked':1.}
