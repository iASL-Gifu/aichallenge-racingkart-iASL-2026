"""Slow-pass release must prove the transition, not just the destination lane."""
from types import SimpleNamespace as NS, MethodType
import ast
import copy
import math
import pytest
from multi_purpose_mpc_ros import collision_geometry as cg
from .test_collision_body_pose import tracker, update
from .test_overtake_session import controller_method
from .test_hybrid_integration import control_tree
from .test_v2x_vehicle_tracker import _msg


def controller():
    t=tracker()
    for time in (0.,.1):
        t.update(_msg(time,[('d2',8.,0.),('d3',12.,0.)]))
    for vid,x in [('d2',8.),('d3',12.)]:
        t.set_measured_body_pose(vid,x,0.,0.,.1)
    xs=[.2,1.,2.,3.,4.,5.,6.,8.,10.,12.,14.]
    ys=[0.,.1,.4,1.,2.,3.,3.5,3.5,3.5,3.5,3.5]
    m=NS(current_prediction=(xs,ys),infeasibility_counter=0,used_prediction_fallback=False,
         recovery_requested=False,time_budget_exceeded=False,last_solution_accurate=True,last_solution_status='solved',_constraint_target_lane=2,_constraint_lane_relaxation=0.)
    r=NS(target_lane_idx=2,is_overtaking=True)
    c=NS(_mpc=m,_reference_path=r,_v2x_tracker=t.snapshot(),_overtake=NS(requested_lane=2),
         _collision_now=.1,_collision_ego_yaw=0.,_collision_ego_origin='center',
         _follow_only=False,_steering_fallback_armed=False,_mpc_safety_recovery_active=False,
         _prepass_fallback_recovery_active=False,_parallel_abort_active=False,
         _close_obstacle_reverse_requested=False,_stuck_recovery_until=None,
         _strict_shadow_commit_creep_max_target_speed=1.5,_parallel_critical_clearance=.3,
         _v2x_t_samples=[i*.2 for i in range(len(xs)+2)],
         _lane_horizon_has_vehicle_width=lambda lane:True,
         _outer_lane_constraint_is_collapsed=lambda lane:(False,None),
         _vehicle_passage=lambda *a:({2:True},None),
         _center_path_collision_prediction=lambda vid:{'collision':False})
    c._live_prediction_context=(m,m.current_prediction,2,r,c._v2x_tracker)
    c.release=MethodType(controller_method('_slow_pass_spacing_release_ids'),c)
    return c


POSE=NS(x=0.,y=0.,theta=0.)


def test_two_stopped_cars_can_be_released_before_lateral_separation():
    c=controller()
    assert c.release(POSE)=={'d2','d3'}


@pytest.mark.parametrize('case',['fallback','inaccurate','time_budget','wrong_lane','old_snapshot',
    'recovery','abort','reverse','width','short_prefix','disconnected','collision',
    'unknown_rear','center_collision','nan','relaxed','collapsed'])
def test_no_release_without_complete_current_safe_proof(case):
    c=controller();m=c._mpc
    if case=='fallback':m.used_prediction_fallback=True
    elif case=='inaccurate':m.last_solution_accurate=False
    elif case=='time_budget':m.time_budget_exceeded=True
    elif case=='wrong_lane':c._overtake.requested_lane=0
    elif case=='old_snapshot':c._v2x_tracker=c._v2x_tracker.snapshot()
    elif case=='recovery':c._mpc_safety_recovery_active=True
    elif case=='abort':c._parallel_abort_active=True
    elif case=='reverse':c._close_obstacle_reverse_requested=True
    elif case=='width':c._lane_horizon_has_vehicle_width=lambda lane:False
    elif case=='short_prefix':m.current_prediction[0][:]=[.2,1.];m.current_prediction[1][:]=[0.,0.]
    elif case=='disconnected':m.current_prediction[0][0]=5.
    elif case=='collision':m.current_prediction[1][:]=[0.]*len(m.current_prediction[0])
    elif case=='unknown_rear':
        c._v2x_tracker.update(_msg(.1,[('d2',8.,0.),('d3',12.,0.),('rear',-4.,3.5)]))
    elif case=='center_collision':c._center_path_collision_prediction=lambda vid:{'collision':True}
    elif case=='relaxed':m._constraint_lane_relaxation=.1
    elif case=='collapsed':c._outer_lane_constraint_is_collapsed=lambda lane:(True,{})
    elif case=='nan':m.current_prediction[0][3]=math.nan
    assert c.release(POSE)==set()


def test_moving_vehicle_is_checked_but_its_spacing_limit_is_not_released():
    c=controller();c._v2x_tracker._velocities['d3']=(2.,0.)
    assert c.release(POSE)=={'d2'}


def test_sweep_catches_between_points_and_initial_connecting_segment():
    g=cg.BodyGeometry();target=cg.body_pose(5.,0.,0.,0.)
    path=[cg.body_pose(x,0.,0.,0.) for x in (0.,10.)]
    assert not cg.overlaps(path[0],target,g)
    assert not cg.overlaps(path[1],target,g)
    assert not cg.swept_path_clear(path,[0.,1.],target,(0.,0.),g)


def test_sweep_preserves_unknown_heading_obstacle_and_observation_age():
    g=cg.BodyGeometry();path=[cg.body_pose(x,0.,0.,0.) for x in (0.,1.)]
    target=cg.body_pose(5.,0.,None,0.)
    assert cg.swept_path_clear(path,[0.,.1],target,(-2.,0.),g)
    assert not cg.swept_path_clear(path,[1.,1.1],target,(-2.,0.),g)


def test_emergency_bypass_is_per_vehicle_and_keeps_other_caps():
    # Run actual emergency-loop bypass before applying an unrelated zero cap.
    loop=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.For)
              and ast.unparse(n.target)=='vid' and ast.unparse(n.iter)=='active_vehicle_ids'
              and n.body and isinstance(n.body[0],ast.If)
              and ast.unparse(n.body[0].test)=='vid in slow_pass_release_ids')
    body=[copy.deepcopy(loop.body[0]),ast.parse('ref_vel_kmph = min(ref_vel_kmph, caps[vid])').body[0]]
    for ids in [('d2','d3'),('d3','d2')]:
        node=ast.For(target=ast.Name(id='vid',ctx=ast.Store()),iter=ast.Name(id='ids',ctx=ast.Load()),body=body,orelse=[])
        ns=dict(ids=ids,slow_pass_release_ids={'d2'},self=NS(_center_path_collision_hazard_until={'d2':1.}),
                ref_vel_kmph=5.,caps={'d2':0.,'d3':0.})
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<emergency>','exec'),ns)
        assert ns['ref_vel_kmph']==0.
        assert 'd2' not in ns['self']._center_path_collision_hazard_until
