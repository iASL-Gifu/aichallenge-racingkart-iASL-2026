from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock
import pytest

from .test_hybrid_integration import controller as hybrid_controller, apply, reference
from .test_hybrid_escape_speed import escape_controller
from .test_overtake_lane_hold import POSE
from .test_overtake_session import controller_method
from .test_v2x_vehicle_tracker import _msg


def group_controller():
    c=escape_controller()
    c._waypoint_in_configured_zones=lambda *a:False
    c._l0_entry_prohibited_zones=[]
    c._overtake_latch_max_distance=35.
    c._stopped_lead_speed_threshold=.3
    c._parallel_critical_clearance=.3
    c._apply_l2_restricted_zone_policy=lambda lane,**kw:lane
    c._vehicle_passage=lambda *a:({0:True,2:True},None)
    c._parallel_abort_active=True
    c._parallel_abort_vehicle_id='d2'
    c._parallel_abort_target_lane_idx=1
    c._parallel_abort_previous_lane=2
    c._mpc_safety_recovery_active=False
    c._post_reverse_full_width_recovery_active=False
    c._cancel_l1_rejoin_for_overtake=Mock()
    for name in ('_stationary_lane_group','_release_stationary_parallel_abort'):
        setattr(c,name,MethodType(controller_method(name),c))
    for t in (0.,.1):c._v2x_tracker.update(_msg(t,[('d2',0.,0.),('d3',4.,0.)]))
    c._v2x_tracker.set_measured_body_pose('d2',0.,0.,0.,.1)
    c._v2x_tracker.set_measured_body_pose('d3',4.,0.,0.,.1)
    return c


def test_reference_completion_keeps_actual_lane_arrival_and_anchor_pending():
    c=hybrid_controller();apply(c);reference(c)
    anchor=c._overtake.hybrid.start_wp
    deadline=c._constraint_transition_until
    c._carN_center.wp_id=8
    c._carN_center.spatial_state.e_y=1.18  # 0.82 m error from target L2
    reference(c,now=11.)
    assert c._overtake.hybrid.reference_completed
    assert not c._overtake.hybrid.completed
    apply(c,now=20.);reference(c,now=20.)
    assert c._overtake.hybrid.start_wp==anchor
    assert c._constraint_transition_until==deadline
    assert not c._overtake.hybrid.completed
    c._carN_center.spatial_state.e_y=1.8
    reference(c,now=20.1)
    assert c._overtake.hybrid.completed


def test_two_stopped_cars_share_destination_but_cannot_occupy_it():
    c=group_controller()
    assert c._stationary_lane_group(POSE,0.,2)=={'d2':0.,'d3':4.}
    c._v2x_tracker.update(_msg(.2,[('d2',0.,0.),('d3',4.,0.),('blocker',7.,2.)]))
    assert c._stationary_lane_group(POSE,0.,2)=={}


def test_stationary_abort_releases_only_to_reacquisition_and_clears_yield_timer():
    c=group_controller()
    assert c._release_stationary_parallel_abort(POSE,0.)
    assert not c._parallel_abort_active
    assert c._parallel_abort_vehicle_id is None
    assert c._parallel_start_time is None
    assert c._overtake.verification.vehicle_id is None
    c._cancel_l1_rejoin_for_overtake.assert_called_once()


@pytest.mark.parametrize('problem',['moving','unknown','overlap','width','policy','recovery'])
def test_abort_cannot_release_without_admissible_stopped_group(problem):
    c=group_controller()
    if problem=='moving':c._v2x_tracker._velocities['d2']=(2.,0.)
    elif problem=='unknown':c._v2x_tracker._velocity_valid['d2']=False
    elif problem=='overlap':c._committed_target_body_overlap=lambda *a:True
    elif problem=='width':c._lane_horizon_has_vehicle_width=lambda *a:False
    elif problem=='policy':c._apply_l2_restricted_zone_policy=lambda *a,**k:1
    elif problem=='recovery':c._mpc_safety_recovery_active=True
    assert not c._release_stationary_parallel_abort(POSE,0.)
    assert c._parallel_abort_active


def test_secondary_stopped_member_gets_only_its_own_geometry_checked_creep():
    c=group_controller()
    c._overtake.target_id=c._overtake.hybrid.vehicle_id='d2'
    assert c._hybrid_escape_speed(POSE,0.,'d3')==.6
    c._current_center_envelopes_are_separated=lambda *a:(False,dict(rectangles_overlap=False,lateral_gap=-.01))
    assert c._hybrid_escape_speed(POSE,0.,'d3')==0.


def test_group_handoff_preserves_anchor_and_deadline_but_discards_target_proof():
    c=group_controller()
    c._overtake.target_id='d2'
    c._overtake.hybrid.vehicle_id='d2'
    c._overtake.hybrid.start_wp=100
    c._overtake.hybrid.travelled=6.
    c._stationary_lane_group=lambda *a:{'d2':-2.5,'d3':4.}
    c._center_path_collision_prediction=lambda *a:{'collision':False}
    c._prediction_is_clear_of_vehicle=lambda *a:True
    c._mpc=NS(last_solution_accurate=True,used_prediction_fallback=False,recovery_requested=False,infeasibility_counter=0)
    c._lane_decision=None
    c._constraint_transition_until=10.
    c._overtake_commit_probe_is_fresh=lambda *a:True
    assert controller_method('_preserve_stationary_group_hybrid')(c,'d3',2,POSE,0.,1.)
    assert c._overtake.target_id=='d2'  # The strict handoff owner installs new identity.
    assert c._overtake.hybrid.vehicle_id=='d3'
    assert c._overtake.hybrid.start_wp==100
    assert c._overtake.hybrid.travelled==6.
    assert c._constraint_transition_until==10.
    assert c._overtake.verification.vehicle_id is None


@pytest.mark.parametrize('problem', ['unverified', 'moving', 'different_lane', 'blocked'])
def test_group_anchor_is_not_inherited_without_same_lane_stationary_shadow(problem):
    c=group_controller()
    c._overtake.target_id=c._overtake.hybrid.vehicle_id='d2'
    c._overtake.hybrid.start_wp=100
    c._lane_decision=None
    c._overtake_commit_probe_is_fresh=lambda *a:problem!='unverified'
    if problem=='moving':c._v2x_tracker._velocities['d2']=(2.,0.)
    if problem=='different_lane':c._overtake.hybrid.lane_idx=0
    if problem=='blocked':c._lane_horizon_has_vehicle_width=lambda *a:False
    assert not controller_method('_preserve_stationary_group_hybrid')(c,'d3',2,POSE,0.,1.)
    assert c._overtake.hybrid.vehicle_id=='d2'
    assert c._overtake.hybrid.start_wp==100


def test_group_evaluation_and_secondary_creep_preserve_live_traffic_history():
    c=group_controller()
    c._overtake.target_id=c._overtake.hybrid.vehicle_id='d2'
    c._overtake.traffic_key=('d2',2)
    c._overtake.traffic_relevant_ids={'remembered_rear'}
    c._committed_lane_unknown_reasons={'remembered_rear':'relevant_position_missing'}
    assert c._stationary_lane_group(POSE,0.,2)
    assert c._hybrid_escape_speed(POSE,0.,'d3')==.6
    assert c._overtake.traffic_key==('d2',2)
    assert c._overtake.traffic_relevant_ids=={'remembered_rear'}
    assert c._committed_lane_unknown_reasons=={'remembered_rear':'relevant_position_missing'}


def test_secondary_creep_stops_when_hybrid_is_paused():
    c=group_controller()
    c._overtake.target_id=c._overtake.hybrid.vehicle_id='d2'
    c._overtake.hybrid.paused=True
    assert c._hybrid_escape_speed(POSE,0.,'d3')==0.


def test_actual_strict_handoff_installs_new_identity_without_resetting_hybrid():
    import ast
    from .test_hybrid_integration import control_tree
    from multi_purpose_mpc_ros.overtake_session import LaneDecision
    c=group_controller()
    c._overtake.target_id=c._overtake.hybrid.vehicle_id='d2'
    c._overtake.hybrid.start_wp=100
    c._overtake.hybrid.travelled=6.
    c._constraint_transition_until=10.
    c._prepass_dynamic_conflict_speed_limit=0.
    c._follow_latched_cache={'vehicle_id':'d2'}
    c._lane_decision=LaneDecision('d2',2,2,'hybrid_lane_transition')
    c._overtake_commit_probe_is_fresh=lambda *a:True
    c._preserve_stationary_group_hybrid=MethodType(controller_method('_preserve_stationary_group_hybrid'),c)
    c._same_lane_target_handoff_available=lambda *a:False
    c._reset_outer_lane_progress=Mock()
    c._clear_consecutive_overtake_handoff=Mock()
    node=next(n for n in ast.walk(control_tree()) if isinstance(n,ast.If) and ast.unparse(n.test)=='handoff_confirmed')
    ns=dict(self=c,opponent_vehicle_id='d3',handoff_lane_idx=2,pose=POSE,v=0.,now_sec=1.,
            active_overtake_target_id='d2',opponent_arc_distance=4.)
    exec(compile(ast.Module(body=node.body,type_ignores=[]),'<actual-strict-handoff>','exec'),ns)
    assert ns['active_overtake_target_id']=='d3'
    assert not ns['active_target_physically_complete']
    assert c._overtake.target_id==c._overtake.hybrid.vehicle_id=='d3'
    assert c._overtake.verification.vehicle_id=='d3'
    assert c._overtake.hybrid.start_wp==100
    assert c._overtake.hybrid.travelled==6.
    assert c._constraint_transition_until==10.
    assert c._lane_decision.target_id=='d3'
    assert c._prepass_dynamic_conflict_speed_limit is None
    assert c._follow_latched_cache is None
