"""Shared collision geometry and actual controller entry points without ROS."""
import math
from types import SimpleNamespace as NS, MethodType
import numpy as np
import pytest
from multi_purpose_mpc_ros import collision_geometry as cg
from multi_purpose_mpc_ros.v2x_vehicle_tracker import V2XVehicleTracker
from .test_v2x_vehicle_tracker import _msg
from .test_overtake_session import controller_method


def tracker():
    return V2XVehicleTracker(20.,5.)


def update(t,time,x,y=0.):
    t.update(_msg(time,[('d2',x,y)]))


def test_motion_axis_survives_stop_and_snapshot_independently_of_zero_speed_hold():
    t=tracker();update(t,0.,0.);update(t,.1,.1,-.1)
    for stamp in (.3,.6,.9):update(t,stamp,.1,-.1)
    assert t.velocity('d2')==(0.,0.)
    body=t.collision_body('d2',.9,origin='center')
    assert body.yaw==pytest.approx(-math.pi/4)
    assert body.yaw_source=='motion_axis_held'
    assert not body.direction_valid  # Axis is known, front/rear are not measured.
    snap=t.snapshot();update(t,1.,.2,-.1)
    assert snap.collision_body('d2',.9,origin='center')==body


@pytest.mark.parametrize('event',['jump','backward_clock','long_gap','missing_id','clear','overspeed'])
def test_discontinuous_observations_cannot_keep_old_heading(event):
    t=tracker();update(t,0.,0.);update(t,.1,.1)
    assert t.collision_body('d2',.1).yaw_valid
    now=.2
    if event=='jump':update(t,now,10.)
    elif event=='backward_clock':now=.05;update(t,now,.2)
    elif event=='long_gap':now=2.;update(t,now,.1)
    elif event=='missing_id':t.update(_msg(.15,[]));update(t,now,.1)
    elif event=='clear':t.clear_active();update(t,now,.1)
    elif event=='overspeed':now=.101;update(t,now,.2)
    assert not t.collision_body('d2',now).yaw_valid


def test_initial_stationary_heading_unknown_is_a_circle_not_course_tangent():
    t=tracker();update(t,0.,0.);update(t,.1,0.)
    b=t.collision_body('d2',.1,origin='center')
    assert not b.yaw_valid and b.yaw_source=='unknown'
    ego=cg.body_pose(0.,0.,0.,.1)
    assert cg.overlaps(ego,b,cg.BodyGeometry()) is True
    assert len(cg.outline(b,cg.BodyGeometry()))==33


def test_real_measured_yaw_supports_stationary_vehicle_and_reverse_offset():
    t=tracker();update(t,0.,0.);update(t,.1,0.)
    t.set_measured_body_pose('d2',.522,0.,0.,.1)
    assert t.collision_body('d2',.1).yaw_source=='measured'
    # Measured center is stale relative to fresh position; retained direction
    # still means forward=+x even while position moves in reverse.
    update(t,.4,-.1)
    b=t.collision_body('d2',.4,origin='rear_axle')
    assert b.yaw==pytest.approx(0.) and b.direction_valid
    assert b.x==pytest.approx(.422)


def test_motion_only_reverse_does_not_guess_direction_for_rear_axle_offset():
    t=tracker();update(t,0.,0.);update(t,.1,-.1)
    b=t.collision_body('d2',.1,origin='rear_axle')
    assert b.yaw_valid and not b.direction_valid
    assert b.center_offset==0. and b.uncertainty==pytest.approx(.522)


def test_origin_offset_applied_once_only_with_directional_yaw():
    b=cg.body_pose(10.,20.,math.pi/2,1.,origin='rear_axle',direction_valid=True)
    assert (b.x,b.y)==pytest.approx((10.,20.522))
    assert b.center_offset==.522
    center=cg.body_pose(10.,20.,math.pi/2,1.,origin='center',direction_valid=True)
    assert (center.x,center.y)==(10.,20.)
    unknown=cg.body_pose(10.,20.,0.,1.,origin='unconfirmed',direction_valid=True)
    assert unknown.center_offset==0. and unknown.uncertainty==.522


@pytest.mark.parametrize('frame,time',[('odom',.1),('map',2.),('map',-.1)])
def test_wrong_frame_or_stale_observation_is_not_motion_permission(frame,time):
    t=tracker();m=_msg(.1,[('d2',0.,0.)]);m.vehicles[0].header.frame_id=frame;t.update(m)
    b=t.collision_body('d2',time,origin='center')
    assert not b.position_valid
    assert cg.overlaps(cg.body_pose(0.,0.,0.,.1),b,cg.BodyGeometry()) is None


def controller(yaw=-math.pi/4):
    t=tracker();update(t,0.,1.84,1.05);update(t,.1,1.84,1.05)
    t.set_measured_body_pose('d2',1.84,1.05,yaw,.1)
    c=NS(_v2x_tracker=t,_collision_now=.1,_collision_ego_origin='center',_collision_v2x_origin='center',
        _collision_geometry=cg.BodyGeometry(2.,1.5),_collision_ego_yaw=0.,
        _center_frenet=lambda x,y:(x,y),_center_arc_total_length=40.,
        _carN_center=NS(get_closest_waypoint=lambda *a:0),
        _reference_pathN_center=NS(get_waypoint=lambda *a:NS(psi=0.)),
        _center_arc_points=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]]),
        _center_arc_cumulative=np.array([0.,10.,20.,30.,40.]),
        _parallel_critical_clearance=.3,
        _mpc=NS(current_prediction=([0.,0.],[0.,0.]),infeasibility_counter=0,
                used_prediction_fallback=False,recovery_requested=False),
        _v2x_t_samples=[0.,0.,0.,.1],_center_path_collision_prediction_horizon_sec=1.5,
        _center_path_collision_prediction_margin=0.)
    for name in ('_committed_target_body_overlap','_current_center_envelopes_are_separated',
                 '_prediction_collision_with_vehicle','_prediction_is_clear_of_vehicle','_center_path_collision_prediction'):
        setattr(c,name,MethodType(controller_method(name),c))
    # This pure function is imported at controller module scope in production.
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import signed_closed_path_arc_distance
    c._current_center_envelopes_are_separated.__func__.__globals__['signed_closed_path_arc_distance']=signed_closed_path_arc_distance
    return c


@pytest.mark.parametrize('yaw,expected',[(0.,True),(-math.pi/4,False)])
def test_current_and_both_prediction_paths_use_same_measured_stationary_body(yaw,expected):
    c=controller(yaw);pose=NS(x=0.,y=0.,theta=0.)
    assert c._committed_target_body_overlap(pose,'d2') is expected
    envelope=c._current_center_envelopes_are_separated(pose,'d2')[1]
    assert envelope['rectangles_overlap'] is expected
    assert envelope['overlap_kind']=='body'
    assert c._prediction_collision_with_vehicle('d2') is expected
    assert c._prediction_is_clear_of_vehicle('d2') is (not expected)
    assert c._center_path_collision_prediction('d2')['collision'] is expected


def test_unknown_stationary_body_is_possible_overlap_in_all_paths():
    c=controller();c._v2x_tracker._body_headings.clear();c._v2x_tracker._measured_bodies.clear()
    pose=NS(x=0.,y=0.,theta=0.)
    env=c._current_center_envelopes_are_separated(pose,'d2')[1]
    assert env['overlap_kind']=='possible' and not env['yaw_known']
    assert c._committed_target_body_overlap(pose,'d2') is True
    assert not c._prediction_is_clear_of_vehicle('d2')
    assert c._center_path_collision_prediction('d2')['collision'] is True


def test_prediction_cannot_release_current_stale_or_wrong_frame_body():
    c=controller();c._collision_now=5.
    assert c._committed_target_body_overlap(NS(x=0.,y=0.,theta=0.),'d2') is None
    assert not c._prediction_is_clear_of_vehicle('d2')
    assert c._center_path_collision_prediction('d2') is None


def test_measured_ego_alignment_is_applied_to_current_and_predicted_pose_once():
    c=controller()
    c._collision_ego_alignment=(.522,0.,.1,.1)
    b=cg.ego_body(c,NS(x=0.,y=0.,theta=0.))
    assert (b.x,b.y,b.yaw)==pytest.approx((.522,0.,.1))
    pred=cg.predicted_ego(c,[0.,1.],[0.,0.],1)
    assert (pred.x,pred.y,pred.yaw)==pytest.approx((1.522,0.,.1))
    assert pred.origin=='center' and pred.uncertainty==0.


def test_stale_measured_pose_cannot_overwrite_recent_motion_heading():
    t=tracker();update(t,1.,0.);update(t,1.1,.1)
    t.set_measured_body_pose('d2',0.,0.,math.pi/2,0.)
    assert t.collision_body('d2',1.1,origin='center').yaw==pytest.approx(0.)


def test_zero_covariance_invalid_nan_position_and_frame_change_are_explicit():
    t=tracker();update(t,0.,0.);update(t,.1,.1)
    m=_msg(.2,[('d2',.2,0.)]);m.vehicles[0].header.frame_id='odom';t.update(m)
    assert not t.collision_body('d2',.2).yaw_valid
    assert not t.collision_body('d2',.2).position_valid
    update(t,.3,float('nan'))
    assert t.collision_body('d2',.3) is None


def test_collision_marker_polygons_match_the_shared_sat_geometry():
    from unittest.mock import Mock
    class Marker:
        DELETEALL=3;ADD=0;LINE_STRIP=4;TEXT_VIEW_FACING=9
        def __init__(self):
            self.header=NS(frame_id='',stamp=None)
            self.pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(w=0.))
            self.scale=NS(x=0.,z=0.)
            self.color=NS(r=0.,g=0.,b=0.,a=0.)
            self.points=[]
    c=controller()
    c.get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:NS(sec=0,nanosec=100000000)))
    c.get_logger=lambda:NS(info=Mock())
    c._collision_body_publisher=Mock()
    fn=controller_method('_publish_collision_bodies')
    fn.__globals__.update(Marker=Marker,MarkerArray=lambda:NS(markers=[]),Point=lambda:NS(x=0.,y=0.,z=0.))
    fn(c,NS(x=0.,y=0.,theta=0.))
    markers=c._collision_body_publisher.publish.call_args.args[0].markers
    polygons=[m for m in markers if getattr(m,'type',None)==Marker.LINE_STRIP]
    assert len(polygons)==2
    target=polygons[1]
    expected=cg.outline(cg.target_body(c,'d2'),c._collision_geometry)
    assert [(p.x,p.y) for p in target.points]==expected
    assert target.color.g==1.  # Measured tilted target is not overlapping.
    assert any('measured' in getattr(m,'text','') for m in markers)


def test_origin_lateral_margin_reduces_width_without_shortening_or_erasing_covariance():
    g = cg.BodyGeometry()
    old = cg.body_pose(0., 0., 0., 0., origin='unconfirmed', uncertainty=.01)
    new = cg.body_pose(0., 0., 0., 0., origin='unconfirmed', uncertainty=.01,
                       origin_lateral_margin=.272)
    assert cg.extents(new, g, 0.)[0] == cg.extents(old, g, 0.)[0]
    assert 2*cg.extents(new, g, 0.)[1] == pytest.approx(2.014)
    assert 2*(cg.extents(old, g, 0.)[1]-cg.extents(new, g, 0.)[1]) == pytest.approx(.5)
    assert max(y for x,y in cg.outline(new,g)) == pytest.approx(1.007)
    other_old = cg.replace(old, y=2.2)
    other_new = cg.replace(new, y=2.2)
    assert cg.overlaps(old, other_old, g)
    assert not cg.overlaps(new, other_new, g)
    # Unknown heading has no established lateral axis: retain its enclosing circle.
    unknown = cg.replace(new, yaw=None)
    assert cg.outline(unknown,g) == cg.outline(cg.replace(old,yaw=None),g)
    assert cg.overlaps(unknown,new,g)


def test_tracker_snapshot_and_controller_predictions_share_lateral_margin():
    t=tracker();update(t,0.,0.);update(t,.1,.1)
    c=NS(_v2x_tracker=t.snapshot(),_collision_now=.1,
         _collision_origin_lateral_margin=.272,_collision_ego_yaw=0.)
    target=cg.target_body(c,'d2')
    raw=t.collision_body('d2',.1)
    assert target.lateral_padding == pytest.approx(raw.lateral_padding-.25)
    ego=cg.ego_body(c,NS(x=0.,y=0.,theta=0.))
    predicted=cg.predicted_ego(c,[0.,1.],[0.,0.],0)
    assert ego.lateral_padding == pytest.approx(.272)
    assert predicted.lateral_padding == ego.lateral_padding
    assert predicted.uncertainty == ego.uncertainty == .522
