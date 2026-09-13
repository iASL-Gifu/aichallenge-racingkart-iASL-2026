"""Boundary escape must evaluate curved motion without ignoring physical collisions."""
import math
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.boundary_recovery import evaluate, Motion
from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.collision_geometry import BodyPose, BodyGeometry
from .test_overtake_session import controller_method


def choose(pose=(0., 0., 0.), **kwargs):
    options = dict(target=(3.5, 0.), distance=2., wheelbase=1.1,
                   steering_limit=.3, clear=lambda path: (True, 'clear'))
    options.update(kwargs)
    return evaluate(pose, **options)


@pytest.mark.parametrize('target,sign', [((3.5,2.),1),((3.5,0.),0),((3.5,-2.),-1)])
def test_compares_all_three_forward_motions_toward_waypoint(target, sign):
    check = Mock(return_value=(True,'clear'))
    motion, _ = choose(target=target, clear=check)
    assert motion.direction == 1
    assert math.copysign(1, motion.steering) == sign if sign else motion.steering == 0.
    assert check.call_count == 3


def test_forward_blocked_selects_straight_reverse():
    motion, _ = choose(clear=lambda path: (path[-1][0] < 0., 'wall'))
    assert motion.direction == -1 and motion.steering == 0.


def test_heading_away_from_waypoint_selects_reverse():
    motion, _ = choose((0.,0.,math.pi))
    assert motion.direction == -1 and motion.steering == 0.


def test_blocked_best_forward_candidate_uses_other_safe_forward():
    motion, _ = choose(target=(3.5,2.), clear=lambda path: (path[-1][2] <= 0.,'wall'))
    assert motion.direction == 1 and motion.steering == 0.


def test_all_colliding_candidates_explain_stop():
    motion, reason = choose(clear=lambda path: (False,'wall'))
    assert motion is None
    assert reason.count('wall') == 4 and 'reverse=' in reason


def test_actual_rate_limited_steering_is_used_for_collision_check():
    check = Mock(return_value=(True,'clear'))
    motion, _ = choose(target=(3.5,-2.), previous_steering=.3, steering_step=.05, clear=check)
    assert motion.steering == pytest.approx(.25)
    assert all(path_call.args[0][-1][2] > 0 for path_call in check.call_args_list)


def static_map():
    m = Map.__new__(Map)
    m.resolution = .1
    m.origin = [-5., -5.]
    m.height = m.width = 101
    m.data_backup = np.ones((101,101), dtype=np.uint8)
    return m


def test_rotated_body_checks_corners_and_out_of_map():
    m = static_map()
    body = BodyPose(0., 0., math.pi/4, 0.)
    geom = BodyGeometry(length=2., width=1.)
    assert m.static_body_is_free(body, geom)
    x,y = m.w2m(.35, 1.)
    m.data_backup[y,x] = 0
    assert not m.static_body_is_free(body, geom)
    assert not m.static_body_is_free(BodyPose(4.9,0.,0.,0.), geom)


def recovery_controller():
    c = NS(_mpc_cfg=NS(control_rate=40.), _reentry_mpc_path_is_valid=Mock(return_value=True), _steering_fallback_armed=False, _request_awsim_control_mode_for_recovery=Mock(),
           _last_u=[0.,0.], _reentry_violation=lambda *a: .3,
           _straight_reentry_active=True, _straight_reentry_returning_drive=False,
           _straight_reentry_success_cycles=0, _straight_reentry_success_cycles_required=3,
           _straight_reentry_direction=0, _reentry_steering=0.,
           _reentry_hold_until=0., _reentry_steering_ready_at=0.,
           _mpc=NS(infeasibility_counter=1, max_steering_rate=1.),
           _velocity_report=NS(longitudinal_velocity=0.),
           _stuck_pre_reverse_duration=1.2, _straight_reentry_speed=1.,
           _stuck_last_drive_request_at=None, _stuck_last_reverse_request_at=None,
           _stuck_drive_request_resend_sec=.5, _stuck_reverse_request_resend_sec=.5,
           _publish_gear_command=Mock(), _gear_drive_command=2,
           _gear_reverse_command=20, _gear_pre_reverse_command=22,
           _current_gear_is_drive=lambda: True, _current_gear_is_reverse=lambda: False,
           _select_reentry_motion=Mock(return_value=(Motion(1,.3,(),.2),'selected')),
           get_logger=Mock(return_value=Mock()))
    c._boundary_recovery_input = lambda pose, now: (c._reentry_violation(pose.x, pose.y, pose.theta), 'valid')
    return c


def tick(c,t):
    u = [5.,.1]
    owns_control = controller_method('_apply_straight_reentry')(
        c, NS(nanoseconds=int(t*1e9)), NS(x=0.,y=-.3,theta=0.), u)
    assert owns_control or not c._straight_reentry_active
    c._last_u = u[:]
    return u


def test_steering_setup_then_forward_then_immediate_stop_for_new_obstacle():
    c = recovery_controller()
    assert tick(c,1.) == [0.,.3]
    assert tick(c,1.6) == [1.,.3]
    c._select_reentry_motion.return_value = (None, 'vehicle_collision=car')
    assert tick(c,1.7)[0] == 0.
    assert c._straight_reentry_active
    assert 'vehicle_collision=car' in c.get_logger().warn.call_args.args[0]
    c._select_reentry_motion.return_value = (Motion(1,.3,(),.2),'selected')
    assert tick(c,1.8)[0] == 1.


def test_reverse_requires_gear_ack_and_has_zero_steering():
    c = recovery_controller()
    c._select_reentry_motion.return_value = (Motion(-1,0.,(),.2),'selected')
    assert tick(c,1.) == [0.,0.]
    assert c._publish_gear_command.call_args.args[1] == 22
    assert tick(c,2.3)[0] == 0.
    assert c._publish_gear_command.call_args.args[1] == 20
    c._current_gear_is_reverse=lambda: True
    assert tick(c,2.9) == [1.,0.]


def test_actual_motion_must_stop_before_direction_change():
    c = recovery_controller()
    c._velocity_report.longitudinal_velocity = -.4
    assert tick(c,1.)[0] == 0.
    assert c._straight_reentry_direction == 0


def test_completion_requires_forward_motion_and_stable_mpc():
    c = recovery_controller()
    c._reentry_violation=lambda *a: 0.
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,
              time_budget_exceeded=False,max_steering_rate=1.)
    c._velocity_report.longitudinal_velocity = .3
    c._straight_reentry_direction = 1
    tick(c,1.); tick(c,1.1)
    assert c._straight_reentry_active
    assert tick(c,1.2) == [5.,.1]
    assert not c._straight_reentry_active


def test_real_physical_check_accepts_corridor_outside_but_rejects_static_wall():
    c = NS(_recovery_localization_available=True, _collision_ego_origin='center',
           _collision_now=0., _map=static_map(), _straight_reentry_speed=.5,
           _v2x_tracker=NS(active_vehicle_ids=lambda: []))
    check = controller_method('_reentry_path_is_clear')
    path = ((0.,-.3,0.), (.05,-.29,.02))
    assert check(c,path) == (True,'clear')
    ix,iy = c._map.w2m(.5,-.3)
    c._map.data_backup[iy,ix]=0
    assert not check(c,path)[0]
    c._recovery_localization_available=False
    assert check(c,path) == (False,'localization_unavailable')


@pytest.mark.parametrize('case,allowed', [('clear',True), ('collision',False), ('unknown',False)])
def test_actual_traffic_check_blocks_collision_and_unknown_velocity(monkeypatch, case, allowed):
    from multi_purpose_mpc_ros import collision_geometry as collision
    c = NS(_recovery_localization_available=True, _collision_ego_origin='center',
           _collision_now=0., _map=static_map(), _straight_reentry_speed=.5,
           _v2x_tracker=NS(active_vehicle_ids=lambda: ['car'],
                           has_velocity_estimate=lambda vid: case != 'unknown',
                           velocity=lambda vid: (0.,0.)))
    target = BodyPose(.5 if case == 'collision' else 4., 0., 0., 0.)
    monkeypatch.setattr(collision, 'target_body', lambda *a: target)
    result, reason = controller_method('_reentry_path_is_clear')(
        c, ((0.,0.,0.),(.1,0.,0.)))
    assert result is allowed
    if case == 'collision':
        assert reason == 'vehicle_collision=car'
    elif case == 'unknown':
        assert reason == 'unknown_vehicle=car'


@pytest.mark.parametrize('enabled,intentional,expected', [(True,False,True), (False,False,False), (True,True,False)])
def test_mode_takes_ownership_before_candidate_selection(enabled, intentional, expected):
    c = NS(_straight_reentry_enabled=True, _enable_control=enabled,
           _collision_evidence_hold=False, _intentional_follow_stop_active=intentional,
           _reentry_violation=lambda *a: .4, get_logger=Mock(return_value=Mock()),
           _boundary_recovery_ready=lambda *a: enabled and not intentional)
    assert controller_method('_start_straight_reentry')(
        c, NS(x=0.,y=-.4,theta=0.), 1.) is expected
    if expected:
        assert c._straight_reentry_active
        assert c._straight_reentry_direction == 0




def test_forward_steering_update_does_not_brake_each_cycle():
    c=recovery_controller()
    c._straight_reentry_direction=1
    c._reentry_steering=.25
    c._velocity_report.longitudinal_velocity=.3
    c._last_u=[.5,.25]
    assert tick(c,2.) == pytest.approx([1.,.275])


def test_stopped_steering_change_waits_before_motion():
    c=recovery_controller()
    c._straight_reentry_direction=1
    c._reentry_steering=-.3
    c._last_u=[0.,-.3]
    assert tick(c,2.) == [0.,.3]
    assert tick(c,3.) == [1.,.3]


def test_solved_mpc_without_motion_keeps_recovery_active():
    c=recovery_controller()
    c._reentry_mpc_path_is_valid.return_value=False
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,
              time_budget_exceeded=False,max_steering_rate=1.)
    for t in (1.,2.,3.,4.):
        tick(c,t)
    assert c._straight_reentry_active
    assert c._straight_reentry_success_cycles == 0


@pytest.mark.parametrize('active,traffic', [(False,False),(True,False),(False,True)])
def test_general_stall_and_explicit_traffic_use_distinct_owners(active,traffic):
    c=NS(_stuck_recovery_enabled=True, _recovery_motion_confirmed=True, _straight_reentry_active=active,
         _enable_control=True,_collision_evidence_hold=False,
         _intentional_follow_stop_active=False,_prepass_retry_after_reverse=traffic,
         _stuck_recovery_until=None, _start_straight_reentry=Mock(return_value=True),
         _apply_straight_reentry=Mock(return_value=True),
         _apply_vehicle_stuck_recovery=Mock(return_value=True),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=2_000_000_000)))
    assert controller_method('_apply_stuck_recovery')(c,NS(nanoseconds=1_000_000_000),[0.,0.],0.,NS(x=0.,y=0.,theta=0.))
    assert c._apply_vehicle_stuck_recovery.called == traffic
    assert c._apply_straight_reentry.called == (not traffic)


def test_controller_selects_waypoint_motion_without_corridor_query():
    points=[NS(x=float(x),y=0.) for x in range(8)]
    c=NS(_reference_path=NS(n_waypoints=8,circular=False,get_waypoint=lambda i:points[i]),
         _car=NS(get_closest_waypoint=lambda *a:0),
         _cfg=NS(bicycle_model=NS(length=1.087)),_mpc_cfg=NS(delta_max=.314,control_rate=40.),
         _mpc=NS(max_steering_rate=2.),_velocity_report=NS(longitudinal_velocity=0.),
         _last_u=[0.,0.],_straight_reentry_probe_distance=2.,
         _reentry_overlap=lambda pose: 0.,
         _reentry_path_is_clear=Mock(return_value=(True,'clear')),
         _physical_corridor_state=Mock(side_effect=AssertionError('corridor must not gate recovery')))
    motion,_=controller_method('_select_reentry_motion')(c,NS(x=0.,y=0.,theta=0.),0.)
    assert motion.direction == 1 and motion.steering == 0.
    assert c._reentry_path_is_clear.call_count == 3


@pytest.mark.parametrize('direction', [1, -1])
def test_initial_overlap_allows_escape_but_not_deeper_or_new_collision(direction):
    m=static_map()
    for ix in range(m.width):
        wx,_=m.m2w(ix,0)
        if direction*wx < -.8:
            m.data_backup[:,ix]=0
    bodies=[BodyPose(direction*d,0.,0.,0.) for d in (0.,.05,.10,.15,.20,.25,.30,.35,.40)]
    assert m.static_recovery_path_is_clear(bodies,BodyGeometry()) == (True,'wall_escape')
    assert not m.static_recovery_path_is_clear(list(reversed(bodies)),BodyGeometry())[0]
    assert not m.static_recovery_path_is_clear([bodies[0],bodies[0]],BodyGeometry())[0]
    ix,iy=m.w2m(direction*1.4,0.)
    m.data_backup[iy,ix]=0
    assert not m.static_recovery_path_is_clear(bodies,BodyGeometry())[0]


def test_wall_escape_forward_is_allowed_even_temporarily_away_from_waypoint():
    motion,_=choose(target=(-3.5,0.),clear=lambda path:(True,'wall_escape'))
    assert motion.direction == 1


def test_wall_escape_still_checks_moving_vehicles(monkeypatch):
    from multi_purpose_mpc_ros import collision_geometry as collision
    c=NS(_recovery_localization_available=True,_collision_ego_origin='center',
         _collision_now=0.,_straight_reentry_speed=.5,
         _map=NS(static_recovery_path_is_clear=lambda *a:(True,'wall_escape')),
         _v2x_tracker=NS(active_vehicle_ids=lambda:['car'],has_velocity_estimate=lambda vid:True,
                         velocity=lambda vid:(0.,0.)))
    monkeypatch.setattr(collision,'target_body',lambda *a:BodyPose(.5,0.,0.,0.))
    assert controller_method('_reentry_path_is_clear')(c,((0.,0.,0.),(.1,0.,0.))) == (False,'vehicle_collision=car')


def test_diagonal_retreat_allows_contact_cells_to_move_along_same_wall():
    m=static_map()
    for ix in range(m.width):
        wx,_=m.m2w(ix,0)
        if wx < -.8:
            m.data_backup[:,ix]=0
    bodies=[BodyPose(i*.05,i*.03,0.,0.) for i in range(12)]
    assert m.static_recovery_path_is_clear(bodies,BodyGeometry()) == (True,'wall_escape')
    assert not m.static_recovery_path_is_clear(list(reversed(bodies)),BodyGeometry())[0]


def test_smaller_total_overlap_cannot_hide_new_separated_wall_contact():
    m=static_map()
    geometry=BodyGeometry()
    # Isolate contact association from raster layout: shrinking old contact must
    # not mask a newly touched, spatially separate obstacle.
    details=[{'reason':'occupied_cell','occupied_cells':{(1,1),(1,2)},
              'overlap_depth':2.,'max_overlap_depth':1.},
             {'reason':'occupied_cell','occupied_cells':{(1,1),(8,8)},
              'overlap_depth':1.,'max_overlap_depth':.5}]
    m._static_body_collision_evidence=Mock(side_effect=details*2)
    safe, reason = m.static_recovery_path_is_clear([BodyPose(0.,0.,0.,0.),BodyPose(.05,0.,0.,0.)],geometry)
    assert not safe and reason.startswith('new_wall_contact_at_step=1,')


@pytest.mark.parametrize('armed,normal_speed',[(True,0.),(True,3.),(False,0.)])
def test_solved_mpc_does_not_end_recovery_while_normal_control_still_stops(armed,normal_speed):
    c=recovery_controller()
    c._steering_fallback_armed=armed
    c._straight_reentry_direction=1
    c._reentry_steering=.3
    c._velocity_report.longitudinal_velocity=.3
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),used_prediction_fallback=False,
              recovery_requested=False,time_budget_exceeded=False,max_steering_rate=1.)
    check=controller_method('_apply_straight_reentry')
    for t in range(1,12):
        u=[normal_speed,.1]
        assert check(c,NS(nanoseconds=t*1_000_000_000),NS(x=0.,y=0.,theta=0.),u)
        assert u[0] == 1.
        assert c._straight_reentry_active
        assert c._straight_reentry_success_cycles == 0


def test_normal_forward_handoff_preserves_command_without_one_cycle_brake():
    c=recovery_controller()
    c._straight_reentry_direction=1
    c._reentry_steering=.3
    c._velocity_report.longitudinal_velocity=.3
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),used_prediction_fallback=False,
              recovery_requested=False,time_budget_exceeded=False,max_steering_rate=1.)
    check=controller_method('_apply_straight_reentry')
    for t in (1,2,3):
        u=[2.,.12]
        owns=check(c,NS(nanoseconds=t*1_000_000_000),NS(x=0.,y=0.,theta=0.),u)
        if t < 3:
            assert owns and u[0] == 1.
        else:
            assert not owns and u == [2.,.12]
            assert not c._straight_reentry_active
            assert c._boundary_stop_samples == []


@pytest.mark.parametrize('sign', [-1, 1])
def test_wall_clearance_precedes_shortest_distance_to_wp(sign):
    # Straight would make the most positional progress, but turning frees the body.
    motion, _ = choose(overlap=lambda p: 10.-sign*p[2],
                       clear=lambda path: (True, 'wall_escape'))
    assert motion.steering*sign > 0.
    assert motion.wall_reduction > 0.


@pytest.mark.parametrize('sign', [-1, 1])
def test_equal_wall_escape_prefers_heading_toward_wp(sign):
    motion, _ = choose(target=(3.5, sign*2.), overlap=lambda p: 0.)
    assert motion.steering*sign > 0.
    assert motion.heading_improvement > 0.


def test_clearance_ranking_cannot_override_collision_rejection():
    motion, _ = choose(overlap=lambda p: 10.-p[2],
                       clear=lambda path: (path[-1][2] <= 0., 'wall'))
    assert motion.steering == 0.


def test_invalid_overlap_does_not_select_motion():
    motion, reason = choose(overlap=lambda p: math.nan)
    assert motion is None and reason == 'invalid_overlap_metric'


def test_reverse_uses_improving_prefix_before_later_wall_increase():
    def clear(path):
        if path[-1][0] > 0.:
            return False, 'wall_overlap_increases_at_step=1'
        if -path[-1][0] > 1.4:
            return False, 'wall_overlap_increases_at_step=30'
        return True, 'wall_escape'
    motion, reason = choose(clear=clear)
    assert motion.direction == -1 and motion.steering == 0.
    assert 1.3 < -motion.poses[-1][0] <= 1.4
    assert 'reverse_prefix=' in reason
    assert motion.speed_limit == 1.


def test_short_reverse_limits_speed_and_respects_current_stopping_distance():
    def clear(path):
        if -.21 < path[-1][0] < 0.:
            return True, 'wall_escape'
        return False, 'wall_overlap_increases_at_step=5'
    motion, _ = choose(clear=clear)
    assert motion.direction == -1
    distance = -motion.poses[-1][0]
    speed = motion.speed_limit
    assert 0. < speed < .3
    assert .5*speed+.5*speed**2+.05 == pytest.approx(distance)
    motion, _ = choose(clear=clear, min_reverse_distance=.4)
    assert motion is None


def test_reverse_prefix_cannot_skip_immediate_failure_or_traffic():
    assert choose(clear=lambda path:(False,'wall_overlap_increases_at_step=1'))[0] is None
    def clear(path):
        if path[-1][0] < -1.:
            return False, 'wall_overlap_increases_at_step=25'
        return False, 'vehicle_collision=car'
    assert choose(clear=clear)[0] is None


def test_reverse_prefix_speed_cap_reaches_published_command():
    c = recovery_controller()
    c._select_reentry_motion.return_value=(Motion(-1,0.,(),0.,speed_limit=.2),'reverse_prefix')
    tick(c,1.)
    c._current_gear_is_reverse=lambda:True
    assert tick(c,3.) == [.2,0.]



def test_valid_forward_mpc_exits_reverse_even_when_fixed_candidates_are_blocked():
    c = recovery_controller()
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,time_budget_exceeded=False)
    c._select_reentry_motion.return_value=(None,'fixed_candidates_blocked')
    c._reentry_mpc_path_is_valid.return_value=True
    c._current_gear_is_drive=lambda:False
    c._straight_reentry_direction=-1
    for t in (1.,2.,3.):
        assert tick(c,t)[0] == 0.
    assert c._straight_reentry_returning_drive
    c._current_gear_is_drive=lambda:True
    assert tick(c,4.) == [5.,.1]
    assert not c._straight_reentry_active


def test_forward_mpc_handoff_brakes_reverse_before_requesting_drive():
    c = recovery_controller()
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,time_budget_exceeded=False)
    c._select_reentry_motion.return_value=(None,'blocked')
    c._reentry_mpc_path_is_valid.return_value=True
    c._velocity_report.longitudinal_velocity=-.5
    for t in (1.,2.,3.,4.):
        assert tick(c,t)[0] == 0.
    c._publish_gear_command.assert_not_called()
    c._reentry_mpc_path_is_valid.return_value=False
    c._velocity_report.longitudinal_velocity=0.
    assert tick(c,5.)[0] == 0.
    assert c._straight_reentry_active



@pytest.mark.parametrize('prefix_reason', ['clear', 'wall_escape'])
def test_reverse_stops_before_new_wall_at_end_of_horizon(prefix_reason):
    def clear(path):
        if path[-1][0] >= 0.:
            return False, 'new_wall_contact_at_step=2'
        if -path[-1][0] > 1.96:
            return False, 'new_wall_contact_at_step=40'
        return True, prefix_reason
    motion, reason = choose(clear=clear)
    assert motion.direction == -1
    assert -motion.poses[-1][0] == pytest.approx(1.95)
    assert motion.speed_limit == 1.
    assert 'reverse_prefix=' in reason


def test_new_wall_prefix_still_checks_traffic_and_minimum_stopping_distance():
    def clear(path):
        if path[-1][0] < -.3:
            return False, 'new_wall_contact_at_step=7'
        return False, 'vehicle_collision=car'
    assert choose(clear=clear)[0] is None
    def close_wall(path):
        if path[-1][0] < 0. and path[-1][0] > -.06:
            return True, 'clear'
        return False, 'new_wall_contact_at_step=2'
    assert choose(clear=close_wall)[0] is None


@pytest.mark.parametrize('safe', [True, False])
def test_mpc_recovery_checks_actual_curved_prediction_and_yaw(safe):
    c=NS(_mpc=NS(last_solution_accurate=True,
                  current_prediction_times=(0., .3, .8),
                  current_recovery_prediction=((0.,0.,0.),(.5,.1,.2),(1.,.4,.5))),
         _prediction_has_forward_progress=lambda *a:True,
         _reentry_path_is_clear=Mock(return_value=(safe,'vehicle_collision=car')),
         get_logger=Mock(return_value=Mock()))
    c._mpc.infeasibility_counter=0; c._mpc.current_prediction=object()
    c._mpc.used_prediction_fallback=False; c._mpc.recovery_requested=False; c._mpc.time_budget_exceeded=False
    assert controller_method('_mpc_prediction_path_is_clear')(c,NS(x=0.,y=0.,theta=0.),(1.,.1)) is safe
    path=c._reentry_path_is_clear.call_args.args[0]
    assert path[-1] == pytest.approx((1.,.4,.5))
    assert any(p == pytest.approx((.5,.1,.2)) for p in path)
    assert all(math.hypot(b[0]-a[0],b[1]-a[1]) <= .050001 for a,b in zip(path,path[1:]))
    assert c._reentry_path_is_clear.call_args.kwargs['times'][-1] == pytest.approx(.8)


@pytest.mark.parametrize('accurate,prediction', [(False,((0.,0.,0.),(1.,0.,0.))), (True,None),
                                                (True,((0.,0.,0.),(math.nan,0.,0.)))])
def test_mpc_recovery_rejects_missing_invalid_or_inaccurate_prediction(accurate,prediction):
    c=NS(_mpc=NS(last_solution_accurate=accurate,current_recovery_prediction=prediction),
         _prediction_has_forward_progress=lambda *a:True,
         _reentry_path_is_clear=Mock())
    c._mpc.infeasibility_counter=0; c._mpc.current_prediction=object()
    c._mpc.used_prediction_fallback=False; c._mpc.recovery_requested=False; c._mpc.time_budget_exceeded=False
    c._mpc_prediction_path_is_clear=MethodType(controller_method('_mpc_prediction_path_is_clear'),c)
    assert not controller_method('_reentry_mpc_path_is_valid')(c,NS(x=0.,y=0.,theta=0.),(1.,0.))
    c._reentry_path_is_clear.assert_not_called()



def test_safe_mpc_confirmation_does_not_execute_reverse_candidate():
    c = recovery_controller()
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,time_budget_exceeded=False)
    c._reentry_mpc_path_is_valid.return_value=True
    c._select_reentry_motion.return_value=(Motion(-1,0.,(),0.),'reverse')
    for t in (1.,2.):
        assert tick(c,t)[0] == 0.
        c._publish_gear_command.assert_not_called()
    assert tick(c,3.) == [5.,.1]
    assert all(call.args[1] == 2 for call in c._publish_gear_command.call_args_list)



def test_moving_fixed_forward_candidate_cannot_bypass_rejected_mpc_path():
    c=recovery_controller()
    c._reentry_mpc_path_is_valid.return_value=False
    c._velocity_report.longitudinal_velocity=.4
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),used_prediction_fallback=False,
              recovery_requested=False,time_budget_exceeded=False,max_steering_rate=1.)
    c._straight_reentry_direction=1
    c._reentry_steering=.3
    for t in (1.,2.,3.,4.):
        tick(c,t)
    assert c._straight_reentry_active
    assert c._straight_reentry_success_cycles == 0


@pytest.mark.parametrize('depths,maxima,expected', [
    ([1.,1.2,.7],[.2,.22,.18],True),
    ([1.,1.2,.7],[.2,.24,.18],False),
    ([1.,1.2,1.1],[.2,.22,.18],False),
    ([1.,1.2,.7],[.2,.22,.21],False),
    ([1.,1.1,1.2,.7],[.2,.22,.24,.18],False),
])
def test_reverse_allows_bounded_temporary_overlap_but_requires_terminal_improvement(depths,maxima,expected):
    m=static_map()
    def details():
        return [{'reason':'occupied_cell','occupied_cells':{(1,1)},
                 'overlap_depth':d,'max_overlap_depth':v} for d,v in zip(depths,maxima)]
    bodies=[BodyPose(i*.05,0.,0.,0.) for i in range(len(depths))]
    m._static_body_collision_evidence=Mock(side_effect=details()*2)
    assert m.static_recovery_path_is_clear(bodies,BodyGeometry(),temporary_depth_increase=.03)[0] is expected
    m._static_body_collision_evidence=Mock(side_effect=details()*2)
    assert not m.static_recovery_path_is_clear(bodies,BodyGeometry())[0]


def test_reverse_allowance_cannot_hide_new_wall_patch():
    m=static_map()
    m._static_body_collision_evidence=Mock(side_effect=[
        {'reason':'occupied_cell','occupied_cells':{(1,1)},'overlap_depth':1.,'max_overlap_depth':.2},
        {'reason':'occupied_cell','occupied_cells':{(9,9)},'overlap_depth':.5,'max_overlap_depth':.1}])
    safe,reason=m.static_recovery_path_is_clear(
        [BodyPose(0.,0.,0.,0.),BodyPose(-.05,0.,0.,0.)],BodyGeometry(),temporary_depth_increase=.03)
    assert not safe and reason.startswith('new_wall_contact')


def test_reverse_checker_is_separate_from_three_forward_checks():
    forward=Mock(return_value=(False,'wall_overlap_increases_at_step=1'))
    reverse=Mock(return_value=(True,'wall_escape'))
    motion,_=choose(clear=forward,reverse_clear=reverse)
    assert motion.direction == -1
    assert forward.call_count > 3 and reverse.call_count == 1
    assert all(call.args[0][-1][0] > 0. for call in forward.call_args_list)
    assert reverse.call_args.args[0][-1][0] < 0.


@pytest.mark.parametrize('yaw', [math.radians(100), math.radians(-130), math.pi])
def test_wrong_way_forward_turn_can_improve_heading_while_away_from_wp(yaw):
    motion,_=choose((0.,0.,yaw),target_heading=0.)
    assert motion.direction == 1 and motion.steering != 0.
    assert motion.heading_improvement > 0.
    assert motion.improvement < 0.


def test_heading_recovery_keeps_turn_side_only_while_safe_and_improving():
    motion,_=choose((0.,0.,math.pi),target_heading=0.,preferred_steering=.3)
    assert motion.steering > 0.
    def clear(path):
        return (path[-1][2] <= math.pi, 'wall')
    motion,_=choose((0.,0.,math.pi),target_heading=0.,preferred_steering=.3,clear=clear)
    assert motion.steering < 0.


def test_recovery_handoff_does_not_carry_negative_acceleration_to_filter():
    c=recovery_controller()
    c._last_acc=-8.
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),used_prediction_fallback=False,
              recovery_requested=False,time_budget_exceeded=False,max_steering_rate=1.)
    for t in (1.,2.,3.):
        tick(c,t)
    assert not c._straight_reentry_active
    filtered=c._last_acc+(2.5-c._last_acc)*.2
    assert filtered == .5



@pytest.mark.parametrize('speed,drive,has_candidate,ready,expected', [
    (.4,True,True,True,1.), (0.,True,True,True,1.),
    (-.4,False,True,True,0.), (.4,True,False,True,0.),
    (.4,True,True,False,0.)])
def test_confirmation_continues_only_ready_safe_forward_motion(speed,drive,has_candidate,ready,expected):
    c=recovery_controller()
    c._straight_reentry_direction=1
    c._reentry_steering=.3
    c._last_u=[0.,.3]
    c._velocity_report.longitudinal_velocity=speed
    c._current_gear_is_drive=lambda:drive
    c._reentry_steering_ready_at=0. if ready else 5.
    if not has_candidate:
        c._select_reentry_motion.return_value=(None,'blocked')
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),used_prediction_fallback=False,
              recovery_requested=False,time_budget_exceeded=False,max_steering_rate=1.)
    assert tick(c,1.)[0] == expected
    assert c._straight_reentry_success_cycles == 1


@pytest.mark.parametrize('sign', [-1, 1])
def test_turning_forward_can_recover_after_bounded_initial_worsening(sign):
    relaxed = Mock(return_value=(True, 'wall_escape'))
    motion, _ = evaluate((0.,0.,0.), target=(3.,sign*2.), target_heading=sign*.8,
        distance=2., wheelbase=1.087, steering_limit=.314,
        clear=lambda path: (False, 'wall_overlap_increases_at_step=1'),
        forward_turn_clear=relaxed, reverse_clear=lambda path: (False,'blocked'))
    assert motion.direction == 1
    assert motion.steering*sign > 0.
    assert motion.heading_improvement > math.radians(2.)
    assert relaxed.call_count == 1  # Neither straight nor wrong-way steering.


@pytest.mark.parametrize('reason', ['vehicle_collision=car', 'new_wall_contact_at_step=1',
                                   'localization_unavailable'])
def test_turning_relaxation_cannot_bypass_other_rejections(reason):
    relaxed = Mock(return_value=(True, 'wall_escape'))
    motion, _ = evaluate((0.,0.,0.), target=(3.,2.), target_heading=.8,
        distance=2., wheelbase=1.087, steering_limit=.314,
        clear=lambda path: (False,reason), forward_turn_clear=relaxed,
        reverse_clear=lambda path: (False,'blocked'))
    assert motion is None
    relaxed.assert_not_called()


def test_turning_relaxation_keeps_no_progress_forward_latch():
    motion, _ = evaluate((0.,0.,0.), target=(3.,2.), target_heading=.8,
        distance=2., wheelbase=1.087, steering_limit=.314, allow_forward=False,
        clear=lambda path: (False,'wall_overlap_increases_at_step=1'),
        forward_turn_clear=lambda path: (True,'wall_escape'),
        reverse_clear=lambda path: (True,'clear'))
    assert motion.direction == -1


@pytest.mark.parametrize('turn,expected', [(True,True),(False,False)])
def test_controller_turning_check_uses_bounded_static_rule(turn,expected):
    m=static_map()
    m._static_body_collision_evidence=Mock(side_effect=[
        {'reason':'occupied_cell','occupied_cells':{(1,1)},'overlap_depth':d,'max_overlap_depth':v}
        for d,v in [(14.87,.3744),(14.30,.3795),(10.,.35)]]*2)
    c=NS(_recovery_localization_available=True, _collision_ego_origin='center',
         _collision_now=0., _map=m, _straight_reentry_speed=1.,
         _forward_turn_overlap_allowance=.03,
         _v2x_tracker=NS(active_vehicle_ids=lambda: []))
    result, _=controller_method('_reentry_path_is_clear')(
        c, ((0.,0.,0.),(.05,0.,.01),(.1,0.,.02)), forward_turn=turn)
    assert result is expected


def test_confirmation_from_rest_prepares_steering_then_starts_before_handoff():
    c=recovery_controller()
    c._straight_reentry_success_cycles_required=8
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,
              time_budget_exceeded=False,max_steering_rate=1.)
    assert tick(c,1.) == [0.,.3]
    assert tick(c,1.2) == [0.,.3]
    assert tick(c,1.6) == [1.,.3]
    assert c._straight_reentry_active
    assert c._straight_reentry_success_cycles == 3


def test_confirmation_from_rest_waits_for_drive_ack():
    c=recovery_controller()
    c._straight_reentry_success_cycles_required=8
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,
              time_budget_exceeded=False,max_steering_rate=1.)
    c._current_gear_is_drive=lambda:False
    assert tick(c,1.)[0] == 0.
    assert tick(c,1.6)[0] == 0.
    c._current_gear_is_drive=lambda:True
    assert tick(c,1.7)[0] == 1.
    assert c._straight_reentry_active


def test_wall_last_resort_ranks_all_directions_and_can_override_forward_hold():
    # Failed normal MPC must not prohibit a different, physically improving turn.
    motion, reason = choose(target=(3.,2.), target_heading=.8, allow_forward=False,
        clear=lambda path:(False,'new_wall_contact_at_step=1'),
        reverse_clear=lambda path:(False,'blocked'),
        overlap=lambda p: 10.-p[1],
        wall_escape_clear=lambda path:(path[-1][1] > .01,'wall_escape'))
    assert reason == 'wall_escape_best_improvement'
    assert motion.direction == 1 and motion.steering > 0.
    assert motion.wall_reduction > 0.
    assert motion.speed_limit <= .5


def test_wall_last_resort_respects_failed_motion_and_traffic_rejection():
    motion, _ = choose(target=(3.,2.), target_heading=.8,
        clear=lambda path:(False,'new_wall_contact_at_step=1'),
        reverse_clear=lambda path:(False,'blocked'),
        overlap=lambda p: 10.-p[1], excluded={(1,1)},
        wall_escape_clear=lambda path:(path[-1][1] > .01,'wall_escape'))
    assert motion is None
    traffic=Mock(return_value=(False,'vehicle_collision=car'))
    motion, _ = choose(clear=lambda path:(False,'wall_overlap_increases'),
        reverse_clear=lambda path:(False,'blocked'),
        overlap=lambda p:10.-p[0], wall_escape_clear=traffic)
    assert motion is None
    assert traffic.called


@pytest.mark.parametrize('bridge,expected', [(True,True),(False,False)])
def test_contact_slide_requires_local_occupied_wall_connection(bridge,expected):
    m=static_map()
    m.data_backup[10,10:13]=0
    if not bridge:
        m.data_backup[10,11]=1
    details=[
        {'reason':'occupied_cell','occupied_cells':{(10,10)},'overlap_depth':1.,'max_overlap_depth':.2},
        {'reason':'occupied_cell','occupied_cells':{(12,10)},'overlap_depth':.8,'max_overlap_depth':.19}]
    bodies=[BodyPose(0.,0.,0.,0.),BodyPose(.05,0.,0.,0.)]
    m._static_body_collision_evidence=Mock(side_effect=details*2)
    assert m.static_recovery_path_is_clear(bodies,BodyGeometry(),
        temporary_depth_increase=.03,recovery_contact_slide=True)[0] is expected
    m._static_body_collision_evidence=Mock(side_effect=details*2)
    assert not m.static_recovery_path_is_clear(bodies,BodyGeometry())[0]


def test_stalled_reverse_is_excluded_until_real_relocation():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    attempts=RecoveryAttempts()
    for i in range(20):
        attempts.observe(i*.1,(0.,0.,0.),1.,-1,0.,True)
    assert (-1,0) in attempts.excluded
    # Switching to a different steering gets its own progress window.
    attempts.observe(2.,(0.,0.,0.),1.,1,.314,True)
    assert (1,1) not in attempts.excluded
    attempts.observe(2.5,(0.,.6,0.),.5,1,.314,True)
    assert not attempts.excluded


def test_wall_last_resort_never_selects_terminal_worsening():
    motion,_=choose(clear=lambda path:(False,'wall_overlap_increases'),
        reverse_clear=lambda path:(False,'blocked'),
        overlap=lambda p:1.+abs(p[0])+abs(p[1]),
        wall_escape_clear=lambda path:(True,'wall_escape'))
    assert motion is None


def test_logged_162543_wall_stop_has_improving_motion_after_local_contact_fix():
    from pathlib import Path
    import yaml
    root=Path(__file__).parents[1]
    cfg=yaml.safe_load((root/'config/config.yaml').read_text())
    road=Map(str(root/cfg['map']['yaml_path']))
    yaw=-1.1433505723986297
    # Reconstruct rear axle from the logged center at the first 5 cm sample.
    pose=(89629.83092129773-.572*math.cos(yaw),
          43158.62386422778-.572*math.sin(yaw),yaw)
    traj=np.genfromtxt(root/cfg['reference_path']['race_csv_path'],delimiter=',',names=True)
    j=int(np.argmin((traj['x_m']-pose[0])**2+(traj['y_m']-pose[1])**2))
    distance=0.
    while distance<3.5:
        k=(j+1)%len(traj)
        distance+=math.hypot(traj['x_m'][k]-traj['x_m'][j],traj['y_m'][k]-traj['y_m'][j])
        j=k
    c=NS(_map=road,_collision_ego_origin='rear_axle',_collision_now=0.,
         _recovery_localization_available=True,_straight_reentry_speed=1.,
         _reverse_overlap_allowance=.03,_forward_turn_overlap_allowance=.03,
         _v2x_tracker=NS(active_vehicle_ids=lambda:[]))
    c._reentry_overlap=MethodType(controller_method('_reentry_overlap'),c)
    c._reentry_path_is_clear=MethodType(controller_method('_reentry_path_is_clear'),c)
    options=dict(target=(traj['x_m'][j],traj['y_m'][j]),target_heading=traj['psi_rad'][j],
        distance=2.,wheelbase=1.087,steering_limit=.314,
        overlap=c._reentry_overlap,clear=c._reentry_path_is_clear,
        reverse_clear=lambda p:c._reentry_path_is_clear(p,reverse=True),
        forward_turn_clear=lambda p:c._reentry_path_is_clear(p,forward_turn=True))
    # Tighter swept coverage may already admit this safe escape without the
    # last-resort allowance. Test the physical outcome, not the old rejection.
    motion,reason=evaluate(pose,**options,
        wall_escape_clear=lambda p:c._reentry_path_is_clear(p,speed=.5,wall_escape=True))
    assert motion.direction==1 and motion.steering<0.
    assert motion.wall_reduction>0.
    assert c._reentry_path_is_clear(motion.poses,speed=motion.speed_limit,wall_escape=True)[0]


def test_forward_prefix_stops_before_distant_wall_without_reversing():
    def check(path):
        length = sum(math.dist(a[:2], b[:2]) for a,b in zip(path,path[1:]))
        return (length <= .6, 'clear' if length <= .6 else 'new_wall_contact_at_step=13')
    motion, _ = choose(clear=check)
    assert motion.direction == 1
    length = sum(math.dist(a[:2], b[:2]) for a,b in zip(motion.poses,motion.poses[1:]))
    assert .1 <= length <= .6
    # The commanded speed can stop inside the checked prefix, including margin.
    v = motion.speed_limit
    assert .05+.5*v+.5*v*v <= length+1e-8


def test_forward_prefix_must_cover_current_speed_stopping_distance():
    def check(path):
        return (path[-1][0] < 0. or path[-1][0] < .6, 'new_wall_contact_at_step=13')
    motion, _ = choose(clear=check, measured_speed=1., min_reverse_distance=1.05)
    assert motion.direction == -1


@pytest.mark.parametrize('reason', ['unknown_vehicle=car',
                                    'localization_unavailable', 'invalid_body'])
def test_forward_prefix_never_relaxes_non_wall_failures(reason):
    check = Mock(return_value=(False,reason))
    assert choose(clear=check)[0] is None
    assert check.call_count == 4


def test_forward_prefix_is_rechecked_against_traffic():
    def check(path):
        if len(path) > 35:
            return False, 'new_wall_contact_at_step=35'
        return False, 'vehicle_collision=car'
    assert choose(clear=check)[0] is None


def test_vehicle_limited_right_turn_rebuilds_safe_stoppable_prefix():
    from multi_purpose_mpc_ros.core.boundary_recovery import rollout
    checked = []
    def check(path):
        checked.append(path)
        length = sum(math.dist(a[:2], b[:2]) for a,b in zip(path,path[1:]))
        safe = path[-1][2] < 0. and length <= .6
        return safe, 'clear' if safe else 'vehicle_collision=d4'
    motion, _ = choose(target=(3.5,-2.), clear=check, steering_rate=.6)
    assert motion.direction == 1 and motion.steering == -.3
    length = sum(math.dist(a[:2], b[:2]) for a,b in zip(motion.poses,motion.poses[1:]))
    assert .05+.5*motion.speed_limit+.5*motion.speed_limit**2 <= length+1e-8
    expected = rollout((0.,0.,0.),1,-.3,length,1.1,
                       initial_steering=0.,steering_rate=.6,speed=motion.speed_limit)
    assert np.asarray(motion.poses) == pytest.approx(np.asarray(expected))
    assert motion.poses in checked


def test_vehicle_limited_prefix_cannot_skip_initial_contact():
    check = Mock(return_value=(False,'vehicle_collision=d4'))
    assert choose(clear=check)[0] is None
    assert check.call_count > 4  # Short paths are checked, never blindly accepted.


def test_vehicle_limited_prefix_must_cover_measured_stopping_distance():
    def check(path):
        length = sum(math.dist(a[:2], b[:2]) for a,b in zip(path,path[1:]))
        return length <= .6, 'vehicle_collision=d4'
    assert choose(clear=check, measured_speed=1.)[0] is None


def test_vehicle_limited_prefix_still_rejects_wall():
    def check(path):
        length = sum(math.dist(a[:2], b[:2]) for a,b in zip(path,path[1:]))
        return False, ('vehicle_collision=d4' if length > .6
                       else 'new_wall_contact_at_step=1')
    assert choose(clear=check)[0] is None


def test_retention_rechecks_same_reverse_despite_safe_forward_becoming_available():
    motion, reason = choose(retained=(-1,0.))
    assert motion.direction == -1 and reason.startswith('retained:')
    motion, _ = choose(retained=(-1,0.),
        clear=lambda path: (path[-1][0] > 0., 'vehicle_collision=car'))
    assert motion.direction == 1


def test_failed_motion_cannot_be_retained():
    motion,_ = choose(retained=(-1,0.), excluded={(-1,0)})
    assert motion.direction == 1


def test_retained_forward_does_not_switch_for_small_waypoint_change():
    motion, reason = choose(target=(3.5,-.01), retained=(1,.3))
    assert motion.steering == .3 and reason.startswith('retained:')


def test_steering_ramp_starts_at_previous_angle_and_reaches_requested_turn():
    from multi_purpose_mpc_ros.core.boundary_recovery import rollout
    ramp = rollout((0.,0.,0.),1,.3,2.,1.1,initial_steering=-.3,steering_rate=.6,speed=1.)
    instant = rollout((0.,0.,0.),1,.3,2.,1.1)
    assert ramp[1][2] < 0.  # Still initially steering right.
    assert ramp[-1][2] > 0.  # Reaches left steering within the horizon.
    assert ramp[-1][2] < instant[-1][2]


def test_stopped_and_moving_selection_use_same_target_steering():
    for measured in (0., .4):
        motion,_ = choose(target=(3.5,2.), previous_steering=0.,
                          steering_rate=1., measured_speed=measured)
        assert motion.steering == .3  # Do not replace target with one cycle's .025 rad.
        assert 0. < motion.poses[1][2] < motion.poses[-1][2]/20.


def test_steering_ramp_collision_is_not_hidden_by_instant_turn():
    # A left goal does not excuse first sweeping right while steering crosses zero.
    motion,_ = choose(target=(3.5,2.), previous_steering=-.3, steering_rate=.5,
        clear=lambda path: (path[-1][0] < 0. or all(p[1] >= -1e-6 for p in path),
                            'vehicle_collision=car'))
    assert motion.direction == -1


def test_controller_retains_preparing_and_measurably_improving_reverse_only():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    points=[NS(x=float(x),y=0.) for x in range(8)]
    c=NS(_reference_path=NS(n_waypoints=8,circular=False,get_waypoint=lambda i:points[i]),
         _car=NS(get_closest_waypoint=lambda *a:0),
         _cfg=NS(bicycle_model=NS(length=1.087)),_mpc_cfg=NS(delta_max=.314,control_rate=40.),
         _mpc=NS(max_steering_rate=2.),_velocity_report=NS(longitudinal_velocity=0.),
         _last_u=[0.,0.],_straight_reentry_probe_distance=2., _straight_reentry_speed=1.,
         _straight_reentry_direction=-1,_reentry_steering=0.,_reentry_hold_until=2.,
         _reentry_overlap=lambda pose:0., _reentry_path_is_clear=Mock(return_value=(True,'clear')),
         _recovery_attempts=RecoveryAttempts())
    select=controller_method('_select_reentry_motion')
    assert select(c,NS(x=0.,y=0.,theta=0.),1.)[0].direction == -1
    # Preparation grace expired without measured progress: recompare forward.
    assert select(c,NS(x=0.,y=0.,theta=0.),3.)[0].direction == 1
    c._last_u=[1.,0.]
    c._velocity_report.longitudinal_velocity=-.5
    c._recovery_attempts.observe(3.,(0.,0.,0.),0.,-1,0.,True)
    assert select(c,NS(x=-.2,y=0.,theta=0.),3.5)[0].direction == -1
    # Do not back forever merely because a clear straight reverse makes progress.
    c._reentry_motion_origin=(0.,0.)
    assert select(c,NS(x=-2.1,y=0.,theta=0.),3.6)[0].direction == 1
    # Moving toward a fresh traffic hazard overrides the hold immediately.
    c._reentry_path_is_clear=lambda path,**kw:(path[-1][0] > path[0][0], 'vehicle_collision=car')
    assert select(c,NS(x=-.25,y=0.,theta=0.),3.6)[0].direction == 1


def test_safe_turn_with_clear_heading_gain_overrides_straight_retention():
    motion, reason=choose(target=(3.5,2.), target_heading=.7, retained=(1,0.))
    assert motion.direction == 1 and motion.steering > 0.
    assert reason == 'turn_improves_held_straight'


def test_straight_retention_ignores_small_gain_and_unsafe_turns():
    motion, reason=choose(target=(3.5,0.),target_heading=.02,retained=(1,0.))
    assert motion.steering == 0. and reason.startswith('retained:')
    motion, reason=choose(target=(3.5,2.), target_heading=.7, retained=(1,0.),
        clear=lambda p:(abs(p[-1][2])<1e-6,'vehicle_collision=car'))
    assert motion.steering == 0. and reason.startswith('retained:')


def test_wall_return_memory_survives_reverse_and_pose_veto_reset():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    a=RecoveryAttempts()
    a.remember_wall_return((1.,0.,0.),0.)
    a.excluded.clear()  # MPC pose reevaluation must not erase this distinct memory.
    assert not a.wall_path_is_clear(((-2.,0.,0.), (1.,0.,0.)))[0]
    assert not a.wall_path_is_clear(((0.,0.,.02), (.9,0.,.02)))[0]
    assert a.wall_path_is_clear(((0.,0.,.3), (.9,0.,.3)))[0]


def test_backward_then_forward_does_not_repeat_remembered_straight():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    a=RecoveryAttempts()
    a.remember_wall_return((1.5,0.,0.),0.)
    motion,_=choose(target=(3.5,2.),target_heading=.7,retained=(1,0.),
                    failure_path_clear=a.wall_path_is_clear)
    assert motion.direction == 1 and motion.steering > 0.


@pytest.mark.parametrize("phase, command, remember", [("preparing",0.,False),("executing",.3,True)])
def test_controller_remembers_only_executed_wall_failure(phase,command,remember):
    points=[NS(x=float(x),y=0.) for x in range(8)]
    c=NS(_reference_path=NS(n_waypoints=8,circular=False,get_waypoint=lambda i:points[i]),
         _car=NS(get_closest_waypoint=lambda *a:0),
         _cfg=NS(bicycle_model=NS(length=1.087)),_mpc_cfg=NS(delta_max=.314,control_rate=40.),
         _mpc=NS(max_steering_rate=2.),_velocity_report=NS(longitudinal_velocity=0.),
         _last_u=[0.,0.],_straight_reentry_probe_distance=2., _straight_reentry_speed=1.,
         _straight_reentry_direction=1,_reentry_steering=0.,_reentry_hold_until=5.,
         _reentry_overlap=lambda pose:0., get_logger=Mock(return_value=Mock()),
         _reentry_path_is_clear=lambda p,**kw:(p[-1][0]<p[0][0], 'new_wall_contact_at_step=1'))
    c._reentry_phase=phase
    c._last_u[0]=command
    select=controller_method('_select_reentry_motion')
    assert select(c,NS(x=1.,y=0.,theta=0.),1.)[0].direction == -1
    assert c._recovery_attempts.wall_path_is_clear(((0.,0.,0.), (1.,0.,0.)))[0] is (not remember)


def test_same_left_steering_allowed_on_different_path_after_backing():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts, rollout
    a=RecoveryAttempts()
    old_path=rollout((0.,0.,0.),1,.3,2.,1.1)
    a.remember_wall_return(old_path[-1],.3)
    assert not a.wall_path_is_clear(old_path)[0]
    shifted=rollout((-1.,0.,0.),1,.3,2.,1.1)
    assert a.wall_path_is_clear(shifted)[0]
    motion,_=choose((-1.,0.,0.),target=(3.5,2.),target_heading=.7,
        failure_path_clear=a.wall_path_is_clear,retained=(1,.3))
    assert motion.direction == 1 and motion.steering == .3


def test_history_is_not_erased_by_heading_change_before_return():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    a=RecoveryAttempts(); a.remember_wall_return((1.,0.,0.),.3)
    assert a.wall_path_is_clear(((0.,0.,.5), (1.,0.,.5)))[0]
    safe,reason=a.wall_path_is_clear(((0.,0.,.5), (1.,0.,0.)))
    assert not safe and reason.startswith('remembered_wall_return')


def test_escape_from_initial_failed_region_is_allowed_but_reentry_is_not():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    a=RecoveryAttempts(); a.remember_wall_return((0.,0.,0.),0.)
    assert a.wall_path_is_clear(((0.,0.,0.), (.1,0.,.1), (.5,.1,.4)))[0]
    assert not a.wall_path_is_clear(((0.,0.,0.), (.5,.1,.4), (.1,0.,0.)))[0]


def test_wall_memory_applies_to_last_resort_forward_too():
    motion,_=choose(overlap=lambda p:10.-p[0],
        clear=lambda p:(False,'new_wall_contact_at_step=1'),
        reverse_clear=lambda p:(False,'wall'),
        wall_escape_clear=lambda p:(True,'wall_escape'),
        failure_path_clear=lambda p:(False,'remembered_wall_return_at_step=3'))
    assert motion is None


def test_mpc_handoff_cannot_bypass_remembered_failed_pose():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    c=NS(_mpc=NS(last_solution_accurate=True,
                  current_prediction_times=(0., .5, 1.),
                  current_recovery_prediction=((0.,0.,0.),(.5,0.,0.),(1.,0.,0.))),
         _prediction_has_forward_progress=lambda *a:True,
         _reentry_path_is_clear=Mock(return_value=(True,'clear')),
         get_logger=Mock(return_value=Mock()), _recovery_attempts=RecoveryAttempts())
    c._mpc.infeasibility_counter=0; c._mpc.current_prediction=object()
    c._mpc.used_prediction_fallback=False; c._mpc.recovery_requested=False; c._mpc.time_budget_exceeded=False
    c._recovery_attempts.remember_wall_return((1.,0.,0.),0.)
    assert not controller_method('_mpc_prediction_path_is_clear')(c,NS(x=0.,y=0.,theta=0.),(1.,0.))
    assert 'remembered_wall_return' in c._mpc_handoff_reason
    c._reentry_path_is_clear.assert_not_called()


def test_steering_estimate_does_not_jump_to_sent_target():
    from multi_purpose_mpc_ros.core.boundary_recovery import SteeringEstimate
    e=SteeringEstimate()
    assert e.update(0.,0.,1.) == 0.
    assert e.update(.1,.314,1.) == pytest.approx(.1)
    assert e.update(.1,.314,1.) == pytest.approx(.1)
    assert e.update(.2,.314,1.) == pytest.approx(.2)
    assert e.update(.4,.314,1.) == pytest.approx(.314)
    assert e.update(.5,-.314,1.) == pytest.approx(.214)
    assert e.update(.6,math.nan,1.) is None
    assert e.update(.7,0.,1.) == 0.


def test_stopped_motion_prediction_is_stable_during_steering_preparation():
    from multi_purpose_mpc_ros.core.boundary_recovery import rollout
    paths=[]
    for estimated in (0.,.1,.2,.3):
        motion,_=choose(target=(3.5,2.), target_heading=.7,
            previous_steering=estimated, steering_rate=1., prepare_steering=True)
        assert motion.steering == .3
        paths.append(motion.poses)
    assert all(path == paths[0] for path in paths)
    assert paths[0] == rollout((0.,0.,0.),1,.3,2.,1.1)


def test_prepared_turn_must_be_safe_even_when_moving_ramp_would_be_safe():
    # The old path could pass a gate only while steering was still near zero.
    def check(path):
        return (path[-1][0] < 0. or abs(path[1][2]) < .002, 'vehicle_collision=car')
    moving,_=choose(target=(3.5,2.),target_heading=.7,steering_rate=1.,clear=check)
    assert moving.steering == .3
    prepared,_=choose(target=(3.5,2.),target_heading=.7,steering_rate=1.,
                      prepare_steering=True,clear=check)
    assert prepared.steering == 0.


def test_left_preparation_deadline_is_not_restarted_by_sent_command():
    from multi_purpose_mpc_ros.core.boundary_recovery import SteeringEstimate
    c=recovery_controller()
    points=[NS(x=float(i),y=.5*i,psi=.7) for i in range(8)]
    c._reference_path=NS(n_waypoints=8,circular=False,get_waypoint=lambda i:points[i])
    c._car=NS(get_closest_waypoint=lambda *a:0)
    c._cfg=NS(bicycle_model=NS(length=1.087))
    c._mpc_cfg.delta_max=.314
    c._straight_reentry_probe_distance=2.
    c._reentry_overlap=lambda p:0.
    c._reentry_path_is_clear=Mock(return_value=(True,'clear'))
    c._select_reentry_motion=MethodType(controller_method('_select_reentry_motion'),c)
    c._reentry_steering_estimate=SteeringEstimate()
    deadlines=[]
    outputs=[]
    for t in (0.,.1,.2,.3,.4,.5,.6):
        u=[0.,0.]
        assert controller_method('_apply_straight_reentry')(
            c,NS(nanoseconds=int(t*1e9)),NS(x=0.,y=0.,theta=0.),u)
        deadlines.append(c._reentry_steering_ready_at)
        outputs.append(u.copy())
        c._last_u=u.copy()  # Simulate the actual publisher's command history.
    assert all(t == pytest.approx(deadlines[0]) for t in deadlines)
    assert all(u[1] == pytest.approx(.314) for u in outputs)
    assert outputs[1][0] == 0.
    assert outputs[-1][0] > 0.
    # A new traffic hazard still stops the retained motion.
    c._reentry_path_is_clear=Mock(return_value=(False,'vehicle_collision=car'))
    u=[0.,0.]
    controller_method('_apply_straight_reentry')(c,NS(nanoseconds=700000000),NS(x=0.,y=0.,theta=0.),u)
    assert u[0] == 0.


def test_same_turn_waits_for_estimated_steering_after_slowing_to_rest():
    from multi_purpose_mpc_ros.core.boundary_recovery import SteeringEstimate
    c = recovery_controller()
    c._straight_reentry_direction = 1
    c._reentry_steering = .3
    c._last_u = [.2, -.2]
    c._reentry_steering_estimate = SteeringEstimate()
    c._reentry_steering_estimate.update(10., -.2, 1.)
    assert tick(c, 10.) == [0., .3]
    assert c._reentry_steering_ready_at == pytest.approx(10.75)
    for t in (10.1, 10.3, 10.5, 10.7, 10.8):
        c._reentry_steering_estimate.update(t, c._last_u[1], 1.)
        u = tick(c, t)
        assert u[0] == (0. if t < 10.75 else 1.)


@pytest.mark.parametrize('excluded,may_handoff', [({(-1, 0)}, True), ({(1, 1)}, False)])
def test_reverse_failure_does_not_veto_safe_forward_mpc(excluded, may_handoff):
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    c = recovery_controller()
    c._mpc = NS(infeasibility_counter=0, current_prediction=object(),
                used_prediction_fallback=False, recovery_requested=False,
                time_budget_exceeded=False, max_steering_rate=1.)
    c._recovery_attempts = RecoveryAttempts()
    c._recovery_attempts.excluded = excluded
    c._select_reentry_motion.return_value = (None, 'no candidate')
    for i in range(8):
        u = tick(c, 10.+i*.025)
    assert c._straight_reentry_active == (not may_handoff)
    assert (u[0] > 0.) == may_handoff
    assert bool(c._reentry_mpc_path_is_valid.call_count) == may_handoff


def test_mpc_confirmation_cannot_bypass_pending_steering_preparation():
    c = recovery_controller()
    c._straight_reentry_direction = 1
    c._reentry_steering = .3
    c._reentry_steering_ready_at = 11.
    c._mpc = NS(infeasibility_counter=0, current_prediction=object(),
                used_prediction_fallback=False, recovery_requested=False,
                time_budget_exceeded=False, max_steering_rate=1.)
    for i in range(8):
        assert tick(c, 10.+i*.025)[0] == 0.
    assert c._straight_reentry_active
    assert tick(c, 11.) == [5., .1]
    assert not c._straight_reentry_active
