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
    c = NS(_reentry_mpc_path_is_valid=Mock(return_value=False), _steering_fallback_armed=False, _request_awsim_control_mode_for_recovery=Mock(),
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
    assert tick(c,2.) == [1.,.3]


def test_stopped_steering_change_waits_before_motion():
    c=recovery_controller()
    c._straight_reentry_direction=1
    c._reentry_steering=-.3
    c._last_u=[0.,-.3]
    assert tick(c,2.) == [0.,.3]
    assert tick(c,3.) == [1.,.3]


def test_solved_mpc_without_motion_keeps_recovery_active():
    c=recovery_controller()
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,
              time_budget_exceeded=False,max_steering_rate=1.)
    for t in (1.,2.,3.,4.):
        tick(c,t)
    assert c._straight_reentry_active
    assert c._straight_reentry_success_cycles == 0


@pytest.mark.parametrize('active,traffic', [(False,False),(True,False),(False,True)])
def test_general_stall_and_explicit_traffic_use_distinct_owners(active,traffic):
    c=NS(_stuck_recovery_enabled=True, _straight_reentry_active=active,
         _enable_control=True,_collision_evidence_hold=False,
         _intentional_follow_stop_active=False,_prepass_retry_after_reverse=traffic,
         _stuck_recovery_until=None, _start_straight_reentry=Mock(return_value=True),
         _apply_straight_reentry=Mock(return_value=True),
         _apply_vehicle_stuck_recovery=Mock(return_value=True),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=2_000_000_000)))
    assert controller_method('_apply_stuck_recovery')(c,NS(nanoseconds=1_000_000_000),[0.,0.],0.,NS())
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
    m.static_body_collision_detail=Mock(side_effect=details)
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
                  current_recovery_prediction=((0.,0.,0.),(.5,.1,.2),(1.,.4,.5))),
         _prediction_has_forward_progress=lambda *a:True,
         _reentry_path_is_clear=Mock(return_value=(safe,'vehicle_collision=car')),
         get_logger=Mock(return_value=Mock()))
    assert controller_method('_reentry_mpc_path_is_valid')(c,NS(x=0.,y=0.,theta=0.),(1.,.1)) is safe
    path=c._reentry_path_is_clear.call_args.args[0]
    assert path[-1] == pytest.approx((1.,.4,.5))
    assert any(p == pytest.approx((.5,.1,.2)) for p in path)
    assert all(math.hypot(b[0]-a[0],b[1]-a[1]) <= .050001 for a,b in zip(path,path[1:]))
    assert c._reentry_path_is_clear.call_args.kwargs['speed'] == 1.


@pytest.mark.parametrize('accurate,prediction', [(False,((0.,0.,0.),(1.,0.,0.))), (True,None),
                                                (True,((0.,0.,0.),(math.nan,0.,0.)))])
def test_mpc_recovery_rejects_missing_invalid_or_inaccurate_prediction(accurate,prediction):
    c=NS(_mpc=NS(last_solution_accurate=accurate,current_recovery_prediction=prediction),
         _prediction_has_forward_progress=lambda *a:True,
         _reentry_path_is_clear=Mock())
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
