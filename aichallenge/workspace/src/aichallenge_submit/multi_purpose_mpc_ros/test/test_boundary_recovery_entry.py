"""Sensor-based mode entry and recovery from invalid input; no AWSIM state gate."""
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock
import math
import pytest
from .test_overtake_session import controller_method
from .test_boundary_recovery import recovery_controller, tick

POSE = NS(x=0., y=0., theta=0.)


def controller():
    c = NS(_straight_reentry_enabled=True, _enable_control=True,
           _collision_evidence_hold=False, _intentional_follow_stop_active=False,
           _stuck_time_threshold=2., _stuck_gnss_distance_threshold=.3,
           _velocity_report=NS(longitudinal_velocity=0.), _stuck_speed_threshold=.15,
           _boundary_stop_samples=[], _last_gnss_received_sec=0.,
           _gnss_pose=NS(pose=NS(pose=NS(position=NS(x=0.,y=0.)))),
           _boundary_recovery_input=Mock(return_value=(.4,'valid')),
           get_logger=Mock(return_value=Mock()))
    c.get_clock = lambda: NS(now=lambda: NS(nanoseconds=int(c._last_gnss_received_sec*1e9)))
    c._boundary_recovery_ready = MethodType(controller_method('_boundary_recovery_ready'), c)
    return c


def observe(c,t,x=0.,y=0.):
    c._last_gnss_received_sec=t
    c._gnss_pose.pose.pose.position=NS(x=x,y=y)
    return c._boundary_recovery_ready(POSE,t)


@pytest.mark.parametrize('state', [None,'Spawned','Grounded','Ready','Start'])
def test_entry_depends_on_gnss_not_sim_state(state):
    c=controller(); c._awsim_state=state
    assert not observe(c,0.)
    assert not observe(c,1.)
    assert not observe(c,1.99)
    assert observe(c,2.)


@pytest.mark.parametrize('input', [(None,'position_unavailable_or_stale'),(None,'heading_or_pose_invalid')])
def test_inside_or_invalid_resets_outside_duration(input):
    c=controller()
    observe(c,0.); observe(c,1.)
    c._boundary_recovery_input.return_value=input
    assert not observe(c,1.5)
    assert not c._boundary_stop_samples
    c._boundary_recovery_input.return_value=(.4,'valid')
    assert not observe(c,2.)
    assert not observe(c,3.)
    assert observe(c,4.)


def test_movement_away_and_back_does_not_count_as_stationary():
    c=controller()
    observe(c,0.)
    observe(c,1.,x=.5)
    assert not observe(c,2.)
    assert not observe(c,3.)
    assert observe(c,4.)


def test_repeated_control_ticks_without_new_gnss_do_not_confirm_stopping():
    c=controller()
    observe(c,0.)
    assert not c._boundary_recovery_ready(POSE,2.)
    assert len(c._boundary_stop_samples)==1


def test_clock_rewind_discards_old_observation():
    c=controller()
    observe(c,10.); observe(c,11.)
    assert not observe(c,1.)
    assert not observe(c,2.)
    assert observe(c,3.)


def input_controller():
    c=NS(_update_localization_consistency=Mock(), _recovery_localization_available=True,
         _odom=NS(pose=NS(pose=NS(orientation=NS(x=0.,y=0.,z=0.,w=1.)))),
         _reentry_violation=Mock(return_value=.4),
         get_clock=lambda: NS(now=lambda: NS(nanoseconds=1_000_000_000)))
    return c


@pytest.mark.parametrize('case,reason', [
    ('valid','valid'),('stale','position_unavailable_or_stale'),
    ('zero_quaternion','heading_or_pose_invalid'), ('nan_yaw','heading_or_pose_invalid'),
    ('invalid_corridor','valid')])
def test_input_validity_is_separate_from_outside(case,reason):
    c=input_controller(); pose=NS(x=0.,y=0.,theta=0.)
    if case=='stale': c._recovery_localization_available=False
    if case=='zero_quaternion': c._odom.pose.pose.orientation.w=0.
    if case=='nan_yaw': pose.theta=math.nan
    if case=='invalid_corridor': c._reentry_violation.return_value=math.inf
    violation,actual_reason=controller_method('_boundary_recovery_input')(c,pose,1.)
    assert actual_reason==reason
    assert violation == (0. if reason=='valid' else None)


def test_active_invalid_input_stops_then_resumes_without_new_mode_entry():
    c=recovery_controller()
    c._straight_reentry_direction=1; c._reentry_steering=.3
    c._boundary_recovery_input=lambda *a: (None,'invalid_corridor')
    c._straight_reentry_success_cycles=2
    assert tick(c,1.)[0]==0.
    assert c._straight_reentry_success_cycles==0
    c._select_reentry_motion.assert_not_called()
    assert c._straight_reentry_active
    c._boundary_recovery_input=lambda *a: (.4,'valid')
    assert tick(c,2.)[0]==1.


@pytest.mark.parametrize('overrun', [0., .4])
def test_stationary_inside_and_outside_both_enter(overrun):
    c=controller()
    c._boundary_recovery_input.return_value=(overrun,'valid')
    assert not observe(c,0.)
    assert not observe(c,1.)
    assert observe(c,2.)


def test_return_to_normal_is_cancelled_if_mpc_invalid_again():
    c=recovery_controller()
    c._straight_reentry_returning_drive=True
    c._straight_reentry_success_cycles=3
    # Even after starting DRIVE confirmation, reassess corrected localization.
    tick(c,1.)
    assert c._straight_reentry_active
    assert not c._straight_reentry_returning_drive
    assert c._straight_reentry_success_cycles==0
    c._select_reentry_motion.assert_called_once()


def test_motion_report_clears_old_stationary_window_before_delayed_gnss_moves():
    c=controller()
    observe(c,0.); observe(c,1.)
    assert observe(c,2.)
    c._velocity_report.longitudinal_velocity=.3
    assert not observe(c,2.1)
    assert not c._boundary_stop_samples
    c._velocity_report.longitudinal_velocity=0.
    assert not observe(c,2.2)
    assert not observe(c,3.2)
    assert observe(c,4.3)


def test_completion_discards_old_stop_samples_and_prevents_immediate_reentry():
    c=recovery_controller()
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,
              time_budget_exceeded=False,max_steering_rate=1.)
    c._straight_reentry_direction=1
    c._velocity_report.longitudinal_velocity=.3
    c._boundary_stop_samples=[(0.,0.,0.),(1.,0.,0.),(2.,0.,0.)]
    for t in (3.,3.1,3.2):
        tick(c,t)
    assert not c._straight_reentry_active
    assert not c._boundary_stop_samples
    c._straight_reentry_enabled=c._enable_control=True
    c._collision_evidence_hold=c._intentional_follow_stop_active=False
    c._stuck_time_threshold=2.; c._stuck_gnss_distance_threshold=.3
    c._stuck_speed_threshold=.15
    c._gnss_pose=NS(pose=NS(pose=NS(position=NS(x=0.,y=0.))))
    c._last_gnss_received_sec=3.21
    c.get_clock=lambda:NS(now=lambda:NS(nanoseconds=3_210_000_000))
    c._velocity_report.longitudinal_velocity=0.
    assert not controller_method('_boundary_recovery_ready')(c,POSE,3.21)


@pytest.mark.parametrize('speed',[math.nan,math.inf,-math.inf])
def test_invalid_speed_neither_starts_nor_drives_recovery(speed):
    c=controller()
    c._velocity_report.longitudinal_velocity=speed
    assert not observe(c,0.)
    assert not observe(c,2.)
    c=recovery_controller()
    c._velocity_report.longitudinal_velocity=speed
    assert tick(c,2.)[0] == 0.
    c._select_reentry_motion.assert_not_called()
