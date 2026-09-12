from multi_purpose_mpc_ros.core.boundary_recovery import ForwardProgress
from multi_purpose_mpc_ros.core.boundary_recovery import evaluate


def tick(guard, i, x=0., overlap=0., forward=True, valid=True):
    guard.update(i/10, x, 0., overlap, valid=valid, forward=forward, reverse=not forward)


def test_stationary_forward_is_latched_until_actual_relocation():
    g = ForwardProgress()
    for i in range(25):
        tick(g, i)
    assert g.blocked and g.reason == 'no_measured_motion'
    # Waiting, repeated valid predictions, and sensor noise cannot clear it.
    for i in range(25, 45):
        tick(g, i, x=.01, forward=False)
    assert g.blocked
    tick(g, 45, x=-.51, forward=False)
    assert not g.blocked


def test_motion_along_wall_without_reduction_requires_reverse():
    g = ForwardProgress()
    for i in range(25):
        tick(g, i, x=.04*i, overlap=1.)
    assert g.blocked and g.reason == 'wall_overlap_not_improving'


def test_shrinking_wall_overlap_and_actual_motion_allow_forward():
    g = ForwardProgress()
    for i in range(50):
        tick(g, i, x=.04*i, overlap=max(0., 1.-i*.03))
    assert not g.blocked


def test_predicted_escape_without_motion_does_not_count():
    g = ForwardProgress()
    for i in range(25):
        tick(g, i, overlap=max(0., 1.-i*.05))
    assert g.blocked


def test_duplicate_and_stale_samples_do_not_accumulate_progress():
    g = ForwardProgress()
    for _ in range(100):
        tick(g, 0)
    assert not g.blocked
    tick(g, 30, valid=False)
    tick(g, 31)
    assert not g.blocked
    tick(g, 32, forward=False)  # intentional stop / gear confirmation
    tick(g, 33)
    assert not g.blocked


def test_backward_time_and_gaps_restart_window_but_keep_failure_latch():
    g = ForwardProgress()
    for i in range(15):
        tick(g, i)
    tick(g, 0)
    tick(g, 30)
    assert not g.blocked
    for i in range(31, 55):
        tick(g, i)
    assert g.blocked
    tick(g, 1)
    assert g.blocked


def test_latch_evaluates_safe_reverse_despite_excellent_forward_prediction():
    motion, _ = evaluate((0., 0., 0.), target=(4., 0.), distance=2.,
        wheelbase=1.087, steering_limit=.314, clear=lambda path: (True, 'clear'),
        allow_forward=False)
    assert motion.direction == -1


def test_latch_cannot_bypass_reverse_safety():
    motion, _ = evaluate((0., 0., 0.), target=(4., 0.), distance=2.,
        wheelbase=1.087, steering_limit=.314, clear=lambda path: (True, 'clear'),
        reverse_clear=lambda path: (False, 'vehicle_collision'), allow_forward=False)
    assert motion is None


def test_valid_mpc_cannot_handoff_when_measured_progress_failed():
    from types import SimpleNamespace as NS
    from .test_boundary_recovery import recovery_controller, tick
    from multi_purpose_mpc_ros.core.boundary_recovery import Motion
    c = recovery_controller()
    c._forward_progress = ForwardProgress()
    for i in range(25):
        c._forward_progress.update(i/10, 0., 0., 0., valid=True, forward=True)
    c._mpc = NS(infeasibility_counter=0, current_prediction=object(),
                used_prediction_fallback=False, recovery_requested=False,
                time_budget_exceeded=False, max_steering_rate=1.)
    c._select_reentry_motion.return_value = (Motion(-1, 0., (), 0.), 'safe_reverse')
    for t in (3., 3.1, 3.2, 3.3):
        tick(c, t)
    assert c._straight_reentry_active
    assert c._straight_reentry_success_cycles == 0
    assert c._straight_reentry_direction == -1
    c._reentry_mpc_path_is_valid.assert_not_called()


def test_measured_wall_free_relocation_allows_fresh_path_evaluation():
    g = ForwardProgress()
    for i in range(25):
        tick(g, i)
    tick(g, 25, x=.6)
    assert not g.blocked


def test_motion_start_ignores_waiting_and_speed_noise():
    from multi_purpose_mpc_ros.core.boundary_recovery import MotionStart
    gate=MotionStart()
    for i in range(100):
        assert not gate.update(i*.1, .01*(i%2), 0., .2, True)
    assert gate.update(10., .4, 0., .3, True)


def test_motion_start_rejects_placement_jump_stale_and_duplicate_samples():
    from multi_purpose_mpc_ros.core.boundary_recovery import MotionStart
    gate=MotionStart()
    assert not gate.update(0.,0.,0.,0.,True)
    assert not gate.update(.1,5.,0.,.3,True)
    assert not gate.update(.1,6.,0.,.3,True)
    assert not gate.update(2.,6.,0.,.3,True)
    assert not gate.update(2.1,6.4,0.,.3,False)
    assert not gate.update(2.2,6.4,0.,.3,True)
    assert gate.update(2.3,6.8,0.,.3,True)


def test_prestart_does_not_enter_reverse_without_awsim_state():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    from .test_overtake_session import controller_method
    c=NS(_stuck_recovery_enabled=True, _straight_reentry_active=False,
         _velocity_report=NS(longitudinal_velocity=0.),
         _collision_ego_metadata=(0.,'map',True),
         _start_straight_reentry=Mock(), _publish_gear_command=Mock())
    # Deliberately no AWSIM state in this fixture.
    for i in range(100):
        c._last_gnss_received_sec=i*.1
        c.get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(c._last_gnss_received_sec*1e9)))
        u=[1.,.1]
        assert not controller_method('_apply_stuck_recovery')(
            c,None,u,0.,NS(x=0.,y=0.,theta=0.))
        assert u==[1.,.1]
    c._start_straight_reentry.assert_not_called()
    c._publish_gear_command.assert_not_called()
    assert not c._forward_progress.blocked


def failed_guard(overlap=1., heading=0.):
    g=ForwardProgress()
    for i in range(25):
        g.update(i/10,0.,0.,overlap,valid=True,forward=True,heading=heading)
    assert g.blocked
    return g


def test_alternative_forward_escape_releases_without_reverse():
    g=failed_guard()
    g.update(2.5,.2,0.,.8,valid=True,forward=True,heading=0.)
    assert not g.blocked


def test_motion_without_contact_improvement_keeps_failed_pose_veto():
    g=failed_guard()
    g.update(2.5,.2,0.,1.,valid=True,forward=True,heading=0.)
    assert g.blocked
    g.update(2.6,.6,0.,1.2,valid=True,forward=True,heading=.4)
    assert g.blocked


def test_new_heading_can_recheck_mpc_while_stopped_but_noise_cannot():
    g=failed_guard()
    g.update(2.5,0.,0.,1.,valid=True,forward=False,heading=.02)
    assert g.blocked
    g.update(2.6,0.,0.,1.,valid=True,forward=False,heading=.2)
    assert not g.blocked


def test_overlap_change_alone_or_invalid_observation_cannot_release():
    g=failed_guard()
    g.update(2.5,0.,0.,.1,valid=True,forward=True,heading=0.)
    assert g.blocked
    g.update(2.6,.6,0.,.1,valid=False,forward=True,heading=.4)
    assert g.blocked


def test_changed_pose_reopens_excluded_motion_evaluation():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    a=RecoveryAttempts()
    a.observe(0.,(0.,0.,0.),1.,1,.3,True)
    a.observe(1.6,(0.,0.,0.),1.,1,.3,True)
    assert a.excluded == {(1,1)}
    a.observe(1.7,(0.,0.,.02),1.,1,.3,False)
    assert a.excluded
    a.observe(1.8,(0.,0.,.2),1.,1,.3,False)
    assert not a.excluded


def test_release_still_requires_fresh_safe_mpc_path():
    from types import SimpleNamespace as NS
    from .test_boundary_recovery import recovery_controller, tick
    c=recovery_controller()
    c._forward_progress=failed_guard()
    c._forward_progress.update(2.5,.2,0.,.8,valid=True,forward=True,heading=0.)
    c._mpc=NS(infeasibility_counter=0,current_prediction=object(),
              used_prediction_fallback=False,recovery_requested=False,
              time_budget_exceeded=False,max_steering_rate=1.)
    c._reentry_mpc_path_is_valid.return_value=False
    tick(c,3.)
    c._reentry_mpc_path_is_valid.assert_called_once()
    assert c._straight_reentry_active
    assert c._straight_reentry_success_cycles == 0
    c._reentry_mpc_path_is_valid.return_value=True
    for t in (3.1,3.2,3.3):
        tick(c,t)
    # MPC confirmation does not bypass the still-pending steering setup.
    assert c._straight_reentry_active
    tick(c,3.6)
    assert not c._straight_reentry_active


def test_controller_release_clears_old_candidate_veto_before_mpc_evaluation():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    from .test_overtake_session import controller_method
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    c=NS(_stuck_recovery_enabled=True,_recovery_motion_confirmed=True,
         _straight_reentry_active=True,_forward_progress=failed_guard(),
         _recovery_attempts=RecoveryAttempts(),_collision_ego_metadata=(0.,'map',True),
         _last_gnss_received_sec=2.5,_reentry_overlap=lambda pose:.8,
         _enable_control=True,_collision_evidence_hold=False,_intentional_follow_stop_active=False,
         _current_gear_is_reverse=lambda:False,_current_gear_is_drive=lambda:True,
         _last_u=[.5,.3],_apply_straight_reentry=Mock(return_value=True),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=2500000000)),get_logger=Mock())
    c._recovery_attempts.excluded={(1,1)}
    c._recovery_attempts.failure_position=(0.,0.,0.)
    assert controller_method('_apply_stuck_recovery')(
        c,NS(nanoseconds=2500000000),[0.,0.],0.,NS(x=.2,y=0.,theta=0.))
    assert not c._forward_progress.blocked
    assert not c._recovery_attempts.excluded
    c._apply_straight_reentry.assert_called_once()


def test_short_path_speed_cap_and_progress_monitor_agree():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts, recovery_speed_limit
    speed = recovery_speed_limit(((0.,0.,0.),(.1,0.,0.)))
    for direction in (1, -1):
        attempts = RecoveryAttempts()
        for i in range(121):
            t = i*.025
            attempts.observe(t, (direction*speed*t,0.,0.), max(0.,1.-speed*t),
                             direction, 0., True, commanded_speed=speed)
        assert not attempts.excluded
        assert attempts.improving_key == (direction, 0)


def test_low_command_still_rejects_noise_or_unimproved_wall_contact():
    from multi_purpose_mpc_ros.core.boundary_recovery import RecoveryAttempts
    for actual_speed in (0., .005, .09):
        attempts = RecoveryAttempts()
        for i in range(61):
            t = i*.025
            attempts.observe(t, (actual_speed*t,0.,0.), 1., 1, .3, True,
                             commanded_speed=.09)
        assert (1,1) in attempts.excluded
