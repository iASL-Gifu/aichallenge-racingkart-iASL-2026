"""Operator takeover owns mode, gear and actuation even during recovery."""
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock
import threading
import pytest
from .test_overtake_session import controller_method


def controller():
    c = NS(_manual_control_lock=threading.RLock(), _manual_control_override=False,
           _manual_recovery_reset_pending=False, _enable_control=True,
           _control_mode_report=None, get_logger=Mock(return_value=Mock()),
           _awsim_control_mode_request_pub=Mock())
    for name in ('_set_manual_control_override', '_control_mode_status_callback',
                 '_control_mode_request_callback'):
        setattr(c, name, MethodType(controller_method(name), c))
    return c


def report(manual):
    return NS(mode=4 if manual else 1, MANUAL=4, AUTONOMOUS=1)


@pytest.mark.parametrize('source', ['request', 'status'])
def test_manual_takeover_suppresses_all_controller_outputs(source):
    c = controller()
    if source == 'request':
        c._control_mode_request_callback(NS(data=False))
        assert c._awsim_control_mode_request_pub.publish.call_args.args[0].data is False
    else:
        c._control_mode_status_callback(report(True))
    assert c._manual_control_override and not c._enable_control
    assert c._manual_recovery_reset_pending
    c._awsim_control_mode_request_pub.reset_mock()
    # No other publisher fields are installed: reaching the old bodies fails.
    for name, args in (
        ('_publish_control_command', (None, [1., .3], 2., False)),
        ('_publish_gear_command', (None, 2)),
        ('_publish_stuck_actuation_command', (None, 1., 0., 0.)),
        ('_request_awsim_control_mode_for_recovery', ())):
        controller_method(name)(c, *args)
    c._awsim_control_mode_request_pub.publish.assert_not_called()


def test_delayed_autonomous_report_does_not_cancel_manual_request():
    c = controller()
    c._control_mode_status_callback(report(False))
    c._control_mode_request_callback(NS(data=False))
    c._control_mode_status_callback(report(False))
    assert c._manual_control_override
    c._control_mode_status_callback(report(True))
    c._control_mode_status_callback(report(False))
    assert not c._manual_control_override and c._enable_control


def test_explicit_auto_request_releases_override():
    c = controller()
    c._control_mode_request_callback(NS(data=False))
    c._control_mode_request_callback(NS(data=True))
    assert not c._manual_control_override and c._enable_control
    assert c._manual_recovery_reset_pending  # control thread discards old recovery


def test_recovery_never_reasserts_auto_before_manual_status_arrives():
    c = controller()
    c._stuck_request_control_mode = True
    controller_method('_request_awsim_control_mode_for_recovery')(c)
    c._awsim_control_mode_request_pub.publish.assert_not_called()


def test_inflight_publish_rechecks_manual_under_output_lock():
    c = controller()
    publish = controller_method('_publish_control_command')
    started = threading.Event()
    errors = []
    def worker():
        started.set()
        try:
            publish(c, None, [1., 0.], 2., False)
        except Exception as e:
            errors.append(e)
    with c._manual_control_lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(1.)
        c._control_mode_request_callback(NS(data=False))
    thread.join(2.)
    assert not thread.is_alive() and not errors


def test_manual_cancels_legacy_and_waypoint_recovery_before_auto_resume():
    c = controller()
    c._stuck_recovery_enabled = True
    c.get_clock = lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000))
    c._straight_reentry_active = True
    c._stuck_recovery_until = 99.
    c._stuck_reverse_drive_active = True
    c._prepass_retry_after_reverse = True
    c._control_mode_request_callback(NS(data=False))
    apply = controller_method('_apply_stuck_recovery')
    pose = NS(x=0., y=0., theta=0.)
    assert not apply(c, None, [1., 0.], 0., pose)
    assert not c._straight_reentry_active and not c._stuck_reverse_drive_active
    assert c._stuck_recovery_until is None and not c._prepass_retry_after_reverse
    assert c._manual_recovery_reset_pending
    c._control_mode_request_callback(NS(data=True))
    c._velocity_report = NS(longitudinal_velocity=0.)
    assert not apply(c, None, [0., 0.], 0., pose)
    assert not c._manual_recovery_reset_pending
