"""Near-zero interpolation has a bounded effect on genuine stop detection."""
import pytest
from multi_purpose_mpc_ros.v2x_vehicle_tracker import V2XVehicleTracker
from .test_v2x_vehicle_tracker import _msg


def moving(**kwargs):
    tracker = V2XVehicleTracker(20., 5., **kwargs)
    tracker.update(_msg(0., [('car', 0., 0.)]))
    tracker.update(_msg(.1, [('car', .4, 0.)]))
    assert tracker.velocity('car') == pytest.approx((4., 0.))
    return tracker


def test_real_stop_is_delayed_only_until_last_motion_plus_point_two_seconds():
    t = moving()
    for stamp in (.15, .2, .25, .299):
        t.update(_msg(stamp, [('car', .4, 0.)]))
        assert t.has_velocity_estimate('car')
        assert t.velocity('car') == pytest.approx((4., 0.))
    t.update(_msg(.301, [('car', .4, 0.)]))
    assert t.has_velocity_estimate('car')
    assert t.velocity('car') == (0., 0.)
    t.update(_msg(.4, [('car', .4, 0.)]))
    assert t.velocity('car') == (0., 0.)


@pytest.mark.parametrize('speed,held', [(0., True), (.09, True), (.11, False)])
def test_near_zero_threshold(speed, held):
    t = moving()
    t.update(_msg(.2, [('car', .4 + speed*.1, 0.)]))
    assert t.velocity('car')[0] == pytest.approx(4. if held else speed)


def test_zero_hold_can_be_disabled():
    t = moving(zero_velocity_hold_sec=0.)
    t.update(_msg(.2, [('car', .4, 0.)]))
    assert t.velocity('car') == (0., 0.)


def test_snapshot_copies_hold_configuration_history_and_future_updates_are_independent():
    t = moving(zero_velocity_hold_sec=.15, zero_velocity_threshold=.08)
    snap = t.snapshot()
    t.update(_msg(.2, [('car', .9, 0.)]))
    snap.update(_msg(.2, [('car', .4, 0.)]))
    assert t.velocity('car')[0] == pytest.approx(5.)
    assert snap.velocity('car')[0] == pytest.approx(4.)
    snap.update(_msg(.251, [('car', .4, 0.)]))
    assert snap.velocity('car') == (0., 0.)
    assert t._last_moving_velocity_times['car'] == pytest.approx(.2)
    assert snap._last_moving_velocity_times['car'] == pytest.approx(.1)
    assert snap._zero_velocity_threshold == .08


@pytest.mark.parametrize('stamp,x', [(.2, 10.), (.11, 1.), (.1, .41), (.05, .4), (.2, float('nan'))])
def test_invalid_measurement_discards_old_motion_before_next_stationary_sample(stamp,x):
    t = moving()
    t.update(_msg(stamp, [('car', x, 0.)]))
    assert not t.has_velocity_estimate('car')
    assert 'car' not in t._last_moving_velocities
    assert 'car' not in t._last_moving_velocity_times
    x = x if x == x else .4
    t.update(_msg(.3, [('car', x, 0.)]))
    t.update(_msg(.4, [('car', x, 0.)]))
    assert t.has_velocity_estimate('car')
    assert t.velocity('car') == (0., 0.)


def test_motion_history_is_per_vehicle():
    t = moving()
    t.update(_msg(.2, [('car', .4, 0.), ('stopped', 10., 0.)]))
    t.update(_msg(.25, [('car', .4, 0.), ('stopped', 10., 0.)]))
    assert t.velocity('car')[0] == pytest.approx(4.)
    assert t.velocity('stopped') == (0., 0.)


def test_exact_retransmission_preserves_samples_velocity_and_hold_deadline():
    t=moving()
    samples=list(t._samples['car'])
    for _ in range(5):
        t.update(_msg(.1,[('car',.4,0.)]))
    assert list(t._samples['car'])==samples
    assert t.velocity('car')[0]==pytest.approx(4.)
    assert t.has_velocity_estimate('car')
    snapshot=t.snapshot()
    snapshot.update(_msg(.2,[('car',.4,0.)]))
    snapshot.update(_msg(.2,[('car',.4,0.)]))
    snapshot.update(_msg(.301,[('car',.4,0.)]))
    assert snapshot.velocity('car')==(0.,0.)
    assert t.velocity('car')[0]==pytest.approx(4.)
