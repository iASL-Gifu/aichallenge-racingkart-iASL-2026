"""Position disagreement must not prevent collision recovery commands."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from .test_four_vehicle_deadlock import method


@pytest.mark.parametrize('error,age,available', [
    (0., 0., True), (1.01, 0., True), (5., 0., True),
    (1.01, 0.6, False), (float('nan'), 0., False),
])
def test_recovery_requires_fresh_positions_but_not_agreement(error, age, available):
    def message(x):
        return NS(pose=NS(pose=NS(position=NS(x=x, y=0.))))
    c = NS(_odom=message(0.), _gnss_pose=message(error),
           _last_odom_received_sec=10.-age, _last_gnss_received_sec=10.,
           _adaptive_reverse_localization_fresh_sec=.5,
           _adaptive_reverse_localization_max_error=.5,
           _adaptive_reverse_localization_confirm_sec=.25,
           _localization_consistent_since=None)
    method('_update_localization_consistency')(c, 10.)
    assert c._recovery_localization_available is available
    if error > .5:
        assert not c._localization_consistent


@pytest.mark.parametrize('available,expected_speed', [(True, 1.), (False, 0.)])
def test_straight_reentry_commands_drive_despite_one_meter_position_error(
        available, expected_speed):
    from .test_boundary_recovery import recovery_controller, tick
    from multi_purpose_mpc_ros.core.boundary_recovery import Motion
    c = recovery_controller()
    c._localization_consistent = False
    c._localization_position_error = 1.01
    c._straight_reentry_direction = 1
    c._select_reentry_motion.return_value = (
        (Motion(1,0.,(),.2), 'selected') if available
        else (None, 'localization_unavailable'))
    assert tick(c,10.) == [expected_speed, 0.]


def localization_controller():
    def message(x):
        return NS(pose=NS(pose=NS(position=NS(x=x, y=0.))))
    return NS(_odom=message(0.), _gnss_pose=message(1.),
              _last_odom_received_sec=100., _last_gnss_received_sec=100.,
              _adaptive_reverse_localization_fresh_sec=.5,
              _adaptive_reverse_localization_max_error=.5,
              _adaptive_reverse_localization_confirm_sec=.25,
              _localization_consistent_since=99.,
              _localization_checked_at_sec=None)


@pytest.mark.parametrize('previous_check', [None, 100.])
def test_clock_rewind_requires_both_streams_again_even_after_clock_catches_up(previous_check):
    c = localization_controller()
    c._localization_checked_at_sec = previous_check
    update = method('_update_localization_consistency')
    update(c, 1.)
    assert not c._recovery_localization_available
    assert c._localization_consistent_since is None
    assert c._last_odom_received_sec is None
    assert c._last_gnss_received_sec is None
    update(c, 100.)
    assert not c._recovery_localization_available
    c._last_odom_received_sec = 100.
    update(c, 100.)
    assert not c._recovery_localization_available
    c._last_gnss_received_sec = 100.
    update(c, 100.)
    assert c._recovery_localization_available
    assert not c._localization_consistent  # position disagreement stays diagnostic


def test_rewind_discards_confirmation_even_if_receipts_have_already_updated():
    c = localization_controller()
    c._localization_checked_at_sec = 100.
    c._last_odom_received_sec = c._last_gnss_received_sec = 1.
    method('_update_localization_consistency')(c, 1.)
    assert not c._recovery_localization_available
    assert c._localization_consistent_since is None


def test_control_validates_with_clock_after_rate_wait():
    import ast
    from .test_hybrid_integration import control_tree
    call = next(n for n in ast.walk(control_tree()) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == 'self._update_localization_consistency')
    c = localization_controller()
    c.get_clock = lambda: NS(now=lambda: NS(nanoseconds=100_000_000_000))
    from types import MethodType
    c._update_localization_consistency = MethodType(method('_update_localization_consistency'), c)
    exec(compile(ast.Module(body=[call], type_ignores=[]), '<clock>', 'exec'),
         dict(self=c, current_time_sec=99.))
    assert c._recovery_localization_available


def boundary_localization_controller():
    from types import MethodType
    c = localization_controller()
    c._odom.pose.pose.orientation = NS(x=0., y=0., z=0., w=1.)
    c._reentry_violation = lambda *args: .4
    c._update_localization_consistency = MethodType(method('_update_localization_consistency'), c)
    c._boundary_recovery_input = MethodType(method('_boundary_recovery_input'), c)
    c.clock_sec = 100.
    c.get_clock = lambda: NS(now=lambda: NS(nanoseconds=int(c.clock_sec*1e9)))
    return c


def test_boundary_recheck_ignores_old_cycle_time_without_discarding_receipts():
    c = boundary_localization_controller()
    c._update_localization_consistency(100.)
    # The same cycle later rechecks with the time captured before waiting.
    assert c._boundary_recovery_input(NS(x=0.,y=0.,theta=0.), 99.) == (0., 'valid')
    assert c._last_odom_received_sec == 100.
    assert c._last_gnss_received_sec == 100.
    assert c._localization_checked_at_sec == 100.


@pytest.mark.parametrize('clock_sec,reason,cleared', [
    (100.6, 'position_unavailable_or_stale', False),
    (1., 'position_unavailable_or_stale', True),
])
def test_boundary_current_clock_still_rejects_stale_data_and_real_rewind(clock_sec, reason, cleared):
    c = boundary_localization_controller()
    c._update_localization_consistency(100.)
    c.clock_sec = clock_sec
    assert c._boundary_recovery_input(NS(x=0.,y=0.,theta=0.), 100.) == (None, reason)
    assert (c._last_gnss_received_sec is None) is cleared
    assert (c._last_odom_received_sec is None) is cleared


def test_stationary_window_survives_cycle_time_older_than_previous_gnss_sample():
    c = boundary_localization_controller()
    c._straight_reentry_enabled = c._enable_control = True
    c._collision_evidence_hold = c._intentional_follow_stop_active = False
    c._velocity_report = NS(longitudinal_velocity=0.)
    c._stuck_speed_threshold = .15
    c._stuck_time_threshold = 2.
    c._stuck_gnss_distance_threshold = .3
    c._boundary_stop_samples = []
    c.get_logger = Mock(return_value=Mock())
    ready = method('_boundary_recovery_ready')
    for i in range(21):
        stamp = 100.+i*.1
        c._last_gnss_received_sec = c._last_odom_received_sec = stamp
        c.clock_sec = stamp+.01
        c._update_localization_consistency(c.get_clock().now().nanoseconds / 1e9)
        result = ready(c, NS(x=0.,y=0.,theta=0.), stamp-.2)
        assert result is (i == 20)
        assert c._last_gnss_received_sec == stamp
    assert len(c._boundary_stop_samples) == 21


@pytest.mark.parametrize('yaw_ns,expected',[(3_589_999_919,True),(3_604_999_919,False),(2_900_000_000,False)])
def test_collision_metadata_uses_current_clock_and_tolerates_float_roundoff(yaw_ns,expected):
    from pathlib import Path
    import textwrap
    source=(Path(__file__).parents[1]/'multi_purpose_mpc_ros/mpc_controller.py').read_text()
    start=source.index('        self._collision_ego_yaw = float(pose.theta)',source.index('        #オドメトリ(x,y,yaw,v)取得'))
    end=source.index('        self._collision_ego_alignment = None',start)
    def msg(ns):
        return NS(header=NS(stamp=NS(sec=ns//1_000_000_000,nanosec=ns%1_000_000_000),frame_id='map'))
    c=NS(_collision_now=3.55,_gnss_pose=msg(3_549_999_920),_odom=msg(yaw_ns),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=3_589_999_919)))
    exec(textwrap.dedent(source[start:end]),{'self':c,'pose':NS(theta=0.)})
    assert c._collision_ego_metadata[2] is expected
    assert c._collision_now == 3_589_999_919/1e9
