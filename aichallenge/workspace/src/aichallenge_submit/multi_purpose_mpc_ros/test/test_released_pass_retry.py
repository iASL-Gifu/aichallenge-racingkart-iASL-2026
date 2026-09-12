"""A released pass may re-enter ordinary verification, never skip it."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from multi_purpose_mpc_ros.core.released_pass_retry import ReleasedPassRetry
from multi_purpose_mpc_ros.overtake_session import OvertakeSession
from .test_four_vehicle_deadlock import method as controller_method


def observe(state, now, speed=7., clear=(0, 2), lane=2, ready=True, target='d3'):
    return state.observe(target=target, now=now, speed=speed, clear_lanes=clear,
                         candidate_lane=lane, ready=ready, slow_speed=4.44)


def test_d3_slowdown_after_rejoin_rearms_only_after_confirmation():
    state = ReleasedPassRetry()
    assert observe(state, 0.) is None
    assert observe(state, 21., speed=1.58) is None
    assert observe(state, 21.49, speed=1.4) is None
    assert observe(state, 21.5, speed=1.4) == 'lead_slowed'


def test_unchanged_slow_vehicle_does_not_cause_periodic_retries():
    state = ReleasedPassRetry()
    for now in (0., 2., 3., 30., 300.):
        assert observe(state, now, speed=1.4) is None


def test_passage_improvement_without_speed_change():
    state = ReleasedPassRetry()
    assert observe(state, 0., clear=(0,)) is None
    assert observe(state, 2.) is None
    assert observe(state, 2.5) == 'passage_improved'


def test_restored_passage_after_a_later_blockage_counts_as_change():
    state = ReleasedPassRetry()
    observe(state, 0.)
    observe(state, 3., clear=())
    assert observe(state, 4.) is None
    assert observe(state, 4.5) == 'passage_improved'


def test_cooldown_busy_owner_and_candidate_flips_restart_confirmation():
    state = ReleasedPassRetry()
    observe(state, 0.)
    assert observe(state, 1., speed=1.) is None
    assert observe(state, 2., speed=1., ready=False) is None
    assert observe(state, 3., speed=1.) is None
    assert observe(state, 3.3, speed=1., lane=0) is None
    assert observe(state, 3.6, speed=1., lane=2) is None
    assert observe(state, 4.1, speed=1.) == 'lead_slowed'


def test_missing_speed_target_change_and_clock_reset_do_not_reuse_pending_event():
    state = ReleasedPassRetry()
    observe(state, 10.)
    observe(state, 12., speed=1.)
    assert observe(state, 12.5, speed=float('nan')) is None
    assert observe(state, 13., speed=1.) is None
    assert observe(state, 13.5, speed=1., target='d4') is None
    assert observe(state, 0., speed=1., target='d4') is None


def controller():
    path = NS(target_lane_idx=None)
    session = OvertakeSession()
    session.target_id, session.requested_lane, session.committed = 'd3', 2, False
    session.probe.vehicle_id, session.probe.lane_idx, session.probe.confirmed = 'd3', 2, True
    session.verification.vehicle_id, session.verification.lane_idx = 'd3', 2
    return NS(_outer_lane_released_vehicle_id='d3', _overtake=session,
              _reference_path=path, _reference_pathN_center=path, _target_lane_idx=None,
              _enable_control=True, _stuck_recovery_until=None, _awsim_state='Start',
              _constraint_transition_until=.6, _overtake_latch_max_distance=35.,
              _slow_lead_overtake_speed=4.44, _cfg=NS(trajectory_switch=NS()),
              _reset_overtake_commit_probe=Mock(), _clear_committed_shadow_verification=Mock(),
              _reset_outer_lane_progress=Mock(), get_logger=Mock(return_value=Mock()))


def tick(c, now, speed=7., **kwargs):
    params = dict(now_sec=now, vehicle_id='d3', speed=speed, velocity_valid=True,
                  clear_lanes={0, 2}, candidate_lane=2, distance=15., recovery_active=False)
    params.update(kwargs)
    return controller_method('_retry_released_outer_pass')(c, **params)


def test_rearm_clears_stale_lane_and_shadow_but_never_applies_candidate():
    c = controller()
    assert not tick(c, 0.)
    assert not tick(c, 21., 1.58)
    assert tick(c, 21.5, 1.4)
    assert c._outer_lane_released_vehicle_id is None
    assert c._overtake.target_id is None and c._overtake.requested_lane is None
    assert not c._overtake.probe.confirmed
    assert c._overtake.verification.vehicle_id is None
    assert c._reference_path.target_lane_idx is None and c._target_lane_idx is None
    c._reset_overtake_commit_probe.assert_called_once()
    assert not tick(c, 22., 1.4)


@pytest.mark.parametrize('flag', [
    '_center_lane_rejoin_active', '_center_lane_rejoin_constraint_released',
    '_l1_probe_active', '_l1_safety_reprobe_pending', '_prepass_fallback_recovery_active',
    '_prepass_fallback_commit_pending', '_prepass_fallback_follow_active',
    '_prepass_fallback_blocked', '_parallel_abort_active', '_follow_escape_active',
    '_mpc_safety_recovery_active', '_post_reverse_full_width_recovery_active',
    '_straight_reentry_active', '_follow_only', '_prepass_retry_after_reverse',
    '_close_obstacle_reverse_requested', '_l1_safety_recovery_active',
    '_race_rejoin_handoff_active', '_initial_start_exclusive_active',
    '_initial_start_post_hold_active'])
def test_rearm_waits_for_existing_owner_to_finish(flag):
    c = controller()
    tick(c, 0.)
    setattr(c, flag, True)
    assert not tick(c, 3., 1.)
    assert not tick(c, 4., 1.)
    assert c._overtake.target_id == 'd3'
    setattr(c, flag, False)
    assert not tick(c, 5., 1.)
    assert tick(c, 5.5, 1.)


@pytest.mark.parametrize('kwargs', [dict(vehicle_id='d4'), dict(velocity_valid=False),
                                     dict(candidate_lane=1), dict(clear_lanes=set()),
                                     dict(distance=-1.), dict(distance=50.),
                                     dict(recovery_active=True)])
def test_invalid_candidate_cannot_clear_released_state(kwargs):
    c = controller()
    tick(c, 0.)
    assert not tick(c, 3., 1., **kwargs)
    assert not tick(c, 4., 1., **kwargs)
    assert c._outer_lane_released_vehicle_id == 'd3'


def test_releasing_again_requires_new_evidence_not_a_timer_alone():
    c = controller()
    tick(c, 0.)
    tick(c, 3., 1.)
    assert tick(c, 3.5, 1.)
    c._outer_lane_released_vehicle_id = c._overtake.target_id = 'd3'
    c._overtake.requested_lane = 2
    for now in (4., 7., 20.):
        assert not tick(c, now, 1.)

