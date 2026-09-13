"""Shared L1 readiness, timeout reference handoff, and collision time origin."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from .test_hybrid_integration import controller, apply
from .test_postpass_rejoin import completed_pass, propose_return
from multi_purpose_mpc_ros.overtake_session import LaneDecision
from multi_purpose_mpc_ros.core.control_continuity import (
    CorridorState, l1_entry_confirmed, prepass_return_alternative)
from multi_purpose_mpc_ros import collision_geometry as g


@pytest.mark.parametrize('prohibited', [False, True])
def test_l1_requests_cannot_bypass_geometry_after_transition_timer(prohibited):
    c = controller()
    apply(c, lane=1, now=1., l1_entry_ready=False, l0_prohibited=prohibited)
    apply(c, lane=1, now=2., l1_entry_ready=False, l0_prohibited=prohibited)
    assert c._lane_decision.applied_lane is None
    assert c._lane_decision.mode == 'l1_geometry_wait'
    assert c._l1_entry_waiting
    apply(c, lane=1, now=3., l1_entry_ready=True, l0_prohibited=prohibited)
    assert c._lane_decision.applied_lane == 1
    # Once admitted, threshold noise must not repeatedly release L1.
    apply(c, lane=1, now=4., l1_entry_ready=False, l0_prohibited=prohibited)
    assert c._lane_decision.applied_lane == 1


def test_l1_probe_does_not_count_unapplied_constraint():
    c = controller()
    c._l1_probe_active = True
    c._l1_probe_constraint_applied = True  # stale success must not survive waiting
    apply(c, lane=1, now=1., l1_entry_ready=False)
    apply(c, lane=1, now=2., l1_entry_ready=False)
    assert not c._l1_probe_constraint_applied
    apply(c, lane=1, now=3., l1_entry_ready=True)
    assert c._l1_probe_constraint_applied


def test_l1_readiness_requires_continuous_time_and_resets_on_clock_rewind():
    c = NS()
    assert not l1_entry_confirmed(c, 1., True, .3)
    assert l1_entry_confirmed(c, 1.3, True, .3)
    assert not l1_entry_confirmed(c, 1.4, False, .3)
    assert not l1_entry_confirmed(c, 1.5, True, .3)
    assert not l1_entry_confirmed(c, 1.2, True, .3)
    assert l1_entry_confirmed(c, 1.5, True, .3)


def timeout_controller():
    c = completed_pass()
    c._postpass_outer_corridor = None
    c._reference_path.target_lane_idx = None
    c._mpc.soft_target_alpha = .7
    c._lane_decision = LaneDecision('d2', None, None, 'full_width_recovery')
    c._committed_corridor = CorridorState.capture(c)
    c._prepass_return_corridor = c._committed_corridor
    # High-level target expires; only geometry is retained.
    c._overtake.release_target()
    return c


def test_timeout_rejected_rejoin_keeps_previous_guidance_not_target_proof(monkeypatch):
    c = timeout_controller()
    propose_return(c, monkeypatch, safe=False)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 2.
    assert c._mpc.soft_target_alpha == .7
    assert c._prepass_return_corridor is not None
    assert c._overtake.target_id is None
    assert c._overtake.accepted_key == (None, None)
    assert not c._corridor_hold_failed
    # A subsequent valid replacement finishes the handoff.
    propose_return(c, monkeypatch, safe=True)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 3.
    assert c._mpc.soft_target_alpha == .6
    assert c._prepass_return_corridor is None


def test_timeout_both_routes_unsafe_requests_recovery(monkeypatch):
    c = timeout_controller()
    propose_return(c, monkeypatch, safe=False, outer_safe=False)
    command, _ = c._solve_with_corridor_commit(NS(), 2.)
    assert command[0] == 0.
    assert c._corridor_hold_failed and c._mpc.recovery_requested
    assert c._prepass_return_corridor is None


@pytest.mark.parametrize('owner', ['_manual_control_override', '_straight_reentry_active',
    '_mpc_safety_recovery_active', '_parallel_abort_active',
    '_prepass_fallback_follow_active', '_prepass_retry_after_reverse',
    '_postpass_rejoin_suppressed'])
def test_timeout_geometry_cannot_override_safety_owner(owner):
    c = timeout_controller()
    setattr(c, owner, True)
    assert prepass_return_alternative(c, CorridorState.capture(c)) is None


def projection(monkeypatch, body, velocity=(4., 0.), known=True, now=10.):
    c = NS(_collision_now=now, _v2x_tracker=NS(
        has_velocity_estimate=lambda v: known, velocity=lambda v: velocity))
    monkeypatch.setattr(g, 'target_body', lambda *a: body)
    return g.current_target_body(c, 'd2')


def test_hold_and_sweep_use_same_current_other_position(monkeypatch):
    ego = g.body_pose(0., 0., 0., 10.)
    observed = g.body_pose(2., 0., 0., 9.75, uncertainty=.01)
    current = projection(monkeypatch, observed)
    assert current.x == 3. and current.stamp == 10.
    assert g.overlaps(ego, observed, g.BodyGeometry())
    assert not g.overlaps(ego, current, g.BodyGeometry())
    times = g.prediction_times_from_observation(observed.stamp, 10., [0., .2])
    assert g.swept_path_clear([ego, replace(ego, x=.2)], times,
                              observed, (4., 0.), g.BodyGeometry())
    assert observed.x == 2. and observed.stamp == 9.75
    assert current.uncertainty == observed.uncertainty


@pytest.mark.parametrize('change', ['stale', 'future', 'unknown_velocity', 'nan_velocity'])
def test_current_overlap_never_treats_missing_evidence_as_safe(monkeypatch, change):
    body = g.body_pose(2., 0., 0., 9.75)
    kwargs = {}
    if change == 'stale': body = replace(body, position_valid=False)
    if change == 'future': body = replace(body, stamp=11.)
    if change == 'unknown_velocity': kwargs['known'] = False
    if change == 'nan_velocity': kwargs['velocity'] = (float('nan'), 0.)
    assert projection(monkeypatch, body, **kwargs) is None


def test_stationary_uncertainty_overlap_remains_rejected(monkeypatch):
    ego = g.body_pose(0., 0., 0., 10.)
    observed = g.body_pose(-2.2, 0., 0., 9.75, uncertainty=.2)
    current = projection(monkeypatch, observed, velocity=(0., 0.))
    assert g.overlaps(ego, current, g.BodyGeometry())
    assert not g.separating_path_clear([ego, replace(ego, x=.2)], [.25, .45],
                                      observed, (0., 0.), g.BodyGeometry())
