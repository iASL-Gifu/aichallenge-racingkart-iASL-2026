"""Reuse cannot bypass changing geometry, traffic gates or Shadow proof counts."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

from multi_purpose_mpc_ros.core.traffic_work import TrafficWork
from multi_purpose_mpc_ros.overtake_session import OvertakeSession, ShadowProbe
from multi_purpose_mpc_ros.v2x_vehicle_tracker import evaluate_lane_width_samples
from .test_overtake_session import controller_method, active_session


def passage_controller():
    wp = NS(x=0., y=0., psi=0., lb=-4., ub=4.)
    path = NS(waypoints=[wp], n_lanes=3, inner_lane_width=.5,
              get_waypoint=lambda i: wp,
              get_lane_bounds=lambda i: [(wp.lb+2, wp.lb), (1, -1), (wp.ub, wp.ub-2)])
    tracker = NS(_samples={'d2': [(0., 0., 0.)]}, velocity=Mock(return_value=(0., 0.)))
    return NS(_traffic_work=TrafficWork(), _reference_pathN_center=path,
              _carN_center=NS(wp_id=0, get_closest_waypoint=Mock(return_value=0)),
              _mpcN_center=NS(N=3), _v2x_tracker=tracker,
              _cfg=NS(bicycle_model=NS(width=1.6)), _v2x_vehicle_radius=.7,
              _passage_clearance=.3, _prepass_lane_fallback_prediction_sec=.3,
              _v2x_t_samples=[0., .1, .2], _passage_lane_width_tolerance=.05,
              _passage_lane_width_tolerance_points=2, get_logger=lambda: Mock())


def test_passage_shared_but_mutable_bounds_and_motion_invalidate():
    c = passage_controller()
    call = controller_method('_vehicle_passage')
    call.__globals__['evaluate_lane_width_samples'] = evaluate_lane_width_samples
    pose = NS(x=0., y=0.)
    first = call(c, 'd2', pose)
    assert first[0] == {0: True, 2: True}
    first[0][0] = False  # callers cannot poison the cache
    assert call(c, 'd2', pose)[0][0]
    assert c._carN_center.get_closest_waypoint.call_count == 3
    c._reference_pathN_center.waypoints[0].lb = -.5
    assert not call(c, 'd2', pose)[0][0]
    assert c._carN_center.get_closest_waypoint.call_count == 6
    c._v2x_tracker.velocity.return_value = (1., 0.)
    call(c, 'd2', pose)
    assert c._carN_center.get_closest_waypoint.call_count == 9
    c._traffic_work.begin_cycle()
    call(c, 'd2', pose)
    assert c._carN_center.get_closest_waypoint.call_count == 12


def test_proposal_hold_expiry_backward_clock_and_changed_gates():
    work = TrafficWork()
    work.remember_proposal(('d2', True), 10., 0)
    assert work.held_proposal(('d2', True), 10.05) == (True, 0)
    assert work.held_proposal(('d3', True), 10.05) == (False, None)
    assert work.held_proposal(('d2', False), 10.05) == (False, None)
    assert work.held_proposal(('d2', True), 10.11) == (False, None)
    assert work.held_proposal(('d2', True), 9.99) == (False, None)


def test_controller_proposal_rechecks_live_gates_and_keeps_active_lane():
    clock = NS(nanoseconds=10_000_000_000)
    c = NS(_traffic_work=TrafficWork(), _overtake=OvertakeSession(),
           _v2x_tracker=NS(active_vehicle_ids=Mock(return_value=['d2']),
                           has_velocity_estimate=lambda vid: True),
           _reference_pathN_center=object(), _overtake_latch_max_distance=35.,
           get_clock=lambda: NS(now=lambda: clock), get_logger=lambda: Mock())
    call = controller_method('_propose_traffic_lane')
    args = (0, {0: True, 2: True}, {0: {}, 2: {}}, 'd2', NS(x=0., y=0.))
    assert call(c, *args) == 0
    clock.nanoseconds += 20_000_000
    assert call(c, *args) == 0
    assert c._traffic_work.proposal_hits == 1
    assert call(c, 0, args[1], {0: {'side': ['d3']}, 2: {}}, *args[3:]) == 2
    assert call(c, 0, args[1], {2: {}}, *args[3:]) == 2
    c._overtake = active_session()
    assert call(c, *args) == 2


def test_probe_failed_retry_pacing_never_counts_skips_as_success():
    work = TrafficWork()
    key = ('overtake', 'd2', 0)
    assert work.probe_due(key, 10.)
    work.record_probe(key, 10., False)
    work.begin_cycle()
    assert not work.probe_due(key, 10.05)
    assert work.probe_due(('overtake', 'd3', 0), 10.05)
    work.record_probe(('overtake', 'd3', 0), 10.05, False)
    work.begin_cycle()
    assert work.probe_due(key, 10.06)  # returning candidate starts fresh
    work.record_probe(key, 10.06, True)
    work.begin_cycle()
    assert work.probe_due(key, 10.07)  # consecutive success remains fast
    assert not work.probe_due(key, 10.07)


def test_skipped_controller_probe_preserves_proof_and_recovery_invalidates():
    path = object()
    c = NS(_traffic_work=TrafficWork(), _overtake=OvertakeSession(
        probe=ShadowProbe(vehicle_id='d2', lane_idx=0)),
        _mpc_safety_recovery_active=False, _post_reverse_full_width_recovery_active=False,
        _reference_path=path, _reference_pathN_center=path,
        get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_050_000_000)))
    c._traffic_work.record_probe(('overtake', 'd2', 0), 10., False)
    call = controller_method('_run_overtake_commit_probe')
    call(c, NS(x=0., y=0., theta=0.), False)
    assert c._overtake.probe.success_cycles == 0
    assert c._overtake.probe.confirmed_at is None
    c._overtake.probe.confirmed = True
    c._overtake.probe.confirmed_at = 10.
    c._overtake.probe.success_cycles = 2
    call(c, NS(x=0., y=0., theta=0.), True)
    assert not c._overtake.probe.confirmed
    assert c._overtake.probe.success_cycles == 0


def test_relative_samples_reuse_and_velocity_validity_path_changes():
    c = passage_controller()
    c._reference_path = c._reference_pathN_center
    c._car = c._carN_center
    c._center_arc_points = ((0., 0.), (10., 0.))
    c._center_arc_cumulative = (0., 10., 20.)
    c._center_arc_total_length = 20.
    c._v2x_tracker.active_vehicle_ids = lambda: ['d2']
    c._v2x_tracker.has_velocity_estimate = Mock(return_value=True)
    c._lane_index_for_position = Mock(return_value=0)
    c._center_longitudinal_between = Mock(return_value=5.)
    call = controller_method('_relative_lane_vehicle_samples')
    pose = NS(x=0., y=0., theta=0.)
    assert len(call(c, pose, 1.)) == 2
    result = call(c, pose, 1.)
    result.clear()
    assert len(call(c, pose, 1.)) == 2
    assert c._lane_index_for_position.call_count == 2
    c._v2x_tracker.has_velocity_estimate.return_value = False
    assert len(call(c, pose, 1.)) == 1
    c._reference_path.waypoints[0].ub = 3.
    call(c, pose, 1.)
    assert c._lane_index_for_position.call_count == 4
    c._traffic_work.begin_cycle()
    call(c, pose, 1.)
    assert c._lane_index_for_position.call_count == 5


def test_confirmed_shadow_refresh_keeps_original_timestamp():
    path = object()
    c = NS(_traffic_work=TrafficWork(), _overtake=OvertakeSession(
        probe=ShadowProbe(vehicle_id='d2', lane_idx=0)),
        _mpc_safety_recovery_active=False, _post_reverse_full_width_recovery_active=False,
        _reference_path=path, _reference_pathN_center=path,
        _overtake_commit_probe_freshness_sec=.75,
        get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_050_000_000)))
    c._overtake.probe.confirmed = True
    c._overtake.probe.confirmed_at = 10.
    c._overtake.probe.success_cycles = 2
    controller_method('_run_overtake_commit_probe')(c, NS(x=0., y=0., theta=0.), False)
    assert c._overtake.probe.confirmed_at == 10.
    assert c._overtake.probe.success_cycles == 2
    assert c._traffic_work.probe_skips == 1


def test_candidate_reset_does_not_inherit_failed_retry_delay():
    c = NS(_traffic_work=TrafficWork(), _overtake=OvertakeSession(
        probe=ShadowProbe(vehicle_id='d2', lane_idx=0)))
    key = ('overtake', 'd2', 0)
    c._traffic_work.record_probe(key, 10., False)
    controller_method('_reset_overtake_commit_probe')(c)
    assert c._traffic_work.probe_due(key, 10.01)
    assert c._overtake.probe.vehicle_id is None
