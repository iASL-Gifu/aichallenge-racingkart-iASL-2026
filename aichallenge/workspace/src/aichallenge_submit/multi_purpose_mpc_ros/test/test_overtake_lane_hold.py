"""Continuation safety, including actual controller evidence methods without ROS."""
from types import SimpleNamespace, MethodType

import pytest

from multi_purpose_mpc_ros.overtake_lane_hold import (
    PassageHold, dynamic_longitudinal_conflict_unsafe,
)
from multi_purpose_mpc_ros.overtake_session import OvertakeSession
from multi_purpose_mpc_ros.v2x_vehicle_tracker import V2XVehicleTracker
from .test_overtake_session import controller_method
from .test_v2x_vehicle_tracker import _msg


def evaluate(hold, time=10., **changes):
    args = dict(key=('d2', 2), now_sec=time, lane_width_valid=True,
                target_geometry_known=True, target_clearance_lost=True,
                body_overlap=False, hybrid_matches=True, hybrid_progress=.1,
                alongside=False, unrelated_unsafe=False, confirm_sec=.4, lock_ratio=.35)
    args.update(changes)
    return hold.evaluate(**args)


def test_target_loss_requires_continuous_confirmation():
    h = PassageHold()
    assert evaluate(h).waiting
    assert evaluate(h, 10.399).waiting
    assert evaluate(h, 10.4).unsafe
    assert not evaluate(h, 10.5, target_clearance_lost=False).unsafe
    assert evaluate(h, 10.6).waiting
    assert evaluate(h, 10.9).waiting
    assert evaluate(h, 11.).unsafe


@pytest.mark.parametrize('change', [dict(alongside=True), dict(hybrid_progress=.35)])
def test_target_only_loss_remains_locked(change):
    h = PassageHold()
    assert evaluate(h, **change).locked
    assert evaluate(h, 100., **change).locked
    assert not evaluate(h, 100., **change).unsafe


@pytest.mark.parametrize('change', [dict(lane_width_valid=False), dict(body_overlap=True),
    dict(unrelated_unsafe=True)])
def test_immediate_safety_overrides_every_lock(change):
    result = evaluate(PassageHold(), alongside=True, hybrid_progress=.9,
                      legacy_target_hold=True, **change)
    assert result.unsafe and not result.waiting and not result.locked


def test_other_target_lane_or_backwards_clock_restarts_confirmation():
    h = PassageHold()
    evaluate(h)
    assert evaluate(h, 11., key=('d3', 2)).waiting
    assert evaluate(h, 12., key=('d3', 0)).waiting
    assert evaluate(h, 9., key=('d3', 0)).waiting
    assert evaluate(h, 9.4, key=('d3', 0)).unsafe


def test_unrelated_hybrid_progress_does_not_lock():
    h = PassageHold()
    evaluate(h, hybrid_matches=False, hybrid_progress=1.)
    assert evaluate(h, 10.4, hybrid_matches=False, hybrid_progress=1.).unsafe


def conflict(lon, ego=5., other=0., **changes):
    args = dict(longitudinal=lon, ego_speed=ego, other_speed=other,
                body_length_sum=2., desired_gap=2., reaction_sec=.5,
                available_deceleration=2.)
    args.update(changes)
    return dynamic_longitudinal_conflict_unsafe(**args)


@pytest.mark.parametrize('lon,ego,other,unsafe', [
    (20., 5., 0., False), (10., 5., 0., True),
    (-20., 5., 15., True), (-20., 5., 5., False),
    (4., 5., 6., False), (-4., 5., 4., False),
    (1., 5., 6., True), (-1., 5., 4., True),
    (30., 5., -10., True), (12.75, 5., 0., True),
    (12.751, 5., 0., False),
])
def test_signed_relative_braking_gap(lon, ego, other, unsafe):
    assert conflict(lon, ego, other) is unsafe


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
@pytest.mark.parametrize('field', ['longitudinal', 'ego_speed', 'other_speed'])
def test_unknown_motion_cannot_establish_safety(field, value):
    args = dict(longitudinal=-20., ego_speed=5., other_speed=0.)
    args[field] = value
    assert dynamic_longitudinal_conflict_unsafe(**args, body_length_sum=2.,
        desired_gap=2., reaction_sec=.5, available_deceleration=2.)


def controller():
    c = SimpleNamespace(_v2x_tracker=V2XVehicleTracker(50., 50.),
        _lane_index_for_position=lambda x, y: 2 if y >= 1. else 1,
        _center_longitudinal_between=lambda ex, ey, x, y: x-ex,
        _reference_pathN_center=SimpleNamespace(get_waypoint=lambda wp: SimpleNamespace(psi=0.)),
        _carN_center=SimpleNamespace(get_closest_waypoint=lambda x,y: 0, wp_id=0),
        _parallel_ego_half_length=1., _parallel_vehicle_half_length=1.,
        _parallel_safety_longitudinal_clearance=.5,
        _moving_emergency_desired_distance=2., _moving_emergency_reaction_sec=.5,
        _moving_emergency_available_deceleration=2., _overtake=OvertakeSession(),
        _prepass_lane_fallback_prediction_sec=1.,
        _lane_horizon_has_vehicle_width=lambda lane: True,
        _hybrid_passage_loss_confirm_sec=.4, _hybrid_side_lock_progress_ratio=.35,
        get_logger=lambda: SimpleNamespace(info=lambda *a, **kw: None))
    for name in ('_committed_lane_traffic_evidence', '_committed_target_body_overlap',
                 '_evaluate_committed_lane_hold'):
        setattr(c, name, MethodType(controller_method(name), c))
    from multi_purpose_mpc_ros.collision_geometry import BodyGeometry
    c._collision_geometry=BodyGeometry(2.,1.5)
    c._collision_ego_origin=c._collision_v2x_origin='center'
    c._oriented_vehicle_rectangles_overlap = controller_method('_oriented_vehicle_rectangles_overlap')
    return c


POSE = SimpleNamespace(x=0., y=2., theta=0.)
EMPTY = dict(front=(), side=(), rear=())


def evidence(c, conflicts=EMPTY):
    return c._committed_lane_traffic_evidence(POSE, 5., 2, conflicts, 'target')


@pytest.mark.parametrize('mode', ['first_sample', 'missing_position', 'missing_distance', 'nan_speed'])
def test_actual_evidence_marks_rear_unknown(mode):
    c = controller()
    c._v2x_tracker.update(_msg(0., [('rear', -21., 2.)]))
    if mode != 'first_sample':
        c._v2x_tracker.update(_msg(.1, [('rear', -20., 2.)]))
    if mode == 'missing_position':
        evidence(c)  # Retain relevance before the localized rear disappears.
        c._v2x_tracker._samples['rear'].clear()
    if mode == 'missing_distance':
        c._center_longitudinal_between = lambda *a: None
    if mode == 'nan_speed':
        c._v2x_tracker._velocities['rear'] = (float('nan'), 0.)
    assert evidence(c) == ((), ('rear',))


def test_actual_evidence_checks_fast_rear_outside_fixed_window():
    c = controller()
    c._v2x_tracker.update(_msg(0., [('rear', -22., 2.)]))
    c._v2x_tracker.update(_msg(.1, [('rear', -20., 2.)]))
    assert evidence(c) == (('rear',), ())
    c._v2x_tracker.update(_msg(.2, [('rear', -20., 2.)]))
    # The retained velocity must still protect against rear approach.
    assert evidence(c) == (('rear',), ())


def test_actual_evidence_ignores_known_other_lane_but_not_predicted_cut_in():
    c = controller()
    c._v2x_tracker.update(_msg(0., [('other', -20., 0.)]))
    assert evidence(c) == ((), ())
    assert evidence(c, dict(front=(), rear=(), side=('other',))) == (('other',), ())


@pytest.mark.parametrize('x', [-1., 0., 1.])
def test_actual_overlap_handles_alongside_on_both_sides_of_zero(x):
    c = controller()
    c._v2x_tracker.update(_msg(0., [('target', x, 0.)]))
    c._v2x_tracker.update(_msg(.1, [('target', x, 0.)]))
    c._v2x_tracker.set_measured_body_pose('target',x,0.,0.,.1)
    assert c._committed_target_body_overlap(POSE, 'target') is False
    assert c._committed_target_body_overlap(SimpleNamespace(x=0., y=0., theta=0.), 'target') is True


def test_actual_lane_hold_width_and_unknown_rear_override_alongside_and_l0_exception():
    c = controller()
    c._v2x_tracker.update(_msg(0., [('target', 0., 0.)]))
    c._v2x_tracker.update(_msg(.1, [('target', 0., 0.)]))
    c._v2x_tracker.set_measured_body_pose('target',0.,0.,0.,.1)
    args = dict(pose=POSE, ego_speed=5., lane_idx=2, target_id='target',
                target_longitudinal=0., live_passage={2: False}, conflicts=EMPTY,
                now_sec=10., legacy_target_hold=True)
    assert c._evaluate_committed_lane_hold(**args).locked
    c._lane_horizon_has_vehicle_width = lambda lane: False
    assert c._evaluate_committed_lane_hold(**args).unsafe
    c._lane_horizon_has_vehicle_width = lambda lane: True
    c._v2x_tracker.update(_msg(.2, [('target', 0., 0.), ('rear', -20., 2.)]))
    assert c._evaluate_committed_lane_hold(**args).unsafe


@pytest.mark.parametrize('method', ['clear_proof', 'clear_hybrid', 'release_target', 'complete_pass'])
def test_session_lifecycle_clears_confirmation(method):
    session = OvertakeSession()
    evaluate(session.passage_hold)
    getattr(session, method)()
    assert session.passage_hold.since is None


def test_duplicate_observation_cannot_cancel_verified_pass_or_zero_target_speed():
    c = controller()
    c._v2x_tracker.update(_msg(0., [('target', 9.6, 0.)]))
    c._v2x_tracker.update(_msg(.1, [('target', 10., 0.)]))
    c._overtake.target_id = 'target'
    c._overtake.requested_lane = 2
    c._overtake.committed = True
    c._overtake.verification.vehicle_id = 'target'
    c._overtake.verification.lane_idx = 2
    c._v2x_tracker.update(_msg(.1, [('target', 10., 0.)]))
    args = dict(pose=POSE, ego_speed=5., lane_idx=2, target_id='target',
                target_longitudinal=10., live_passage={2:True}, conflicts=EMPTY, now_sec=10.)
    assert not c._evaluate_committed_lane_hold(**args).unsafe
    assert c._v2x_tracker.velocity('target')[0] == pytest.approx(4.)
    assert c._v2x_tracker.has_velocity_estimate('target')


@pytest.mark.parametrize('x,y,overlap', [(10.,0.,False), (0.,4.1,False), (0.,4.,True), (0.,2.,True), (1.,2.,True)])
def test_unknown_target_velocity_uses_position_and_conservative_body_bound(x,y,overlap):
    c=controller()
    c._v2x_tracker.update(_msg(0., [('target',x,y)]))
    assert not c._v2x_tracker.has_velocity_estimate('target')
    assert c._committed_target_body_overlap(POSE,'target') is overlap


@pytest.mark.parametrize('known_lane', [None, 2])
def test_distant_unknown_speed_cannot_release_lane(known_lane):
    c=controller()
    c._lane_index_for_position=lambda *a: known_lane
    c._v2x_tracker.update(_msg(0., [('remote',2000.,10.)]))
    assert evidence(c)==((),())


def test_remote_missing_position_cannot_create_relevance_but_identified_conflict_can():
    c=controller()
    c._v2x_tracker.update(_msg(0., [('remote',2000.,10.)]))
    assert evidence(c)==((),())
    c._v2x_tracker._samples['remote'].clear()
    assert evidence(c)==((),())
    assert evidence(c,dict(front=(),side=(),rear=('remote',)))==((),('remote',))


def test_nearby_unknown_lane_remains_conservative_with_diagnostic_reason():
    c=controller()
    c._lane_index_for_position=lambda *a: None
    c._v2x_tracker.update(_msg(0., [('car',-10.,2.)]))
    assert evidence(c)==((),('car',))
    assert c._committed_lane_unknown_reasons == {'car':'nearby_lane_unknown'}


def test_missing_previous_rear_remains_unsafe_until_located_elsewhere():
    c=controller()
    c._v2x_tracker.update(_msg(0., [('rear',-10.,2.)]))
    assert evidence(c)==((),('rear',))
    c._v2x_tracker._samples['rear'].clear()
    assert evidence(c)==((),('rear',))
    c._v2x_tracker.update(_msg(.2,[('rear',-10.,0.)]))
    assert evidence(c)==((),())
    c._v2x_tracker._samples['rear'].clear()
    assert evidence(c)==((),())


def test_explicit_predicted_cut_in_is_not_removed_by_distance_prefilter():
    c=controller()
    c._v2x_tracker.update(_msg(0., [('car',2000.,0.)]))
    assert evidence(c,dict(front=(),side=('car',),rear=())) == (('car',),())


@pytest.mark.parametrize('missing',[dict(target_geometry_known=False),dict(body_overlap=None)])
def test_transient_unknown_stops_motion_without_discarding_lane(missing):
    h=PassageHold()
    evaluate(h,9.,target_clearance_lost=False)
    result=evaluate(h,10.,**missing)
    assert result.waiting and result.motion_blocked and not result.unsafe
    assert not evaluate(h,10.2,target_clearance_lost=False).motion_blocked
    assert evaluate(h,11.,**missing).motion_blocked
    assert evaluate(h,11.4,**missing).unsafe
    # An actual road-width loss never waits for unknown-observation recovery.
    assert evaluate(h,12.,lane_width_valid=False,**missing).unsafe
