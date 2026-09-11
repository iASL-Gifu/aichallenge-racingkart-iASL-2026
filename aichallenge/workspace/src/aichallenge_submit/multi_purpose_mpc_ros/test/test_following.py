"""Launch regression plus braking, stale observations and target handoff tests."""
import ast
from collections import deque
from functools import lru_cache
import math
from pathlib import Path
import textwrap
from types import SimpleNamespace, MethodType
from unittest.mock import Mock

import pytest

from multi_purpose_mpc_ros.core.following import following_state
from multi_purpose_mpc_ros.core.final_emergency_limit import enforce_emergency_limit
from multi_purpose_mpc_ros.v2x_vehicle_tracker import startup_follow_restart_gap


SOURCE = Path(__file__).parents[1] / 'multi_purpose_mpc_ros/mpc_controller.py'


def state(gap=1.5, lead=.56, ego=0., **overrides):
    args = dict(body_gap=gap, lead_speed=lead, ego_speed=ego, minimum_gap=1.,
                desired_gap=3., reaction_sec=.4, deceleration=2.5, spacing_kp=.5)
    args.update(overrides)
    return following_state(**args)


@pytest.mark.parametrize('gap,lead,ego', [
    (1.5, .56, 0.), (1.53, .85, .3),
    (1.85, 1.29, .8), (1.99, 1.61, 1.6), (2.95, 2.28, 1.5),
])
def test_recorded_opening_launch_gaps_do_not_produce_stop(gap, lead, ego):
    result = state(gap, lead, ego)
    assert 0 < result.target_speed <= result.speed_limit


@pytest.mark.parametrize('lead', [0., .56, 2., 5., 9.])
@pytest.mark.parametrize('ego', [0., 1., 5., 10.])
@pytest.mark.parametrize('gap', [1.01, 1.5, 3., 10.])
def test_cap_covers_reaction_and_equal_deceleration_stopping_distances(lead, ego, gap):
    cap = state(gap, lead, ego).speed_limit
    if cap > 0:
        ego_travel = max(ego, cap) * .4 + .5 * 2.5 * .4 ** 2 + (cap + 2.5 * .4) ** 2 / 5.
        lead_travel = lead ** 2 / 5.
        assert ego_travel - lead_travel <= gap - 1. + 1e-10


def test_closing_fast_and_lead_braking_reduce_cap():
    assert state(1.5, .5, 5).target_speed == 0
    assert state(2., 0., 3.).target_speed == 0
    assert state(3., .5, 2).target_speed < state(3., 2., 2).target_speed


@pytest.mark.parametrize('gap', [-2., 0., .99, 1.])
def test_overlap_or_missing_minimum_reserve_stops(gap):
    assert state(gap, 3).target_speed == 0


@pytest.mark.parametrize('key', ['body_gap', 'lead_speed', 'ego_speed'])
def test_nonfinite_observation_cannot_accelerate(key):
    assert state(**{key: math.nan}).target_speed == 0


def controller():
    c = SimpleNamespace(
        _follow_lateral_distance=1.2, _prepass_fallback_follow_active=False,
        _follow_only=False, _follow_engage_distance=35.,
        _follow_spacing_kp=.5, _follow_desired_distance=6.,
        _follow_target_lost_max_speed=0., _follow_restart_ego_stopped_speed=.3,
        _grounded_start_boost_eligible=False, _has_moved_once=False,
        _follow_restart_min_gap=3., _follow_restart_start_min_gap=1.,
        _follow_minimum_body_gap=1., _stopped_lead_speed_threshold=.3,
        _follow_stopped_vehicle_id=None, _follow_restart_vehicle_id=None,
        _follow_restart_until=0., _follow_restart_lead_moving_speed=.5,
        _follow_restart_duration=1., _follow_restart_max_speed=3.,
        _follow_restart_speed_margin=1.5, _enable_control=True, KP=2.,
        _mpc_cfg=SimpleNamespace(a_min=-2.5, a_max=2.5),
        get_logger=Mock(return_value=Mock()))
    return c


def run_acc(c, observation, *, lead=.56, ego=0., target='d3', now=10., ceiling=9., stale=False):
    # Execute the actual ACC/restart block, including the ordering of its caps.
    source = SOURCE.read_text()
    block = source.split('            # --- Standard Follow (Same Lane) ---\n')[1]
    block = block.split('            # --- Emergency Proximity Brake')[0]
    c._moving_following_state = Mock(return_value=observation)
    scope = dict(self=c, pose=None, v=ego, current_time_sec=now,
                 lat_dist=0., opponent_ahead=object(), latched_follow_state=(
                     {'stale': True} if stale else None),
                 forced_overtake_active=False, acc_vehicle_id=target,
                 slow_pass_release_ids=set(), initial_start_boost_active=False,
                 follow_target_expired=False, acc_distance=4.61,
                 acc_lead_speed=lead, acc_velocity_valid=True,
                 ref_vel_kmph=ceiling, hybrid_escape_speeds={},
                 shared_follow_state=None, follow_restart_active=False,
                 startup_follow_restart_gap=startup_follow_restart_gap)
    exec(compile(textwrap.dedent(block), str(SOURCE), 'exec'), scope)
    return scope


def emergency_follow_target(observation):
    tree = ast.parse(SOURCE.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'shared_follow_limit is not None'
                  and any(isinstance(a, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'v_ref_emg'
                                  for t in a.targets) for a in n.body))
    scope = dict(shared_follow_limit=observation)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['v_ref_emg']


def test_restart_and_emergency_share_target_through_launch():
    c = controller()
    run_acc(c, None, lead=0.)
    assert c._follow_stopped_vehicle_id == 'd3'
    observation = state()
    start = run_acc(c, observation, now=10.1)
    assert start['follow_restart_active']
    assert start['follow_restart_target'] == pytest.approx(observation.target_speed)
    assert emergency_follow_target(observation) == start['ref_vel_kmph']
    c._has_moved_once = True
    moving = state(1.99, 1.61, 1.6)
    later = run_acc(c, moving, lead=1.61, ego=1.6, now=11.5)
    assert not later['follow_restart_active']
    assert later['ref_vel_kmph'] == emergency_follow_target(moving) > 0


def test_restart_cannot_override_prior_stop_or_transfer_to_different_vehicle():
    c = controller()
    run_acc(c, None, lead=0.)
    result = run_acc(c, state(), ceiling=0.)
    assert result['ref_vel_kmph'] == 0
    changed = run_acc(c, state(), target='d2', now=10.1)
    assert not changed['follow_restart_active']
    assert enforce_emergency_limit(
        state().target_speed, 2., True, limit=0., measured_speed=.3,
        kp=2., a_min=-2.5, a_max=2.5, active=True) == (0., -.6, False)


def test_stale_target_cannot_use_restart_even_with_cached_moving_speed():
    c = controller()
    run_acc(c, None, lead=0.)
    result = run_acc(c, state(), stale=True)
    assert not result['follow_restart_active']
    assert result['ref_vel_kmph'] == 0
    c._moving_following_state.assert_not_called()


def observation_controller():
    # Execute the production snapshot method without loading ROS.
    tree = ast.parse(SOURCE.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_moving_following_state')
    scope = {'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), scope)
    c = controller()
    c._follow_observation_max_age = .5
    c._moving_emergency_desired_distance = 3.
    c._moving_emergency_reaction_sec = .4
    c._moving_emergency_available_deceleration = 2.5
    c._moving_emergency_spacing_kp = .5
    c._v2x_tracker = SimpleNamespace(
        _samples={'d3': [(9.9, 0., 4.6)]},
        has_velocity_estimate=lambda _: True, velocity=lambda _: (0., .56))
    c._carN_center = SimpleNamespace(get_closest_waypoint=lambda x, y: 2)
    c._reference_pathN_center = SimpleNamespace(
        get_waypoint=lambda _: SimpleNamespace(psi=math.pi / 2))
    c._current_center_envelopes_are_separated = Mock(return_value=(
        True, {'arc_gap': 1.5, 'rectangles_overlap': False}))
    c.observe = MethodType(scope['_moving_following_state'], c)
    return c


def test_snapshot_uses_body_gap_and_signed_center_velocity():
    c = observation_controller()
    assert c.observe(None, 'd3', 0., 10.) == state()
    c._v2x_tracker.velocity = lambda _: (0., -.56)
    assert c.observe(None, 'd3', 0., 10.) is None
    c._v2x_tracker.velocity = lambda _: (2., .56)
    assert c.observe(None, 'd3', 0., 10.) is None
    c._v2x_tracker.velocity = lambda _: (0., .28)
    assert c.observe(None, 'd3', 0., 10.) is None


@pytest.mark.parametrize('stamp', [9., 10.1, math.nan])
def test_snapshot_rejects_stale_future_and_invalid_time(stamp):
    c = observation_controller()
    c._v2x_tracker._samples['d3'] = [(stamp, 0., 4.6)]
    assert c.observe(None, 'd3', 0., 10.) is None


@pytest.mark.parametrize('envelope', [None, {'arc_gap': -1., 'rectangles_overlap': True}])
def test_snapshot_does_not_relax_overlap_or_unknown_geometry(envelope):
    c = observation_controller()
    c._current_center_envelopes_are_separated.return_value = (False, envelope)
    assert c.observe(None, 'd3', 0., 10.) is None


@lru_cache(maxsize=1)
def final_follow_guard():
    tree = ast.parse(SOURCE.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == (
                      'shared_follow_state is not None and (not recovering_from_stuck) and self._enable_control'))
    return compile(ast.Module(body=[branch], type_ignores=[]), str(SOURCE), 'exec')


def apply_final_guard(c, observation, ego, acc, speed=None):
    scope = dict(self=c, shared_follow_state=observation, recovering_from_stuck=False,
                 u=[observation.target_speed if speed is None else speed, 0.], acc=acc,
                 bug_acc_enabled=True, ref_vel_kmph=9., v=ego,
                 acc_vehicle_id='d3', follow_restart_active=True)
    exec(final_follow_guard(), scope)
    return scope


def test_final_follow_guard_caps_fallback_and_demands_braking_when_needed():
    c = controller()
    result = apply_final_guard(c, state(1.5, .5, 5.), ego=5., acc=2.5, speed=7.5)
    assert result['u'][0] == 0
    assert result['acc'] == -2.5
    assert not result['bug_acc_enabled']
    result = apply_final_guard(c, state(), ego=0., acc=-2.5, speed=0.)
    assert result['u'][0] == 0
    assert result['acc'] == -2.5


@pytest.mark.parametrize('delay', [0., .2, .4])
def test_launch_then_braking_lead_with_delayed_ego_commands(delay):
    c = controller()
    dt = .025
    pending = deque([0.] * round(delay / dt))
    gap, ego, lead = 1.5, 0., 0.
    minimum_gap = gap
    peak_ego_speed = 0.
    for i in range(600):
        t = i * dt
        lead_acc = 2. if t < 2. else (-2.5 if t >= 4. and lead > 0 else 0.)
        observation = state(gap, lead, ego)
        acc = max(-2.5, min(2.5, 2. * (observation.target_speed - ego)))
        final = apply_final_guard(c, observation, ego, acc)
        pending.append(final['acc'])
        next_ego = max(0., ego + pending.popleft() * dt)
        next_lead = max(0., lead + lead_acc * dt)
        gap += (lead + next_lead - ego - next_ego) * dt / 2
        ego, lead = next_ego, next_lead
        peak_ego_speed = max(peak_ego_speed, ego)
        minimum_gap = min(minimum_gap, gap)
    assert peak_ego_speed > 2.5  # The model must actually launch, not just stop safely.
    assert minimum_gap >= 1.
    assert ego < .01
