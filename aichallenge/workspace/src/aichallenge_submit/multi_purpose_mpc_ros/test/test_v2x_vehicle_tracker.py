"""Unit tests for V2XVehicleTracker (pure Python, no rclpy)."""

from dataclasses import dataclass
from typing import List

import pytest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    V2XVehicleTracker,
    follow_emergency_reacquire_blocked,
    l0_restricted_follow_can_ignore_passage,
    outer_lane_problem_slow_override_active,
    outer_prediction_bypass_target_matches,
    overtake_shadow_solution_acceptable,
    prepass_recovery_timeout_expired,
    resolve_applied_corridor,
    select_l2_restricted_zone_lane,
    strict_shadow_slow_commit_creep_allowed,
)


def test_l0_restricted_follow_ignores_target_only_passage_loss():
    assert l0_restricted_follow_can_ignore_passage(
        restriction_active=True,
        lane_has_vehicle_width=True,
        non_target_conflicts={"front": [], "side": [], "rear": ["d3"]},
    )


@pytest.mark.parametrize("conflict_group", ["front", "side"])
def test_l0_restricted_follow_rejects_unrelated_collision(conflict_group):
    conflicts = {"front": [], "side": [], "rear": []}
    conflicts[conflict_group] = ["d3"]
    assert not l0_restricted_follow_can_ignore_passage(
        restriction_active=True,
        lane_has_vehicle_width=True,
        non_target_conflicts=conflicts,
    )


def test_l0_restricted_follow_rejects_lane_width_loss():
    assert not l0_restricted_follow_can_ignore_passage(
        restriction_active=True,
        lane_has_vehicle_width=False,
        non_target_conflicts={"front": [], "side": [], "rear": []},
    )


def test_strict_shadow_stopped_commit_creep_accepts_exact_safe_commit():
    assert strict_shadow_slow_commit_creep_allowed(
        target_matches=True,
        shadow_verified=True,
        committed_outer_lane=True,
        target_is_slow=True,
        current_envelopes_separated=True,
        candidate_passable=True,
        candidate_conflicts={"front": [], "side": [], "rear": []},
    )


@pytest.mark.parametrize(
    "failed_gate",
    [
        "target_matches",
        "shadow_verified",
        "committed_outer_lane",
        "target_is_slow",
        "current_envelopes_separated",
        "candidate_passable",
    ],
)
def test_strict_shadow_stopped_commit_creep_rejects_failed_gate(failed_gate):
    gates = {
        "target_matches": True,
        "shadow_verified": True,
        "committed_outer_lane": True,
        "target_is_slow": True,
        "current_envelopes_separated": True,
        "candidate_passable": True,
    }
    gates[failed_gate] = False
    assert not strict_shadow_slow_commit_creep_allowed(
        **gates,
        candidate_conflicts={"front": [], "side": [], "rear": []},
    )


@pytest.mark.parametrize("group", ["front", "side"])
def test_strict_shadow_stopped_commit_creep_rejects_live_conflict(group):
    conflicts = {"front": [], "side": [], "rear": []}
    conflicts[group] = ["d3"]
    assert not strict_shadow_slow_commit_creep_allowed(
        target_matches=True,
        shadow_verified=True,
        committed_outer_lane=True,
        target_is_slow=True,
        current_envelopes_separated=True,
        candidate_passable=True,
        candidate_conflicts=conflicts,
    )


def test_strict_shadow_stopped_commit_creep_ignores_rear_conflict():
    assert strict_shadow_slow_commit_creep_allowed(
        target_matches=True,
        shadow_verified=True,
        committed_outer_lane=True,
        target_is_slow=True,
        current_envelopes_separated=True,
        candidate_passable=True,
        candidate_conflicts={"front": [], "side": [], "rear": ["d3"]},
    )


def test_problem_zone_slow_override_rejects_missing_target():
    assert not outer_lane_problem_slow_override_active(
        latched_target_id=None,
        opponent_vehicle_id=None,
        velocity_valid=False,
    )
    assert not outer_lane_problem_slow_override_active(
        latched_target_id=None,
        opponent_vehicle_id="d2",
        velocity_valid=True,
    )
    assert not outer_lane_problem_slow_override_active(
        latched_target_id="d2",
        opponent_vehicle_id="d2",
        velocity_valid=False,
    )
    assert outer_lane_problem_slow_override_active(
        latched_target_id="d2",
        opponent_vehicle_id="d2",
        velocity_valid=True,
    )


def test_l2_restricted_zone_prefers_l0_and_ignores_rear_vehicle():
    assert select_l2_restricted_zone_lane(
        2,
        restriction_active=True,
        slow_lead_override=False,
        l0_physically_passable=True,
        l0_conflicts={"front": [], "side": [], "rear": ["d3"]},
    ) == 0


def test_l2_restricted_zone_follows_l0_front_vehicle():
    assert select_l2_restricted_zone_lane(
        2,
        restriction_active=True,
        slow_lead_override=False,
        l0_physically_passable=False,
        l0_conflicts={"front": ["d2"], "side": [], "rear": ["d3"]},
    ) == 0


def test_l2_restricted_zone_uses_l1_when_l0_side_is_unsafe():
    assert select_l2_restricted_zone_lane(
        2,
        restriction_active=True,
        slow_lead_override=False,
        l0_physically_passable=True,
        l0_conflicts={"front": [], "side": ["d3"], "rear": []},
    ) == 1


def test_l2_restricted_zone_slow_override_preserves_candidate():
    assert select_l2_restricted_zone_lane(
        2,
        restriction_active=True,
        slow_lead_override=True,
        l0_physically_passable=False,
        l0_conflicts={"front": [], "side": ["d3"], "rear": []},
    ) == 2


# Lightweight stand-ins for v2x_msgs / std_msgs / geometry_msgs so tests
# do not require the ROS message DLLs to be importable.
@dataclass
class _Stamp:
    sec: int
    nanosec: int


@dataclass
class _Header:
    stamp: _Stamp


@dataclass
class _Point:
    x: float
    y: float
    z: float = 0.0


@dataclass
class _V2XVehiclePosition:
    header: _Header
    vehicle_id: str
    position: _Point


@dataclass
class _V2XVehiclePositionArray:
    header: _Header
    vehicles: List[_V2XVehiclePosition]


def _msg(stamp_sec: float, vehicles):
    """Build a fake V2XVehiclePositionArray with the given (vehicle_id, x, y)."""
    sec = int(stamp_sec)
    nanosec = int((stamp_sec - sec) * 1e9)
    array_header = _Header(_Stamp(sec, nanosec))
    out = []
    for vid, x, y in vehicles:
        out.append(_V2XVehiclePosition(
            header=_Header(_Stamp(sec, nanosec)),
            vehicle_id=vid,
            position=_Point(x=x, y=y),
        ))
    return _V2XVehiclePositionArray(header=array_header, vehicles=out)


def test_two_samples_constant_velocity_yields_finite_difference():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)

    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 2.5)]))

    vx, vy = tracker.velocity("d2")
    assert vx == pytest.approx(10.0)
    assert vy == pytest.approx(5.0)


def test_single_sample_yields_zero_velocity():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)

    tracker.update(_msg(0.0, [("d2", 1.0, 2.0)]))

    assert tracker.velocity("d2") == (0.0, 0.0)
    assert tracker.has_velocity_estimate("d2") is False


def test_unknown_vehicle_velocity_is_zero():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)
    assert tracker.velocity("d9") == (0.0, 0.0)


def test_predict_positions_constant_velocity():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 2.5)]))  # vx=10, vy=5, latest (5,2.5)

    points = tracker.predict_positions("d2", [0.0, 0.5, 1.0])

    assert points[0] == pytest.approx((5.0, 2.5))
    assert points[1] == pytest.approx((10.0, 5.0))
    assert points[2] == pytest.approx((15.0, 7.5))


def test_position_jump_resets_velocity_and_drops_old_sample():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.1, [("d2", 100.0, 0.0)]))  # 100 m jump > 5 m

    assert tracker.velocity("d2") == (0.0, 0.0)
    # Predictions should anchor at the *new* position with zero velocity.
    points = tracker.predict_positions("d2", [0.0, 0.5])
    assert points[0] == pytest.approx((100.0, 0.0))
    assert points[1] == pytest.approx((100.0, 0.0))


def test_velocity_above_safety_cap_is_zeroed():
    # 50 m / 0.05 s = 1000 m/s, well above v_max_safety=30
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=200.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.05, [("d2", 50.0, 0.0)]))

    assert tracker.velocity("d2") == (0.0, 0.0)


def test_two_vehicles_tracked_independently():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0), ("d3", 10.0, 10.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 0.0), ("d3", 10.0, 11.1)]))

    assert tracker.velocity("d2") == pytest.approx((10.0, 0.0))
    assert tracker.velocity("d3") == pytest.approx((0.0, 5.0))
    assert tracker.has_velocity_estimate("d2") is True
    assert tracker.has_velocity_estimate("d3") is True


def test_active_ids_reflect_latest_message_only():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0), ("d3", 10.0, 10.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 0.0)]))  # d3 dropped this tick

    assert tracker.active_vehicle_ids() == ["d2"]
    # d3 is still in the internal state but not reported as active.
    assert tracker.velocity("d3") == pytest.approx((0.0, 0.0))


def test_predict_all_returns_only_active_vehicles():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0), ("d3", 10.0, 10.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 0.0)]))  # d3 dropped

    out = tracker.predict_all([0.0, 1.0])
    assert set(out.keys()) == {"d2"}
    assert out["d2"][0] == pytest.approx((5.0, 0.0))
    assert out["d2"][1] == pytest.approx((15.0, 0.0))


@dataclass
class _StubObstacle:
    cx: float
    cy: float
    radius: float


def test_predictions_to_obstacles_flattens_with_radius():
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles

    predictions = {
        "d2": [(1.0, 2.0), (3.0, 4.0)],
        "d3": [(5.0, 6.0)],
    }
    obstacles = predictions_to_obstacles(
        predictions, vehicle_radius=0.5, obstacle_cls=_StubObstacle)

    centers = sorted((o.cx, o.cy, o.radius) for o in obstacles)
    assert centers == sorted([
        (1.0, 2.0, 0.5),
        (3.0, 4.0, 0.5),
        (5.0, 6.0, 0.5),
    ])


def test_predictions_to_obstacles_empty_input():
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles
    assert predictions_to_obstacles(
        {}, vehicle_radius=0.5, obstacle_cls=_StubObstacle) == []


def test_position_jump_invokes_warn_callback():
    msgs = []
    tracker = V2XVehicleTracker(
        v_max_safety=30.0,
        position_jump_threshold=5.0,
        warn_callback=msgs.append,
    )
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.1, [("d2", 100.0, 0.0)]))

    assert any("position jump" in m for m in msgs)
    assert any("d2" in m for m in msgs)


def test_velocity_cap_invokes_warn_callback():
    msgs = []
    tracker = V2XVehicleTracker(
        v_max_safety=30.0,
        position_jump_threshold=200.0,
        warn_callback=msgs.append,
    )
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.05, [("d2", 50.0, 0.0)]))

    assert any("velocity" in m for m in msgs)
    assert any("d2" in m for m in msgs)


def test_warn_callback_optional_default_is_silent():
    # Construct without a callback; clamp fires must not raise.
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.1, [("d2", 100.0, 0.0)]))  # would warn if a callback existed

    assert tracker.velocity("d2") == (0.0, 0.0)  # clamp still fires


def test_snapshot_is_detached_from_later_callback_updates():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    snapshot = tracker.snapshot()
    snapshot_generation = snapshot.generation

    tracker.update(_msg(1.0, [("d2", 5.0, 0.0), ("d3", 8.0, 1.0)]))

    assert snapshot.active_vehicle_ids() == ["d2"]
    assert snapshot.predict_positions("d2", [0.0])[0] == (0.0, 0.0)
    assert snapshot.predict_positions("d3", [0.0]) == []
    assert snapshot.generation == snapshot_generation
    assert tracker.generation == snapshot_generation + 1


@pytest.mark.parametrize(
    "requested,prohibited,recovery,transition,expected",
    [
        (0, True, True, True, (None, "full_width_recovery")),
        (2, True, True, True, (None, "full_width_recovery")),
        (0, False, True, True, (None, "full_width_recovery")),
        (0, False, False, True, (None, "lane_transition")),
        (0, False, False, False, (0, "requested_lane")),
    ],
)
def test_applied_corridor_has_one_explicit_owner(
    requested, prohibited, recovery, transition, expected,
):
    assert resolve_applied_corridor(
        requested_lane=requested,
        l0_prohibited=prohibited,
        full_width_recovery=recovery,
        transition_active=transition,
    ) == expected


def test_soft_guidance_has_absolute_watchdog_when_prediction_is_safe():
    assert prepass_recovery_timeout_expired(
        elapsed=6.0, recovery_timeout=5.0,
        soft_guidance_timeout=10.0, full_width_prediction_safe=True,
    ) == (False, False)
    assert prepass_recovery_timeout_expired(
        elapsed=10.0, recovery_timeout=5.0,
        soft_guidance_timeout=10.0, full_width_prediction_safe=True,
    ) == (False, True)


def test_follow_emergency_reacquire_hysteresis_only_blocks_same_target():
    assert follow_emergency_reacquire_blocked(
        vehicle_id="d2", released_vehicle_id="d2", released_at=10.0,
        now_sec=10.5, hysteresis_sec=1.0) is True
    assert follow_emergency_reacquire_blocked(
        vehicle_id="d3", released_vehicle_id="d2", released_at=10.0,
        now_sec=10.5, hysteresis_sec=1.0) is False
    assert follow_emergency_reacquire_blocked(
        vehicle_id="d2", released_vehicle_id="d2", released_at=10.0,
        now_sec=11.0, hysteresis_sec=1.0) is False


def _valid_shadow_kwargs():
    return dict(
        accurate=True,
        used_prediction_fallback=False,
        time_budget_exceeded=False,
        recovery_requested=False,
        infeasibility_counter=0,
        has_prediction=True,
        constraint_collapsed=False,
        lane_relaxation=0.2,
        max_lane_relaxation=0.2,
        forward_width_valid=True,
    )


def test_shadow_accepts_exact_solution_at_relaxation_limit():
    assert overtake_shadow_solution_acceptable(**_valid_shadow_kwargs())


def test_outer_prediction_bypass_accepts_latched_target():
    assert outer_prediction_bypass_target_matches(
        vehicle_id="d2", latched_target_id="d2",
        handoff_target_id=None, handoff_lane_idx=None,
        verified_outer_lane=0,
    )


def test_outer_prediction_bypass_accepts_same_lane_handoff_target():
    assert outer_prediction_bypass_target_matches(
        vehicle_id="d3", latched_target_id="d2",
        handoff_target_id="d3", handoff_lane_idx=0,
        verified_outer_lane=0,
    )


def test_outer_prediction_bypass_rejects_unverified_handoff_lane():
    assert not outer_prediction_bypass_target_matches(
        vehicle_id="d3", latched_target_id="d2",
        handoff_target_id="d3", handoff_lane_idx=0,
        verified_outer_lane=2,
    )


def test_outer_prediction_bypass_rejects_unrelated_vehicle():
    assert not outer_prediction_bypass_target_matches(
        vehicle_id="d4", latched_target_id="d2",
        handoff_target_id="d3", handoff_lane_idx=0,
        verified_outer_lane=0,
    )


@pytest.mark.parametrize("override", [
    {"accurate": False},
    {"used_prediction_fallback": True},
    {"time_budget_exceeded": True},
    {"constraint_collapsed": True},
    {"lane_relaxation": 0.2001},
    {"forward_width_valid": False},
])
def test_shadow_rejects_any_failed_commit_gate(override):
    values = _valid_shadow_kwargs()
    values.update(override)
    assert not overtake_shadow_solution_acceptable(**values)
