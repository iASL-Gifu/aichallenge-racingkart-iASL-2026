"""Unit tests for V2XVehicleTracker (pure Python, no rclpy)."""

from dataclasses import dataclass
from typing import List

import pytest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    V2XVehicleTracker,
    l0_prohibited_l1_hard_defer_reasons,
    resolve_applied_corridor,
    should_defer_l0_prohibited_l1_hard,
    should_defer_l0_prohibited_physical_l2_l1_hard,
)


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
        (0, True, True, True, (1, "l0_prohibited")),
        (2, True, True, True, (2, "l0_prohibited")),
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


def test_l0_prohibited_physical_l1_keeps_hard_l1_when_v2x_clear():
    assert not should_defer_l0_prohibited_l1_hard(
        l0_prohibited=True,
        physical_lane_idx=1,
        l1_conflicts={"front": [], "side": [], "rear": []},
    )


def test_l0_prohibited_physical_l1_defers_hard_l1_when_v2x_blocks():
    assert should_defer_l0_prohibited_l1_hard(
        l0_prohibited=True,
        physical_lane_idx=1,
        l1_conflicts={"front": ["d3"], "side": [], "rear": []},
    )


def test_l0_prohibited_l1_deferral_does_not_change_physical_l0_path():
    assert not should_defer_l0_prohibited_l1_hard(
        l0_prohibited=True,
        physical_lane_idx=0,
        l1_conflicts={"front": ["d3"], "side": [], "rear": []},
    )
    # The existing physical-L0 staged-recovery path suppresses the geographic
    # hard-L1 override, allowing full-width recovery to retain ownership.
    assert resolve_applied_corridor(
        requested_lane=None,
        l0_prohibited=False,
        full_width_recovery=True,
        transition_active=False,
    ) == (None, "full_width_recovery")


def _l1_entry_defer_reasons(**overrides):
    values = {
        "l0_prohibited": True,
        "physical_lane_idx": 1,
        "l1_conflicts": {"front": [], "side": [], "rear": []},
        "heading_error": 0.05,
        "max_heading_error": 0.10,
        "lateral_speed": 0.2,
        "max_lateral_speed": 0.8,
        "yaw_rate": 0.2,
        "max_yaw_rate": 1.0,
        "l1_lateral_error": 0.1,
        "max_l1_lateral_error": 0.4,
        "prediction_available": True,
        "prediction_fit": 0.9,
        "minimum_prediction_fit": 0.75,
        "l1_hard_already_owned": False,
    }
    values.update(overrides)
    return l0_prohibited_l1_hard_defer_reasons(**values)


def test_l0_prohibited_l1_hard_defers_for_unstable_yaw_rate():
    assert _l1_entry_defer_reasons(yaw_rate=1.01) == (
        "yaw_rate_unstable",)


def test_l0_prohibited_l1_hard_defers_for_low_prediction_fit():
    assert _l1_entry_defer_reasons(prediction_fit=0.74) == (
        "prediction_fit_low",)


def test_l0_prohibited_l1_hard_defers_for_l1_lateral_error():
    assert _l1_entry_defer_reasons(l1_lateral_error=0.41) == (
        "l1_lateral_error",)


def test_l0_prohibited_l1_hard_allows_fully_ready_physical_l1():
    assert _l1_entry_defer_reasons() == ()


def test_l0_prohibited_physical_l2_defers_new_l1_hard_when_not_ready():
    reasons = _l1_entry_defer_reasons(
        physical_lane_idx=2,
        yaw_rate=1.41,
        l1_lateral_error=0.76,
        prediction_available=False,
        prediction_fit=0.0,
    )
    assert reasons == (
        "yaw_rate_unstable",
        "l1_lateral_error",
        "prediction_unavailable",
    )
    # The controller suppresses the geographic L1-hard override while these
    # reasons are present, leaving the existing staged recovery as owner.
    assert resolve_applied_corridor(
        requested_lane=None,
        l0_prohibited=not bool(reasons),
        full_width_recovery=bool(reasons),
        transition_active=False,
    ) == (None, "full_width_recovery")


def test_l0_prohibited_physical_l2_allows_new_l1_hard_when_ready():
    assert _l1_entry_defer_reasons(physical_lane_idx=2) == ()


def test_l0_prohibited_physical_l2_keeps_valid_l2_request():
    assert resolve_applied_corridor(
        requested_lane=2,
        l0_prohibited=True,
        full_width_recovery=False,
        transition_active=False,
    ) == (2, "l0_prohibited")


def test_physical_l2_readiness_does_not_apply_outside_l0_prohibited_zone():
    assert _l1_entry_defer_reasons(
        l0_prohibited=False,
        physical_lane_idx=2,
        yaw_rate=1.41,
        l1_lateral_error=0.76,
        prediction_available=False,
    ) == ()


@pytest.mark.parametrize(
    "target_lane_idx,requested_lane_idx", [(1, 1), (None, None)])
def test_l0_prohibited_physical_l2_defers_unready_l1_hard_regardless_of_source(
    target_lane_idx, requested_lane_idx,
):
    physical_lane_idx = 2
    reasons = _l1_entry_defer_reasons(
        physical_lane_idx=physical_lane_idx,
        yaw_rate=1.41,
        l1_lateral_error=0.76,
        prediction_available=False,
        # A target request is not safe existing ownership from physical L2.
        l1_hard_already_owned=(
            target_lane_idx == 1 and physical_lane_idx == 1
        ),
    )
    assert should_defer_l0_prohibited_physical_l2_l1_hard(
        l0_prohibited=True,
        physical_lane_idx=physical_lane_idx,
        requested_lane_idx=requested_lane_idx,
        failed_readiness_conditions=reasons,
    )
    assert resolve_applied_corridor(
        requested_lane=None,
        l0_prohibited=False,
        full_width_recovery=True,
        transition_active=False,
    ) == (None, "full_width_recovery")


def test_l0_prohibited_physical_l2_ready_l1_request_is_allowed():
    reasons = _l1_entry_defer_reasons(physical_lane_idx=2)
    assert not should_defer_l0_prohibited_physical_l2_l1_hard(
        l0_prohibited=True,
        physical_lane_idx=2,
        requested_lane_idx=1,
        failed_readiness_conditions=reasons,
    )


def test_l0_prohibited_physical_l2_valid_l2_request_is_not_deferred():
    reasons = _l1_entry_defer_reasons(
        physical_lane_idx=2,
        yaw_rate=1.41,
        prediction_available=False,
    )
    assert not should_defer_l0_prohibited_physical_l2_l1_hard(
        l0_prohibited=True,
        physical_lane_idx=2,
        requested_lane_idx=2,
        failed_readiness_conditions=reasons,
    )


def test_physical_l1_existing_l1_hard_ownership_remains_exempt():
    assert _l1_entry_defer_reasons(
        physical_lane_idx=1,
        yaw_rate=1.41,
        prediction_available=False,
        l1_hard_already_owned=True,
    ) == ()


def test_l0_prohibited_l1_hard_reports_v2x_and_readiness_failures():
    assert _l1_entry_defer_reasons(
        l1_conflicts={"front": ["d3"], "side": [], "rear": []},
        prediction_available=False,
    ) == ("predicted_v2x_conflict", "prediction_unavailable")


def test_verified_l1_hard_ownership_is_not_reentered_after_probe_success():
    assert _l1_entry_defer_reasons(
        yaw_rate=1.2,
        prediction_available=False,
        l1_hard_already_owned=True,
    ) == ()
