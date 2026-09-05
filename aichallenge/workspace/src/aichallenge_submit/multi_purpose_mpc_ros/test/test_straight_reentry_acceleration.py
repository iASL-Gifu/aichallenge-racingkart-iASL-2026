"""Regression tests for StraightReentry longitudinal ownership."""

from types import SimpleNamespace
import inspect
import unittest

import numpy as np

from multi_purpose_mpc_ros.mpc_controller import MPCController
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    should_count_mpc_recovery_success,
)


def _controller(*, direction=1, gear=2, reverse_drive=False):
    controller = MPCController.__new__(MPCController)
    controller._straight_reentry_active = True
    controller._straight_reentry_direction = direction
    controller._straight_reentry_returning_drive = False
    controller._gear_report = SimpleNamespace(report=gear)
    controller._gear_drive_command = 2
    controller._gear_reverse_reports = {20, 21}
    controller._drive_confirmed_by_command_fallback = False
    controller._stuck_reverse_drive_active = reverse_drive
    controller._stuck_reverse_command_mode = "awsim_reverse_button"
    controller._stuck_reverse_acceleration_positive = True
    controller._stuck_reverse_acceleration = -2.5
    controller._mpc_cfg = SimpleNamespace(a_min=-2.0, a_max=2.5)
    controller._cfg = SimpleNamespace(
        bicycle_model=SimpleNamespace(width=1.6),
        mpc=SimpleNamespace(prediction_outer_boundary_guard=0.1),
    )
    return controller


class StraightReentryAccelerationTest(unittest.TestCase):
    @staticmethod
    def _recovery_success_allowed(
        *, straight_reentry_active, generic_recovery_active=False,
        post_reverse_active=False, gear_is_drive=True,
    ):
        recovery_motion_active = bool(
            generic_recovery_active or straight_reentry_active)
        recovery_gear_ready = bool(
            not post_reverse_active or gear_is_drive)
        return should_count_mpc_recovery_success(
            stuck_recovery_active=recovery_motion_active,
            gear_is_drive=recovery_gear_ready,
            infeasibility_counter=0,
            has_current_prediction=True,
            used_prediction_fallback=False,
        )

    def test_ordinary_mpc_recovery_success_remains_countable(self):
        self.assertTrue(self._recovery_success_allowed(
            straight_reentry_active=False))

    def test_reverse_reentry_blocks_mpc_recovery_success(self):
        self.assertFalse(self._recovery_success_allowed(
            straight_reentry_active=True,
            gear_is_drive=False,
        ))

    def test_drive_reentry_blocks_mpc_recovery_success(self):
        self.assertFalse(self._recovery_success_allowed(
            straight_reentry_active=True,
            gear_is_drive=True,
        ))

    def test_post_reverse_drive_recovery_remains_countable(self):
        self.assertTrue(self._recovery_success_allowed(
            straight_reentry_active=False,
            post_reverse_active=True,
            gear_is_drive=True,
        ))

    def test_control_success_gate_includes_straight_reentry_ownership(self):
        source = inspect.getsource(MPCController._control)

        self.assertIn(
            "recovery_active or self._straight_reentry_active", source)
        self.assertIn(
            "stuck_recovery_active=recovery_motion_active", source)

    def test_wall_violation_uses_vehicle_footprint_and_existing_guard(self):
        controller = _controller()
        controller._car = SimpleNamespace(
            get_closest_waypoint=lambda _x, _y: 198)
        controller._reference_path = SimpleNamespace(
            get_waypoint=lambda _wp: SimpleNamespace(
                x=0.0, y=0.0, psi=0.0),
            get_lane_bounds=lambda _wp: [(4.034, -2.624)],
        )

        violation = controller._full_corridor_violation(0.0, 4.110)

        self.assertAlmostEqual(violation, 0.976)

    def test_center_inside_raw_bound_but_footprint_at_wall_is_violation(self):
        controller = _controller()
        controller._car = SimpleNamespace(
            get_closest_waypoint=lambda _x, _y: 10)
        controller._reference_path = SimpleNamespace(
            get_waypoint=lambda _wp: SimpleNamespace(
                x=0.0, y=0.0, psi=0.0),
            get_lane_bounds=lambda _wp: [(4.0, -4.0)],
        )

        self.assertAlmostEqual(
            controller._full_corridor_violation(0.0, 3.9), 0.8)
        self.assertEqual(
            controller._full_corridor_violation(0.0, 0.0), 0.0)

    def test_pp_wall_check_keeps_the_same_vehicle_center_bounds(self):
        controller = _controller()

        lower, upper = controller._physical_wall_safe_bounds(-4.0, 4.0)

        self.assertAlmostEqual(lower, -3.1)
        self.assertAlmostEqual(upper, 3.1)
        source = inspect.getsource(MPCController._pure_pursuit_feedback_is_safe)
        self.assertIn("_physical_wall_safe_bounds", source)

    def test_safe_drive_rollout_keeps_existing_selection_conditions(self):
        controller = _controller()
        controller._straight_reentry_min_improvement = 0.20
        controller._full_corridor_violation = lambda _x, _y: 0.976
        controller._straight_reentry_rollout = lambda _pose, direction: (
            np.linspace(0.976, 0.60, 11)
            if direction > 0 else np.linspace(0.976, 1.10, 11)
        )
        controller._relative_lane_vehicle_samples = (
            lambda _pose, _speed: [])

        self.assertEqual(
            controller._select_straight_reentry_direction(
                SimpleNamespace(x=0.0, y=0.0)),
            1,
        )

    def test_intermediate_worsening_still_rejects_drive_rollout(self):
        controller = _controller()
        controller._straight_reentry_min_improvement = 0.20
        controller._full_corridor_violation = lambda _x, _y: 0.976
        controller._straight_reentry_rollout = lambda _pose, direction: (
            np.array([0.976, 1.08, 1.09, 0.60])
            if direction > 0 else np.linspace(0.976, 1.10, 11)
        )

        self.assertEqual(
            controller._select_straight_reentry_direction(
                SimpleNamespace(x=0.0, y=0.0)),
            0,
        )

    def test_safe_reverse_still_uses_existing_rear_clear_gate(self):
        controller = _controller()
        controller._straight_reentry_enabled = True
        controller._straight_reentry_min_improvement = 0.20
        controller._straight_reentry_probe_distance = 2.0
        controller._stuck_pre_reverse_duration = 0.0
        controller._full_corridor_violation = lambda _x, _y: 0.976
        controller._straight_reentry_rollout = lambda _pose, direction: (
            np.linspace(0.976, 1.10, 11)
            if direction > 0 else np.linspace(0.976, 0.60, 11)
        )
        controller._reverse_rear_is_clear = lambda *_args, **_kwargs: True
        controller._last_stuck_gear_command = object()
        controller._stuck_last_drive_request_at = 1.0
        controller._stuck_last_reverse_request_at = 1.0
        controller.get_logger = lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None)

        self.assertTrue(controller._start_straight_reentry(
            SimpleNamespace(x=0.0, y=0.0), 2.0))
        self.assertEqual(controller._straight_reentry_direction, -1)

    def test_failed_reentry_still_falls_through_to_generic_reverse(self):
        source = inspect.getsource(MPCController._apply_stuck_recovery)
        reentry_position = source.index("self._start_straight_reentry")
        generic_reverse_position = source.index(
            "[MPCStallRecovery] starting reverse", reentry_position)

        self.assertLess(reentry_position, generic_reverse_position)
        self.assertNotIn(
            "StuckRecoveryBlindReverseBlocked",
            source[reentry_position:generic_reverse_position],
        )

    def test_drive_reentry_uses_normal_forward_acceleration_from_rest(self):
        controller = _controller()

        acceleration = controller._stuck_recovery_acceleration(1.0, 0.0)

        self.assertAlmostEqual(acceleration, 2.5)
        self.assertNotEqual(acceleration, -8.0)

    def test_drive_reentry_acceleration_reduces_near_target(self):
        controller = _controller()

        from_rest = controller._stuck_recovery_acceleration(1.0, 0.0)
        near_target = controller._stuck_recovery_acceleration(1.0, 0.99)
        at_target = controller._stuck_recovery_acceleration(1.0, 1.0)

        self.assertGreater(from_rest, near_target)
        self.assertGreater(near_target, at_target)
        self.assertAlmostEqual(at_target, 0.0)

    def test_non_driving_recovery_keeps_forced_stop_acceleration(self):
        controller = _controller()
        controller._straight_reentry_active = False

        self.assertEqual(
            controller._stuck_recovery_acceleration(0.0, 0.0), -8.0)

    def test_reverse_reentry_uses_existing_acceleration_after_gear_confirm(self):
        controller = _controller(direction=-1, gear=20, reverse_drive=False)

        self.assertAlmostEqual(
            controller._stuck_recovery_acceleration(1.0, 0.0), 2.5)
        self.assertFalse(controller._stuck_reverse_drive_active)

    def test_reverse_reentry_honors_existing_negative_acceleration_semantics(self):
        controller = _controller(direction=-1, gear=20, reverse_drive=False)
        controller._stuck_reverse_acceleration_positive = False

        self.assertAlmostEqual(
            controller._stuck_recovery_acceleration(1.0, 0.0), -2.5)

    def test_reverse_reentry_before_reverse_gear_confirmation_keeps_stop(self):
        controller = _controller(direction=-1, gear=2, reverse_drive=False)

        self.assertEqual(
            controller._stuck_recovery_acceleration(0.0, 0.0), -8.0)

    def test_reverse_reentry_in_pre_reverse_gear_keeps_stop(self):
        controller = _controller(direction=-1, gear=22, reverse_drive=False)

        self.assertEqual(
            controller._stuck_recovery_acceleration(0.0, 0.0), -8.0)

    def test_normal_stuck_reverse_keeps_existing_acceleration_semantics(self):
        controller = _controller(direction=0, gear=20, reverse_drive=True)
        controller._straight_reentry_active = False

        self.assertAlmostEqual(
            controller._stuck_recovery_acceleration(1.5, -0.5), 2.5)

    def test_drive_reentry_without_confirmed_drive_gear_keeps_stop(self):
        controller = _controller(gear=20)

        self.assertEqual(
            controller._stuck_recovery_acceleration(0.0, 0.0), -8.0)

    def test_localization_inconsistent_still_stops_straight_reentry(self):
        controller = _controller()
        controller._request_awsim_control_mode_for_recovery = lambda: None
        controller._full_corridor_violation = lambda _x, _y: 0.24
        controller._mpc = SimpleNamespace(
            infeasibility_counter=1,
            current_prediction=None,
            used_prediction_fallback=False,
            recovery_requested=False,
        )
        controller._straight_reentry_success_cycles = 0
        controller._straight_reentry_success_cycles_required = 8
        controller._straight_reentry_started_at = 1.0
        controller._straight_reentry_timeout = 8.0
        controller._straight_reentry_last_violation = 0.24
        controller._localization_consistent = False
        controller._gnss_history = [object()]
        controller._stuck_since = 1.0
        controller.get_logger = lambda: SimpleNamespace(
            warn=lambda *_args, **_kwargs: None)
        command = np.array([1.0, 0.1])

        controller._apply_straight_reentry(
            SimpleNamespace(nanoseconds=int(2.0e9)),
            SimpleNamespace(x=0.0, y=0.0),
            command,
        )

        self.assertFalse(controller._straight_reentry_active)
        self.assertEqual(controller._straight_reentry_direction, 0)
        self.assertEqual(command.tolist(), [0.0, 0.0])

    def test_normal_control_acceleration_branch_remains_separate(self):
        source = inspect.getsource(MPCController._control)

        self.assertIn("if recovering_from_stuck:", source)
        self.assertIn("else:\n            acc =  self.KP * (u[0] - v)", source)

    def test_reentry_direction_and_timeout_logic_remain_in_existing_owner(self):
        source = inspect.getsource(MPCController._apply_straight_reentry)

        self.assertIn("self._straight_reentry_timeout", source)
        self.assertIn("self._straight_reentry_direction > 0", source)
        self.assertIn("self._straight_reentry_direction < 0", source)


if __name__ == "__main__":
    unittest.main()
