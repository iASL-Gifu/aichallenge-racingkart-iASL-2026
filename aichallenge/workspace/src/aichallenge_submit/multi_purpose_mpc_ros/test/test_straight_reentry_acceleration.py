"""Regression tests for StraightReentry longitudinal ownership."""

from types import SimpleNamespace
import inspect
import unittest

import numpy as np

from multi_purpose_mpc_ros.mpc_controller import MPCController


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
    return controller


class StraightReentryAccelerationTest(unittest.TestCase):
    def test_physical_violation_blocks_blind_generic_reverse(self):
        controller = _controller()
        controller._full_corridor_violation = lambda _x, _y: 0.051

        blocked, violation = (
            controller._physical_violation_blocks_blind_reverse(
                SimpleNamespace(x=1.0, y=2.0)
            )
        )

        self.assertTrue(blocked)
        self.assertEqual(violation, 0.051)

    def test_inside_physical_corridor_preserves_generic_reverse(self):
        controller = _controller()
        controller._full_corridor_violation = lambda _x, _y: 0.05

        blocked, violation = (
            controller._physical_violation_blocks_blind_reverse(
                SimpleNamespace(x=1.0, y=2.0)
            )
        )

        self.assertFalse(blocked)
        self.assertEqual(violation, 0.05)

    def test_safe_straight_reentry_directions_still_start(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                controller = _controller(direction=direction)
                controller._straight_reentry_enabled = True
                controller._select_straight_reentry_direction = (
                    lambda _pose, selected=direction: selected
                )
                controller._reverse_rear_is_clear = lambda *_args, **_kwargs: True
                controller._straight_reentry_probe_distance = 2.0
                controller._stuck_pre_reverse_duration = 0.0
                controller._straight_reentry_min_improvement = 0.2
                controller._straight_reentry_active = False
                controller._full_corridor_violation = lambda _x, _y: 0.25
                controller._last_stuck_gear_command = object()
                controller._stuck_last_drive_request_at = 1.0
                controller._stuck_last_reverse_request_at = 1.0
                controller.get_logger = lambda: SimpleNamespace(
                    warn=lambda *_args, **_kwargs: None)

                started = controller._start_straight_reentry(
                    SimpleNamespace(x=0.0, y=0.0), 2.0)

                self.assertTrue(started)
                self.assertTrue(controller._straight_reentry_active)
                self.assertEqual(controller._straight_reentry_direction, direction)

    def test_failed_reentry_is_checked_before_generic_reverse_initialization(self):
        source = inspect.getsource(MPCController._apply_stuck_recovery)
        block_position = source.index(
            "_physical_violation_blocks_blind_reverse")
        reverse_position = source.index(
            "self._stuck_recovery_until = now_sec + self._stuck_max_shift_wait"
        )

        self.assertLess(block_position, reverse_position)
        self.assertIn("u[0] = 0.0", source[block_position:reverse_position])
        self.assertIn("return True", source[block_position:reverse_position])

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
