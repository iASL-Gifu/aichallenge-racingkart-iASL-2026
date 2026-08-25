import unittest
from types import SimpleNamespace

import numpy as np
import osqp

from multi_purpose_mpc_ros.core.MPC import (
    apply_outer_boundary_guard,
    can_reuse_prediction_fallback,
    is_plausible_mpc_prediction,
    is_plausible_world_prediction,
    is_primal_infeasible,
    is_valid_osqp_solution,
    zero_inverted_bounds,
)


def result(status_name, values):
    return SimpleNamespace(
        x=values,
        info=SimpleNamespace(status_val=osqp.constant(status_name)),
    )


class TestOsqpSolutionStatus(unittest.TestCase):
    def test_solved_zero_steering_is_valid(self):
        solved = result('OSQP_SOLVED', np.zeros(10))

        self.assertTrue(is_valid_osqp_solution(solved))

    def test_solved_inaccurate_finite_solution_is_valid(self):
        solved = result('OSQP_SOLVED_INACCURATE', np.array([1.0, 0.0]))

        self.assertTrue(is_valid_osqp_solution(solved))

    def test_non_finite_solution_is_invalid(self):
        solved = result('OSQP_SOLVED', np.array([1.0, np.nan]))

        self.assertFalse(is_valid_osqp_solution(solved))

    def test_primal_infeasible_is_relaxable(self):
        infeasible = result('OSQP_PRIMAL_INFEASIBLE', None)

        self.assertTrue(is_primal_infeasible(infeasible))
        self.assertFalse(is_valid_osqp_solution(infeasible))

    def test_maximum_iterations_is_not_relaxable(self):
        max_iter = result('OSQP_MAX_ITER_REACHED', np.zeros(10))

        self.assertFalse(is_primal_infeasible(max_iter))
        self.assertFalse(is_valid_osqp_solution(max_iter))


class TestPredictionPlausibility(unittest.TestCase):
    def setUp(self):
        self.states = np.zeros((5, 3))
        self.prediction = ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        self.lower = np.full(4, -2.0)
        self.upper = np.full(4, 2.0)

    def is_plausible(self, **overrides):
        values = {
            "spatial_states": self.states,
            "world_prediction": self.prediction,
            "current_position": (0.0, 0.0),
            "lower_bounds": self.lower,
            "upper_bounds": self.upper,
        }
        values.update(overrides)
        return is_plausible_mpc_prediction(**values)

    def test_regular_prediction_is_valid(self):
        self.assertTrue(self.is_plausible())

    def test_lateral_constraint_violation_is_invalid(self):
        states = self.states.copy()
        states[2, 0] = 3.0

        self.assertFalse(self.is_plausible(spatial_states=states))

    def test_configured_small_lateral_violation_is_invalid(self):
        states = self.states.copy()
        states[2, 0] = 2.03

        self.assertFalse(self.is_plausible(
            spatial_states=states,
            lateral_tolerance=0.02,
        ))

    def test_distant_prediction_start_is_invalid(self):
        prediction = ([20.0, 21.0, 22.0], [0.0, 0.0, 0.0])

        self.assertFalse(self.is_plausible(world_prediction=prediction))

    def test_prediction_point_jump_is_invalid(self):
        prediction = ([1.0, 2.0, 20.0], [0.0, 0.0, 0.0])

        self.assertFalse(self.is_plausible(world_prediction=prediction))

    def test_non_finite_prediction_is_invalid(self):
        prediction = ([1.0, np.nan, 3.0], [0.0, 0.0, 0.0])

        self.assertFalse(self.is_plausible(world_prediction=prediction))

    def test_stale_fallback_prediction_is_invalid(self):
        self.assertFalse(is_plausible_world_prediction(
            self.prediction,
            current_position=(20.0, 0.0),
        ))


class TestOuterBoundaryGuard(unittest.TestCase):
    def setUp(self):
        self.lower = np.array([-3.0, -3.1])
        self.upper = np.array([3.0, 3.1])

    def test_full_width_insets_both_physical_edges(self):
        lower, upper = apply_outer_boundary_guard(
            self.lower, self.upper, None, 0.1)
        np.testing.assert_allclose(lower, [-2.9, -3.0])
        np.testing.assert_allclose(upper, [2.9, 3.0])

    def test_outer_lanes_inset_only_their_physical_edge(self):
        l0_lower, l0_upper = apply_outer_boundary_guard(
            self.lower, self.upper, 0, 0.1)
        l2_lower, l2_upper = apply_outer_boundary_guard(
            self.lower, self.upper, 2, 0.1)
        np.testing.assert_allclose(l0_lower, self.lower + 0.1)
        np.testing.assert_allclose(l0_upper, self.upper)
        np.testing.assert_allclose(l2_lower, self.lower)
        np.testing.assert_allclose(l2_upper, self.upper - 0.1)

    def test_l1_is_unchanged_and_inputs_are_not_mutated(self):
        lower_before = self.lower.copy()
        upper_before = self.upper.copy()
        lower, upper = apply_outer_boundary_guard(
            self.lower, self.upper, 1, 0.1)
        np.testing.assert_allclose(lower, lower_before)
        np.testing.assert_allclose(upper, upper_before)
        np.testing.assert_array_equal(self.lower, lower_before)
        np.testing.assert_array_equal(self.upper, upper_before)

    def test_inverted_narrow_corridor_becomes_zero_width(self):
        lower, upper = apply_outer_boundary_guard(
            np.array([-0.04]), np.array([0.04]), None, 0.1)
        self.assertGreater(lower[0], upper[0])
        lower, upper = zero_inverted_bounds(lower, upper)
        np.testing.assert_array_equal(lower, [0.0])
        np.testing.assert_array_equal(upper, [0.0])


class TestPredictionFallbackLimit(unittest.TestCase):
    def test_first_three_failures_may_reuse_valid_prediction(self):
        for failure_cycle in (1, 2, 3):
            self.assertTrue(can_reuse_prediction_fallback(
                failure_cycle,
                max_fallback_cycles=3,
                prediction_valid=True,
                control_valid=True,
            ))

    def test_fourth_failure_must_stop_reusing_prediction(self):
        self.assertFalse(can_reuse_prediction_fallback(
            failure_cycle=4,
            max_fallback_cycles=3,
            prediction_valid=True,
            control_valid=True,
        ))

    def test_invalid_prediction_is_never_reused(self):
        self.assertFalse(can_reuse_prediction_fallback(
            failure_cycle=1,
            max_fallback_cycles=3,
            prediction_valid=False,
            control_valid=True,
        ))


if __name__ == '__main__':
    unittest.main()
