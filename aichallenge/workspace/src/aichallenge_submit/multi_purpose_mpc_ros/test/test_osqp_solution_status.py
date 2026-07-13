import unittest
from types import SimpleNamespace

import numpy as np
import osqp

from multi_purpose_mpc_ros.core.MPC import (
    is_plausible_mpc_prediction,
    is_plausible_world_prediction,
    is_primal_infeasible,
    is_valid_osqp_solution,
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


if __name__ == '__main__':
    unittest.main()
