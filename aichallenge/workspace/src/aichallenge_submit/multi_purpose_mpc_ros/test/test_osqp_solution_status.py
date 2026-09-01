import unittest
from types import MethodType, SimpleNamespace

import numpy as np
import osqp

from multi_purpose_mpc_ros.core.MPC import (
    MPC,
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
        info=SimpleNamespace(
            status=status_name,
            status_val=osqp.constant(status_name),
        ),
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


class TestMpcGetControlTiming(unittest.TestCase):
    def test_phase_accounting_matches_total_without_changing_control(self):
        mpc = MPC.__new__(MPC)
        mpc.nx = 3
        mpc.nu = 2
        mpc.N = 3
        mpc.model = SimpleNamespace(
            reference_path=SimpleNamespace(
                n_waypoints=10,
                circular=True,
            ),
            wp_id=0,
            current_waypoint=object(),
            temporal_state=SimpleNamespace(x=0.0, y=0.0),
            spatial_state=SimpleNamespace(e_y=0.0),
            safety_margin=0.0,
            length=1.0,
            Ts=0.025,
        )
        mpc.model.get_current_waypoint = lambda: None
        mpc.model.t2s = lambda **_kwargs: SimpleNamespace(e_y=0.0)
        mpc.used_prediction_fallback = False
        mpc.time_budget_exceeded = False
        mpc.recovery_requested = False
        mpc.failure_reason = None
        mpc.last_solution_status = None
        mpc.last_solution_accurate = False
        mpc.current_prediction = None
        mpc.current_prediction_times = None
        mpc.current_constraint_prediction_times = None
        mpc.current_control = np.zeros(mpc.nu * mpc.N)
        mpc.previous_steering = 0.0
        mpc.max_steering_rate = 1.0
        mpc.prediction_lateral_tolerance = 0.02
        mpc.infeasibility_counter = 0
        mpc.last_solved_wp_id = 0
        mpc.debug_counter = 1
        mpc.lane_constraint_retry_relaxation_m = ()
        mpc.solve_time_budget_ms = 20.0

        solution = np.zeros((mpc.N + 1) * mpc.nx + mpc.N * mpc.nu)
        mpc.optimizer = SimpleNamespace(solve=lambda: result(
            'OSQP_SOLVED', solution))

        phase_names = (
            "problem_setup_ms",
            "linearization_reference_ms",
            "path_constraints_ms",
            "sparse_matrix_ms",
            "constraint_bounds_ms",
            "cost_vector_ms",
            "optimizer_update_ms",
        )

        def fake_init_problem(self, _N, _safety_margin,
                              lane_relaxation=0.0):
            del lane_relaxation
            self._prediction_lower_bounds = np.full(_N, -1.0)
            self._prediction_upper_bounds = np.full(_N, 1.0)
            self.last_problem_build_phase_wall_ms.append({
                **{name: 0.0 for name in phase_names},
                "reference_free_segments_ms": 0.0,
                "reference_combination_ms": 0.0,
                "reference_smoothing_ms": 0.0,
                "reference_other_ms": 0.0,
                "reference_internal_total_ms": 0.0,
                "reference_free_line_aa_ms": 0.0,
                "reference_free_static_map_check_ms": 0.0,
                "reference_free_dynamic_obstacle_check_ms": 0.0,
                "reference_free_dynamic_geometry_prepare_ms": 0.0,
                "reference_free_conversion_ms": 0.0,
                "reference_free_other_ms": 0.0,
                "reference_free_other_setup_ms": 0.0,
                "reference_free_other_endpoint_map_prepare_ms": 0.0,
                "reference_free_other_cell_loop_bookkeeping_ms": 0.0,
                "reference_free_other_post_loop_finalization_ms": 0.0,
                "reference_free_other_detail_total_ms": 0.0,
                "reference_free_other_remaining_ms": 0.0,
                "reference_free_internal_total_ms": 0.0,
                "reference_free_section_overhead_ms": 0.0,
                "reference_free_timing_call_count": 3,
                "reference_timing_available": True,
            })

        mpc._init_problem = MethodType(fake_init_problem, mpc)
        mpc.update_prediction_with_times = MethodType(
            lambda self, _states, _N: (([0.0], [0.0]), [0.1]),
            mpc,
        )
        mpc.constraint_prediction_times = MethodType(
            lambda self, _states, _N: [0.1] * _N,
            mpc,
        )

        control, _ = mpc.get_control()

        np.testing.assert_array_equal(control, [0.0, 0.0])
        timing = mpc.last_get_control_timing_ms
        self.assertEqual(timing["problem_build_call_count"], 1)
        accounted = sum(timing[name] for name in phase_names) + sum((
            timing["state_prepare_ms"],
            timing["solve_ms"],
            timing["result_postprocess_ms"],
            timing["fallback_ms"],
            timing["tail_ms"],
            timing["other_ms"],
        ))
        self.assertAlmostEqual(accounted, timing["total_ms"], places=6)
        self.assertAlmostEqual(
            timing["non_solve_ms"],
            timing["total_ms"] - timing["solve_ms"],
            places=6,
        )
        self.assertEqual(timing["reference_free_timing_call_count"], 3)
        self.assertAlmostEqual(
            timing["reference_free_other_setup_ms"]
            + timing["reference_free_other_endpoint_map_prepare_ms"]
            + timing["reference_free_other_cell_loop_bookkeeping_ms"]
            + timing["reference_free_other_post_loop_finalization_ms"]
            + timing["reference_free_other_remaining_ms"],
            timing["reference_free_other_ms"],
            places=6,
        )
        self.assertAlmostEqual(
            timing["reference_free_line_aa_ms"]
            + timing["reference_free_static_map_check_ms"]
            + timing["reference_free_dynamic_obstacle_check_ms"]
            + timing["reference_free_dynamic_geometry_prepare_ms"]
            + timing["reference_free_conversion_ms"]
            + timing["reference_free_other_ms"],
            timing["reference_free_internal_total_ms"],
            places=6,
        )
        self.assertAlmostEqual(
            timing["reference_free_internal_total_ms"]
            + timing["reference_free_section_overhead_ms"],
            timing["reference_free_segments_ms"],
            places=6,
        )
        path_detail_names = (
            "path_constraint_metadata_ms",
            "path_constraint_reference_bounds_ms",
            "path_constraint_boundary_guard_ms",
            "path_constraint_state_bounds_ms",
            "path_constraint_lateral_reference_ms",
        )
        self.assertAlmostEqual(
            sum(timing[name] for name in path_detail_names)
            + timing["path_constraint_other_ms"],
            timing["path_constraints_ms"],
            places=6,
        )
        self.assertEqual(timing["reference_timing_call_count"], 1)
        self.assertAlmostEqual(
            timing["reference_free_segments_ms"]
            + timing["reference_combination_ms"]
            + timing["reference_smoothing_ms"]
            + timing["reference_other_ms"],
            timing["reference_internal_total_ms"],
            places=6,
        )


if __name__ == '__main__':
    unittest.main()
