"""Tests for the speed-dependent simple understeer model."""

import unittest

from multi_purpose_mpc_ros.core.spatial_bicycle_models import (
    BicycleModel,
    understeer_curvature_gain,
)


class UndersteerModelTest(unittest.TestCase):
    def test_zero_coefficient_preserves_kinematic_model(self):
        self.assertEqual(understeer_curvature_gain(10.0, 0.0), 1.0)

    def test_curvature_gain_decreases_with_speed(self):
        low_speed_gain = understeer_curvature_gain(5.0, 0.002)
        high_speed_gain = understeer_curvature_gain(10.0, 0.002)

        self.assertLess(high_speed_gain, low_speed_gain)
        self.assertLess(low_speed_gain, 1.0)
        self.assertAlmostEqual(high_speed_gain, 1.0 / 1.2)

    def test_negative_coefficient_is_disabled(self):
        self.assertEqual(understeer_curvature_gain(10.0, -0.002), 1.0)

    def test_linearized_model_applies_gain_to_curvature_input(self):
        model = object.__new__(BicycleModel)
        _, _, input_matrix = model.linearize(
            v_ref=10.0,
            kappa_ref=0.1,
            delta_s=0.6,
            understeer_coeff=0.002,
        )

        expected_gain = understeer_curvature_gain(10.0, 0.002)
        self.assertAlmostEqual(input_matrix[1, 1], expected_gain * 0.6)


if __name__ == "__main__":
    unittest.main()
