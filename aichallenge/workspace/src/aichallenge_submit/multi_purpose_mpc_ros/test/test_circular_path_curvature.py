import math
from types import SimpleNamespace

import pytest

from multi_purpose_mpc_ros.core.reference_path import (
    ReferencePath,
    smooth_circular_waypoint_curvatures,
)


def construct_waypoints(coordinates, circular):
    reference_path = object.__new__(ReferencePath)
    reference_path.circular = circular
    reference_path.eps = 1e-12
    return reference_path._construct_waypoints(coordinates)


def test_circular_first_waypoint_uses_final_incoming_segment():
    waypoints = construct_waypoints(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        circular=True,
    )

    assert waypoints[0].kappa == pytest.approx(math.pi / 2.0)


def test_non_circular_first_waypoint_keeps_zero_curvature():
    waypoints = construct_waypoints(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        circular=False,
    )

    assert waypoints[0].kappa == 0.0


def test_curvature_smoothing_wraps_across_circular_seam():
    waypoints = [SimpleNamespace(kappa=value) for value in [1.0] + [0.0] * 8]

    applied = smooth_circular_waypoint_curvatures(waypoints)

    assert applied
    assert waypoints[0].kappa != pytest.approx(1.0)
    assert waypoints[-1].kappa != pytest.approx(0.0)


def test_curvature_smoothing_skips_paths_shorter_than_window():
    waypoints = [SimpleNamespace(kappa=value) for value in [1.0, 0.0, 0.0]]

    applied = smooth_circular_waypoint_curvatures(waypoints)

    assert not applied
    assert [waypoint.kappa for waypoint in waypoints] == [1.0, 0.0, 0.0]
