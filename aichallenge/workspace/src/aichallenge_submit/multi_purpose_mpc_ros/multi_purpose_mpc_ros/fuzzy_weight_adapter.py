"""Lightweight fuzzy adaptation of MPC path-tracking weights.

The rule tables follow Tables 2--4 of Wang et al., "Path Tracking
Control for Autonomous Vehicles Based on an Improved MPC".  The runtime
calculation operates on two five-element membership vectors and does not
construct optimization matrices.
"""

from dataclasses import dataclass
import math

import numpy as np


# Output linguistic values: ZO, PS, PM, PB.
_OUTPUT_SINGLETONS = np.asarray([0.0, 0.35, 0.65, 1.0], dtype=float)

# Rows are heading error NB, NS, ZO, PS, PB. Columns are lateral error.
_Q_LATERAL_RULES = np.asarray([
    [0, 1, 2, 1, 0],
    [0, 1, 2, 1, 0],
    [1, 2, 3, 2, 1],
    [0, 1, 2, 1, 0],
    [0, 1, 2, 1, 0],
], dtype=np.int8)

_Q_HEADING_RULES = np.asarray([
    [3, 2, 1, 1, 0],
    [2, 1, 0, 1, 0],
    [1, 0, 0, 2, 1],
    [2, 1, 0, 1, 0],
    [3, 2, 1, 1, 0],
], dtype=np.int8)

_STEER_DELTA_RULES = np.asarray([
    [2, 1, 0, 1, 2],
    [3, 2, 1, 2, 3],
    [3, 2, 1, 2, 3],
    [3, 2, 1, 2, 3],
    [2, 1, 0, 1, 2],
], dtype=np.int8)


@dataclass(frozen=True)
class FuzzyWeightRatios:
    q_lateral: float
    q_heading: float
    steer_delta: float


def _lateral_memberships(normalized_error: float) -> np.ndarray:
    """Five Gaussian memberships matching the paper's lateral-error shape."""
    value = float(np.clip(normalized_error, -1.0, 1.0))
    centers = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    sigma = 0.21
    memberships = np.exp(-0.5 * ((value - centers) / sigma) ** 2)
    return memberships / max(float(np.max(memberships)), 1e-12)


def _heading_memberships(normalized_error: float) -> np.ndarray:
    """Five triangular/shoulder memberships matching the paper's heading plot."""
    value = float(np.clip(normalized_error, -1.0, 1.0))
    result = np.zeros(5, dtype=float)
    result[0] = float(np.clip((-0.5 - value) / 0.5, 0.0, 1.0))
    result[1] = float(np.clip(1.0 - abs((value + 0.5) / 0.5), 0.0, 1.0))
    result[2] = float(np.clip(1.0 - abs(value / 0.5), 0.0, 1.0))
    result[3] = float(np.clip(1.0 - abs((value - 0.5) / 0.5), 0.0, 1.0))
    result[4] = float(np.clip((value - 0.5) / 0.5, 0.0, 1.0))
    return result


def _infer_ratio(
    heading_memberships: np.ndarray,
    lateral_memberships: np.ndarray,
    rules: np.ndarray,
) -> float:
    """Mamdani min/max inference with lightweight singleton defuzzification."""
    strengths = np.minimum(
        heading_memberships[:, np.newaxis],
        lateral_memberships[np.newaxis, :],
    )
    aggregated = np.asarray([
        np.max(strengths[rules == output_index])
        for output_index in range(len(_OUTPUT_SINGLETONS))
    ])
    total = float(np.sum(aggregated))
    if total <= 1e-12:
        return 1.0
    return float(np.dot(aggregated, _OUTPUT_SINGLETONS) / total)


class FuzzyWeightAdapter:
    """Compute smoothed fuzzy weight ratios with a small fixed operation count."""

    def __init__(
        self,
        lateral_error_full_scale: float,
        heading_error_full_scale_rad: float,
        q_lateral_min_ratio: float,
        q_heading_min_ratio: float,
        steer_delta_min_ratio: float,
        smoothing_sec: float,
        control_period_sec: float,
    ) -> None:
        self.lateral_scale = max(abs(float(lateral_error_full_scale)), 1e-6)
        self.heading_scale = max(abs(float(heading_error_full_scale_rad)), 1e-6)
        self.minimums = np.clip(np.asarray([
            q_lateral_min_ratio,
            q_heading_min_ratio,
            steer_delta_min_ratio,
        ], dtype=float), 0.0, 1.0)
        smoothing = max(float(smoothing_sec), 0.0)
        period = max(float(control_period_sec), 1e-6)
        self.alpha = 1.0 if smoothing <= 0.0 else 1.0 - math.exp(-period / smoothing)
        # Start from the configured maximum weights and approach the fuzzy
        # output smoothly, avoiding a discontinuity on the first control cycle.
        self._smoothed = np.ones(3, dtype=float)

    def update(self, lateral_error: float, heading_error_rad: float) -> FuzzyWeightRatios:
        lateral = _lateral_memberships(float(lateral_error) / self.lateral_scale)
        heading = _heading_memberships(float(heading_error_rad) / self.heading_scale)
        raw = np.asarray([
            _infer_ratio(heading, lateral, _Q_LATERAL_RULES),
            _infer_ratio(heading, lateral, _Q_HEADING_RULES),
            _infer_ratio(heading, lateral, _STEER_DELTA_RULES),
        ])
        raw = np.maximum(raw, self.minimums)
        self._smoothed += self.alpha * (raw - self._smoothed)
        return FuzzyWeightRatios(*map(float, self._smoothed))

