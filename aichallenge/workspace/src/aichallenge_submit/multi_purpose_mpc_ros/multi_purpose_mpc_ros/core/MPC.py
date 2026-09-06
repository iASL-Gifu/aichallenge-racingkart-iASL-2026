from numpy.core import function_base
import math
import decimal
import decimal
from numpy import typing
from typing import Tuple
import numpy as np
import osqp
from scipy import sparse
import time
from datetime import datetime

from multi_purpose_mpc_ros.core.spatial_bicycle_models import (
    understeer_curvature_gain,
)
from multi_purpose_mpc_ros.core.reference_path import (
    collapsed_constraint_snapshot,
    retain_first_collapsed_constraint,
)

# Colors
PREDICTION = '#BA4A00'

OSQP_SOLVED_STATUSES = {
    osqp.constant('OSQP_SOLVED'),
    osqp.constant('OSQP_SOLVED_INACCURATE'),
}
OSQP_PRIMAL_INFEASIBLE_STATUSES = {
    osqp.constant('OSQP_PRIMAL_INFEASIBLE'),
    osqp.constant('OSQP_PRIMAL_INFEASIBLE_INACCURATE'),
}


def curvature_lateral_shift(
    max_abs_kappa,
    threshold,
    gain,
    max_shift,
) -> float:
    """Return a bounded L1-side offset requested by upcoming curvature."""
    values = (max_abs_kappa, threshold, gain, max_shift)
    if not all(np.isfinite(float(value)) for value in values):
        return 0.0
    excess_curvature = max(
        float(max_abs_kappa) - max(float(threshold), 0.0),
        0.0,
    )
    requested_shift = max(float(gain), 0.0) * excess_curvature
    return float(np.clip(requested_shift, 0.0, max(float(max_shift), 0.0)))


def is_valid_osqp_solution(result) -> bool:
    """Return whether OSQP produced a usable optimal solution."""
    return (
        result is not None
        and result.x is not None
        and result.info.status_val in OSQP_SOLVED_STATUSES
        and np.all(np.isfinite(result.x))
    )


def is_primal_infeasible(result) -> bool:
    """Return whether relaxing path constraints could make the QP feasible."""
    return (
        result is not None
        and result.info.status_val in OSQP_PRIMAL_INFEASIBLE_STATUSES
    )


def is_plausible_mpc_prediction(
    spatial_states,
    world_prediction,
    current_position,
    lower_bounds,
    upper_bounds,
    lateral_tolerance=0.5,
    max_start_distance=8.0,
    max_step_distance=5.0,
) -> bool:
    """Reject finite but physically implausible solver predictions."""
    states = np.asarray(spatial_states)
    if states.ndim != 2 or states.shape[0] < 3 or states.shape[1] < 2:
        return False
    if not np.all(np.isfinite(states)):
        return False

    lower = np.asarray(lower_bounds).reshape(-1)
    upper = np.asarray(upper_bounds).reshape(-1)
    n_bounds = min(states.shape[0] - 1, lower.size, upper.size)
    if n_bounds == 0:
        return False
    lateral = states[1:n_bounds + 1, 0]
    if np.any(lateral < lower[:n_bounds] - lateral_tolerance):
        return False
    if np.any(lateral > upper[:n_bounds] + lateral_tolerance):
        return False

    return is_plausible_world_prediction(
        world_prediction,
        current_position,
        max_start_distance=max_start_distance,
        max_step_distance=max_step_distance,
    )


def apply_outer_boundary_guard(
    lower_bounds, upper_bounds, target_lane, guard_margin,
):
    """Inset only the physical course edge(s) used by the active corridor."""
    lower = np.asarray(lower_bounds, dtype=float).copy()
    upper = np.asarray(upper_bounds, dtype=float).copy()
    guard = float(guard_margin)
    if not np.isfinite(guard) or guard <= 0.0:
        return lower, upper
    if target_lane is None:
        lower += guard
        upper -= guard
    elif target_lane == 0:
        lower += guard
    elif target_lane == 2:
        upper -= guard
    return lower, upper


def zero_inverted_bounds(lower_bounds, upper_bounds):
    """Replace inverted corridor samples with the zero-width sentinel."""
    lower = np.asarray(lower_bounds, dtype=float).copy()
    upper = np.asarray(upper_bounds, dtype=float).copy()
    inverted = upper < lower
    lower[inverted] = 0.0
    upper[inverted] = 0.0
    return lower, upper


def has_active_offset_limits(offsets) -> bool:
    """Return whether an objective-offset profile requests real movement."""
    if offsets is None:
        return False
    values = np.asarray(offsets, dtype=float).reshape(-1)
    return bool(values.size and np.any(np.isfinite(values) & (values > 0.0)))


def build_arc_length_steering_reservation(
    distances,
    steering_refs,
    speeds,
    delay_sec,
    steering_rate,
    current_steering,
    command_period,
):
    """Build a delayed, steering-rate-feasible reference along arc length."""
    distance_array = np.asarray(distances, dtype=float)
    reference_array = np.asarray(steering_refs, dtype=float)
    speed_array = np.asarray(speeds, dtype=float)
    if (
        distance_array.size == 0
        or distance_array.size != reference_array.size
        or distance_array.size != speed_array.size
        or steering_rate <= 0.0
    ):
        return reference_array.copy()

    count = distance_array.size
    delayed = reference_array.copy()
    for index in range(count):
        target_distance = (
            distance_array[index]
            + max(float(speed_array[index]), 0.0)
            * max(float(delay_sec), 0.0)
        )
        target_index = int(np.searchsorted(
            distance_array, target_distance, side="left"))
        target_index = min(max(target_index, index), count - 1)
        delayed[index] = reference_array[target_index]

    # Propagate future steering requirements backward so the actuator begins
    # moving before the curve, while respecting its physical slew rate.
    reserved = delayed.copy()
    for index in range(count - 2, -1, -1):
        ds = max(float(distance_array[index + 1] - distance_array[index]), 0.0)
        speed = max(float(speed_array[index]), 0.5)
        max_change = float(steering_rate) * ds / speed
        reserved[index] = float(np.clip(
            reserved[index],
            reserved[index + 1] - max_change,
            reserved[index + 1] + max_change,
        ))

    previous = float(current_steering)
    for index in range(count):
        if index == 0:
            max_change = float(steering_rate) * max(float(command_period), 0.0)
        else:
            ds = max(float(
                distance_array[index] - distance_array[index - 1]), 0.0)
            speed = max(float(speed_array[index - 1]), 0.5)
            max_change = float(steering_rate) * ds / speed
        reserved[index] = float(np.clip(
            reserved[index], previous - max_change, previous + max_change))
        previous = reserved[index]
    return reserved


def is_plausible_world_prediction(
    world_prediction,
    current_position,
    max_start_distance=8.0,
    max_step_distance=5.0,
) -> bool:
    """Validate a world-frame prediction, including a stored fallback."""
    x_pred, y_pred = world_prediction
    points = np.column_stack((x_pred, y_pred))
    if points.shape[0] == 0 or not np.all(np.isfinite(points)):
        return False

    current_xy = np.asarray(current_position, dtype=float)
    if current_xy.shape != (2,) or not np.all(np.isfinite(current_xy)):
        return False
    if np.linalg.norm(points[0] - current_xy) > max_start_distance:
        return False
    if (
        len(points) > 1
        and np.any(np.linalg.norm(np.diff(points, axis=0), axis=1)
                   > max_step_distance)
    ):
        return False
    return True


def can_reuse_prediction_fallback(
    failure_cycle,
    max_fallback_cycles,
    prediction_valid,
    control_valid,
) -> bool:
    """Allow a stored prediction/control pair for only a bounded time."""
    return (
        1 <= int(failure_cycle) <= max(int(max_fallback_cycles), 0)
        and bool(prediction_valid)
        and bool(control_valid)
    )


def blend_lateral_reference(start_e_y, lane_center_e_y, alpha) -> float:
    """Blend a full-width lateral reference toward a lane center."""
    blend = float(np.clip(alpha, 0.0, 1.0))
    return (
        (1.0 - blend) * float(start_e_y)
        + blend * float(lane_center_e_y)
    )


def lateral_reference_ramp_duration(
    start_e_y, lane_center_e_y, minimum_sec, max_speed
) -> float:
    """Return a ramp duration that respects a lateral-reference speed cap."""
    minimum = max(float(minimum_sec), 0.0)
    speed = max(float(max_speed), 0.0)
    if speed <= 0.0:
        return minimum
    distance = abs(float(lane_center_e_y) - float(start_e_y))
    return max(minimum, distance / speed)

def smootherstep01(values):
    """Quintic [0, 1] easing with continuous slope and curvature."""
    t = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def spatial_lane_transition_reference(
    distances, start_e_y, lane_centers, transition_length,
):
    """Build an N+1 lane-change reference as a function of path distance."""
    distance_values = np.asarray(distances, dtype=float).reshape(-1)
    center_values = np.asarray(lane_centers, dtype=float).reshape(-1)
    if distance_values.size != center_values.size:
        raise ValueError("distances and lane_centers must have equal length")
    length = max(float(transition_length), 1e-6)
    weights = smootherstep01(distance_values / length)
    targets = (
        (1.0 - weights) * float(start_e_y)
        + weights * center_values
    )
    return targets, weights


def blend_previous_lateral_prediction(
    nominal, previous, continuity_weight, max_deviation,
):
    """Blend a valid shifted MPC prediction without overpowering the plan."""
    nominal_values = np.asarray(nominal, dtype=float).reshape(-1)
    if previous is None:
        return nominal_values.copy()
    previous_values = np.asarray(previous, dtype=float).reshape(-1)
    if previous_values.size != nominal_values.size:
        return nominal_values.copy()
    if not np.all(np.isfinite(previous_values)):
        return nominal_values.copy()
    if np.max(np.abs(previous_values - nominal_values)) > max(float(max_deviation), 0.0):
        return nominal_values.copy()
    weight = float(np.clip(continuity_weight, 0.0, 1.0))
    return (1.0 - weight) * nominal_values + weight * previous_values


##################
# MPC Controller #
##################

class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, max_steering_rate, wp_id_offset, use_obstacle_avoidance,
                 use_path_constraints_topic, use_max_kappa_pred=True,
                 understeer_coeff=0.0, steering_command_delay=0.15,
                 steering_reservation_enabled=False):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param N: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param ay_max: maximum allowed lateral acceleration in curves
        :param wp_id_offset: offset for waypoint id to consider control delay
        :param use_obstacle_avoidance: flag to enable obstacle avoidance
        :param use_path_constraints_topic: flag to use path constraints from topic
        :param max_steering_rate: maximum allowed steering rate in rad/s
        :param understeer_coeff: coefficient K in
            kappa_actual = kappa_cmd / (1 + K * v^2), in s^2/m^2
        """
        # MPCの設定値と内部変数を初期化する
        # 既存の初期化パラメータ
        self.N = N #予測ホライズン
        self.Q = Q #状態誤差の重み
        self.R = R #入力コスト
        #R =[[1,0],[0,20]]なら速度変更は許す、操舵変更は嫌う
        self.QN = QN
        self.wp_id_offset = wp_id_offset
        self.use_obstacle_avoidance = use_obstacle_avoidance
        self.use_path_constraints_topic = use_path_constraints_topic
        self.model = model #車両モデル
        self.nx = self.model.n_states
        self.nu = 2
        # This objective-only target is independent from target_lane_idx.
        # It lets recovery keep full-width bounds while gently attracting the
        # prediction toward L1 with the existing, fixed Q matrix.
        self.soft_target_lane_idx = None
        self.soft_target_start_e_y = 0.0
        self.soft_target_alpha = 0.0
        self.soft_target_lateral_offset = 0.0
        self.soft_lateral_targets = None
        self.lane_transition_weights = None
        # Per-horizon objective offsets. These alter xr only; hard lane and
        # physical-course bounds remain unchanged.
        self.target_lane_lateral_offsets = None
        self.full_width_l1_offset_limits = None
        self.full_width_l0_offset_limits = None

        # setupが済んでいるかどうか
        self.osqp_initialized = False

        # ステアリングレート制約用の事前計算変数
        self.nx_N = self.nx * (self.N + 1)
        self.nu_N = self.nu * self.N

        # ステアリングレート制約行列の作成
        self.n_rate_constraints = N - 1
        steering_rate_matrix = sparse.lil_matrix(
            (self.n_rate_constraints, self.nx_N + self.nu_N)
        )
        for i in range(self.n_rate_constraints):
            steering_rate_matrix[i, self.nx_N + self.nu*i + 1] = -1
            steering_rate_matrix[i, self.nx_N + self.nu*(i+1) + 1] = 1
        self.steering_rate_matrix = steering_rate_matrix.tocsc()
        #sparse.eyeの固定`
        self.basic_constraint_matrix = sparse.eye(
            self.nx_N + self.nu_N,
            format='csc'
        )
        self.A_inequality = sparse.vstack([
            self.basic_constraint_matrix,
            self.steering_rate_matrix
        ], format='csc')        


        #Axのsparse.eye()固定
        self.Ax_base = sparse.kron(
            sparse.eye(self.N + 1),
            -sparse.eye(self.nx),
            format='csc'
        )
        #Pも固定
        self.P_base = sparse.block_diag([
            sparse.kron(sparse.eye(self.N), self.Q),
            self.QN,
            sparse.kron(sparse.eye(self.N), self.R)
        ], format='csc')

        # A, B行列のスパース構造を完全に固定するためのインデックス事前計算
        row_A, col_A = [], []
        row_B, col_B = [], []
        for n in range(self.N):
            for i in range(self.nx):
                for j in range(self.nx):
                    row_A.append((n+1)*self.nx + i)
                    col_A.append(n*self.nx + j)
                for j in range(self.nu):
                    row_B.append((n+1)*self.nx + i)
                    col_B.append(n*self.nu + j)
        self.row_A = np.array(row_A)
        self.col_A = np.array(col_A)
        self.row_B = np.array(row_B)
        self.col_B = np.array(col_B)

        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints
        self.ay_max = ay_max
        self.understeer_coeff = max(float(understeer_coeff), 0.0)

        # 追加: ステアリングレート制限関連のパラメータ
        self.max_steering_rate = max_steering_rate
        self.previous_steering = 0.0  # 前回のステア角
        self.steering_command_delay = max(float(steering_command_delay), 0.0)
        self.steering_reservation_enabled = bool(
            steering_reservation_enabled)

        # 追加: ay_maxによる速度制限の方式切り替え
        self.use_max_kappa_pred = use_max_kappa_pred
        # 既存の初期化
        self.current_prediction = None
        self.infeasibility_counter = 0
        self.solve_time_budget_ms = 20.0
        self.max_prediction_fallback_cycles = 3
        self.prediction_outer_boundary_guard = 0.0
        self.prediction_lateral_tolerance = 0.02
        self.lane_constraint_retry_relaxation_m = (
            0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.20)
        self.lane_constraint_retry_relax_toward_center_only = True
        self.lane_constraint_retry_taper_over_horizon = True
        self.lane_constraint_retry_terminal_ratio = 0.35
        self.lane_constraint_connection_points = 10
        self.used_prediction_fallback = False
        self.time_budget_exceeded = False
        self.recovery_requested = False
        self.failure_reason = None
        # Shadow/recovery callers distinguish an exact OSQP solution from
        # SOLVED_INACCURATE without reaching into the solver result object.
        self.last_solution_status = None
        self.last_solution_accurate = False
        self.soft_target_lane_idx = None
        self.soft_target_start_e_y = 0.0
        self.soft_target_alpha = 0.0
        self.soft_target_lateral_offset = 0.0
        self.soft_lateral_targets = None
        self.target_lane_lateral_offsets = None
        self.full_width_l1_offset_limits = None
        self.full_width_l0_offset_limits = None
        # Snapshot of the exact corridor used by the latest solve attempt.
        # These values remain available after an infeasible solve so the
        # controller can diagnose lane-bound and obstacle-induced failures.
        self._constraint_wp_ids = np.array([], dtype=int)
        self._constraint_target_lane = None
        self._constraint_safety_margin = 0.0
        self._constraint_lane_relaxation = 0.0
        # Preserve the first invalid corridor seen in the initial solve or
        # any retry. A later relaxed solution must not hide the collapse.
        self._constraint_collapse_detected = False
        self._constraint_collapse_detail = None
        self._current_constraint_bounds_invalid = False
        self.last_solved_wp_id = 0
        self.current_control = np.zeros((self.nu*self.N))
        self.optimizer = osqp.OSQP()

        self.debug_counter = 0

        self.startup =0
        self.linearize =0
        self.path_constraints =0
        self.sparse =0
        self.constraints2 =0
        self.vector =0
        self.update =0


        if not self.use_obstacle_avoidance:
            self.model.reference_path.update_simple_path_constraints(
                N,
                self.model.safety_margin)

    def update_v_max(self, v_max: float):
        self.input_constraints['umax'][0] = v_max

    def update_ay_max(self, ay_max: float):
        self.ay_max = ay_max

    def update_understeer_coeff(self, understeer_coeff: float):
        self.understeer_coeff = max(float(understeer_coeff), 0.0)

    def update_wp_id_offset(self, wp_id_offset: int):
        self.wp_id_offset = wp_id_offset

    def update_Q(self, Q: np.ndarray):
        self.Q = Q

    def update_R(self, R: np.ndarray):
        self.R = R

    def update_QN(self, QN: np.ndarray):
        self.QN = QN

    def set_soft_lateral_reference(
        self, lane_idx=None, start_e_y=0.0, alpha=0.0,
        lateral_offset=0.0, lateral_targets=None,
    ) -> None:
        """Set an objective-only lane reference without narrowing bounds."""
        self.soft_target_lane_idx = (
            int(lane_idx) if lane_idx is not None else None
        )
        self.soft_target_start_e_y = float(start_e_y)
        self.soft_target_alpha = float(np.clip(alpha, 0.0, 1.0))
        self.soft_target_lateral_offset = float(lateral_offset)
        targets = (
            None if lateral_targets is None
            else np.asarray(lateral_targets, dtype=float).reshape(-1).copy()
        )
        self.soft_lateral_targets = (
            targets if targets is not None and targets.size else None)

    def set_target_lane_lateral_offsets(self, offsets=None) -> None:
        """Offset a hard target lane's objective without changing bounds."""
        values = (
            None if offsets is None
            else np.asarray(offsets, dtype=float).reshape(-1).copy()
        )
        self.target_lane_lateral_offsets = (
            values if values is not None and values.size else None)

    def set_lane_transition_weights(self, weights=None) -> None:
        """Set per-horizon artificial-lane contraction progress."""
        values = (
            None if weights is None
            else np.asarray(weights, dtype=float).reshape(-1).copy()
        )
        self.lane_transition_weights = (
            np.clip(values, 0.0, 1.0)
            if values is not None and values.size else None)


    def set_full_width_l1_offset_limits(self, offsets=None) -> None:
        """Cap objective-only motion from full-width midpoint toward L1."""
        values = (
            None if offsets is None
            else np.asarray(offsets, dtype=float).reshape(-1).copy()
        )
        self.full_width_l1_offset_limits = (
            values if values is not None and values.size else None)

    def set_full_width_l0_offset_limits(self, offsets=None) -> None:
        """Cap objective-only motion from full-width midpoint toward L0."""
        values = (
            None if offsets is None
            else np.asarray(offsets, dtype=float).reshape(-1).copy()
        )
        self.full_width_l0_offset_limits = (
            values if values is not None and values.size else None)

    def _compute_lane_center(self, wp_id: int, target_lane: int) -> float:
        lanes = self.model.reference_path.get_lane_bounds(wp_id)
        if not lanes or target_lane >= len(lanes):
            return 0.0
        ub_lane, lb_lane = lanes[target_lane]
        lane_center = (ub_lane + lb_lane) / 2.0
        margin_from_edge = (self.model.width / 2.0) + 0.80
        min_center = lb_lane + margin_from_edge
        max_center = ub_lane - margin_from_edge
        if min_center > max_center:
            return lane_center
        return float(np.clip(lane_center, min_center, max_center))

    def _init_problem(self, N, safety_margin, lane_relaxation=0.0):
        """
        Initialize optimization problem for current time step with steering rate constraints.
        """

        t_start = time.perf_counter()
        self._current_constraint_bounds_invalid = False
        
        # 既存の制約設定
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # Precompute common terms
        nx_N = self.nx * (N + 1)
        nu_N = self.nu * N

        # LTV System Matrices
        A_data = np.zeros(N * self.nx * self.nx)
        B_data = np.zeros(N * self.nx * self.nu)

        # Reference vector
        ur = np.zeros(nu_N)
        xr = np.zeros(nx_N)
        uq = np.zeros(N * self.nx)

        # Dynamic constraints
        xmin_dyn = np.kron(np.ones(N + 1), xmin)
        xmax_dyn = np.kron(np.ones(N + 1), xmax)
        umax_dyn = np.kron(np.ones(N), umax)

        # Get curvature predictions
        kappa_pred = np.tan(np.append(np.array(self.current_control[3::self.nu]), self.current_control[-1])) / self.model.length

        # Consider control delay
        self.model.wp_id += self.wp_id_offset

        reserved_steering = None
        if self.steering_reservation_enabled:
            horizon_distance = np.zeros(N + 1, dtype=float)
            steering_reference = np.zeros(N + 1, dtype=float)
            reservation_speeds = np.zeros(N + 1, dtype=float)
            delta_limit = math.atan(
                abs(float(self.input_constraints['umax'][1]))
                * self.model.length)
            for index in range(N + 1):
                waypoint = self.model.reference_path.get_waypoint(
                    self.model.wp_id + index)
                speed = float(np.clip(
                    waypoint.v_ref,
                    self.input_constraints['umin'][0],
                    self.input_constraints['umax'][0],
                ))
                gain = understeer_curvature_gain(
                    speed, self.understeer_coeff)
                commanded_curvature = waypoint.kappa / max(gain, 1e-3)
                steering_reference[index] = float(np.clip(
                    math.atan(self.model.length * commanded_curvature),
                    -delta_limit,
                    delta_limit,
                ))
                reservation_speeds[index] = max(speed, 0.0)
                if index:
                    previous_waypoint = self.model.reference_path.get_waypoint(
                        self.model.wp_id + index - 1)
                    horizon_distance[index] = (
                        horizon_distance[index - 1]
                        + float(waypoint - previous_waypoint)
                    )
            reserved_steering = build_arc_length_steering_reservation(
                horizon_distance,
                steering_reference,
                reservation_speeds,
                self.steering_command_delay,
                self.max_steering_rate,
                self.previous_steering,
                self.model.Ts,
            )

        # Iterate over horizon
        t_pref = time.perf_counter()

        for n in range(N):
            # Get waypoint information
            current_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
            delta_s = next_waypoint - current_waypoint
            kappa_ref = current_waypoint.kappa

            # Clip reference velocity
            v_ref = np.clip(current_waypoint.v_ref, self.input_constraints['umin'][0], self.input_constraints['umax'][0])
            #v_ref = 12.5 

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(
                v_ref, kappa_ref, delta_s, self.understeer_coeff)
            eps = 1e-9
            A_lin[np.abs(A_lin) < eps] = eps
            B_lin[np.abs(B_lin) < eps] = eps
            #print(np.abs(A_lin) < 1e-12, flush=True)
            #print(np.count_nonzero(A_lin),flush=True)
            A_data[n*self.nx*self.nx : (n+1)*self.nx*self.nx] = A_lin.flatten()
            B_data[n*self.nx*self.nu : (n+1)*self.nx*self.nu] = B_lin.flatten()

            # Set reference
            curvature_gain = understeer_curvature_gain(
                v_ref, self.understeer_coeff)
            # Request enough commanded curvature for the speed-dependent
            # model to achieve the reference-path curvature.
            kappa_cmd_ref = kappa_ref / max(curvature_gain, 1e-3)
            if reserved_steering is not None:
                kappa_cmd_ref = (
                    math.tan(float(reserved_steering[n]))
                    / self.model.length
                )
            ur[n*self.nu:(n+1)*self.nu] = [v_ref, kappa_cmd_ref]
            uq[n * self.nx:(n+1)*self.nx] = B_lin.dot(
                [v_ref, kappa_cmd_ref]) - f

            # Set spatial reference e_y to target lane center with vehicle safety offset
            target_lane = getattr(self.model.reference_path, 'target_lane_idx', None)
            if target_lane is not None:
                xr[n * self.nx] = self._compute_lane_center(self.model.wp_id + n, target_lane)

            # Constrain maximum speed based on curvature
            # 曲率にもとづいた最大速度の制約
            if self.use_max_kappa_pred:
                max_kappa_pred = np.max(np.abs(kappa_pred[n:]))
                vmax_dyn = np.sqrt(self.ay_max / (np.abs(max_kappa_pred) + 1e-12))
            else:
                vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
                
            umax_dyn[self.nu*n] = min(vmax_dyn, umax_dyn[self.nu*n])

            if n == 0:
                self.deibug_max_kappa_pred = max_kappa_pred if self.use_max_kappa_pred else kappa_pred[n]
                self.debug_vmax_dyn = vmax_dyn

            #if n == 0 and self.debug_counter % 20 == 0:
            #    print(
        #        f"kappa_pred={self.debug_max_kappa_pred:.4f} "
        #        f"vmax_dyn={self.debug_vmax_dyn:.2f}",
        #        flush=True
        #    )

        # 終端状態に対する目標
        target_lane = getattr(self.model.reference_path, 'target_lane_idx', None)
        if target_lane is not None:
            xr[N * self.nx] = self._compute_lane_center(self.model.wp_id + N, target_lane)

        t_linearize = time.perf_counter()

        # Update path constraints
        self._constraint_wp_ids = np.array([
            (self.model.wp_id + 1 + n)
            % self.model.reference_path.n_waypoints
            for n in range(N)
        ], dtype=int)
        self._constraint_target_lane = getattr(
            self.model.reference_path, 'target_lane_idx', None)
        # Selected lanes no longer receive the former symmetric 40%-of-lane
        # margin.  Keep the diagnostic snapshot aligned with the bounds that
        # reference_path actually sends to the solver.
        self._constraint_safety_margin = (
            0.0 if self._constraint_target_lane in (0, 1, 2)
            else float(safety_margin)
        )
        self._constraint_lane_relaxation = max(float(lane_relaxation), 0.0)
        if self.use_obstacle_avoidance and not self.use_path_constraints_topic:
            ub, lb, _ = self.model.reference_path.update_path_constraints(
                self.model.wp_id + 1,
                [self.model.temporal_state.x, self.model.temporal_state.y, self.model.temporal_state.psi],
                N, self.model.length, self.model.width, safety_margin,
                lane_relaxation=self._constraint_lane_relaxation,
                toward_center_only=(
                    self.lane_constraint_retry_relax_toward_center_only),
                taper_retry_over_horizon=(
                    self.lane_constraint_retry_taper_over_horizon),
                retry_terminal_ratio=(
                    self.lane_constraint_retry_terminal_ratio),
                connect_lane_from_current_pose=True,
                lane_connection_points=(
                    self.lane_constraint_connection_points),
                lane_transition_weights=self.lane_transition_weights)
        else:
            ref_wp_id = (self.model.wp_id + 1) % len(self.model.reference_path.path_constraints[0])
            ub = self.model.reference_path.path_constraints[0][ref_wp_id]
            lb = self.model.reference_path.path_constraints[1][ref_wp_id]
            self.model.reference_path.border_cells.current_wp_id = ref_wp_id

            # Update safety margin if provided as argument and different from current value
            if self.model.safety_margin != safety_margin:
                safety_margin_diff = safety_margin - self.model.safety_margin
                ub -= safety_margin_diff
                lb += safety_margin_diff

                infeasible_index = ub < lb
                ub[infeasible_index] = 0.0
                lb[infeasible_index] = 0.0

        lb, ub = apply_outer_boundary_guard(
            lb,
            ub,
            self._constraint_target_lane,
            self.prediction_outer_boundary_guard,
        )
        # A guard wider than an already narrowed corridor must never invert
        # the bounds passed to OSQP.  The zero-width sentinel makes the
        # problem safely infeasible and lets the existing recovery run.
        lb, ub = zero_inverted_bounds(lb, ub)

        # A zero-width, inverted, or non-finite corridor is not an OSQP
        # infeasibility sentinel: lb==ub==0 can be solved as the equality
        # e_y==0. Reject it explicitly before optimizer.solve().
        current_collapse = collapsed_constraint_snapshot(
            ub, lb, self._constraint_wp_ids)
        self._current_constraint_bounds_invalid = bool(
            current_collapse is not None)
        self._constraint_collapse_detail = retain_first_collapsed_constraint(
            self._constraint_collapse_detail,
            ub,
            lb,
            self._constraint_wp_ids,
        )
        self._constraint_collapse_detected = bool(
            self._constraint_collapse_detail is not None)

        # Update dynamic state constraints
        xmin_dyn[0] = xmax_dyn[0] = self.model.spatial_state.e_y
        #print("N =", N)
        #print("len(lb) =", len(lb))
        #print("xmin_dyn slice =", len(xmin_dyn[self.nx::self.nx]))
        
        xmin_dyn[self.nx::self.nx] = lb
        xmax_dyn[self.nx::self.nx] = ub
        xr[self.nx::self.nx] = (lb + ub) / 2
        self._prediction_lower_bounds = np.array(lb, copy=True)
        self._prediction_upper_bounds = np.array(ub, copy=True)
        # If a target lane is active, preserve lane-center targets for the e_y references.
        target_lane = getattr(self.model.reference_path, 'target_lane_idx', None)
        # Only a synchronized Hybrid reference may override an applied lane.
        # Existing L1/startup/offset precedence remains hard-lane-first.
        if target_lane is not None and not (
            self.lane_transition_weights is not None
            and self.soft_lateral_targets is not None
        ):
            lane_centers = []
            for n in range(N):
                lane_center = self._compute_lane_center(
                    self.model.wp_id + n, target_lane)
                if (
                    target_lane == 2
                    and self.target_lane_lateral_offsets is not None
                ):
                    lane_center += self.target_lane_lateral_offsets[
                        min(n, len(self.target_lane_lateral_offsets) - 1)]
                lane_centers.append(lane_center)
            xr[0:N*self.nx:self.nx] = lane_centers
            terminal_center = self._compute_lane_center(
                self.model.wp_id + N, target_lane)
            if (
                target_lane == 2
                and self.target_lane_lateral_offsets is not None
            ):
                terminal_center += self.target_lane_lateral_offsets[
                    min(N, len(self.target_lane_lateral_offsets) - 1)]
            xr[N * self.nx] = terminal_center
        elif (
            self.soft_target_lane_idx is not None
            or self.soft_lateral_targets is not None
        ):
            # Only xr changes here. lb/ub above remain the full-width corridor,
            # and P/Q stay fixed, so obstacles may still move the solution away
            # from L1 when required.
            for n in range(N):
                if self.soft_lateral_targets is not None:
                    lane_center = self.soft_lateral_targets[
                        min(n, len(self.soft_lateral_targets) - 1)]
                else:
                    lane_center = self._compute_lane_center(
                        self.model.wp_id + n, self.soft_target_lane_idx)
                    lane_center += self.soft_target_lateral_offset
                xr[n * self.nx] = blend_lateral_reference(
                    self.soft_target_start_e_y,
                    lane_center,
                    self.soft_target_alpha,
                )
            if self.soft_lateral_targets is not None:
                terminal_center = self.soft_lateral_targets[
                    min(N, len(self.soft_lateral_targets) - 1)]
            else:
                terminal_center = self._compute_lane_center(
                    self.model.wp_id + N, self.soft_target_lane_idx)
                terminal_center += self.soft_target_lateral_offset
            xr[N * self.nx] = blend_lateral_reference(
                self.soft_target_start_e_y,
                terminal_center,
                self.soft_target_alpha,
            )
        elif has_active_offset_limits(self.full_width_l0_offset_limits):
            # Bounds remain full width. Move only xr toward L0 by at most the
            # configured amount. Higher-priority soft/hard lane targets above
            # continue to own the objective while a manoeuvre is active.
            for n in range(N):
                midpoint = xr[n * self.nx]
                l0_center = self._compute_lane_center(
                    self.model.wp_id + n, 0)
                limit = max(float(self.full_width_l0_offset_limits[
                    min(n, len(self.full_width_l0_offset_limits) - 1)]), 0.0)
                xr[n * self.nx] = midpoint + np.clip(
                    l0_center - midpoint, -limit, limit)
            midpoint = xr[N * self.nx]
            l0_center = self._compute_lane_center(self.model.wp_id + N, 0)
            limit = max(float(self.full_width_l0_offset_limits[
                min(N, len(self.full_width_l0_offset_limits) - 1)]), 0.0)
            xr[N * self.nx] = midpoint + np.clip(
                l0_center - midpoint, -limit, limit)
        elif has_active_offset_limits(self.full_width_l1_offset_limits):
            # Bounds remain full width. Move only xr toward L1 by at most the
            # configured amount, avoiding a discontinuous lane-center jump.
            for n in range(N):
                midpoint = xr[n * self.nx]
                l1_center = self._compute_lane_center(
                    self.model.wp_id + n, 1)
                limit = max(float(self.full_width_l1_offset_limits[
                    min(n, len(self.full_width_l1_offset_limits) - 1)]), 0.0)
                xr[n * self.nx] = midpoint + np.clip(
                    l1_center - midpoint, -limit, limit)
            midpoint = xr[N * self.nx]
            l1_center = self._compute_lane_center(self.model.wp_id + N, 1)
            limit = max(float(self.full_width_l1_offset_limits[
                min(N, len(self.full_width_l1_offset_limits) - 1)]), 0.0)
            xr[N * self.nx] = midpoint + np.clip(
                l1_center - midpoint, -limit, limit)

        t_constraints = time.perf_counter()

        # Get equality matrix
        A_sparse = sparse.csc_matrix(
            (A_data, (self.row_A[:len(A_data)], self.col_A[:len(A_data)])),
            shape=(nx_N, nx_N)
        )
        Bu = sparse.csc_matrix((B_data, (self.row_B[:len(B_data)], self.col_B[:len(B_data)])), shape=(nx_N, nu_N))
        Ax = self.Ax_base + A_sparse

        Aeq = sparse.hstack([Ax, Bu])

        A_inequality = self.A_inequality     

        # 完全な制約行列
        A_full = sparse.vstack([Aeq, A_inequality], format='csc')

        t_matrix = time.perf_counter()

        # 境界制約の構築
        x0 = np.array(self.model.spatial_state[:])
        leq = np.hstack([-x0, uq])
        ueq = leq

        # 入力と状態の制約境界
        lineq_basic = np.hstack([xmin_dyn, np.kron(np.ones(N), umin)])
        uineq_basic = np.hstack([xmax_dyn, umax_dyn])

        # ステアリングレート制約の境界
        max_delta_change = self.max_steering_rate * self.model.Ts
        lineq_rate = -max_delta_change * np.ones(self.n_rate_constraints)
        uineq_rate = max_delta_change * np.ones(self.n_rate_constraints)

        t_constraints2 = time.perf_counter()

        # 全ての境界を結合
        l = np.hstack([leq, lineq_basic, lineq_rate])
        u = np.hstack([ueq, uineq_basic, uineq_rate])

        # コスト行列
        P = self.P_base

        q = np.hstack([
            -np.tile(np.diag(self.Q.toarray()), N) * xr[:-self.nx],
            -self.QN.dot(xr[-self.nx:]),
            -np.tile(np.diag(self.R.toarray()), N) * ur
        ])

        t_vector = time.perf_counter()

        # オプティマイザの設定
        if not self.osqp_initialized:
            # osqp_initialized=False でリセット後に既存インスタンスへ setup() を呼ぶと
            # "Workspace already setup!" エラーになるため、必ず新しいインスタンスを生成する。
            self.optimizer = osqp.OSQP()
            self.A0 = A_full.copy()
            self.optimizer.setup(P=P, q=q, A=A_full, l=l, u=u, warm_start=False, verbose=False)
            self.osqp_initialized = True

            
        else:
            #PはQ,R,QNが変わらないなら固定なので更新しない
            #qは参照(v_refとkappa_ref)によって毎回変わるので更新する
            #A_fullはLTVモデルの更新により毎回変わるためAxも更新する
            #print("A_full",A_full[:20,:20].toarray())
            #print("A0",self.A0[:20,:20].toarray())
            #print(np.max(np.abs(A_full.data - self.A_data_ref)),flush=True)

        
            #self.optimizer.update(q=q, l=l, u=u)
          
            self.optimizer.update(q=q, l=l, u=u, Ax=A_full.data)

        t_update = time.perf_counter()
        self.startup +=(t_pref-t_start)
        self.linearize +=(t_linearize-t_pref)
        self.path_constraints +=(t_constraints-t_linearize)
        self.sparse +=(t_matrix-t_constraints)
        self.constraints2 +=(t_constraints2-t_matrix)
        self.vector +=(t_vector-t_constraints2)
        self.update +=(t_update-t_vector)
        if self.debug_counter % 80 == 0:
            
            print(
                f"startup={self.startup*1000:.2f} "
                f"linearize={self.linearize*1000:.2f} "
                f"path_constraints={self.path_constraints*1000:.2f} "
                f"sparse={self.sparse*1000:.2f} "
                f"constraints2={self.constraints2*1000:.2f} "
                f"vector={self.vector*1000:.2f} "
                f"update={self.update*1000:.2f}",
                flush=True
            )
            


            # リセット
            self.startup = 0
            self.linearize = 0
            self.path_constraints = 0
            self.sparse = 0
            self.constraints2 = 0
            self.vector = 0
            self.update = 0
                        
    def get_control(self) -> Tuple[np.ndarray, float]:
        """
        Get control signal given the current position of the car.
        """
        nx = self.nx
        nu = self.nu
        self.used_prediction_fallback = False
        self.time_budget_exceeded = False
        self.recovery_requested = False
        self.failure_reason = None
        self.last_solution_status = None
        self.last_solution_accurate = False
        self._constraint_collapse_detected = False
        self._constraint_collapse_detail = None
        self._current_constraint_bounds_invalid = False

        #最近傍Waypointを取得
        self.model.get_current_waypoint()

        N = min(self.N, self.model.reference_path.n_waypoints - self.model.wp_id) \
            if not self.model.reference_path.circular else self.N
        #世界座標を経路座標へ変換
        self.model.spatial_state = self.model.t2s(
            reference_state=self.model.temporal_state,
            reference_waypoint=self.model.current_waypoint)

        t0 = time.perf_counter()

        base_wp_id = self.model.wp_id
        self._init_problem(N, self.model.safety_margin)

        t1 = time.perf_counter()
        t2 = t1

        # Preserve last prediction as fallback when the solver temporarily fails
        prediction_backup = self.current_prediction

        try:

            dec = None
            if not self._current_constraint_bounds_invalid:
                dec = self.optimizer.solve()
            if self.debug_counter % 20 == 0:
                print(
                    "invalid constraint bounds"
                    if dec is None else dec.info.status,
                    flush=True,
                )
            t2 = time.perf_counter()

            if self._current_constraint_bounds_invalid or is_primal_infeasible(dec):
                # Limit only the additional relaxed retries. The initial
                # problem build and solve are normal MPC work and are not
                # included in this deadline.
                retry_started_at = time.perf_counter()
                retry_steps = tuple(
                    max(float(value), 0.0)
                    for value in self.lane_constraint_retry_relaxation_m)
                # A small first retry is useless when the current vehicle
                # center is already far outside the selected artificial lane,
                # and one solve may consume the whole retry budget. Start at
                # the first configured step that can cover the measured gap.
                minimum_relaxation = 0.0
                if self._constraint_target_lane in (0, 1, 2):
                    lanes = self.model.reference_path.get_lane_bounds(
                        self.model.wp_id + 1)
                    if (
                        lanes
                        and self._constraint_target_lane < len(lanes)
                    ):
                        lane_ub, lane_lb = lanes[
                            self._constraint_target_lane]
                        current_e_y = float(self.model.spatial_state.e_y)
                        minimum_relaxation = max(
                            float(lane_lb) - current_e_y,
                            current_e_y - float(lane_ub),
                            0.0,
                        ) + 0.10
                eligible_steps = tuple(
                    value for value in retry_steps
                    if value + 1e-9 >= minimum_relaxation)
                if eligible_steps:
                    retry_steps = eligible_steps
                elif retry_steps:
                    retry_steps = (max(retry_steps),)
                for lane_relaxation in retry_steps:
                    if lane_relaxation <= 0.0:
                        continue
                    if self._constraint_target_lane not in (0, 1, 2):
                        break
                    elapsed_ms = (
                        time.perf_counter() - retry_started_at) * 1000.0
                    if elapsed_ms >= self.solve_time_budget_ms:
                        self.time_budget_exceeded = True
                        break
                    # _init_problem applies wp_id_offset, so restore the
                    # unshifted waypoint before every retry.
                    self.model.wp_id = base_wp_id
                    self._init_problem(
                        N, self.model.safety_margin,
                        lane_relaxation=lane_relaxation)
                    if self._current_constraint_bounds_invalid:
                        dec = None
                        continue
                    dec = self.optimizer.solve()
                    t2 = time.perf_counter()

                    if is_valid_osqp_solution(dec):
                        if self.last_solved_wp_id != self.model.wp_id:
                            print(
                                "[LaneConstraintRetry] solved with "
                                f"lane_relaxation={lane_relaxation:.2f}m",
                                flush=True,
                            )
                        break
                    if not is_primal_infeasible(dec):
                        break

            if not is_valid_osqp_solution(dec):
                if self._current_constraint_bounds_invalid:
                    detail = self._constraint_collapse_detail or {}
                    raise ValueError(
                        "invalid/collapsed MPC constraint bounds: "
                        f"wp={detail.get('wp')}, "
                        f"width={detail.get('width')}"
                    )
                if self.time_budget_exceeded:
                    raise ValueError(
                        "MPC retry time budget exceeded "
                        f"({self.solve_time_budget_ms:.1f}ms)")
                raise ValueError(
                    f"OSQP failed with status '{dec.info.status}'")

            self.last_solution_status = str(dec.info.status)
            solution_is_accurate = bool(
                dec.info.status_val == osqp.constant('OSQP_SOLVED'))

            control_signals = np.array(dec.x[-N*nu:])

            # ステア角の計算と保存
            control_signals[1::2] = np.arctan(control_signals[1::2] * self.model.length)
            x = np.reshape(dec.x[:(N+1)*nx], (N+1, nx))
            candidate_prediction = self.update_prediction(x, N)
            if not is_plausible_mpc_prediction(
                x,
                candidate_prediction,
                (self.model.temporal_state.x, self.model.temporal_state.y),
                self._prediction_lower_bounds,
                self._prediction_upper_bounds,
                lateral_tolerance=self.prediction_lateral_tolerance,
            ):
                raise ValueError("OSQP returned an implausible prediction")

            v = control_signals[0]
            delta = control_signals[1]

            # ステアレートの制限を適用
            max_delta_change = self.max_steering_rate * self.model.Ts
            delta = np.clip(
                delta,
                self.previous_steering - max_delta_change,
                self.previous_steering + max_delta_change
            )

            self.previous_steering = delta

            # Commit the candidate only after solver and geometry validation.
            self.current_control = control_signals
            self.current_prediction = candidate_prediction
            self.collision_prediction_context = (
                candidate_prediction, self.update_prediction(x,N,start_index=0))
            self.last_solution_accurate = solution_is_accurate

            u = np.array([v, delta])
            max_delta = np.max(np.abs(control_signals[1:len(control_signals)//3*2:2]))

            if self.infeasibility_counter > (N - 1):
                print(f'Problem solved after {self.infeasibility_counter} infeasible iterations')
            self.infeasibility_counter = 0
            self.last_solved_wp_id = self.model.wp_id

        except (TypeError, ValueError) as error:
            self.failure_reason = str(error)
            if self.debug_counter % 20 == 0:
                print(f"[MPCFallback] {error}", flush=True)
            failure_cycle = self.infeasibility_counter + 1
            fallback_id = nu * failure_cycle
            fallback_prediction_valid = (
                prediction_backup is not None
                and is_plausible_world_prediction(
                    prediction_backup,
                    (self.model.temporal_state.x, self.model.temporal_state.y),
                )
            )
            fallback_control_valid = (
                fallback_id + 2 <= len(self.current_control)
                and not np.all(
                    self.current_control[
                        fallback_id:fallback_id + 2] == 0.0)
            )
            fallback_allowed = can_reuse_prediction_fallback(
                failure_cycle,
                self.max_prediction_fallback_cycles,
                fallback_prediction_valid,
                fallback_control_valid,
            )

            if fallback_allowed:
                u = np.array(
                    self.current_control[fallback_id:fallback_id + 2])
                max_delta = np.abs(u[1])
                self.current_prediction = prediction_backup
                self.used_prediction_fallback = True
            else:
                # Do not drive indefinitely on an old prediction. Preserve
                # steering continuity while commanding a full stop.
                u = np.array([0.0, self.previous_steering])
                max_delta = np.abs(self.previous_steering)
                self.current_prediction = None
                self.current_control = np.zeros_like(self.current_control)
                self.recovery_requested = True

            self.infeasibility_counter += 1

        if self.infeasibility_counter > (N - 1) and self.infeasibility_counter % 100 == 0:
            now = datetime.now().strftime("%H:%M:%S.%f")
            print('No control signal computed!')
            print(now)

        self.debug_counter += 1

        '''
        if self.debug_counter % 20 == 0:    
                    print(
                        f"status={dec.info.status} "
                        f"v={v:.3f} "
                        f"delta={delta:.3f}",
                        flush=True
                    )
        '''
        
        if self.debug_counter % 80 == 0:
            now = datetime.now().strftime("%H:%M:%S.%f")
            total_ms = (t2-t0)*1000
            print(
                f"N={N} "
                f"build={(t1-t0)*1000:.1f}ms "
                f"solve={(t2-t1)*1000:.1f}ms "
                f"total={total_ms:.1f}ms "
                f"target={1000*self.model.Ts:.1f}ms "
                f"waypoint={self.model.wp_id} "
                f"time={now}",
                flush=True
            )
        

        return u, max_delta

    def update_prediction(self, spatial_state_prediction, N, start_index=2):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(start_index, N):
            # Get associated waypoint
            associated_waypoint = self.model.reference_path.\
                get_waypoint(self.model.wp_id+n)
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[n, :])

            # Save predicted coordinates in world coordinate frame
            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred

    def show_prediction(self, ax):
        """
        Display predicted car trajectory on the provided axis.
        :param ax: Matplotlib axis object to plot on
        """

        if self.current_prediction is not None:
            # ax.scatter(self.current_prediction[0], self.current_prediction[1],
            #            c=PREDICTION, s=5)
            ax.plot(self.current_prediction[0], self.current_prediction[1], c=PREDICTION)
