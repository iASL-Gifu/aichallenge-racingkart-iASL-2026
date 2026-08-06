from numpy.core import function_base
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
from multi_purpose_mpc_ros.fuzzy_weight_adapter import FuzzyWeightAdapter

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
    lateral_tolerance=0.02,
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
    lower_bounds,
    upper_bounds,
    target_lane,
    guard_margin,
):
    """Keep the MPC corridor away from the two physical course edges.

    L0 touches the lower/right course edge and L2 touches the upper/left
    edge. Full-width/Race driving touches both. L1 has only internal lane
    edges, so it does not receive this additional wall guard.
    """
    lower = np.asarray(lower_bounds, dtype=float).copy()
    upper = np.asarray(upper_bounds, dtype=float).copy()
    guard = max(float(guard_margin), 0.0)
    if target_lane is None:
        lower += guard
        upper -= guard
    elif target_lane == 0:
        lower += guard
    elif target_lane == 2:
        upper -= guard
    return lower, upper


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

##################
# MPC Controller #
##################

class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, max_steering_rate, wp_id_offset, use_obstacle_avoidance,
                 use_path_constraints_topic, use_max_kappa_pred=True,
                 understeer_coeff=0.0):
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
        self.soft_lateral_targets = None

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
        # The ordinary rate rows above constrain only adjacent predicted
        # inputs.  Keep a structurally fixed row for the transition from the
        # steering angle that is currently being applied to the first MPC
        # input.  It is disabled for normal MPCs and enabled for the outer-lane
        # shadow probe, so the shadow prediction is generated from a command
        # the live controller can actually reach in one control period.
        initial_steering_rate_matrix = sparse.lil_matrix(
            (1, self.nx_N + self.nu_N)
        )
        initial_steering_rate_matrix[0, self.nx_N + 1] = 1.0
        self.initial_steering_rate_matrix = (
            initial_steering_rate_matrix.tocsc()
        )
        #sparse.eyeの固定`
        self.basic_constraint_matrix = sparse.eye(
            self.nx_N + self.nu_N,
            format='csc'
        )
        self.A_inequality = sparse.vstack([
            self.basic_constraint_matrix,
            self.initial_steering_rate_matrix,
            self.steering_rate_matrix
        ], format='csc')        
        

        #Axのsparse.eye()固定
        self.Ax_base = sparse.kron(
            sparse.eye(self.N + 1),
            -sparse.eye(self.nx),
            format='csc'
        )
        # Cost-matrix sparsity is fixed once. Fuzzy adaptation updates only
        # this matrix's numeric CSC data through OSQP Px updates.
        self._base_Q_diag = np.asarray(self.Q.diagonal(), dtype=float).copy()
        self._base_R_diag = np.asarray(self.R.diagonal(), dtype=float).copy()
        self._base_QN_diag = np.asarray(self.QN.diagonal(), dtype=float).copy()
        self._active_Q_diag = self._base_Q_diag.copy()
        self._fuzzy_weight_adapter = None
        self._fuzzy_steer_delta_max_weight = 0.0
        self._active_steer_delta_weight = 0.0
        self._cost_constant_data = None
        self._cost_q_lateral_data = None
        self._cost_q_heading_data = None
        self._cost_steer_delta_data = None
        self._cost_values_dirty = False
        self._rebuild_cost_matrix_structure()

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
        self.enforce_initial_steering_rate_constraint = False

        # 追加: ay_maxによる速度制限の方式切り替え
        self.use_max_kappa_pred = use_max_kappa_pred
        # 既存の初期化
        self.current_prediction = None
        self.infeasibility_counter = 0
        self.solve_time_budget_ms = 20.0
        self.max_prediction_fallback_cycles = 3
        # OUTER_COURSE_MARGIN is already included in the static bounds. This
        # small extra guard prevents accepted predictions from riding exactly
        # on that boundary, while the tolerance only absorbs solver noise.
        self.prediction_outer_boundary_guard = 0.10
        self.prediction_lateral_tolerance = 0.02
        self.used_prediction_fallback = False
        self.time_budget_exceeded = False
        self.recovery_requested = False
        self.failure_reason = None
        # Ordinary control may accept OSQP_SOLVED_INACCURATE, but recovery
        # after reverse requires an exact OSQP_SOLVED result.
        self.last_solution_status = None
        self.last_solution_accurate = False
        self.last_compute_time_ms = 0.0
        self.last_build_time_ms = 0.0
        self.soft_target_lane_idx = None
        self.soft_target_start_e_y = 0.0
        self.soft_target_alpha = 0.0
        self.soft_target_lateral_offset = 0.0
        self.soft_lateral_targets = None
        # Snapshot of the exact corridor used by the latest solve attempt.
        # These values remain available after an infeasible solve so the
        # controller can diagnose lane-bound and obstacle-induced failures.
        self._constraint_wp_ids = np.array([], dtype=int)
        self._constraint_target_lane = None
        self._constraint_safety_margin = 0.0
        self._path_constraints_valid = True
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

    def set_initial_steering_rate_constraint(self, enabled: bool) -> None:
        """Constrain the first predicted steer to one reachable control step."""
        self.enforce_initial_steering_rate_constraint = bool(enabled)

    def update_Q(self, Q: np.ndarray):
        self.Q = Q
        self._base_Q_diag = np.asarray(Q.diagonal(), dtype=float).copy()
        self._rebuild_cost_matrix_structure()
        self.osqp_initialized = False

    def update_R(self, R: np.ndarray):
        self.R = R
        self._base_R_diag = np.asarray(R.diagonal(), dtype=float).copy()
        self._rebuild_cost_matrix_structure()
        self.osqp_initialized = False

    def update_QN(self, QN: np.ndarray):
        self.QN = QN
        self._base_QN_diag = np.asarray(QN.diagonal(), dtype=float).copy()
        self._rebuild_cost_matrix_structure()
        self.osqp_initialized = False

    @staticmethod
    def _component_data_on_pattern(component, pattern):
        """Align a sparse component with a fixed CSC matrix data order."""
        aligned = np.zeros_like(pattern.data, dtype=float)
        positions = {}
        for col in range(pattern.shape[1]):
            for data_index in range(pattern.indptr[col], pattern.indptr[col + 1]):
                positions[(int(pattern.indices[data_index]), col)] = data_index
        component = sparse.triu(component, format='coo')
        for row, col, value in zip(component.row, component.col, component.data):
            data_index = positions.get((int(row), int(col)))
            if data_index is not None:
                aligned[data_index] = float(value)
        return aligned

    def _rebuild_cost_matrix_structure(self) -> None:
        """Build the fixed P sparsity and cache its four numeric components."""
        variable_count = self.nx_N + self.nu_N
        constant_diag = np.concatenate([
            np.tile(
                np.asarray([0.0, 0.0, self._base_Q_diag[2]]),
                self.N,
            ),
            self._base_QN_diag,
            np.tile(self._base_R_diag, self.N),
        ])
        constant = sparse.diags(
            constant_diag, shape=(variable_count, variable_count), format='csc')

        q_lateral = sparse.lil_matrix((variable_count, variable_count))
        q_heading = sparse.lil_matrix((variable_count, variable_count))
        for step in range(self.N):
            q_lateral[step * self.nx, step * self.nx] = self._base_Q_diag[0]
            q_heading[
                step * self.nx + 1, step * self.nx + 1
            ] = self._base_Q_diag[1]
        q_lateral = q_lateral.tocsc()
        q_heading = q_heading.tocsc()

        # 0.5*w*(kappa[0]-kappa_previous)^2 plus the adjacent increments.
        # The previous-curvature linear term is added to q in _init_problem.
        steer_delta = sparse.lil_matrix((variable_count, variable_count))
        steering_indices = [
            self.nx_N + step * self.nu + 1 for step in range(self.N)
        ]
        if steering_indices:
            steer_delta[steering_indices[0], steering_indices[0]] += 1.0
        for previous, current in zip(steering_indices[:-1], steering_indices[1:]):
            steer_delta[previous, previous] += 1.0
            steer_delta[current, current] += 1.0
            steer_delta[previous, current] -= 1.0
            steer_delta[current, previous] -= 1.0
        steer_delta = steer_delta.tocsc()

        include_delta = (
            self._fuzzy_weight_adapter is not None
            and self._fuzzy_steer_delta_max_weight > 0.0
        )
        pattern_matrix = constant + q_lateral + q_heading
        if include_delta:
            # A positive coefficient is guaranteed by the configured floor,
            # so these off-diagonal entries remain in OSQP's fixed structure.
            pattern_matrix = (
                pattern_matrix
                + self._fuzzy_steer_delta_max_weight * steer_delta
            )
        self.P_base = sparse.triu(pattern_matrix, format='csc')
        self.P_base.sort_indices()

        self._cost_constant_data = self._component_data_on_pattern(
            constant, self.P_base)
        self._cost_q_lateral_data = self._component_data_on_pattern(
            q_lateral, self.P_base)
        self._cost_q_heading_data = self._component_data_on_pattern(
            q_heading, self.P_base)
        self._cost_steer_delta_data = self._component_data_on_pattern(
            steer_delta, self.P_base)
        self._active_Q_diag = self._base_Q_diag.copy()
        self._active_steer_delta_weight = (
            self._fuzzy_steer_delta_max_weight if include_delta else 0.0)
        self._apply_cost_matrix_values(1.0, 1.0, 1.0)

    def _apply_cost_matrix_values(
        self, q_lateral_ratio, q_heading_ratio, steer_delta_ratio
    ) -> None:
        q_lateral_ratio = float(q_lateral_ratio)
        q_heading_ratio = float(q_heading_ratio)
        steer_delta_ratio = float(steer_delta_ratio)
        self._active_Q_diag = self._base_Q_diag.copy()
        self._active_Q_diag[0] *= q_lateral_ratio
        self._active_Q_diag[1] *= q_heading_ratio
        self._active_steer_delta_weight = (
            self._fuzzy_steer_delta_max_weight * steer_delta_ratio
            if self._fuzzy_weight_adapter is not None else 0.0
        )
        self.P_base.data[:] = (
            self._cost_constant_data
            + q_lateral_ratio * self._cost_q_lateral_data
            + q_heading_ratio * self._cost_q_heading_data
            + self._active_steer_delta_weight * self._cost_steer_delta_data
        )
        self._cost_values_dirty = True

    def configure_fuzzy_weights(
        self,
        *,
        enabled,
        lateral_error_full_scale,
        heading_error_full_scale_rad,
        q_lateral_min_ratio,
        q_heading_min_ratio,
        steer_delta_min_ratio,
        steer_delta_max_weight,
        smoothing_sec,
    ) -> None:
        """Configure lightweight fuzzy weights before the first OSQP setup."""
        if not enabled:
            self._fuzzy_weight_adapter = None
            self._fuzzy_steer_delta_max_weight = 0.0
        else:
            self._fuzzy_weight_adapter = FuzzyWeightAdapter(
                lateral_error_full_scale=lateral_error_full_scale,
                heading_error_full_scale_rad=heading_error_full_scale_rad,
                q_lateral_min_ratio=q_lateral_min_ratio,
                q_heading_min_ratio=q_heading_min_ratio,
                steer_delta_min_ratio=steer_delta_min_ratio,
                smoothing_sec=smoothing_sec,
                control_period_sec=self.model.Ts,
            )
            self._fuzzy_steer_delta_max_weight = max(
                float(steer_delta_max_weight), 0.0)
        self._rebuild_cost_matrix_structure()
        self.osqp_initialized = False

    def _update_fuzzy_cost(self) -> None:
        if self._fuzzy_weight_adapter is None:
            return
        ratios = self._fuzzy_weight_adapter.update(
            self.model.spatial_state.e_y,
            self.model.spatial_state.e_psi,
        )
        self._apply_cost_matrix_values(
            ratios.q_lateral,
            ratios.q_heading,
            ratios.steer_delta,
        )

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
        explicit_targets = (
            None if lateral_targets is None
            else np.asarray(lateral_targets, dtype=float).reshape(-1).copy()
        )
        self.soft_lateral_targets = (
            explicit_targets
            if explicit_targets is not None and explicit_targets.size > 0
            else None
        )

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

    def _init_problem(self, N, safety_margin):
        """
        Initialize optimization problem for current time step with steering rate constraints.
        """

        t_start = time.perf_counter()
        
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
        if self.use_obstacle_avoidance and not self.use_path_constraints_topic:
            ub, lb, _ = self.model.reference_path.update_path_constraints(
                self.model.wp_id + 1,
                [self.model.temporal_state.x, self.model.temporal_state.y, self.model.temporal_state.psi],
                N, self.model.length, self.model.width, safety_margin)
            self._path_constraints_valid = bool(
                self.model.reference_path.last_constraints_valid)
        else:
            ref_wp_id = (self.model.wp_id + 1) % len(self.model.reference_path.path_constraints[0])
            ub = self.model.reference_path.path_constraints[0][ref_wp_id]
            lb = self.model.reference_path.path_constraints[1][ref_wp_id]
            self.model.reference_path.border_cells.current_wp_id = ref_wp_id
            self._path_constraints_valid = True

            # Update safety margin if provided as argument and different from current value
            if self.model.safety_margin != safety_margin:
                safety_margin_diff = safety_margin - self.model.safety_margin
                ub -= safety_margin_diff
                lb += safety_margin_diff

                infeasible_index = ub < lb
                if np.any(infeasible_index):
                    self._path_constraints_valid = False
                    invalid_wp_ids = getattr(
                        self.model.reference_path,
                        'invalid_constraint_wp_ids',
                        None,
                    )
                    if invalid_wp_ids is not None:
                        for index in np.flatnonzero(infeasible_index):
                            invalid_wp_id = int(
                                (ref_wp_id + int(index))
                                % self.model.reference_path.n_waypoints)
                            if invalid_wp_id not in invalid_wp_ids:
                                invalid_wp_ids.append(invalid_wp_id)

        lb, ub = apply_outer_boundary_guard(
            lb,
            ub,
            self._constraint_target_lane,
            self.prediction_outer_boundary_guard,
        )

        # Validate the final bounds, including the extra physical-edge guard,
        # before passing them to OSQP.  update_bounds() raises an uncaught
        # ValueError when even one lower bound exceeds its upper bound.  Treat
        # non-finite, mismatched, or inverted bounds as an ordinary infeasible
        # MPC cycle so the controller can enter SafetyRecovery instead.
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        bounds_shape_valid = (
            lb.shape == ub.shape
            and lb.ndim == 1
            and lb.size == N
        )
        if bounds_shape_valid:
            invalid_bounds = (
                ~np.isfinite(lb)
                | ~np.isfinite(ub)
                | (lb > ub)
            )
        else:
            invalid_bounds = np.ones(max(lb.size, ub.size, 1), dtype=bool)

        if np.any(invalid_bounds):
            self._path_constraints_valid = False
            invalid_indices = np.flatnonzero(invalid_bounds)
            invalid_wp_ids = [
                int((self.model.wp_id + 1 + int(index))
                    % self.model.reference_path.n_waypoints)
                for index in invalid_indices[:5]
            ]
            reference_invalid_wp_ids = getattr(
                self.model.reference_path, 'invalid_constraint_wp_ids', None)
            if reference_invalid_wp_ids is not None:
                for invalid_wp_id in invalid_wp_ids:
                    if invalid_wp_id not in reference_invalid_wp_ids:
                        reference_invalid_wp_ids.append(invalid_wp_id)
            raise ValueError(
                "Invalid MPC path constraint bounds after outer boundary "
                f"guard at waypoints {invalid_wp_ids}")

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
        if target_lane is not None:
            lane_centers = []
            for n in range(N):
                lane_centers.append(self._compute_lane_center(self.model.wp_id + n, target_lane))
            xr[0:N*self.nx:self.nx] = lane_centers
        elif (
            self.soft_target_lane_idx is not None
            or self.soft_lateral_targets is not None
        ):
            # Only xr changes here. lb/ub above remain the full-width corridor,
            # and P/Q stay fixed, so obstacles may still move the solution away
            # from L1 when required.
            for n in range(N):
                if self.soft_lateral_targets is not None:
                    target_index = min(n, len(self.soft_lateral_targets) - 1)
                    lane_center = self.soft_lateral_targets[target_index]
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
                terminal_index = min(N, len(self.soft_lateral_targets) - 1)
                terminal_center = self.soft_lateral_targets[terminal_index]
            else:
                terminal_center = self._compute_lane_center(
                    self.model.wp_id + N, self.soft_target_lane_idx)
                terminal_center += self.soft_target_lateral_offset
            xr[N * self.nx] = blend_lateral_reference(
                self.soft_target_start_e_y,
                terminal_center,
                self.soft_target_alpha,
            )

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
        if self.enforce_initial_steering_rate_constraint:
            # The optimizer's second input is curvature.  Convert the
            # reachable steering-angle interval back to curvature before
            # applying it to the first input variable.
            delta_lower = self.previous_steering - max_delta_change
            delta_upper = self.previous_steering + max_delta_change
            kappa_lower = np.tan(delta_lower) / self.model.length
            kappa_upper = np.tan(delta_upper) / self.model.length
            lineq_initial_rate = np.array([
                np.clip(kappa_lower, umin[1], umax[1])
            ])
            uineq_initial_rate = np.array([
                np.clip(kappa_upper, umin[1], umax[1])
            ])
        else:
            lineq_initial_rate = np.array([-np.inf])
            uineq_initial_rate = np.array([np.inf])
        lineq_rate = -max_delta_change * np.ones(self.n_rate_constraints)
        uineq_rate = max_delta_change * np.ones(self.n_rate_constraints)

        t_constraints2 = time.perf_counter()

        # 全ての境界を結合
        l = np.hstack([
            leq, lineq_basic, lineq_initial_rate, lineq_rate
        ])
        u = np.hstack([
            ueq, uineq_basic, uineq_initial_rate, uineq_rate
        ])

        # コスト行列
        P = self.P_base

        q = np.hstack([
            -np.tile(self._active_Q_diag, N) * xr[:-self.nx],
            -self.QN.dot(xr[-self.nx:]),
            -np.tile(self._base_R_diag, N) * ur
        ])
        if self._active_steer_delta_weight > 0.0:
            previous_kappa = (
                np.tan(self.previous_steering) / self.model.length)
            q[self.nx_N + 1] -= (
                self._active_steer_delta_weight * previous_kappa)

        t_vector = time.perf_counter()

        # オプティマイザの設定
        if not self.osqp_initialized:
            # osqp_initialized=False でリセット後に既存インスタンスへ setup() を呼ぶと
            # "Workspace already setup!" エラーになるため、必ず新しいインスタンスを生成する。
            self.optimizer = osqp.OSQP()
            self.A0 = A_full.copy()
            self.optimizer.setup(P=P, q=q, A=A_full, l=l, u=u, warm_start=False, verbose=False)
            self.osqp_initialized = True
            self._cost_values_dirty = False

            
        else:
            #PはQ,R,QNが変わらないなら固定なので更新しない
            #qは参照(v_refとkappa_ref)によって毎回変わるので更新する
            #A_fullはLTVモデルの更新により毎回変わるためAxも更新する
            #print("A_full",A_full[:20,:20].toarray())
            #print("A0",self.A0[:20,:20].toarray())
            #print(np.max(np.abs(A_full.data - self.A_data_ref)),flush=True)

        
            #self.optimizer.update(q=q, l=l, u=u)
          
            update_arguments = {
                "q": q,
                "l": l,
                "u": u,
                "Ax": A_full.data,
            }
            if (
                self._fuzzy_weight_adapter is not None
                and self._cost_values_dirty
            ):
                update_arguments["Px"] = P.data
            self.optimizer.update(**update_arguments)
            self._cost_values_dirty = False

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

        #最近傍Waypointを取得
        self.model.get_current_waypoint()

        N = min(self.N, self.model.reference_path.n_waypoints - self.model.wp_id) \
            if not self.model.reference_path.circular else self.N
        #世界座標を経路座標へ変換
        self.model.spatial_state = self.model.t2s(
            reference_state=self.model.temporal_state,
            reference_waypoint=self.model.current_waypoint)
        self._update_fuzzy_cost()

        t0 = time.perf_counter()

        # Preserve last prediction as fallback when the solver temporarily fails
        prediction_backup = self.current_prediction
        base_wp_id = self.model.wp_id
        t1 = t0
        t2 = t1

        try:
            self._init_problem(N, self.model.safety_margin)
            t1 = time.perf_counter()
            t2 = t1

            if not self._path_constraints_valid:
                invalid_wp_ids = getattr(
                    self.model.reference_path,
                    'invalid_constraint_wp_ids',
                    [],
                )
                raise ValueError(
                    "No obstacle-free path corridor wide enough for vehicle "
                    f"at waypoints {invalid_wp_ids[:5]}")

            dec = self.optimizer.solve()
            t2 = time.perf_counter()

            if self.debug_counter % 20 == 0:
                print(dec.info.status, flush=True)

            if is_primal_infeasible(dec):
                # Limit only the additional relaxed retries. The initial
                # problem build and solve are normal MPC work and are not
                # included in this deadline.
                retry_started_at = time.perf_counter()
                for i in range(1, 6):
                    elapsed_ms = (
                        time.perf_counter() - retry_started_at) * 1000.0
                    if elapsed_ms >= self.solve_time_budget_ms:
                        self.time_budget_exceeded = True
                        break
                    relaxed_safety_margin = self.model.safety_margin * ((5-i) / 5.0)
                    # _init_problem applies wp_id_offset, so restore the
                    # unshifted waypoint before every retry.
                    self.model.wp_id = base_wp_id
                    self._init_problem(N, relaxed_safety_margin)
                    dec = self.optimizer.solve()
                    t2 = time.perf_counter()

                    if is_valid_osqp_solution(dec):
                        if self.last_solved_wp_id != self.model.wp_id:
                            print(
                                f"Relaxed safety margin by {relaxed_safety_margin} "
                                f"({5-i}/5) to solve the problem")
                        break
                    if not is_primal_infeasible(dec):
                        break

            if not is_valid_osqp_solution(dec):
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
            self.last_solution_accurate = solution_is_accurate

            u = np.array([v, delta])
            max_delta = np.max(np.abs(control_signals[1:len(control_signals)//3*2:2]))

            if self.infeasibility_counter > (N - 1):
                print(
                    f'Problem solved after {self.infeasibility_counter} '
                    'infeasible iterations')

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
                fallback_prediction_valid and self._path_constraints_valid,
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

        if (
            self.infeasibility_counter > (N - 1)
            and self.infeasibility_counter % 100 == 0
        ):
            now = datetime.now().strftime("%H:%M:%S.%f")
            print('No control signal computed!')
            print(now)

        self.debug_counter += 1
        self.last_build_time_ms = max((t1 - t0) * 1000.0, 0.0)
        self.last_compute_time_ms = max(
            (time.perf_counter() - t0) * 1000.0, 0.0)

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

    def update_prediction(self, spatial_state_prediction, N):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(2, N):
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
