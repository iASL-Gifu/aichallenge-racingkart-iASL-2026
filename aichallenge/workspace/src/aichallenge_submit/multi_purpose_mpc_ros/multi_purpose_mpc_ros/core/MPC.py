from numpy.core import function_base
import decimal
import decimal
from numpy import typing
from typing import Tuple
import math
import numpy as np
import osqp
from scipy import sparse
import time
from datetime import datetime

from multi_purpose_mpc_ros.core.spatial_bicycle_models import (
    understeer_curvature_gain,
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


def steering_preview_index(
    distances,
    steering_refs,
    start_index,
    speed,
    delay_sec,
    steering_rate,
    max_preview_distance,
) -> int:
    """Select a future steering reference early enough for the actuator.

    The selection accounts for both command delay and the time needed to slew
    from the reference at ``start_index`` to each future candidate.
    """
    distance_array = np.asarray(distances, dtype=float)
    steering_array = np.asarray(steering_refs, dtype=float)
    if distance_array.size == 0 or distance_array.size != steering_array.size:
        return int(start_index)
    start = int(np.clip(start_index, 0, distance_array.size - 1))
    if speed <= 0.0 or steering_rate <= 0.0:
        return start
    preview_limit = max(float(max_preview_distance), 0.0)
    selected = start
    for candidate in range(start + 1, distance_array.size):
        candidate_distance = distance_array[candidate] - distance_array[start]
        if candidate_distance > preview_limit:
            break
        angle_change = abs(
            steering_array[candidate] - steering_array[start])
        required_start_distance = max(float(speed), 0.0) * (
            max(float(delay_sec), 0.0) + angle_change / steering_rate)
        # This future target has reached the point at which steering must
        # already start. Select the furthest such target within the bounded
        # preview so a sharp change just beyond the pure-delay point is not
        # missed while traversing a straight segment.
        if candidate_distance <= required_start_distance:
            selected = candidate
    return selected


def steering_reachability_speed_cap(
    distance,
    angle_change,
    delay_sec,
    steering_rate,
    minimum_angle_change=0.03,
) -> float:
    """Return speed that leaves enough time for delay plus steering slew."""
    if (
        distance <= 0.0
        or steering_rate <= 0.0
        or abs(angle_change) <= max(float(minimum_angle_change), 0.0)
    ):
        return math.inf
    required_time = (
        max(float(delay_sec), 0.0)
        + abs(float(angle_change)) / float(steering_rate)
    )
    return float(distance) / max(required_time, 1e-6)


def build_arc_length_steering_reservation(
    distances,
    steering_refs,
    speeds,
    delay_sec,
    steering_rate,
    current_steering,
    command_period,
):
    """Build a delayed, rate-feasible steering schedule along arc length."""
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
            distance_array, target_distance, side='left'))
        target_index = min(max(target_index, index), count - 1)
        delayed[index] = reference_array[target_index]

    # Backward propagation reserves the latest feasible start of each future
    # steering change instead of waiting for lateral error at the corner.
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

    # Anchor the first scheduled command to the command actually owned by the
    # actuator, then keep the complete schedule rate-feasible going forward.
    previous = float(current_steering)
    for index in range(count):
        if index == 0:
            max_change = (
                float(steering_rate) * max(float(command_period), 0.0))
        else:
            ds = max(float(
                distance_array[index] - distance_array[index - 1]), 0.0)
            speed = max(float(speed_array[index - 1]), 0.5)
            max_change = float(steering_rate) * ds / speed
        reserved[index] = float(np.clip(
            reserved[index], previous - max_change, previous + max_change))
        previous = reserved[index]
    return reserved


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
    lower_bounds,
    upper_bounds,
    target_lane,
    guard_margin,
):
    """Inset only the physical course edge(s) used by the active corridor.

    L0 touches the lower/right physical edge and L2 touches the upper/left
    edge.  Full-width/Race driving touches both.  L1 is bounded only by
    internal lane boundaries, so applying this wall guard there would narrow
    an already constrained rejoin corridor without adding wall clearance.

    Always copy the arrays: the non-obstacle path can hand us views into the
    reference path's cached constraints, which must not be modified in place.
    """
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
                 understeer_coeff=0.0, use_steering_state=False,
                 steering_state_weight=1.0e6,
                 terminal_steering_state_weight=1.0e6,
                 steering_preview_enabled=True,
                 steering_command_delay=0.15,
                 steering_preview_max_distance=6.0,
                 steering_reservation_enabled=False,
                 steering_reachability_speed_enabled=True,
                 steering_reachability_min_speed=2.0,
                 steering_reachability_min_angle=0.03):
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
        self.model = model #車両モデル
        self.uses_steering_state = bool(use_steering_state)
        self.model_nx = self.model.n_states
        if self.uses_steering_state:
            self.Q = sparse.block_diag([
                Q, sparse.csc_matrix([[float(steering_state_weight)]])
            ], format='csc')
            self.QN = sparse.block_diag([
                QN,
                sparse.csc_matrix(
                    [[float(terminal_steering_state_weight)]])
            ], format='csc')
        else:
            self.Q = Q
            self.QN = QN
        self.R = R #入力コスト
        #R =[[1,0],[0,20]]なら速度変更は許す、操舵変更は嫌う
        self.wp_id_offset = wp_id_offset
        self.use_obstacle_avoidance = use_obstacle_avoidance
        self.use_path_constraints_topic = use_path_constraints_topic
        self.nx = self.model_nx + (1 if self.uses_steering_state else 0)
        self.nu = 2
        # This objective-only target is independent from target_lane_idx.
        # It lets recovery keep full-width bounds while gently attracting the
        # prediction toward L1 with the existing, fixed Q matrix.
        self.soft_target_lane_idx = None
        self.soft_target_start_e_y = 0.0
        self.soft_target_alpha = 0.0
        # Optional post-L1-release steering envelope.  While active, the
        # excess steering relative to the path feed-forward angle must shrink
        # toward the configured neutral band at every prediction step.  This
        # prevents receding-horizon MPC from postponing corner-exit unwind.
        self.steering_unwind_active = False
        self.steering_unwind_initial_excess = 0.0
        self.steering_unwind_rate = 0.0
        self.steering_unwind_neutral_band = 0.0

        # setupが済んでいるかどうか
        self.osqp_initialized = False

        # ステアリングレート制約用の事前計算変数
        self.nx_N = self.nx * (self.N + 1)
        self.nu_N = self.nu * self.N

        # With delta as a state, delta_rate is constrained directly by the
        # ordinary input bounds. The legacy model instead constrains adjacent
        # curvature inputs.
        self.n_rate_constraints = 0 if self.uses_steering_state else N - 1
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

        if self.uses_steering_state:
            delta_limit = np.arctan(
                float(InputConstraints['umax'][1]) * self.model.length)
            self.state_constraints = {
                'xmin': np.append(
                    np.asarray(StateConstraints['xmin'], dtype=float),
                    -delta_limit),
                'xmax': np.append(
                    np.asarray(StateConstraints['xmax'], dtype=float),
                    delta_limit),
            }
            self.input_constraints = {
                'umin': np.array([
                    float(InputConstraints['umin'][0]),
                    -float(max_steering_rate),
                ]),
                'umax': np.array([
                    float(InputConstraints['umax'][0]),
                    float(max_steering_rate),
                ]),
            }
        else:
            self.state_constraints = StateConstraints
            self.input_constraints = InputConstraints
        self.ay_max = ay_max
        self.understeer_coeff = max(float(understeer_coeff), 0.0)
        self.steering_preview_enabled = bool(steering_preview_enabled)
        self.steering_command_delay = max(float(steering_command_delay), 0.0)
        self.steering_preview_max_distance = max(
            float(steering_preview_max_distance), 0.0)
        self.steering_reservation_enabled = bool(
            steering_reservation_enabled)
        self.steering_reachability_speed_enabled = bool(
            steering_reachability_speed_enabled)
        self.steering_reachability_min_speed = max(
            float(steering_reachability_min_speed), 0.0)
        self.steering_reachability_min_angle = max(
            float(steering_reachability_min_angle), 0.0)

        # 追加: ステアリングレート制限関連のパラメータ
        self.max_steering_rate = max_steering_rate
        self.previous_steering = 0.0  # 前回のステア角
        # Last QP attempt diagnostics.  These are deliberately independent
        # from current_control/current_prediction, which may contain a reused
        # fallback after a failed solve.
        self.last_attempt_feasible = False
        self.last_attempt_status = "not_solved"
        self.last_attempt_initial_steering = 0.0
        self.last_attempt_predicted_steering = np.array([], dtype=float)
        self.last_attempt_steering_rate = np.array([], dtype=float)
        self.last_attempt_command_steering = np.array([], dtype=float)
        self.last_attempt_target_steering = np.array([], dtype=float)
        self.last_attempt_steering_angle_margin = np.nan
        self.last_attempt_steering_rate_margin = np.nan
        self.steering_rate_limited = False
        self.steering_rate_speed_cap = np.inf

        # 追加: ay_maxによる速度制限の方式切り替え
        self.use_max_kappa_pred = use_max_kappa_pred
        # 既存の初期化
        self.current_prediction = None
        self.infeasibility_counter = 0
        self.solve_time_budget_ms = 20.0
        self.max_prediction_fallback_cycles = 3
        # Keep predicted vehicle centers away from physical course edges and
        # reject solver solutions outside the exact corridor beyond ordinary
        # OSQP numerical error.  Both are configurable by the controller.
        self.prediction_outer_boundary_guard = 0.0
        self.prediction_lateral_tolerance = 0.02
        self.used_prediction_fallback = False
        self.time_budget_exceeded = False
        self.recovery_requested = False
        self.failure_reason = None
        self.steering_rate_limited = False
        self.steering_rate_speed_cap = np.inf
        self.soft_target_lane_idx = None
        self.soft_target_start_e_y = 0.0
        self.soft_target_alpha = 0.0
        self.soft_target_lateral_offset = 0.0
        # Snapshot of the exact corridor used by the latest solve attempt.
        # These values remain available after an infeasible solve so the
        # controller can diagnose lane-bound and obstacle-induced failures.
        self._constraint_wp_ids = np.array([], dtype=int)
        self._constraint_target_lane = None
        self._constraint_safety_margin = 0.0
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

    def configure_steering_unwind(
            self, active: bool, initial_excess: float = 0.0,
            unwind_rate: float = 0.0, neutral_band: float = 0.0):
        """Configure the temporary post-lane-release steering envelope."""
        self.steering_unwind_active = bool(active and self.uses_steering_state)
        self.steering_unwind_initial_excess = float(initial_excess)
        self.steering_unwind_rate = max(float(unwind_rate), 0.0)
        self.steering_unwind_neutral_band = max(float(neutral_band), 0.0)

    def update_Q(self, Q: np.ndarray):
        if self.uses_steering_state and Q.shape == (self.model_nx, self.model_nx):
            steering_weight = float(self.Q[-1, -1])
            self.Q = sparse.block_diag([
                Q, sparse.csc_matrix([[steering_weight]])
            ], format='csc')
        else:
            self.Q = Q

    def update_R(self, R: np.ndarray):
        self.R = R

    def update_QN(self, QN: np.ndarray):
        if self.uses_steering_state and QN.shape == (self.model_nx, self.model_nx):
            steering_weight = float(self.QN[-1, -1])
            self.QN = sparse.block_diag([
                QN, sparse.csc_matrix([[steering_weight]])
            ], format='csc')
        else:
            self.QN = QN

    def set_soft_lateral_reference(
        self, lane_idx=None, start_e_y=0.0, alpha=0.0,
        lateral_offset=0.0,
    ) -> None:
        """Set an objective-only lane reference without narrowing bounds."""
        self.soft_target_lane_idx = (
            int(lane_idx) if lane_idx is not None else None
        )
        self.soft_target_start_e_y = float(start_e_y)
        self.soft_target_alpha = float(np.clip(alpha, 0.0, 1.0))
        self.soft_target_lateral_offset = float(lateral_offset)

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

        # Get curvature predictions.  In the legacy formulation the second
        # input is curvature.  In the steering-state formulation it is a
        # steering *rate*, so interpreting current_control[1::2] as a steering
        # angle produces a completely fictitious lateral-acceleration limit.
        kappa_pred = None
        if not self.uses_steering_state:
            kappa_pred = np.tan(np.append(
                np.array(self.current_control[3::self.nu]),
                self.current_control[-1])) / self.model.length

        # Consider control delay
        self.model.wp_id += self.wp_id_offset

        # Build the path-required steering sequence once.  The vehicle model
        # remains linearised at the local path curvature, while the steering
        # state objective is advanced by actuator delay plus slew time.  This
        # makes steering start before lateral/heading error has already grown.
        steering_reference = np.zeros(N + 1, dtype=float)
        horizon_distance = np.zeros(N + 1, dtype=float)
        if self.uses_steering_state:
            delta_limit = float(self.state_constraints['xmax'][3])
            for index in range(N + 1):
                waypoint = self.model.reference_path.get_waypoint(
                    self.model.wp_id + index)
                gain = understeer_curvature_gain(
                    waypoint.v_ref, self.understeer_coeff)
                kappa_cmd = waypoint.kappa / max(gain, 1e-3)
                steering_reference[index] = float(np.clip(
                    math.atan(self.model.length * kappa_cmd),
                    -delta_limit,
                    delta_limit,
                ))
                if index:
                    previous_waypoint = self.model.reference_path.get_waypoint(
                        self.model.wp_id + index - 1)
                    horizon_distance[index] = (
                        horizon_distance[index - 1]
                        + float(waypoint - previous_waypoint)
                    )

            preview_reference = steering_reference.copy()
            if self.steering_reservation_enabled:
                reservation_speeds = np.asarray([
                    max(float(self.model.reference_path.get_waypoint(
                        self.model.wp_id + index).v_ref), 0.0)
                    for index in range(N + 1)
                ], dtype=float)
                preview_reference = build_arc_length_steering_reservation(
                    horizon_distance,
                    steering_reference,
                    reservation_speeds,
                    self.steering_command_delay,
                    self.max_steering_rate,
                    self.previous_steering,
                    self.model.Ts,
                )
            elif self.steering_preview_enabled:
                for index in range(N + 1):
                    waypoint = self.model.reference_path.get_waypoint(
                        self.model.wp_id + index)
                    preview_index = steering_preview_index(
                        horizon_distance,
                        steering_reference,
                        index,
                        max(float(waypoint.v_ref), 0.0),
                        self.steering_command_delay,
                        self.max_steering_rate,
                        self.steering_preview_max_distance,
                    )
                    preview_reference[index] = steering_reference[preview_index]

            # If even full-rate steering cannot reach an upcoming reference,
            # constrain only the controls before that point.  A floor prevents
            # an abrupt curve at the current waypoint from immobilising cars.
            if self.steering_reachability_speed_enabled:
                # This optional cap is for an upcoming change in path demand,
                # not for correcting the current tracking error.  Using the
                # measured/current delta here made every nearby waypoint apply
                # the minimum-speed floor whenever the actuator lagged by only
                # a few hundredths of a radian.
                initial_delta = float(steering_reference[0])
                for target_index in range(1, N + 1):
                    speed_cap = steering_reachability_speed_cap(
                        horizon_distance[target_index],
                        steering_reference[target_index] - initial_delta,
                        self.steering_command_delay,
                        self.max_steering_rate,
                        self.steering_reachability_min_angle,
                    )
                    if not np.isfinite(speed_cap):
                        continue
                    speed_cap = max(
                        speed_cap, self.steering_reachability_min_speed)
                    for control_index in range(min(target_index, N)):
                        velocity_index = control_index * self.nu
                        umax_dyn[velocity_index] = min(
                            umax_dyn[velocity_index], speed_cap)

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

            # Compute the original [e_y, e_psi, t] model first.
            f_model, A_model, B_model = self.model.linearize(
                v_ref, kappa_ref, delta_s, self.understeer_coeff)

            curvature_gain = understeer_curvature_gain(
                v_ref, self.understeer_coeff)
            kappa_cmd_ref = kappa_ref / max(curvature_gain, 1e-3)

            if self.uses_steering_state:
                # State: [e_y, e_psi, t, delta]
                # Input: [v, delta_rate]
                delta_limit = float(self.state_constraints['xmax'][3])
                # Use the advanced value only as a steering objective.  The
                # lateral dynamics below are still linearised about the local
                # path curvature, avoiding fictitious early road curvature.
                delta_ref = float(steering_reference[n])
                objective_delta_ref = float(preview_reference[n])
                next_delta_ref = float(preview_reference[n + 1])
                dkappa_ddelta = (
                    1.0
                    / (self.model.length * np.cos(delta_ref) ** 2)
                )
                dt_ref = float(delta_s) / max(float(v_ref), 0.5)
                delta_change_ref = next_delta_ref - objective_delta_ref
                delta_rate_ref = float(np.clip(
                    delta_change_ref / max(dt_ref, 1e-3),
                    -self.max_steering_rate,
                    self.max_steering_rate))

                # The rest of this model is spatially discretised: one state
                # transition spans delta_s metres, not one controller tick.
                # Therefore steering is the same continuous delta_rate law
                # integrated over the predicted travel time delta_s / v.
                # Linearise that duration around (v_ref, delta_rate_ref).
                steering_step_ref = delta_rate_ref * float(delta_s) / max(
                    float(v_ref), 0.5)
                dstep_dv = (
                    -delta_rate_ref * float(delta_s)
                    / max(float(v_ref), 0.5) ** 2
                )
                dstep_drate = dt_ref

                A_lin = np.zeros((self.nx, self.nx), dtype=float)
                B_lin = np.zeros((self.nx, self.nu), dtype=float)
                A_lin[:self.model_nx, :self.model_nx] = A_model
                A_lin[:self.model_nx, 3] = (
                    B_model[:, 1] * dkappa_ddelta)
                A_lin[3, 3] = 1.0
                B_lin[:self.model_nx, 0] = B_model[:, 0]
                B_lin[3, 0] = dstep_dv
                B_lin[3, 1] = dstep_drate

                # Preserve the affine operating point of the original
                # curvature-input model after substituting the linearised
                # kappa(delta) relation.
                uq_model = (
                    B_model[:, 0] * v_ref
                    - f_model
                    + B_model[:, 1] * dkappa_ddelta * delta_ref
                )
                uq_delta = (
                    -steering_step_ref
                    + dstep_dv * v_ref
                    + dstep_drate * delta_rate_ref
                )
                uq[n * self.nx:(n + 1) * self.nx] = np.append(
                    uq_model, uq_delta)
                ur[n * self.nu:(n + 1) * self.nu] = [
                    v_ref, delta_rate_ref]
                xr[n * self.nx + 3] = objective_delta_ref
            else:
                A_lin = A_model
                B_lin = B_model
                ur[n*self.nu:(n+1)*self.nu] = [v_ref, kappa_cmd_ref]
                uq[n * self.nx:(n+1)*self.nx] = B_lin.dot(
                    [v_ref, kappa_cmd_ref]) - f_model

            eps = 1e-9
            A_lin[np.abs(A_lin) < eps] = eps
            B_lin[np.abs(B_lin) < eps] = eps
            #print(np.abs(A_lin) < 1e-12, flush=True)
            #print(np.count_nonzero(A_lin),flush=True)
            A_data[n*self.nx*self.nx : (n+1)*self.nx*self.nx] = A_lin.flatten()
            B_data[n*self.nx*self.nu : (n+1)*self.nx*self.nu] = B_lin.flatten()

            # Set spatial reference e_y to target lane center with vehicle safety offset
            target_lane = getattr(self.model.reference_path, 'target_lane_idx', None)
            if target_lane is not None:
                xr[n * self.nx] = self._compute_lane_center(self.model.wp_id + n, target_lane)

            # Constrain maximum speed based on curvature
            # 曲率にもとづいた最大速度の制約
            if self.uses_steering_state:
                # Use path curvature, not the second MPC input (delta_rate),
                # for the lateral-acceleration speed limit.
                if self.use_max_kappa_pred:
                    horizon_kappa = [
                        abs(self.model.reference_path.get_waypoint(
                            self.model.wp_id + j).kappa)
                        for j in range(n, N)
                    ]
                    max_kappa_pred = max(horizon_kappa, default=0.0)
                else:
                    max_kappa_pred = abs(kappa_ref)
                vmax_dyn = np.sqrt(
                    self.ay_max / (max_kappa_pred + 1e-12))
                debug_kappa_pred = max_kappa_pred

                # Slow down when the distance interval would otherwise pass
                # before the actuator can complete the reference change.
                # With preview enabled, adjacent objective values can skip
                # several waypoints. Treating that objective jump as road
                # curvature change imposes an artificial near-crawl speed.
                # Preview reachability has its own optional prefix cap, while
                # the committed first command retains the existing measured
                # steering-rate saturation speed protection.
                if (
                    not self.steering_preview_enabled
                    and abs(delta_change_ref) > 1e-4
                ):
                    steering_vmax = (
                        self.max_steering_rate * float(delta_s)
                        / abs(delta_change_ref)
                    )
                    vmax_dyn = min(vmax_dyn, steering_vmax)
            elif self.use_max_kappa_pred:
                max_kappa_pred = np.max(np.abs(kappa_pred[n:]))
                vmax_dyn = np.sqrt(self.ay_max / (np.abs(max_kappa_pred) + 1e-12))
                debug_kappa_pred = max_kappa_pred
            else:
                vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
                debug_kappa_pred = kappa_pred[n]
                
            umax_dyn[self.nu*n] = min(vmax_dyn, umax_dyn[self.nu*n])

            if n == 0:
                self.deibug_max_kappa_pred = debug_kappa_pred
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
        if self.uses_steering_state:
            xr[N * self.nx + 3] = preview_reference[N]

            if self.steering_unwind_active:
                # State zero is fixed to the measured/last-issued steering
                # angle.  Constrain states 1..N relative to each point's own
                # path-required angle.  Only the release-side bound is
                # tightened, so a changing road curvature may still request
                # steering in the opposite direction when necessary.
                release_sign = math.copysign(
                    1.0, self.steering_unwind_initial_excess)
                initial_magnitude = abs(self.steering_unwind_initial_excess)
                for index in range(1, N + 1):
                    waypoint = self.model.reference_path.get_waypoint(
                        self.model.wp_id + index)
                    gain = understeer_curvature_gain(
                        waypoint.v_ref, self.understeer_coeff)
                    kappa_cmd = waypoint.kappa / max(gain, 1e-3)
                    delta_ref = float(np.clip(
                        math.atan(self.model.length * kappa_cmd),
                        self.state_constraints['xmin'][3],
                        self.state_constraints['xmax'][3],
                    ))
                    allowed_excess = max(
                        self.steering_unwind_neutral_band,
                        initial_magnitude
                        - self.steering_unwind_rate * index * self.model.Ts,
                    )
                    delta_state = index * self.nx + 3
                    if release_sign > 0.0:
                        xmax_dyn[delta_state] = min(
                            xmax_dyn[delta_state],
                            delta_ref + allowed_excess,
                        )
                    else:
                        xmin_dyn[delta_state] = max(
                            xmin_dyn[delta_state],
                            delta_ref - allowed_excess,
                        )

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
        # Obstacle narrowing may already leave a corridor thinner than the
        # physical-edge guard.  Never send inverted bounds to OSQP: represent
        # those invalid prediction points with the existing zero-width
        # sentinel so the QP becomes safely infeasible instead of raising an
        # uncaught bounds-update exception.
        lb, ub = zero_inverted_bounds(lb, ub)

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
        elif self.soft_target_lane_idx is not None:
            # Only xr changes here. lb/ub above remain the full-width corridor,
            # and P/Q stay fixed, so obstacles may still move the solution away
            # from L1 when required.
            for n in range(N):
                lane_center = self._compute_lane_center(
                    self.model.wp_id + n, self.soft_target_lane_idx)
                lane_center += self.soft_target_lateral_offset
                xr[n * self.nx] = blend_lateral_reference(
                    self.soft_target_start_e_y,
                    lane_center,
                    self.soft_target_alpha,
                )
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
        if self.uses_steering_state:
            x0 = np.append(x0, self.previous_steering)
        leq = np.hstack([-x0, uq])
        ueq = leq

        # 入力と状態の制約境界
        lineq_basic = np.hstack([xmin_dyn, np.kron(np.ones(N), umin)])
        uineq_basic = np.hstack([xmax_dyn, umax_dyn])

        # Legacy curvature-input MPC needs adjacent-input constraints. In the
        # steering-state model delta_rate is already bounded in umin/umax.
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
        self.last_attempt_feasible = False
        self.last_attempt_status = "not_solved"
        self.last_attempt_initial_steering = float(self.previous_steering)
        self.last_attempt_predicted_steering = np.array([], dtype=float)
        self.last_attempt_steering_rate = np.array([], dtype=float)
        self.last_attempt_command_steering = np.array([], dtype=float)
        self.last_attempt_target_steering = np.array([], dtype=float)
        self.last_attempt_steering_angle_margin = np.nan
        self.last_attempt_steering_rate_margin = np.nan

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
        diagnostic_delta_limit = float(self.state_constraints['xmax'][3])
        diagnostic_target_delta = []
        for index in range(N + 1):
            diagnostic_waypoint = self.model.reference_path.get_waypoint(
                self.model.wp_id + index)
            diagnostic_gain = understeer_curvature_gain(
                diagnostic_waypoint.v_ref, self.understeer_coeff)
            diagnostic_kappa = (
                diagnostic_waypoint.kappa
                / max(diagnostic_gain, 1e-3)
            )
            diagnostic_target_delta.append(float(np.clip(
                math.atan(self.model.length * diagnostic_kappa),
                -diagnostic_delta_limit,
                diagnostic_delta_limit,
            )))
        self.last_attempt_target_steering = np.asarray(
            diagnostic_target_delta, dtype=float)

        t1 = time.perf_counter()
        t2 = t1

        # Preserve last prediction as fallback when the solver temporarily fails
        prediction_backup = self.current_prediction

        try:

            dec = self.optimizer.solve()
            if self.debug_counter % 20 == 0:
                print(dec.info.status,flush=True)
            t2 = time.perf_counter()

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
                            print(f"Relaxed safety margin by {relaxed_safety_margin} ({5-i}/5) to solve the problem")
                        break
                    if not is_primal_infeasible(dec):
                        break

            if not is_valid_osqp_solution(dec):
                self.last_attempt_status = str(dec.info.status)
                if self.time_budget_exceeded:
                    raise ValueError(
                        "MPC retry time budget exceeded "
                        f"({self.solve_time_budget_ms:.1f}ms)")
                raise ValueError(
                    f"OSQP failed with status '{dec.info.status}'")

            optimizer_controls = np.array(dec.x[-N*nu:])
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

            if self.uses_steering_state:
                # Expose [v, delta] to the existing controller/fallback code.
                # There is one continuous steering-rate solution.  The state
                # prediction samples it after each spatial interval (ds / v),
                # while actuator commands sample the same rate every fixed Ts.
                # These arrays must not be compared by index because their
                # sample times are intentionally different.
                control_signals = np.empty_like(optimizer_controls)
                control_signals[0::nu] = optimizer_controls[0::nu]
                predicted_delta = np.asarray(x[:, 3], dtype=float)
                solved_delta_rate = np.asarray(
                    optimizer_controls[1::nu], dtype=float)
                command_delta = float(self.previous_steering)
                for index, delta_rate in enumerate(solved_delta_rate):
                    command_delta += float(delta_rate) * self.model.Ts
                    command_delta = float(np.clip(
                        command_delta,
                        self.state_constraints['xmin'][3],
                        self.state_constraints['xmax'][3],
                    ))
                    control_signals[index * nu + 1] = command_delta
                command_delta_sequence = np.asarray(
                    control_signals[1::nu], dtype=float)
                delta_limit = float(self.state_constraints['xmax'][3])
                self.last_attempt_predicted_steering = predicted_delta
                self.last_attempt_steering_rate = solved_delta_rate
                self.last_attempt_command_steering = command_delta_sequence
                self.last_attempt_steering_angle_margin = float(
                    delta_limit - np.max(np.abs(predicted_delta)))
                self.last_attempt_steering_rate_margin = float(
                    self.max_steering_rate
                    - np.max(np.abs(solved_delta_rate)))
                v = control_signals[0]
                delta = control_signals[1]
            else:
                control_signals = optimizer_controls
                control_signals[1::2] = np.arctan(
                    control_signals[1::2] * self.model.length)
                v = control_signals[0]
                desired_first_delta = float(control_signals[1])

                # Keep the complete stored command sequence consistent with
                # the actuator. Previously only the live first command was
                # clamped, while RViz/fallback/current_control retained the
                # impossible unconstrained steering sequence.
                max_delta_change = self.max_steering_rate * self.model.Ts
                bounded_delta = float(self.previous_steering)
                for index in range(N):
                    desired_delta = float(control_signals[index * nu + 1])
                    bounded_delta = float(np.clip(
                        desired_delta,
                        bounded_delta - max_delta_change,
                        bounded_delta + max_delta_change,
                    ))
                    control_signals[index * nu + 1] = bounded_delta
                delta = float(control_signals[1])

                requested_change = abs(
                    desired_first_delta - self.previous_steering)
                self.steering_rate_limited = (
                    requested_change > max_delta_change + 1e-6)
                if self.steering_rate_limited:
                    tracking_ratio = np.clip(
                        max_delta_change / max(requested_change, 1e-6),
                        0.0, 1.0)
                    self.steering_rate_speed_cap = max(
                        1.0, float(v) * float(tracking_ratio))
                    v = min(float(v), self.steering_rate_speed_cap)
                    control_signals[0] = v
                    candidate_prediction = self._rollout_bounded_prediction(
                        control_signals, N)
                else:
                    self.steering_rate_speed_cap = np.inf

            self.previous_steering = delta
            self.last_attempt_feasible = True
            self.last_attempt_status = str(dec.info.status)

            # Commit the candidate only after solver and geometry validation.
            self.current_control = control_signals
            self.current_prediction = candidate_prediction

            u = np.array([v, delta])
            max_delta = np.max(np.abs(control_signals[1:len(control_signals)//3*2:2]))

            if self.infeasibility_counter > (N - 1):
                print(f'Problem solved after {self.infeasibility_counter} infeasible iterations')
            self.infeasibility_counter = 0
            self.last_solved_wp_id = self.model.wp_id

        except (TypeError, ValueError) as error:
            self.failure_reason = str(error)
            if self.last_attempt_status == "not_solved":
                self.last_attempt_status = str(error)
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

    def _rollout_bounded_prediction(self, control_signals, N):
        """Predict world motion using the steering commands actually usable."""
        x = float(self.model.temporal_state.x)
        y = float(self.model.temporal_state.y)
        psi = float(self.model.temporal_state.psi)
        x_pred, y_pred = [], []

        for n in range(N):
            current_waypoint = self.model.reference_path.get_waypoint(
                self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(
                self.model.wp_id + n + 1)
            delta_s = max(float(next_waypoint - current_waypoint), 0.0)
            speed = max(float(control_signals[n * self.nu]), 0.0)
            delta = float(control_signals[n * self.nu + 1])
            curvature_gain = understeer_curvature_gain(
                speed, self.understeer_coeff)
            heading_change = (
                delta_s * curvature_gain * np.tan(delta)
                / self.model.length)
            mid_psi = psi + 0.5 * heading_change
            x += delta_s * np.cos(mid_psi)
            y += delta_s * np.sin(mid_psi)
            psi += heading_change
            if 1 <= n < N - 1:
                x_pred.append(x)
                y_pred.append(y)

        return x_pred, y_pred

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
                                            spatial_state_prediction[
                                                n, :self.model_nx])

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
