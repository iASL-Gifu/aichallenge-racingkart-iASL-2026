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

        # 追加: ay_maxによる速度制限の方式切り替え
        self.use_max_kappa_pred = use_max_kappa_pred
        # 既存の初期化
        self.current_prediction = None
        self.infeasibility_counter = 0
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

            dec = self.optimizer.solve()
            if self.debug_counter % 20 == 0:
                print(dec.info.status,flush=True)
            t2 = time.perf_counter()

            if is_primal_infeasible(dec):
                for i in range(1, 6):
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
                raise ValueError(
                    f"OSQP failed with status '{dec.info.status}'")

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

            u = np.array([v, delta])
            max_delta = np.max(np.abs(control_signals[1:len(control_signals)//3*2:2]))

            if self.infeasibility_counter > (N - 1):
                print(f'Problem solved after {self.infeasibility_counter} infeasible iterations')
            self.infeasibility_counter = 0
            self.last_solved_wp_id = self.model.wp_id

        except (TypeError, ValueError) as error:
            if self.debug_counter % 20 == 0:
                print(f"[MPCFallback] {error}", flush=True)
            id = nu * (self.infeasibility_counter + 1)
            if id + 2 < len(self.current_control) and not np.all(self.current_control[id:id+2] == 0.0):
                u = np.array(self.current_control[id:id+2])
                max_delta = np.abs(u[1])
            else:
                # Keep last steering angle and use safe minimum speed (1.0 m/s)
                u = np.array([1.0, self.previous_steering])
                max_delta = np.abs(self.previous_steering)

            # Keep the last valid prediction when solver fails
            if (
                prediction_backup is not None
                and is_plausible_world_prediction(
                    prediction_backup,
                    (self.model.temporal_state.x, self.model.temporal_state.y),
                )
            ):
                self.current_prediction = prediction_backup
            else:
                self.current_prediction = None

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
