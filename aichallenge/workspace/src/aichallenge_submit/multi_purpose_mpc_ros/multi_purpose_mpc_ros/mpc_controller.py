#!/usr/bin/env python3

import yaml
import math
from typing import List, Tuple, Optional, NamedTuple
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import numpy as np
import copy
from contextlib import contextmanager
from multi_purpose_mpc_ros.overtake_session import OvertakeSession, decide_lane
from multi_purpose_mpc_ros.overtake_lane_hold import dynamic_longitudinal_conflict_unsafe
import os
import shutil
from collections import deque
from datetime import datetime

# ROS 2
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray
from . import collision_geometry as collision
from . import lane_evaluation
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty, Bool, Float32MultiArray, Int32, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion, Pose2D, Point, Vector3, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA

from rcl_interfaces.msg import SetParametersResult

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_planning_msgs.msg import Trajectory
try:
    from autoware_auto_vehicle_msgs.msg import GearCommand, GearReport
except ModuleNotFoundError:
    GearCommand = None
    GearReport = None
try:
    from autoware_auto_vehicle_msgs.msg import ControlModeReport, VelocityReport
except (ModuleNotFoundError, ImportError):
    ControlModeReport = None
    VelocityReport = None
try:
    from tier4_vehicle_msgs.msg import ActuationCommandStamped
except ModuleNotFoundError:
    ActuationCommandStamped = None
from v2x_msgs.msg import V2XVehiclePositionArray
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    V2XVehicleTracker,
    absolute_heading_difference,
    build_closed_path_arc_lengths,
    classify_prepass_timeout_reasons,
    classify_lane_conflicts,
    circular_forward_progress,
    continuous_condition_confirmed,
    evaluate_stopped_lead_overtake,
    evaluate_lane_width_samples,
    evaluate_overtake_commit_gate,
    follow_stop_deadlock_conditions_met,
    is_follow_target_ahead,
    is_follow_retry_within_distance,
    is_prepass_fallback_lane_change,
    lane_conflicts_are_clear,
    l0_restricted_follow_can_ignore_passage,
    lateral_vehicle_clearance,
    longitudinal_vehicle_clearance,
    is_parallel_vehicle,
    ordered_outer_lane_candidates,
    ordered_prepass_fallback_candidates,
    outer_lane_problem_slow_override_active,
    outer_prediction_bypass_target_matches,
    overtake_shadow_solution_acceptable,
    prepass_recovery_timeout_expired,
    prepass_recovery_owns_lane_selection,
    prediction_clears_moving_vehicle,
    predictions_to_obstacles,
    project_to_closed_path_frenet,
    reverse_path_has_vehicle_conflict,
    follow_emergency_reacquire_blocked,
    select_safe_outer_lane,
    select_l2_restricted_zone_lane,
    select_latched_overtake_lane,
    slow_lead_commit_distance,
    hybrid_lateral_escape_creep_allowed,
    rolling_precommit_speed_margin,
    l1_rejoin_preemption_target_relevant,
    select_parallel_abort_lane,
    signed_closed_path_arc_distance,
    should_release_latched_overtake_lane,
    should_reevaluate_follow_overtake,
    should_count_mpc_recovery_success,
    should_recover_from_mpc_stall,
    should_hold_generic_reverse_for_minimum_distance,
    should_hold_follow_escape_exclusive,
    should_release_active_overtake_distance_gate,
    should_reset_overtake_latch_for_target_change,
    should_reset_motion_latch,
    should_suppress_overtake_before_grounded_snapshot,
    should_release_prepass_distance_gate,
    should_start_l1_recovery_from_safety,
    should_start_prepass_recovery_from_safety,
    strict_shadow_slow_commit_creep_allowed,
    should_exit_l1_probe_backoff,
    update_continuous_condition_since,
    update_fallback_commit_success_since,
    update_motion_latch,
    update_follow_escape_probe_success_cycles,
    startup_follow_restart_gap,
    startup_same_lane_lead_key,
)

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import (
    ReferencePath,
    collapsed_constraint_snapshot,
    lane_minimum_free_segment_width,
)
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import (
    MPC,
    curvature_lateral_shift,
    lateral_reference_ramp_duration,
    spatial_lane_transition_reference,
    blend_previous_lateral_prediction,
)
from multi_purpose_mpc_ros.core.utils import load_waypoints, kmh_to_m_per_sec, load_ref_path

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists
from multi_purpose_mpc_ros.simulation_logger import SimulationLogger
from multi_purpose_mpc_ros.exexution_stats import ExecutionStats
from multi_purpose_mpc_ros_msgs.msg import AckermannControlBoostCommand, PathConstraints, BorderCells
from multi_purpose_mpc_ros.tools.reference_velocity_configulator import ReferenceVelocityConfigulator


RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
YELLOW = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
CYAN = ColorRGBA(r=0.0, g=156.0 / 255.0, b=209.0 / 255.0, a=1.0)

def array_to_ackermann_control_command(stamp, u: np.ndarray, acc: float) -> AckermannControlCommand:
    msg = AckermannControlCommand()
    msg.stamp = stamp
    msg.lateral.stamp = stamp
    msg.lateral.steering_tire_angle = u[1]
    msg.lateral.steering_tire_rotation_rate = 2.0
    msg.longitudinal.stamp = stamp
    msg.longitudinal.speed = u[0]
    msg.longitudinal.acceleration = acc
    return msg

def yaw_from_quaternion(q: Quaternion):
    sqx = q.x * q.x
    sqy = q.y * q.y
    sqz = q.z * q.z
    sqw = q.w * q.w

    # Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
    sarg = -2 * (q.x*q.z - q.w*q.y) / (sqx + sqy + sqz + sqw) # normalization added from urdfom_headers

    if sarg <= -0.99999:
        yaw = -2. * np.arctan2(q.y, q.x)
    elif sarg >= 0.99999:
        yaw = 2. * np.arctan2(q.y, q.x)
    else:
        yaw = np.arctan2(2. * (q.x*q.y + q.w*q.z), sqw + sqx - sqy - sqz)

    return yaw

def odom_to_pose_2d(odom: Odometry) -> Pose2D:
    pose = Pose2D()
    pose.x = odom.pose.pose.position.x
    pose.y = odom.pose.pose.position.y
    pose.theta = yaw_from_quaternion(odom.pose.pose.orientation)

    return pose

@dataclasses.dataclass
class MPCConfig:
    N: int
    Q: dia_matrix
    R: dia_matrix
    QN: dia_matrix
    v_max: float
    a_min: float
    a_max: float
    ay_max: float
    understeer_coeff: float
    delta_max: float
    steer_rate_max: float
    control_rate: float
    steering_tire_angle_gain_var: float
    accel_low_pass_gain: float
    steer_low_pass_gain: float
    wp_id_offset: int
    use_max_kappa_pred: bool


class MPCController(Node):

    PKG_PATH: str = get_package_share_directory('multi_purpose_mpc_ros') + "/"
    # MAX_LAPS = 6
    MAX_LAPS = 10000
    BUG_VEL = 40.0 # km/h
    BUG_ACC = 400.0

    SHOW_PLOT_ANIMATION = False
    PLOT_RESULTS = False
    ANIMATION_INTERVAL = 20

    KP = 100.0

    def __init__(self, config_path: str, ref_vel_config_path: Optional[str]) -> None:
        super().__init__("mpc_controller") # type: ignore

        # declare parameters
        self.declare_parameter("use_boost_acceleration", False)
        self.declare_parameter("use_obstacle_avoidance", False)
        self.declare_parameter("use_stats", False)

        # get parameters
        self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.USE_BUG_ACC = self.get_parameter("use_boost_acceleration").get_parameter_value().bool_value
        self.USE_OBSTACLE_AVOIDANCE = self.get_parameter("use_obstacle_avoidance").get_parameter_value().bool_value
        self.use_stats = self.get_parameter("use_stats").get_parameter_value().bool_value

        self._config_path = config_path
        self._ref_vel_config_path: Optional[str] = ref_vel_config_path
        self._cfg = self._load_config()
        self._default_wp_id_offset = self._cfg.mpc.wp_id_offset
        self._odom: Optional[Odometry] = None
        self._gnss_pose: Optional[PoseWithCovarianceStamped] = None
        self._enable_control = True
        self._initialize()
        self._setup_parameters_callback()
        self._setup_pub_sub()

        # Determine ego vehicle ID from ROS_DOMAIN_ID
        domain_id = int(os.environ.get("ROS_DOMAIN_ID", "1"))
        self._ego_vehicle_id = f"d{domain_id}"
        self.get_logger().info(f"Initialized MPC Controller. Ego vehicle ID set to: {self._ego_vehicle_id}")

        if self.use_sim_time:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("use_sim_time is enabled!")
            self.get_logger().warn("------------------------------------")
        if self.USE_BUG_ACC:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("USE_BUG_ACC is enabled!")
            self.get_logger().warn("------------------------------------")
        if self.USE_OBSTACLE_AVOIDANCE:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("USE_OBSTACLE_AVOIDANCE is enabled!")
            self.get_logger().warn("------------------------------------")

    def _load_config(self) -> NamedTuple:

        # logging content
        with open(self._config_path, "r") as f:
            config_content = f.read()
            self.get_logger().info(
                "\n" +
                "----- config.yaml -----\n"+
                config_content + "\n" +
                "-----------------------")

        if self._ref_vel_config_path is not None:
            with open(self._ref_vel_config_path, "r") as f:
                ref_vel_config_content = f.read()
                self.get_logger().info(
                    "\n" +
                    "----- ref_vel.yaml -----\n"+
                    ref_vel_config_content + "\n" +
                    "-----------------------")

        with open(self._config_path, "r") as f:
            cfg: NamedTuple = convert_to_namedtuple(yaml.safe_load(f)) # type: ignore

        # Check if the files exist
        mandatory_files = [cfg.map.yaml_path, cfg.waypoints.csv_path] # type: ignore
        for file_path in mandatory_files:
            file_exists(self.in_pkg_share(file_path))
        return cfg

    def _create_reference_path_from_autoware_trajectory(self, trajectory: Trajectory) -> Optional[ReferencePath]:
        wp_x = [0] * len(trajectory.points)
        wp_y = [0] * len(trajectory.points)
        for i, p in enumerate(trajectory.points):
            wp_x[i] = p.pose.position.x
            wp_y[i] = p.pose.position.y

        cfg_ref_path = self._cfg.reference_path # type: ignore
        reference_path = ReferencePath(
            self._map,
            wp_x,
            wp_y,
            cfg_ref_path.resolution,
            cfg_ref_path.smoothing_distance,
            cfg_ref_path.max_width,
            cfg_ref_path.circular)

        mpc_config = self._mpc_cfg
        speed_profile_constraints = {
            "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
            "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}

        if not reference_path.compute_speed_profile(speed_profile_constraints):
            return None

        return reference_path

    def _setup_parameters_callback(self) -> None:
        def declatre_parameters():
            cfg_mpc = self._cfg.mpc
            self.declare_parameter("v_max", cfg_mpc.v_max)
            self.declare_parameter("steering_tire_angle_gain_var", cfg_mpc.steering_tire_angle_gain_var)
            self.declare_parameter("Q0", cfg_mpc.Q[0])
            self.declare_parameter("Q1", cfg_mpc.Q[1])
            self.declare_parameter("Q2", cfg_mpc.Q[2])
            self.declare_parameter("R0", cfg_mpc.R[0])
            self.declare_parameter("R1", cfg_mpc.R[1])
            self.declare_parameter("QN0", cfg_mpc.QN[0])
            self.declare_parameter("QN1", cfg_mpc.QN[1])
            self.declare_parameter("QN2", cfg_mpc.QN[2])

            mpc_cfg = self._mpc_cfg
            self.declare_parameter("ay_max", mpc_cfg.ay_max)
            self.declare_parameter("understeer_coeff", mpc_cfg.understeer_coeff)
            self.declare_parameter("accel_low_pass_gain", mpc_cfg.accel_low_pass_gain)
            self.declare_parameter("steer_low_pass_gain", mpc_cfg.steer_low_pass_gain)
            self.declare_parameter("wp_id_offset", mpc_cfg.wp_id_offset)

        def param_cb(parameters):
            cfg_mpc = self._cfg.mpc # type: ignore
            mpc_cfg = self._mpc_cfg

            def update_Q(index: int, value: float):
                cfg_mpc.Q[index] = value
                mpc_cfg.Q = sparse.diags(cfg_mpc.Q)
                self._mpc.update_Q(mpc_cfg.Q)
                self.get_logger().warn(f"Q[{index}] was updated to '{value}'")

            def update_R(index: int, value: float):
                cfg_mpc.R[index] = value
                mpc_cfg.R = sparse.diags(cfg_mpc.R)
                self._mpc.update_R(mpc_cfg.R)
                self.get_logger().warn(f"R[{index}] was updated to '{value}'")

            def update_QN(index: int, value: float):
                cfg_mpc.QN[index] = value
                mpc_cfg.QN = sparse.diags(cfg_mpc.QN)
                self._mpc.update_QN(mpc_cfg.QN)
                self.get_logger().warn(f"QN[{index}] was updated to '{value}'")

            for param in parameters:
                if param.name == "v_max" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.v_max = param.value
                    self._mpc.update_v_max(kmh_to_m_per_sec(param.value))
                    v_ref: List[float] = [kmh_to_m_per_sec(param.value)] * len(self._reference_path.waypoints)
                    self._reference_path.set_v_ref(v_ref)

                    self.get_logger().warn(f"v_max was updated to '{param.value}' [km/h]")

                elif param.name == "steering_tire_angle_gain_var" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.steering_tire_angle_gain_var = param.value
                    self.get_logger().warn(f"steering_tire_angle_gain_var was updated to '{param.value}'")

                elif param.name == "Q0" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(0, param.value)
                elif param.name == "Q1" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(1, param.value)
                elif param.name == "Q2" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(2, param.value)


                elif param.name == "R0" and param.type_ == Parameter.Type.DOUBLE:
                    update_R(0, param.value)
                elif param.name == "R1" and param.type_ == Parameter.Type.DOUBLE:
                    update_R(1, param.value)

                elif param.name == "QN0" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(0, param.value)
                elif param.name == "QN1" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(1, param.value)
                elif param.name == "QN2" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(2, param.value)

                elif param.name == "ay_max" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.ay_max = param.value
                    self._mpc.update_ay_max(param.value)
                    self.get_logger().warn(f"ay_max was updated to '{param.value}'")

                elif param.name == "understeer_coeff" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.understeer_coeff = max(param.value, 0.0)
                    self._mpc.update_understeer_coeff(param.value)
                    self.get_logger().warn(
                        f"understeer_coeff was updated to '{mpc_cfg.understeer_coeff}'")

                elif param.name == "accel_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.accel_low_pass_gain = param.value
                    self.get_logger().warn(f"accel_low_pass_gain was updated to '{param.value}'")

                elif param.name == "steer_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.steer_low_pass_gain = param.value
                    self.get_logger().warn(f"steer_low_pass_gain was updated to '{param.value}'")

                elif param.name == "wp_id_offset" and param.type_ == Parameter.Type.INTEGER:
                    mpc_cfg.wp_id_offset = param.value
                    self._default_wp_id_offset = param.value
                    self._mpc.update_wp_id_offset(param.value)
                    self.get_logger().warn(f"wp_id_offset was updated to '{param.value}'")


            return SetParametersResult(successful=True)

        declatre_parameters()
        self.add_on_set_parameters_callback(param_cb)

    def _initialize(self) -> None:
        self._map_z = 0.02

        def create_map() -> Map:
            return Map(self.in_pkg_share(self._cfg.map.yaml_path)) # type: ignore

        def create_ref_path(map: Map, custom_csv_path: str = None, bounds_csv_path: str = None) -> ReferencePath:
            cfg_ref_path = self._cfg.reference_path # type: ignore
            target_csv = custom_csv_path if custom_csv_path is not None else cfg_ref_path.csv_path

            is_ref_path_given = target_csv != "" # type: ignore
            if is_ref_path_given:
                print(f"Using given reference path: {target_csv}")
                wp_x, wp_y, wp_psi, _ = load_ref_path(self.in_pkg_share(target_csv)) # type: ignore
                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular,
                    wp_psi=wp_psi,
                    bounds_csv_path=bounds_csv_path)

            else:
                print("Using waypoints to create reference path")
                wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path)) # type: ignore

                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular,
                    bounds_csv_path=bounds_csv_path)


        def create_obstacles() -> List[Obstacle]:
            use_csv_obstacles = self._cfg.obstacles.csv_path != "" # type: ignore
            if use_csv_obstacles:
                obstacles_file_path = self.in_pkg_share(self._cfg.obstacles.csv_path) # type: ignore
                obs_x, obs_y = load_waypoints(obstacles_file_path)
                obstacles = []
                for cx, cy in zip(obs_x, obs_y):
                    obstacles.append(Obstacle(cx=cx, cy=cy, radius=self._cfg.obstacles.radius)) # type: ignore
                return obstacles
            else:
                return []

        def create_car(ref_path: ReferencePath) -> BicycleModel:
            cfg_model = self._cfg.bicycle_model # type: ignore
            return BicycleModel(
                ref_path,
                cfg_model.length,
                cfg_model.width,
                1.0 / self._cfg.mpc.control_rate) # type: ignore

        def create_mpc(car: BicycleModel, N, R=None) -> Tuple[MPCConfig, MPC]:
            cfg_mpc = self._cfg.mpc # type: ignore
            mpc_R = R if R is not None else cfg_mpc.R

            mpc_cfg = MPCConfig(
                N,
                sparse.diags(cfg_mpc.Q),
                sparse.diags(mpc_R),
                sparse.diags(cfg_mpc.QN),
                kmh_to_m_per_sec(self.BUG_VEL if self.USE_BUG_ACC else cfg_mpc.v_max),
                cfg_mpc.a_min,
                cfg_mpc.a_max,
                cfg_mpc.ay_max,
                getattr(cfg_mpc, "understeer_coeff", 0.0),
                np.deg2rad(cfg_mpc.delta_max_deg),
                cfg_mpc.steer_rate_max,
                cfg_mpc.control_rate,
                cfg_mpc.steering_tire_angle_gain_var,
                cfg_mpc.accel_low_pass_gain,
                cfg_mpc.steer_low_pass_gain,
                cfg_mpc.wp_id_offset,
                cfg_mpc.use_max_kappa_pred)

            state_constraints = {
                "xmin": np.array([-np.inf, -np.inf, -np.inf]),
                "xmax": np.array([np.inf, np.inf, np.inf])}
            input_constraints = {
                "umin": np.array([0.0, -np.tan(mpc_cfg.delta_max) / car.length]),
                "umax": np.array([mpc_cfg.v_max, np.tan(mpc_cfg.delta_max) / car.length])}

            # mpcからのsteer指令出力は、gainを掛けて出力され、その状態で車体のsteer rate limit が適用されるため、
            # mpcの制御計算におけるsteer_rate_maxは、実際のsteer_rate_maxをgainで除した値で設定する
            scaled_steer_rate_max = mpc_cfg.steer_rate_max / mpc_cfg.steering_tire_angle_gain_var

            mpc = MPC(
                car,
                N,
                mpc_cfg.Q,
                mpc_cfg.R,
                mpc_cfg.QN,
                state_constraints,
                input_constraints,
                mpc_cfg.ay_max,
                scaled_steer_rate_max,
                mpc_cfg.wp_id_offset,
                self.USE_OBSTACLE_AVOIDANCE,
                self._cfg.reference_path.use_path_constraints_topic,
                mpc_cfg.use_max_kappa_pred,
                mpc_cfg.understeer_coeff,
                steering_command_delay=float(getattr(
                    cfg_mpc, "steering_command_delay", 0.15)),
                steering_reservation_enabled=bool(getattr(
                    cfg_mpc, "steering_reservation_enabled", False)))

            mpc.solve_time_budget_ms = max(float(getattr(
                cfg_mpc, "solve_time_budget_ms", 20.0)), 0.0)
            mpc.max_prediction_fallback_cycles = max(int(getattr(
                cfg_mpc, "max_prediction_fallback_cycles", 3)), 0)
            mpc.prediction_outer_boundary_guard = max(float(getattr(
                cfg_mpc, "prediction_outer_boundary_guard", 0.0)), 0.0)
            mpc.prediction_lateral_tolerance = max(float(getattr(
                cfg_mpc, "prediction_lateral_tolerance", 0.02)), 0.0)
            mpc.lane_constraint_retry_relaxation_m = tuple(
                max(float(value), 0.0) for value in getattr(
                    cfg_mpc,
                    "lane_constraint_retry_relaxation_m",
                    [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.20],
                )
            )
            mpc.lane_constraint_retry_relax_toward_center_only = bool(getattr(
                cfg_mpc, "lane_constraint_retry_relax_toward_center_only", True))
            mpc.lane_constraint_retry_taper_over_horizon = bool(getattr(
                cfg_mpc, "lane_constraint_retry_taper_over_horizon", True))
            mpc.lane_constraint_retry_terminal_ratio = float(np.clip(
                getattr(
                    cfg_mpc,
                    "lane_constraint_retry_terminal_ratio",
                    0.35,
                ),
                0.0,
                1.0,
            ))
            mpc.lane_constraint_connection_points = max(int(getattr(
                cfg_mpc, "lane_constraint_connection_points", 10)), 1)


            return mpc_cfg, mpc

        def compute_speed_profile(car: BicycleModel, mpc_config: MPCConfig) -> None:
            speed_profile_constraints = {
                "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
                "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}
            car.reference_path.compute_speed_profile(speed_profile_constraints)

        def create_ref_vel_configulator() -> Optional[ReferenceVelocityConfigulator]:
            if self._ref_vel_config_path is None:
                return None
            return ReferenceVelocityConfigulator(self, self._config_path, self._ref_vel_config_path)

        self._map = create_map()

        cfg_ref_path = self._cfg.reference_path  # type: ignore

        # Race セットの初期化
        # race_csv_path が指定されていなければ config の csv_path (デフォルト) を使用
        race_csv = getattr(cfg_ref_path, 'race_csv_path', None)
        race_bounds_csv = getattr(cfg_ref_path, 'race_bounds_csv_path', 'env/waypoint_bounds.csv')
        print(f"[init] load race path: {race_csv if race_csv else '(config csv_path)'}, bounds: {race_bounds_csv}")
        self._reference_pathN_race = create_ref_path(
            self._map,
            custom_csv_path=race_csv,
            bounds_csv_path=race_bounds_csv
        )
        #self._reference_path10_race = create_ref_path(self._map)
        self._carN_race = create_car(self._reference_pathN_race)
        #self._car10_race = create_car(self._reference_path10_race)
        self._mpc_cfg_race, self._mpcN_race = create_mpc(self._carN_race, self._cfg.mpc.N)
        #_, self._mpc10_race = create_mpc(self._car10_race, 9, self._cfg.mpc.R10)
        compute_speed_profile(self._carN_race, self._mpc_cfg_race)
        #compute_speed_profile(self._car10_race, self._mpc_cfg_race)

        # Center セットの初期化 (追い越し・追従フラグ時に使用する安定化軌道)
        # USE_OBSTACLE_AVOIDANCE の有無に関わらず常に読み込む
        center_csv = getattr(cfg_ref_path, 'center_csv_path', 'env/centerline/traj_center313.csv')
        center_bounds_csv = getattr(cfg_ref_path, 'center_bounds_csv_path', 'env/centerline/waypoint_bounds_center.csv')
        print(f"[init] load center path: {center_csv}, bounds: {center_bounds_csv}")
        self._reference_pathN_center = create_ref_path(
            self._map,
            custom_csv_path=center_csv,
            bounds_csv_path=center_bounds_csv
        )
        unsafe_static_fallback = bool(getattr(
            cfg_ref_path, "unsafe_static_fallback_on_narrow", False))
        self._reference_pathN_race.unsafe_static_fallback_on_narrow = (
            unsafe_static_fallback)
        self._reference_pathN_center.unsafe_static_fallback_on_narrow = (
            unsafe_static_fallback)
        if unsafe_static_fallback:
            self.get_logger().warn(
                "[UnsafeStaticBoundaryFallback] enabled for comparison: "
                "vehicle-width-infeasible obstacle corridors will be replaced "
                "with static bounds and may permit collision")
        #self._reference_path10_center = create_ref_path(self._map, custom_csv_path=center_csv)
        self._carN_center = create_car(self._reference_pathN_center)
        #self._car10_center = create_car(self._reference_path10_center)
        self._mpc_cfg_center, self._mpcN_center = create_mpc(self._carN_center, self._cfg.mpc.N)
        # A dedicated solver verifies a candidate outer lane without changing
        # the live Center MPC prediction, counters, or warm start.
        self._carN_overtake_commit_probe = create_car(
            self._reference_pathN_center)
        (
            self._mpc_cfg_overtake_commit_probe,
            self._mpcN_overtake_commit_probe,
        ) = create_mpc(
            self._carN_overtake_commit_probe,
            self._cfg.mpc.N,
        )
        #_, self._mpc10_center = create_mpc(self._car10_center, 9, self._cfg.mpc.R10)
        compute_speed_profile(self._carN_center, self._mpc_cfg_center)
        center_arc_points = [
            (waypoint.x, waypoint.y)
            for waypoint in self._reference_pathN_center.waypoints
        ]
        (
            self._center_arc_points,
            self._center_arc_cumulative,
            self._center_arc_total_length,
        ) = build_closed_path_arc_lengths(center_arc_points)
        self._center_arc_mean_wp_spacing = (
            self._center_arc_total_length / max(len(center_arc_points), 1)
        )
        #compute_speed_profile(self._car10_center, self._mpc_cfg_center)

        # Lane lock on curves configuration
        self._curve_lane_lock_enabled = bool(getattr(cfg_ref_path, "curve_lane_lock_enabled", True))
        self._curve_lane_lock_wps = getattr(cfg_ref_path, "curve_lane_lock_wps", [])

        # デフォルトは Race セット
        self._reference_pathN = self._reference_pathN_race
        #self._reference_path10 = self._reference_path10_race
        self._carN = self._carN_race
        #self._car10 = self._car10_race
        self._mpc_cfg = self._mpc_cfg_race
        self._mpcN = self._mpcN_race
        #self._mpc10 = self._mpc10_race

        self._car = self._carN
        self._reference_path = self._reference_pathN
        self._mpc = self._mpcN

        cfg_mpc = self._cfg.mpc  # type: ignore
        self._delay_prediction_enabled = bool(
            getattr(cfg_mpc, "delay_prediction_enabled", True))
        self._steering_command_delay = max(
            float(getattr(cfg_mpc, "steering_command_delay", 0.15)), 0.0)
        self._delay_prediction_steer_gain = max(
            float(getattr(cfg_mpc, "delay_prediction_steer_gain", 1.0)), 0.0)
        self._delay_prediction_steps = max(
            int(getattr(cfg_mpc, "delay_prediction_steps", 9)), 1)
        history_length = max(
            int(self._mpc_cfg.control_rate *
                (self._steering_command_delay + 1.0)),
            self._delay_prediction_steps + 2,
        )
        self._steering_command_history = deque(maxlen=history_length)

        self._ref_vel_configulator: Optional[ReferenceVelocityConfigulator] = create_ref_vel_configulator()

        self._trajectory: Optional[Trajectory] = None
        self._last_lane_change_time = None
        self._target_lane_idx = None
        self._mpc_safety_recovery_active = False
        self._mpc_safety_recovery_success_cycles = 0
        self._mpc_recovery_request_cycles = 0
        self._mpc_recovery_confirm_cycles = max(int(getattr(
            cfg_mpc, "safety_recovery_confirm_cycles", 2)), 1)
        self._mpc_pp_safe_recovery_confirm_cycles = max(int(getattr(
            cfg_mpc, "pp_safe_safety_recovery_confirm_cycles", 6)),
            self._mpc_recovery_confirm_cycles)
        self._mpc_pp_safe_zone_recovery_confirm_cycles = max(int(getattr(
            cfg_mpc, "pp_safe_zone_safety_recovery_confirm_cycles", 20)),
            self._mpc_pp_safe_recovery_confirm_cycles)
        self._race_to_center_failure_grace_sec = max(float(getattr(
            cfg_mpc, "race_to_center_failure_grace_sec", 0.6)), 0.0)
        self._post_reverse_full_width_recovery_active = False
        self._mpc_safety_recovery_success_sec = max(float(getattr(
            cfg_mpc, "safety_recovery_success_sec", 0.5)), 0.0)
        self._mpc_safety_recovery_pp_creep_speed = max(float(getattr(
            cfg_mpc,
            "safety_recovery_pure_pursuit_creep_speed",
            1.5,
        )), 0.0)
        self._mpc_safety_recovery_pp_fast_creep_speed = max(float(getattr(
            cfg_mpc,
            "safety_recovery_pure_pursuit_fast_creep_speed",
            4.0,
        )), self._mpc_safety_recovery_pp_creep_speed)
        self._mpc_safety_recovery_success_required_cycles = max(
            1,
            int(math.ceil(
                self._mpc_safety_recovery_success_sec
                * float(cfg_mpc.control_rate)
            )),
        )
        self._mpc_prediction_fallback_cycles = 0
        # MPC failure fallback order: Pure Pursuit, then legacy feedback.
        self._steering_fallback_enabled = bool(getattr(
            cfg_mpc, "active_path_steering_fallback_enabled", True))
        self._steering_fallback_success_required = max(int(getattr(
            cfg_mpc, "active_path_steering_fallback_success_cycles", 8)), 1)
        self._steering_fallback_lookahead_gain = max(float(getattr(
            cfg_mpc, "active_path_steering_fallback_lookahead_gain", 0.5)), 0.0)
        self._steering_fallback_min_distance = max(float(getattr(
            cfg_mpc, "active_path_steering_fallback_min_distance", 3.5)), 0.1)
        self._steering_fallback_max_distance = max(float(getattr(
            cfg_mpc, "active_path_steering_fallback_max_distance", 8.0)),
            self._steering_fallback_min_distance)
        self._steering_fallback_prediction_sec = max(float(getattr(
            cfg_mpc, "active_path_steering_fallback_prediction_sec", 1.0)), 0.05)
        self._steering_fallback_prediction_steps = max(int(getattr(
            cfg_mpc, "active_path_steering_fallback_prediction_steps", 20)), 1)
        self._steering_fallback_ey_gain = max(float(getattr(
            cfg_mpc, "active_path_steering_fallback_ey_gain", 0.35)), 0.0)
        self._steering_fallback_heading_gain = max(float(getattr(
            cfg_mpc, "active_path_steering_fallback_heading_gain", 0.8)), 0.0)
        self._steering_fallback_speed = max(float(getattr(
            cfg_mpc, "active_path_steering_fallback_speed", 7.5)), 0.0)
        self._steering_fallback_armed = False
        self._steering_fallback_success_cycles = 0

        self._overtake = OvertakeSession()
        self._lane_decision = None
        self._hybrid_reference_key = None
        self._overtake_soft_transition_lane_idx = None
        self._overtake_soft_transition_start_e_y = None
        switch_cfg = getattr(self._cfg, "trajectory_switch", None)
        self._trajectory_enter_center_wps = int(
            getattr(switch_cfg, "enter_center_wps", 25))
        self._trajectory_exit_center_wps = int(
            getattr(switch_cfg, "exit_center_wps", 35))
        self._overtake_latch_max_distance = max(float(getattr(
            switch_cfg, "overtake_latch_max_distance", 35.0)), 0.0)
        self._overtake_commit_min_distance = min(max(float(getattr(
            switch_cfg, "overtake_commit_min_distance", 8.0)), 0.0),
            self._overtake_latch_max_distance)
        self._overtake_commit_preview_distance = max(float(getattr(
            switch_cfg, "overtake_commit_preview_distance", 12.0)), 0.0)
        self._overtake_outside_curvature_threshold = max(float(getattr(
            switch_cfg, "overtake_outside_curvature_threshold", 0.12)), 0.0)
        self._overtake_commit_probe_required_success_cycles = max(int(getattr(
            switch_cfg, "overtake_commit_probe_success_cycles", 2)), 1)
        self._overtake_commit_max_relaxation = max(float(getattr(
            switch_cfg, "overtake_commit_max_relaxation_m", 0.20)), 0.0)
        self._overtake_commit_valid_width_distance = max(float(getattr(
            switch_cfg, "overtake_commit_valid_width_distance_m", 20.0)), 0.0)
        self._overtake_commit_probe_freshness_sec = max(float(getattr(
            switch_cfg, "overtake_commit_probe_freshness_sec", 0.75)), 0.0)
        self._overtake_commit_verified_candidate_hold_sec = max(float(getattr(
            switch_cfg, "overtake_commit_verified_candidate_hold_sec", 1.5)),
            self._overtake_commit_probe_freshness_sec)
        self._hybrid_overtake_enabled = bool(getattr(
            switch_cfg, "hybrid_overtake_enabled", True))
        self._hybrid_passage_loss_confirm_sec = max(float(getattr(
            switch_cfg, "hybrid_passage_loss_confirm_sec", 0.40)), 0.0)
        self._hybrid_side_lock_progress_ratio = float(np.clip(getattr(
            switch_cfg, "hybrid_side_lock_progress_ratio", 0.35), 0.0, 1.0))
        self._ultra_slow_early_commit_speed = max(float(getattr(
            switch_cfg, "ultra_slow_early_commit_speed_kmh", 5.4)) / 3.6, 0.0)
        self._hybrid_overtake_base_length = max(float(getattr(
            switch_cfg, "hybrid_overtake_base_length_m", 6.0)), 0.1)
        self._hybrid_overtake_offset_gain = max(float(getattr(
            switch_cfg, "hybrid_overtake_offset_gain", 2.5)), 0.0)
        self._hybrid_overtake_speed_gain = max(float(getattr(
            switch_cfg, "hybrid_overtake_speed_gain", 0.8)), 0.0)
        self._hybrid_overtake_min_length = max(float(getattr(
            switch_cfg, "hybrid_overtake_min_length_m", 8.0)), 0.1)
        self._hybrid_overtake_max_length = max(float(getattr(
            switch_cfg, "hybrid_overtake_max_length_m", 20.0)),
            self._hybrid_overtake_min_length)
        self._hybrid_overtake_low_speed_start_threshold = max(float(getattr(
            switch_cfg,
            "hybrid_overtake_low_speed_start_threshold_mps", 2.0)), 0.0)
        self._hybrid_overtake_low_speed_max_length = max(float(getattr(
            switch_cfg,
            "hybrid_overtake_low_speed_max_length_m", 8.0)),
            self._hybrid_overtake_min_length)
        self._hybrid_overtake_continuity_weight = float(np.clip(getattr(
            switch_cfg, "hybrid_overtake_continuity_weight", 0.30),
            0.0, 1.0))
        self._hybrid_overtake_continuity_max_deviation = max(float(getattr(
            switch_cfg, "hybrid_overtake_continuity_max_deviation_m", 0.75)),
            0.0)
        self._hybrid_overtake_transition_timeout = max(float(getattr(
            switch_cfg, "hybrid_overtake_transition_timeout_sec", 8.0)), 0.6)
        self._committed_overtake_acceleration_boost_enabled = bool(getattr(
            switch_cfg, "committed_overtake_acceleration_boost_enabled", True))
        self._committed_overtake_acceleration_min_speed_error = max(float(
            getattr(switch_cfg,
                    "committed_overtake_acceleration_min_speed_error", 0.5)),
            0.0)
        # A successful Shadow probe is initially pending state, but its safety
        # result must survive the latch/Soft Transition for the same target and
        # lane.  Otherwise the next cycle forgets the proof and re-applies the
        # slow-lead speed cap until lateral convergence completes.
        self._slow_lead_overtake_speed = max(float(getattr(
            switch_cfg, "slow_lead_overtake_speed_kmh", 10.0)) / 3.6, 0.0)
        self._slow_lead_overtake_far_speed_bonus = max(float(getattr(
            switch_cfg, "slow_lead_overtake_far_speed_bonus", 2.2)), 0.0)
        self._slow_lead_overtake_speed_margin_fade_distance = max(float(getattr(
            switch_cfg, "slow_lead_overtake_speed_margin_fade_distance", 10.0)), 0.0)
        self._slow_lead_overtake_speed_margin = max(float(getattr(
            switch_cfg, "slow_lead_overtake_speed_margin", 0.8)), 0.0)
        self._slow_lead_overtake_committed_speed_margin = max(float(getattr(
            switch_cfg, "slow_lead_overtake_committed_speed_margin", 2.5)),
            self._slow_lead_overtake_speed_margin)
        self._slow_lead_speed_match_release_lateral_error = max(float(getattr(
            switch_cfg, "slow_lead_speed_match_release_lateral_error", 0.45)),
            0.0)
        self._slow_lead_overtake_infeasible_cycles = max(int(getattr(
            switch_cfg, "slow_lead_overtake_infeasible_cycles", 8)), 0)
        self._moving_lead_mpc_grace_speed = max(float(getattr(
            switch_cfg, "moving_lead_mpc_grace_speed_kmh", 10.0)) / 3.6,
            0.0)
        self._moving_lead_mpc_grace_cycles = max(int(getattr(
            switch_cfg, "moving_lead_mpc_grace_cycles", 20)), 0)
        self._slow_lead_overtake_prepare_distance = max(float(getattr(
            switch_cfg, "slow_lead_overtake_prepare_distance", 35.0)), 0.0)
        self._slow_lead_overtake_commit_distance = min(max(float(getattr(
            switch_cfg, "slow_lead_overtake_commit_distance", 28.0)), 0.0),
            self._slow_lead_overtake_prepare_distance)
        self._late_defense_slow_lead_overtake_commit_distance = min(max(
            float(getattr(
                switch_cfg,
                "late_defense_slow_lead_overtake_commit_distance",
                28.0,
            )), 0.0), self._slow_lead_overtake_prepare_distance)
        self._slow_lead_overtake_retry_sec = max(float(getattr(
            switch_cfg, "slow_lead_overtake_retry_sec", 0.25)), 0.0)
        self._overtake_release_distance = max(float(getattr(
            switch_cfg, "overtake_release_distance",
            42.0)),
            self._overtake_latch_max_distance)
        self._overtake_target_switch_margin_m = max(float(getattr(
            switch_cfg, "overtake_target_switch_margin_m", 3.0)), 0.0)
        self._overtake_target_switch_confirm_sec = max(float(getattr(
            switch_cfg, "overtake_target_switch_confirm_sec", 0.5)), 0.0)
        self._overtake_target_immediate_switch_distance = max(float(getattr(
            switch_cfg, "overtake_target_immediate_switch_distance", 5.0)),
            0.0)
        self._overtake_completed_target_id = None
        self._consecutive_overtake_handoff_target_id = None
        self._consecutive_overtake_handoff_lane_idx = None
        self._consecutive_overtake_handoff_started_at = None
        self._consecutive_overtake_handoff_timeout_sec = max(float(getattr(
            switch_cfg, "consecutive_overtake_handoff_timeout_sec", 1.0)),
            self._overtake_commit_probe_freshness_sec)
        self._outer_lane_progress_timeout = max(float(getattr(
            switch_cfg, "outer_lane_progress_timeout_sec", 2.5)), 0.0)
        self._outer_lane_min_progress = max(float(getattr(
            switch_cfg, "outer_lane_min_progress_m", 0.5)), 0.0)
        self._outer_lane_progress_vehicle_id = None
        self._outer_lane_progress_best_longitudinal = None
        self._outer_lane_last_progress_at = None
        self._overtake_switch_candidate_id = None
        self._overtake_switch_candidate_since = None
        # A locked-corridor conflict is more urgent than an ordinary lead
        # replacement.  Keep its confirmation independent so a one-frame V2X
        # or passage-classification dropout cannot restart the timer.
        self._urgent_overtake_switch_confirm_sec = 0.2
        self._urgent_overtake_switch_dropout_grace_sec = 0.25
        self._clear_urgent_overtake_switch_candidate()
        self._center_lane_rejoin_behind_wps = int(
            getattr(switch_cfg, "center_lane_rejoin_behind_wps", 6))
        self._trajectory_behind_release_wps = int(
            getattr(switch_cfg, "behind_release_wps", 12))
        self._trajectory_switch_min_hold = float(
            getattr(switch_cfg, "min_hold_sec", 1.0))
        self._lane_change_cooldown_sec = max(float(
            getattr(switch_cfg, "lane_change_cooldown_sec", 2.5)), 0.0)
        self._l0_entry_prohibited_zones = []
        for zone in getattr(switch_cfg, "l0_entry_prohibited_zones", []):
            start_wp = int(getattr(zone, "start_wp", -1))
            end_wp = int(getattr(zone, "end_wp", -1))
            if start_wp >= 0 and end_wp >= 0:
                self._l0_entry_prohibited_zones.append(
                    (start_wp, end_wp))
        self._l2_entry_restricted_zones = []
        for zone in getattr(switch_cfg, "l2_entry_restricted_zones", []):
            start_wp = int(getattr(zone, "start_wp", -1))
            end_wp = int(getattr(zone, "end_wp", -1))
            if start_wp >= 0 and end_wp >= 0:
                self._l2_entry_restricted_zones.append(
                    (start_wp, end_wp))
        self._l2_entry_restricted_override_speed = max(float(getattr(
            switch_cfg,
            "l2_entry_restricted_override_speed_kmh",
            10.0,
        )) / 3.6, 0.0)
        self._l2_inward_offset_zones = []
        for zone in getattr(switch_cfg, "l2_inward_offset_zones", []):
            start_wp = int(getattr(zone, "start_wp", -1))
            end_wp = int(getattr(zone, "end_wp", -1))
            offset_m = max(float(getattr(zone, "offset_m", 0.0)), 0.0)
            if start_wp >= 0 and end_wp >= 0 and offset_m > 0.0:
                self._l2_inward_offset_zones.append(
                    (start_wp, end_wp, offset_m))
        self._full_width_l1_offset_zones = []
        for zone in getattr(switch_cfg, "full_width_l1_offset_zones", []):
            start_wp = int(getattr(zone, "start_wp", -1))
            end_wp = int(getattr(zone, "end_wp", -1))
            offset_m = max(float(getattr(zone, "offset_m", 0.0)), 0.0)
            if start_wp >= 0 and end_wp >= 0 and offset_m > 0.0:
                self._full_width_l1_offset_zones.append(
                    (start_wp, end_wp, offset_m))
        self._full_width_l0_offset_zones = []
        for zone in getattr(switch_cfg, "full_width_l0_offset_zones", []):
            start_wp = int(getattr(zone, "start_wp", -1))
            end_wp = int(getattr(zone, "end_wp", -1))
            offset_m = max(float(getattr(zone, "offset_m", 0.0)), 0.0)
            if start_wp >= 0 and end_wp >= 0 and offset_m > 0.0:
                self._full_width_l0_offset_zones.append(
                    (start_wp, end_wp, offset_m))
        self._pp_safe_recovery_fast_zones = []
        for zone in getattr(switch_cfg, "pp_safe_recovery_fast_zones", []):
            start_wp = int(getattr(zone, "start_wp", -1))
            end_wp = int(getattr(zone, "end_wp", -1))
            if start_wp >= 0 and end_wp >= 0:
                self._pp_safe_recovery_fast_zones.append(
                    (start_wp, end_wp))
        self._outer_lane_mpc_problem_zones = []
        for zone in getattr(switch_cfg, "outer_lane_mpc_problem_zones", []):
            start_wp = int(getattr(zone, "start_wp", -1))
            end_wp = int(getattr(zone, "end_wp", -1))
            if start_wp >= 0 and end_wp >= 0:
                self._outer_lane_mpc_problem_zones.append(
                    (start_wp, end_wp))
        self._outer_lane_problem_override_speed = max(float(getattr(
            switch_cfg,
            "outer_lane_mpc_problem_override_speed_kmh",
            5.0,
        )) / 3.6, 0.0)
        self._outer_lane_problem_override_release_speed = max(float(getattr(
            switch_cfg,
            "outer_lane_mpc_problem_override_release_speed_kmh",
            6.0,
        )) / 3.6, self._outer_lane_problem_override_speed)
        self._outer_lane_problem_slow_override_target_id = None
        self._trajectory_exit_confirm = float(
            getattr(switch_cfg, "exit_confirm_sec", 0.5))
        self._race_rejoin_max_heading = math.radians(float(
            getattr(switch_cfg, "race_rejoin_max_heading_deg", 10.0)))
        self._race_rejoin_probe_start_heading = max(
            math.radians(max(float(getattr(
                switch_cfg, "race_rejoin_probe_start_heading_deg", 12.0)), 0.0)),
            self._race_rejoin_max_heading)
        self._race_rejoin_probe_release_heading = max(
            math.radians(max(float(getattr(
                switch_cfg, "race_rejoin_probe_release_heading_deg", 15.0)), 0.0)),
            self._race_rejoin_probe_start_heading)
        self._race_rejoin_direct_max_position_gap = max(float(getattr(
            switch_cfg, "race_rejoin_direct_max_position_gap", 0.75)), 0.0)
        self._race_rejoin_probe_required_success_cycles = max(int(getattr(
            switch_cfg, "race_rejoin_probe_success_cycles", 3)), 1)
        self._race_handoff_ramp_sec = max(float(getattr(
            switch_cfg, "race_handoff_ramp_sec", 1.5)), 0.0)
        self._race_handoff_max_reference_speed = max(float(getattr(
            switch_cfg, "race_handoff_max_reference_speed", 0.7)), 0.0)
        self._race_handoff_max_position_gap = max(float(getattr(
            switch_cfg, "race_handoff_max_position_gap", 3.0)), 0.0)
        self._race_handoff_max_target_step = max(float(getattr(
            switch_cfg, "race_handoff_max_target_step", 0.75)), 0.0)
        self._race_handoff_guard_extra_lookahead_wps = max(int(getattr(
            switch_cfg, "race_handoff_guard_extra_lookahead_wps", 10)), 0)
        self._race_rejoin_probe_timeout_sec = max(float(getattr(
            switch_cfg, "race_rejoin_probe_timeout_sec", 1.0)), 0.0)
        self._race_rejoin_retry_backoff_sec = max(float(getattr(
            switch_cfg, "race_rejoin_retry_backoff_sec", 2.0)), 0.0)
        self._race_targets_in_center_frame = self._build_race_targets_in_center_frame()
        self._race_rejoin_handoff_active = False
        self._race_rejoin_handoff_soft = False
        self._race_rejoin_handoff_started_at = None
        self._race_rejoin_handoff_start_e_y = None
        self._race_rejoin_handoff_effective_ramp_sec = None
        self._race_rejoin_handoff_guidance_ready = False
        self._race_rejoin_probe_success_cycles = 0
        self._race_rejoin_probe_confirmed = False
        self._race_rejoin_probe_started_at = None
        self._race_rejoin_retry_not_before = None
        self._race_rejoin_latched_targets = None
        self._race_rejoin_latched_start_wp = None
        self._trajectory_last_switch_time = None
        self._trajectory_clear_since = None
        self._trajectory_switch_reason = "initial"
        self._trajectory_vehicle_id = None
        self._prev_opponent_ahead_detected = False
        self._center_lane_rejoin_active = False
        self._center_lane_rejoin_constraint_released = False
        self._center_lane_rejoin_stable_since = None
        self._l1_probe_active = False
        self._l1_probe_context = None
        self._l1_probe_success_cycles = 0
        self._l1_probe_constraint_applied = False
        self._l1_safety_recovery_active = False
        self._l1_safety_recovery_stable_since = None
        self._l1_safety_recovery_context = None
        self._l1_safety_reprobe_pending = False
        rejoin_stability_cfg = getattr(
            self._cfg, "center_lane_rejoin_stability", None)
        self._center_lane_rejoin_stability_enabled = bool(getattr(
            rejoin_stability_cfg, "enabled", True))
        self._constraint_diagnostics_enabled = bool(getattr(
            rejoin_stability_cfg, "constraint_diagnostics_enabled", True))
        self._constraint_diagnostics_points = max(int(getattr(
            rejoin_stability_cfg, "constraint_diagnostics_points", 5)), 1)
        self._center_lane_rejoin_max_heading = math.radians(max(float(getattr(
            rejoin_stability_cfg, "max_heading_error_deg", 5.0)), 0.0))
        self._center_lane_rejoin_max_lateral_speed = max(float(getattr(
            rejoin_stability_cfg, "max_lateral_speed", 0.3)), 0.0)
        self._center_lane_rejoin_max_yaw_rate = max(float(getattr(
            rejoin_stability_cfg, "max_yaw_rate", 0.25)), 0.0)
        self._center_lane_rejoin_stable_sec = max(float(getattr(
            rejoin_stability_cfg, "stable_sec", 0.5)), 0.0)
        self._center_lane_rejoin_release_infeasible_cycles = max(int(getattr(
            rejoin_stability_cfg, "release_infeasible_cycles", 8)), 1)
        self._l1_probe_required_success_cycles = max(int(getattr(
            rejoin_stability_cfg, "probe_success_cycles", 3)), 1)
        self._l1_rejoin_max_lateral_error = max(float(getattr(
            rejoin_stability_cfg, "max_l1_lateral_error", 0.25)), 0.0)
        self._l1_prediction_fit_points = max(int(getattr(
            rejoin_stability_cfg, "prediction_fit_points", 0)), 0)
        self._l1_prediction_min_fit_ratio = min(max(float(getattr(
            rejoin_stability_cfg, "prediction_fit_ratio", 1.0)), 0.0), 1.0)
        self._l1_probe_retry_cooldown_sec = max(float(getattr(
            rejoin_stability_cfg, "probe_retry_cooldown_sec", 2.0)), 0.0)
        self._l1_probe_retry_min_wp_progress = max(int(getattr(
            rejoin_stability_cfg, "probe_retry_min_wp_progress", 5)), 0)
        self._l1_backoff_full_width_success_sec = max(float(getattr(
            rejoin_stability_cfg, "backoff_full_width_success_sec", 0.5)), 0.0)
        self._l1_soft_rejoin_enabled = bool(getattr(
            rejoin_stability_cfg, "soft_rejoin_enabled", True))
        self._l1_soft_rejoin_ramp_sec = max(float(getattr(
            rejoin_stability_cfg, "soft_rejoin_ramp_sec", 1.5)), 0.0)
        self._l1_soft_rejoin_max_reference_speed = max(float(getattr(
            rejoin_stability_cfg,
            "soft_rejoin_max_reference_speed", 1.0)), 0.0)
        self._l1_soft_rejoin_max_ramp_sec = max(float(getattr(
            rejoin_stability_cfg, "soft_rejoin_max_ramp_sec", 3.0)), 0.0)
        self._l1_soft_rejoin_started_at = None
        self._l1_soft_rejoin_start_e_y = None
        self._l1_soft_rejoin_effective_ramp_sec = None
        self._l1_soft_rejoin_full_strength_logged = False
        self._l1_rejoin_backoff_active = False
        self._l1_rejoin_backoff_started_at = None
        self._l1_rejoin_backoff_failed_wp = None
        self._l1_rejoin_backoff_full_width_success_since = None
        self._outer_lane_released_vehicle_id = None
        self._post_overtake_vehicle_id = None
        self._race_return_time = None
        self._outer_lane_release_infeasible_cycles = max(
            int(getattr(
                switch_cfg, "outer_lane_release_infeasible_cycles", 8)), 1)
        self._outer_lane_release_behind_m = max(
            float(getattr(switch_cfg, "outer_lane_release_behind_m", 1.0)), 0.0)
        prepass_fallback_cfg = getattr(
            self._cfg, "prepass_lane_fallback", None)
        self._prepass_lane_fallback_enabled = bool(getattr(
            prepass_fallback_cfg, "enabled", True))
        self._prepass_lane_fallback_infeasible_cycles = max(int(getattr(
            prepass_fallback_cfg, "infeasible_cycles", 5)), 1)
        self._prepass_lane_fallback_recovery_stable_sec = max(float(getattr(
            prepass_fallback_cfg, "recovery_stable_sec", 0.5)), 0.0)
        self._prepass_lane_fallback_recovery_timeout_sec = max(float(getattr(
            prepass_fallback_cfg, "recovery_timeout_sec", 5.0)), 0.0)
        self._prepass_soft_guidance_timeout_sec = max(float(getattr(
            prepass_fallback_cfg, "soft_guidance_timeout_sec", 10.0)), 0.0)
        self._prepass_follow_retry_sec = max(float(getattr(
            prepass_fallback_cfg, "follow_retry_sec", 2.0)), 0.0)
        self._prepass_follow_retry_max_distance = max(float(getattr(
            prepass_fallback_cfg, "follow_retry_max_distance", 10.0)), 0.0)
        self._prepass_behind_release_longitudinal = max(float(getattr(
            prepass_fallback_cfg, "behind_release_longitudinal", 0.5)), 0.0)
        self._prepass_behind_release_confirm_sec = max(float(getattr(
            prepass_fallback_cfg, "behind_release_confirm_sec", 0.25)), 0.0)
        self._prepass_max_heading = math.radians(max(float(getattr(
            prepass_fallback_cfg, "max_heading_error_deg", 10.0)), 0.0))
        self._prepass_heading_lookahead_wps = max(int(getattr(
            prepass_fallback_cfg, "heading_lookahead_wps", 3)), 1)
        self._prepass_commit_required_success_sec = max(float(getattr(
            prepass_fallback_cfg, "commit_success_sec", 0.3)), 0.0)
        self._prepass_lane_fallback_front_distance = max(float(getattr(
            prepass_fallback_cfg, "front_distance", 8.0)), 0.0)
        self._prepass_lane_fallback_side_distance = max(float(getattr(
            prepass_fallback_cfg, "side_distance", 2.0)), 0.0)
        self._prepass_lane_fallback_rear_distance = max(float(getattr(
            prepass_fallback_cfg, "rear_distance", 5.0)), 0.0)
        self._prepass_lane_fallback_prediction_sec = max(float(getattr(
            prepass_fallback_cfg, "prediction_sec", 1.0)), 0.0)
        self._follow_target_lost_hold_sec = max(float(getattr(
            prepass_fallback_cfg, "follow_target_lost_hold_sec", 1.0)), 0.0)
        self._follow_emergency_reacquire_sec = max(float(getattr(
            prepass_fallback_cfg, "follow_emergency_reacquire_sec", 1.0)), 0.0)
        self._follow_last_released_vehicle_id = None
        self._follow_last_released_at = None
        self._follow_target_lost_max_speed = max(float(getattr(
            prepass_fallback_cfg, "follow_target_lost_max_speed", 1.0)), 0.0)
        follow_cfg = getattr(self._cfg, "follow_control", None)
        self._follow_only = bool(getattr(follow_cfg, "follow_only", False))
        self._follow_engage_distance = max(float(getattr(
            follow_cfg, "engage_distance", 15.0)), 0.0)
        self._follow_desired_distance = max(float(getattr(
            follow_cfg, "desired_distance", 8.0)), 0.0)
        self._follow_lateral_distance = max(float(getattr(
            follow_cfg, "lateral_distance", 1.2)), 0.0)
        self._follow_spacing_kp = max(float(getattr(
            follow_cfg, "spacing_kp", 1.2)), 0.0)
        self._follow_deadlock_escape_enabled = bool(getattr(
            follow_cfg, "deadlock_escape_enabled", True))
        self._follow_deadlock_ego_speed_threshold = max(float(getattr(
            follow_cfg, "deadlock_ego_speed_threshold", 0.15)), 0.0)
        self._follow_deadlock_lead_speed_threshold = max(float(getattr(
            follow_cfg, "deadlock_lead_speed_threshold", 0.3)), 0.0)
        self._follow_deadlock_forward_command_threshold = max(float(getattr(
            follow_cfg, "deadlock_forward_command_threshold", 0.3)), 0.0)
        self._follow_deadlock_gnss_distance_threshold = max(float(getattr(
            follow_cfg, "deadlock_gnss_distance_threshold", 0.3)), 0.0)
        self._follow_deadlock_hold_sec = max(float(getattr(
            follow_cfg, "deadlock_hold_sec", 2.0)), 0.0)
        self._follow_escape_probe_success_cycles_required = max(int(getattr(
            follow_cfg, "escape_probe_success_cycles", 3)), 1)
        self._follow_escape_probe_timeout_sec = max(float(getattr(
            follow_cfg, "escape_probe_timeout_sec", 1.0)), 0.1)
        self._follow_escape_reevaluate_sec = max(float(getattr(
            follow_cfg, "escape_reevaluate_sec", 0.75)), 0.1)
        self._follow_escape_creep_speed = max(float(getattr(
            follow_cfg, "escape_creep_speed", 1.0)), 0.0)
        self._follow_escape_forward_sec = max(float(getattr(
            follow_cfg, "escape_forward_sec", 1.0)), 0.1)
        self._follow_deadlock_since = None
        self._follow_deadlock_start_xy = None
        self._follow_escape_active = False
        self._follow_escape_target_id = None
        self._follow_escape_probe_lane_idx = None
        self._follow_escape_probe_started_at = None
        self._follow_escape_probe_success_cycles = 0
        self._follow_escape_attempted_lanes = set()
        self._follow_escape_forward_active = False
        self._follow_escape_forward_until = None
        self._follow_escape_last_reevaluate_at = None
        self._follow_escape_target_prediction_blocked = False
        self._prepass_fallback_lane_idx = None
        self._prepass_fallback_blocked = False
        self._prepass_fallback_follow_active = False
        self._prepass_follow_last_retry_at = None
        self._prepass_retry_after_reverse = False
        self._prepass_retry_lane_idx = None
        self._prepass_reverse_motion_started = False
        self._prepass_reverse_start_xy = None
        self._prepass_reverse_distance = 0.0
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_recovery_started_at = None
        self._prepass_dynamic_conflict_speed_limit = None
        self._prepass_soft_switch_confirm_sec = 0.3
        self._prepass_soft_dropout_grace_sec = 0.25
        self._clear_prepass_soft_guidance()
        self._prepass_target_behind_since = None
        self._prepass_failed_lane_idx = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_fallback_commit_success_since = None
        self._prepass_attempted_outer_lanes = set()
        self._follow_latched_cache = None
        overtake_cfg = getattr(self._cfg, "stopped_vehicle_overtake", None)
        self._stopped_lead_speed_threshold = float(
            getattr(overtake_cfg, "speed_threshold", 0.3))
        self._forced_overtake_speed = float(
            getattr(overtake_cfg, "max_speed", 10.0))
        self._close_obstacle_reverse_distance = float(
            getattr(overtake_cfg, "reverse_distance", 3.0))
        self._close_obstacle_infeasible_cycles = int(
            getattr(overtake_cfg, "infeasible_cycles", 5))
        self._passage_clearance = max(float(getattr(
            overtake_cfg, "passage_clearance", 0.30)), 0.0)
        self._passage_lane_width_tolerance = max(float(getattr(
            overtake_cfg, "lane_width_tolerance", 0.05)), 0.0)
        self._passage_lane_width_tolerance_points = max(int(getattr(
            overtake_cfg, "lane_width_tolerance_points", 2)), 0)
        self._hybrid_escape_creep_speed = max(float(getattr(
            overtake_cfg, "hybrid_escape_creep_speed", 0.6)), 0.0)
        self._hybrid_escape_min_lateral_gap = max(float(getattr(
            overtake_cfg, "hybrid_escape_min_lateral_gap", 0.10)), 0.0)
        self._strict_shadow_stopped_commit_creep_speed = max(float(getattr(
            overtake_cfg, "strict_shadow_commit_creep_speed", 1.0)), 0.0)
        self._strict_shadow_commit_creep_max_target_speed = kmh_to_m_per_sec(
            max(float(getattr(
                overtake_cfg,
                "strict_shadow_commit_creep_max_target_speed_kmh",
                5.0)), 0.0))
        self._forced_overtake_vehicle_id = None
        self._close_obstacle_reverse_requested = False
        self._parallel_abort_active = False
        self._parallel_abort_vehicle_id = None
        self._parallel_abort_target_lane_idx = None
        self._parallel_timer_vehicle_id = None
        self._parallel_start_time = None

        restart_cfg = getattr(self._cfg, "follow_restart", None)
        self._follow_restart_lead_moving_speed = float(getattr(
            restart_cfg, "lead_moving_speed", 0.5))
        self._follow_restart_ego_stopped_speed = float(getattr(
            restart_cfg, "ego_stopped_speed", 0.3))
        self._follow_restart_min_gap = float(getattr(
            restart_cfg, "min_gap", 6.0))
        self._follow_restart_start_min_gap = float(getattr(
            restart_cfg, "start_min_gap", 4.0))
        self._follow_restart_speed_margin = float(getattr(
            restart_cfg, "speed_margin", 1.5))
        self._follow_restart_max_speed = float(getattr(
            restart_cfg, "max_speed", 3.0))
        self._follow_restart_acceleration = float(getattr(
            restart_cfg, "acceleration", 2.0))
        self._follow_restart_duration = float(getattr(
            restart_cfg, "duration", 1.0))
        self._follow_stopped_vehicle_id = None
        self._follow_restart_until = 0.0
        self._intentional_follow_stop_active = False

        start_boost_cfg = getattr(self._cfg, "initial_start_boost", None)
        self._initial_start_boost_enabled = bool(getattr(
            start_boost_cfg, "enabled", True))
        self._initial_start_boost_duration = max(float(getattr(
            start_boost_cfg, "duration", 3.0)), 0.0)
        self._initial_start_turbo_enabled = bool(getattr(
            start_boost_cfg, "turbo_enabled", True))
        self._initial_start_motion_speed_threshold = max(float(getattr(
            start_boost_cfg, "motion_speed_threshold", 0.1)), 0.0)
        self._initial_start_hold_l0 = bool(getattr(
            start_boost_cfg, "hold_l0_during_boost", True))
        self._initial_start_soft_l0_enabled = bool(getattr(
            start_boost_cfg, "soft_l0_guidance_enabled", False))
        self._initial_start_soft_l0_l1_offset = max(float(getattr(
            start_boost_cfg, "soft_l0_l1_offset", 0.3)), 0.0)
        self._initial_start_soft_l0_ramp_sec = max(float(getattr(
            start_boost_cfg, "soft_l0_ramp_sec", 1.5)), 0.0)
        self._initial_start_soft_l0_max_reference_speed = max(float(getattr(
            start_boost_cfg, "soft_l0_max_reference_speed", 0.5)), 0.0)
        self._initial_start_soft_l0_curvature_threshold = max(float(getattr(
            start_boost_cfg, "soft_l0_curvature_threshold", 0.04)), 0.0)
        self._initial_start_soft_l0_curvature_gain = max(float(getattr(
            start_boost_cfg, "soft_l0_curvature_gain", 6.0)), 0.0)
        self._initial_start_soft_l0_curvature_max_shift = max(float(getattr(
            start_boost_cfg, "soft_l0_curvature_max_shift", 1.0)), 0.0)
        self._initial_start_soft_l0_started_at = None
        self._initial_start_soft_l0_start_e_y = None
        self._initial_start_soft_l0_effective_ramp_sec = None
        self._initial_start_soft_l0_full_strength_logged = False
        self._initial_start_soft_l0_curvature_shift = 0.0
        self._initial_start_soft_l0_last_update_sec = None
        self._initial_start_post_hold_min_sec = max(float(getattr(
            start_boost_cfg, "post_boost_hold_min_sec", 1.5)), 0.0)
        self._initial_start_post_hold_stable_sec = max(float(getattr(
            start_boost_cfg, "post_boost_stable_sec", 0.4)), 0.0)
        self._initial_start_post_hold_max_kappa = max(float(getattr(
            start_boost_cfg, "post_boost_max_kappa", 0.12)), 0.0)
        self._initial_start_post_hold_lookahead_wps = max(int(getattr(
            start_boost_cfg, "post_boost_lookahead_wps", 12)), 1)
        self._initial_start_boost_armed = False
        self._initial_start_boost_until = None
        self._initial_start_boost_done = False
        self._initial_start_boost_logged = False
        self._initial_start_boost_decision_logged = False
        self._initial_start_turbo_published = False
        self._initial_start_lane_hold_logged = False
        self._initial_start_exclusive_active = False
        self._initial_start_post_hold_active = False
        self._initial_start_post_hold_l0_active = False
        self._initial_start_post_hold_started_at = None
        self._initial_start_post_hold_stable_since = None
        self._grounded_start_boost_eligible = None
        self._grounded_start_boost_capture_state = None
        self._grounded_ego_lane_idx = None
        self._grounded_l0_vehicle_ids = []
        self._v2x_received_once = False
        self._grounded_snapshot_wait_logged = False
        self._startup_priority_target_id = None

        # Obstacles
        if self.USE_OBSTACLE_AVOIDANCE:
            #固定障害物(地図や事前に定義する)
            self._static_obstacles: List[Obstacle] = create_obstacles()
            #動的障害物(V2Xなど他車両)
            self._dynamic_obstacles: List[Obstacle] = []
            self._obstacles_updated = bool(self._static_obstacles)
            v2x_cfg = self._cfg.v2x_obstacle_avoidance  # type: ignore
            #V2Xの位置追跡からtrackingを行うモジュール(id管理、位置のスムージング、ジャンプ除去、速度推定)
            self._v2x_tracker = V2XVehicleTracker(
                v_max_safety=float(v2x_cfg.v_max_safety),
                position_jump_threshold=float(v2x_cfg.position_jump_threshold),
                warn_callback=self.get_logger().warn,
                zero_velocity_hold_sec=float(getattr(v2x_cfg, "zero_velocity_hold_sec", 0.20)),
                zero_velocity_threshold=float(getattr(v2x_cfg, "zero_velocity_threshold", 0.10)),
            )
            # Only callbacks mutate this live tracker.  _v2x_tracker is replaced
            # by an immutable-for-the-cycle snapshot at the start of _control.
            self._v2x_input_tracker = self._v2x_tracker
            body_cfg = getattr(self._cfg, 'collision_geometry', None)
            self._collision_geometry = collision.BodyGeometry(
                float(getattr(body_cfg,'length',2.064)),float(getattr(body_cfg,'width',1.45)))
            self._collision_ego_origin = str(getattr(body_cfg,'ego_position_origin','unconfirmed'))
            self._collision_v2x_origin = str(getattr(body_cfg,'v2x_position_origin','unconfirmed'))
            self._collision_center_offset = float(getattr(body_cfg,'rear_axle_to_center',0.522))
            self._collision_origin_lateral_margin = max(float(getattr(body_cfg,'origin_lateral_margin',0.272)),0.0)
            self._collision_max_age = float(getattr(body_cfg,'max_observation_age',0.5))
            self._collision_body_publisher = self.create_publisher(MarkerArray,'/mpc/collision_bodies',1)
            self._collision_pose_subscriptions = [self.create_subscription(
                PoseStamped, f'/v2x/{vid}/body_pose',
                lambda msg, vehicle_id=vid: self._measured_body_pose_callback(vehicle_id,msg), 1)
                for vid in getattr(body_cfg,'measured_pose_vehicle_ids',['d1','d2','d3','d4'])]
            self._v2x_applied_generation = -1
            self._v2x_vehicle_radius = float(v2x_cfg.vehicle_radius)
            self._moving_vehicle_brake_bypass_min_speed = 2.0
            self._moving_vehicle_brake_bypass_confirm_sec = 0.2
            self._moving_vehicle_brake_bypass_horizon_sec = 1.5
            self._moving_vehicle_brake_bypass_preview_distance = 8.0
            self._moving_vehicle_brake_bypass_since = {}
            self._center_path_collision_prediction_horizon_sec = max(float(
                getattr(v2x_cfg,
                        "center_path_collision_prediction_horizon_sec", 1.5)),
                0.0)
            self._center_path_collision_prediction_margin = max(float(
                getattr(v2x_cfg,
                        "center_path_collision_prediction_margin", 0.15)),
                0.0)
            self._center_path_collision_hazard_hold_sec = max(float(
                getattr(v2x_cfg,
                        "center_path_collision_hazard_hold_sec", 0.50)),
                0.0)
            self._center_path_collision_hazard_until = {}
            self._moving_emergency_preview_distance = max(float(getattr(
                v2x_cfg, "moving_emergency_preview_distance", 12.0)), 0.0)
            self._moving_emergency_reaction_sec = max(float(getattr(
                v2x_cfg, "moving_emergency_reaction_sec", 0.40)), 0.0)
            self._moving_emergency_available_deceleration = max(float(getattr(
                v2x_cfg, "moving_emergency_available_deceleration", 2.5)),
                0.1)
            self._moving_emergency_desired_distance = max(float(getattr(
                v2x_cfg, "moving_emergency_desired_distance", 3.0)), 0.0)
            self._moving_emergency_spacing_kp = max(float(getattr(
                v2x_cfg, "moving_emergency_spacing_kp", 0.5)), 0.0)
            self._moving_emergency_max_speed_deficit = max(float(getattr(
                v2x_cfg, "moving_emergency_max_speed_deficit", 0.5)), 0.0)
            self._moving_emergency_critical_distance = max(float(getattr(
                v2x_cfg, "moving_emergency_critical_distance", 2.0)), 0.0)
            self._parallel_safety_enabled = bool(getattr(
                v2x_cfg, "parallel_safety_enabled", True))
            self._v2x_parallel_vehicle_half_width = max(float(getattr(
                v2x_cfg, "parallel_vehicle_half_width",
                self._v2x_vehicle_radius)), 0.0)
            self._parallel_ego_half_length = max(float(getattr(
                v2x_cfg, "parallel_ego_half_length", 1.0)), 0.0)
            self._parallel_vehicle_half_length = max(float(getattr(
                v2x_cfg, "parallel_vehicle_half_length", 1.0)), 0.0)
            self._parallel_critical_clearance = max(float(getattr(
                v2x_cfg, "parallel_critical_clearance", 0.30)), 0.0)
            self._parallel_warning_clearance = max(float(getattr(
                v2x_cfg, "parallel_warning_clearance", 1.30)),
                self._parallel_critical_clearance)
            self._parallel_overtake_speed_margin = max(float(getattr(
                v2x_cfg, "parallel_overtake_speed_margin", 0.50)), 0.0)
            self._parallel_safety_longitudinal_clearance = max(float(getattr(
                v2x_cfg, "parallel_safety_longitudinal_clearance", 0.50)),
                0.0)
            self._parallel_abort_longitudinal_clearance = max(float(getattr(
                v2x_cfg, "parallel_abort_longitudinal_clearance", 0.50)),
                0.0)
            self._parallel_safety_lon_behind = max(float(getattr(
                v2x_cfg, "parallel_safety_lon_behind", 2.0)), 0.0)
            self._parallel_safety_lon_ahead = max(float(getattr(
                v2x_cfg, "parallel_safety_lon_ahead", 4.5)), 0.0)
            self._parallel_abort_lon_behind = max(float(getattr(
                v2x_cfg, "parallel_abort_lon_behind", 0.5)), 0.0)
            self._parallel_abort_lon_ahead = max(float(getattr(
                v2x_cfg, "parallel_abort_lon_ahead", 4.5)), 0.0)
            self._parallel_safety_arc_behind = max(float(getattr(
                v2x_cfg, "parallel_safety_arc_behind", 2.0)), 0.0)
            self._parallel_safety_arc_ahead = max(float(getattr(
                v2x_cfg, "parallel_safety_arc_ahead", 5.0)), 0.0)
            self._parallel_abort_arc_behind = max(float(getattr(
                v2x_cfg, "parallel_abort_arc_behind", 0.5)), 0.0)
            self._parallel_abort_arc_ahead = max(float(getattr(
                v2x_cfg, "parallel_abort_arc_ahead", 5.0)), 0.0)
            self._parallel_abort_sec = max(float(getattr(
                v2x_cfg, "parallel_abort_sec", 4.0)), 0.0)
            mpc_N = int(self._cfg.mpc.N)  # type: ignore
            t_horizon = mpc_N / float(self._cfg.mpc.control_rate)  # type: ignore
            self._v2x_t_samples = [
                k * t_horizon / max(mpc_N - 1, 1) for k in range(mpc_N)
            ]
            # コリドー外の V2X 障害物で MPC のコリドー狭窄/反転が起きないよう、
            # ref-path 近傍のみに絞り込む。閾値 = max_width/2 + vehicle_radius + 余白。
            ref_max_width = float(self._cfg.reference_path.max_width)  # type: ignore
            self._v2x_corridor_threshold_sq = (
                ref_max_width / 2.0 + self._v2x_vehicle_radius + 0.5
            ) ** 2
            wps = self._reference_path.waypoints
            self._waypoint_xy = np.asarray(
                [(wp.x, wp.y) for wp in wps], dtype=np.float64)

        # Laps
        self._current_laps = 1
        self._last_lap_time = 0.0
        self._lap_times = [None] * (self.MAX_LAPS + 1) # +1 means include lap 0

        # condition
        self._last_condition = None
        self._last_colliding_time = None
        self._configure_stuck_recovery()

        # stats
        self._stats = ExecutionStats(self.get_logger(), window_size=50, record_count_threshold=1000)

        # save config
        if self._cfg.common.save_config:
            self._save_config()

    def _save_config(self) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_dir = self.PKG_PATH + f"log/{now}"
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(self._config_path, os.path.join(dst_dir, "config.yaml"))

    def _configure_stuck_recovery(self) -> None:
        cfg = getattr(self._cfg, "stuck_recovery", None)

        def get_cfg(name: str, default):
            return getattr(cfg, name, default) if cfg is not None else default

        self._stuck_recovery_enabled = bool(get_cfg("enabled", True))
        self._stuck_speed_threshold = float(get_cfg("speed_threshold", 0.15))
        self._mpc_stall_speed_threshold = max(float(get_cfg(
            "mpc_stall_speed_threshold", 0.4)), 0.0)
        self._stuck_forward_cmd_threshold = float(get_cfg("forward_cmd_threshold", 0.8))
        self._stuck_time_threshold = float(get_cfg("stuck_time_threshold", 2.0))
        self._stuck_gnss_distance_threshold = float(get_cfg("gnss_distance_threshold", 0.3))
        self._stuck_forward_resume_gap = max(float(get_cfg(
            "forward_resume_gap", 4.0)), 0.0)
        self._stuck_forward_resume_lead_speed = max(float(get_cfg(
            "forward_resume_lead_speed", 0.5)), 0.0)
        self._stuck_forward_resume_min_command = max(float(get_cfg(
            "forward_resume_min_command", 0.3)), 0.0)
        self._stuck_reverse_duration = float(get_cfg("reverse_duration", 3.0))
        self._adaptive_reverse_enabled = bool(get_cfg(
            "adaptive_reverse_enabled", True))
        self._adaptive_reverse_max_distance = max(float(get_cfg(
            "adaptive_reverse_max_distance", 6.0)), 0.0)
        self._adaptive_reverse_wall_margin = max(float(get_cfg(
            "adaptive_reverse_wall_margin", 0.3)), 0.0)
        self._adaptive_reverse_footprint_radius = max(float(get_cfg(
            "adaptive_reverse_footprint_radius",
            float(self._cfg.bicycle_model.width) / math.sqrt(2.0),
        )), 0.1)
        self._adaptive_reverse_min_clear_distance = max(float(get_cfg(
            "adaptive_reverse_min_clear_distance", 0.3)), 0.0)
        self._adaptive_reverse_localization_max_error = max(float(get_cfg(
            "adaptive_reverse_localization_max_error", 0.5)), 0.0)
        self._adaptive_reverse_localization_fresh_sec = max(float(get_cfg(
            "adaptive_reverse_localization_fresh_sec", 0.5)), 0.0)
        self._adaptive_reverse_localization_confirm_sec = max(float(get_cfg(
            "adaptive_reverse_localization_confirm_sec", 0.25)), 0.0)
        self._adaptive_reverse_forward_success_cycles_required = max(int(
            get_cfg("adaptive_reverse_forward_success_cycles", 3)), 1)
        self._generic_reverse_min_distance = max(float(get_cfg(
            "generic_reverse_min_distance", 0.5)), 0.0)
        self._stuck_cooldown = float(get_cfg("cooldown", 2.0))
        self._stuck_forward_reverse_speed = abs(float(get_cfg("reverse_speed", 1.0)))
        self._stuck_reverse_speed = -abs(float(get_cfg("reverse_speed", 1.0)))
        self._stuck_reverse_acceleration = -abs(float(get_cfg("reverse_acceleration", 1.5)))
        self._stuck_reverse_acceleration_positive = bool(
            get_cfg("reverse_acceleration_positive", True))
        self._stuck_reverse_steering_scale = float(get_cfg("reverse_steering_scale", 0.0))
        self._stuck_reverse_command_mode = str(
            get_cfg("reverse_command_mode", "awsim_reverse_button"))
        self._stuck_request_control_mode = bool(get_cfg("request_control_mode", False))
        self._stuck_control_mode_request_value = bool(
            get_cfg("control_mode_request_value", True))
        self._stuck_shift_control_mode_request_value = bool(
            get_cfg("shift_control_mode_request_value", self._stuck_control_mode_request_value))
        self._stuck_drive_control_mode_request_value = bool(
            get_cfg("drive_control_mode_request_value", self._stuck_control_mode_request_value))
        self._stuck_send_gear_command = bool(get_cfg("send_gear_command", True))
        self._stuck_wait_for_reverse_gear = bool(get_cfg("wait_for_reverse_gear", True))
        self._stuck_pre_reverse_duration = float(get_cfg("pre_reverse_duration", 0.0))
        self._stuck_gear_shift_delay = float(get_cfg("gear_shift_delay", 1.0))
        self._stuck_max_shift_wait = float(get_cfg("max_shift_wait", 5.0))
        self._stuck_drive_request_resend_sec = max(float(get_cfg(
            "drive_request_resend_sec", 0.5)), 0.0)
        self._stuck_reverse_request_resend_sec = max(float(get_cfg(
            "reverse_request_resend_sec", 0.5)), 0.0)
        self._stuck_collision_window = float(get_cfg("collision_window", 20.0))
        self._stuck_collision_count_threshold = int(get_cfg("collision_count_threshold", 3))
        self._stuck_use_actuation_cmd = bool(get_cfg("use_actuation_cmd", True))
        self._stuck_actuation_accel_cmd = abs(float(get_cfg("actuation_accel_cmd", 1.0)))
        self._stuck_actuation_brake_cmd = abs(float(get_cfg("actuation_brake_cmd", 0.0)))
        self._gear_reverse_reports = {
            int(value)
            for value in str(get_cfg("reverse_gear_reports", "20")).split(",")
            if value.strip()
        }
        self._stuck_reverse_gear_command_override = get_cfg("reverse_gear_command", None)
        self._stuck_drive_gear_command_override = get_cfg("drive_gear_command", None)
        self._stuck_pre_reverse_gear_command_override = get_cfg("pre_reverse_gear_command", None)
        self._last_stuck_gear_command = None
        self._stuck_last_drive_request_at = None
        self._stuck_last_reverse_request_at = None
        self._stuck_reverse_drive_after = None
        self._stuck_reverse_drive_active = False
        self._stuck_recovery_started_at = None
        self._stuck_reverse_start_target_longitudinal = None
        self._stuck_reverse_target_distance = None
        self._stuck_reverse_start_heading = None
        self._adaptive_reverse_active = False
        self._adaptive_reverse_static_clearance = None
        self._adaptive_reverse_forward_success_cycles = 0
        self._straight_reentry_enabled = bool(get_cfg(
            "straight_reentry_enabled", True))
        self._straight_reentry_speed = max(float(get_cfg(
            "straight_reentry_speed", 1.0)), 0.0)
        self._straight_reentry_probe_distance = max(float(get_cfg(
            "straight_reentry_probe_distance", 2.0)), 0.1)
        self._straight_reentry_timeout = max(float(get_cfg(
            "straight_reentry_timeout", 8.0)), 0.5)
        self._straight_reentry_localization_wait_sec = max(float(get_cfg(
            "straight_reentry_localization_wait_sec", 1.0)),
            self._adaptive_reverse_localization_confirm_sec)
        self._straight_reentry_min_improvement = max(float(get_cfg(
            "straight_reentry_min_improvement", 0.20)), 0.0)
        self._straight_reentry_success_cycles_required = max(int(get_cfg(
            "straight_reentry_success_cycles", 8)), 1)
        self._straight_reentry_active = False
        self._straight_reentry_direction = 0
        self._straight_reentry_started_at = None
        self._straight_reentry_pre_reverse_until = None
        self._straight_reentry_returning_drive = False
        self._straight_reentry_success_cycles = 0
        self._straight_reentry_last_violation = None
        self._straight_reentry_localization_wait_started_at = None
        self._straight_reentry_allow_boundary_step_increase = False
        self._post_reverse_straight_reentry_pending = False
        self._localization_consistent_since = None
        self._localization_consistent = False
        self._localization_position_error = math.inf
        self._last_odom_received_sec = None
        self._last_gnss_received_sec = None
        self._stuck_pre_reverse_until = None
        self._gear_report = None
        self._last_gear_report_received_sec = None
        self._stuck_drive_transition_started_at = None
        self._drive_confirmed_by_command_fallback = False
        self._control_mode_report = None
        self._velocity_report = None
        self._awsim_state = None
        self._actuation_cmd_pub = None
        self._gear_drive_command = (
            int(self._stuck_drive_gear_command_override)
            if self._stuck_drive_gear_command_override is not None
            else getattr(GearCommand, "DRIVE", 2) if GearCommand is not None else 2
        )
        self._gear_reverse_command = (
            int(self._stuck_reverse_gear_command_override)
            if self._stuck_reverse_gear_command_override is not None
            else getattr(GearCommand, "REVERSE", 20) if GearCommand is not None else 20
        )
        self._gear_pre_reverse_command = (
            int(self._stuck_pre_reverse_gear_command_override)
            if self._stuck_pre_reverse_gear_command_override is not None
            else getattr(GearCommand, "NEUTRAL", 1) if GearCommand is not None else 1
        )
        if GearReport is not None and hasattr(GearReport, "REVERSE"):
            self._gear_reverse_reports.add(int(getattr(GearReport, "REVERSE")))

        self._stuck_since = None
        self._stuck_recovery_until = None
        self._stuck_cooldown_until = None
        self._has_moved_once = False
        self._stuck_pre_drive_until = None
        self._stuck_wait_for_drive = False
        self._gnss_history = []
        self._collision_times = []

        if self._stuck_recovery_enabled:
            self.get_logger().info(
                "[StuckRecovery] enabled: "
                f"speed<{self._stuck_speed_threshold:.2f}m/s for "
                f"{self._stuck_time_threshold:.1f}s -> reverse "
                f"(MPC recovery: speed<={self._mpc_stall_speed_threshold:.2f}m/s "
                "with GNSS stationary) "
                f"mode={self._stuck_reverse_command_mode} "
                f"speed={self._stuck_forward_reverse_speed:.2f} "
                f"accel={abs(self._stuck_reverse_acceleration):.2f} "
                f"accel_positive={self._stuck_reverse_acceleration_positive} "
                f"use_actuation_cmd={self._stuck_use_actuation_cmd} "
                f"control_mode_request={self._stuck_control_mode_request_value} "
                f"shift_control_mode={self._stuck_shift_control_mode_request_value} "
                f"drive_control_mode={self._stuck_drive_control_mode_request_value} "
                f"send_gear={self._stuck_send_gear_command} "
                f"wait_gear={self._stuck_wait_for_reverse_gear} "
                f"pre_gear={self._gear_pre_reverse_command} "
                f"pre_duration={self._stuck_pre_reverse_duration:.2f} "
                f"gear_cmd={self._gear_reverse_command} "
                f"reverse_reports={sorted(self._gear_reverse_reports)} "
                f"for {self._stuck_reverse_duration:.1f}s "
                f"adaptive={self._adaptive_reverse_enabled} "
                f"adaptive_max={self._adaptive_reverse_max_distance:.2f}m "
                f"localization_error_max="
                f"{self._adaptive_reverse_localization_max_error:.2f}m "
                f"source={__file__}"
            )

    def _setup_pub_sub(self) -> None:
        # Publishers
        if self.USE_BUG_ACC:
          self._command_pub = self.create_publisher(
            AckermannControlBoostCommand, "/boost_commander/command", 1)
        else:
          self._command_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1)
          self._command_raw_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd_raw", 1)
          print("use normal ackermann control command")

        # NOTE:評価環境での可視化のためにダミーのトピック名を使用
        self._mpc_pred_pub = self.create_publisher(
            MarkerArray, "/mpc/prediction", 1)
        self._mpc_pred_pub_dummy = self.create_publisher(
            MarkerArray, "/planning/scenario_planning/lane_driving/motion_planning/obstacle_stop_planner/virtual_wall", 1)

        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # NOTE:評価環境での可視化のためにダミーのトピック名を使用
        self._ref_path_pub = self.create_publisher(
            MarkerArray, "/mpc/ref_path", latching_qos)
        self._ref_path_pub_dummy = self.create_publisher(
            MarkerArray, "/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner/debug/bound", latching_qos)

        # 3車線境界・追い越しゾーン可視化
        self._lane_marker_pub = self.create_publisher(
            MarkerArray, "/mpc/lane_bounds", latching_qos)

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)
        self._gnss_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/sensing/gnss/pose_with_covariance", self._gnss_callback, 1)
        self._control_mode_request_sub = self.create_subscription(
            Bool, "control/control_mode_request_topic", self._control_mode_request_callback, 1)
        # simple_trajectory_generator publishes with BEST_EFFORT/KEEP_LAST(1) — match it
        # so the subscription is QoS-compatible (rclpy default is RELIABLE).
        trajectory_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._trajectory_sub = self.create_subscription(
            Trajectory, "planning/scenario_planning/trajectory", self._trajectory_callback, trajectory_qos)
        self._stop_request_sub = self.create_subscription(
            Empty, "/control/mpc/stop_request", self._stop_request_callback, 1)

        if self.use_sim_time:
            self._awsim_status_sub = self.create_subscription(
                Float32MultiArray, "/awsim/status", self._awsim_status_callback, 1)
            self._awsim_state_sub = self.create_subscription(
                String, "/awsim/state", self._awsim_state_callback, 1)
            self._condition_sub = self.create_subscription(
                Int32, "/aichallenge/pitstop/condition", self._condition_callback, 1)

        self._awsim_control_mode_request_pub = self.create_publisher(
            Bool, "/awsim/control_mode_request_topic", 1)
        self._awsim_turbo_pub = self.create_publisher(
            Float32MultiArray, "/awsim/cmd", 10)
        self._gear_cmd_pub = None
        if GearCommand is not None:
            self._gear_cmd_pub = self.create_publisher(
                GearCommand, "/control/command/gear_cmd", 10)
            if GearReport is not None:
                self._gear_status_sub = self.create_subscription(
                    GearReport, "/vehicle/status/gear_status", self._gear_status_callback, 1)
        else:
            self.get_logger().warn(
                "autoware_auto_vehicle_msgs/GearCommand is unavailable; "
                "stuck recovery cannot shift AWSIM gear from ROS."
            )
        if ActuationCommandStamped is not None:
            self._actuation_cmd_pub = self.create_publisher(
                ActuationCommandStamped, "/control/command/actuation_cmd", 1)
        else:
            self.get_logger().warn(
                "tier4_vehicle_msgs/ActuationCommandStamped is unavailable; "
                "stuck recovery cannot publish AWSIM actuation_cmd."
            )

        if ControlModeReport is not None:
            self._control_mode_status_sub = self.create_subscription(
                ControlModeReport,
                "/vehicle/status/control_mode",
                self._control_mode_status_callback,
                1,
            )
        if VelocityReport is not None:
            self._velocity_status_sub = self.create_subscription(
                VelocityReport,
                "/vehicle/status/velocity_status",
                self._velocity_status_callback,
                1,
            )

        if self.USE_OBSTACLE_AVOIDANCE:
            if self._cfg.reference_path.use_path_constraints_topic: # type: ignore
                self._path_constraints_sub = self.create_subscription(
                    PathConstraints, "/path_constraints_provider/path_constraints", self._path_constraints_callback, 1)

            if self._cfg.reference_path.use_border_cells_topic: # type: ignore
                self._border_cells_sub = self.create_subscription(
                    BorderCells, "/path_constraints_provider/border_cells", self._border_cells_callback, 1)

            self._v2x_sub = self.create_subscription(
                V2XVehiclePositionArray,
                "/v2x/vehicle_positions",
                self._v2x_callback,
                1)

        # マップ境界線のマーカーを取得するサブスクライバ (TRANSIENT_LOCAL QoS)
        map_marker_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self._map_marker_sub = self.create_subscription(
            MarkerArray, "/map/vector_map_marker", self._map_marker_callback, map_marker_qos
        )

    def _create_ackerman_control_command(self, stamp, u, acc, bug_acc_enabled):
        v_cmd = u[0]
        steer_cmd = u[1]

        ackerman_cmd = array_to_ackermann_control_command(stamp.to_msg(), [v_cmd, steer_cmd], acc)

        if not self.USE_BUG_ACC:
            return ackerman_cmd

        ackerman_boost_cmd = AckermannControlBoostCommand()
        ackerman_boost_cmd.command = ackerman_cmd
        ackerman_boost_cmd.boost_mode = bug_acc_enabled
        return ackerman_boost_cmd

    def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):
        cmd = self._create_ackerman_control_command(stamp, u, acc, bug_acc_enabled)

        # publish raw control command
        self._command_raw_pub.publish(cmd)

        # Store the measured-response-equivalent angle for delay prediction.
        prediction_steer = float(u[1]) * self._delay_prediction_steer_gain
        self._steering_command_history.append((
            float(stamp.nanoseconds) / 1e9,
            prediction_steer,
        ))

        # Apply the actuator command gain only to the published command.
        cmd.lateral.steering_tire_angle *= (
            self._mpc_cfg.steering_tire_angle_gain_var)
        self._command_pub.publish(cmd)

    def _predict_pose_after_steering_delay(
        self, pose: Pose2D, speed: float, now_sec: float,
    ) -> Pose2D:
        """Predict the state when the newly calculated command reaches the actuator."""
        predicted = Pose2D()
        predicted.x = pose.x
        predicted.y = pose.y
        predicted.theta = pose.theta

        if (
            not self._delay_prediction_enabled
            or self._steering_command_delay <= 0.0
            or abs(speed) < 1e-3
        ):
            return predicted

        step_dt = self._steering_command_delay / self._delay_prediction_steps
        history = list(self._steering_command_history)
        fallback_steer = (
            float(self._last_u[1]) * self._delay_prediction_steer_gain)

        for step in range(self._delay_prediction_steps):
            source_time = (
                now_sec + step * step_dt - self._steering_command_delay)
            steering = fallback_steer
            for command_time, command_steer in reversed(history):
                if command_time <= source_time:
                    steering = command_steer
                    break

            yaw_rate = speed / self._car.length * math.tan(steering)
            mid_yaw = predicted.theta + 0.5 * yaw_rate * step_dt
            predicted.x += speed * math.cos(mid_yaw) * step_dt
            predicted.y += speed * math.sin(mid_yaw) * step_dt
            predicted.theta += yaw_rate * step_dt

        predicted.theta = math.atan2(
            math.sin(predicted.theta), math.cos(predicted.theta))
        return predicted

    def _limit_fallback_steering(self, requested_delta: float) -> float:
        """Apply the MPC tire-angle and steering-rate limits."""
        delta_limit = float(self._mpc_cfg.delta_max)
        rate_step = float(self._mpc.max_steering_rate) / max(
            float(self._mpc_cfg.control_rate), 1e-6)
        previous = float(self._mpc.previous_steering)
        return float(np.clip(
            requested_delta,
            max(-delta_limit, previous - rate_step),
            min(delta_limit, previous + rate_step)))

    def _active_path_pure_pursuit_feedback(
        self, pose: Pose2D, speed: float,
    ):
        """Calculate Pure Pursuit feedback on the currently selected path."""
        ref_path = self._reference_path
        lookahead = float(np.clip(
            self._steering_fallback_lookahead_gain * abs(speed)
            + self._steering_fallback_min_distance,
            self._steering_fallback_min_distance,
            self._steering_fallback_max_distance))
        wp_id = int(self._car.get_closest_waypoint(pose.x, pose.y))
        target_wp = ref_path.get_waypoint(wp_id)
        previous_wp = target_wp
        travelled = 0.0
        target_wp_id = wp_id
        for offset in range(1, int(ref_path.n_waypoints) + 1):
            candidate_id = (wp_id + offset) % int(ref_path.n_waypoints)
            candidate = ref_path.get_waypoint(candidate_id)
            travelled += math.hypot(
                float(candidate.x) - float(previous_wp.x),
                float(candidate.y) - float(previous_wp.y))
            target_wp = candidate
            target_wp_id = candidate_id
            if travelled >= lookahead:
                break
            previous_wp = candidate
        if travelled < 1e-3:
            return None, lookahead, target_wp_id, "degenerate_path"

        wheelbase = float(self._cfg.bicycle_model.length)
        rear_x = pose.x - 0.5 * wheelbase * math.cos(pose.theta)
        rear_y = pose.y - 0.5 * wheelbase * math.sin(pose.theta)
        alpha = math.atan2(
            float(target_wp.y) - rear_y,
            float(target_wp.x) - rear_x) - pose.theta
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        requested = math.atan2(
            2.0 * wheelbase * math.sin(alpha), lookahead)
        if not math.isfinite(requested):
            return None, lookahead, target_wp_id, "non_finite_steering"
        return (self._limit_fallback_steering(requested), lookahead,
                target_wp_id, "ok")

    def _legacy_active_path_feedback(self) -> float:
        """Legacy feedback used when Pure Pursuit itself cannot be computed."""
        waypoint = self._reference_path.get_waypoint(int(self._car.wp_id))
        requested = (
            math.atan(float(self._cfg.bicycle_model.length)
                      * float(waypoint.kappa))
            - self._steering_fallback_ey_gain
            * float(self._car.spatial_state.e_y)
            - self._steering_fallback_heading_gain
            * float(self._car.spatial_state.e_psi))
        return self._limit_fallback_steering(requested)

    def _pure_pursuit_feedback_is_safe(
        self, pose: Pose2D, speed: float, steering: float,
    ):
        """Validate the PP command by rolling the bicycle model inside walls."""
        predicted = Pose2D()
        predicted.x = float(pose.x)
        predicted.y = float(pose.y)
        predicted.theta = float(pose.theta)
        rollout_speed = max(abs(float(speed)), 0.0)
        steps = self._steering_fallback_prediction_steps
        dt = self._steering_fallback_prediction_sec / steps
        wheelbase = max(float(self._cfg.bicycle_model.length), 1e-6)
        for index in range(steps + 1):
            wp_id, e_y, lower, upper = self._physical_corridor_state(
                predicted.x, predicted.y)
            if not lower <= e_y <= upper:
                return False, (
                    f"wall_at_step={index},wp={wp_id},e_y={e_y:.3f},"
                    f"allowed=[{lower:.3f},{upper:.3f}]")
            if index < steps:
                yaw_rate = rollout_speed / wheelbase * math.tan(steering)
                mid_yaw = predicted.theta + 0.5 * yaw_rate * dt
                predicted.x += rollout_speed * math.cos(mid_yaw) * dt
                predicted.y += rollout_speed * math.sin(mid_yaw) * dt
                predicted.theta += yaw_rate * dt
        return True, "ok"

    def _apply_stuck_reverse_command(self, u) -> None:
        if self._stuck_reverse_command_mode in ("teleop", "awsim_reverse_button"):
            # In AWSIM reverse gear, keep speed positive and let the gear decide
            # the vehicle direction.  A zero target speed can cancel the throttle.
            u[0] = self._stuck_forward_reverse_speed
        elif self._stuck_reverse_command_mode == "negative_speed_positive_accel":
            u[0] = -abs(self._stuck_reverse_speed)
        else:
            u[0] = -abs(self._stuck_reverse_speed)
        u[1] *= self._stuck_reverse_steering_scale

    def _publish_gear_command(
        self, now, command: int, *, force: bool = False
    ) -> None:
        if not self._stuck_send_gear_command or GearCommand is None or self._gear_cmd_pub is None:
            return
        if not force and self._last_stuck_gear_command == command:
            return
        msg = GearCommand()
        msg.stamp = now.to_msg()
        msg.command = command
        self._gear_cmd_pub.publish(msg)
        self._last_stuck_gear_command = command

    def _publish_stuck_actuation_command(self, now, accel: float, brake: float, steer_cmd: float) -> None:
        if (
            not self._stuck_use_actuation_cmd
            or ActuationCommandStamped is None
            or self._actuation_cmd_pub is None
        ):
            return
        msg = ActuationCommandStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "base_link"
        msg.actuation.accel_cmd = accel
        msg.actuation.brake_cmd = brake
        msg.actuation.steer_cmd = steer_cmd
        self._actuation_cmd_pub.publish(msg)

    def _current_gear_is_reverse(self) -> bool:
        if self._gear_report is None:
            return False
        return int(getattr(self._gear_report, "report", -1)) in self._gear_reverse_reports

    def _current_gear_is_drive(self) -> bool:
        if self._drive_confirmed_by_command_fallback:
            return True
        if self._gear_report is None:
            return False
        return int(getattr(
            self._gear_report, "report", -1)) == self._gear_drive_command

    def _gear_status_callback(self, msg) -> None:
        self._gear_report = msg
        self._last_gear_report_received_sec = (
            float(self.get_clock().now().nanoseconds) / 1e9)

    def _control_mode_status_callback(self, msg) -> None:
        self._control_mode_report = msg

    def _velocity_status_callback(self, msg) -> None:
        self._velocity_report = msg

    def _awsim_state_callback(self, msg) -> None:
        previous_state = self._awsim_state
        self._awsim_state = getattr(msg, "data", None)
        if should_reset_motion_latch(self._awsim_state):
            if self._has_moved_once:
                self.get_logger().info(
                    "[StuckRecovery] vehicle-motion latch reset by AWSIM "
                    f"state={self._awsim_state}."
                )
            self._has_moved_once = False
            self._stuck_since = None
            self._gnss_history = []
            self._follow_deadlock_since = None
            self._follow_deadlock_start_xy = None
            self._close_obstacle_reverse_requested = False
            self._prepass_retry_after_reverse = False
            self._prepass_retry_lane_idx = None
            self._stuck_reverse_target_distance = None
            self._stuck_reverse_start_heading = None
            self._adaptive_reverse_active = False
            self._adaptive_reverse_static_clearance = None
            self._adaptive_reverse_forward_success_cycles = 0
            self._localization_consistent_since = None
            self._localization_consistent = False
            self._reset_follow_escape()
        if (
            self._initial_start_boost_enabled
            and not self._initial_start_boost_done
            and self._awsim_state == "Start"
            and previous_state != "Start"
            and self._grounded_start_boost_eligible is True
            and not self._initial_start_boost_armed
            and self._initial_start_boost_until is None
        ):
            # Start may re-arm only an already verified eligible layout. Turbo
            # is sent after measured vehicle motion begins, never while the car
            # is still waiting on the grid.
            self._initial_start_boost_armed = True

    def _lane_index_for_position(self, x: float, y: float):
        wp_id = self._car.get_closest_waypoint(x, y)
        wp = self._reference_path.get_waypoint(wp_id)
        normal_angle = wp.psi + math.pi / 2.0
        offset = (
            (x - wp.x) * math.cos(normal_angle)
            + (y - wp.y) * math.sin(normal_angle)
        )
        for lane_idx, (ub_lane, lb_lane) in enumerate(
            self._reference_path.get_lane_bounds(wp_id)
        ):
            if lb_lane <= offset <= ub_lane:
                return lane_idx
        return None

    def _relative_lane_vehicle_samples(self, pose, ego_speed: float):
        """Return current and predicted lane/longitudinal samples for V2X cars."""
        samples = []
        prediction_sec = self._prepass_lane_fallback_prediction_sec
        ego_future_x = pose.x + ego_speed * math.cos(pose.theta) * prediction_sec
        ego_future_y = pose.y + ego_speed * math.sin(pose.theta) * prediction_sec
        for vehicle_id in self._v2x_tracker.active_vehicle_ids():
            buf = self._v2x_tracker._samples.get(vehicle_id)
            if not buf:
                continue
            _, vehicle_x, vehicle_y = buf[-1]
            lane_idx = self._lane_index_for_position(vehicle_x, vehicle_y)
            longitudinal = self._center_longitudinal_between(
                pose.x, pose.y, vehicle_x, vehicle_y)
            if longitudinal is None:
                continue
            samples.append((vehicle_id, lane_idx, longitudinal))

            if prediction_sec <= 0.0 or not self._v2x_tracker.has_velocity_estimate(vehicle_id):
                continue
            velocity_x, velocity_y = self._v2x_tracker.velocity(vehicle_id)
            future_x = vehicle_x + velocity_x * prediction_sec
            future_y = vehicle_y + velocity_y * prediction_sec
            future_lane_idx = self._lane_index_for_position(future_x, future_y)
            future_longitudinal = self._center_longitudinal_between(
                ego_future_x, ego_future_y, future_x, future_y)
            if future_longitudinal is None:
                continue
            samples.append((vehicle_id, future_lane_idx, future_longitudinal))
        return samples

    def _committed_lane_traffic_evidence(self, pose, ego_speed, lane_idx, conflicts, target_id, *, remember=True):
        """Evaluate every possibly relevant other vehicle, retaining unknowns.

        Fixed-distance front/rear categories alone miss fast closers outside
        their window. Current-lane traffic is checked at any arc distance;
        predicted side/cut-in conflicts remain immediate release conditions.
        """
        unsafe, unknown = [], []
        unknown_reasons = {}
        key = (target_id, lane_idx)
        previous_relevant = (self._overtake.traffic_relevant_ids
                             if self._overtake.traffic_key == key else set())
        relevant = set()
        identified = set().union(*(set(conflicts.get(group, ()))
                                  for group in ("front", "side", "rear")))
        candidates = dict.fromkeys((*self._v2x_tracker.active_vehicle_ids(), *sorted(identified)))
        for vehicle_id in candidates:
            if vehicle_id == target_id:
                continue
            buf = self._v2x_tracker._samples.get(vehicle_id)
            if not buf or not all(math.isfinite(float(value)) for value in buf[-1]):
                # No location cannot establish new relevance. Retain hazards
                # already localized to this manoeuvre, including a lost rear.
                if vehicle_id in identified or vehicle_id in previous_relevant:
                    relevant.add(vehicle_id)
                    unknown.append(vehicle_id)
                    unknown_reasons[vehicle_id] = "relevant_position_missing"
                continue
            _, x, y = buf[-1]
            longitudinal = self._center_longitudinal_between(pose.x, pose.y, x, y)
            vx, vy = self._v2x_tracker.velocity(vehicle_id)
            velocity_known = (self._v2x_tracker.has_velocity_estimate(vehicle_id)
                              and math.isfinite(vx) and math.isfinite(vy))
            # Bound relevance before treating an unknown lane/speed as a
            # corridor hazard. The radius covers prediction travel AND the
            # worst closing-speed braking gap. It is not a fixed rear window.
            other_bound = (math.hypot(vx, vy) if velocity_known
                           else self._v2x_tracker._v_max_safety)
            closing_bound = abs(float(ego_speed)) + other_bound
            reach = (self._parallel_ego_half_length + self._parallel_vehicle_half_length
                     + self._moving_emergency_desired_distance
                     + closing_bound * max(self._prepass_lane_fallback_prediction_sec,
                                           self._moving_emergency_reaction_sec)
                     + closing_bound ** 2 / (2.0 * max(
                         self._moving_emergency_available_deceleration, 0.1)))
            # Euclidean separation is a lower bound on travel distance, so
            # rejecting a car outside this radius also works across lap wrap.
            if (vehicle_id not in identified and math.isfinite(reach)
                    and math.hypot(x - pose.x, y - pose.y) > reach):
                continue
            current_lane = self._lane_index_for_position(x, y)
            if current_lane is not None and current_lane != lane_idx and vehicle_id not in identified:
                continue
            relevant.add(vehicle_id)
            if vehicle_id in conflicts.get("side", ()):
                unsafe.append(vehicle_id)
                continue
            if current_lane is None:
                unknown.append(vehicle_id)
                unknown_reasons[vehicle_id] = "nearby_lane_unknown"
                continue
            if longitudinal is None or not math.isfinite(float(longitudinal)):
                unknown.append(vehicle_id)
                unknown_reasons[vehicle_id] = "relevant_distance_unknown"
                continue
            if not velocity_known:
                unknown.append(vehicle_id)
                unknown_reasons[vehicle_id] = "relevant_velocity_unknown"
                continue
            if not all(math.isfinite(float(value)) for value in (ego_speed, pose.theta)):
                unknown.append(vehicle_id)
                unknown_reasons[vehicle_id] = "ego_motion_unknown"
                continue
            wp = self._reference_pathN_center.get_waypoint(
                self._carN_center.get_closest_waypoint(x, y))
            if not math.isfinite(float(wp.psi)):
                unknown.append(vehicle_id)
                unknown_reasons[vehicle_id] = "course_heading_unknown"
                continue
            other_speed = vx * math.cos(wp.psi) + vy * math.sin(wp.psi)
            if dynamic_longitudinal_conflict_unsafe(
                longitudinal=longitudinal, ego_speed=float(ego_speed), other_speed=other_speed,
                body_length_sum=self._parallel_ego_half_length + self._parallel_vehicle_half_length,
                desired_gap=self._moving_emergency_desired_distance,
                reaction_sec=self._moving_emergency_reaction_sec,
                available_deceleration=self._moving_emergency_available_deceleration,
            ):
                unsafe.append(vehicle_id)
        if remember:
            self._overtake.traffic_key = key
            self._overtake.traffic_relevant_ids = relevant
            self._committed_lane_unknown_reasons = unknown_reasons
        return tuple(unsafe), tuple(unknown)

    def _committed_target_body_overlap(self, pose, target_id):
        target = collision.target_body(self, target_id)
        if target is None:
            return None
        return collision.overlaps(collision.ego_body(self, pose), target, collision.geometry(self))


    def _evaluate_committed_lane_hold(
        self, *, pose, ego_speed, lane_idx, target_id, target_longitudinal,
        live_passage, conflicts, now_sec, legacy_target_hold=False,
    ):
        width_valid = self._lane_horizon_has_vehicle_width(lane_idx)
        overlap = self._committed_target_body_overlap(pose, target_id)
        unsafe, unknown = self._committed_lane_traffic_evidence(
            pose, ego_speed, lane_idx, conflicts, target_id)
        geometry_known = bool(
            overlap is not None and target_longitudinal is not None
            and math.isfinite(float(target_longitudinal))
            and lane_idx in live_passage)
        hybrid = self._overtake.hybrid
        matches = bool(hybrid.vehicle_id == target_id and hybrid.lane_idx == lane_idx
                       and hybrid.length is not None and hybrid.length > 0.0
                       and not hybrid.completed)
        # Sample current signed progress before the reference writer runs later
        # in the same control cycle. Repeating this call at the same WP adds zero.
        progress = (float(self._hybrid_horizon_distances(self._carN_center.wp_id)[0]) / hybrid.length
                    if matches else 0.0)
        alongside = bool(
            geometry_known and abs(target_longitudinal) <= (
                self._parallel_ego_half_length + self._parallel_vehicle_half_length
                + self._parallel_safety_longitudinal_clearance))
        result = self._overtake.passage_hold.evaluate(
            key=(target_id, lane_idx), now_sec=now_sec, lane_width_valid=width_valid,
            target_geometry_known=geometry_known,
            target_clearance_lost=not live_passage.get(lane_idx, False),
            body_overlap=overlap, hybrid_matches=matches, hybrid_progress=progress,
            alongside=alongside, unrelated_unsafe=bool(unsafe or unknown),
            confirm_sec=self._hybrid_passage_loss_confirm_sec,
            lock_ratio=self._hybrid_side_lock_progress_ratio,
            legacy_target_hold=legacy_target_hold)
        if result.motion_blocked or not geometry_known:
            self._collision_evidence_hold = True
        if result.unsafe or result.waiting or result.locked:
            self.get_logger().info(
                f"[OvertakeLaneHold] vehicle={target_id}, lane=L{lane_idx}, "
                f"unsafe={result.unsafe}, waiting={result.waiting}, locked={result.locked}, "
                f"width_valid={width_valid}, progress={progress:.2f}, "
                f"reasons={result.reasons}, traffic_unsafe={unsafe}, traffic_unknown={unknown}, "
                f"unknown_details={self._committed_lane_unknown_reasons}, "
                f"body_overlap={overlap}, longitudinal={target_longitudinal}, "
                f"passage_available={lane_idx in live_passage}, "
                f"ego_body={collision.ego_body(self,pose)}, "
                f"target_body={collision.target_body(self,target_id)}",
                throttle_duration_sec=0.5)
        return result

    def _reverse_rear_is_clear(
        self, pose, ego_speed: float = 0.0, reverse_distance=None
    ) -> bool:
        """Check current and predicted V2X occupancy of the reverse path."""
        reverse_distance = (
            self._prepass_lane_fallback_rear_distance
            if reverse_distance is None else max(float(reverse_distance), 0.0)
        )
        path_lanes = set()
        # Recovery reverses straight by default. Sample its swept center line
        # and retain every lane crossed by that path.
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            distance = reverse_distance * ratio
            lane_idx = self._lane_index_for_position(
                pose.x - distance * math.cos(pose.theta),
                pose.y - distance * math.sin(pose.theta),
            )
            if lane_idx is not None:
                path_lanes.add(lane_idx)

        corridor_half_width = (
            0.5 * float(self._cfg.bicycle_model.width)
            + self._v2x_vehicle_radius
            + 0.20
        )
        prediction_sec = self._prepass_lane_fallback_prediction_sec
        conflicts = []
        for vehicle_id in self._v2x_tracker.active_vehicle_ids():
            buf = self._v2x_tracker._samples.get(vehicle_id)
            if not buf:
                continue
            _, vehicle_x, vehicle_y = buf[-1]
            velocity_x, velocity_y = self._v2x_tracker.velocity(vehicle_id)
            samples = [(vehicle_x, vehicle_y, "current")]
            if prediction_sec > 0.0:
                samples.append((
                    vehicle_x + velocity_x * prediction_sec,
                    vehicle_y + velocity_y * prediction_sec,
                    "predicted",
                ))
            for sample_x, sample_y, sample_kind in samples:
                vehicle_lane = self._lane_index_for_position(sample_x, sample_y)
                if reverse_path_has_vehicle_conflict(
                    ego_x=pose.x,
                    ego_y=pose.y,
                    ego_heading=pose.theta,
                    reverse_distance=reverse_distance,
                    corridor_half_width=corridor_half_width,
                    reverse_path_lanes=path_lanes,
                    vehicle_x=sample_x,
                    vehicle_y=sample_y,
                    vehicle_lane=vehicle_lane,
                ):
                    conflicts.append((vehicle_id, vehicle_lane, sample_kind))
                    break
        self._reverse_rear_conflicts = conflicts
        if getattr(self, "_adaptive_reverse_active", False):
            if self._reverse_vehicle_clear_distance(pose, reverse_distance) + 1e-6 < reverse_distance:
                return False
        return not conflicts

    def _reverse_vehicle_clear_distance(self, pose, maximum):
        """Bound straight reverse by vehicle bodies and closing motion, not centers."""
        maximum = max(float(maximum), 0.0)
        reverse_speed = max(self._stuck_forward_reverse_speed, 0.1)
        reaction = self._moving_emergency_reaction_sec
        horizon = maximum / reverse_speed + reaction
        other_radius = math.hypot(self._parallel_vehicle_half_length, self._v2x_vehicle_radius)
        body_length = self._parallel_ego_half_length + other_radius
        body_gap = (body_length
                    + self._adaptive_reverse_wall_margin
                    + reverse_speed ** 2 / (2.0 * max(self._moving_emergency_available_deceleration, 0.1)))
        half_width = 0.5 * self._cfg.bicycle_model.width + other_radius + 0.20
        heading_x, heading_y = math.cos(pose.theta), math.sin(pose.theta)
        available = maximum
        for vid in self._v2x_tracker.active_vehicle_ids():
            buf = self._v2x_tracker._samples.get(vid)
            if not buf or not all(math.isfinite(float(v)) for v in buf[-1]):
                return 0.0
            _, x, y = buf[-1]
            dx, dy = x - pose.x, y - pose.y
            longitudinal = dx * heading_x + dy * heading_y
            lateral = -dx * heading_y + dy * heading_x
            vx, vy = self._v2x_tracker.velocity(vid)
            if (not self._v2x_tracker.has_velocity_estimate(vid)
                    or not all(math.isfinite(v) for v in (vx, vy))):
                # Unknown nearby traffic cannot be assumed stationary.
                reach = maximum + body_gap + self._v2x_tracker._v_max_safety * horizon
                if math.hypot(dx, dy) <= reach:
                    return 0.0
                continue
            forward_speed = vx * heading_x + vy * heading_y
            lateral_speed = -vx * heading_y + vy * heading_x
            lateral_end = lateral + lateral_speed * horizon
            if min(lateral, lateral_end) > half_width or max(lateral, lateral_end) < -half_width:
                continue
            # A vehicle wholly ahead and not approaching the reverse sweep
            # does not block a retreat. Crossing/approaching traffic is retained.
            nearest_lon = min(longitudinal, longitudinal + forward_speed * horizon)
            if nearest_lon > body_length:
                continue
            if longitudinal >= 0.0:
                return 0.0
            closing = max(forward_speed, 0.0)
            gap = -longitudinal - body_gap - closing * reaction
            limit = gap / (1.0 + closing / reverse_speed)
            available = min(available, max(limit, 0.0))
        return available

    def _reverse_rear_conflict_summary(self) -> str:
        conflicts = getattr(self, "_reverse_rear_conflicts", [])
        if not conflicts:
            return "none"
        return ",".join(
            f"{vehicle_id}:lane={lane_idx}:sample={sample_kind}"
            for vehicle_id, lane_idx, sample_kind in conflicts
        )

    def _update_localization_consistency(self, now_sec: float) -> None:
        """Track fresh GNSS/odometry position agreement continuously."""
        fresh = bool(
            self._odom is not None
            and self._gnss_pose is not None
            and self._last_odom_received_sec is not None
            and self._last_gnss_received_sec is not None
            and now_sec - self._last_odom_received_sec
                <= self._adaptive_reverse_localization_fresh_sec
            and now_sec - self._last_gnss_received_sec
                <= self._adaptive_reverse_localization_fresh_sec
        )
        if fresh:
            odom_position = self._odom.pose.pose.position
            gnss_position = self._gnss_pose.pose.pose.position
            self._localization_position_error = math.hypot(
                float(odom_position.x) - float(gnss_position.x),
                float(odom_position.y) - float(gnss_position.y),
            )
        else:
            self._localization_position_error = math.inf
        condition = bool(
            fresh
            and self._localization_position_error
                <= self._adaptive_reverse_localization_max_error
        )
        self._localization_consistent_since = update_continuous_condition_since(
            self._localization_consistent_since,
            now_sec=now_sec,
            condition=condition,
        )
        self._localization_consistent = continuous_condition_confirmed(
            self._localization_consistent_since,
            now_sec=now_sec,
            confirm_sec=self._adaptive_reverse_localization_confirm_sec,
        )

    def _latched_target_is_primary_v2x_blocker(
        self, pose, ego_speed: float, target_id
    ) -> bool:
        """Require the latched target to be the nearest current lead."""
        leads = [
            (float(longitudinal), vehicle_id)
            for vehicle_id, _, longitudinal
            in self._relative_lane_vehicle_samples(pose, ego_speed)
            if 0.0 < float(longitudinal) <= self._follow_engage_distance
        ]
        if not leads:
            return False
        _, nearest_id = min(leads, key=lambda item: item[0])
        return nearest_id == target_id

    def _static_reverse_clearance(self, pose, max_distance: float) -> float:
        return self._map.static_straight_path_clearance(
            pose.x,
            pose.y,
            pose.theta,
            max_distance,
            self._adaptive_reverse_footprint_radius,
        )

    def _static_current_footprint_is_free(self, pose) -> bool:
        return self._map.static_disk_is_free(
            pose.x, pose.y, self._adaptive_reverse_footprint_radius)

    def _prepare_follow_deadlock_reverse(
        self, *, pose, now_sec: float, ego_speed: float, target_id
    ) -> str:
        """Plan a wall-bounded reverse for a stopped V2X lead."""
        self._stuck_reverse_target_distance = None
        self._adaptive_reverse_active = False
        self._adaptive_reverse_static_clearance = None
        self._adaptive_reverse_forward_success_cycles = 0
        if not self._adaptive_reverse_enabled:
            return "legacy"

        target = self._latched_follow_target_state(pose, now_sec)
        physical_passage, _ = self._latched_target_passage(pose)
        eligible = bool(
            target is not None
            and not target.get("expired", False)
            and target.get("vehicle_id") == target_id
            and is_follow_target_ahead(target.get("longitudinal"))
            and target.get("velocity_valid", False)
            and float(target.get("speed", math.inf))
                < self._stopped_lead_speed_threshold
            and any(physical_passage.get(lane, False) for lane in (0, 2))
            and self._latched_target_is_primary_v2x_blocker(
                pose, ego_speed, target_id)
            and self._localization_consistent
            and self._static_current_footprint_is_free(pose)
            and self._follow_escape_target_prediction_blocked
        )
        if not eligible:
            return "legacy"

        scan_distance = (
            self._adaptive_reverse_max_distance
            + self._adaptive_reverse_wall_margin
        )
        static_clearance = self._static_reverse_clearance(pose, scan_distance)
        self._adaptive_reverse_static_clearance = static_clearance
        available = max(
            static_clearance - self._adaptive_reverse_wall_margin, 0.0)
        if available < self._adaptive_reverse_min_clear_distance:
            self.get_logger().warn(
                "[AdaptiveReverseBlocked] stopped lead caused the deadlock "
                "but the static rear sweep has no safe travel: "
                f"vehicle_id={target_id}, static_clearance="
                f"{static_clearance:.2f}m, wall_margin="
                f"{self._adaptive_reverse_wall_margin:.2f}m, "
                f"localization_error={self._localization_position_error:.2f}m"
            )
            return "blocked"

        target_distance = min(available, self._adaptive_reverse_max_distance)
        mode = (
            "adaptive"
            if available >= self._adaptive_reverse_max_distance
            else "wall_bounded"
        )
        if target_distance < self._adaptive_reverse_min_clear_distance:
            return "blocked"
        vehicle_distance = self._reverse_vehicle_clear_distance(pose, target_distance)
        if vehicle_distance < target_distance:
            target_distance = vehicle_distance
            mode = "vehicle_bounded"
        if (target_distance < self._adaptive_reverse_min_clear_distance
                or not self._reverse_rear_is_clear(
                    pose, ego_speed, reverse_distance=target_distance)):
            self.get_logger().warn(
                "[AdaptiveReverseBlocked] static sweep is free but a V2X "
                "vehicle occupies the planned reverse path: "
                f"vehicle_id={target_id}, target={target_distance:.2f}m, "
                f"conflicts={self._reverse_rear_conflict_summary()}"
            )
            return "blocked"

        self._stuck_reverse_target_distance = target_distance
        self._adaptive_reverse_active = True
        self.get_logger().warn(
            "[AdaptiveReversePlan] planned distance-controlled reverse: "
            f"mode={mode}, vehicle_id={target_id}, lead_longitudinal="
            f"{float(target['longitudinal']):.2f}m, stop_condition="
            f"forward_mpc_{self._adaptive_reverse_forward_success_cycles_required}_cycles, "
            f"distance_limit={target_distance:.2f}m, static_clearance="
            f"{static_clearance:.2f}m, usable={available:.2f}m, "
            f"localization_error={self._localization_position_error:.2f}m"
        )
        return mode

    def _reverse_forward_path_is_valid(self, pose, u) -> bool:
        """Check whether any active reverse may stop for a fresh MPC path."""
        target_id = self._overtake.target_id
        fresh_solution = bool(
            self._mpc.infeasibility_counter == 0
            and self._mpc.current_prediction is not None
            and not self._mpc.used_prediction_fallback
            and not self._mpc.recovery_requested
            and not self._mpc.time_budget_exceeded
            and getattr(self._mpc, "last_solution_accurate", False)
        )
        return bool(
            fresh_solution
            and self._prediction_has_forward_progress(pose, u)
            and (
                target_id is None
                or self._prediction_is_clear_of_vehicle(target_id)
            )
        )

    def _follow_deadlock_position(self, pose):
        """Use raw GNSS for the deadlock movement gate when it is available."""
        if self._gnss_pose is not None:
            position = self._gnss_pose.pose.pose.position
            return float(position.x), float(position.y)
        return float(pose.x), float(pose.y)

    def _reset_follow_escape(self, reason=None) -> None:
        was_active = self._follow_escape_active
        self._follow_deadlock_since = None
        self._follow_deadlock_start_xy = None
        self._follow_escape_active = False
        self._follow_escape_target_id = None
        self._follow_escape_probe_lane_idx = None
        self._follow_escape_probe_started_at = None
        self._follow_escape_probe_success_cycles = 0
        self._follow_escape_attempted_lanes.clear()
        self._follow_escape_forward_active = False
        self._follow_escape_forward_until = None
        self._follow_escape_last_reevaluate_at = None
        self._follow_escape_target_prediction_blocked = False
        if was_active and reason:
            self.get_logger().info(
                f"[FollowDeadlockEscape] released: reason={reason}"
            )

    def _invalidate_follow_escape_probe_for_recovery(self, reason: str) -> None:
        """Discard a FollowEscape decision when another recovery takes over.

        FollowEscape itself remains active so the stopped target retains
        exclusive ownership.  The candidate lane and every consecutive-success
        result are deliberately forgotten: after full-width recovery the lane
        must be selected and verified again from the current vehicle state.
        """
        if not self._follow_escape_active:
            return
        had_probe_state = bool(
            self._follow_escape_probe_lane_idx is not None
            or self._follow_escape_probe_success_cycles > 0
            or self._follow_escape_forward_active
        )
        self._follow_escape_probe_lane_idx = None
        self._follow_escape_probe_started_at = None
        self._follow_escape_probe_success_cycles = 0
        self._follow_escape_attempted_lanes.clear()
        self._follow_escape_forward_active = False
        self._follow_escape_forward_until = None
        self._follow_escape_last_reevaluate_at = None
        self._follow_escape_target_prediction_blocked = False
        if had_probe_state:
            self.get_logger().warn(
                "[FollowDeadlockRecoveryHandoff] discarded the stale escape "
                "lane/probe result; a fresh evaluation is required after "
                f"full-width recovery: reason={reason}"
            )

    def _clear_urgent_overtake_switch_candidate(self) -> None:
        self._urgent_overtake_switch_candidate_id = None
        self._urgent_overtake_switch_candidate_since = None
        self._urgent_overtake_switch_last_seen_at = None

    def _clear_prepass_soft_guidance(self) -> None:
        self._prepass_soft_candidate_lane_idx = None
        self._prepass_soft_pending_lane_idx = None
        self._prepass_soft_pending_since = None
        self._prepass_soft_candidate_last_seen_at = None
        self._prepass_soft_guidance_started_at = None
        self._prepass_soft_guidance_start_e_y = None
        self._prepass_soft_guidance_ramp_sec = None

    def _clear_consecutive_overtake_handoff(self) -> None:
        self._consecutive_overtake_handoff_target_id = None
        self._consecutive_overtake_handoff_lane_idx = None
        self._consecutive_overtake_handoff_started_at = None

    def _reset_overtake_state_for_target_change(
        self, new_target_id, *, reason: str
    ) -> None:
        """Drop every passing-side decision owned by a previous vehicle."""
        old_target_id = (
            self._overtake.target_id
            if self._overtake.target_id is not None
            else self._forced_overtake_vehicle_id
        )
        old_lane_idx = self._overtake.requested_lane
        if (
            self._follow_escape_active
            and self._follow_escape_target_id != new_target_id
        ):
            self._reset_follow_escape(
                "overtake target changed before escape completed"
            )

        self._overtake.release_target()
        self._overtake.committed = False
        self._overtake_completed_target_id = None
        self._reset_outer_lane_progress()
        self._reset_overtake_commit_probe()
        self._clear_committed_shadow_verification()
        self._clear_consecutive_overtake_handoff()
        self._overtake_switch_candidate_id = None
        self._overtake_switch_candidate_since = None
        self._clear_urgent_overtake_switch_candidate()
        self._forced_overtake_vehicle_id = new_target_id
        self._outer_lane_released_vehicle_id = None
        self._follow_latched_cache = None
        self._prepass_fallback_lane_idx = None
        self._prepass_fallback_blocked = False
        self._prepass_fallback_follow_active = False
        self._prepass_follow_last_retry_at = None
        self._prepass_dynamic_conflict_speed_limit = None
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_recovery_started_at = None
        self._prepass_target_behind_since = None
        self._prepass_failed_lane_idx = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_fallback_commit_success_since = None
        self._clear_prepass_soft_guidance()
        self._prepass_attempted_outer_lanes.clear()
        self._prepass_retry_after_reverse = False
        self._prepass_retry_lane_idx = None
        self._prepass_reverse_motion_started = False
        self._prepass_reverse_start_xy = None
        self._prepass_reverse_distance = 0.0
        self._close_obstacle_reverse_requested = False
        self._target_lane_idx = None
        self._mpc.osqp_initialized = False
        self._mpc.current_prediction = None
        self.get_logger().warn(
            "[OvertakeTargetChange] discarded the previous target's latch "
            "and passing side before re-evaluation: "
            f"old_vehicle_id={old_target_id}, old_lane=L{old_lane_idx}, "
            f"new_vehicle_id={new_target_id}, reason={reason}"
        )

    def _select_follow_escape_lane(self, pose, ego_speed: float):
        """Prefer a physically passable outer lane, then collision-free L1."""
        outer_order = ordered_outer_lane_candidates(self._overtake.requested_lane)
        physical_passage, _ = self._latched_target_passage(pose)
        samples = self._relative_lane_vehicle_samples(pose, ego_speed)
        all_conflicts = {
            lane_idx: classify_lane_conflicts(
                lane_idx,
                samples,
                front_distance=self._prepass_lane_fallback_front_distance,
                side_distance=self._prepass_lane_fallback_side_distance,
                rear_distance=self._prepass_lane_fallback_rear_distance,
            )
            for lane_idx in (0, 1, 2)
        }
        for lane_idx in outer_order:
            if (
                lane_idx not in self._follow_escape_attempted_lanes
                and physical_passage.get(lane_idx, False)
                and lane_conflicts_are_clear(all_conflicts[lane_idx])
            ):
                selected_lane_idx = self._apply_l2_restricted_zone_policy(
                    lane_idx,
                    target_vehicle_id=self._follow_escape_target_id,
                    physical_passage=physical_passage,
                    conflicts_by_lane=all_conflicts,
                )
                if selected_lane_idx == lane_idx and lane_idx in (0, 2):
                    if (self._lane_horizon_has_vehicle_width(lane_idx)
                            and not (lane_idx == 0 and self._waypoint_in_configured_zones(
                                int(self._carN_center.wp_id), self._l0_entry_prohibited_zones))):
                        unsafe, unknown = self._committed_lane_traffic_evidence(
                            pose, ego_speed, lane_idx, all_conflicts[lane_idx],
                            None, remember=False)
                        if not unsafe and not unknown:
                            return lane_idx, all_conflicts
        if (
            1 not in self._follow_escape_attempted_lanes
            and lane_conflicts_are_clear(all_conflicts[1])
        ):
            return 1, all_conflicts
        return None, all_conflicts

    def _waypoint_in_configured_zones(self, wp_id: int, zones) -> bool:
        """Return whether a circular Center waypoint lies in any zone."""
        return any(
            (start_wp <= wp_id <= end_wp)
            if start_wp <= end_wp
            else (wp_id >= start_wp or wp_id <= end_wp)
            for start_wp, end_wp in zones
        )

    def _l2_inward_offsets(self, center_wp: int, count: int):
        """Return L1-directed L2 objective offsets for Center waypoints."""
        waypoint_count = len(self._reference_pathN_center.waypoints)
        offsets = []
        for step in range(max(int(count), 0)):
            wp_id = (int(center_wp) + step) % waypoint_count
            inward = 0.0
            for start_wp, end_wp, offset_m in self._l2_inward_offset_zones:
                if self._waypoint_in_configured_zones(
                    wp_id, ((start_wp, end_wp),)
                ):
                    inward = max(inward, offset_m)
            offsets.append(-inward)  # Positive e_y is L2; negative is inward.
        return np.asarray(offsets, dtype=float)

    def _update_l2_target_objective_offsets(self, mpc, center_wp: int):
        """Apply configured L2 objective offsets without widening bounds."""
        if not self._l2_inward_offset_zones:
            mpc.set_target_lane_lateral_offsets()
            return
        mpc.set_target_lane_lateral_offsets(self._l2_inward_offsets(
            center_wp, mpc.N + 1))

    def _update_full_width_l1_objective_offsets(self, mpc, center_wp: int):
        """Move only a full-width objective toward L1 in local zones."""
        if (
            not self._full_width_l1_offset_zones
            or self._reference_pathN_center.target_lane_idx is not None
        ):
            mpc.set_full_width_l1_offset_limits()
            return
        waypoint_count = len(self._reference_pathN_center.waypoints)
        limits = []
        for step in range(mpc.N + 1):
            wp_id = (int(center_wp) + step) % waypoint_count
            limit = 0.0
            for start_wp, end_wp, offset_m in self._full_width_l1_offset_zones:
                if self._waypoint_in_configured_zones(
                    wp_id, ((start_wp, end_wp),)
                ):
                    limit = max(limit, offset_m)
            limits.append(limit)
        mpc.set_full_width_l1_offset_limits(limits)

    def _update_full_width_l0_objective_offsets(self, mpc, center_wp: int):
        """Move only a full-width objective lightly toward L0 in local zones."""
        if (
            not self._full_width_l0_offset_zones
            or self._reference_pathN_center.target_lane_idx is not None
        ):
            mpc.set_full_width_l0_offset_limits()
            return
        waypoint_count = len(self._reference_pathN_center.waypoints)
        limits = []
        for step in range(mpc.N + 1):
            wp_id = (int(center_wp) + step) % waypoint_count
            limit = 0.0
            for start_wp, end_wp, offset_m in self._full_width_l0_offset_zones:
                if self._waypoint_in_configured_zones(
                    wp_id, ((start_wp, end_wp),)
                ):
                    limit = max(limit, offset_m)
            limits.append(limit)
        mpc.set_full_width_l0_offset_limits(limits)

    def _l2_inward_targets(self, center_wp: int):
        """Build the offset L2 xr sequence used during soft guidance."""
        offsets = self._l2_inward_offsets(
            center_wp, self._mpcN_center.N + 1)
        return np.asarray([
            self._mpcN_center._compute_lane_center(center_wp + step, 2)
            + offsets[step]
            for step in range(self._mpcN_center.N + 1)
        ], dtype=float)

    def _l2_restricted_slow_override(self, vehicle_id) -> bool:
        """Allow L2 locally only for a fresh lead speed below the threshold."""
        if (
            vehicle_id is None
            or not self._v2x_tracker.has_velocity_estimate(vehicle_id)
        ):
            return False
        velocity_x, velocity_y = self._v2x_tracker.velocity(vehicle_id)
        return (
            math.hypot(velocity_x, velocity_y)
            < self._l2_entry_restricted_override_speed
        )

    def _slow_lead_commit_distance_at(
        self, center_wp: int, *, lead_speed=None, lead_is_stationary=False,
    ) -> float:
        """Distance eligibility only; the existing strict Shadow gates still apply."""
        ordinary = (
            self._late_defense_slow_lead_overtake_commit_distance
            if self._waypoint_in_configured_zones(int(center_wp), self._l2_entry_restricted_zones)
            else self._slow_lead_overtake_commit_distance)
        if lead_speed is None:
            return ordinary
        return slow_lead_commit_distance(
            lead_is_stationary=lead_is_stationary, lead_speed=lead_speed,
            ultra_slow_speed_threshold=self._ultra_slow_early_commit_speed,
            early_commit_distance=self._slow_lead_overtake_prepare_distance,
            ordinary_slow_commit_distance=ordinary)

    def _lane_horizon_has_vehicle_width(self, lane_idx: int) -> bool:
        """Return whether a lane alone can contain ego over the MPC horizon."""
        lane_widths = []
        for offset in range(self._mpcN_center.N + 1):
            lanes = self._reference_pathN_center.get_lane_bounds(
                int(self._carN_center.wp_id) + offset)
            if lane_idx >= len(lanes):
                lane_widths.append(0.0)
                continue
            lane_ub, lane_lb = lanes[lane_idx]
            lane_widths.append(float(lane_ub) - float(lane_lb))
        if not lane_widths or not all(math.isfinite(width) for width in lane_widths):
            return False
        result = evaluate_lane_width_samples(
            lane_widths,
            required_width=float(self._cfg.bicycle_model.width),
            tolerance=self._passage_lane_width_tolerance,
            max_consecutive_tolerated=(
                self._passage_lane_width_tolerance_points),
        )
        return bool(result["passable"])

    def _apply_l2_restricted_zone_policy(
        self, candidate_lane_idx, *, target_vehicle_id,
        physical_passage, conflicts_by_lane, center_wp=None,
    ):
        """Prefer L0, otherwise L1, while locally prohibiting ordinary L2."""
        wp_id = (
            int(self._carN_center.wp_id)
            if center_wp is None else int(center_wp)
        )
        restriction_active = self._waypoint_in_configured_zones(
            wp_id, self._l2_entry_restricted_zones)
        slow_override = self._l2_restricted_slow_override(target_vehicle_id)
        l0_conflicts = conflicts_by_lane.get(0, {})
        # In the final defensive section, lack of room to pass the lead on L0
        # must not be confused with lack of room to *follow* it on L0.  The
        # ordinary passage result includes target-to-boundary clearance and
        # therefore becomes false when the lead itself occupies L0.  Permit
        # the L0 follow state whenever the lane's own horizon is wide enough;
        # side traffic and genuine lane-width collapse remain blockers.
        l0_follow_geometry_override = bool(
            restriction_active
            and not slow_override
            and target_vehicle_id is not None
            and not physical_passage.get(0, False)
            and not l0_conflicts.get("side")
            and self._lane_horizon_has_vehicle_width(0)
        )
        selected = select_l2_restricted_zone_lane(
            candidate_lane_idx,
            restriction_active=restriction_active,
            slow_lead_override=slow_override,
            l0_physically_passable=(
                physical_passage.get(0, False)
                or l0_follow_geometry_override
            ),
            l0_conflicts=l0_conflicts,
        )
        if l0_follow_geometry_override and selected == 0:
            self.get_logger().info(
                "[L0ZoneFollowGeometryOverride] L0 passing clearance is "
                "unavailable, but the lane itself has vehicle width; "
                "selecting L0 to follow the lead instead of returning to L1: "
                f"center_wp={wp_id}, vehicle_id={target_vehicle_id}",
                throttle_duration_sec=1.0,
            )
        if restriction_active and not slow_override and selected != candidate_lane_idx:
            self.get_logger().info(
                "[L2EntryRestricted] ordinary lead cannot create an L2 "
                "request in this section; applying L0->L1 priority while "
                "ignoring L0 rear traffic: "
                f"center_wp={wp_id}, vehicle_id={target_vehicle_id}, "
                f"requested=L{candidate_lane_idx}, selected=L{selected}, "
                f"l0_passable={physical_passage.get(0, False)}, "
                f"l0_conflicts={conflicts_by_lane.get(0, {})}",
                throttle_duration_sec=1.0,
            )
        return selected

    def _prediction_collision_with_vehicle(self, target_id):
        if (self._mpc.current_prediction is None or self._mpc.infeasibility_counter != 0
                or self._mpc.used_prediction_fallback or self._mpc.recovery_requested):
            return None
        target = collision.target_body(self, target_id)
        if target is None or not target.position_valid:
            return None
        xs, ys = self._mpc.current_prediction
        if not xs or len(xs) != len(ys):
            return None
        vx, vy = self._v2x_tracker.velocity(target_id)
        if not self._v2x_tracker.has_velocity_estimate(target_id):
            return None
        for i in range(len(xs)):
            t = self._v2x_t_samples[min(i+2, len(self._v2x_t_samples)-1)]
            predicted = dataclasses.replace(target, x=target.x+vx*t, y=target.y+vy*t)
            result = collision.overlaps(collision.predicted_ego(self,xs,ys,i), predicted, collision.geometry(self))
            if result is not False:
                return result
        return False


    def _prediction_is_clear_of_vehicle(self, target_id) -> bool:
        return self._prediction_collision_with_vehicle(target_id) is False


    def _slow_pass_spacing_release_ids(self, pose):
        """Per-cycle proof for slow traffic, including the unfinished lane transition.

        Do not require present lateral separation: prove the connecting swept
        path instead. All observed traffic must be clear before releasing any
        spacing cap; recovery, physical bounds and later safety retain priority.
        """
        def reject(reason):
            self._slow_pass_release_reason = reason
            return set()
        self._slow_pass_release_reason = 'no_slow_vehicle_with_terminal_clearance'
        mpc = self._mpc
        lane = self._reference_path.target_lane_idx
        if (self._follow_only or not self._reference_path.is_overtaking
                or lane not in (0, 2) or lane != self._overtake.requested_lane
                or self._steering_fallback_armed or self._mpc_safety_recovery_active
                or self._prepass_fallback_recovery_active or self._parallel_abort_active
                or self._close_obstacle_reverse_requested
                or self._stuck_recovery_until is not None
                or mpc.current_prediction is None or mpc.infeasibility_counter != 0
                or mpc.used_prediction_fallback or mpc.recovery_requested
                or mpc.time_budget_exceeded or not mpc.last_solution_accurate
                or str(mpc.last_solution_status).lower() != 'solved'
                or getattr(mpc,'_constraint_target_lane',None) != lane
                or getattr(mpc,'_constraint_lane_relaxation',math.inf) != 0.0
                or self._outer_lane_constraint_is_collapsed(lane)[0]):
            return reject(f'controller_state: lane={lane}, prepass={self._prepass_fallback_recovery_active}, mpc_recovery={self._mpc_safety_recovery_active}, status={mpc.last_solution_status}')
        context = self._live_prediction_context
        if (context is None or context[0] is not mpc
                or context[1] is not mpc.current_prediction or context[2] != lane
                or context[3] is not self._reference_path or context[4] is not self._v2x_tracker
                or not self._lane_horizon_has_vehicle_width(lane)):
            return reject('prediction_context_or_width')
        # Visualization starts at waypoint +2. Use the actual near states for
        # the swept connection, bound to this exact successful solve.
        near = getattr(mpc,'collision_prediction_context',None)
        xs, ys = (near[1] if near is not None and near[0] is mpc.current_prediction
                  else mpc.current_prediction)
        if (len(xs) < 2 or len(xs) != len(ys)
                or not all(math.isfinite(float(v)) for v in (*xs,*ys))):
            return reject('invalid_prediction')
        ego = collision.ego_body(self,pose)
        path = [ego] + [collision.predicted_ego(self,xs,ys,i) for i in range(len(xs))]
        # Reject disconnected/stationary predictions, including stale spatial solutions.
        if (math.hypot(path[1].x-ego.x,path[1].y-ego.y) > 1.0
                or math.hypot(path[-1].x-ego.x,path[-1].y-ego.y) < 0.5):
            return reject('disconnected_or_stationary_prediction')
        times = [0.] + [float(self._v2x_t_samples[min(i+2,len(self._v2x_t_samples)-1)])
                        for i in range(len(xs))]
        approved = set()
        geometry = collision.geometry(self)
        for vid in self._v2x_tracker.active_vehicle_ids():
            target = collision.target_body(self,vid)
            if (target is None or not target.position_valid
                    or not self._v2x_tracker.has_velocity_estimate(vid)):
                return reject(f'observation_unknown:{vid}')
            velocity = self._v2x_tracker.velocity(vid)
            age = max(self._collision_now-target.stamp,0.)
            if not collision.swept_path_clear(path,[t+age for t in times],target,
                                             velocity,geometry):
                return reject(f'swept_path_not_clear:{vid}')
            center = self._center_path_collision_prediction(vid)
            if center is None or center.get('collision') is not False:
                return reject(f'center_prediction_not_clear:{vid}')
            speed = math.hypot(*velocity)
            if speed > self._strict_shadow_commit_creep_max_target_speed:
                continue
            passage, _ = self._vehicle_passage(vid,pose)
            if not passage.get(lane,False):
                continue
            # Horizon must reach lateral clearance or pass the vehicle; a short
            # collision-free prefix ending behind the blocker cannot release ACC.
            end = path[-1]
            dx = target.x+velocity[0]*(times[-1]+age)-end.x
            dy = target.y+velocity[1]*(times[-1]+age)-end.y
            longitudinal = dx*math.cos(end.yaw)+dy*math.sin(end.yaw)
            lateral = abs(-dx*math.sin(end.yaw)+dy*math.cos(end.yaw))
            ego_long,ego_lat = collision.extents(end,geometry,end.yaw)
            other_long,other_lat = collision.extents(target,geometry,end.yaw)
            if (lateral > ego_lat+other_lat+self._parallel_critical_clearance
                    or longitudinal < -ego_long-other_long-self._parallel_critical_clearance):
                approved.add(vid)
        return approved


    def _fresh_outer_prediction_releases_center_stop(self, vehicle_id, envelope):
        """Release only this vehicle's stale Center hazard with live motion proof."""
        mpc = self._mpc
        if (
            envelope is None or envelope.get("rectangles_overlap", True)
            or not math.isfinite(float(envelope.get("lateral_gap", math.nan)))
            or float(envelope["lateral_gap"]) < self._parallel_warning_clearance
            or not self._reference_path.is_overtaking
            or self._reference_path.target_lane_idx not in (0, 2)
            or mpc.current_prediction is None
            or mpc.infeasibility_counter != 0
            or mpc.used_prediction_fallback or mpc.recovery_requested
            or mpc.time_budget_exceeded
            or not mpc.last_solution_accurate
            or str(mpc.last_solution_status).lower() != "solved"
            or self._steering_fallback_armed
            or self._mpc_safety_recovery_active
            or not self._v2x_tracker.has_velocity_estimate(vehicle_id)
        ):
            return False
        context = self._live_prediction_context
        if (context is None or context[0] is not mpc
                or context[1] is not mpc.current_prediction
                or context[2] != self._reference_path.target_lane_idx
                or context[3] is not self._reference_path
                or context[4] is not self._v2x_tracker):
            return False
        xs, ys = mpc.current_prediction
        if (len(xs) < 2 or len(xs) != len(ys)
                or not all(math.isfinite(float(v)) for v in (*xs, *ys))
                or math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < 0.5
                or not self._prediction_is_clear_of_vehicle(vehicle_id)):
            return False
        center_prediction = self._center_path_collision_prediction(vehicle_id)
        return bool(center_prediction is not None
                    and center_prediction.get("collision") is False)

    def _center_path_collision_prediction(self, target_id):
        """Predict an opponent along Center and check ego/opponent OBBs.

        V2X velocity is used only for progress speed.  Keeping its current
        Center lateral offset while advancing on the closed Center arc makes
        the predicted position and yaw follow the bend instead of continuing
        along a straight world-frame velocity vector.
        """
        if (
            target_id is None
            or self._mpc.current_prediction is None
            or self._mpc.infeasibility_counter != 0
            or self._mpc.used_prediction_fallback
            or self._mpc.recovery_requested
        ):
            return None
        target_body = collision.target_body(self, target_id)
        if target_body is None or not target_body.position_valid:
            return None
        target_buf = self._v2x_tracker._samples.get(target_id)
        if not target_buf:
            return None
        _, target_x, target_y = target_buf[-1]
        target_frenet = self._center_frenet(target_x, target_y)
        if target_frenet is None or self._center_arc_total_length <= 1e-6:
            return None
        target_vx, target_vy = self._v2x_tracker.velocity(target_id)
        target_wp_id = self._carN_center.get_closest_waypoint(
            target_x, target_y)
        target_center_heading = float(
            self._reference_pathN_center.get_waypoint(target_wp_id).psi)
        # Signed Center progress also supports reversing opponents. Body yaw
        # remains independent of the direction of travel.
        target_speed = (
            target_vx * math.cos(target_center_heading)
            + target_vy * math.sin(target_center_heading)
        )
        if not self._v2x_tracker.has_velocity_estimate(target_id):
            return None

        pred_x, pred_y = self._mpc.current_prediction
        if not pred_x or len(pred_x) != len(pred_y):
            return None
        points = self._center_arc_points
        cumulative = self._center_arc_cumulative
        total = float(self._center_arc_total_length)
        target_s, target_lateral = target_frenet

        def center_pose_at_s(path_s):
            wrapped_s = float(path_s) % total
            segment_index = int(np.searchsorted(
                cumulative, wrapped_s, side="right") - 1)
            segment_index = max(0, min(segment_index, len(points) - 1))
            next_index = (segment_index + 1) % len(points)
            x0, y0 = points[segment_index]
            x1, y1 = points[next_index]
            segment_start = float(cumulative[segment_index])
            segment_length = float(cumulative[segment_index + 1]) - segment_start
            ratio = (
                (wrapped_s - segment_start) / segment_length
                if segment_length > 1e-9 else 0.0
            )
            heading = math.atan2(y1 - y0, x1 - x0)
            center_x = float(x0) + ratio * (float(x1) - float(x0))
            center_y = float(y0) + ratio * (float(y1) - float(y0))
            # Positive Frenet lateral is left of the Center tangent.
            return (
                center_x - math.sin(heading) * float(target_lateral),
                center_y + math.cos(heading) * float(target_lateral),
                heading,
            )

        prediction_times = [
            self._v2x_t_samples[min(index + 2,
                                   len(self._v2x_t_samples) - 1)]
            for index in range(len(pred_x))
        ]
        for index, (ego_x, ego_y, prediction_time) in enumerate(zip(
            pred_x, pred_y, prediction_times
        )):
            if prediction_time > (
                self._center_path_collision_prediction_horizon_sec
            ):
                break
            path_x, path_y, tangent = center_pose_at_s(target_s + target_speed * prediction_time)
            base_x, base_y, base_tangent = center_pose_at_s(target_s)
            rotation = math.atan2(math.sin(tangent-base_tangent),math.cos(tangent-base_tangent)) if abs(target_speed) > 0.15 else 0.0
            target_heading = target_body.yaw + rotation if target_body.yaw_valid else None
            predicted_target_x = target_body.x + path_x-base_x
            predicted_target_y = target_body.y + path_y-base_y
            predicted = dataclasses.replace(target_body, x=predicted_target_x, y=predicted_target_y, yaw=target_heading)
            overlap = collision.overlaps(collision.predicted_ego(self,pred_x,pred_y,index),
                predicted, collision.geometry(self), margin=self._center_path_collision_prediction_margin)
            if overlap is None:
                return None
            if overlap:
                return {
                    "collision": True,
                    "time": float(prediction_time),
                    "index": index,
                    "target_x": predicted_target_x,
                    "target_y": predicted_target_y,
                    "target_heading": target_heading,
                }
        return {"collision": False}

    @staticmethod
    def _oriented_vehicle_rectangles_overlap(ego_x, ego_y, ego_heading,
                                             target_x, target_y, target_heading, margin=0.0):
        """Compatibility entry point using the shared physical body dimensions."""
        return collision.overlaps(collision.body_pose(ego_x,ego_y,ego_heading,0.),
            collision.body_pose(target_x,target_y,target_heading,0.),collision.BodyGeometry(),margin)

    def _current_center_envelopes_are_separated(self, pose, target_id):
        ego = collision.ego_body(self,pose)
        target = collision.target_body(self,target_id)
        if target is None:
            return False, None
        result = collision.overlaps(ego,target,collision.geometry(self))
        if result is None:
            return False, None
        ego_frenet = self._center_frenet(ego.x,ego.y)
        target_frenet = self._center_frenet(target.x,target.y)
        if ego_frenet is None or target_frenet is None:
            return False, None
        arc_delta = signed_closed_path_arc_distance(ego_frenet[0],target_frenet[0],self._center_arc_total_length)
        if arc_delta is None or arc_delta <= 0.0:
            return False, None
        wp = self._carN_center.get_closest_waypoint(target.x,target.y)
        heading = float(self._reference_pathN_center.get_waypoint(wp).psi)
        ego_lon,ego_lat = collision.extents(ego,collision.geometry(self),heading)
        target_lon,target_lat = collision.extents(target,collision.geometry(self),heading)
        arc_gap = arc_delta-ego_lon-target_lon
        lateral_gap = abs(target_frenet[1]-ego_frenet[1])-ego_lat-target_lat
        clearance_result = collision.overlaps(ego,target,collision.geometry(self),
                                               margin=self._parallel_critical_clearance)
        return bool(not result and clearance_result is False), {
            'arc_delta':float(arc_delta),'arc_gap':float(arc_gap),'lateral_gap':float(lateral_gap),
            'rectangles_overlap':result, 'yaw_known':ego.yaw_valid and target.yaw_valid,
            'overlap_kind':'possible' if (not target.yaw_valid or target.uncertainty or ego.uncertainty) else 'body',
            'ego_body':ego,'target_body':target}


    def _stationary_lane_group(self, pose, ego_speed, lane):
        """Current stopped blockers sharing one physical passage; never motion proof."""
        if lane not in (0, 2) or not self._lane_horizon_has_vehicle_width(lane):
            return {}
        wp = int(self._carN_center.wp_id)
        if lane == 0 and self._waypoint_in_configured_zones(wp, self._l0_entry_prohibited_zones):
            return {}
        members = {}
        blocked_at = math.inf
        samples = self._relative_lane_vehicle_samples(pose, ego_speed)
        conflicts = classify_lane_conflicts(
            lane, samples, front_distance=self._prepass_lane_fallback_front_distance,
            side_distance=self._prepass_lane_fallback_side_distance,
            rear_distance=self._prepass_lane_fallback_rear_distance)
        for vid in self._v2x_tracker.active_vehicle_ids():
            buf = self._v2x_tracker._samples.get(vid)
            if not buf or not self._v2x_tracker.has_velocity_estimate(vid):
                continue
            _, x, y = buf[-1]
            lon = self._center_longitudinal_between(pose.x, pose.y, x, y)
            speed = math.hypot(*self._v2x_tracker.velocity(vid))
            if (lon is None or not math.isfinite(lon) or not math.isfinite(speed)
                    or not -3.0 <= lon < self._overtake_latch_max_distance
                    or speed > self._stopped_lead_speed_threshold):
                continue
            # Stopped cars occupying the destination cannot become members
            # just because some other car has room on that side.
            passage, _ = self._vehicle_passage(vid, pose)
            if self._committed_target_body_overlap(pose, vid) is not False:
                return {}
            if not passage.get(lane, False):
                blocked_at = min(blocked_at, lon)
            else:
                members[vid] = lon
        if math.isfinite(blocked_at):
            # Keep only a prefix that can be fully passed before the next
            # blockage's stopping line. The excluded car stays in all MPC,
            # traffic and emergency evaluations; it gets no creep exemption.
            speed = max(abs(float(ego_speed)), self._hybrid_escape_creep_speed)
            body = self._parallel_ego_half_length + self._parallel_vehicle_half_length
            stopping_gap = (body + self._moving_emergency_desired_distance
                            + speed * self._moving_emergency_reaction_sec
                            + speed ** 2 / (2.0 * max(self._moving_emergency_available_deceleration, 0.1)))
            if blocked_at <= stopping_gap:
                return {}
            last_passable = blocked_at - stopping_gap - body - self._parallel_critical_clearance
            members = {vid: lon for vid, lon in members.items() if lon < last_passable}
        if not members:
            return {}
        unsafe, unknown = self._committed_lane_traffic_evidence(pose, ego_speed, lane, conflicts, self._overtake.target_id, remember=False)
        if unsafe or unknown or any(conflicts.get(key) for key in ("front", "side", "rear")):
            return {}
        policy_lane = self._apply_l2_restricted_zone_policy(
            lane, target_vehicle_id=self._overtake.target_id,
            physical_passage={lane: True}, conflicts_by_lane={lane: conflicts}, center_wp=wp)
        return members if policy_lane == lane else {}

    def _release_stationary_parallel_abort(self, pose, ego_speed):
        """Transfer a stopped yield to ordinary Shadow acquisition, not forward control."""
        target = self._parallel_abort_vehicle_id
        if (not self._parallel_abort_active or self._follow_only
                or self._mpc_safety_recovery_active
                or self._post_reverse_full_width_recovery_active
                or not self._v2x_tracker.has_velocity_estimate(target)
                or math.hypot(*self._v2x_tracker.velocity(target)) > self._stopped_lead_speed_threshold):
            return False
        preferred = getattr(self, '_parallel_abort_previous_lane', None)
        for lane in dict.fromkeys((preferred, 0, 2)):
            if lane not in (0, 2):
                continue
            members = self._stationary_lane_group(pose, ego_speed, lane)
            if target not in members:
                continue
            self._parallel_abort_active = False
            self._parallel_abort_vehicle_id = None
            self._parallel_abort_target_lane_idx = None
            self._parallel_timer_vehicle_id = None
            self._parallel_start_time = None
            self._cancel_l1_rejoin_for_overtake()
            self.get_logger().info(
                f"[StationaryAbortReacquire] stopped group={tuple(members)}, candidate=L{lane}; "
                "yield released for fresh Shadow acquisition; no motion permission")
            return True
        return False

    def _hybrid_escape_speed(self, pose, ego_speed, vehicle_id=None):
        """Permission for this target's ACC/emergency cap only, not a global floor."""
        hybrid = self._overtake.hybrid
        target = hybrid.vehicle_id if vehicle_id is None else vehicle_id
        lane = hybrid.lane_idx
        grouped = (target != hybrid.vehicle_id and target in self._stationary_lane_group(pose, ego_speed, lane))
        if (target is None or lane not in (0, 2)
                or (target != self._overtake.target_id and not grouped)
                or lane != self._overtake.requested_lane
                or lane != self._reference_path.target_lane_idx
                or hybrid.paused or hybrid.completed
                or not self._reference_path.is_overtaking
                or self._follow_only
                or not self._v2x_tracker.has_velocity_estimate(target)):
            return 0.0
        _, envelope = self._current_center_envelopes_are_separated(pose, target)
        if envelope is None:
            return 0.0
        passage, _ = self._vehicle_passage(target, pose)
        conflicts = classify_lane_conflicts(
            lane, self._relative_lane_vehicle_samples(pose, ego_speed),
            front_distance=self._prepass_lane_fallback_front_distance,
            side_distance=self._prepass_lane_fallback_side_distance,
            rear_distance=self._prepass_lane_fallback_rear_distance)
        unsafe, unknown = self._committed_lane_traffic_evidence(
            pose, ego_speed, lane, conflicts, target, remember=False)
        speed = math.hypot(*self._v2x_tracker.velocity(target))
        allowed = hybrid_lateral_escape_creep_allowed(
            target_matches=True, transition_active=True,
            target_is_slow=math.isfinite(speed) and speed <= self._strict_shadow_commit_creep_max_target_speed,
            rectangles_overlap=envelope.get("rectangles_overlap", True),
            lateral_body_gap=envelope.get("lateral_gap", -math.inf),
            minimum_lateral_body_gap=self._hybrid_escape_min_lateral_gap,
            candidate_passable=bool(passage.get(lane, False)
                                    and self._lane_horizon_has_vehicle_width(lane)
                                    and not unsafe and not unknown),
            candidate_conflicts=conflicts)
        return self._hybrid_escape_creep_speed if allowed else 0.0

    def _moving_vehicle_will_clear_after_brief_conflict(
        self, target_id, pose, ego_speed: float
    ) -> bool:
        """Check a fast mover without ignoring an imminent initial collision."""
        if (
            target_id is None
            or not self._v2x_tracker.has_velocity_estimate(target_id)
            or self._mpc.current_prediction is None
            or self._mpc.infeasibility_counter != 0
            or self._mpc.used_prediction_fallback
            or self._mpc.recovery_requested
        ):
            return False
        target_buf = self._v2x_tracker._samples.get(target_id)
        if not target_buf:
            return False
        _, target_x, target_y = target_buf[-1]
        target_vx, target_vy = self._v2x_tracker.velocity(target_id)
        if math.hypot(target_vx, target_vy) < (
            self._moving_vehicle_brake_bypass_min_speed
        ):
            return False

        clearance = (
            0.5 * float(self._cfg.bicycle_model.width)
            + self._v2x_vehicle_radius
        )
        relative_x = float(target_x) - float(pose.x)
        relative_y = float(target_y) - float(pose.y)
        ego_vx = float(ego_speed) * math.cos(float(pose.theta))
        ego_vy = float(ego_speed) * math.sin(float(pose.theta))
        relative_vx = float(target_vx) - ego_vx
        relative_vy = float(target_vy) - ego_vy
        relative_speed_sq = relative_vx ** 2 + relative_vy ** 2
        early_horizon = self._moving_vehicle_brake_bypass_confirm_sec
        closest_time = 0.0
        if relative_speed_sq > 1e-9:
            closest_time = float(np.clip(
                -(relative_x * relative_vx + relative_y * relative_vy)
                / relative_speed_sq,
                0.0,
                early_horizon,
            ))
        early_clearance = math.hypot(
            relative_x + relative_vx * closest_time,
            relative_y + relative_vy * closest_time,
        )
        if early_clearance < clearance:
            return False

        pred_x, pred_y = self._mpc.current_prediction
        prediction_times = [
            self._v2x_t_samples[
                min(index + 2, len(self._v2x_t_samples) - 1)
            ]
            for index in range(len(pred_x))
        ]
        return prediction_clears_moving_vehicle(
            pred_x,
            pred_y,
            prediction_times,
            vehicle_x=target_x,
            vehicle_y=target_y,
            vehicle_vx=target_vx,
            vehicle_vy=target_vy,
            minimum_clearance=clearance,
            minimum_prediction_time=early_horizon,
            maximum_prediction_time=(
                self._moving_vehicle_brake_bypass_horizon_sec),
        )

    def _follow_escape_lane_traffic_is_clear(
        self, pose, ego_speed: float, lane_idx
    ) -> bool:
        if lane_idx not in (0, 1, 2):
            return False
        conflicts = classify_lane_conflicts(
            lane_idx,
            self._relative_lane_vehicle_samples(pose, ego_speed),
            front_distance=self._prepass_lane_fallback_front_distance,
            side_distance=self._prepass_lane_fallback_side_distance,
            rear_distance=self._prepass_lane_fallback_rear_distance,
        )
        return lane_conflicts_are_clear(conflicts)

    def _prediction_has_forward_progress(self, pose, u) -> bool:
        if self._mpc.current_prediction is None or float(u[0]) < (
            self._follow_deadlock_forward_command_threshold
        ):
            return False
        pred_x, pred_y = self._mpc.current_prediction
        if not pred_x or len(pred_x) != len(pred_y):
            return False
        progress = self._center_longitudinal_between(
            float(pred_x[0]),
            float(pred_y[0]),
            float(pred_x[-1]),
            float(pred_y[-1]),
        )
        if progress is None:
            return False
        return progress >= self._follow_deadlock_gnss_distance_threshold

    def _handoff_prepass_to_follow_probe(self, pose, ego_speed, now_sec):
        """Give a stopped, validated lane trial exclusive ownership of Prepass.

        This grants a constrained solve, not permission to move. Solver/reverse
        recovery and real traffic hazards remain exclusive higher priorities.
        """
        lane = self._follow_escape_probe_lane_idx
        target = self._follow_escape_target_id
        if (not self._follow_escape_active or lane not in (0,2)
                or not self._prepass_fallback_recovery_active
                or target != self._overtake.target_id or abs(ego_speed) > .3
                or self._mpc_safety_recovery_active or self._post_reverse_full_width_recovery_active
                or self._parallel_abort_active or self._stuck_recovery_until is not None
                or self._close_obstacle_reverse_requested
                or self._follow_only
                or not self._v2x_tracker.has_velocity_estimate(target)
                or math.hypot(*self._v2x_tracker.velocity(target))
                    > self._strict_shadow_commit_creep_max_target_speed
                or not self._lane_horizon_has_vehicle_width(lane)
                or not self._follow_escape_lane_traffic_is_clear(pose,ego_speed,lane)):
            return False
        passage,_ = self._vehicle_passage(target,pose)
        if not passage.get(lane,False):
            return False
        for vid in self._v2x_tracker.active_vehicle_ids():
            if (not self._v2x_tracker.has_velocity_estimate(vid)
                    or self._committed_target_body_overlap(pose,vid) is not False):
                return False
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_started_at = None
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_blocked = False
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_fallback_commit_success_since = None
        self._prepass_fallback_lane_idx = None
        self._clear_prepass_soft_guidance()
        self._overtake.requested_lane = lane
        self._follow_escape_probe_started_at = now_sec
        self._follow_escape_probe_success_cycles = 0
        self._mpc.osqp_initialized = False
        self.get_logger().info(
            f"[FollowProbePrepassHandoff] vehicle={target}, lane=L{lane}; "
            "holding output until fresh constrained forward proof succeeds")
        return True

    def _update_follow_deadlock_escape(
        self,
        *,
        now_sec: float,
        pose,
        ego_speed: float,
        follow_active: bool,
        target_id,
        lead_speed: float,
        forward_command: float,
        emergency_brake_active: bool,
        emergency_brake_vehicle_id=None,
    ) -> None:
        """Detect and resolve a stopped-follow deadlock without blind motion."""
        if not self._follow_deadlock_escape_enabled:
            return

        position = self._follow_deadlock_position(pose)
        if self._follow_deadlock_start_xy is None:
            self._follow_deadlock_start_xy = position
        gnss_moved_distance = math.hypot(
            position[0] - self._follow_deadlock_start_xy[0],
            position[1] - self._follow_deadlock_start_xy[1],
        )

        if self._follow_escape_active:
            target = self._latched_follow_target_state(pose, now_sec)
            if (
                target is None
                or target.get("expired", False)
            ):
                self._reset_follow_escape("target disappeared")
                return
            if not is_follow_target_ahead(target.get("longitudinal")):
                self._reset_follow_escape("target was passed")
                return
            if not should_hold_follow_escape_exclusive(
                target_longitudinal=target.get("longitudinal"),
                target_distance=target.get("distance"),
                safe_distance=self._follow_desired_distance,
            ):
                self._reset_follow_escape(
                    "safe distance established: "
                    f"distance={target.get('distance', math.inf):.2f}m/"
                    f"{self._follow_desired_distance:.2f}m"
                )
                return

            if self._follow_escape_forward_active:
                if emergency_brake_active:
                    failed_lane = self._follow_escape_probe_lane_idx
                    if failed_lane is not None:
                        self._follow_escape_attempted_lanes.add(failed_lane)
                    self._follow_escape_forward_active = False
                    self._follow_escape_forward_until = None
                    self._follow_escape_probe_lane_idx = None
                    self._follow_escape_probe_started_at = None
                    self._follow_escape_probe_success_cycles = 0
                    self.get_logger().warn(
                        "[FollowDeadlockForwardAbort] EmergencyBrake became "
                        f"active during low-speed escape: lane=L{failed_lane}"
                    )
                elif now_sec < self._follow_escape_forward_until:
                    return
                else:
                    # A one-second creep is only one step of the escape. Keep
                    # exclusive lateral ownership and revalidate the same lane
                    # before any further forward motion. Ordinary selection may
                    # resume only after the target is passed or the configured
                    # following distance has been established.
                    self._follow_escape_forward_active = False
                    self._follow_escape_forward_until = None
                    self._follow_escape_probe_started_at = now_sec
                    self._follow_escape_probe_success_cycles = 0
                    self.get_logger().info(
                        "[FollowDeadlockForwardHold] creep interval completed; "
                        "keeping exclusive lane ownership until target pass or "
                        "safe distance: "
                        f"vehicle_id={self._follow_escape_target_id}, "
                        f"lane=L{self._follow_escape_probe_lane_idx}, "
                        f"distance={target.get('distance', math.inf):.2f}m/"
                        f"{self._follow_desired_distance:.2f}m"
                    )

            if self._follow_escape_probe_lane_idx is None:
                reevaluate_due = (
                    self._follow_escape_last_reevaluate_at is None
                    or now_sec - self._follow_escape_last_reevaluate_at
                        >= self._follow_escape_reevaluate_sec
                )
                if not reevaluate_due:
                    return
                self._follow_escape_last_reevaluate_at = now_sec
                lane_idx, conflicts = self._select_follow_escape_lane(
                    pose, ego_speed)
                if lane_idx is not None:
                    self._follow_escape_probe_lane_idx = lane_idx
                    self._follow_escape_probe_started_at = now_sec
                    self._follow_escape_probe_success_cycles = 0
                    self._mpc.osqp_initialized = False
                    self._handoff_prepass_to_follow_probe(pose,ego_speed,now_sec)
                    self.get_logger().warn(
                        "[FollowDeadlockForwardProbe] probing a stopped-follow "
                        f"escape lane: vehicle_id={self._follow_escape_target_id}, "
                        f"lane=L{lane_idx}, conflicts={conflicts}"
                    )
                    return

                # Reconsider every lane when traffic changes. Between checks,
                # keep zero speed rather than selecting a stale candidate.
                self._follow_escape_attempted_lanes.clear()
                reverse_mode = self._prepare_follow_deadlock_reverse(
                    pose=pose,
                    now_sec=now_sec,
                    ego_speed=ego_speed,
                    target_id=self._follow_escape_target_id,
                )
                reverse_distance = self._stuck_reverse_target_distance
                rear_clear = (
                    reverse_mode != "blocked"
                    and self._reverse_rear_is_clear(
                        pose,
                        ego_speed,
                        reverse_distance=reverse_distance,
                    )
                )
                if rear_clear:
                    self._prepass_retry_after_reverse = True
                    self._prepass_retry_lane_idx = None
                    self._prepass_reverse_motion_started = False
                    self._prepass_reverse_start_xy = None
                    self._prepass_reverse_distance = 0.0
                    self._close_obstacle_reverse_requested = True
                    # The dedicated detector already observed two seconds of
                    # immobility. Do not wait for the generic timer again.
                    self._stuck_since = now_sec - self._stuck_time_threshold
                    self.get_logger().warn(
                        "[FollowDeadlockReverseRequest] no safe forward lane; "
                        "rear corridor is clear, requesting reverse: "
                        f"vehicle_id={self._follow_escape_target_id}, "
                        f"mode={reverse_mode}, target_distance="
                        f"{reverse_distance if reverse_distance is not None else 'timed'}"
                    )
                else:
                    self._close_obstacle_reverse_requested = False
                    self.get_logger().warn(
                        "[FollowDeadlockBlocked] forward candidates and rear "
                        "corridor are unsafe; holding zero speed until the "
                        f"next check in {self._follow_escape_reevaluate_sec:.2f}s: "
                        f"vehicle_id={self._follow_escape_target_id}, "
                        f"conflicts={conflicts}"
                    )
                return

            applied_lane_idx = (
                self._reference_path.target_lane_idx
                if self._reference_path.is_overtaking else None
            )
            lane_applied = (
                applied_lane_idx == self._follow_escape_probe_lane_idx
            )
            if not lane_applied:
                # No constrained trial occurred: do not consume this lane's
                # failure budget or reverse because another owner deferred it.
                self._follow_escape_probe_started_at = now_sec
                self._follow_escape_probe_success_cycles = 0
                self._handoff_prepass_to_follow_probe(pose,ego_speed,now_sec)
                self.get_logger().info(
                    f"[FollowProbeDeferred] lane=L{self._follow_escape_probe_lane_idx}, "
                    f"applied={applied_lane_idx}; awaiting lane ownership",
                    throttle_duration_sec=1.0)
                return
            feasible_solution = (
                self._mpc.infeasibility_counter == 0
                and self._mpc.current_prediction is not None
                and not self._mpc.used_prediction_fallback
                and not self._mpc.recovery_requested
                and not self._mpc_safety_recovery_active
            )
            executable_prediction = self._prediction_has_forward_progress(
                pose, [forward_command, 0.0])
            prediction_clear = (
                self._follow_escape_target_id in self._slow_pass_spacing_release_ids(pose)
                if self._follow_escape_probe_lane_idx in (0,2)
                else self._prediction_is_clear_of_vehicle(self._follow_escape_target_id))
            self._follow_escape_probe_success_cycles = (
                update_follow_escape_probe_success_cycles(
                    self._follow_escape_probe_success_cycles,
                    lane_applied=lane_applied,
                    feasible_solution=feasible_solution,
                    executable_forward_prediction=executable_prediction,
                    prediction_clear=prediction_clear,
                    emergency_brake_active=emergency_brake_active,
                )
            )
            if self._follow_escape_probe_success_cycles >= (
                self._follow_escape_probe_success_cycles_required
            ):
                self._follow_escape_forward_active = True
                self._follow_escape_forward_until = (
                    now_sec + self._follow_escape_forward_sec)
                self.get_logger().warn(
                    "[FollowDeadlockForwardCommit] lane MPC, forward "
                    "prediction and target clearance succeeded continuously; "
                    f"creeping forward: vehicle_id={self._follow_escape_target_id}, "
                    f"lane=L{self._follow_escape_probe_lane_idx}, "
                    f"success_cycles={self._follow_escape_probe_success_cycles}, "
                    f"speed={self._follow_escape_creep_speed:.2f}m/s"
                )
                return

            probe_elapsed = now_sec - self._follow_escape_probe_started_at
            if probe_elapsed >= self._follow_escape_probe_timeout_sec:
                failed_lane = self._follow_escape_probe_lane_idx
                if (
                    emergency_brake_active
                    and emergency_brake_vehicle_id
                        == self._follow_escape_target_id
                    and not prediction_clear
                ):
                    self._follow_escape_target_prediction_blocked = True
                self._follow_escape_attempted_lanes.add(failed_lane)
                self._follow_escape_probe_lane_idx = None
                self._follow_escape_probe_started_at = None
                self._follow_escape_probe_success_cycles = 0
                self._follow_escape_last_reevaluate_at = None
                self.get_logger().warn(
                    "[FollowDeadlockForwardProbeFailed] candidate did not "
                    "produce three consecutive safe forward predictions: "
                    f"lane=L{failed_lane}, elapsed={probe_elapsed:.2f}s, "
                    f"lane_applied={lane_applied}, feasible={feasible_solution}, "
                    f"forward={executable_prediction}, "
                    f"prediction_clear={prediction_clear}, "
                    f"emergency_brake={emergency_brake_active}"
                )
            return

        deadlock_conditions = follow_stop_deadlock_conditions_met(
            follow_active=follow_active,
            ego_speed=ego_speed,
            lead_speed=lead_speed,
            gnss_moved_distance=gnss_moved_distance,
            forward_command=forward_command,
            ego_speed_threshold=self._follow_deadlock_ego_speed_threshold,
            lead_speed_threshold=self._follow_deadlock_lead_speed_threshold,
            gnss_distance_threshold=(
                self._follow_deadlock_gnss_distance_threshold),
            forward_command_threshold=(
                self._follow_deadlock_forward_command_threshold),
        )
        if not deadlock_conditions or target_id is None:
            self._follow_deadlock_since = None
            self._follow_deadlock_start_xy = position
            return
        if self._follow_deadlock_since is None:
            self._follow_deadlock_since = now_sec
            self._follow_deadlock_start_xy = position
            return
        if now_sec - self._follow_deadlock_since < self._follow_deadlock_hold_sec:
            return

        # FollowDeadlockEscape is the exclusive lateral owner.  In particular,
        # do not leave an L1 rejoin/probe armed behind it: that stale state could
        # otherwise resume without a fresh feasibility check when Escape ends.
        self._cancel_l1_rejoin_for_overtake()
        self._follow_escape_active = True
        self._follow_escape_target_id = target_id
        self._overtake.target_id = target_id
        self._forced_overtake_vehicle_id = target_id
        self._follow_latched_cache = None
        self._follow_escape_last_reevaluate_at = None
        self.get_logger().warn(
            "[FollowDeadlockDetected] ego, lead, GNSS progress and forward "
            "command stayed below thresholds; starting safe escape evaluation: "
            f"vehicle_id={target_id}, elapsed="
            f"{now_sec - self._follow_deadlock_since:.2f}s, "
            f"ego_speed={abs(ego_speed):.2f}m/s, "
            f"lead_speed={lead_speed:.2f}m/s, "
            f"gnss_moved={gnss_moved_distance:.2f}m, "
            f"forward_command={forward_command:.2f}m/s"
        )

    def _vehicle_passage(self, target_id, pose):
        """Return horizon-passable outer lanes and distance for one vehicle.

        Passage is not decided from only the target's current waypoint.  The
        target's short V2X prediction and every ego MPC-horizon lane segment
        must retain enough physical width.  Traffic occupancy is intersected
        separately by ``classify_lane_conflicts`` at the selection site.
        """
        if target_id is None:
            return {}, None
        target_buf = self._v2x_tracker._samples.get(target_id)
        if not target_buf:
            return {}, None
        _, target_x, target_y = target_buf[-1]
        min_space = (
            0.5 * float(self._cfg.bicycle_model.width)
            + float(self._v2x_vehicle_radius)
            + self._passage_clearance
        )
        target_vx, target_vy = self._v2x_tracker.velocity(target_id)
        prediction_limit = self._prepass_lane_fallback_prediction_sec
        prediction_times = [0.0]
        prediction_times.extend(
            float(t) for t in self._v2x_t_samples
            if 0.0 < float(t) <= prediction_limit
        )
        passage = {0: True, 2: True}
        clearance_min = {0: math.inf, 2: math.inf}
        clearance_failure = {0: None, 2: None}
        for prediction_time in prediction_times:
            predicted_x = target_x + target_vx * prediction_time
            predicted_y = target_y + target_vy * prediction_time
            target_wp_id = self._carN_center.get_closest_waypoint(
                predicted_x, predicted_y)
            target_wp = self._reference_pathN_center.get_waypoint(target_wp_id)
            normal_angle = target_wp.psi + math.pi / 2.0
            target_offset = (
                (predicted_x - target_wp.x) * math.cos(normal_angle)
                + (predicted_y - target_wp.y) * math.sin(normal_angle)
            )
            clearances = {
                0: float(target_offset - target_wp.lb),
                2: float(target_wp.ub - target_offset),
            }
            for lane_idx in (0, 2):
                clearance_min[lane_idx] = min(
                    clearance_min[lane_idx], clearances[lane_idx])
                if clearances[lane_idx] < min_space:
                    passage[lane_idx] = False
                    if clearance_failure[lane_idx] is None:
                        clearance_failure[lane_idx] = (
                            prediction_time, target_wp_id,
                            clearances[lane_idx])

        # Reject a nominal lane which becomes narrower than the vehicle in
        # the ego prediction horizon. This catches a taper/track-width change
        # which the old single target-waypoint test could not see.
        horizon_path = self._reference_pathN_center
        horizon_wp = self._carN_center.wp_id
        required_width = float(self._cfg.bicycle_model.width)
        lane_widths = {0: [], 2: []}
        lane_width_wps = {0: [], 2: []}
        for offset in range(self._mpcN_center.N + 1):
            lanes = horizon_path.get_lane_bounds(horizon_wp + offset)
            for lane_idx in (0, 2):
                if lane_idx >= len(lanes):
                    lane_widths[lane_idx].append(0.0)
                    lane_width_wps[lane_idx].append(horizon_wp + offset)
                    continue
                lane_ub, lane_lb = lanes[lane_idx]
                lane_widths[lane_idx].append(
                    float(lane_ub) - float(lane_lb))
                lane_width_wps[lane_idx].append(horizon_wp + offset)

        for lane_idx in (0, 2):
            width_result = evaluate_lane_width_samples(
                lane_widths[lane_idx],
                required_width=required_width,
                tolerance=self._passage_lane_width_tolerance,
                max_consecutive_tolerated=(
                    self._passage_lane_width_tolerance_points),
            )
            if not width_result["passable"]:
                passage[lane_idx] = False
            failure = clearance_failure[lane_idx]
            failed_width_index = width_result["first_failed_index"]
            failed_width_wp = (
                lane_width_wps[lane_idx][failed_width_index]
                if failed_width_index is not None else None
            )
            reasons = []
            if failure is not None:
                reasons.append("target_boundary_clearance")
            if width_result["failure_reason"] is not None:
                reasons.append(width_result["failure_reason"])
            diagnostic_message = (
                "[PhysicalPassageDiagnostic] "
                f"vehicle_id={target_id}, lane=L{lane_idx}, "
                f"passable={passage[lane_idx]}, "
                f"target_clearance_min={clearance_min[lane_idx]:.3f}m/"
                f"{min_space:.3f}m, passage_clearance="
                f"{self._passage_clearance:.3f}m, "
                f"lane_width_min={width_result['minimum_width']:.3f}m/"
                f"{required_width:.3f}m, tolerance="
                f"{self._passage_lane_width_tolerance:.3f}m, "
                f"minor_run={width_result['longest_minor_run']}/"
                f"{self._passage_lane_width_tolerance_points}, "
                f"clearance_failure={failure}, "
                f"width_failure_wp={failed_width_wp}, "
                f"reasons={reasons if reasons else ['none']}"
            )
            # Keep distinct call sites so rclpy's caller-based throttle emits
            # diagnostics for both L0 and L2 rather than suppressing the
            # second lane in this loop.
            if lane_idx == 0:
                self.get_logger().info(
                    diagnostic_message, throttle_duration_sec=1.0)
            else:
                self.get_logger().info(
                    diagnostic_message, throttle_duration_sec=1.0)
        return passage, math.hypot(target_x - pose.x, target_y - pose.y)

    def _latched_target_passage(self, pose):
        """Return passable outer lanes and distance for the latched target only."""
        return self._vehicle_passage(self._overtake.target_id, pose)

    def _propose_traffic_lane(self, preferred, passage, conflicts, target_id, pose):
        """Read-only ranking. Never feed future blockers into live lane release."""
        active = (self._overtake.requested_lane if self._overtake.committed
                  or self._overtake.can_resume_hybrid(self._overtake.requested_lane) else None)
        if active in (0, 2) and lane_evaluation.propose_lane(
                preferred, passage, conflicts, active_lane=active) == active:
            return active
        rows = []
        unknown = []
        for vid in self._v2x_tracker.active_vehicle_ids():
            if vid == target_id:
                continue
            body = collision.target_body(self, vid)
            if body is None or not body.position_valid:
                unknown.append(vid)
                continue
            distance = self._center_longitudinal_between(pose.x, pose.y, body.x, body.y)
            if distance is None or not math.isfinite(distance):
                unknown.append(vid)
                continue
            if not 0. <= distance <= self._overtake_latch_max_distance:
                continue
            if not self._v2x_tracker.has_velocity_estimate(vid):
                unknown.append(vid)
                continue
            other_passage, _ = self._vehicle_passage(vid, pose)
            rows.append(lane_evaluation.LaneObstruction(
                vid, distance, tuple(lane for lane in (0, 2)
                                     if not other_passage.get(lane, False))))
        proposed = lane_evaluation.propose_lane(preferred, passage, conflicts, rows)
        self.get_logger().info(
            f"[TrafficLaneProposal] target={target_id}, proposed={proposed}, "
            f"future_obstructions={rows}, unknown={unknown}; ranking_only=True",
            throttle_duration_sec=1.0)
        return proposed

    def _same_lane_target_handoff_available(self, successor, pose, ego_speed):
        """Geometry admission for preparing a successor Shadow without tearing down."""
        lane = self._overtake.requested_lane
        if (successor is None or not self._overtake.can_resume_hybrid(lane)
                or self._overtake.hybrid.paused
                or any(getattr(self, flag, False) for flag in (
                    '_parallel_abort_active', '_follow_escape_active',
                    '_mpc_safety_recovery_active', '_prepass_fallback_recovery_active',
                    '_post_reverse_full_width_recovery_active'))
                or self._stuck_recovery_until is not None):
            return False
        passage, _ = self._vehicle_passage(successor, pose)
        if (not passage.get(lane, False)
                or not self._lane_horizon_has_vehicle_width(lane)
                or not self._follow_escape_lane_traffic_is_clear(pose, ego_speed, lane)):
            return False
        for vid in self._v2x_tracker.active_vehicle_ids():
            if (not self._v2x_tracker.has_velocity_estimate(vid)
                    or self._committed_target_body_overlap(pose, vid) is not False):
                return False
        return True

    def _accept_same_lane_target_handoff(self, successor, lane):
        """Called only with fresh successor Shadow; leave boundaries and timer intact."""
        old = self._overtake.target_id
        if not self._overtake.handoff_target(successor, lane):
            return False
        self._overtake.verification.vehicle_id = successor
        self._overtake.verification.lane_idx = lane
        self._overtake.committed = True
        if self._lane_decision is not None:
            self._lane_decision = dataclasses.replace(self._lane_decision, target_id=successor)
        self._hybrid_reference_key = (successor, lane)
        self._follow_latched_cache = None
        self._prepass_dynamic_conflict_speed_limit = None
        self._prepass_target_behind_since = None
        self._overtake_completed_target_id = None
        self._forced_overtake_vehicle_id = None
        self._reset_outer_lane_progress()
        self._clear_consecutive_overtake_handoff()
        self._overtake_switch_candidate_id = None
        self._overtake_switch_candidate_since = None
        self._clear_urgent_overtake_switch_candidate()
        self.get_logger().info(
            f"[SameLaneTargetHandoff] {old}->{successor}, lane=L{lane}; "
            "fresh Shadow accepted; spatial anchor and boundary deadline retained")
        return True

    def _reset_outer_lane_progress(self):
        self._outer_lane_progress_vehicle_id = None
        self._outer_lane_progress_best_longitudinal = None
        self._outer_lane_last_progress_at = None

    def _l1_rejoin_traffic_is_clear(self, pose, ego_speed: float):
        """Require the merge destination to be clear before leaving L0/L2."""
        conflicts = classify_lane_conflicts(
            1,
            self._relative_lane_vehicle_samples(pose, ego_speed),
            front_distance=self._prepass_lane_fallback_front_distance,
            side_distance=self._prepass_lane_fallback_side_distance,
            rear_distance=self._prepass_lane_fallback_rear_distance,
        )
        return not any(
            conflicts.get(key) for key in ("front", "side", "rear")
        ), conflicts

    def _update_outer_lane_progress_state(
        self, *, target_id, longitudinal, pose, ego_speed: float, now_sec: float
    ) -> bool:
        """Release a stalled outer-lane pass only when L1 is clear."""
        outer_lane_active = bool(
            target_id is not None
            and longitudinal is not None
            and self._overtake.committed
            and self._reference_path.target_lane_idx in (0, 2)
            and self._reference_path.is_overtaking
            and float(now_sec) >= getattr(
                self, "_constraint_transition_until", 0.0)
        )
        if not outer_lane_active:
            self._reset_outer_lane_progress()
            return False

        lane_idx = int(self._reference_path.target_lane_idx)
        lateral_error = self._lane_lateral_error(pose.x, pose.y, lane_idx)
        lateral_move_established = bool(
            lateral_error
            <= self._slow_lead_speed_match_release_lateral_error
        )
        if not lateral_move_established:
            # Arm only after ego reaches the selected lane center; otherwise
            # lane-change time would be mistaken for missing pass progress.
            self._reset_outer_lane_progress()
            self.get_logger().info(
                "[OuterLaneProgressArmHold] waiting until lateral movement "
                "is established before evaluating passing progress: "
                f"vehicle_id={target_id}, lane=L{lane_idx}, "
                f"lateral_error={lateral_error:.2f}/"
                f"{self._slow_lead_speed_match_release_lateral_error:.2f}m",
                throttle_duration_sec=0.5,
            )
            return False

        longitudinal = float(longitudinal)
        if self._outer_lane_progress_vehicle_id != target_id:
            self._outer_lane_progress_vehicle_id = target_id
            self._outer_lane_progress_best_longitudinal = longitudinal
            self._outer_lane_last_progress_at = float(now_sec)
            return False

        if (
            self._outer_lane_progress_best_longitudinal is None
            or longitudinal
                <= self._outer_lane_progress_best_longitudinal
                - self._outer_lane_min_progress
        ):
            self._outer_lane_progress_best_longitudinal = longitudinal
            self._outer_lane_last_progress_at = float(now_sec)
            return False

        last_progress_at = self._outer_lane_last_progress_at
        stalled_sec = (
            float(now_sec) - float(last_progress_at)
            if last_progress_at is not None else 0.0
        )
        if stalled_sec < self._outer_lane_progress_timeout:
            return False

        l1_clear, conflicts = self._l1_rejoin_traffic_is_clear(
            pose, ego_speed)
        if not l1_clear:
            self.get_logger().info(
                "[OuterLanePassContinue] no passing progress but L1 is "
                "occupied; continuing the pass instead of following: "
                f"vehicle_id={target_id}, stalled={stalled_sec:.2f}s, "
                f"longitudinal={longitudinal:.2f}m, conflicts={conflicts}",
                throttle_duration_sec=0.5,
            )
            return False

        self.get_logger().warn(
            "[OuterLaneRejoin] no passing progress and L1 is clear; "
            "releasing the outer constraint for full-width soft L1 rejoin: "
            f"vehicle_id={target_id}, stalled={stalled_sec:.2f}s, "
            f"longitudinal={longitudinal:.2f}m, "
            f"lateral_error={lateral_error:.2f}m"
        )
        self._outer_lane_released_vehicle_id = target_id
        self._overtake.committed = False
        # Preserve the target/lane as manoeuvre metadata. The released-target
        # flag makes constraint resolution select full width and prevents a
        # fresh L0/L2 request for the same pass.
        self._forced_overtake_vehicle_id = None
        self._prepass_fallback_follow_active = False
        self._prepass_fallback_lane_idx = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._clear_prepass_soft_guidance()
        self._reset_overtake_commit_probe()
        self._clear_committed_shadow_verification()
        self._mpc.osqp_initialized = False
        self._reset_outer_lane_progress()
        return True

    def _reset_overtake_commit_probe(self):
        """Discard a pending L0/L2 pre-commit feasibility check."""
        self._overtake.probe.vehicle_id = None
        self._overtake.probe.lane_idx = None
        self._overtake.probe.success_cycles = 0
        self._overtake.probe.confirmed = False
        self._overtake.probe.confirmed_at = None

    def _clear_committed_shadow_verification(self):
        """Invalidate the Shadow proof retained by an active commitment."""
        self._overtake.verification.vehicle_id = None
        self._overtake.verification.lane_idx = None

    def _prepare_overtake_commit_probe(self, vehicle_id, lane_idx):
        """Latch one candidate without granting it control of the live MPC."""
        if vehicle_id is None or lane_idx not in (0, 2):
            self._reset_overtake_commit_probe()
            return
        if (
            self._overtake.probe.vehicle_id != vehicle_id
            or self._overtake.probe.lane_idx != int(lane_idx)
        ):
            self._overtake.probe.vehicle_id = vehicle_id
            self._overtake.probe.lane_idx = int(lane_idx)
            self._overtake.probe.success_cycles = 0
            self._overtake.probe.confirmed = False
            self._overtake.probe.confirmed_at = None
            probe = self._mpcN_overtake_commit_probe
            probe.osqp_initialized = False
            probe.current_prediction = None
            probe.current_control = np.zeros_like(probe.current_control)
            probe.infeasibility_counter = 0

    def _candidate_lane_has_valid_width_ahead(
        self, start_wp: int, lane_idx: int
    ) -> tuple[bool, Optional[dict]]:
        """Check static candidate-lane width over a true forward distance."""
        reference_path = self._reference_pathN_center
        required_width = float(self._carN_center.width)
        travelled = 0.0
        wp_id = int(start_wp)
        for _ in range(int(reference_path.n_waypoints)):
            lanes = reference_path.get_lane_bounds(wp_id)
            if not lanes or lane_idx >= len(lanes):
                return False, {"wp": wp_id, "width": 0.0}
            upper, lower = lanes[lane_idx]
            width = float(upper) - float(lower)
            if not math.isfinite(width) or width + 1e-6 < required_width:
                return False, {"wp": wp_id, "width": width}
            if travelled >= self._overtake_commit_valid_width_distance:
                return True, None
            waypoint = reference_path.get_waypoint(wp_id)
            next_waypoint = reference_path.get_waypoint(wp_id + 1)
            travelled += math.hypot(
                float(next_waypoint.x) - float(waypoint.x),
                float(next_waypoint.y) - float(waypoint.y),
            )
            wp_id = (wp_id + 1) % int(reference_path.n_waypoints)
        return False, {"wp": wp_id, "width": 0.0}

    @contextmanager
    def _probe_corridor(self, model, lane_idx):
        """Probe a private boundary snapshot without changing live paths."""
        original = model.reference_path
        snapshot = copy.copy(original)
        snapshot.waypoints = [copy.copy(wp) for wp in original.waypoints]
        snapshot.border_cells = copy.deepcopy(original.border_cells)
        snapshot.unsafe_static_fallback_wp_ids = []
        snapshot.target_lane_idx = lane_idx
        snapshot.is_overtaking = lane_idx is not None
        model.reference_path = snapshot
        try:
            if hasattr(model, "current_waypoint"):
                model.current_waypoint = snapshot.get_waypoint(model.wp_id)
            yield snapshot
        finally:
            model.reference_path = original
            if hasattr(model, "current_waypoint"):
                model.current_waypoint = original.get_waypoint(model.wp_id)


    def _run_overtake_commit_probe(self, predicted_pose, recovery_active):
        """Solve a candidate L0/L2 without changing the live controller."""
        lane_idx = self._overtake.probe.lane_idx
        vehicle_id = self._overtake.probe.vehicle_id
        if (
            lane_idx not in (0, 2)
            or vehicle_id is None
        ):
            return
        if (
            recovery_active
            or self._mpc_safety_recovery_active
            or self._post_reverse_full_width_recovery_active
            or self._reference_path is not self._reference_pathN_center
        ):
            self._overtake.probe.success_cycles = 0
            self._overtake.probe.confirmed = False
            self._overtake.probe.confirmed_at = None
            return

        probe = self._mpcN_overtake_commit_probe
        collapse_detail = None
        width_failure = None
        with self._probe_corridor(self._carN_overtake_commit_probe, int(lane_idx)):
            self._carN_overtake_commit_probe.update_states(
                predicted_pose.x, predicted_pose.y, predicted_pose.theta)
            probe.set_soft_lateral_reference()
            probe.set_lane_transition_weights()
            self._update_l2_target_objective_offsets(
                probe, self._carN_overtake_commit_probe.wp_id)
            probe.set_full_width_l1_offset_limits()
            probe.set_full_width_l0_offset_limits()
            probe.update_wp_id_offset(0)
            probe.previous_steering = self._mpcN_center.previous_steering
            probe.get_control()
            collapse_detail = getattr(
                probe, "_constraint_collapse_detail", None)
            width_valid, width_failure = (
                self._candidate_lane_has_valid_width_ahead(
                    self._carN_overtake_commit_probe.wp_id, lane_idx)
            )
            relaxation = float(getattr(
                probe, "_constraint_lane_relaxation", math.inf))
            success = overtake_shadow_solution_acceptable(
                accurate=getattr(probe, "last_solution_accurate", False),
                used_prediction_fallback=probe.used_prediction_fallback,
                time_budget_exceeded=probe.time_budget_exceeded,
                recovery_requested=probe.recovery_requested,
                infeasibility_counter=probe.infeasibility_counter,
                has_prediction=probe.current_prediction is not None,
                constraint_collapsed=getattr(
                    probe, "_constraint_collapse_detected", False),
                lane_relaxation=relaxation,
                max_lane_relaxation=self._overtake_commit_max_relaxation,
                forward_width_valid=width_valid,
            )

        was_confirmed = self._overtake.probe.confirmed
        self._overtake.probe.success_cycles = (
            min(
                self._overtake.probe.success_cycles + 1,
                self._overtake_commit_probe_required_success_cycles,
            )
            if success else 0)
        if not success and not was_confirmed:
            # Before confirmation, consecutive success is mandatory.  After
            # confirmation, however, one transient solver miss must not erase
            # the exact-lane proof immediately and let the high-level selector
            # jump to the opposite side.  The original confirmation timestamp
            # is intentionally not extended here: _is_fresh() still prevents
            # commitment once the strict proof ages past its short freshness
            # window, while the candidate hold lets this same lane re-probe.
            self._overtake.probe.confirmed = False
            self._overtake.probe.confirmed_at = None
        if collapse_detail is not None:
            self.get_logger().warn(
                "[OvertakeCommitProbeConstraintCollapse] rejecting shadow "
                "MPC because the candidate corridor collapsed: "
                f"vehicle_id={vehicle_id}, lane=L{lane_idx}, "
                f"i={collapse_detail['index']}, wp={collapse_detail['wp']}, "
                f"width={collapse_detail['width']:.3f}m",
                throttle_duration_sec=0.5)
        if width_failure is not None:
            self.get_logger().info(
                "[OvertakeCommitProbeWidthHold] candidate lane has no valid "
                f"vehicle-width corridor for "
                f"{self._overtake_commit_valid_width_distance:.1f}m: "
                f"vehicle_id={vehicle_id}, lane=L{lane_idx}, "
                f"wp={width_failure['wp']}, width={width_failure['width']:.3f}m",
                throttle_duration_sec=0.5)
        self.get_logger().info(
            "[OvertakeCommitProbe] shadow verification: "
            f"vehicle_id={vehicle_id}, lane=L{lane_idx}, success={success}, "
            f"accurate={getattr(probe, 'last_solution_accurate', False)}, "
            f"fallback={probe.used_prediction_fallback}, "
            f"budget_exceeded={probe.time_budget_exceeded}, "
            f"relaxation={float(getattr(probe, '_constraint_lane_relaxation', math.inf)):.2f}/"
            f"{self._overtake_commit_max_relaxation:.2f}m, cycles="
            f"{self._overtake.probe.success_cycles}/"
            f"{self._overtake_commit_probe_required_success_cycles}",
            throttle_duration_sec=0.25)
        if self._overtake.probe.success_cycles >= (
            self._overtake_commit_probe_required_success_cycles
        ):
            self._overtake.probe.confirmed = True
            self._overtake.probe.confirmed_at = float(
                self.get_clock().now().nanoseconds) / 1e9
            if not was_confirmed:
                self.get_logger().info(
                    "[OvertakeCommitProbeConfirmed] candidate outer lane "
                    "passed all pre-commit gates: "
                    f"vehicle_id={vehicle_id}, lane=L{lane_idx}")

    def _overtake_commit_probe_is_fresh(
        self, vehicle_id, lane_idx, now_sec: float
    ) -> bool:
        """Return whether the exact candidate has a recent strict Shadow proof."""
        return bool(
            self._overtake.probe.confirmed
            and self._overtake.probe.vehicle_id == vehicle_id
            and self._overtake.probe.lane_idx == lane_idx
            and self._overtake.probe.confirmed_at is not None
            and now_sec - self._overtake.probe.confirmed_at
                <= self._overtake_commit_probe_freshness_sec
        )

    def _overtake_commit_curvature_preview(self, center_wp: int):
        """Return signed Center-path curvature samples over the preview distance."""
        ref_path = self._reference_pathN_center
        samples = []
        travelled = 0.0
        wp_id = int(center_wp)
        for _ in range(int(ref_path.n_waypoints)):
            waypoint = ref_path.get_waypoint(wp_id)
            samples.append(float(waypoint.kappa))
            if travelled >= self._overtake_commit_preview_distance:
                break
            next_waypoint = ref_path.get_waypoint(wp_id + 1)
            travelled += math.hypot(
                float(next_waypoint.x) - float(waypoint.x),
                float(next_waypoint.y) - float(waypoint.y),
            )
            wp_id += 1
        return samples

    def _arm_emergency_blocker_recovery(
        self, vehicle_id, pose, ego_speed: float
    ) -> None:
        """Route an unlatched, close stopped blocker into follow/reverse recovery."""
        if vehicle_id is None:
            return
        now_sec = float(self.get_clock().now().nanoseconds) / 1e9
        if follow_emergency_reacquire_blocked(
            vehicle_id=vehicle_id,
            released_vehicle_id=self._follow_last_released_vehicle_id,
            released_at=self._follow_last_released_at,
            now_sec=now_sec,
            hysteresis_sec=self._follow_emergency_reacquire_sec,
        ):
            self.get_logger().info(
                "[EmergencyBlockerReacquireHold] suppressing immediate "
                "re-latch of the just-released Follow target: "
                f"vehicle_id={vehicle_id}, elapsed="
                f"{now_sec - self._follow_last_released_at:.2f}s/"
                f"{self._follow_emergency_reacquire_sec:.2f}s",
                throttle_duration_sec=0.5)
            return
        if self._stuck_recovery_until is not None:
            # Do not reset reverse-distance bookkeeping after the shift/reverse
            # sequence has already started.
            return
        if (
            self._prepass_retry_after_reverse
            and self._overtake.target_id == vehicle_id
        ):
            self._close_obstacle_reverse_requested = True
            return

        passage, distance = self._vehicle_passage(vehicle_id, pose)
        if distance is None:
            return

        # The vehicle which actually forced the emergency stop must become the
        # recovery/follow target. Otherwise EmergencyBrake holds u[0] at zero,
        # while StuckRecovery keeps waiting for the old overtake target.
        target_changed = self._overtake.target_id != vehicle_id
        if target_changed:
            old_target = self._overtake.target_id
            self._reset_overtake_state_for_target_change(
                vehicle_id,
                reason="EmergencyBrake selected a different stopped blocker",
            )
            self.get_logger().warn(
                "[EmergencyBlockerLatch] emergency-stop target replaces "
                f"old target: old={old_target}, new={vehicle_id}, "
                f"distance={distance:.2f}m"
            )

        # ``_reset_overtake_state_for_target_change`` deliberately clears the
        # old latch.  Promote the emergency blocker immediately, before the
        # no-passing-lane branch enters Follow.  Otherwise
        # ``_switch_prepass_to_follow`` clears only the forced-overtake marker
        # and Follow becomes active with no vehicle ID to track.
        self._overtake.target_id = vehicle_id

        samples = self._relative_lane_vehicle_samples(pose, ego_speed)
        conflicts = {
            lane_idx: classify_lane_conflicts(
                lane_idx,
                samples,
                front_distance=self._prepass_lane_fallback_front_distance,
                side_distance=self._prepass_lane_fallback_side_distance,
                rear_distance=self._prepass_lane_fallback_rear_distance,
            )
            for lane_idx in (0, 2)
        }
        candidates = [
            lane_idx for lane_idx in (0, 2)
            if passage.get(lane_idx, False)
            and lane_conflicts_are_clear(conflicts[lane_idx])
        ]
        if not candidates:
            self._switch_prepass_to_follow(
                "emergency blocker leaves no physically safe passing lane: "
                f"vehicle_id={vehicle_id}, passage={passage}, "
                f"conflicts={conflicts}"
            )
            return

        preferred = self._overtake.requested_lane
        retry_lane_idx = next(
            (lane for lane in candidates if lane == preferred), candidates[0])
        self._prepass_fallback_follow_active = False
        self._prepass_fallback_blocked = False
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_recovery_started_at = None
        self._prepass_fallback_lane_idx = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_retry_after_reverse = True
        self._prepass_retry_lane_idx = retry_lane_idx
        self._prepass_reverse_motion_started = False
        self._prepass_reverse_start_xy = None
        self._prepass_reverse_distance = 0.0
        self._close_obstacle_reverse_requested = True
        if target_changed:
            self.get_logger().warn(
                "[EmergencyBlockerRecovery] close stopped blocker has a safe "
                f"passing candidate L{retry_lane_idx}; requesting reverse "
                f"before retry: vehicle_id={vehicle_id}, distance={distance:.2f}m, "
                f"passage={passage}, conflicts={conflicts}"
            )

    def _select_prepass_retry_lane(
        self, pose, ego_speed: float, preferred=None, exclude_lane_idx=None
    ):
        """Re-evaluate a retry lane from the latest samples and latched target."""
        passage, target_distance = self._latched_target_passage(pose)
        samples = self._relative_lane_vehicle_samples(pose, ego_speed)
        conflicts = {
            lane_idx: classify_lane_conflicts(
                lane_idx,
                samples,
                front_distance=self._prepass_lane_fallback_front_distance,
                side_distance=self._prepass_lane_fallback_side_distance,
                rear_distance=self._prepass_lane_fallback_rear_distance,
            )
            for lane_idx in (0, 2)
        }
        preferred = preferred if preferred in (0, 2) else self._overtake.requested_lane
        order = ordered_outer_lane_candidates(preferred, exclude_lane_idx)
        lane_idx = next((
            candidate for candidate in order
            if candidate in (0, 2)
            and passage.get(candidate, False)
            and lane_conflicts_are_clear(conflicts[candidate])
        ), None)
        lane_idx = self._apply_l2_restricted_zone_policy(
            lane_idx,
            target_vehicle_id=self._overtake.target_id,
            physical_passage=passage,
            conflicts_by_lane=conflicts,
        )
        return lane_idx, target_distance, conflicts

    def _select_prepass_fallback_lane(
        self, pose, ego_speed: float, failed_lane_idx
    ):
        """Select opposite outer, failed-lane re-probe, then L1."""
        physical_passage, target_distance = self._latched_target_passage(pose)
        samples = self._relative_lane_vehicle_samples(pose, ego_speed)
        target_id = self._overtake.target_id
        target_is_ultra_slow = False
        if (
            target_id is not None
            and self._v2x_tracker.has_velocity_estimate(target_id)
        ):
            target_vx, target_vy = self._v2x_tracker.velocity(target_id)
            target_is_ultra_slow = bool(
                math.hypot(target_vx, target_vy)
                <= self._strict_shadow_commit_creep_max_target_speed
            )
        candidate_order = ordered_prepass_fallback_candidates(
            failed_lane_idx,
            self._prepass_attempted_outer_lanes,
            # A previous transient Shadow failure must not make L1 the final
            # state behind a stopped/ultra-slow car. Reconsider both outer
            # lanes from current geometry and traffic; the strict Shadow MPC
            # below still has to pass before either lane is committed.
            reconsider_attempted=target_is_ultra_slow,
        )
        conflicts = {
            lane_idx: classify_lane_conflicts(
                lane_idx,
                samples,
                front_distance=self._prepass_lane_fallback_front_distance,
                side_distance=self._prepass_lane_fallback_side_distance,
                rear_distance=self._prepass_lane_fallback_rear_distance,
            )
            # Keep diagnostics for every lane even when a previously attempted
            # lane is excluded from this selection pass.
            for lane_idx in (0, 1, 2)
        }
        selected_lane_idx = next((
            lane_idx for lane_idx in candidate_order
            if lane_conflicts_are_clear(conflicts[lane_idx])
            and (
                lane_idx == 1
                or physical_passage.get(lane_idx, False)
            )
        ), None)
        selected_lane_idx = self._apply_l2_restricted_zone_policy(
            selected_lane_idx,
            target_vehicle_id=self._overtake.target_id,
            physical_passage=physical_passage,
            conflicts_by_lane=conflicts,
        )
        if target_is_ultra_slow and selected_lane_idx in (0, 2):
            self.get_logger().info(
                "[PrepassUltraSlowOuterRetry] current outer corridor is "
                "clear; reconsidering it ahead of L1 despite an earlier "
                f"attempt: vehicle_id={target_id}, lane=L{selected_lane_idx}, "
                f"attempted={sorted(self._prepass_attempted_outer_lanes)}",
                throttle_duration_sec=1.0,
            )
        return selected_lane_idx, target_distance, conflicts, physical_passage

    def _candidate_lane_heading(self, lane_idx, wp_id: int) -> float:
        """Estimate heading along a candidate lane's actual center geometry."""
        def lane_center_point(candidate_wp_id):
            candidate_wp = self._reference_path.get_waypoint(candidate_wp_id)
            lanes = self._reference_path.get_lane_bounds(candidate_wp_id)
            if lane_idx not in (0, 1, 2) or lane_idx >= len(lanes):
                return candidate_wp.x, candidate_wp.y
            upper, lower = lanes[lane_idx]
            lateral_offset = 0.5 * (upper + lower)
            normal_angle = candidate_wp.psi + math.pi / 2.0
            return (
                candidate_wp.x + lateral_offset * math.cos(normal_angle),
                candidate_wp.y + lateral_offset * math.sin(normal_angle),
            )

        x0, y0 = lane_center_point(wp_id)
        x1, y1 = lane_center_point(
            wp_id + self._prepass_heading_lookahead_wps)
        if math.hypot(x1 - x0, y1 - y0) < 1e-6:
            return self._reference_path.get_waypoint(wp_id).psi
        return math.atan2(y1 - y0, x1 - x0)

    def _build_race_targets_in_center_frame(self):
        """Project the Race line onto Center arc length and lateral offset."""
        samples = []
        total = float(self._center_arc_total_length)
        if total <= 0.0:
            return None
        for wp in self._reference_pathN_race.waypoints:
            frenet = project_to_closed_path_frenet(
                wp.x, wp.y, self._center_arc_points,
                self._center_arc_cumulative, total)
            if frenet is not None:
                samples.append((float(frenet[0]), float(frenet[1])))
        if len(samples) < 2:
            self.get_logger().error(
                "[RaceCenterMapping] insufficient projected Race samples")
            return None
        samples.sort(key=lambda value: value[0])
        unique_s, unique_y = [], []
        for center_s, lateral in samples:
            if unique_s and center_s - unique_s[-1] <= 1e-3:
                unique_y[-1] = 0.5 * (unique_y[-1] + lateral)
            else:
                unique_s.append(center_s)
                unique_y.append(lateral)
        if len(unique_s) < 2:
            return None
        sample_s = np.asarray(unique_s, dtype=float)
        sample_y = np.asarray(unique_y, dtype=float)
        periodic_s = np.concatenate((sample_s - total, sample_s, sample_s + total))
        periodic_y = np.tile(sample_y, 3)
        center_s = np.asarray(self._center_arc_cumulative[:-1], dtype=float)
        targets = np.interp(center_s, periodic_s, periodic_y)
        for index, target in enumerate(targets):
            wp = self._reference_pathN_center.get_waypoint(index)
            if wp.lb is None or wp.ub is None:
                self.get_logger().error(
                    "[RaceCenterMapping] Center bounds unavailable")
                return None
            targets[index] = np.clip(target, float(wp.lb), float(wp.ub))
        max_step = float(np.max(np.abs(np.diff(np.r_[targets, targets[0]]))))
        if not np.all(np.isfinite(targets)) or max_step > self._race_handoff_max_target_step:
            self.get_logger().error(
                "[RaceCenterMapping] rejected discontinuous mapping: "
                f"max_step={max_step:.3f}m/wp/")
            return None
        self.get_logger().info(
            "[RaceCenterMapping] mapping ready: "
            f"targets={len(targets)}, max_step={max_step:.3f}m/wp")
        return targets

    def _reset_race_rejoin_handoff(self) -> None:
        self._race_rejoin_handoff_active = False
        self._race_rejoin_handoff_soft = False
        self._race_rejoin_handoff_started_at = None
        self._race_rejoin_handoff_start_e_y = None
        self._race_rejoin_handoff_effective_ramp_sec = None
        self._race_rejoin_handoff_guidance_ready = False
        self._race_rejoin_probe_success_cycles = 0
        self._race_rejoin_probe_confirmed = False
        self._race_rejoin_probe_started_at = None
        self._race_rejoin_latched_targets = None
        self._race_rejoin_latched_start_wp = None
        self._mpcN_center.set_soft_lateral_reference()

    def _race_handoff_lateral_targets(self, center_wp=None, extra=0):
        source = self._race_rejoin_latched_targets
        if source is None:
            source = self._race_targets_in_center_frame
        if source is None or len(source) == 0:
            return None
        if center_wp is None:
            center_wp = self._carN_center.wp_id
        count = len(source)
        return np.asarray([
            source[(int(center_wp) + n) % count]
            for n in range(self._mpcN_center.N + 1 + max(int(extra), 0))
        ], dtype=float)

    def _validate_race_handoff_targets(self, targets, current_e_y):
        if targets is None or len(targets) == 0 or not np.all(np.isfinite(targets)):
            return False, "missing_or_non_finite"
        gap = abs(float(current_e_y) - float(targets[0]))
        if gap > self._race_handoff_max_position_gap:
            return False, f"position_gap={gap:.2f}m"
        max_step = float(np.max(np.abs(np.diff(targets)))) if len(targets) > 1 else 0.0
        if max_step > self._race_handoff_max_target_step:
            return False, f"target_step={max_step:.2f}m/wp"
        return True, "ok"

    def _start_race_rejoin_handoff(self, position_gap, now_sec, center_wp):
        if self._race_rejoin_handoff_active:
            return
        if self._race_rejoin_retry_not_before is not None and now_sec < self._race_rejoin_retry_not_before:
            return
        self._cancel_normal_l1_rejoin_for_prepass()
        self._target_lane_idx = None
        self._race_rejoin_handoff_active = True
        self._race_rejoin_handoff_soft = (
            position_gap > self._race_rejoin_direct_max_position_gap)
        self._race_rejoin_handoff_guidance_ready = not self._race_rejoin_handoff_soft
        self._race_rejoin_handoff_started_at = None
        self._race_rejoin_probe_started_at = None
        self._race_rejoin_probe_success_cycles = 0
        self._race_rejoin_probe_confirmed = False
        self._race_rejoin_latched_targets = np.array(
            self._race_targets_in_center_frame, copy=True)
        self._race_rejoin_latched_start_wp = int(center_wp)
        self._mpcN_race.osqp_initialized = False
        self._mpcN_race.current_prediction = None
        self._mpcN_race.previous_steering = self._mpcN_center.previous_steering
        self._mpcN_race.current_control = np.array(
            self._mpcN_center.current_control, copy=True)
        self.get_logger().info(
            "[RaceRejoin] handoff started: "
            f"mode={'soft' if self._race_rejoin_handoff_soft else 'direct'}, "
            f"gap={position_gap:.2f}m")

    def _update_race_handoff_reference(self, enabled, now_sec):
        if not enabled:
            return
        targets = self._race_handoff_lateral_targets()
        guard = self._race_handoff_lateral_targets(
            extra=self._race_handoff_guard_extra_lookahead_wps)
        valid, reason = self._validate_race_handoff_targets(
            guard, self._carN_center.spatial_state.e_y)
        if not valid:
            self.get_logger().warn(
                f"[RaceHandoffGuard] cancelled: {reason}")
            self._reset_race_rejoin_handoff()
            self._race_rejoin_retry_not_before = now_sec + self._race_rejoin_retry_backoff_sec
            return
        if self._race_rejoin_handoff_started_at is None:
            self._race_rejoin_handoff_started_at = now_sec
            self._race_rejoin_handoff_start_e_y = float(
                self._carN_center.spatial_state.e_y)
            furthest = max(targets, key=lambda value: abs(
                value - self._race_rejoin_handoff_start_e_y))
            self._race_rejoin_handoff_effective_ramp_sec = lateral_reference_ramp_duration(
                self._race_rejoin_handoff_start_e_y, furthest,
                self._race_handoff_ramp_sec,
                self._race_handoff_max_reference_speed)
        duration = self._race_rejoin_handoff_effective_ramp_sec
        alpha = 1.0 if duration <= 0.0 else min(
            (now_sec - self._race_rejoin_handoff_started_at) / duration, 1.0)
        self._mpcN_center.set_soft_lateral_reference(
            start_e_y=self._race_rejoin_handoff_start_e_y,
            alpha=alpha, lateral_targets=targets)
        if alpha >= 1.0:
            self._race_rejoin_handoff_guidance_ready = True

    def _run_race_rejoin_probe(self, predicted_pose, recovery_active, now_sec,
                               heading_ok, heading_released):
        if not self._race_rejoin_handoff_active or not self._race_rejoin_handoff_guidance_ready:
            return
        if recovery_active or self._mpc_safety_recovery_active or heading_released:
            self._race_rejoin_probe_success_cycles = 0
            self._race_rejoin_probe_confirmed = False
            self._race_rejoin_probe_started_at = None
            return
        if not heading_ok:
            return
        if self._race_rejoin_probe_started_at is None:
            self._race_rejoin_probe_started_at = now_sec
        elif (self._race_rejoin_probe_timeout_sec > 0.0
              and now_sec - self._race_rejoin_probe_started_at >= self._race_rejoin_probe_timeout_sec
              and not self._race_rejoin_probe_confirmed):
            self.get_logger().warn("[RaceRejoinProbe] timeout; falling back to L1 rejoin")
            self._reset_race_rejoin_handoff()
            self._race_rejoin_retry_not_before = now_sec + self._race_rejoin_retry_backoff_sec
            return
        with self._probe_corridor(self._carN_race, None):
            self._carN_race.update_states(
                predicted_pose.x, predicted_pose.y, predicted_pose.theta)
            self._mpcN_race.set_soft_lateral_reference()
            self._mpcN_race.set_lane_transition_weights()
            self._mpcN_race.update_wp_id_offset(0)
            self._mpcN_race.previous_steering = self._mpcN_center.previous_steering
            self._mpcN_race.get_control()
        success = (
            not self._mpcN_race.recovery_requested
            and self._mpcN_race.infeasibility_counter == 0
            and self._mpcN_race.current_prediction is not None
            and not self._mpcN_race.used_prediction_fallback
            and not self._mpcN_race.time_budget_exceeded
            and bool(getattr(
                self._mpcN_race, "last_solution_accurate", False)))
        self._race_rejoin_probe_success_cycles = (
            self._race_rejoin_probe_success_cycles + 1 if success else 0)
        if self._race_rejoin_probe_success_cycles >= self._race_rejoin_probe_required_success_cycles:
            self._race_rejoin_probe_confirmed = True

    def _lane_lateral_error(self, x: float, y: float, lane_idx: int):
        """Return distance from a point to the selected lane center."""
        # L1 rejoin is defined on the Center path even when the currently
        # active path is still Race during a trajectory transition.
        car = self._carN_center
        reference_path = self._reference_pathN_center
        wp_id = car.get_closest_waypoint(x, y)
        wp = reference_path.get_waypoint(wp_id)
        lanes = reference_path.get_lane_bounds(wp_id)
        if lane_idx < 0 or lane_idx >= len(lanes):
            return math.inf
        upper, lower = lanes[lane_idx]
        normal_angle = wp.psi + math.pi / 2.0
        lateral_offset = (
            (x - wp.x) * math.cos(normal_angle)
            + (y - wp.y) * math.sin(normal_angle)
        )
        return abs(lateral_offset - 0.5 * (upper + lower))

    def _update_l1_soft_rejoin_reference(
        self, *, enabled: bool, now_sec: float
    ) -> None:
        """Ramp xr toward L1 while leaving MPC constraints at full width."""
        for mpc in (self._mpcN_race, self._mpcN_center):
            mpc.set_soft_lateral_reference()

        if not enabled:
            self._l1_soft_rejoin_started_at = None
            self._l1_soft_rejoin_start_e_y = None
            self._l1_soft_rejoin_effective_ramp_sec = None
            self._l1_soft_rejoin_full_strength_logged = False
            return

        if self._l1_soft_rejoin_started_at is None:
            self._l1_soft_rejoin_started_at = float(now_sec)
            self._l1_soft_rejoin_start_e_y = float(
                self._carN_center.spatial_state.e_y)
            center_wp = self._carN_center.wp_id
            l1_center_references = [
                self._mpcN_center._compute_lane_center(center_wp + n, 1)
                for n in range(self._mpcN_center.N + 1)
            ]
            l1_center_e_y = l1_center_references[0]
            furthest_l1_center_e_y = max(
                l1_center_references,
                key=lambda e_y: abs(
                    e_y - self._l1_soft_rejoin_start_e_y),
            )
            max_horizon_distance = abs(
                furthest_l1_center_e_y
                - self._l1_soft_rejoin_start_e_y
            )
            uncapped_ramp_sec = lateral_reference_ramp_duration(
                self._l1_soft_rejoin_start_e_y,
                furthest_l1_center_e_y,
                self._l1_soft_rejoin_ramp_sec,
                self._l1_soft_rejoin_max_reference_speed,
            )
            self._l1_soft_rejoin_effective_ramp_sec = (
                min(uncapped_ramp_sec, self._l1_soft_rejoin_max_ramp_sec)
                if self._l1_soft_rejoin_max_ramp_sec > 0.0
                else uncapped_ramp_sec
            )
            self._l1_soft_rejoin_full_strength_logged = False
            self.get_logger().info(
                "[L1SoftRejoin] starting objective-only L1 guidance with "
                "full-width constraints: "
                f"start_e_y={self._l1_soft_rejoin_start_e_y:.2f}m, "
                f"target_e_y={l1_center_e_y:.2f}m, "
                f"max_horizon_distance={max_horizon_distance:.2f}m, "
                f"ramp_sec={self._l1_soft_rejoin_effective_ramp_sec:.2f}, "
                f"uncapped_ramp_sec={uncapped_ramp_sec:.2f}, "
                f"max_reference_speed="
                f"{self._l1_soft_rejoin_max_reference_speed:.2f}m/s"
            )

        elapsed = max(
            float(now_sec) - self._l1_soft_rejoin_started_at, 0.0)
        alpha = (
            1.0
            if self._l1_soft_rejoin_effective_ramp_sec <= 0.0
            else min(
                elapsed / self._l1_soft_rejoin_effective_ramp_sec, 1.0)
        )
        self._mpcN_center.set_soft_lateral_reference(
            lane_idx=1,
            start_e_y=self._l1_soft_rejoin_start_e_y,
            alpha=alpha,
        )
        if alpha >= 1.0 and not self._l1_soft_rejoin_full_strength_logged:
            self._l1_soft_rejoin_full_strength_logged = True
            self.get_logger().info(
                "[L1SoftRejoin] L1 guidance reached full strength; "
                "constraints remain full width until L1 probe."
            )

    def _update_initial_start_soft_l0_reference(
        self, *, enabled: bool, now_sec: float
    ) -> None:
        """Ramp full-width MPC's objective toward the L1 side of L0."""
        if not enabled:
            self._initial_start_soft_l0_started_at = None
            self._initial_start_soft_l0_start_e_y = None
            self._initial_start_soft_l0_effective_ramp_sec = None
            self._initial_start_soft_l0_full_strength_logged = False
            self._initial_start_soft_l0_curvature_shift = 0.0
            self._initial_start_soft_l0_last_update_sec = None
            return

        center_wp = self._carN_center.wp_id
        horizon_max_kappa = max(
            abs(self._reference_pathN_center.get_waypoint(
                center_wp + n).kappa)
            for n in range(self._mpcN_center.N + 1)
        )
        desired_curvature_shift = curvature_lateral_shift(
            horizon_max_kappa,
            self._initial_start_soft_l0_curvature_threshold,
            self._initial_start_soft_l0_curvature_gain,
            self._initial_start_soft_l0_curvature_max_shift,
        )

        if self._initial_start_soft_l0_started_at is None:
            self._initial_start_soft_l0_started_at = float(now_sec)
            self._initial_start_soft_l0_last_update_sec = float(now_sec)
            self._initial_start_soft_l0_start_e_y = float(
                self._carN_center.spatial_state.e_y)
            # Alpha starts at zero, so accepting the first curvature request
            # here does not jump xr. Later curvature changes are rate-limited.
            self._initial_start_soft_l0_curvature_shift = (
                desired_curvature_shift)
            total_l1_offset = (
                self._initial_start_soft_l0_l1_offset
                + self._initial_start_soft_l0_curvature_shift
            )
            target_references = [
                self._mpcN_center._compute_lane_center(center_wp + n, 0)
                + total_l1_offset
                for n in range(self._mpcN_center.N + 1)
            ]
            current_target_e_y = target_references[0]
            furthest_target_e_y = max(
                target_references,
                key=lambda e_y: abs(
                    e_y - self._initial_start_soft_l0_start_e_y),
            )
            max_horizon_distance = abs(
                furthest_target_e_y
                - self._initial_start_soft_l0_start_e_y
            )
            self._initial_start_soft_l0_effective_ramp_sec = (
                lateral_reference_ramp_duration(
                    self._initial_start_soft_l0_start_e_y,
                    furthest_target_e_y,
                    self._initial_start_soft_l0_ramp_sec,
                    self._initial_start_soft_l0_max_reference_speed,
                )
            )
            self._initial_start_soft_l0_full_strength_logged = False
            self.get_logger().info(
                "[InitialStartSoftL0] starting objective-only guidance with "
                "full-width constraints: "
                f"start_e_y={self._initial_start_soft_l0_start_e_y:.2f}m, "
                f"target_e_y={current_target_e_y:.2f}m, "
                "base_l1_offset="
                f"{self._initial_start_soft_l0_l1_offset:.2f}m, "
                f"max_kappa={horizon_max_kappa:.4f}1/m, "
                f"curvature_shift="
                f"{self._initial_start_soft_l0_curvature_shift:.2f}m, "
                f"max_horizon_distance={max_horizon_distance:.2f}m, "
                "ramp_sec="
                f"{self._initial_start_soft_l0_effective_ramp_sec:.2f}, "
                "max_reference_speed="
                f"{self._initial_start_soft_l0_max_reference_speed:.2f}m/s"
            )

        last_update_sec = self._initial_start_soft_l0_last_update_sec
        dt = (
            0.0
            if last_update_sec is None
            else max(float(now_sec) - float(last_update_sec), 0.0)
        )
        self._initial_start_soft_l0_last_update_sec = float(now_sec)
        max_shift_step = (
            self._initial_start_soft_l0_max_reference_speed * dt)
        shift_error = (
            desired_curvature_shift
            - self._initial_start_soft_l0_curvature_shift
        )
        if self._initial_start_soft_l0_max_reference_speed <= 0.0:
            self._initial_start_soft_l0_curvature_shift = (
                desired_curvature_shift)
        else:
            self._initial_start_soft_l0_curvature_shift += float(np.clip(
                shift_error,
                -max_shift_step,
                max_shift_step,
            ))
        total_l1_offset = (
            self._initial_start_soft_l0_l1_offset
            + self._initial_start_soft_l0_curvature_shift
        )

        elapsed = max(
            float(now_sec) - self._initial_start_soft_l0_started_at, 0.0)
        alpha = (
            1.0
            if self._initial_start_soft_l0_effective_ramp_sec <= 0.0
            else min(
                elapsed
                / self._initial_start_soft_l0_effective_ramp_sec,
                1.0,
            )
        )
        self._mpcN_center.set_soft_lateral_reference(
            lane_idx=0,
            start_e_y=self._initial_start_soft_l0_start_e_y,
            alpha=alpha,
            lateral_offset=total_l1_offset,
        )
        self.get_logger().info(
            "[InitialStartSoftL0Curvature] "
            f"wp={center_wp}, max_kappa={horizon_max_kappa:.4f}1/m, "
            f"desired_shift={desired_curvature_shift:.2f}m, "
            "applied_shift="
            f"{self._initial_start_soft_l0_curvature_shift:.2f}m, "
            f"total_l1_offset={total_l1_offset:.2f}m",
            throttle_duration_sec=1.0,
        )
        if (
            alpha >= 1.0
            and not self._initial_start_soft_l0_full_strength_logged
        ):
            self._initial_start_soft_l0_full_strength_logged = True
            self.get_logger().info(
                "[InitialStartSoftL0] L0-side guidance reached full strength; "
                "constraints remain full width."
            )

    def _update_legacy_overtake_transition_soft_reference(
        self, *, enabled: bool, lane_idx, now_sec: float,
        transition_end_sec: float, transition_duration_sec: float,
    ) -> None:
        """Guide full-width MPC toward a confirmed outer lane during taper."""
        if not enabled or lane_idx not in (0, 2):
            self._overtake_soft_transition_lane_idx = None
            self._overtake_soft_transition_start_e_y = None
            return

        if self._overtake_soft_transition_lane_idx != lane_idx:
            self._overtake_soft_transition_lane_idx = int(lane_idx)
            self._overtake_soft_transition_start_e_y = float(
                self._carN_center.spatial_state.e_y)
            self.get_logger().info(
                "[OvertakeSoftTransition] guiding full-width MPC toward "
                f"L{lane_idx} while the lane constraint is tapered in: "
                f"start_e_y={self._overtake_soft_transition_start_e_y:.2f}m, "
                f"duration={transition_duration_sec:.2f}s"
            )

        remaining = max(float(transition_end_sec) - float(now_sec), 0.0)
        alpha = (
            1.0 if transition_duration_sec <= 0.0
            else float(np.clip(
                1.0 - remaining / float(transition_duration_sec), 0.0, 1.0
            ))
        )
        # Called after the other objective-only guidance updaters, so a newly
        # confirmed overtake owns the reference for this transition only.
        self._mpcN_center.set_soft_lateral_reference(
            lane_idx=lane_idx,
            start_e_y=self._overtake_soft_transition_start_e_y,
            alpha=alpha,
            lateral_targets=(
                self._l2_inward_targets(self._carN_center.wp_id)
                if lane_idx == 2 and self._l2_inward_offset_zones
                else None
            ),
        )


    def _closed_path_distance(self, start_wp: int, end_wp: int) -> float:
        """Return forward center-path distance between two waypoint indices."""
        path = self._reference_pathN_center
        count = int(path.n_waypoints)
        if count <= 0:
            return 0.0
        start = int(start_wp) % count
        end = int(end_wp) % count
        distance = 0.0
        index = start
        while index != end:
            distance += float(path.segment_lengths[index])
            index = (index + 1) % count
            if index == start:
                break
        return distance


    def _hybrid_horizon_distances(self, current_wp: int):
        """Distances from the latched manoeuvre start through the MPC horizon."""
        start_wp = self._overtake.hybrid.start_wp
        if start_wp is None:
            return np.zeros(self._mpcN_center.N + 1, dtype=float)
        path = self._reference_pathN_center
        hybrid = self._overtake.hybrid
        previous_wp = hybrid.last_wp if hybrid.last_wp is not None else start_wp
        forward = self._closed_path_distance(previous_wp, current_wp)
        backward = self._closed_path_distance(current_wp, previous_wp)
        delta = forward if forward <= backward else -backward
        hybrid.travelled += delta
        hybrid.last_wp = int(current_wp)
        distances = [max(hybrid.travelled, 0.0)]
        for n in range(self._mpcN_center.N):
            segment_index = (int(current_wp) + n) % path.n_waypoints
            distances.append(
                distances[-1] + float(path.segment_lengths[segment_index]))
        return np.asarray(distances, dtype=float)


    def _shifted_center_prediction_lateral(self, current_wp: int):
        """Project the previous XY prediction onto the next Center horizon."""
        prediction = self._mpcN_center.current_prediction
        if (
            prediction is None
            or self._hybrid_reference_key != (
                self._overtake.hybrid.vehicle_id, self._overtake.hybrid.lane_idx)
            or self._mpcN_center.used_prediction_fallback
            or self._mpcN_center.recovery_requested
            or not bool(getattr(self._mpcN_center, "last_solution_accurate", False))
        ):
            return None
        pred_x, pred_y = prediction
        if len(pred_x) < 2 or len(pred_y) < 2:
            return None
        values = []
        for n in range(self._mpcN_center.N + 1):
            prediction_index = min(n + 1, len(pred_x) - 1, len(pred_y) - 1)
            wp = self._reference_pathN_center.get_waypoint(current_wp + n)
            values.append(
                np.cos(wp.psi) * (float(pred_y[prediction_index]) - wp.y)
                - np.sin(wp.psi) * (float(pred_x[prediction_index]) - wp.x)
            )
        return np.asarray(values, dtype=float)


    def _update_overtake_transition_soft_reference(
        self, *, enabled: bool, lane_idx, now_sec: float,
        ego_speed: float = 0.0, vehicle_id=None,
    ) -> None:
        """Apply a spatial quintic reference and matching lane contraction."""
        session_matches = bool(
            self._overtake.hybrid.lane_idx in (0, 2)
            and not self._overtake.hybrid.completed
            and self._overtake.hybrid.vehicle_id is not None
            and vehicle_id == self._overtake.hybrid.vehicle_id
            and (
                lane_idx == self._overtake.hybrid.lane_idx
                or self._overtake.requested_lane
                    == self._overtake.hybrid.lane_idx
            )
        )
        if not enabled and session_matches:
            self._hybrid_reference_key = None
            # Full-width safety recovery temporarily owns constraints. Keep
            # the manoeuvre's spatial anchor so the same target/side resumes
            # from travelled distance instead of restarting at zero.
            self._mpcN_center.set_lane_transition_weights()
            if not self._overtake.hybrid.paused:
                self._overtake.hybrid.paused = True
                self.get_logger().info(
                    "[HybridOvertakePaused] preserving spatial progress "
                    "during temporary full-width/recovery ownership: "
                    f"vehicle_id={vehicle_id}, lane="
                    f"L{self._overtake.hybrid.lane_idx}"
                )
            return
        if not enabled or lane_idx not in (0, 2):

            self._overtake.clear_hybrid()
            self._hybrid_reference_key = None
            self._mpcN_center.set_lane_transition_weights()
            return

        current_wp = int(self._carN_center.wp_id)
        if (
            self._overtake.hybrid.lane_idx != lane_idx
            or self._overtake.hybrid.vehicle_id != vehicle_id
        ):
            start_e_y = float(self._carN_center.spatial_state.e_y)
            first_center = self._mpcN_center._compute_lane_center(
                current_wp, lane_idx)
            lateral_distance = abs(
                first_center - start_e_y)
            requested_length = (
                self._hybrid_overtake_base_length
                + self._hybrid_overtake_offset_gain * lateral_distance
                + self._hybrid_overtake_speed_gain * max(float(ego_speed), 0.0)
            )
            low_speed_start = bool(
                max(float(ego_speed), 0.0)
                <= self._hybrid_overtake_low_speed_start_threshold
            )
            if low_speed_start:
                # After a stop/recovery there is little longitudinal room to
                # spend on a long, gentle lateral transition.  Keeping the
                # quintic profile but completing it sooner establishes body
                # separation before the kart reaches the parallel phase, so
                # EmergencyBrake need not insert a one-cycle full stop.
                requested_length = min(
                    requested_length,
                    self._hybrid_overtake_low_speed_max_length,
                )
            length = float(np.clip(
                requested_length,
                self._hybrid_overtake_min_length,
                self._hybrid_overtake_max_length,
            ))
            self._overtake.start_hybrid(
                vehicle_id=vehicle_id, lane_idx=int(lane_idx),
                start_wp=current_wp, started_at=float(now_sec),
                start_e_y=start_e_y, length=length,
            )

            self.get_logger().info(
                "[HybridOvertake] starting spatial quintic transition toward "
                f"L{lane_idx}: "
                f"start_e_y={self._overtake.hybrid.start_e_y:.2f}m, "
                f"length={self._overtake.hybrid.length:.2f}m, "
                f"speed={max(float(ego_speed), 0.0):.2f}m/s, "
                f"low_speed_shortened={low_speed_start}"
            )
        elif self._overtake.hybrid.paused:
            self._overtake.hybrid.paused = False
            self.get_logger().info(
                "[HybridOvertakeResumed] continuing the existing spatial "
                "transition without resetting progress: "
                f"vehicle_id={vehicle_id}, lane=L{lane_idx}, "
                f"travelled={self._closed_path_distance(self._overtake.hybrid.start_wp, current_wp):.2f}m/"
                f"{self._overtake.hybrid.length:.2f}m"
            )

        distances = self._hybrid_horizon_distances(current_wp)
        lane_centers = np.asarray([
            self._mpcN_center._compute_lane_center(current_wp + n, lane_idx)
            for n in range(self._mpcN_center.N + 1)
        ], dtype=float)
        if lane_idx == 2 and self._l2_inward_offset_zones:
            lane_centers = np.asarray(
                self._l2_inward_targets(current_wp), dtype=float)
        targets, weights = spatial_lane_transition_reference(
            distances,
            self._overtake.hybrid.start_e_y,
            lane_centers,
            self._overtake.hybrid.length,
        )
        previous_lateral = self._shifted_center_prediction_lateral(current_wp)
        continuity_used = bool(
            previous_lateral is not None
            and len(previous_lateral) == len(targets)
            and np.max(np.abs(previous_lateral - targets))
                <= self._hybrid_overtake_continuity_max_deviation
        )
        targets = blend_previous_lateral_prediction(
            targets,
            previous_lateral,
            self._hybrid_overtake_continuity_weight,
            self._hybrid_overtake_continuity_max_deviation,
        )
        self._mpcN_center.set_soft_lateral_reference(
            lane_idx=lane_idx,
            start_e_y=self._overtake.hybrid.start_e_y,
            alpha=1.0,
            lateral_targets=targets,
        )
        self._mpcN_center.set_lane_transition_weights(weights[1:])
        self._hybrid_reference_key = (vehicle_id, lane_idx)
        self.get_logger().info(
            "[HybridOvertakeProgress] "
            f"vehicle_id={vehicle_id}, lane=L{lane_idx}, "
            f"travelled={distances[0]:.2f}m/"
            f"{self._overtake.hybrid.length:.2f}m, "
            f"weight_near={weights[0]:.3f}, weight_far={weights[-1]:.3f}, "
            f"continuity={'used' if continuity_used else 'nominal'}",
            throttle_duration_sec=0.5,
        )
        hybrid = self._overtake.hybrid
        hybrid.reference_completed = bool(distances[0] >= 0.98 * hybrid.length)
        lateral_error = abs(float(self._carN_center.spatial_state.e_y) - float(lane_centers[0]))
        # Finishing the reference is not evidence that the kart reached it.
        # Keep saturated spatial guidance and side-lock/creep ownership until
        # actual lateral tracking has caught up. Never restart the anchor.
        if hybrid.reference_completed and lateral_error <= self._slow_lead_speed_match_release_lateral_error:
            self._constraint_transition_until = min(float(self._constraint_transition_until), float(now_sec))
            if not hybrid.completed:
                hybrid.completed = True
                self.get_logger().info(
                    f"[HybridOvertakeCompleted] reference and actual lane arrival confirmed: "
                    f"vehicle_id={vehicle_id}, lane=L{lane_idx}, lateral_error={lateral_error:.2f}m")
        elif hybrid.reference_completed:
            self.get_logger().info(
                f"[HybridLaneArrivalWait] vehicle_id={vehicle_id}, lane=L{lane_idx}, "
                f"lateral_error={lateral_error:.2f}m; retaining spatial manoeuvre",
                throttle_duration_sec=0.5)

    def _latch_prepass_soft_candidate(self, lane_idx, now_sec: float):
        """Debounce the L0/L2 candidate used only by Prepass soft guidance."""
        raw_lane = int(lane_idx) if lane_idx in (0, 2) else None
        latched_lane = self._prepass_soft_candidate_lane_idx

        if latched_lane not in (0, 2):
            if raw_lane is not None:
                self._prepass_soft_candidate_lane_idx = raw_lane
                self._prepass_soft_candidate_last_seen_at = float(now_sec)
            self._prepass_soft_pending_lane_idx = None
            self._prepass_soft_pending_since = None
            return self._prepass_soft_candidate_lane_idx

        if raw_lane == latched_lane:
            self._prepass_soft_candidate_last_seen_at = float(now_sec)
            self._prepass_soft_pending_lane_idx = None
            self._prepass_soft_pending_since = None
            return latched_lane

        if raw_lane is None:
            last_seen = self._prepass_soft_candidate_last_seen_at
            if (
                last_seen is not None
                and float(now_sec) - last_seen
                <= self._prepass_soft_dropout_grace_sec
            ):
                return latched_lane
            self._prepass_soft_candidate_lane_idx = None
            self._prepass_soft_pending_lane_idx = None
            self._prepass_soft_pending_since = None
            return None

        if self._prepass_soft_pending_lane_idx != raw_lane:
            self._prepass_soft_pending_lane_idx = raw_lane
            self._prepass_soft_pending_since = float(now_sec)
            return latched_lane

        pending_for = (
            float(now_sec) - self._prepass_soft_pending_since
            if self._prepass_soft_pending_since is not None else 0.0
        )
        if pending_for < self._prepass_soft_switch_confirm_sec:
            return latched_lane

        old_lane = latched_lane
        self._prepass_soft_candidate_lane_idx = raw_lane
        self._prepass_soft_candidate_last_seen_at = float(now_sec)
        self._prepass_soft_pending_lane_idx = None
        self._prepass_soft_pending_since = None
        self.get_logger().info(
            "[PrepassSoftCandidateSwitch] opposite outer lane remained "
            f"stable for {pending_for:.2f}s: L{old_lane} -> L{raw_lane}"
        )
        return raw_lane

    def _update_prepass_soft_reference(
        self, *, enabled: bool, lane_idx, now_sec: float
    ) -> None:
        """Turn full-width Prepass recovery toward its verified candidate."""
        if not enabled or lane_idx not in (0, 2):
            self._clear_prepass_soft_guidance()
            return

        if (
            self._prepass_soft_candidate_lane_idx != lane_idx
            or self._prepass_soft_guidance_started_at is None
            or self._prepass_soft_guidance_start_e_y is None
            or self._prepass_soft_guidance_ramp_sec is None
        ):
            self._prepass_soft_candidate_lane_idx = int(lane_idx)
            self._prepass_soft_guidance_started_at = float(now_sec)
            self._prepass_soft_guidance_start_e_y = float(
                self._carN_center.spatial_state.e_y)
            center_wp = self._carN_center.wp_id
            target_centers = [
                self._mpcN_center._compute_lane_center(center_wp + n, lane_idx)
                for n in range(self._mpcN_center.N + 1)
            ]
            furthest_target = max(
                target_centers,
                key=lambda value: abs(
                    value - self._prepass_soft_guidance_start_e_y),
            )
            self._prepass_soft_guidance_ramp_sec = (
                lateral_reference_ramp_duration(
                    self._prepass_soft_guidance_start_e_y,
                    furthest_target,
                    self._l1_soft_rejoin_ramp_sec,
                    self._race_handoff_max_reference_speed,
                )
            )
            self.get_logger().info(
                "[PrepassSoftGuidance] turning full-width recovery toward "
                f"verified L{lane_idx}: start_e_y="
                f"{self._prepass_soft_guidance_start_e_y:.2f}m, "
                f"ramp_sec={self._prepass_soft_guidance_ramp_sec:.2f}s"
            )

        elapsed = max(
            float(now_sec) - self._prepass_soft_guidance_started_at, 0.0)
        alpha = (
            1.0 if self._prepass_soft_guidance_ramp_sec <= 0.0
            else min(elapsed / self._prepass_soft_guidance_ramp_sec, 1.0)
        )
        self._mpcN_center.set_soft_lateral_reference(
            lane_idx=lane_idx,
            start_e_y=self._prepass_soft_guidance_start_e_y,
            alpha=alpha,
            lateral_targets=(
                self._l2_inward_targets(self._carN_center.wp_id)
                if lane_idx == 2 and self._l2_inward_offset_zones
                else None
            ),
        )

    def _current_prediction_l1_fit_ratio(self) -> float:
        """Measure how much of the full MPC center prediction lies in L1."""
        if self._mpc.current_prediction is None:
            return 0.0
        pred_x, pred_y = self._mpc.current_prediction
        available_count = min(len(pred_x), len(pred_y))
        count = (
            available_count
            if self._l1_prediction_fit_points <= 0
            else min(available_count, self._l1_prediction_fit_points)
        )
        if count <= 0:
            return 0.0
        inside = 0
        car = self._carN_center
        reference_path = self._reference_pathN_center
        for x, y in zip(pred_x[:count], pred_y[:count]):
            wp_id = car.get_closest_waypoint(float(x), float(y))
            wp = reference_path.get_waypoint(wp_id)
            lanes = reference_path.get_lane_bounds(wp_id)
            if len(lanes) <= 1:
                continue
            upper, lower = lanes[1]
            normal_angle = wp.psi + math.pi / 2.0
            lateral_offset = (
                (float(x) - wp.x) * math.cos(normal_angle)
                + (float(y) - wp.y) * math.sin(normal_angle)
            )
            if lower <= lateral_offset <= upper:
                inside += 1
        return float(inside) / float(count)

    def _log_lane_constraint_diagnostics(
        self, *, lane_idx: int, context: str, failed_wp: int, reason: str
    ) -> None:
        """Log the exact selected-lane corridor after a failed solve.

        The bound snapshot retained by MPC represents the final relaxed retry,
        so this deliberately does not recompute a potentially different
        corridor.  For an outer lane, a prediction horizon crossing the
        circular waypoint seam is reported separately.
        """
        if not self._constraint_diagnostics_enabled:
            return
        lane_idx = int(lane_idx)
        lane_tag = f"L{lane_idx}"
        if lane_idx not in (0, 1, 2):
            self.get_logger().warn(
                "[LaneConstraintDiagnostic] invalid lane index: "
                f"lane_idx={lane_idx}, context={context}, failed_wp={failed_wp}"
            )
            return
        try:
            upper = np.asarray(
                getattr(self._mpc, "_prediction_upper_bounds", []),
                dtype=float,
            ).reshape(-1)
            lower = np.asarray(
                getattr(self._mpc, "_prediction_lower_bounds", []),
                dtype=float,
            ).reshape(-1)
            wp_ids = np.asarray(
                getattr(self._mpc, "_constraint_wp_ids", []),
                dtype=int,
            ).reshape(-1)
            count = min(len(upper), len(lower), len(wp_ids))
            if count <= 0:
                self.get_logger().warn(
                    f"[{lane_tag}ConstraintDiagnostic] corridor snapshot unavailable: "
                    f"context={context}, failed_wp={failed_wp}, reason={reason}"
                )
                return

            reference_path = self._reference_path
            rows = []
            horizon_centers = []
            for i in range(count):
                wp_id = int(wp_ids[i])
                wp = reference_path.get_waypoint(wp_id)
                lanes = reference_path.get_lane_bounds(wp_id)
                if lane_idx >= len(lanes):
                    base_upper = math.nan
                    base_lower = math.nan
                    base_width = math.nan
                    lane_center = 0.0
                else:
                    base_upper, base_lower = lanes[lane_idx]
                    base_width = float(base_upper - base_lower)
                    lane_center = 0.5 * float(base_upper + base_lower)
                effective_width = float(upper[i] - lower[i])
                center_margin = min(
                    float(upper[i]) - lane_center,
                    lane_center - float(lower[i]),
                )
                narrowing = (
                    float(base_width - effective_width)
                    if math.isfinite(base_width) else math.nan
                )
                normal_angle = wp.psi + math.pi / 2.0
                center_x = wp.x + lane_center * math.cos(normal_angle)
                center_y = wp.y + lane_center * math.sin(normal_angle)
                horizon_centers.append((center_x, center_y, wp_id))
                rows.append({
                    "index": i,
                    "wp": wp_id,
                    "base_upper": float(base_upper),
                    "base_lower": float(base_lower),
                    "base_width": base_width,
                    "upper": float(upper[i]),
                    "lower": float(lower[i]),
                    "width": effective_width,
                    "center_margin": center_margin,
                    "narrowing": narrowing,
                    "x": float(wp.x),
                    "y": float(wp.y),
                    "psi": float(wp.psi),
                    "kappa": float(wp.kappa),
                    "lane_center_x": float(center_x),
                    "lane_center_y": float(center_y),
                })

            widths = np.asarray([row["width"] for row in rows])
            base_widths = np.asarray([row["base_width"] for row in rows])
            center_margins = np.asarray([
                row["center_margin"] for row in rows])
            invalid_count = int(np.count_nonzero(
                ~np.isfinite(widths) | (widths <= 0.0)))
            vehicle_width = float(self._cfg.bicycle_model.width)
            required_width = lane_minimum_free_segment_width(
                lane_idx,
                vehicle_width,
                self._reference_path.inner_lane_width,
            )
            below_vehicle_count = int(np.count_nonzero(
                np.isfinite(widths) & (widths < required_width)))
            center_excluded_count = int(np.count_nonzero(
                np.isfinite(center_margins) & (center_margins < 0.0)))

            obstacle_records = []
            seen_obstacles = set()
            for obstacle in getattr(self._map, "obstacles", []):
                key = (
                    round(float(obstacle.cx), 2),
                    round(float(obstacle.cy), 2),
                    round(float(obstacle.radius), 2),
                )
                if key in seen_obstacles:
                    continue
                seen_obstacles.add(key)
                nearest = min(
                    (
                        math.hypot(
                            float(obstacle.cx) - center_x,
                            float(obstacle.cy) - center_y,
                        ) - float(obstacle.radius),
                        index,
                        wp_id,
                    )
                    for index, (center_x, center_y, wp_id)
                    in enumerate(horizon_centers)
                )
                obstacle_records.append((
                    nearest[0], nearest[1], nearest[2],
                    float(obstacle.cx), float(obstacle.cy),
                    float(obstacle.radius),
                ))
            obstacle_records.sort(key=lambda item: item[0])
            nearest_obstacle_clearance = (
                obstacle_records[0][0] if obstacle_records else math.inf)

            finite_narrowing = [
                row["narrowing"] for row in rows
                if math.isfinite(row["narrowing"])
            ]
            max_narrowing = max(finite_narrowing, default=0.0)
            if invalid_count:
                likely_cause = "invalid_or_collapsed_bounds"
            elif below_vehicle_count:
                likely_cause = "effective_width_below_required_width"
            elif center_excluded_count:
                likely_cause = (
                    f"{lane_tag.lower()}_center_excluded_from_free_segment")
            elif max_narrowing > 0.20:
                likely_cause = (
                    "obstacle_narrowing"
                    if nearest_obstacle_clearance < required_width
                    else "static_map_or_boundary_narrowing"
                )
            else:
                likely_cause = "solver_dynamics_or_rate_constraints"

            min_row = min(rows, key=lambda row: row["width"])
            self.get_logger().warn(
                f"[{lane_tag}ConstraintDiagnostic] "
                f"context={context}, failed_wp={failed_wp}, reason={reason}, "
                f"mpc_failure={self._mpc.failure_reason}, "
                f"target_lane={getattr(self._mpc, '_constraint_target_lane', None)}, "
                f"safety_margin={getattr(self._mpc, '_constraint_safety_margin', math.nan):.3f}m, "
                f"horizon={count}, base_width_min={np.nanmin(base_widths):.3f}m, "
                f"effective_width_min={min_row['width']:.3f}m@"
                f"i{min_row['index']}/wp{min_row['wp']}, "
                f"invalid={invalid_count}, below_required_width="
                f"{below_vehicle_count}/{count}, required_width="
                f"{required_width:.3f}m, center_excluded="
                f"{center_excluded_count}/{count}, max_narrowing="
                f"{max_narrowing:.3f}m, nearest_obstacle_clearance="
                f"{nearest_obstacle_clearance:.3f}m, likely_cause={likely_cause}"
            )

            detail_rows = sorted(rows, key=lambda row: row["width"])[
                :self._constraint_diagnostics_points]
            detail = "; ".join(
                f"i{row['index']}/wp{row['wp']}:"
                f"base=[{row['base_lower']:.2f},{row['base_upper']:.2f}]"
                f"({row['base_width']:.2f}m),"
                f"effective=[{row['lower']:.2f},{row['upper']:.2f}]"
                f"({row['width']:.2f}m),"
                f"center_margin={row['center_margin']:.2f}m"
                for row in detail_rows
            )
            self.get_logger().warn(
                f"[{lane_tag}ConstraintDetail] {detail}"
            )

            # A circular horizon should contain exactly one descending
            # waypoint transition (for example 311 -> 0).  Compare the seam
            # against every other adjacent horizon step so discontinuities in
            # geometry, curvature, or bounds are visible in one log record.
            seam_indices = [
                i for i in range(1, len(rows))
                if rows[i]["wp"] < rows[i - 1]["wp"]
            ]
            if lane_idx in (0, 2) and seam_indices:
                adjacent_metrics = []
                for i in range(1, len(rows)):
                    previous = rows[i - 1]
                    current = rows[i]
                    heading_delta = abs(math.atan2(
                        math.sin(current["psi"] - previous["psi"]),
                        math.cos(current["psi"] - previous["psi"]),
                    ))
                    adjacent_metrics.append({
                        "index": i,
                        "path_step": math.hypot(
                            current["x"] - previous["x"],
                            current["y"] - previous["y"],
                        ),
                        "lane_center_step": math.hypot(
                            current["lane_center_x"]
                            - previous["lane_center_x"],
                            current["lane_center_y"]
                            - previous["lane_center_y"],
                        ),
                        "heading_delta_deg": math.degrees(heading_delta),
                        "kappa_delta": abs(
                            current["kappa"] - previous["kappa"]),
                    })
                max_path_step = max(
                    metric["path_step"] for metric in adjacent_metrics)
                max_lane_center_step = max(
                    metric["lane_center_step"] for metric in adjacent_metrics)
                max_heading_delta = max(
                    metric["heading_delta_deg"] for metric in adjacent_metrics)
                max_kappa_delta = max(
                    metric["kappa_delta"] for metric in adjacent_metrics)
                for seam_index in seam_indices:
                    previous = rows[seam_index - 1]
                    current = rows[seam_index]
                    seam_metric = next(
                        metric for metric in adjacent_metrics
                        if metric["index"] == seam_index)
                    self.get_logger().warn(
                        f"[{lane_tag}CircularSeamDiagnostic] "
                        f"context={context}, transition=i{seam_index - 1}/"
                        f"wp{previous['wp']}->i{seam_index}/wp{current['wp']}, "
                        f"path_step={seam_metric['path_step']:.3f}m/"
                        f"horizon_max={max_path_step:.3f}m, "
                        "lane_center_step="
                        f"{seam_metric['lane_center_step']:.3f}m/"
                        f"horizon_max={max_lane_center_step:.3f}m, "
                        "heading_jump="
                        f"{seam_metric['heading_delta_deg']:.2f}deg/"
                        f"horizon_max={max_heading_delta:.2f}deg, "
                        f"kappa={previous['kappa']:.5f}->"
                        f"{current['kappa']:.5f}1/m, "
                        f"kappa_jump={seam_metric['kappa_delta']:.5f}/"
                        f"horizon_max={max_kappa_delta:.5f}1/m, "
                        "base_bounds="
                        f"[{previous['base_lower']:.3f},"
                        f"{previous['base_upper']:.3f}]->"
                        f"[{current['base_lower']:.3f},"
                        f"{current['base_upper']:.3f}], "
                        "effective_bounds="
                        f"[{previous['lower']:.3f},{previous['upper']:.3f}]->"
                        f"[{current['lower']:.3f},{current['upper']:.3f}], "
                        "effective_width="
                        f"{previous['width']:.3f}->{current['width']:.3f}m"
                    )

            if obstacle_records:
                obstacle_detail = "; ".join(
                    f"i{index}/wp{wp_id}:clearance={clearance:.2f}m,"
                    f"pos=({cx:.2f},{cy:.2f}),r={radius:.2f}m"
                    for clearance, index, wp_id, cx, cy, radius
                    in obstacle_records[:3]
                )
                self.get_logger().warn(
                    f"[{lane_tag}ObstacleDiagnostic] {obstacle_detail}"
                )

            if hasattr(self, "_v2x_tracker"):
                v2x_records = []
                for vehicle_id in self._v2x_tracker.active_vehicle_ids():
                    samples = self._v2x_tracker._samples.get(vehicle_id)
                    if not samples:
                        continue
                    _, vehicle_x, vehicle_y = samples[-1]
                    clearance, index, wp_id = min(
                        (
                            math.hypot(
                                float(vehicle_x) - center_x,
                                float(vehicle_y) - center_y,
                            ) - self._v2x_vehicle_radius,
                            index,
                            wp_id,
                        )
                        for index, (center_x, center_y, wp_id)
                        in enumerate(horizon_centers)
                    )
                    velocity_x, velocity_y = self._v2x_tracker.velocity(
                        vehicle_id)
                    v2x_records.append((
                        clearance, str(vehicle_id), index, wp_id,
                        math.hypot(velocity_x, velocity_y),
                        float(vehicle_x), float(vehicle_y),
                    ))
                v2x_records.sort(key=lambda item: item[0])
                if v2x_records:
                    v2x_detail = "; ".join(
                        f"vehicle_id={vehicle_id},i{index}/wp{wp_id},"
                        f"clearance={clearance:.2f}m,speed={speed:.2f}m/s,"
                        f"pos=({vehicle_x:.2f},{vehicle_y:.2f})"
                        for clearance, vehicle_id, index, wp_id, speed,
                        vehicle_x, vehicle_y in v2x_records[:3]
                    )
                    self.get_logger().warn(
                        f"[{lane_tag}V2XDiagnostic] {v2x_detail}"
                    )
        except Exception as error:
            # Diagnostics must never interfere with the safety state machine.
            self.get_logger().warn(
                f"[{lane_tag}ConstraintDiagnostic] collection failed: "
                f"context={context}, failed_wp={failed_wp}, error={error}"
            )

    def _log_l1_constraint_diagnostics(
        self, *, context: str, failed_wp: int, reason: str
    ) -> None:
        """Backward-compatible entry point for existing L1 failure paths."""
        self._log_lane_constraint_diagnostics(
            lane_idx=1,
            context=context,
            failed_wp=failed_wp,
            reason=reason,
        )

    def _reset_l1_rejoin_backoff(self) -> None:
        self._l1_rejoin_backoff_active = False
        self._l1_rejoin_backoff_started_at = None
        self._l1_rejoin_backoff_failed_wp = None
        self._l1_rejoin_backoff_full_width_success_since = None

    def _start_l1_rejoin_backoff(
        self, now_sec: float, failed_wp: int, reason: str
    ) -> None:
        self._l1_rejoin_backoff_active = True
        self._l1_rejoin_backoff_started_at = float(now_sec)
        self._l1_rejoin_backoff_failed_wp = int(failed_wp)
        self._l1_rejoin_backoff_full_width_success_since = None
        self._center_lane_rejoin_stable_since = None
        self.get_logger().warn(
            "[L1ProbeBackoff] L1 probe failed; holding full width until "
            "cooldown, waypoint progress, and fresh MPC recovery complete: "
            f"failed_wp={failed_wp}, reason={reason}, "
            f"cooldown={self._l1_probe_retry_cooldown_sec:.2f}s, "
            f"min_wp_progress={self._l1_probe_retry_min_wp_progress}, "
            f"full_width_success="
            f"{self._l1_backoff_full_width_success_sec:.2f}s"
        )

    def _switch_prepass_to_follow(self, reason: str) -> None:
        self._prepass_fallback_blocked = False
        self._prepass_fallback_follow_active = True
        self._prepass_follow_last_retry_at = (
            float(self.get_clock().now().nanoseconds) / 1e9
        )
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_recovery_started_at = None
        self._prepass_target_behind_since = None
        self._prepass_failed_lane_idx = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_fallback_commit_success_since = None
        self._prepass_attempted_outer_lanes.clear()
        self._prepass_retry_after_reverse = False
        self._prepass_retry_lane_idx = None
        self._forced_overtake_vehicle_id = None
        self._close_obstacle_reverse_requested = False
        if (
            self._follow_latched_cache is not None
            and self._follow_latched_cache["vehicle_id"]
                != self._overtake.target_id
        ):
            self._follow_latched_cache = None
        self.get_logger().warn(f"[PrepassLaneFallbackFollow] {reason}")

    def _cancel_normal_l1_rejoin_for_prepass(self) -> None:
        """Give a newly started Prepass recovery exclusive lateral ownership."""
        self._center_lane_rejoin_active = False
        self._center_lane_rejoin_constraint_released = False
        self._center_lane_rejoin_stable_since = None
        self._reset_l1_rejoin_backoff()
        if self._l1_probe_active and self._l1_probe_context == "rejoin":
            self._l1_probe_active = False
            self._l1_probe_context = None
            self._l1_probe_success_cycles = 0
            self._l1_probe_constraint_applied = False

    def _cancel_l1_rejoin_for_overtake(self) -> None:
        """Let a newly verified outer-lane pass preempt normal L1 rejoin."""
        self._cancel_normal_l1_rejoin_for_prepass()
        self._l1_safety_recovery_active = False
        self._l1_safety_recovery_stable_since = None
        self._l1_safety_recovery_context = None
        self._l1_safety_reprobe_pending = False
        self._l1_probe_active = False
        self._l1_probe_context = None
        self._l1_probe_success_cycles = 0
        self._l1_probe_constraint_applied = False

    def _latched_follow_target_state(self, pose, now_sec: float):
        """Resolve ACC input from the latched vehicle, with short loss hold."""
        target_id = self._overtake.target_id
        if target_id is None:
            return None
        active = target_id in self._v2x_tracker.active_vehicle_ids()
        target_buf = self._v2x_tracker._samples.get(target_id) if active else None
        if target_buf:
            _, target_x, target_y = target_buf[-1]
            velocity_x, velocity_y = self._v2x_tracker.velocity(target_id)
            velocity_valid = self._v2x_tracker.has_velocity_estimate(target_id)
            self._follow_latched_cache = {
                "vehicle_id": target_id,
                "x": target_x,
                "y": target_y,
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "velocity_valid": velocity_valid,
                "last_seen_at": now_sec,
            }
            stale = False
        else:
            cache = self._follow_latched_cache
            if cache is None or cache["vehicle_id"] != target_id:
                return {"vehicle_id": target_id, "expired": True}
            target_x = cache["x"]
            target_y = cache["y"]
            velocity_x = cache["velocity_x"]
            velocity_y = cache["velocity_y"]
            velocity_valid = cache["velocity_valid"]
            stale = True
            if now_sec - cache["last_seen_at"] > self._follow_target_lost_hold_sec:
                return {"vehicle_id": target_id, "expired": True}

        target_wp_id = self._car.get_closest_waypoint(target_x, target_y)
        target_wp = self._reference_path.get_waypoint(target_wp_id)
        normal_angle = target_wp.psi + math.pi / 2.0
        target_offset = (
            (target_x - target_wp.x) * math.cos(normal_angle)
            + (target_y - target_wp.y) * math.sin(normal_angle)
        )
        return {
            "vehicle_id": target_id,
            "x": target_x,
            "y": target_y,
            "distance": math.hypot(target_x - pose.x, target_y - pose.y),
            "longitudinal": self._center_longitudinal_between(
                pose.x, pose.y, target_x, target_y),
            "speed": math.hypot(velocity_x, velocity_y),
            "offset": target_offset,
            "velocity_valid": velocity_valid,
            "stale": stale,
            "expired": False,
        }

    def _reverse_forward_resume_ready(
        self, pose, now_sec: float, u, ego_speed: float
    ):
        """Return whether an active reverse can safely hand back to DRIVE."""
        target = self._latched_follow_target_state(pose, now_sec)
        if target is None or target.get("expired", False):
            return False, ""

        longitudinal = float(target["longitudinal"])
        lead_speed = float(target["speed"])
        start_longitudinal = self._stuck_reverse_start_target_longitudinal
        lead_advance = (
            longitudinal - start_longitudinal
            if start_longitudinal is not None else None
        )
        front_has_advanced = (
            lead_advance is not None
            and lead_advance >= self._stuck_forward_resume_gap
            and target.get("velocity_valid", False)
            and lead_speed >= self._stuck_forward_resume_lead_speed
        )
        valid_forward_mpc = (
            longitudinal > 0.0
            and self._mpc.infeasibility_counter == 0
            and self._mpc.current_prediction is not None
            and not getattr(self._mpc, "used_prediction_fallback", False)
            and not self._mpc_safety_recovery_active
            and float(u[0]) >= self._stuck_forward_resume_min_command
        )
        retry_lane_idx, _, _ = self._select_prepass_retry_lane(
            pose, ego_speed, self._overtake.requested_lane)
        current_pose_has_passage = retry_lane_idx is not None

        if front_has_advanced and valid_forward_mpc and current_pose_has_passage:
            return True, (
                "latched lead advanced with a valid forward path: "
                f"advance={lead_advance:.2f}m, "
                f"longitudinal={longitudinal:.2f}m, "
                f"speed={lead_speed:.2f}m/s, lane=L{retry_lane_idx}, "
                f"u0={float(u[0]):.2f}m/s"
            )
        return False, ""

    def _release_lost_follow_target(self, reason="hold expired") -> None:
        lost_id = self._overtake.target_id
        if lost_id is not None:
            self._follow_last_released_vehicle_id = lost_id
            self._follow_last_released_at = (
                float(self.get_clock().now().nanoseconds) / 1e9)
        self._prepass_fallback_follow_active = False
        self._prepass_follow_last_retry_at = None
        self._overtake.release_target()
        self._forced_overtake_vehicle_id = None
        self._prepass_fallback_lane_idx = None
        self._prepass_dynamic_conflict_speed_limit = None
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_recovery_started_at = None
        self._prepass_failed_lane_idx = None
        self._prepass_target_behind_since = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_fallback_commit_success_since = None
        self._prepass_attempted_outer_lanes.clear()
        self._follow_latched_cache = None
        self._clear_committed_shadow_verification()
        self.get_logger().warn(
            "[FollowTargetRelease] releasing latched follow target for "
            f"full-traffic re-evaluation: vehicle_id={lost_id}, reason={reason}"
        )

    def _preserve_stationary_group_hybrid(self, successor, lane, pose, ego_speed, now_sec):
        """Preserve geometry only after the existing next-target Shadow gate succeeds."""
        old_target = self._overtake.target_id
        hybrid = self._overtake.hybrid
        if ((hybrid.vehicle_id, hybrid.lane_idx) != (old_target, lane)
                or hybrid.start_wp is None
                or not self._v2x_tracker.has_velocity_estimate(old_target)
                or not math.isfinite(math.hypot(*self._v2x_tracker.velocity(old_target)))
                or math.hypot(*self._v2x_tracker.velocity(old_target)) > self._stopped_lead_speed_threshold
                or successor not in self._stationary_lane_group(pose, ego_speed, lane)
                or not self._overtake_commit_probe_is_fresh(successor, lane, now_sec)):
            return False
        hybrid.vehicle_id = successor
        self._overtake.accepted_key = (successor, lane)
        if self._lane_decision is not None:
            previous = self._lane_decision
            self._lane_decision = type(previous)(target_id=successor,
                requested_lane=previous.requested_lane, applied_lane=previous.applied_lane,
                mode=previous.mode)
        # The caller installs successor's fresh verification. Never reuse the
        # completed target's speed cap, passage-loss timer or traffic history.
        self._overtake.clear_verification()
        self._overtake.passage_hold.reset()
        self._overtake.traffic_key = (None, None)
        self._overtake.traffic_relevant_ids.clear()
        self._follow_latched_cache = None
        self._prepass_dynamic_conflict_speed_limit = None
        self._prepass_target_behind_since = None
        self._hybrid_reference_key = (successor, lane)
        self.get_logger().info(
            f"[StationaryGroupHandoff] {old_target}->{successor}, lane=L{lane}; "
            "fresh successor Shadow accepted; retaining spatial anchor")
        return True

    def _complete_overtake_target_behind(
        self, target_id, longitudinal: float, *, source: str
    ) -> None:
        """End all Prepass ownership and begin full-width soft L1 rejoin."""
        self._overtake.complete_pass()
        self._overtake_completed_target_id = target_id
        self._outer_lane_released_vehicle_id = target_id
        self._prepass_fallback_lane_idx = None
        self._prepass_fallback_blocked = False
        self._prepass_fallback_follow_active = False
        self._prepass_follow_last_retry_at = None
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_recovery_started_at = None
        self._prepass_dynamic_conflict_speed_limit = None
        self._prepass_target_behind_since = None
        self._prepass_failed_lane_idx = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_fallback_commit_success_since = None
        self._prepass_attempted_outer_lanes.clear()
        self._prepass_retry_after_reverse = False
        self._prepass_retry_lane_idx = None
        self._prepass_reverse_motion_started = False
        self._prepass_reverse_start_xy = None
        self._prepass_reverse_distance = 0.0
        self._clear_prepass_soft_guidance()
        self._clear_committed_shadow_verification()
        self._mpc.osqp_initialized = False
        self.get_logger().info(
            "[OvertakeTargetPhysicalComplete] target is behind with "
            "separated vehicle envelopes; ending Prepass and starting "
            "full-width soft L1/Race return immediately: "
            f"vehicle_id={target_id}, longitudinal={longitudinal:.2f}m, "
            f"source={source}",
            throttle_duration_sec=1.0,
        )

    def _overtake_prediction_is_clear(self, target_id) -> bool:
        """Accept brake bypass only for a fresh, collision-free MPC prediction."""
        if (
            target_id is None
            or self._mpc.infeasibility_counter != 0
            or self._mpc.current_prediction is None
            or self._reference_path.target_lane_idx not in (0, 2)
            or not self._reference_path.is_overtaking
        ):
            return False
        target_buf = self._v2x_tracker._samples.get(target_id)
        if not target_buf:
            return False
        _, target_x, target_y = target_buf[-1]
        velocity_x, velocity_y = self._v2x_tracker.velocity(target_id)
        pred_x, pred_y = self._mpc.current_prediction
        if not pred_x or len(pred_x) != len(pred_y):
            return False
        minimum_clearance = (
            0.5 * float(self._cfg.bicycle_model.width)
            + self._v2x_vehicle_radius
        )
        prediction_times = []
        for index in range(len(pred_x)):
            time_index = min(index + 2, len(self._v2x_t_samples) - 1)
            prediction_times.append(self._v2x_t_samples[time_index])
        return prediction_clears_moving_vehicle(
            pred_x,
            pred_y,
            prediction_times,
            vehicle_x=target_x,
            vehicle_y=target_y,
            vehicle_vx=velocity_x,
            vehicle_vy=velocity_y,
            minimum_clearance=minimum_clearance,
        )

    def _capture_grounded_start_boost_layout(self) -> None:
        if (
            self._grounded_start_boost_eligible is not None
            or self._awsim_state not in ("Grounded", "Ready")
            or self._odom is None
            or not self._v2x_received_once
            or not hasattr(self, '_v2x_tracker')
        ):
            return

        pose = self.get_ego_pose()
        ego_lane_idx = self._lane_index_for_position(pose.x, pose.y)
        l0_vehicle_ids = []
        for vid in self._v2x_tracker.active_vehicle_ids():
            buf = self._v2x_tracker._samples.get(vid)
            if not buf:
                continue
            _, other_x, other_y = buf[-1]
            if self._lane_index_for_position(other_x, other_y) == 0:
                l0_vehicle_ids.append(vid)

        self._grounded_ego_lane_idx = ego_lane_idx
        self._grounded_l0_vehicle_ids = l0_vehicle_ids
        self._grounded_start_boost_capture_state = self._awsim_state
        self._grounded_start_boost_eligible = (
            ego_lane_idx == 0 and not l0_vehicle_ids)
        if (
            self._grounded_start_boost_eligible
            and self._initial_start_boost_enabled
            and not self._initial_start_boost_done
            and not self._initial_start_boost_armed
            and self._initial_start_boost_until is None
        ):
            # Grounded/Ready only records eligibility. Starting the timer or
            # toggling turbo here can waste the boost before the race moves.
            self._initial_start_boost_armed = True
            self._activate_initial_start_exclusive()
        self.get_logger().info(
            "[InitialStartBoostGroundedSnapshot] "
            f"eligible={self._grounded_start_boost_eligible}, "
            f"capture_state={self._grounded_start_boost_capture_state}, "
            f"ego_lane={ego_lane_idx}, "
            f"L0_vehicle_ids={l0_vehicle_ids}"
        )

    def _activate_initial_start_exclusive(self) -> None:
        """Give an eligible L0 start exclusive trajectory ownership."""
        if self._initial_start_exclusive_active:
            return

        self._initial_start_exclusive_active = True

        # Grid vehicles are stationary before Start and must not be treated as
        # ordinary stopped-overtake targets.  Drop every competing lateral
        # owner when the Grounded snapshot confirms an eligible L0 start.
        self._overtake.release_target()
        self._forced_overtake_vehicle_id = None
        self._outer_lane_released_vehicle_id = None
        self._prepass_fallback_lane_idx = None
        self._prepass_fallback_blocked = False
        self._prepass_fallback_follow_active = False
        self._prepass_follow_last_retry_at = None
        self._prepass_retry_after_reverse = False
        self._prepass_retry_lane_idx = None
        self._prepass_reverse_motion_started = False
        self._prepass_reverse_start_xy = None
        self._prepass_reverse_distance = 0.0
        self._prepass_fallback_recovery_active = False
        self._prepass_fallback_recovery_stable_since = None
        self._prepass_fallback_recovery_started_at = None
        self._prepass_target_behind_since = None
        self._prepass_failed_lane_idx = None
        self._prepass_fallback_commit_pending = False
        self._prepass_fallback_commit_lane_idx = None
        self._prepass_fallback_commit_success_since = None
        self._prepass_attempted_outer_lanes.clear()
        self._follow_latched_cache = None
        self._close_obstacle_reverse_requested = False
        self._reset_follow_escape()

        self._center_lane_rejoin_active = False
        self._center_lane_rejoin_constraint_released = False
        self._center_lane_rejoin_stable_since = None
        self._l1_probe_active = False
        self._l1_probe_context = None
        self._l1_probe_success_cycles = 0
        self._l1_probe_constraint_applied = False
        self._l1_safety_recovery_active = False
        self._l1_safety_recovery_stable_since = None
        self._l1_safety_recovery_context = None
        self._l1_safety_reprobe_pending = False
        self._reset_l1_rejoin_backoff()
        self._trajectory_clear_since = None
        self._trajectory_vehicle_id = None

        self.get_logger().info(
            "[InitialStartExclusive] Grounded snapshot confirmed an eligible "
            "L0 start; locking the Center trajectory while suppressing "
            "ordinary overtake, Prepass, and Race/Center selection until "
            "boost completion."
        )

    def _publish_initial_turbo(self) -> None:
        """Toggle AWSIM turbo once, matching teleop_manager's button command."""
        if (
            not self._initial_start_turbo_enabled
            or self._initial_start_turbo_published
        ):
            return
        turbo_on = Float32MultiArray()
        turbo_on.data = [1.0]
        self._awsim_turbo_pub.publish(turbo_on)
        turbo_release = Float32MultiArray()
        turbo_release.data = [0.0]
        self._awsim_turbo_pub.publish(turbo_release)
        self._initial_start_turbo_published = True
        self.get_logger().info(
            "[InitialStartTurbo] published AWSIM turbo toggle [1.0] -> [0.0] "
            "on /awsim/cmd."
        )

    def _request_awsim_control_mode_for_recovery(self) -> None:
        if not self._stuck_request_control_mode:
            return
        msg = Bool()
        msg.data = True
        self._awsim_control_mode_request_pub.publish(msg)
        self.get_logger().info(
            "[StuckRecovery] requested AWSIM control mode (data=True).",
            throttle_duration_sec=1.0,
        )

    def _begin_stuck_drive_transition(self, now_sec: float, reason: str) -> None:
        """Stop and request DRIVE until AWSIM reports that the shift completed."""
        self._stuck_reverse_drive_active = False
        self._stuck_pre_reverse_until = None
        self._stuck_pre_drive_until = None
        self._stuck_wait_for_drive = True
        if self._stuck_drive_transition_started_at is None:
            self._stuck_drive_transition_started_at = now_sec
        self._stuck_recovery_until = now_sec + self._stuck_max_shift_wait
        self._stuck_last_drive_request_at = None
        self.get_logger().info(
            f"[StuckRecovery] {reason}; starting DRIVE confirmation."
        )

    def _outer_lane_constraint_is_collapsed(self, lane_idx: int):
        """Detect an invalid/zero-width corridor in the latest outer solve."""
        lane_idx = int(lane_idx)
        if lane_idx not in (0, 2):
            return False, None
        if getattr(self._mpc, "_constraint_target_lane", None) != lane_idx:
            return False, None
        # MPC retains the first collapse across every relaxation retry in the
        # same solve. Prefer that snapshot so a later relaxed solution cannot
        # hide the original zero-width corridor.
        detail = getattr(self._mpc, "_constraint_collapse_detail", None)
        if detail is None:
            detail = collapsed_constraint_snapshot(
                getattr(self._mpc, "_prediction_upper_bounds", []),
                getattr(self._mpc, "_prediction_lower_bounds", []),
                getattr(self._mpc, "_constraint_wp_ids", []),
            )
        return detail is not None, detail

    def _physical_corridor_state(self, x: float, y: float):
        """Return the same guarded center corridor used by PP wall checks."""
        wp_id = self._car.get_closest_waypoint(x, y)
        wp = self._reference_path.get_waypoint(wp_id)
        normal_angle = wp.psi + math.pi / 2.0
        offset = (
            (x - wp.x) * math.cos(normal_angle)
            + (y - wp.y) * math.sin(normal_angle)
        )
        half_width = 0.5 * float(self._cfg.bicycle_model.width)
        guard = float(getattr(
            self._cfg.mpc, "prediction_outer_boundary_guard", 0.10))
        lower = float(wp.lb) + half_width + guard
        upper = float(wp.ub) - half_width - guard
        return wp_id, offset, lower, upper

    def _full_corridor_violation(self, x: float, y: float) -> float:
        """Return PP-equivalent guarded vehicle-center boundary violation."""
        _, offset, lower, upper = self._physical_corridor_state(x, y)
        return max(lower - offset, offset - upper, 0.0)

    def _straight_reentry_rollout(self, pose, direction: int):
        distances = np.linspace(
            0.0, self._straight_reentry_probe_distance, 11)
        violations = []
        for distance in distances:
            signed_distance = float(direction) * float(distance)
            violations.append(self._full_corridor_violation(
                pose.x + signed_distance * math.cos(pose.theta),
                pose.y + signed_distance * math.sin(pose.theta),
            ))
        return np.asarray(violations, dtype=float)

    def _select_straight_reentry_direction(self, pose) -> int:
        current_violation = self._full_corridor_violation(pose.x, pose.y)
        if not np.isfinite(current_violation) or current_violation <= 0.05:
            return 0
        candidates = []
        for direction in (1, -1):
            violations = self._straight_reentry_rollout(pose, direction)
            if not np.all(np.isfinite(violations)):
                continue
            improvement = current_violation - float(violations[-1])
            # Permit small waypoint-projection noise, but reject a direction
            # which first drives materially farther out of the corridor.
            if (
                improvement >= self._straight_reentry_min_improvement
                and float(np.max(violations)) <= current_violation + 0.10
            ):
                candidates.append((float(violations[-1]), direction))
        if not candidates:
            return 0
        for _, direction in sorted(candidates, key=lambda item: item[0]):
            if direction > 0:
                forward_blocked = any(
                    0.0 < float(longitudinal)
                    <= self._straight_reentry_probe_distance + 2.0
                    for _, _, longitudinal
                    in self._relative_lane_vehicle_samples(pose, 0.0)
                )
                if forward_blocked:
                    continue
            return direction
        return 0

    def _start_straight_reentry(self, pose, now_sec: float) -> bool:
        if not self._straight_reentry_enabled:
            return False
        direction = self._select_straight_reentry_direction(pose)
        if direction == 0:
            return False
        if direction < 0 and not self._reverse_rear_is_clear(
            pose, 0.0, reverse_distance=self._straight_reentry_probe_distance
        ):
            return False
        self._straight_reentry_active = True
        self._straight_reentry_direction = direction
        self._straight_reentry_started_at = now_sec
        self._straight_reentry_pre_reverse_until = (
            now_sec + self._stuck_pre_reverse_duration
            if direction < 0 and self._stuck_pre_reverse_duration > 0.0
            else None
        )
        self._straight_reentry_returning_drive = False
        self._straight_reentry_success_cycles = 0
        self._straight_reentry_last_violation = self._full_corridor_violation(
            pose.x, pose.y)
        self._straight_reentry_localization_wait_started_at = now_sec
        self._straight_reentry_allow_boundary_step_increase = False
        self._last_stuck_gear_command = None
        self._stuck_last_drive_request_at = None
        self._stuck_last_reverse_request_at = None
        self.get_logger().warn(
            "[StraightReentryStart] MPC-independent straight recovery "
            f"selected direction={'DRIVE' if direction > 0 else 'REVERSE'}, "
            f"violation={self._straight_reentry_last_violation:.2f}m, "
            f"probe={self._straight_reentry_probe_distance:.2f}m"
        )
        return True

    def _apply_straight_reentry(self, now, pose, u) -> bool:
        now_sec = float(now.nanoseconds) / 1e9
        u[0] = 0.0
        u[1] = 0.0
        self._request_awsim_control_mode_for_recovery()

        if self._straight_reentry_returning_drive:
            drive_request_due = (
                self._stuck_last_drive_request_at is None
                or now_sec - self._stuck_last_drive_request_at
                    >= self._stuck_drive_request_resend_sec
            )
            if drive_request_due:
                self._publish_gear_command(
                    now, self._gear_drive_command, force=True)
                self._stuck_last_drive_request_at = now_sec
            if self._current_gear_is_drive():
                self._straight_reentry_active = False
                self._straight_reentry_returning_drive = False
                self._straight_reentry_direction = 0
                self._straight_reentry_localization_wait_started_at = None
                self._stuck_since = None
                self._gnss_history = []
                self._post_reverse_full_width_recovery_active = True
                self._mpc_safety_recovery_active = True
                self.get_logger().info(
                    "[StraightReentryComplete] DRIVE confirmed; returning "
                    "to full-width MPC recovery."
                )
            return True

        violation = self._full_corridor_violation(pose.x, pose.y)
        fresh_mpc = bool(
            self._mpc.infeasibility_counter == 0
            and self._mpc.current_prediction is not None
            and not self._mpc.used_prediction_fallback
            and not self._mpc.recovery_requested
        )
        inside_and_stable = violation <= 0.02 and fresh_mpc
        self._straight_reentry_success_cycles = (
            self._straight_reentry_success_cycles + 1
            if inside_and_stable else 0
        )
        timed_out = (
            self._straight_reentry_started_at is not None
            and now_sec - self._straight_reentry_started_at
                >= self._straight_reentry_timeout
        )
        localization_wait_elapsed = (
            now_sec - self._straight_reentry_localization_wait_started_at
            if self._straight_reentry_localization_wait_started_at is not None
            else 0.0
        )
        localization_waiting = bool(
            not self._localization_consistent
            and localization_wait_elapsed
                < self._straight_reentry_localization_wait_sec
        )
        localization_failed = bool(
            not self._localization_consistent
            and not localization_waiting
        )
        if localization_waiting:
            self.get_logger().info(
                "[StraightReentryLocalizationWait] holding zero speed until "
                "GNSS/odom agreement is confirmed: "
                f"elapsed={localization_wait_elapsed:.2f}s/"
                f"{self._straight_reentry_localization_wait_sec:.2f}s, "
                f"position_error={self._localization_position_error:.2f}m",
                throttle_duration_sec=0.25,
            )
            return True
        moving_wrong_way = bool(
            not self._straight_reentry_allow_boundary_step_increase
            and
            self._straight_reentry_last_violation is not None
            and violation > self._straight_reentry_last_violation + 0.15
        )
        completed = self._straight_reentry_success_cycles >= (
            self._straight_reentry_success_cycles_required)
        if timed_out or localization_failed or moving_wrong_way or completed:
            reason = (
                "corridor_and_mpc_recovered" if completed
                else "localization_inconsistent" if localization_failed
                else "corridor_violation_increased" if moving_wrong_way
                else "timeout"
            )
            self.get_logger().warn(
                "[StraightReentryStop] stopping continuous recovery: "
                f"reason={reason}, violation={violation:.2f}m"
            )
            if self._straight_reentry_direction < 0:
                self._straight_reentry_returning_drive = True
            else:
                self._straight_reentry_active = False
                self._straight_reentry_direction = 0
                self._straight_reentry_allow_boundary_step_increase = False
                self._straight_reentry_localization_wait_started_at = None
                self._stuck_since = None
                self._gnss_history = []
            return True

        self._straight_reentry_last_violation = min(
            float(self._straight_reentry_last_violation), violation)
        if self._straight_reentry_direction > 0:
            drive_request_due = (
                self._stuck_last_drive_request_at is None
                or now_sec - self._stuck_last_drive_request_at
                    >= self._stuck_drive_request_resend_sec
            )
            if drive_request_due:
                self._publish_gear_command(
                    now, self._gear_drive_command, force=True)
                self._stuck_last_drive_request_at = now_sec
            if self._current_gear_is_drive():
                u[0] = self._straight_reentry_speed
        else:
            if (
                self._straight_reentry_pre_reverse_until is not None
                and now_sec < self._straight_reentry_pre_reverse_until
            ):
                self._publish_gear_command(now, self._gear_pre_reverse_command)
            else:
                self._straight_reentry_pre_reverse_until = None
                reverse_request_due = (
                    self._stuck_last_reverse_request_at is None
                    or now_sec - self._stuck_last_reverse_request_at
                        >= self._stuck_reverse_request_resend_sec
                )
                if reverse_request_due:
                    self._publish_gear_command(
                        now, self._gear_reverse_command, force=True)
                    self._stuck_last_reverse_request_at = now_sec
                if self._current_gear_is_reverse():
                    u[0] = self._straight_reentry_speed
        self.get_logger().info(
            "[StraightReentry] continuous straight recovery active: "
            f"direction={'DRIVE' if self._straight_reentry_direction > 0 else 'REVERSE'}, "
            f"violation={violation:.2f}m, speed={float(u[0]):.2f}m/s",
            throttle_duration_sec=0.5,
        )
        return True

    def _apply_stuck_recovery(self, now, u, actual_speed: float, pose) -> bool:
        if not self._stuck_recovery_enabled:
            return False

        now_sec = float(now.nanoseconds) / 1e9

        if self._straight_reentry_active:
            return self._apply_straight_reentry(now, pose, u)

        # Grid waiting is intentional.  Never convert a Grounded/Ready stop or
        # a stale close-obstacle request into a reverse manoeuvre.
        if (
            should_reset_motion_latch(self._awsim_state)
            and self._stuck_recovery_until is None
        ):
            self._stuck_since = None
            self._gnss_history = []
            self._close_obstacle_reverse_requested = False
            return False

        # A reverse request may wait in the normal stuck timer for several
        # seconds. Keep checking the rear corridor during that wait as well.
        if (
            self._prepass_retry_after_reverse
            and self._stuck_recovery_until is None
            and not self._reverse_rear_is_clear(
                pose, actual_speed,
                reverse_distance=self._stuck_reverse_target_distance)
        ):
            self._switch_prepass_to_follow(
                "rear corridor became occupied while waiting to start reverse: "
                f"conflicts={self._reverse_rear_conflict_summary()}"
            )
            self._stuck_since = None
            return False

        if self._stuck_recovery_until is not None:
            in_drive_transition = (
                self._stuck_pre_drive_until is not None
                or self._stuck_wait_for_drive
            )
            # Measure signed reverse progress for every recovery type.  The
            # old bookkeeping was limited to stopped-lead adaptive reverse,
            # leaving generic MPC-stall recovery unable to reject an
            # R->D transition after only a few solver cycles.
            if (
                self._stuck_reverse_drive_active
                and self._prepass_reverse_start_xy is not None
            ):
                reverse_arc_delta = self._center_longitudinal_between(
                    self._prepass_reverse_start_xy[0],
                    self._prepass_reverse_start_xy[1],
                    pose.x,
                    pose.y,
                )
                if reverse_arc_delta is not None:
                    self._prepass_reverse_distance = max(
                        0.0, -reverse_arc_delta)
            # Both distance-controlled and ordinary timed reverse must stop
            # as soon as a fresh executable forward MPC path is stable. The
            # former implementation evaluated this only in adaptive mode, so
            # generic MPC-stall recovery always reversed for the full timeout.
            if (
                self._stuck_reverse_drive_active
                and not in_drive_transition
            ):
                forward_path_valid = self._reverse_forward_path_is_valid(
                    pose, u)
                self._adaptive_reverse_forward_success_cycles = (
                    self._adaptive_reverse_forward_success_cycles + 1
                    if forward_path_valid else 0
                )
                if self._adaptive_reverse_forward_success_cycles >= (
                    self._adaptive_reverse_forward_success_cycles_required
                ):
                    remaining_minimum = max(
                        self._generic_reverse_min_distance
                        - self._prepass_reverse_distance,
                        0.0,
                    )
                    clearance_probe = (
                        remaining_minimum
                        + self._adaptive_reverse_wall_margin
                    )
                    static_clearance = self._static_reverse_clearance(
                        pose, clearance_probe)
                    boundary_has_remaining_clearance = bool(
                        static_clearance + 0.05 >= clearance_probe
                    )
                    hold_for_minimum = (
                        should_hold_generic_reverse_for_minimum_distance(
                            front_vehicle_origin=
                                self._prepass_retry_after_reverse,
                            travelled_distance=
                                self._prepass_reverse_distance,
                            minimum_distance=
                                self._generic_reverse_min_distance,
                            localization_consistent=
                                self._localization_consistent,
                            boundary_has_remaining_clearance=
                                boundary_has_remaining_clearance,
                        )
                    )
                    if hold_for_minimum:
                        self._adaptive_reverse_forward_success_cycles = (
                            self._adaptive_reverse_forward_success_cycles_required
                        )
                        self.get_logger().warn(
                            "[GenericReverseMinimumHold] forward MPC is ready "
                            "but generic stall recovery has not moved the "
                            "minimum reverse distance: "
                            f"travelled={self._prepass_reverse_distance:.2f}m/"
                            f"{self._generic_reverse_min_distance:.2f}m, "
                            f"remaining={remaining_minimum:.2f}m, "
                            f"static_clearance={static_clearance:.2f}m",
                            throttle_duration_sec=0.5,
                        )
                    else:
                        u[0] = 0.0
                        u[1] = 0.0
                        current_violation = self._full_corridor_violation(
                            pose.x, pose.y)
                        self._post_reverse_straight_reentry_pending = bool(
                            self._straight_reentry_enabled
                            and current_violation > 0.02
                        )
                        exception_reason = "minimum_reached"
                        if not self._prepass_retry_after_reverse:
                            if not self._localization_consistent:
                                exception_reason = "localization_inconsistent"
                            elif not boundary_has_remaining_clearance:
                                exception_reason = "boundary_clearance_insufficient"
                        self.get_logger().info(
                            "[ReverseForwardReady] fresh collision-free "
                            "forward MPC path confirmed; ending reverse "
                            "early: "
                            f"success_cycles="
                            f"{self._adaptive_reverse_forward_success_cycles}, "
                            f"adaptive={self._adaptive_reverse_active}, "
                            f"travelled={self._prepass_reverse_distance:.2f}m, "
                            f"minimum_gate={exception_reason}, "
                            f"corridor_violation={current_violation:.2f}m, "
                            "forward_reentry_after_drive="
                            f"{self._post_reverse_straight_reentry_pending}"
                        )
                        self._begin_stuck_drive_transition(
                            now_sec, "valid forward MPC path confirmed")
                        in_drive_transition = True
            if (
                self._adaptive_reverse_active
                and not in_drive_transition
                and self._stuck_recovery_started_at is not None
                and self._prepass_reverse_start_xy is not None
            ):
                reverse_arc_delta = self._center_longitudinal_between(
                    self._prepass_reverse_start_xy[0],
                    self._prepass_reverse_start_xy[1],
                    pose.x,
                    pose.y,
                )
                if reverse_arc_delta is not None:
                    self._prepass_reverse_distance = max(
                        0.0, -reverse_arc_delta)
                target_distance = float(
                    self._stuck_reverse_target_distance or 0.0)
                remaining = max(
                    target_distance - self._prepass_reverse_distance, 0.0)
                localization_lost = not self._localization_consistent
                static_remaining = self._static_reverse_clearance(
                    pose, remaining + self._adaptive_reverse_wall_margin)
                wall_became_unsafe = (
                    remaining > 0.05
                    and static_remaining - self._adaptive_reverse_wall_margin
                        + 0.05 < remaining
                )
                if localization_lost or wall_became_unsafe:
                    u[0] = 0.0
                    u[1] = 0.0
                    reason = (
                        "localization_lost"
                        if localization_lost else "static_path_blocked"
                    )
                    self.get_logger().warn(
                        "[AdaptiveReverseAbort] stopping distance-controlled "
                        "reverse and returning to DRIVE: "
                        f"reason={reason}, travelled="
                        f"{self._prepass_reverse_distance:.2f}m/"
                        f"{target_distance:.2f}m, remaining={remaining:.2f}m, "
                        f"static_remaining={static_remaining:.2f}m, "
                        f"localization_error={self._localization_position_error:.2f}m"
                    )
                    self._begin_stuck_drive_transition(
                        now_sec, "adaptive reverse safety condition failed")
                    in_drive_transition = True
                elif self._prepass_reverse_distance >= max(
                    target_distance - 0.05, 0.0
                ):
                    u[0] = 0.0
                    u[1] = 0.0
                    self.get_logger().info(
                        "[AdaptiveReverseDistanceLimit] no forward MPC was "
                        "confirmed before the safe distance limit: "
                        f"travelled={self._prepass_reverse_distance:.2f}m, "
                        f"limit={target_distance:.2f}m"
                    )
                    self._begin_stuck_drive_transition(
                        now_sec, "adaptive reverse target reached")
                    in_drive_transition = True
            if not in_drive_transition:
                forward_resume_ready, forward_resume_reason = (
                    self._reverse_forward_resume_ready(
                        pose, now_sec, u, actual_speed)
                )
                if forward_resume_ready:
                    u[0] = 0.0
                    u[1] = 0.0
                    self._switch_prepass_to_follow(
                        "front path became available during reverse; "
                        "cancelling reverse and following the latched target"
                    )
                    self._begin_stuck_drive_transition(
                        now_sec,
                        "cancelling REVERSE for forward recovery: "
                        f"{forward_resume_reason}",
                    )
                    in_drive_transition = True
            if (
                self._prepass_retry_after_reverse
                and not in_drive_transition
                and not self._reverse_rear_is_clear(
                    pose,
                    actual_speed,
                    reverse_distance=(
                        max(
                            float(self._stuck_reverse_target_distance)
                            - self._prepass_reverse_distance,
                            0.0,
                        )
                        if self._stuck_reverse_target_distance is not None
                        else None
                    ),
                )
            ):
                u[0] = 0.0
                u[1] = 0.0
                self._switch_prepass_to_follow(
                    "rear corridor became occupied before/during reverse; "
                    "aborting reverse and following the latched target: "
                    f"conflicts={self._reverse_rear_conflict_summary()}"
                )
                self._begin_stuck_drive_transition(
                    now_sec,
                    "rear corridor became occupied before/during reverse: "
                    f"conflicts={self._reverse_rear_conflict_summary()}",
                )

            # 1. タイムアウト判定
            if now_sec >= self._stuck_recovery_until:
                if self._stuck_reverse_drive_active:
                    # 後退駆動が終わったので、前進復帰シーケンスを開始する
                    self._begin_stuck_drive_transition(
                        now_sec, "REVERSE drive finished"
                    )
                elif in_drive_transition:
                    # Never resume normal control only because the nominal
                    # shift wait expired. AWSIM may still be in REVERSE.
                    self._stuck_recovery_until = (
                        now_sec + self._stuck_max_shift_wait)
                    self.get_logger().warn(
                        "[StuckRecovery] DRIVE confirmation timed out; "
                        "holding zero speed and retrying DRIVE request.",
                        throttle_duration_sec=1.0,
                    )
                else:
                    # A REVERSE request may have been accepted even if its
                    # report was delayed. Always leave through DRIVE confirm.
                    self._begin_stuck_drive_transition(
                        now_sec,
                        "REVERSE confirmation timed out before drive began",
                    )

            # 2. リカバリー動作中の処理
            if self._stuck_recovery_until is not None:
                # 前進復帰シーケンス
                in_drive_transition = (
                    self._stuck_pre_drive_until is not None
                    or self._stuck_wait_for_drive
                )

                if in_drive_transition:
                    self._request_awsim_control_mode_for_recovery()
                    pre_driving = (
                        self._stuck_pre_drive_until is not None
                        and now_sec < self._stuck_pre_drive_until
                    )
                    gear_report_age = (
                        now_sec - self._last_gear_report_received_sec
                        if self._last_gear_report_received_sec is not None
                        else math.inf
                    )
                    gear_status_known = bool(
                        self._gear_report is not None
                        and gear_report_age <= max(
                            2.0, 4.0 * self._stuck_drive_request_resend_sec)
                    )
                    reported_drive = (
                        gear_status_known
                        and getattr(self._gear_report, 'report', None)
                            == self._gear_drive_command
                    )
                    drive_wait_elapsed = (
                        now_sec - self._stuck_drive_transition_started_at
                        if self._stuck_drive_transition_started_at is not None
                        else 0.0
                    )
                    missing_report_drive_fallback = bool(
                        not gear_status_known
                        and drive_wait_elapsed >= self._stuck_max_shift_wait
                        and abs(actual_speed) <= 0.1
                    )
                    gear_is_drive = bool(
                        reported_drive
                        or not self._stuck_wait_for_reverse_gear
                        or missing_report_drive_fallback
                    )
                    if missing_report_drive_fallback:
                        self._drive_confirmed_by_command_fallback = True
                        self.get_logger().error(
                            "[StuckRecoveryGearFallback] GearReport remained "
                            "unavailable after repeated DRIVE requests; vehicle "
                            "is stopped, continuing with commanded DRIVE state.",
                            throttle_duration_sec=1.0,
                        )

                    if pre_driving:
                        self._publish_gear_command(now, self._gear_reverse_command)
                    else:
                        self._stuck_pre_drive_until = None
                        self._stuck_wait_for_drive = True
                        drive_request_due = (
                            self._stuck_last_drive_request_at is None
                            or now_sec - self._stuck_last_drive_request_at
                                >= self._stuck_drive_request_resend_sec
                        )
                        if drive_request_due:
                            self._publish_gear_command(
                                now, self._gear_drive_command, force=True)
                            self._stuck_last_drive_request_at = now_sec

                    waiting_for_drive = (
                        pre_driving or not gear_is_drive
                    )

                    if waiting_for_drive:
                        # A stale forward/reverse MPC command must never leak
                        # through while AWSIM is still changing back to DRIVE.
                        u[0] = 0.0
                        u[1] = 0.0
                        if gear_status_known:
                            self.get_logger().warn(
                                "[StuckRecovery] pre-shift before drive (stopping)..."
                                if pre_driving
                                else "[StuckRecovery] waiting for AWSIM gear to become DRIVE "
                                f"(current={getattr(self._gear_report, 'report', None)}).",
                                throttle_duration_sec=1.0,
                            )
                        else:
                            self.get_logger().warn(
                                "[StuckRecovery] waiting for AWSIM GearReport "
                                "to confirm DRIVE; holding zero speed.",
                                throttle_duration_sec=1.0,
                            )
                        return True
                    else:
                        self._stuck_recovery_until = None  # シフト完了につき正常終了へ

                # 後退（REVERSE）リカバリーシーケンス
                else:
                    gear_is_reverse = self._current_gear_is_reverse()
                    if (
                        gear_is_reverse
                        and self._stuck_recovery_started_at is None
                    ):
                        self._stuck_recovery_started_at = now_sec
                        self._prepass_reverse_start_xy = (pose.x, pose.y)
                        self._stuck_reverse_start_heading = pose.theta
                        if self._stuck_reverse_target_distance is not None:
                            nominal_travel_sec = (
                                self._stuck_reverse_target_distance
                                / max(self._stuck_forward_reverse_speed, 0.1)
                            )
                            reverse_timeout = max(
                                self._stuck_reverse_duration,
                                nominal_travel_sec + 2.0,
                            )
                        else:
                            reverse_timeout = self._stuck_reverse_duration
                        self._stuck_recovery_until = now_sec + reverse_timeout
                        self.get_logger().info(
                            "[StuckRecovery] AWSIM gear is REVERSE; starting reverse drive "
                            f"with timeout={reverse_timeout:.1f}s, "
                            f"target_distance="
                            f"{self._stuck_reverse_target_distance if self._stuck_reverse_target_distance is not None else 'timed'}."
                        )

                    self._request_awsim_control_mode_for_recovery()
                    pre_shifting = (
                        self._stuck_pre_reverse_until is not None
                        and now_sec < self._stuck_pre_reverse_until
                    )
                    if pre_shifting:
                        self._publish_gear_command(now, self._gear_pre_reverse_command)
                    else:
                        self._stuck_pre_reverse_until = None
                        reverse_request_due = (
                            self._stuck_last_reverse_request_at is None
                            or now_sec - self._stuck_last_reverse_request_at
                                >= self._stuck_reverse_request_resend_sec
                        )
                        if reverse_request_due:
                            self._publish_gear_command(
                                now, self._gear_reverse_command, force=True)
                            self._stuck_last_reverse_request_at = now_sec
                    waiting_for_reverse = (
                        pre_shifting
                        or (
                            self._stuck_wait_for_reverse_gear
                            and not gear_is_reverse
                        )
                    )
                    if waiting_for_reverse:
                        # Never send a positive speed while AWSIM may still be
                        # in DRIVE. Hold completely still until REVERSE is
                        # confirmed (or the shift wait times out).
                        u[0] = 0.0
                        u[1] = 0.0
                        self._stuck_reverse_drive_active = False
                        if self._gear_report is None:
                            self.get_logger().warn(
                                "[StuckRecovery] waiting for AWSIM GearReport "
                                "to confirm REVERSE; holding zero speed.",
                                throttle_duration_sec=1.0,
                            )
                    else:
                        self._apply_stuck_reverse_command(u)
                        self._publish_stuck_actuation_command(now, self._stuck_actuation_accel_cmd, 0.0, u[1])
                        self._stuck_reverse_drive_active = True
                        if self._prepass_retry_after_reverse:
                            if self._prepass_reverse_start_xy is None:
                                self._prepass_reverse_start_xy = (pose.x, pose.y)
                            if self._stuck_reverse_start_heading is None:
                                self._stuck_reverse_start_heading = pose.theta
                            reverse_arc_delta = self._center_longitudinal_between(
                                self._prepass_reverse_start_xy[0],
                                self._prepass_reverse_start_xy[1],
                                pose.x,
                                pose.y,
                            )
                            if reverse_arc_delta is not None:
                                self._prepass_reverse_distance = max(
                                    0.0, -reverse_arc_delta)
                            if self._prepass_reverse_distance >= max(
                                0.1, self._stuck_gnss_distance_threshold
                            ):
                                self._prepass_reverse_motion_started = True
                    return True

            # 3. 正常終了・タイムアウト終了後のリセット処理
            reverse_drive_completed = (
                self._stuck_recovery_started_at is not None
            )
            self._stuck_recovery_until = None
            self._stuck_cooldown_until = now_sec + self._stuck_cooldown
            self._stuck_since = None
            self._stuck_reverse_start_target_longitudinal = None
            self._stuck_reverse_target_distance = None
            self._stuck_reverse_start_heading = None
            self._adaptive_reverse_active = False
            self._adaptive_reverse_static_clearance = None
            self._adaptive_reverse_forward_success_cycles = 0
            self._gnss_history = []
            self._last_stuck_gear_command = None
            self._stuck_last_drive_request_at = None
            self._stuck_last_reverse_request_at = None
            self._stuck_reverse_drive_after = None
            self._stuck_reverse_drive_active = False
            self._stuck_recovery_started_at = None
            self._stuck_pre_reverse_until = None
            self._stuck_pre_drive_until = None
            self._stuck_wait_for_drive = False
            self._stuck_drive_transition_started_at = None
            self._publish_gear_command(now, self._gear_drive_command)
            if reverse_drive_completed:
                # DRIVE confirmation is only the end of the gear transition.
                # Keep the next MPC problems full-width until a fresh
                # prediction succeeds continuously for the normal safety
                # recovery duration.
                self._post_reverse_full_width_recovery_active = True
                self._mpc_safety_recovery_active = True
                self._mpc_safety_recovery_success_cycles = 0
                self._stuck_since = None
                self._gnss_history = []
                self.get_logger().info(
                    "[PostReverseFullWidthRecovery] DRIVE confirmed; holding "
                    "full width until fresh MPC predictions recover "
                    f"continuously for {self._mpc_safety_recovery_success_sec:.2f}s."
                )
                if self._post_reverse_straight_reentry_pending:
                    self._straight_reentry_active = True
                    self._straight_reentry_direction = 1
                    self._straight_reentry_started_at = now_sec
                    self._straight_reentry_pre_reverse_until = None
                    self._straight_reentry_returning_drive = False
                    self._straight_reentry_success_cycles = 0
                    self._straight_reentry_localization_wait_started_at = now_sec
                    self._straight_reentry_last_violation = (
                        self._full_corridor_violation(pose.x, pose.y)
                    )
                    # Consecutive collision-free forward MPC checks already
                    # authorized DRIVE. A waypoint-boundary discontinuity may
                    # temporarily increase the numeric violation, so let
                    # corridor entry or timeout terminate this forward creep.
                    self._straight_reentry_allow_boundary_step_increase = True
                    self._post_reverse_straight_reentry_pending = False
                    self.get_logger().warn(
                        "[PostReverseStraightReentry] transient forward MPC "
                        "was confirmed while still outside the full corridor; "
                        "continuing low-speed DRIVE until corridor and MPC "
                        "recover: violation="
                        f"{self._straight_reentry_last_violation:.2f}m, "
                        f"speed={self._straight_reentry_speed:.2f}m/s"
                    )
            if self._prepass_retry_after_reverse:
                requested_lane_idx = self._prepass_retry_lane_idx
                reverse_succeeded = self._prepass_reverse_motion_started
                self._prepass_retry_after_reverse = False
                self._prepass_retry_lane_idx = None
                if not reverse_succeeded:
                    self._switch_prepass_to_follow(
                        "reverse did not move the vehicle"
                    )
                else:
                    # Do not choose L0/L2 here.  A fresh conflict/passability
                    # check is deferred until post-reverse full-width MPC has
                    # recovered continuously.
                    self._prepass_fallback_blocked = False
                    self._prepass_fallback_follow_active = False
                    self._prepass_fallback_recovery_active = True
                    self._prepass_fallback_recovery_stable_since = None
                    self._prepass_fallback_recovery_started_at = None
                    self._prepass_target_behind_since = None
                    self._prepass_fallback_lane_idx = None
                    self._prepass_failed_lane_idx = (
                        2 if requested_lane_idx == 0 else 0
                        if requested_lane_idx == 2 else None
                    )
                    self._prepass_fallback_commit_pending = False
                    self._prepass_fallback_commit_lane_idx = None
                    self._prepass_fallback_commit_success_since = None
                    self._mpc.osqp_initialized = False
                    self.get_logger().warn(
                        "[PrepassLaneFallbackRetryDeferred] reverse motion "
                        f"confirmed ({self._prepass_reverse_distance:.2f}m); "
                        "deferring lane selection until post-reverse "
                        "full-width MPC recovery completes."
                    )
                self._prepass_reverse_motion_started = False
                self._prepass_reverse_start_xy = None
                self._prepass_reverse_distance = 0.0
            # 自動運転モードを明示的にONにする
            if self._stuck_request_control_mode:
                msg = Bool()
                msg.data = True
                self._awsim_control_mode_request_pub.publish(msg)
            return False

        # 4. 通常時の判定（誤爆防止マスク＆スタック検知）
        in_cooldown = (
            self._stuck_cooldown_until is not None
            and now_sec < self._stuck_cooldown_until
        )
        if in_cooldown:
            return False

        # Once motion has been observed, keep the latch across lap wrap-around
        # and ordinary stops. Only Grounded/Ready may clear it in the AWSIM
        # state callback.
        self._has_moved_once = update_motion_latch(
            self._has_moved_once,
            actual_speed,
        )

        if not self._has_moved_once:
            self._stuck_since = None
            return False

        if self._intentional_follow_stop_active:
            # Stopping at the configured following gap is commanded behavior,
            # not a vehicle stall. Do not turn that stop into reverse recovery.
            self._stuck_since = None
            self._gnss_history = []
            return False

        # GNSSによる位置変化のチェック
        gnss_is_stuck = False
        gnss_moved_dist = None
        if self._gnss_pose is not None:
            self._gnss_history.append((now_sec, self._gnss_pose.pose.pose.position.x, self._gnss_pose.pose.pose.position.y))
        
        # 不要になった古い履歴を削除
        cutoff = now_sec - (self._stuck_time_threshold + 1.0)
        self._gnss_history = [item for item in self._gnss_history if item[0] >= cutoff]

        if len(self._gnss_history) > 1:
            target_t = now_sec - self._stuck_time_threshold
            ref_item = min(self._gnss_history, key=lambda item: abs(item[0] - target_t))
            if now_sec - ref_item[0] >= self._stuck_time_threshold * 0.8:
                curr_item = self._gnss_history[-1]
                gnss_moved_dist = math.hypot(curr_item[1] - ref_item[1], curr_item[2] - ref_item[2])
                if gnss_moved_dist < self._stuck_gnss_distance_threshold:
                    gnss_is_stuck = True

        # スピードが遅い、またはGNSS位置に変化がない場合をスタック状態とする
        is_stuck_state = (
            abs(actual_speed) < self._stuck_speed_threshold
            or gnss_is_stuck
        )

        should_recover_from_close_obstacle = (
            self._close_obstacle_reverse_requested and is_stuck_state
        )
        # MPC safety recovery intentionally sets u[0] to zero when no valid
        # prediction exists. In that state, use low actual speed plus GNSS
        # immobility instead of requiring a positive forward command.
        has_fresh_valid_prediction = (
            self._mpc.infeasibility_counter == 0
            and self._mpc.current_prediction is not None
            and not self._mpc.used_prediction_fallback
            and not self._mpc.recovery_requested
        )
        if (
            self._mpc_safety_recovery_active
            and has_fresh_valid_prediction
        ):
            # A newly solved prediction takes priority over every stall path,
            # including the generic positive-command detector.  Restart the
            # observation window if MPC becomes invalid again later.
            self._stuck_since = None
            self._gnss_history = []
            return False
        recover_from_mpc_stall = should_recover_from_mpc_stall(
            safety_recovery_active=self._mpc_safety_recovery_active,
            actual_speed=actual_speed,
            stall_speed_threshold=self._mpc_stall_speed_threshold,
            gnss_is_stuck=gnss_is_stuck,
            infeasibility_counter=self._mpc.infeasibility_counter,
            has_fresh_valid_prediction=has_fresh_valid_prediction,
        )
        if is_stuck_state and (
            u[0] > self._stuck_forward_cmd_threshold
            or should_recover_from_close_obstacle
            or recover_from_mpc_stall
        ):
            if self._stuck_since is None:
                # gnss_is_stuck already covers the observation period, so an
                # MPC stall must not wait for the same duration a second time.
                self._stuck_since = (
                    now_sec - self._stuck_time_threshold
                    if recover_from_mpc_stall
                    else now_sec
                )
            elif now_sec - self._stuck_since >= self._stuck_time_threshold:
                if (
                    recover_from_mpc_stall
                    and not self._prepass_retry_after_reverse
                    and self._start_straight_reentry(pose, now_sec)
                ):
                    return self._apply_straight_reentry(now, pose, u)
                if recover_from_mpc_stall:
                    self.get_logger().warn(
                        "[MPCStallRecovery] starting reverse after GNSS "
                        f"movement stayed below "
                        f"{self._stuck_gnss_distance_threshold:.2f}m for "
                        f"{self._stuck_time_threshold:.1f}s at "
                        f"speed={abs(actual_speed):.2f}m/s; "
                        f"mpc_infeasible={self._mpc.infeasibility_counter}."
                    )
                if not self._prepass_retry_after_reverse:
                    self._stuck_reverse_target_distance = None
                    self._stuck_reverse_start_heading = None
                    self._adaptive_reverse_active = False
                    self._adaptive_reverse_static_clearance = None
                    self._adaptive_reverse_forward_success_cycles = 0
                self._stuck_recovery_until = now_sec + self._stuck_max_shift_wait
                self._stuck_reverse_drive_after = now_sec + self._stuck_gear_shift_delay
                self._stuck_pre_reverse_until = (
                    now_sec + self._stuck_pre_reverse_duration
                    if self._stuck_pre_reverse_duration > 0.0
                    else None
                )
                self._stuck_recovery_started_at = None
                self._stuck_drive_transition_started_at = None
                self._drive_confirmed_by_command_fallback = False
                target = self._latched_follow_target_state(pose, now_sec)
                self._stuck_reverse_start_target_longitudinal = (
                    float(target["longitudinal"])
                    if target is not None and target["longitudinal"] > 0.0
                    else None
                )
                self._last_stuck_gear_command = None
                self._stuck_last_drive_request_at = None
                self._stuck_last_reverse_request_at = None
                self._request_awsim_control_mode_for_recovery()
                if self._stuck_pre_reverse_until is not None:
                    self._publish_gear_command(now, self._gear_pre_reverse_command)
                else:
                    self._publish_gear_command(now, self._gear_reverse_command)
                    self._stuck_last_reverse_request_at = now_sec
                u[0] = 0.0
                u[1] = 0.0
                self._stuck_reverse_drive_active = False
                self._close_obstacle_reverse_requested = False
                self._reset_follow_escape(
                    "reverse recovery sequence started"
                )
                
                return True
        else:
            self._stuck_since = None

        return False


    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg
        self._last_odom_received_sec = (
            float(self.get_clock().now().nanoseconds) / 1e9)

    def _gnss_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._gnss_pose = msg
        self._last_gnss_received_sec = (
            float(self.get_clock().now().nanoseconds) / 1e9)

    def _center_frenet(self, x: float, y: float):
        """Return Center arc position and signed lateral offset."""
        return project_to_closed_path_frenet(
            x,
            y,
            self._center_arc_points,
            self._center_arc_cumulative,
            self._center_arc_total_length,
        )

    def _center_longitudinal_between(
        self, ego_x: float, ego_y: float, other_x: float, other_y: float
    ):
        """Return signed other-minus-ego distance along the Center arc."""
        ego_frenet = self._center_frenet(ego_x, ego_y)
        other_frenet = self._center_frenet(other_x, other_y)
        if ego_frenet is None or other_frenet is None:
            return None
        return signed_closed_path_arc_distance(
            ego_frenet[0], other_frenet[0], self._center_arc_total_length)

    def _center_arc_as_waypoint_delta(self, arc_delta):
        """Convert Center arc metres to legacy waypoint-count thresholds."""
        if arc_delta is None or self._center_arc_mean_wp_spacing <= 1e-6:
            return None
        return float(arc_delta) / self._center_arc_mean_wp_spacing

    def _center_lane_index_for_offset(
        self, x: float, y: float, lateral_offset: float
    ):
        """Classify a Center-Frenet offset using Center lane boundaries."""
        wp_id = self._carN_center.get_closest_waypoint(x, y)
        for lane_idx, (ub_lane, lb_lane) in enumerate(
            self._reference_pathN_center.get_lane_bounds(wp_id)
        ):
            if lb_lane <= lateral_offset <= ub_lane:
                return lane_idx
        return None

    def _control_mode_request_callback(self, msg):
        if msg.data and not self._enable_control:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _path_constraints_callback(self, msg: PathConstraints):
        self._reference_path.set_path_constraints(
            msg.upper_bounds, msg.lower_bounds, msg.rows, msg.cols)

    def _measured_body_pose_callback(self, vehicle_id, msg):
        q = msg.pose.orientation
        norm = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
        if abs(norm-1.0) > 0.01:
            return
        self._v2x_input_tracker.set_measured_body_pose(vehicle_id,
            msg.pose.position.x,msg.pose.position.y,yaw_from_quaternion(q),
            msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,msg.header.frame_id)

    def _publish_collision_bodies(self, pose):
        markers = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        ego = collision.ego_body(self,pose)
        diagnostic_lines = []
        for i,vid in enumerate(self._v2x_tracker.active_vehicle_ids()):
            target = collision.target_body(self,vid)
            if target is None:
                continue
            overlap = collision.overlaps(ego,target,collision.geometry(self))
            for j,body in enumerate((ego,target)):
                marker = Marker(); marker.header.frame_id='map'
                marker.header.stamp=self.get_clock().now().to_msg()
                marker.ns=f'collision/{vid}';marker.id=j;marker.type=Marker.LINE_STRIP
                marker.action=Marker.ADD;marker.pose.orientation.w=1.0;marker.scale.x=0.04
                marker.color.a=1.0;marker.color.r=1.0 if overlap is not False else 0.0
                marker.color.g=0.5 if not body.yaw_valid else (1.0 if overlap is False else 0.0)
                if body.position_valid:
                    for x,y in collision.outline(body,collision.geometry(self)):
                        point=Point();point.x=float(x);point.y=float(y);point.z=0.3
                        marker.points.append(point)
                    markers.markers.append(marker)
                    label=Marker();label.header=marker.header;label.ns=marker.ns;label.id=j+2
                    label.type=Marker.TEXT_VIEW_FACING;label.action=Marker.ADD;label.pose.orientation.w=1.0
                    label.pose.position.x=float(body.x);label.pose.position.y=float(body.y);label.pose.position.z=1.0+j*0.4
                    label.scale.z=0.22;label.color=marker.color
                    label.text=(f"{'ego' if j==0 else vid}: {body.yaw_source} "
                                f"origin={body.origin} offset={body.center_offset:.3f} "
                                f"uncertainty={body.uncertainty:.3f} lateral={body.lateral_padding:.3f} age={self._collision_now-body.stamp:.2f}s")
                    markers.markers.append(label)
            diagnostic_lines.append(
                f'[CollisionBodyPair] target={vid}, overlap={overlap}, '
                f'ego={ego}, target_body={target}, '
                f'ego_age={self._collision_now-ego.stamp:.3f}, target_age={self._collision_now-target.stamp:.3f}')
        if diagnostic_lines:
            self.get_logger().info(' | '.join(diagnostic_lines),throttle_duration_sec=1.0)
        self._collision_body_publisher.publish(markers)

    def _v2x_callback(self, msg: V2XVehiclePositionArray) -> None:
        # If obstacle avoidance is disabled, clear tracker and bypass V2X processing entirely.
        if not self.USE_OBSTACLE_AVOIDANCE:
            if hasattr(self, '_v2x_tracker'):
                self._v2x_input_tracker.clear_active()
            return

        # Create a new list excluding the ego vehicle
        filtered_vehicles = []
        for v in msg.vehicles:
            if hasattr(self, '_ego_vehicle_id') and v.vehicle_id == self._ego_vehicle_id:
                continue
            filtered_vehicles.append(v)
        
        # Override msg.vehicles with the filtered list
        msg.vehicles = filtered_vehicles

        self._v2x_input_tracker.update(msg)
        self._v2x_received_once = True
        self._obstacles_updated = True

    def _map_marker_callback(self, msg: MarkerArray) -> None:
        left_pts = []
        right_pts = []
        road_zs = []
        for m in msg.markers:
            if m.ns == 'left_lane_bound':
                left_pts.extend([[p.x, p.y] for p in m.points])
            elif m.ns == 'right_lane_bound':
                right_pts.extend([[p.x, p.y] for p in m.points])
            elif m.ns == 'road_lanelets':
                road_zs.extend([p.z for p in m.points])

        if len(road_zs) > 0:
            self._map_z = float(np.mean(road_zs))
            self.get_logger().info(f"[MPC] Detected map z-coordinate: {self._map_z:.3f}")

        if len(left_pts) > 0 and len(right_pts) > 0:
            self.get_logger().info(
                f"[MPC] Received vector map boundaries. Left pts: {len(left_pts)}, Right pts: {len(right_pts)}"
            )
            left_arr = np.array(left_pts)
            right_arr = np.array(right_pts)
            # Race/CenterのCSV境界は生成時に確定済み。受信で幅を変更しない。
            self._reference_pathN_race.update_boundaries_from_markers(left_arr, right_arr)
            self._reference_pathN_center.update_boundaries_from_markers(left_arr, right_arr)

            self.destroy_subscription(self._map_marker_sub)

            self._map_marker_sub = None


    #　経路付近の障害物だけをMPCに渡す関数
    # 一番近いWaypointとの距離をみて近ければ採用
    def _filter_obstacles_to_corridor(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        if not obstacles or self._waypoint_xy.size == 0:
            return obstacles
        thr_sq = self._v2x_corridor_threshold_sq# 判定距離は2乗で取得している
        # 車両位置から最も近い40個の点を採用(追い越されるときに経路を譲るのを見越して後ろの点も考慮)
        wp = self._car.wp_id
        N = len(self._waypoint_xy)
        indices = [(wp + i) % N for i in range(-40, 41)]
        wps = self._waypoint_xy[indices]
        kept: List[Obstacle] = []
        for ob in obstacles:
            dxy = wps - np.array([ob.cx, ob.cy], dtype=np.float64)
            if np.min(np.einsum('ij,ij->i', dxy, dxy)) <= thr_sq: #もしもあるWaypointとの距離がthr_sq以下なら
                kept.append(ob)
        return kept

    def _border_cells_callback(self, msg: BorderCells):
        self._reference_path.set_border_cells(
            msg.dynamic_upper_bounds, msg.dynamic_lower_bounds, msg.rows, msg.cols)

    def _trajectory_callback(self, msg):
        self._trajectory = msg

    def _awsim_status_callback(self, msg):
        laps = int(msg.data[1])
        lap_time = msg.data[2]

        if self._current_laps is None:
            self._current_laps = 1 if laps == 0 else laps

        if laps > self._current_laps:
            self.get_logger().info(f'\033[32mLap {self._current_laps} completed! Lap time: {self._last_lap_time} s\033[0m')
            self._lap_times[self._current_laps] = self._last_lap_time
            self._current_laps = laps

        self._last_lap_time = lap_time

    def _condition_callback(self, msg: Int32):
        if self._last_condition is None:
            self._last_condition = msg.data

        diff_condition = msg.data - self._last_condition
        if diff_condition > 30.0:
            now = self.get_clock().now()
            self._last_colliding_time = now
            self._collision_times.append(now.nanoseconds / 1e9)
            self.get_logger().warning(f"Collision detected! Total recent: {len(self._collision_times)}")
        self._last_condition = msg.data

    def _stop_request_callback(self, msg: Empty) -> None:
        if self._enable_control:
            self.get_logger().warn(f"Stop request received {self._enable_control}")
            self._enable_control = False

    def _wait_until_clock_received(self) -> None:
        if self.use_sim_time:
            self.get_logger().info(f"wait until clock received...")
            rate = self.create_rate(10)
            rate.sleep()
            self.get_logger().info(f">> OK!")

    def _wait_until_message_received(self, message_getter, message_name: str, timeout: float, rate_hz: int = 30) -> None:

        t_start = self.get_clock().now()
        rate = self.create_rate(rate_hz)

        self.get_logger().info(f"wait until {message_name} received...")

        while message_getter() is None:
            now = self.get_clock().now()
            if (now - t_start).nanoseconds > timeout * 1e9:
                self.get_logger().info(f"now: {now}, t_start: {t_start}")
                raise TimeoutError(f"Timeout while waiting for {message_name} message")
            rate.sleep()

        self.get_logger().info(f">> OK!")

    def _wait_until_odom_received(self, timeout: float = 30.) -> None:
        self._wait_until_message_received(lambda: self._odom, 'odometry', timeout)

    def _wait_until_gnss_received(self, timeout: float = 30.) -> None:
        self._wait_until_message_received(lambda: self._gnss_pose, 'gnss_pose', timeout)

    def get_ego_pose(self) -> Pose2D:
        pose = odom_to_pose_2d(self._odom)
        if self._gnss_pose is not None:
            pose.x = self._gnss_pose.pose.pose.position.x
            pose.y = self._gnss_pose.pose.pose.position.y
        return pose

    def _wait_until_trajectory_received(self, timeout: float = 30.) -> None:
        if self._cfg.reference_path.update_by_topic:
            self._wait_until_message_received(lambda: self._trajectory, 'trajectory', timeout)

    def _wait_until_path_constraints_received(self, timeout: float = 30.) -> None:
        if self.USE_OBSTACLE_AVOIDANCE and self._cfg.reference_path.use_path_constraints_topic: # type: ignore
            self._wait_until_message_received(lambda: self._reference_path.path_constraints, 'path constraints', timeout)

    def _publish_mpc_pred_marker(self, x_pred, y_pred):
        pred_marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.frame_id = "map"
        clear_marker.ns = "mpc_pred"
        clear_marker.action = Marker.DELETEALL
        pred_marker_array.markers.append(clear_marker)
        m_base = Marker()
        m_base.header.frame_id = "map"
        m_base.ns = "mpc_pred"
        m_base.type = Marker.SPHERE
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.0
        m_base.scale = Vector3(x=0.5, y=0.5, z=0.5)
        m_base.color = self._pred_marker_color
        for i in range(len(x_pred)):
            m = copy.deepcopy(m_base)
            m.id = i
            m.pose.position.x = x_pred[i]
            m.pose.position.y = y_pred[i]
            pred_marker_array.markers.append(m) # type: ignore
        self._mpc_pred_pub.publish(pred_marker_array)
        self._mpc_pred_pub_dummy.publish(pred_marker_array)

    def _clear_mpc_pred_markers(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "mpc_pred"
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self._mpc_pred_pub.publish(marker_array)
        self._mpc_pred_pub_dummy.publish(marker_array)

    def _publish_lane_markers(self, ref_path: ReferencePath, n_lanes: int = 3) -> None:
        """
        3車線の範囲を塗りつぶしたMarkerArray(TRIANGLE_LIST)としてRvizに描画
        追い越し許可ゾーン → L0=赤, L1=黄, L2=緑
        追い越し不可ゾーン → グレー 
        ターゲット車線はより不透明度を高くして強調表示する
        """
        import math
        markers = MarkerArray()

        N = ref_path.n_waypoints
        has_overtake = hasattr(ref_path, 'overtake_zone')
        limit = N if ref_path.circular else N - 1

        for lane_idx in range(n_lanes):
            m = Marker()
            m.header.frame_id = "map"
            m.ns = f"lane_L{lane_idx}"
            m.id = lane_idx
            m.type = Marker.TRIANGLE_LIST
            m.action = Marker.ADD
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) # 頂点カラー(m.colors)を表示するために必須
            m.pose.orientation.w = 1.0

            # ターゲット車線（強調表示）か通常の車線かで不透明度(Alpha)を変える
            is_target_lane = hasattr(ref_path, 'target_lane_idx') and ref_path.target_lane_idx == lane_idx
            if is_target_lane:
                lane_colors = [
                    ColorRGBA(r=0.9, g=0.2, b=0.2, a=0.55),   # L0 内側  赤
                    ColorRGBA(r=0.9, g=0.8, b=0.1, a=0.55),   # L1 中央  黄
                    ColorRGBA(r=0.2, g=0.85, b=0.3, a=0.55),  # L2 外側  緑
                ]
                grey = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.25)
            else:
                lane_colors = [
                    ColorRGBA(r=0.9, g=0.2, b=0.2, a=0.30),   # L0 内側  赤
                    ColorRGBA(r=0.9, g=0.8, b=0.1, a=0.30),   # L1 中央  黄
                    ColorRGBA(r=0.2, g=0.85, b=0.3, a=0.30),  # L2 外側  緑
                ]
                grey = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.12)

            for wp_id in range(limit):
                next_wp_id = (wp_id + 1) % N

                wp = ref_path.get_waypoint(wp_id)
                next_wp = ref_path.get_waypoint(next_wp_id)

                if wp.ub is None or wp.lb is None or next_wp.ub is None or next_wp.lb is None:
                    continue

                # このwaypointの車線境界を取得
                lanes = ref_path.get_lane_bounds(wp_id, n_lanes)
                next_lanes = ref_path.get_lane_bounds(next_wp_id, n_lanes)
                if not lanes or not next_lanes:
                    continue

                ub_l, lb_l = lanes[lane_idx]
                next_ub_l, next_lb_l = next_lanes[lane_idx]

                # 世界座標へ変換 (normal_angle が利用可能な場合は -cos, -sin を使用し、無ければ従来の psi + pi/2 を使用)
                if wp.normal_angle is not None:
                    nx = -math.cos(wp.normal_angle)
                    ny = -math.sin(wp.normal_angle)
                    curr_left_x = wp.x + ub_l * nx
                    curr_left_y = wp.y + ub_l * ny
                    curr_right_x = wp.x + lb_l * nx
                    curr_right_y = wp.y + lb_l * ny
                else:
                    angle_ub = math.fmod(math.pi / 2.0 + wp.psi + math.pi, 2 * math.pi) - math.pi
                    curr_left_x = wp.x + ub_l * math.cos(angle_ub)
                    curr_left_y = wp.y + ub_l * math.sin(angle_ub)
                    curr_right_x = wp.x + lb_l * math.cos(angle_ub)
                    curr_right_y = wp.y + lb_l * math.sin(angle_ub)

                if next_wp.normal_angle is not None:
                    next_nx = -math.cos(next_wp.normal_angle)
                    next_ny = -math.sin(next_wp.normal_angle)
                    next_left_x = next_wp.x + next_ub_l * next_nx
                    next_left_y = next_wp.y + next_ub_l * next_ny
                    next_right_x = next_wp.x + next_lb_l * next_nx
                    next_right_y = next_wp.y + next_lb_l * next_ny
                else:
                    next_angle_ub = math.fmod(math.pi / 2.0 + next_wp.psi + math.pi, 2 * math.pi) - math.pi
                    next_left_x = next_wp.x + next_ub_l * math.cos(next_angle_ub)
                    next_left_y = next_wp.y + next_ub_l * next_sin(next_angle_ub)
                    next_right_x = next_wp.x + next_lb_l * math.cos(next_angle_ub)
                    next_right_y = next_wp.y + next_lb_l * math.sin(next_angle_ub)

                # 頂点データ
                p_curr_left = Point(x=curr_left_x, y=curr_left_y, z=self._map_z)
                p_curr_right = Point(x=curr_right_x, y=curr_right_y, z=self._map_z)
                p_next_left = Point(x=next_left_x, y=next_left_y, z=self._map_z)
                p_next_right = Point(x=next_right_x, y=next_right_y, z=self._map_z)

                # 追い越し許可されている部分（overtake_zoneがTrue）は車線ごとの色、それ以外はグレー
                if has_overtake and not ref_path.overtake_zone[wp_id]:
                    color_curr = grey
                else:
                    color_curr = lane_colors[lane_idx]

                if has_overtake and not ref_path.overtake_zone[next_wp_id]:
                    color_next = grey
                else:
                    color_next = lane_colors[lane_idx]

                # 三角形1 (表面 - CCW)
                m.points.append(p_curr_left)
                m.colors.append(color_curr)
                m.points.append(p_next_right)
                m.colors.append(color_next)
                m.points.append(p_curr_right)
                m.colors.append(color_curr)

                # 三角形1 (裏面 - CW)
                m.points.append(p_curr_left)
                m.colors.append(color_curr)
                m.points.append(p_curr_right)
                m.colors.append(color_curr)
                m.points.append(p_next_right)
                m.colors.append(color_next)

                # 三角形2 (表面 - CCW)
                m.points.append(p_curr_left)
                m.colors.append(color_curr)
                m.points.append(p_next_left)
                m.colors.append(color_next)
                m.points.append(p_next_right)
                m.colors.append(color_next)

                # 三角形2 (裏面 - CW)
                m.points.append(p_curr_left)
                m.colors.append(color_curr)
                m.points.append(p_next_right)
                m.colors.append(color_next)
                m.points.append(p_next_left)
                m.colors.append(color_next)

            markers.markers.append(m)

        self._lane_marker_pub.publish(markers)

    def _publish_ref_path_marker(self, ref_path: ReferencePath):
        WP_SPHERE_ENABLED = False

        ref_path_marker_array = MarkerArray()

        m_base = Marker()
        m_base.header.frame_id = "map"
        m_base.ns = "ref_path"
        m_base.type = Marker.LINE_STRIP
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.0
        m_base.scale.x = 0.2
        m_base.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.7)

        for i in range(len(ref_path.waypoints) - 1):
            m = copy.deepcopy(m_base)
            m.id = i
            start = Point()
            start.x = ref_path.waypoints[i].x
            start.y = ref_path.waypoints[i].y
            end = Point()
            end.x = ref_path.waypoints[i + 1].x
            end.y = ref_path.waypoints[i + 1].y
            m.points.append(start) # type: ignore
            m.points.append(end) # type: ignore
            ref_path_marker_array.markers.append(m) # type: ignore

        if WP_SPHERE_ENABLED:
            spheres = Marker()
            spheres.header.frame_id = "map"
            spheres.ns = "ref_path_point"
            spheres.type = Marker.SPHERE_LIST
            spheres.action = Marker.ADD
            radius = 0.2
            spheres.scale = Vector3(x=radius, y=radius, z=radius)
            spheres.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7)
            for i in range(len(ref_path.waypoints) - 1):
                p = Point()
                p.x = ref_path.waypoints[i].x
                p.y = ref_path.waypoints[i].y
                p.z = 0.
                spheres.points.append(p) #type: ignore
            ref_path_marker_array.markers.append(spheres) # type: ignore

        self._ref_path_pub.publish(ref_path_marker_array)
        self._ref_path_pub_dummy.publish(ref_path_marker_array)

    def _apply_lane_decision(
        self, *, requested_lane, now_sec, l0_prohibited, full_width_recovery,
        preserve_manoeuvre=False, allow_hybrid=True,
    ):
        """The only writer of the accepted session and live MPC corridor."""
        previous = self._lane_decision
        changed_lane = previous is None or previous.requested_lane != requested_lane
        changed_target = (
            previous is not None and previous.target_id != self._overtake.target_id
        )
        continuing_hybrid = bool(
            not changed_target
            and self._overtake.can_resume_hybrid(self._overtake.requested_lane)
            and (requested_lane == self._overtake.requested_lane
                 or (preserve_manoeuvre and requested_lane is None)))
        if (changed_lane or changed_target) and not continuing_hybrid:
            duration = (
                self._hybrid_overtake_transition_timeout
                if self._hybrid_overtake_enabled and allow_hybrid
                and self._overtake.target_id is not None and requested_lane in (0, 2)
                else 0.6
            )
            # Startup's first untracked L0 hold was immediate in submit.
            # Only a real Hybrid pass needs a new window on first application.
            if previous is None and not (
                self._hybrid_overtake_enabled and allow_hybrid
                and self._overtake.target_id is not None and requested_lane in (0, 2)
            ):
                duration = 0.0
            self._constraint_transition_until = now_sec + duration
        transition_active = now_sec < getattr(self, "_constraint_transition_until", 0.0)
        hybrid = self._overtake.hybrid
        resume_hybrid = (
            hybrid.vehicle_id == self._overtake.target_id
            and hybrid.lane_idx == requested_lane
            and not hybrid.completed
        )
        decision = decide_lane(
            target_id=self._overtake.target_id,
            requested_lane=requested_lane,
            l0_prohibited=l0_prohibited,
            full_width_recovery=full_width_recovery,
            transition_active=transition_active,
            hybrid_active=(
                self._hybrid_overtake_enabled and allow_hybrid
                and self._overtake.target_id is not None
                and (transition_active or resume_hybrid)),
        )
        if self._overtake.apply(decision, preserve_manoeuvre=continuing_hybrid):
            self._reset_outer_lane_progress()
            self._hybrid_reference_key = None
            self._mpcN_center.set_lane_transition_weights()
        self._lane_decision = decision
        self._target_lane_idx = decision.requested_lane
        self._applied_corridor_mode = decision.mode
        for path in (self._reference_path, self._reference_pathN):
            path.target_lane_idx = decision.applied_lane
            path.is_overtaking = decision.applied_lane is not None
        if previous != decision:
            if changed_lane and not continuing_hybrid and not (
                (previous is None or previous.requested_lane is None)
                and requested_lane in (None, 1)
            ):
                self._last_lane_change_time = now_sec
            self.get_logger().info(
                f"[LaneChange] request={requested_lane}, "
                f"applied={decision.applied_lane}, mode={decision.mode}, "
                f"vehicle_id={decision.target_id}"
            )
        if self._l1_probe_active and decision.applied_lane == 1:
            self._l1_probe_constraint_applied = True
        return (
            decision.applied_lane,
            decision.mode == "hybrid_lane_transition",
            decision.mode == "lane_transition",
        )


    def _control(self):
        # No pre-solve/early-return path may reuse last cycle's release proof.
        self._live_prediction_context = None
        now = self.get_clock().now()
        t = (now - self._t_start).nanoseconds / 1e9
        dt = (now - self._last_t).nanoseconds / 1e9

        self._last_t = now
        self._loop += 1

        # MPCの実行時間計測
        if self.use_stats:
            self._stats.record()

        # 制御周期を維持
        self._control_rate.sleep()

        self._collision_evidence_hold = False

        # Take the snapshot after the rate wait and apply exactly that generation
        # to the obstacle map.  A boolean notification can be overwritten by the
        # control thread and lose the last callback; generations cannot.
        if self.USE_OBSTACLE_AVOIDANCE:
            self._v2x_tracker = self._v2x_input_tracker.snapshot()
            if self._v2x_tracker.generation != self._v2x_applied_generation:
                predictions = self._v2x_tracker.predict_all(self._v2x_t_samples)
                self._dynamic_obstacles = predictions_to_obstacles(
                    predictions, self._v2x_vehicle_radius)
                self._map.reset_map()
                filtered_dynamic = self._filter_obstacles_to_corridor(
                    self._dynamic_obstacles)
                self._map.add_obstacles(
                    self._static_obstacles + filtered_dynamic)
                self._reference_path.reset_dynamic_constraints()
                self._v2x_applied_generation = self._v2x_tracker.generation
                self._obstacles_updated = False

        if self._loop % 100 == 0:
            # update reference path
            if self._cfg.reference_path.update_by_topic: # type: ignore
                new_referece_path = self._create_reference_path_from_autoware_trajectory(self._trajectory)
                if new_referece_path is not None:
                    self._car.reference_path = new_referece_path
                    self._car.update_reference_path(self._car.reference_path)

        #可視化用衝突判定
        is_colliding = False 
        if self._last_colliding_time is not None:
            elapsed_from_last_colliding = (now - self._last_colliding_time).nanoseconds / 1e9
            if elapsed_from_last_colliding < 5.0:
                is_colliding = True

        #オドメトリ(x,y,yaw,v)取得
        pose = self.get_ego_pose()
        self._collision_now = float(self.get_clock().now().nanoseconds)/1e9
        self._collision_ego_yaw = float(pose.theta)
        position_msg = self._gnss_pose if self._gnss_pose is not None else self._odom
        position_stamp = position_msg.header.stamp.sec + position_msg.header.stamp.nanosec/1e9
        yaw_stamp = self._odom.header.stamp.sec + self._odom.header.stamp.nanosec/1e9
        valid = (0.0 <= self._collision_now-position_stamp <= getattr(self,"_collision_max_age",0.5)
                 and 0.0 <= self._collision_now-yaw_stamp <= getattr(self,"_collision_max_age",0.5)
                 and abs(position_stamp-yaw_stamp) <= 0.2)
        self._collision_ego_metadata = (position_stamp,position_msg.header.frame_id,valid)
        self._collision_ego_alignment = None
        if self.USE_OBSTACLE_AVOIDANCE:
            measured = self._v2x_tracker._measured_bodies.get(self._ego_vehicle_id)
            if (measured and valid and 0.0 <= self._collision_now-measured.stamp <= self._collision_max_age
                    and abs(position_stamp-measured.stamp) <= 0.2):
                dx,dy=measured.x-pose.x,measured.y-pose.y
                c,s=math.cos(pose.theta),math.sin(pose.theta)
                self._collision_ego_alignment = (dx*c+dy*s,-dx*s+dy*c,measured.yaw-pose.theta,measured.stamp)
        if self.USE_OBSTACLE_AVOIDANCE:
            self._publish_collision_bodies(pose)
        v = self._odom.twist.twist.linear.x

        # Capture the L0 grid layout before any opponent-driven trajectory
        # decision.  Once eligible, the start sequence owns Center/L0 through
        # the end of boost, including the stationary wait before motion.
        self._capture_grounded_start_boost_layout()
        initial_start_exclusive_active = (
            self._initial_start_exclusive_active)
        grounded_snapshot_pending = (
            should_suppress_overtake_before_grounded_snapshot(
                self._awsim_state,
                self._grounded_start_boost_eligible is not None,
            )
        )
        startup_overtake_suppressed = (
            initial_start_exclusive_active or grounded_snapshot_pending)
        prestart_reverse_suppressed = should_reset_motion_latch(
            self._awsim_state)
        if grounded_snapshot_pending and not self._grounded_snapshot_wait_logged:
            self._grounded_snapshot_wait_logged = True
            self.get_logger().info(
                "[InitialGridSnapshotWait] suppressing ordinary overtake, "
                "Prepass, stopped-vehicle overtake, and Race/Center "
                "switching until the Grounded/Ready layout is captured."
            )
        
        # --- Dynamic Trajectory Switching (Race ↔ Center) ---
        opponent_ahead_detected = getattr(self, '_opponent_ahead_detected', False)
        
        # Estimate closest opponent distance in waypoint steps (signed)
        # temp_car は Race 軌道のウェイポイント参照専用。
        # update_states の代わりに _wp_xy キャッシュを使って直接 wp_id を取得し
        # BicycleModel の get_closest_waypoint 呼び出し（全点スキャン）を1回に抑える。
        wp_temp = self._carN_race.get_closest_waypoint(pose.x, pose.y)
        center_wp_temp = self._carN_center.get_closest_waypoint(pose.x, pose.y)
        race_heading = self._reference_pathN_race.get_waypoint(wp_temp).psi
        center_heading = self._reference_pathN_center.get_waypoint(
            center_wp_temp).psi
        race_rejoin_heading_diff = absolute_heading_difference(
            race_heading, center_heading)
        race_rejoin_heading_ok = (
            race_rejoin_heading_diff <= self._race_rejoin_max_heading)
        race_rejoin_probe_heading_ok = (
            race_rejoin_heading_diff <= self._race_rejoin_probe_start_heading)
        race_rejoin_probe_heading_released = (
            race_rejoin_heading_diff > self._race_rejoin_probe_release_heading)
        closest_opp_ahead = 99999
        closest_opp_behind = 99999
        closest_opp_ahead_id = None
        closest_opp_behind_id = None
        closest_opp_ahead_arc = math.inf
        closest_opp_ahead_speed = math.inf
        closest_opp_ahead_velocity_valid = False
        if self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker'):
            for vid in self._v2x_tracker.active_vehicle_ids():
                buf = self._v2x_tracker._samples.get(vid)
                if buf:
                    _, opp_x, opp_y = buf[-1]
                    # Euclidean 距離で事前フィルタ（過度に遠い車は全点スキャンをスキップ）
                    if math.hypot(opp_x - pose.x, opp_y - pose.y) > 35.0:
                        continue
                    arc_delta = self._center_longitudinal_between(
                        pose.x, pose.y, opp_x, opp_y)
                    wp_diff = self._center_arc_as_waypoint_delta(arc_delta)
                    if wp_diff is None:
                        continue
                    
                    if wp_diff >= 0:
                        if wp_diff < closest_opp_ahead:
                            closest_opp_ahead = wp_diff
                            closest_opp_ahead_id = vid
                            closest_opp_ahead_arc = float(arc_delta)
                            closest_vx, closest_vy = self._v2x_tracker.velocity(
                                vid)
                            closest_opp_ahead_speed = math.hypot(
                                closest_vx, closest_vy)
                            closest_opp_ahead_velocity_valid = (
                                self._v2x_tracker.has_velocity_estimate(vid))
                    else:
                        if abs(wp_diff) < closest_opp_behind:
                            closest_opp_behind = abs(wp_diff)
                            closest_opp_behind_id = vid

        # Enter Center at the near threshold and return to Race only after the
        # farther threshold remains clear. This prevents 26--34 waypoint
        # opponents from flipping the selected trajectory every cycle.
        now_sec = float(now.nanoseconds) / 1e9
        hold_elapsed = (
            self._trajectory_last_switch_time is None
            or now_sec - self._trajectory_last_switch_time
                >= self._trajectory_switch_min_hold
        )
        recovery_active = self._stuck_recovery_until is not None
        forced_overtake_pending = (
            self._forced_overtake_vehicle_id is not None
            or self._parallel_abort_active
        )
        l1_probe_failed_this_cycle = False
        center_lane_rejoin_clear = False

        # These values are also needed by Prepass recovery after the opponent
        # or AWSIM state changes. Compute them unconditionally so recovery/log
        # paths can never reference branch-local, uninitialized variables.
        center_heading_error = absolute_heading_difference(
            pose.theta, center_heading)
        measured_lateral_speed = abs(float(self._odom.twist.twist.linear.y))
        estimated_lateral_speed = abs(
            float(v) * math.sin(center_heading_error))
        center_lateral_speed = max(
            measured_lateral_speed, estimated_lateral_speed)
        center_yaw_rate = abs(float(self._odom.twist.twist.angular.z))
        center_rejoin_stable = (
            center_heading_error <= self._center_lane_rejoin_max_heading
            and center_lateral_speed
                <= self._center_lane_rejoin_max_lateral_speed
            and center_yaw_rate <= self._center_lane_rejoin_max_yaw_rate
            and self._mpc.infeasibility_counter == 0
        )
        l1_lateral_error = self._lane_lateral_error(pose.x, pose.y, 1)
        fresh_full_width_prediction = (
            self._reference_path is self._reference_pathN_center
            and self._reference_path.target_lane_idx is None
            and self._mpc.infeasibility_counter == 0
            and self._mpc.current_prediction is not None
            and not getattr(self._mpc, "used_prediction_fallback", False)
            and not self._mpc_safety_recovery_active
            and not recovery_active
            and not self._post_reverse_full_width_recovery_active
        )
        l1_prediction_fit_ratio = (
            self._current_prediction_l1_fit_ratio()
            if fresh_full_width_prediction else 0.0
        )
        l1_entry_geometry_ready = (
            l1_lateral_error < self._l1_rejoin_max_lateral_error
            and l1_prediction_fit_ratio >= self._l1_prediction_min_fit_ratio
        )

        # An L1 failure attributed by SafetyRecovery owns lateral selection
        # until full-width MPC and the vehicle state are both stable.  Only
        # then is L1 allowed back as an explicit feasibility probe.
        if self._l1_safety_recovery_active:
            l1_full_width_stable = (
                not self._mpc_safety_recovery_active
                and not recovery_active
                and center_rejoin_stable
                and fresh_full_width_prediction
                and l1_entry_geometry_ready
            )
            if l1_full_width_stable:
                if self._l1_safety_recovery_stable_since is None:
                    self._l1_safety_recovery_stable_since = now_sec
                stable_elapsed = (
                    now_sec - self._l1_safety_recovery_stable_since)
                if stable_elapsed >= self._center_lane_rejoin_stable_sec:
                    probe_context = (
                        self._l1_safety_recovery_context or "rejoin")
                    self._l1_safety_recovery_active = False
                    self._l1_safety_recovery_stable_since = None
                    self._l1_probe_active = True
                    self._l1_probe_context = probe_context
                    self._l1_probe_success_cycles = 0
                    self._l1_probe_constraint_applied = False
                    self.get_logger().info(
                        "[L1SafetyRecovery] full-width vehicle state stable; "
                        "starting controlled L1 feasibility probe: "
                        f"context={probe_context}, "
                        f"stable_sec={self._center_lane_rejoin_stable_sec:.2f}, "
                        f"heading_error="
                        f"{math.degrees(center_heading_error):.1f}deg, "
                        f"l1_lateral_error={l1_lateral_error:.2f}m, "
                        f"prediction_fit={l1_prediction_fit_ratio:.2f}, "
                        f"lateral_speed={center_lateral_speed:.2f}m/s, "
                        f"yaw_rate={center_yaw_rate:.2f}rad/s"
                    )
            else:
                self._l1_safety_recovery_stable_since = None
                self.get_logger().info(
                    "[L1SafetyRecoveryHold] keeping full width until MPC "
                    "and vehicle state are stable: "
                    f"generic_recovery={self._mpc_safety_recovery_active}, "
                    f"heading_error="
                    f"{math.degrees(center_heading_error):.1f}deg/"
                    f"{math.degrees(self._center_lane_rejoin_max_heading):.1f}, "
                    f"l1_lateral_error={l1_lateral_error:.2f}/"
                    f"{self._l1_rejoin_max_lateral_error:.2f}m, "
                    f"prediction_fit={l1_prediction_fit_ratio:.2f}/"
                    f"{self._l1_prediction_min_fit_ratio:.2f}, "
                    f"lateral_speed={center_lateral_speed:.2f}/"
                    f"{self._center_lane_rejoin_max_lateral_speed:.2f}m/s, "
                    f"yaw_rate={center_yaw_rate:.2f}/"
                    f"{self._center_lane_rejoin_max_yaw_rate:.2f}rad/s, "
                    f"mpc_infeasible={self._mpc.infeasibility_counter}",
                    throttle_duration_sec=1.0,
                )

        # Evaluate a SafetyRecovery-originated L1 probe independently from
        # opponent/trajectory state.  L1 may also be requested by fallback,
        # parallel-yield, or curve logic while Race is active.
        if (
            self._l1_safety_reprobe_pending
            and self._l1_probe_active
            and self._l1_probe_constraint_applied
            and not recovery_active
        ):
            if self._mpc.infeasibility_counter > 0:
                failed_context = self._l1_probe_context or "rejoin"
                self._log_l1_constraint_diagnostics(
                    context=f"safety_reprobe:{failed_context}",
                    failed_wp=center_wp_temp,
                    reason="safety-reprobe L1 infeasible",
                )
                self._l1_probe_active = False
                self._l1_probe_context = None
                self._l1_probe_success_cycles = 0
                self._l1_probe_constraint_applied = False
                self._center_lane_rejoin_stable_since = None
                l1_probe_failed_this_cycle = True
                if failed_context == "fallback":
                    self._l1_safety_recovery_active = False
                    self._l1_safety_recovery_stable_since = None
                    self._l1_safety_recovery_context = None
                    self._l1_safety_reprobe_pending = False
                    self._switch_prepass_to_follow(
                        "L1 fallback probe is infeasible after outer-lane "
                        "candidates were exhausted"
                    )
                else:
                    self._l1_safety_recovery_active = False
                    self._l1_safety_recovery_stable_since = None
                    self._l1_safety_recovery_context = None
                    self._l1_safety_reprobe_pending = False
                    self._start_l1_rejoin_backoff(
                        now_sec, center_wp_temp,
                        "safety-reprobe L1 infeasible")
                    self.get_logger().warn(
                        "[L1SafetyRecoveryProbe] L1 probe infeasible; returning "
                        "to dedicated full-width BACKOFF: "
                        f"context={failed_context}, "
                        f"mpc_infeasible={self._mpc.infeasibility_counter}"
                    )
            else:
                self._l1_probe_success_cycles += 1
                if (
                    self._l1_probe_success_cycles
                    >= self._l1_probe_required_success_cycles
                ):
                    confirmed_context = self._l1_probe_context or "rejoin"
                    self._l1_probe_active = False
                    self._l1_probe_context = None
                    self._l1_probe_success_cycles = 0
                    self._l1_probe_constraint_applied = False
                    if confirmed_context == "rejoin":
                        self._center_lane_rejoin_active = True
                    elif confirmed_context == "fallback":
                        self._prepass_fallback_lane_idx = 1
                        self._overtake.requested_lane = None
                        self._outer_lane_released_vehicle_id = None
                        self._prepass_attempted_outer_lanes.clear()
                    self._l1_safety_recovery_active = False
                    self._l1_safety_recovery_stable_since = None
                    self._l1_safety_recovery_context = None
                    self._l1_safety_reprobe_pending = False
                    self.get_logger().info(
                        "[L1SafetyRecoveryProbe] L1 constraint confirmed "
                        "feasible; normal L1 ownership may resume: "
                        f"context={confirmed_context}, success_cycles="
                        f"{self._l1_probe_required_success_cycles}"
                    )

        initial_start_lateral_hold_active = (
            self._initial_start_hold_l0
            and (
                self._initial_start_boost_armed
                or self._initial_start_boost_until is not None
                or self._initial_start_post_hold_active
            )
        )
        initial_start_l0_hold_active = (
            self._initial_start_hold_l0
            and (
                self._initial_start_boost_armed
                or self._initial_start_boost_until is not None
                or self._initial_start_post_hold_l0_active
            )
        )
        if initial_start_lateral_hold_active and (
            self._center_lane_rejoin_active
            or (
                self._l1_probe_active
                and self._l1_probe_context == "rejoin"
            )
        ):
            self._center_lane_rejoin_active = False
            self._center_lane_rejoin_constraint_released = False
            self._center_lane_rejoin_stable_since = None
            self._l1_probe_active = False
            self._l1_probe_context = None
            self._l1_probe_success_cycles = 0
            self._l1_probe_constraint_applied = False
            if self._l1_safety_reprobe_pending:
                self._l1_safety_recovery_active = True
                self._l1_safety_recovery_stable_since = None
            self.get_logger().info(
                "[InitialStartLaneHold] cancelled pending L1 rejoin while "
                "initial/post-boost L0 hold is active."
            )
        initial_post_hold_max_kappa = 0.0
        if self._initial_start_post_hold_active:
            center_wp_id = self._carN_center.get_closest_waypoint(
                pose.x, pose.y)
            center_n_wps = self._reference_pathN_center.n_waypoints
            initial_post_hold_max_kappa = max(
                abs(self._reference_pathN_center.get_waypoint(
                    (center_wp_id + offset) % center_n_wps).kappa)
                for offset in range(
                    self._initial_start_post_hold_lookahead_wps)
            )
            post_hold_elapsed = (
                0.0 if self._initial_start_post_hold_started_at is None
                else now_sec - self._initial_start_post_hold_started_at
            )
            if (
                self._initial_start_post_hold_l0_active
                and post_hold_elapsed
                    >= self._initial_start_post_hold_min_sec
            ):
                self._initial_start_post_hold_l0_active = False
                initial_start_l0_hold_active = False
                self.get_logger().info(
                    "[InitialStartPostBoostHold] minimum L0 hold complete; "
                    "releasing L0 constraint to full width before L1 rejoin: "
                    f"elapsed={post_hold_elapsed:.2f}s"
                )
            post_hold_state_stable = (
                not self._initial_start_post_hold_l0_active
                and initial_post_hold_max_kappa
                    <= self._initial_start_post_hold_max_kappa
            )
            if post_hold_state_stable:
                if self._initial_start_post_hold_stable_since is None:
                    self._initial_start_post_hold_stable_since = now_sec
                elif (
                    now_sec - self._initial_start_post_hold_stable_since
                    >= self._initial_start_post_hold_stable_sec
                ):
                    self._initial_start_post_hold_active = False
                    self._initial_start_post_hold_l0_active = False
                    self._initial_start_post_hold_stable_since = None
                    initial_start_lateral_hold_active = False
                    self.get_logger().info(
                        "[InitialStartPostBoostHold] predicted curvature "
                        "confirmed; handing off to normal L1 rejoin checks: "
                        f"elapsed={post_hold_elapsed:.2f}s, speed={abs(v):.2f}m/s, "
                        f"max_kappa={initial_post_hold_max_kappa:.4f}, "
                        f"heading_error={math.degrees(center_heading_error):.1f}deg, "
                        f"lateral_speed={center_lateral_speed:.2f}m/s, "
                        f"yaw_rate={center_yaw_rate:.2f}rad/s"
                    )
            else:
                self._initial_start_post_hold_stable_since = None
                self.get_logger().info(
                    "[InitialStartPostBoostHold] waiting for low-curvature "
                    "window before L1 rejoin: "
                    f"elapsed={post_hold_elapsed:.2f}/"
                    f"{self._initial_start_post_hold_min_sec:.2f}s, "
                    f"max_kappa={initial_post_hold_max_kappa:.4f}/"
                    f"{self._initial_start_post_hold_max_kappa:.4f}, "
                    f"heading_error={math.degrees(center_heading_error):.1f}deg, "
                    f"lateral_speed={center_lateral_speed:.2f}m/s, "
                    f"yaw_rate={center_yaw_rate:.2f}rad/s, "
                    "constraint="
                    f"{'L0' if self._initial_start_post_hold_l0_active else 'full_width'}",
                    throttle_duration_sec=1.0,
                )

        if startup_overtake_suppressed:
            # Stationary grid vehicles are not overtake candidates.  Keep one
            # deterministic Center trajectory for the complete L0 boost and
            # do not advance normal Race/Center or rejoin state machines.
            opponent_ahead_detected = True
            forced_overtake_pending = False
            self._trajectory_clear_since = None
            self._center_lane_rejoin_stable_since = None
            self._trajectory_switch_reason = "initial_start_exclusive_hold"
        elif recovery_active:
            self._trajectory_clear_since = None
            self._center_lane_rejoin_stable_since = None
            if self._l1_safety_reprobe_pending:
                self._l1_safety_recovery_active = True
                self._l1_safety_recovery_stable_since = None
            self._l1_probe_active = False
            self._l1_probe_context = None
            self._l1_probe_success_cycles = 0
            self._l1_probe_constraint_applied = False
            self._trajectory_switch_reason = "stuck_recovery_hold"
        elif opponent_ahead_detected:
            center_lane_rejoin_clear = (
                closest_opp_ahead > self._trajectory_exit_center_wps
                and closest_opp_behind >= self._center_lane_rejoin_behind_wps
            )
            opponent_is_clear = (
                closest_opp_ahead > self._trajectory_exit_center_wps
                and closest_opp_behind > self._trajectory_behind_release_wps
            )
            race_targets = self._race_handoff_lateral_targets(
                center_wp=center_wp_temp,
                extra=self._race_handoff_guard_extra_lookahead_wps)
            race_targets_valid, race_targets_reason = (
                self._validate_race_handoff_targets(
                    race_targets, self._carN_center.spatial_state.e_y)
            )
            race_handoff_blocked = (
                forced_overtake_pending
                or recovery_active
                or self._mpc_safety_recovery_active
                or self._post_reverse_full_width_recovery_active
                or self._prepass_fallback_recovery_active
                or self._prepass_fallback_commit_pending
                or self._parallel_abort_active
            )
            if (
                opponent_is_clear
                and race_rejoin_probe_heading_ok
                and race_targets_valid
                and not race_handoff_blocked
                and not self._race_rejoin_handoff_active
                and self._reference_path is self._reference_pathN_center
            ):
                self._start_race_rejoin_handoff(
                    abs(float(self._carN_center.spatial_state.e_y)
                        - float(race_targets[0])),
                    now_sec, center_wp_temp)
            elif (
                opponent_is_clear
                and race_rejoin_probe_heading_ok
                and not race_targets_valid
            ):
                self.get_logger().warn(
                    "[RaceHandoffGuard] keeping conventional L1 rejoin: "
                    f"reason={race_targets_reason}",
                    throttle_duration_sec=2.0)
            if self._l1_rejoin_backoff_active:
                self._l1_rejoin_backoff_full_width_success_since = (
                    update_continuous_condition_since(
                        self._l1_rejoin_backoff_full_width_success_since,
                        now_sec=now_sec,
                        condition=fresh_full_width_prediction,
                    )
                )
                backoff_elapsed = (
                    0.0 if self._l1_rejoin_backoff_started_at is None
                    else now_sec - self._l1_rejoin_backoff_started_at
                )
                backoff_wp_progress = circular_forward_progress(
                    self._l1_rejoin_backoff_failed_wp,
                    center_wp_temp,
                    self._reference_pathN_center.n_waypoints,
                )
                full_width_success_elapsed = (
                    0.0
                    if self._l1_rejoin_backoff_full_width_success_since is None
                    else now_sec
                        - self._l1_rejoin_backoff_full_width_success_since
                )
                if should_exit_l1_probe_backoff(
                    elapsed_sec=backoff_elapsed,
                    cooldown_sec=self._l1_probe_retry_cooldown_sec,
                    waypoint_progress=backoff_wp_progress,
                    minimum_waypoint_progress=(
                        self._l1_probe_retry_min_wp_progress),
                    full_width_success_sec=full_width_success_elapsed,
                    required_full_width_success_sec=(
                        self._l1_backoff_full_width_success_sec),
                ):
                    self.get_logger().info(
                        "[L1ProbeBackoffRelease] full-width recovery complete; "
                        "returning to FULL_WIDTH_WAIT: "
                        f"elapsed={backoff_elapsed:.2f}s, "
                        f"wp_progress={backoff_wp_progress}, "
                        f"full_width_success="
                        f"{full_width_success_elapsed:.2f}s"
                    )
                    self._reset_l1_rejoin_backoff()
                else:
                    self.get_logger().info(
                        "[L1ProbeBackoffHold] keeping full width: "
                        f"elapsed={backoff_elapsed:.2f}/"
                        f"{self._l1_probe_retry_cooldown_sec:.2f}s, "
                        f"wp_progress={backoff_wp_progress}/"
                        f"{self._l1_probe_retry_min_wp_progress}, "
                        f"full_width_success={full_width_success_elapsed:.2f}/"
                        f"{self._l1_backoff_full_width_success_sec:.2f}s",
                        throttle_duration_sec=1.0,
                    )
            # L1 is first applied as a short feasibility probe. The counter
            # observed here is the result of the previous cycle's L1-constrained
            # solve, so no second MPC solve is needed.
            if (
                self._l1_probe_active
                and self._l1_probe_constraint_applied
                and not self._l1_safety_reprobe_pending
            ):
                if self._mpc.infeasibility_counter > 0:
                    failed_context = self._l1_probe_context
                    self._log_l1_constraint_diagnostics(
                        context=f"probe:{failed_context}",
                        failed_wp=center_wp_temp,
                        reason="L1 constrained MPC infeasible",
                    )
                    self._l1_probe_active = False
                    self._l1_probe_context = None
                    self._l1_probe_success_cycles = 0
                    self._l1_probe_constraint_applied = False
                    l1_probe_failed_this_cycle = True
                    self._center_lane_rejoin_stable_since = None
                    if failed_context == "fallback":
                        self._switch_prepass_to_follow(
                            "L1 fallback probe is infeasible after outer-lane "
                            "candidates were exhausted"
                        )
                    elif failed_context == "rejoin":
                        self._start_l1_rejoin_backoff(
                            now_sec, center_wp_temp,
                            "L1 constrained MPC infeasible")
                    self.get_logger().warn(
                        "[L1Probe] L1 constraint infeasible; returning to "
                        f"full width: context={failed_context}, "
                        f"mpc_infeasible={self._mpc.infeasibility_counter}"
                    )
                else:
                    self._l1_probe_success_cycles += 1
                    if (
                        self._l1_probe_success_cycles
                        >= self._l1_probe_required_success_cycles
                    ):
                        confirmed_context = self._l1_probe_context
                        self._l1_probe_active = False
                        self._l1_probe_context = None
                        self._l1_probe_success_cycles = 0
                        self._l1_probe_constraint_applied = False
                        if confirmed_context == "rejoin":
                            self._center_lane_rejoin_active = True
                        elif confirmed_context == "fallback":
                            self._prepass_fallback_lane_idx = 1
                            self._overtake.requested_lane = None
                            self._outer_lane_released_vehicle_id = None
                            self._prepass_attempted_outer_lanes.clear()
                        self.get_logger().info(
                            "[L1Probe] L1 constraint confirmed feasible: "
                            f"context={confirmed_context}, "
                            f"success_cycles="
                            f"{self._l1_probe_required_success_cycles}"
                        )

            # Stability gates only the start of L1 rejoin. Once started, keep
            # the L1 target latched so threshold noise cannot cause weaving.
            if (
                not self._center_lane_rejoin_active
                and not self._l1_probe_active
                and not self._l1_safety_reprobe_pending
                and not self._l1_rejoin_backoff_active
                and not self._race_rejoin_handoff_active
            ):
                can_start_center_rejoin = (
                    center_lane_rejoin_clear
                    and not forced_overtake_pending
                    and not initial_start_lateral_hold_active
                    and not self._prepass_fallback_recovery_active
                    and not self._prepass_fallback_commit_pending
                    and fresh_full_width_prediction
                    and l1_entry_geometry_ready
                    and (
                        not self._center_lane_rejoin_stability_enabled
                        or center_rejoin_stable
                    )
                )
                if can_start_center_rejoin:
                    if self._center_lane_rejoin_stable_since is None:
                        self._center_lane_rejoin_stable_since = now_sec
                    stable_elapsed = (
                        now_sec - self._center_lane_rejoin_stable_since)
                    required_stable_sec = (
                        self._center_lane_rejoin_stable_sec
                        if self._center_lane_rejoin_stability_enabled else 0.0
                    )
                    if stable_elapsed >= required_stable_sec:
                        self._l1_probe_active = True
                        self._l1_probe_context = "rejoin"
                        self._l1_probe_success_cycles = 0
                        self._l1_probe_constraint_applied = False
                        self._center_lane_rejoin_stable_since = None
                        self.get_logger().info(
                            "[CenterLaneRejoin] stability confirmed; starting "
                            "L1 feasibility probe: "
                            f"heading_error="
                            f"{math.degrees(center_heading_error):.1f}deg, "
                            f"l1_lateral_error={l1_lateral_error:.2f}m/"
                            f"{self._l1_rejoin_max_lateral_error:.2f}m, "
                            f"prediction_fit={l1_prediction_fit_ratio:.2f}/"
                            f"{self._l1_prediction_min_fit_ratio:.2f}, "
                            f"lateral_speed={center_lateral_speed:.2f}m/s, "
                            f"yaw_rate={center_yaw_rate:.2f}rad/s."
                        )
                else:
                    self._center_lane_rejoin_stable_since = None
                    if center_lane_rejoin_clear and not forced_overtake_pending:
                        self.get_logger().info(
                            "[CenterLaneRejoinHold] waiting for stable state: "
                            f"heading_error="
                            f"{math.degrees(center_heading_error):.1f}deg/"
                            f"{math.degrees(self._center_lane_rejoin_max_heading):.1f}, "
                            f"l1_lateral_error={l1_lateral_error:.2f}/"
                            f"{self._l1_rejoin_max_lateral_error:.2f}m, "
                            f"prediction_fit={l1_prediction_fit_ratio:.2f}/"
                            f"{self._l1_prediction_min_fit_ratio:.2f}, "
                            f"full_width_prediction="
                            f"{fresh_full_width_prediction}, "
                            f"lateral_speed={center_lateral_speed:.2f}/"
                            f"{self._center_lane_rejoin_max_lateral_speed:.2f}m/s, "
                            f"yaw_rate={center_yaw_rate:.2f}/"
                            f"{self._center_lane_rejoin_max_yaw_rate:.2f}rad/s, "
                            f"mpc_infeasible={self._mpc.infeasibility_counter}",
                            throttle_duration_sec=1.0,
                        )

            constraint_transition_until = getattr(
                self, '_constraint_transition_until', 0.0)
            center_lane_rejoin_ready = (
                self._center_lane_rejoin_active
                and not self._l1_safety_reprobe_pending
                and (
                    self._target_lane_idx == 1
                    or self._center_lane_rejoin_constraint_released
                )
                and now_sec >= constraint_transition_until
            )
            conventional_race_rejoin_ready = (
                opponent_is_clear
                and not forced_overtake_pending
                and center_lane_rejoin_ready
                and race_rejoin_heading_ok
            )
            handoff_race_rejoin_ready = (
                opponent_is_clear
                and not forced_overtake_pending
                and self._race_rejoin_handoff_active
                and self._race_rejoin_probe_confirmed
                and race_rejoin_heading_ok
            )
            race_rejoin_ready = (
                conventional_race_rejoin_ready
                or handoff_race_rejoin_ready)
            if race_rejoin_ready:
                if self._trajectory_clear_since is None:
                    self._trajectory_clear_since = now_sec
                clear_confirmed = (
                    now_sec - self._trajectory_clear_since
                    >= self._trajectory_exit_confirm
                )
                if clear_confirmed and hold_elapsed:
                    opponent_ahead_detected = False
                    self._trajectory_switch_reason = "opponent_clear"
            else:
                self._trajectory_clear_since = None
                if (
                    opponent_is_clear
                    and not forced_overtake_pending
                    and not race_rejoin_heading_ok
                ):
                    self._trajectory_switch_reason = "race_rejoin_heading_hold"
                    self.get_logger().info(
                        "[TrajectorySwitchHold] waiting for Race/Center "
                        f"alignment: heading_diff="
                        f"{math.degrees(race_rejoin_heading_diff):.1f}deg "
                        f"> {math.degrees(self._race_rejoin_max_heading):.1f}deg",
                        throttle_duration_sec=1.0,
                    )
        else:
            if self._race_rejoin_handoff_active:
                self._reset_race_rejoin_handoff()
            self._reset_l1_rejoin_backoff()
            self._center_lane_rejoin_active = False
            self._center_lane_rejoin_constraint_released = False
            self._center_lane_rejoin_stable_since = None
            if not self._l1_safety_reprobe_pending:
                self._l1_probe_active = False
                self._l1_probe_context = None
                self._l1_probe_success_cycles = 0
                self._l1_probe_constraint_applied = False
            self._trajectory_clear_since = None
            slow_lead_prepare_detected = bool(
                closest_opp_ahead_velocity_valid
                and closest_opp_ahead_speed
                    <= self._slow_lead_overtake_speed
                and closest_opp_ahead_arc
                    <= self._slow_lead_overtake_prepare_distance
            )
            if (
                (closest_opp_ahead < self._trajectory_enter_center_wps
                 or slow_lead_prepare_detected)
                and hold_elapsed
            ):
                opponent_ahead_detected = True
                self._trajectory_switch_reason = (
                    "slow_lead_prepare" if slow_lead_prepare_detected
                    else "opponent_ahead")

        # Enforce racing-line-only behavior when obstacle avoidance is disabled
        if not self.USE_OBSTACLE_AVOIDANCE:
            opponent_ahead_detected = False

        # Lock trajectory (CSV) switching during curves (when following centerline)
        if self._curve_lane_lock_enabled:
            was_following_centerline = getattr(self, '_opponent_ahead_detected', False)
            if was_following_centerline:
                is_in_curve = False
                current_wp = self._car.get_closest_waypoint(pose.x, pose.y)
                N_wps = self._reference_path.n_waypoints
                reversed_wp = N_wps - 1 - current_wp
                for r in self._curve_lane_lock_wps:
                    if len(r) == 2:
                        start, end = r[0], r[1]
                        if start <= end:
                            if start <= reversed_wp <= end:
                                is_in_curve = True
                                break
                        else:
                            if reversed_wp >= start or reversed_wp <= end:
                                is_in_curve = True
                                break
                
                if is_in_curve:
                    opponent_ahead_detected = True
                    self._trajectory_switch_reason = "curve_hold"

        # Check if we should force centerline due to multiple recent collisions
        self._collision_times = [t for t in self._collision_times if now_sec - t < self._stuck_collision_window]
        force_centerline_by_collision = (len(self._collision_times) >= self._stuck_collision_count_threshold)

        if force_centerline_by_collision and not recovery_active:
            opponent_ahead_detected = True
            self._trajectory_switch_reason = "collision_hold"
            #self.get_logger().warn(
            #    f"[CollisionSwitch] Multiple recent collisions detected ({len(self._collision_times)} in {self._stuck_collision_window:.1f}s). "
            #    "Forcing centerline for safety.",
            #    throttle_duration_sec=2.0
            #)

        if startup_overtake_suppressed:
            # This final trajectory override also wins over curve/collision
            # logic.  Those safety systems may still limit speed, but they may
            # not change the boost trajectory or create passing state.
            opponent_ahead_detected = True
            self._trajectory_switch_reason = "initial_start_exclusive_hold"

        self._opponent_ahead_detected = opponent_ahead_detected

        # 追い越し・追従フラグが立っている場合: Center軌道 (traj_center313.csv)
        # 通常走行時: Race軌道 (traj_race_cl_mpc.csv)
        # --- 切り替え検出: 実際に軌道が変わったときだけ OSQP を再初期化 ---
        trajectory_switched = (opponent_ahead_detected != self._prev_opponent_ahead_detected)
        self._prev_opponent_ahead_detected = opponent_ahead_detected

        if opponent_ahead_detected:
            self._reference_pathN = self._reference_pathN_center
            #self._reference_path10 = self._reference_path10_center
            self._carN = self._carN_center
            #self._car10 = self._car10_center
            self._mpc_cfg = self._mpc_cfg_center
            self._mpcN = self._mpcN_center
            #self._mpc10 = self._mpc10_center

        else:
            self._reference_pathN = self._reference_pathN_race
            #self._reference_path10 = self._reference_path10_race
            self._carN = self._carN_race
            #self._car10 = self._car10_race
            self._mpc_cfg = self._mpc_cfg_race
            self._mpcN = self._mpcN_race
            #self._mpc10 = self._mpc10_race

        # 軌道が切り替わった場合: OSQP ソルバーの内部状態(A行列構造)を強制リセット。
        # これにより、旧軌道向けに warm-start された状態が新軌道の制約と食い違って
        # infeasible になるリスクを防ぐ。
        if trajectory_switched:
            # Race/Center use separate MPC instances. Preserve the physical
            # steering state owned by the outgoing controller, while still
            # discarding its path-specific warm start and prediction.
            inherited_steering = float(getattr(
                self._mpc, "previous_steering", float("nan")))
            if not np.isfinite(inherited_steering):
                inherited_steering = float(self._last_u[1])
            destination_delta_limit = float(self._mpc_cfg.delta_max)
            inherited_steering = float(np.clip(
                inherited_steering,
                -destination_delta_limit,
                destination_delta_limit,
            ))
            self._trajectory_last_switch_time = now_sec
            label = "Race→Center" if opponent_ahead_detected else "Center→Race"
            center_wp_at_switch = center_wp_temp
            if (
                opponent_ahead_detected
                and not startup_overtake_suppressed
                and closest_opp_ahead_id is not None
            ):
                self._trajectory_vehicle_id = closest_opp_ahead_id
            if startup_overtake_suppressed:
                switch_vehicle_id = None
            else:
                switch_vehicle_id = (
                    closest_opp_ahead_id or self._trajectory_vehicle_id
                    if opponent_ahead_detected
                    else closest_opp_behind_id or self._trajectory_vehicle_id
                )
            self.get_logger().info(
                f"[TrajectorySwitch] {label}: "
                f"race_wp={wp_temp}, center_wp={center_wp_at_switch}, "
                f"reason={self._trajectory_switch_reason}, "
                f"vehicle_id={switch_vehicle_id}, "
                f"ahead_wps={closest_opp_ahead}, "
                f"behind_wps={closest_opp_behind}, "
                f"pose=({pose.x:.2f}, {pose.y:.2f}); "
                "resetting OSQP solver state."
            )
            for switched_mpc in (self._mpcN_race, self._mpcN_center):
                switched_mpc.osqp_initialized = False
                switched_mpc.current_prediction = None
                switched_mpc.current_control = np.zeros_like(
                    switched_mpc.current_control)
                switched_mpc.infeasibility_counter = 0
            self._mpcN.previous_steering = inherited_steering
            if not opponent_ahead_detected:
                # The confirmed probe has handed ownership to Race. Stop the
                # shadow state before the live Race solve later this cycle.
                self._reset_race_rejoin_handoff()
            self.get_logger().info(
                "[TrajectorySwitchSteeringSync] inherited current steering "
                f"into the destination MPC: steering="
                f"{inherited_steering:+.4f}rad, destination="
                f"{'Center' if opponent_ahead_detected else 'Race'}"
            )
            self._clear_mpc_pred_markers()
            if opponent_ahead_detected:
                self._post_overtake_vehicle_id = None
                self._race_return_time = None
                self._center_lane_rejoin_constraint_released = False
                self._prepass_fallback_lane_idx = None
                self._prepass_fallback_blocked = False
                self._prepass_fallback_follow_active = False
                self._prepass_retry_after_reverse = False
                self._prepass_retry_lane_idx = None
                self._prepass_fallback_recovery_active = False
                self._prepass_fallback_recovery_stable_since = None
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_lane_idx = None
            else:
                # L1 recovery is defined against the Center reference.  Carrying
                # it into Race makes fresh_full_width_prediction impossible and
                # leaves lane selection suppressed indefinitely.
                if self._l1_safety_recovery_active:
                    self.get_logger().info(
                        "[L1SafetyRecoveryRelease] cancelling Center-only L1 "
                        "recovery on confirmed Race return")
                self._l1_safety_recovery_active = False
                self._l1_safety_recovery_stable_since = None
                self._l1_safety_recovery_context = None
                self._l1_safety_reprobe_pending = False
                self._l1_probe_active = False
                self._l1_probe_context = None
                self._l1_probe_success_cycles = 0
                self._l1_probe_constraint_applied = False
                self._post_overtake_vehicle_id = (
                    self._overtake.target_id
                    or self._trajectory_vehicle_id
                    or closest_opp_behind_id
                )
                self.get_logger().info(
                    "[OvertakeLatch] released after Race return: "
                    f"vehicle_id={self._post_overtake_vehicle_id}, "
                    f"lane={self._overtake.requested_lane}"
                )
                self._overtake.release_target()
                self._overtake.committed = False
                self._overtake_completed_target_id = None
                self._outer_lane_released_vehicle_id = None
                self._center_lane_rejoin_constraint_released = False
                self._prepass_fallback_lane_idx = None
                self._prepass_fallback_blocked = False
                self._prepass_fallback_follow_active = False
                self._prepass_fallback_recovery_active = False
                self._prepass_fallback_recovery_stable_since = None
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_lane_idx = None
                self._trajectory_vehicle_id = None

        # --- 重要: self._car / self._mpc / self._reference_path を _carN / _mpcN / _reference_pathN に同期 ---
        # self._carN / _mpcN / _reference_pathN は切り替えブロックで更新されているが、
        # 実際に制御計算に使われるのは self._car / _mpc / _reference_path なので必ず反映する。
        self._car = self._carN
        self._mpc = self._mpcN
        self._reference_path = self._reference_pathN

        # Save the stationary initial-grid layout during Grounded or Ready.
        # Ready is needed when this node starts after the one-shot Grounded
        # notification. This also covers a state callback arriving after V2X.
        self._capture_grounded_start_boost_layout()

        # Solve from the state at which this cycle's command reaches the actuator.
        predicted_pose = self._predict_pose_after_steering_delay(
            pose, v, float(now.nanoseconds) / 1e9)
        self._car.update_states(
            predicted_pose.x, predicted_pose.y, predicted_pose.theta)
        wp = self._car.wp_id  # update_states 内で get_closest_waypoint が実行済み

        # Initial race-start boost. Grounded/Ready only arms it. Turbo and the
        # duration timer start once measured forward motion begins. A later
        # Start notification may only re-arm an already verified layout, and
        # this can never re-arm on later laps.
        initial_start_boost_active = False
        now_sec_for_start = float(now.nanoseconds) / 1e9
        if (
            self._initial_start_boost_armed
            and not self._initial_start_boost_done
            and self._initial_start_boost_until is None
            and abs(v) >= self._initial_start_motion_speed_threshold
        ):
            self._initial_start_boost_until = (
                now_sec_for_start + self._initial_start_boost_duration)
            self._initial_start_boost_armed = False
            self._publish_initial_turbo()
            self.get_logger().info(
                "[InitialStartMotion] measured vehicle motion; starting turbo "
                f"and maximum acceleration: speed={abs(v):.2f}m/s, "
                f"threshold={self._initial_start_motion_speed_threshold:.2f}m/s"
            )
        if (
            self._initial_start_boost_until is not None
            and now_sec_for_start >= self._initial_start_boost_until
        ):
            self._initial_start_boost_done = True
            self._initial_start_boost_until = None
            if self._initial_start_exclusive_active:
                self._initial_start_exclusive_active = False
                self.get_logger().info(
                    "[InitialStartExclusive] boost completed; releasing "
                    "ordinary overtake, Prepass, and Race/Center selection."
                )
            if self._initial_start_hold_l0:
                self._initial_start_post_hold_active = True
                self._initial_start_post_hold_l0_active = True
                self._initial_start_post_hold_started_at = now_sec_for_start
                self._initial_start_post_hold_stable_since = None
                initial_start_lateral_hold_active = True
                self.get_logger().info(
                    "[InitialStartPostBoostHold] boost finished; keeping L0 "
                    "for the configured minimum time, then recovering with "
                    "full-width constraints until L1 rejoin is safe."
                )
        if (
            not self._initial_start_boost_done
            and self._initial_start_boost_until is not None
        ):
            initial_start_boost_active = (
                self._grounded_start_boost_eligible is True)
            if not self._initial_start_boost_decision_logged:
                self._initial_start_boost_decision_logged = True
                if self._grounded_start_boost_eligible is None:
                    decision = "skipped"
                    reason = "Grounded snapshot unavailable"
                elif initial_start_boost_active:
                    decision = "enabled"
                    reason = "Grounded snapshot has only ego in right-side L0"
                elif self._grounded_ego_lane_idx != 0:
                    decision = "skipped"
                    reason = (
                        f"Grounded ego_lane=L{self._grounded_ego_lane_idx} "
                        "is not right-side L0"
                    )
                else:
                    decision = "skipped"
                    reason = (
                        "Grounded right-side L0 occupied by "
                        f"{self._grounded_l0_vehicle_ids}"
                    )
                self.get_logger().info(
                    "[InitialStartBoostDecision] "
                    f"decision={decision}, reason={reason}, "
                    "capture_state="
                    f"{self._grounded_start_boost_capture_state}, "
                    f"grounded_ego_lane={self._grounded_ego_lane_idx}, "
                    "grounded_L0_vehicle_ids="
                    f"{self._grounded_l0_vehicle_ids}"
                )
            if initial_start_boost_active and not self._initial_start_boost_logged:
                self._initial_start_boost_logged = True
                # Usually already sent by the measured-motion trigger above.
                self._publish_initial_turbo()
                self.get_logger().info(
                    "[InitialStartBoost] Grounded snapshot shows right-side "
                    "L0 contains only ego; "
                    f"turbo={self._initial_start_turbo_enabled}, "
                    "using maximum acceleration for "
                    f"{self._initial_start_boost_duration:.1f}s."
                )


        # --- Overtaking Lane Selection Logic ---
        if not hasattr(self, '_target_lane_idx'):
            self._target_lane_idx = None

        opponent_ahead = None
        opponent_offset = 0.0
        opponent_distance = 99999.0 #前方車両との距離
        opponent_arc_distance = 99999.0
        opponent_v_lead = 0.0 #前方車両の速度
        opponent_vehicle_id = None
        opponent_velocity_valid = False
        min_wp_diff = 99999

        if self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker'):
            startup_same_lane_candidate = None
            startup_follow_priority_active = not self._has_moved_once
            ego_start_lane_idx = (
                self._lane_index_for_position(pose.x, pose.y)
                if startup_follow_priority_active else None
            )
            for vid in self._v2x_tracker.active_vehicle_ids():
                buf = self._v2x_tracker._samples.get(vid)
                if buf:
                    _, opp_x, opp_y = buf[-1]
                    opp_wp_id = self._car.get_closest_waypoint(opp_x, opp_y)
                    rel_forward = self._center_longitudinal_between(
                        pose.x, pose.y, opp_x, opp_y)
                    if rel_forward is None:
                        continue
                    wp_diff = self._center_arc_as_waypoint_delta(rel_forward)
                    if wp_diff is None:
                        continue
                    if startup_follow_priority_active:
                        start_distance = math.hypot(opp_x - pose.x, opp_y - pose.y)
                        priority_key = startup_same_lane_lead_key(
                            vehicle_id=vid,
                            vehicle_lane_idx=self._lane_index_for_position(
                                opp_x, opp_y),
                            ego_lane_idx=ego_start_lane_idx,
                            longitudinal=rel_forward,
                            distance=start_distance,
                        )
                        candidate = (
                            priority_key, vid, opp_wp_id, opp_x, opp_y,
                        )
                        if (
                            priority_key is not None
                            and (
                                startup_same_lane_candidate is None
                                or priority_key
                                    < startup_same_lane_candidate[0]
                            )
                        ):
                            startup_same_lane_candidate = candidate
                    if 0.0 < wp_diff < 40.0:
                        if wp_diff < min_wp_diff:
                            min_wp_diff = wp_diff
                            opponent_ahead = opp_wp_id
                            opponent_vehicle_id = vid
                            opponent_arc_distance = rel_forward
                            
                            # 前方車両の横オフセットを算出
                            opp_wp = self._reference_path.get_waypoint(opp_wp_id)
                            angle_ub = opp_wp.psi + math.pi / 2.0
                            dx = opp_x - opp_wp.x
                            dy = opp_y - opp_wp.y
                            opponent_offset = dx * math.cos(angle_ub) + dy * math.sin(angle_ub)

                            # 車間距離（Euclidean距離）と前方車両の速度を取得
                            opponent_distance = math.hypot(opp_x - pose.x, opp_y - pose.y)
                            opp_vx, opp_vy = self._v2x_tracker.velocity(vid)
                            opponent_v_lead = math.hypot(opp_vx, opp_vy)
                            opponent_velocity_valid = (
                                self._v2x_tracker.has_velocity_estimate(vid))

            if startup_same_lane_candidate is not None:
                (
                    startup_priority_key,
                    opponent_vehicle_id,
                    opponent_ahead,
                    opp_x,
                    opp_y,
                ) = startup_same_lane_candidate
                start_longitudinal, opponent_distance, _ = startup_priority_key
                opponent_arc_distance = start_longitudinal
                min_wp_diff = self._center_arc_as_waypoint_delta(
                    start_longitudinal)
                opp_wp = self._reference_path.get_waypoint(opponent_ahead)
                angle_ub = opp_wp.psi + math.pi / 2.0
                opponent_offset = (
                    (opp_x - opp_wp.x) * math.cos(angle_ub)
                    + (opp_y - opp_wp.y) * math.sin(angle_ub)
                )
                opp_vx, opp_vy = self._v2x_tracker.velocity(
                    opponent_vehicle_id)
                opponent_v_lead = math.hypot(opp_vx, opp_vy)
                opponent_velocity_valid = (
                    self._v2x_tracker.has_velocity_estimate(
                        opponent_vehicle_id))
                opponent_ahead_detected = True
                if self._startup_priority_target_id != opponent_vehicle_id:
                    self._startup_priority_target_id = opponent_vehicle_id
                    self.get_logger().info(
                        "[InitialStartFollowTarget] prioritizing nearest "
                        "same-lane forward vehicle: "
                        f"vehicle_id={opponent_vehicle_id}, "
                        f"lane=L{ego_start_lane_idx}, "
                        f"longitudinal={start_longitudinal:.2f}m, "
                        f"distance={opponent_distance:.2f}m"
                    )

        # 常に追い越しを許可する
        is_overtake_zone = True

        # Calculate candidate target lane based on opponent position
        new_target_lane_idx = self._target_lane_idx  # Keep current active lane by default
        left_is_free = False
        right_is_free = False

        if (
            self._follow_only
            and opponent_ahead_detected
            and opponent_ahead is not None
        ):
            # Follow-only mode keeps the Center trajectory and constrains the
            # vehicle to L1. It must never create an L0/L2 overtake request.
            new_target_lane_idx = 1
            left_is_free = False
            right_is_free = False
        elif opponent_ahead_detected and opponent_ahead is not None:
            if is_overtake_zone:
                horizon_passage, _ = self._vehicle_passage(
                    opponent_vehicle_id, pose)
                horizon_samples = self._relative_lane_vehicle_samples(pose, v)
                horizon_conflicts = {
                    lane_idx: classify_lane_conflicts(
                        lane_idx,
                        horizon_samples,
                        front_distance=(
                            self._prepass_lane_fallback_front_distance),
                        side_distance=(
                            self._prepass_lane_fallback_side_distance),
                        rear_distance=(
                            self._prepass_lane_fallback_rear_distance),
                    )
                    for lane_idx in (0, 2)
                }
                selected_outer_lane = self._propose_traffic_lane(
                    new_target_lane_idx,
                    horizon_passage,
                    horizon_conflicts, opponent_vehicle_id, pose,
                )
                selected_outer_lane = self._apply_l2_restricted_zone_policy(
                    selected_outer_lane,
                    target_vehicle_id=opponent_vehicle_id,
                    physical_passage=horizon_passage,
                    conflicts_by_lane=horizon_conflicts,
                    center_wp=center_wp_temp,
                )
                # Do not discard a strictly verified candidate merely because
                # the high-level lane preference flips on the next geometry
                # sample.  Keep the exact target/lane long enough to enter the
                # commit branch.  Physical passage, live traffic conflicts and
                # geographic restrictions remain hard invalidation gates.
                verified_lane_idx = self._overtake.probe.lane_idx
                verified_lane_conflicts = horizon_conflicts.get(
                    verified_lane_idx, {})
                verified_lane_clear = bool(
                    verified_lane_idx in (0, 2)
                    and self._overtake.probe.confirmed
                    and self._overtake.probe.vehicle_id
                        == opponent_vehicle_id
                    and self._overtake.probe.confirmed_at is not None
                    and now_sec - self._overtake.probe.confirmed_at
                        <= self._overtake_commit_verified_candidate_hold_sec
                    and horizon_passage.get(verified_lane_idx, False)
                    and not verified_lane_conflicts.get("front", [])
                    and not verified_lane_conflicts.get("side", [])
                    and not verified_lane_conflicts.get("rear", [])
                    and self._apply_l2_restricted_zone_policy(
                        verified_lane_idx,
                        target_vehicle_id=opponent_vehicle_id,
                        physical_passage=horizon_passage,
                        conflicts_by_lane=horizon_conflicts,
                        center_wp=center_wp_temp,
                    ) == verified_lane_idx
                )
                if (
                    verified_lane_clear
                    and selected_outer_lane != verified_lane_idx
                ):
                    self.get_logger().info(
                        "[OvertakeVerifiedCandidateHold] preserving the "
                        "strictly verified candidate through high-level lane "
                        "preference changes: "
                        f"vehicle_id={opponent_vehicle_id}, "
                        f"lane=L{verified_lane_idx}, requested="
                        f"L{selected_outer_lane}",
                        throttle_duration_sec=0.5,
                    )
                    selected_outer_lane = verified_lane_idx
                left_is_free = selected_outer_lane == 2
                right_is_free = selected_outer_lane == 0
                new_target_lane_idx = (
                    selected_outer_lane
                    if selected_outer_lane in (0, 2)
                    else 1
                )
                if selected_outer_lane not in (0, 2):
                    self.get_logger().info(
                        "[OvertakeHorizonProbe] no safe outer corridor over "
                        "the prediction horizon; keeping L1: "
                        f"vehicle_id={opponent_vehicle_id}, "
                        f"physical_passage={horizon_passage}, "
                        f"conflicts={horizon_conflicts}",
                        throttle_duration_sec=1.0,
                    )
                else:
                    self.get_logger().info(
                        "[OvertakeHorizonProbe] selected safe outer corridor: "
                        f"vehicle_id={opponent_vehicle_id}, "
                        f"lane=L{selected_outer_lane}, "
                        f"physical_passage={horizon_passage}, "
                        f"conflicts={horizon_conflicts}",
                        throttle_duration_sec=1.0,
                    )
            else:
                # 追い越し不可エリア -> 中央車線 (L1) を走行して追従
                new_target_lane_idx = 1
        elif not opponent_ahead_detected:
            # 追い越しモード自体が終了した場合はターゲット車線をクリア
            new_target_lane_idx = None

        # A stopped lead selected after another pass is a new manoeuvre.  The
        # old target's sticky L0/L2 latch must not decide the passing side for
        # this vehicle. Clear all target-owned recovery state, then evaluate
        # physical passage and V2X conflicts again for the new target.
        lead_is_stationary = (
            opponent_vehicle_id is not None
            and opponent_velocity_valid
            and opponent_v_lead < self._stopped_lead_speed_threshold
        )
        lead_is_special_slow = bool(
            opponent_vehicle_id is not None
            and opponent_velocity_valid
            and self._stopped_lead_speed_threshold <= opponent_v_lead
            <= self._slow_lead_overtake_speed
        )
        slow_lead_prepare_active = bool(
            lead_is_special_slow
            and is_follow_retry_within_distance(
                opponent_arc_distance,
                self._slow_lead_overtake_prepare_distance,
            )
        )
        active_overtake_target_id = (
            self._overtake.target_id
            if self._overtake.target_id is not None
            else self._forced_overtake_vehicle_id
        )
        active_target_distance = None
        active_target_longitudinal = None
        if active_overtake_target_id is not None:
            active_buf = self._v2x_tracker._samples.get(
                active_overtake_target_id)
            if active_buf:
                _, active_opp_x, active_opp_y = active_buf[-1]
                active_rel_forward = self._center_longitudinal_between(
                    pose.x, pose.y, active_opp_x, active_opp_y)
                active_target_longitudinal = active_rel_forward
                if active_rel_forward is not None and active_rel_forward > 0.0:
                    active_target_distance = active_rel_forward
        candidate_target_is_relevant = bool(
            lead_is_stationary
            or lead_is_special_slow
            or (
                opponent_ahead_detected
                and is_follow_retry_within_distance(
                    opponent_arc_distance,
                    self._overtake_latch_max_distance,
                )
            )
        )
        # Physical completion is independent of the committed flag: an unsafe
        # lane release may clear that flag while the same pass continues under
        # Prepass ownership.
        active_target_envelopes_separated = bool(
            active_target_longitudinal is not None
            and active_target_longitudinal < 0.0
            and longitudinal_vehicle_clearance(
                active_target_longitudinal,
                self._parallel_ego_half_length,
                self._parallel_vehicle_half_length,
            ) > self._parallel_critical_clearance
        )
        active_target_has_manoeuvre_state = bool(
            self._overtake.committed
            or self._overtake.requested_lane in (0, 2)
            or self._prepass_fallback_recovery_active
            or self._prepass_fallback_commit_pending
            or self._prepass_fallback_lane_idx in (0, 1, 2)
        )
        active_target_physically_complete = bool(
            active_target_has_manoeuvre_state
            and active_target_envelopes_separated
            and active_overtake_target_id
                != self._overtake_completed_target_id
        )
        if active_target_physically_complete:
            # If another lead is already ahead in the same committed outer
            # lane, do not start L1 rejoin before giving that exact target/lane
            # a strict Shadow MPC check.  This preserves a safe continuous
            # pass while still falling back to the ordinary completion path
            # when width, live traffic, policy, or Shadow feasibility fails.
            handoff_lane_idx = (
                int(self._overtake.requested_lane)
                if self._overtake.requested_lane in (0, 2) else None
            )
            handoff_candidate = bool(
                handoff_lane_idx in (0, 2)
                and opponent_vehicle_id is not None
                and opponent_vehicle_id != active_overtake_target_id
                and candidate_target_is_relevant
                and 0.0 < float(opponent_arc_distance)
                <= self._slow_lead_overtake_prepare_distance
            )
            handoff_passage = {}
            handoff_conflicts = {}
            handoff_lane_clear = False
            if handoff_candidate:
                handoff_passage, _ = self._vehicle_passage(
                    opponent_vehicle_id, pose)
                handoff_samples = self._relative_lane_vehicle_samples(pose, v)
                handoff_conflicts = classify_lane_conflicts(
                    handoff_lane_idx,
                    handoff_samples,
                    front_distance=self._prepass_lane_fallback_front_distance,
                    side_distance=self._prepass_lane_fallback_side_distance,
                    rear_distance=self._prepass_lane_fallback_rear_distance,
                )
                # The completed vehicle is expected immediately behind in the
                # same corridor. It is already envelope-separated above; only
                # unrelated rear traffic invalidates the handoff.
                unrelated_rear = [
                    vehicle_id
                    for vehicle_id in handoff_conflicts.get("rear", [])
                    if vehicle_id != active_overtake_target_id
                ]
                policy_lane = self._apply_l2_restricted_zone_policy(
                    handoff_lane_idx,
                    target_vehicle_id=opponent_vehicle_id,
                    physical_passage=handoff_passage,
                    conflicts_by_lane={
                        handoff_lane_idx: {
                            **handoff_conflicts,
                            "rear": unrelated_rear,
                        }
                    },
                    center_wp=center_wp_temp,
                )
                handoff_lane_clear = bool(
                    handoff_passage.get(handoff_lane_idx, False)
                    and not handoff_conflicts.get("front", [])
                    and not handoff_conflicts.get("side", [])
                    and not unrelated_rear
                    and policy_lane == handoff_lane_idx
                )

            handoff_pending = bool(
                handoff_lane_clear
                and self._consecutive_overtake_handoff_target_id
                    == opponent_vehicle_id
                and self._consecutive_overtake_handoff_lane_idx
                    == handoff_lane_idx
            )
            if handoff_lane_clear and not handoff_pending:
                self._consecutive_overtake_handoff_target_id = (
                    opponent_vehicle_id)
                self._consecutive_overtake_handoff_lane_idx = handoff_lane_idx
                self._consecutive_overtake_handoff_started_at = now_sec
                self._prepare_overtake_commit_probe(
                    opponent_vehicle_id, handoff_lane_idx)
                handoff_pending = True

            handoff_confirmed = bool(
                handoff_pending
                and self._overtake_commit_probe_is_fresh(
                    opponent_vehicle_id, handoff_lane_idx, now_sec)
            )
            handoff_elapsed = (
                now_sec - self._consecutive_overtake_handoff_started_at
                if (
                    handoff_pending
                    and self._consecutive_overtake_handoff_started_at
                        is not None
                ) else 0.0
            )
            if handoff_confirmed:
                old_target_id = active_overtake_target_id
                if self._same_lane_target_handoff_available(opponent_vehicle_id, pose, v):
                    self._accept_same_lane_target_handoff(opponent_vehicle_id, handoff_lane_idx)
                else:
                    self._preserve_stationary_group_hybrid(
                        opponent_vehicle_id, handoff_lane_idx, pose, v, now_sec)
                self._overtake.target_id = opponent_vehicle_id
                self._forced_overtake_vehicle_id = None
                self._overtake.requested_lane = handoff_lane_idx
                self._overtake.committed = True
                self._overtake_completed_target_id = None
                self._outer_lane_released_vehicle_id = None
                self._overtake.verification.vehicle_id = opponent_vehicle_id
                self._overtake.verification.lane_idx = handoff_lane_idx
                self._reset_outer_lane_progress()
                self._clear_consecutive_overtake_handoff()
                active_overtake_target_id = opponent_vehicle_id
                active_target_longitudinal = float(opponent_arc_distance)
                active_target_distance = float(opponent_arc_distance)
                candidate_target_is_relevant = False
                active_target_physically_complete = False
                self.get_logger().info(
                    "[ConsecutiveOvertakeHandoff] strict Shadow verification "
                    "passed for the next lead; preserving the current outer "
                    "lane without L1 rejoin: "
                    f"old_vehicle_id={old_target_id}, "
                    f"new_vehicle_id={opponent_vehicle_id}, "
                    f"lane=L{handoff_lane_idx}, "
                    f"distance={opponent_arc_distance:.2f}m"
                )
            elif (
                handoff_pending
                and handoff_elapsed
                    <= self._consecutive_overtake_handoff_timeout_sec
            ):
                self._prepare_overtake_commit_probe(
                    opponent_vehicle_id, handoff_lane_idx)
                candidate_target_is_relevant = False
                self.get_logger().info(
                    "[ConsecutiveOvertakeHandoffHold] keeping the current "
                    "outer lane while strictly probing the next lead: "
                    f"old_vehicle_id={active_overtake_target_id}, "
                    f"new_vehicle_id={opponent_vehicle_id}, "
                    f"lane=L{handoff_lane_idx}, elapsed="
                    f"{handoff_elapsed:.2f}/"
                    f"{self._consecutive_overtake_handoff_timeout_sec:.2f}s, "
                    f"cycles={self._overtake.probe.success_cycles}/"
                    f"{self._overtake_commit_probe_required_success_cycles}",
                    throttle_duration_sec=0.25,
                )
                active_target_physically_complete = False
            else:
                self._clear_consecutive_overtake_handoff()

        if active_target_physically_complete:
            # The pass itself is complete even though Center/Race handoff may
            # still be pending.  Release the *artificial* outer-lane bound in
            # this cycle so full-width MPC plus the soft L1 reference can
            # begin bringing the vehicle back immediately.  Keep the target
            # ID/lane only as manoeuvre metadata; a confirmed different lead
            # may still replace it through the target-switch path below.
            #
            # Merely clearing ``_overtake_target_committed`` is insufficient:
            # select_latched_overtake_lane() preserves an existing L0/L2
            # latch, which used to re-apply the completed target's lane until
            # the larger distance/behind release gate fired.
            self._complete_overtake_target_behind(
                active_overtake_target_id,
                active_target_longitudinal,
                source=(
                    "committed"
                    if self._overtake.committed
                    else "lane-unsafe/prepass"
                ),
            )
        elif active_overtake_target_id is None:
            self._overtake.committed = False
            self._overtake_completed_target_id = None
        elif (
            active_overtake_target_id != self._overtake_completed_target_id
            and self._reference_path.is_overtaking
            and self._reference_path.target_lane_idx in (0, 2)
            and float(now_sec) >= getattr(
                self, "_constraint_transition_until", 0.0)
        ):
            self._overtake.committed = True
        overtake_target_locked = bool(
            active_overtake_target_id is not None
            and self._overtake.committed)
        if (
            overtake_target_locked
            and opponent_vehicle_id is not None
            and opponent_vehicle_id != active_overtake_target_id
            and candidate_target_is_relevant
        ):
            self.get_logger().info(
                "[OvertakeTargetLock] keeping active target; another vehicle "
                "remains monitored only by ParallelSafety/EmergencyBrake: "
                f"active={active_overtake_target_id}, "
                f"candidate={opponent_vehicle_id}",
                throttle_duration_sec=0.5,
            )
        urgent_locked_target_switch = False
        if (
            overtake_target_locked
            and opponent_vehicle_id is not None
            and opponent_vehicle_id != active_overtake_target_id
            and candidate_target_is_relevant
            and active_target_distance is not None
            and math.isfinite(float(active_target_distance))
            and math.isfinite(float(opponent_arc_distance))
            and opponent_arc_distance <= (
                active_target_distance
                - self._overtake_target_switch_margin_m)
            and opponent_arc_distance <= self._overtake_latch_max_distance
            and self._overtake.requested_lane in (0, 2)
        ):
            locked_lane_conflicts = classify_lane_conflicts(
                self._overtake.requested_lane,
                self._relative_lane_vehicle_samples(pose, v),
                front_distance=self._prepass_lane_fallback_front_distance,
                side_distance=self._prepass_lane_fallback_side_distance,
                rear_distance=self._prepass_lane_fallback_rear_distance,
            )
            urgent_locked_target_switch = bool(
                opponent_vehicle_id
                in (
                    locked_lane_conflicts.get("front", [])
                    + locked_lane_conflicts.get("side", [])
                )
            )
            if urgent_locked_target_switch:
                self.get_logger().warn(
                    "[OvertakeTargetUrgentSwitch] a clearly closer candidate "
                    "is entering the committed corridor; allowing confirmed "
                    "target replacement despite the manoeuvre lock: "
                    f"active={active_overtake_target_id}, "
                    f"candidate={opponent_vehicle_id}, "
                    f"active_arc={active_target_distance:.2f}m, "
                    f"candidate_arc={opponent_arc_distance:.2f}m, "
                    f"lane=L{self._overtake.requested_lane}, "
                    f"conflicts={locked_lane_conflicts}",
                    throttle_duration_sec=0.5,
                )

        switch_candidate_eligible = should_reset_overtake_latch_for_target_change(
            active_target_id=active_overtake_target_id,
            candidate_target_id=opponent_vehicle_id,
            candidate_is_relevant=candidate_target_is_relevant,
            active_target_distance=active_target_distance,
            candidate_target_distance=opponent_arc_distance,
            switch_margin_m=self._overtake_target_switch_margin_m,
            target_locked=(
                overtake_target_locked and not urgent_locked_target_switch),
        )
        immediate_close_target_switch = bool(
            opponent_vehicle_id is not None
            and opponent_vehicle_id != active_overtake_target_id
            and candidate_target_is_relevant
            and active_target_distance is not None
            and math.isfinite(float(active_target_distance))
            and math.isfinite(float(opponent_arc_distance))
            and 0.0 < float(opponent_arc_distance)
            <= self._overtake_target_immediate_switch_distance
            and float(opponent_arc_distance) <= (
                float(active_target_distance)
                - self._overtake_target_switch_margin_m)
        )
        if immediate_close_target_switch:
            switch_candidate_eligible = True
            self.get_logger().warn(
                "[OvertakeTargetImmediateSwitch] a much closer forward "
                "vehicle entered the emergency acquisition range; bypassing "
                "the ordinary target confirmation: "
                f"active={active_overtake_target_id}, "
                f"candidate={opponent_vehicle_id}, "
                f"active_arc={active_target_distance:.2f}m, "
                f"candidate_arc={opponent_arc_distance:.2f}m/"
                f"{self._overtake_target_immediate_switch_distance:.2f}m"
            )
        urgent_switch_confirmed = False
        if urgent_locked_target_switch:
            if (
                self._urgent_overtake_switch_candidate_id
                != opponent_vehicle_id
            ):
                self._urgent_overtake_switch_candidate_id = (
                    opponent_vehicle_id)
                self._urgent_overtake_switch_candidate_since = now_sec
            self._urgent_overtake_switch_last_seen_at = now_sec
            urgent_confirmed_for = (
                now_sec - self._urgent_overtake_switch_candidate_since
                if self._urgent_overtake_switch_candidate_since is not None
                else 0.0
            )
            urgent_switch_confirmed = bool(
                urgent_confirmed_for
                >= self._urgent_overtake_switch_confirm_sec
            )
        else:
            urgent_last_seen = self._urgent_overtake_switch_last_seen_at
            urgent_dropout_expired = bool(
                urgent_last_seen is None
                or now_sec - urgent_last_seen
                > self._urgent_overtake_switch_dropout_grace_sec
            )
            if urgent_dropout_expired:
                self._clear_urgent_overtake_switch_candidate()

        target_switch_confirmed = bool(
            urgent_switch_confirmed or immediate_close_target_switch)
        if (
            switch_candidate_eligible
            and not urgent_locked_target_switch
            and not immediate_close_target_switch
        ):
            if self._overtake_switch_candidate_id != opponent_vehicle_id:
                self._overtake_switch_candidate_id = opponent_vehicle_id
                self._overtake_switch_candidate_since = now_sec
                self.get_logger().info(
                    "[OvertakeTargetSwitchCandidate] new closer target must "
                    "remain stable before replacing the active target: "
                    f"active={active_overtake_target_id}, "
                    f"candidate={opponent_vehicle_id}, confirm_sec="
                    f"{self._overtake_target_switch_confirm_sec:.2f}")
            confirmed_for = (
                now_sec - self._overtake_switch_candidate_since
                if self._overtake_switch_candidate_since is not None
                else 0.0)
            target_switch_confirmed = bool(
                confirmed_for >= self._overtake_target_switch_confirm_sec)
        elif not urgent_locked_target_switch:
            self._overtake_switch_candidate_id = None
            self._overtake_switch_candidate_since = None

        same_lane_handoff = False
        if (target_switch_confirmed
                and self._same_lane_target_handoff_available(opponent_vehicle_id, pose, v)):
            handoff_lane = self._overtake.requested_lane
            if self._overtake_commit_probe_is_fresh(opponent_vehicle_id, handoff_lane, now_sec):
                same_lane_handoff = self._accept_same_lane_target_handoff(
                    opponent_vehicle_id, handoff_lane)
                if same_lane_handoff:
                    active_overtake_target_id = opponent_vehicle_id
                    active_target_longitudinal = opponent_arc_distance
                    active_target_distance = opponent_arc_distance
                    active_target_physically_complete = False
                    new_target_lane_idx = handoff_lane
            else:
                self._prepare_overtake_commit_probe(opponent_vehicle_id, handoff_lane)
                target_switch_confirmed = False
                self.get_logger().info(
                    f"[SameLaneTargetProbe] keeping lane=L{handoff_lane} and spatial anchor "
                    f"while verifying successor={opponent_vehicle_id}", throttle_duration_sec=0.5)

        if target_switch_confirmed and not same_lane_handoff:
            old_target_id = active_overtake_target_id
            self._reset_overtake_state_for_target_change(
                opponent_vehicle_id,
                reason=(
                    "closer forward vehicle entered immediate acquisition "
                    "range"
                    if immediate_close_target_switch
                    else (
                        "urgent committed-corridor conflict remained closer "
                        "for 0.2s"
                        if urgent_switch_confirmed
                        else "new relevant lead remained clearly closer"
                    )
                ),
            )
            target_passage, _ = self._vehicle_passage(
                opponent_vehicle_id, pose)
            target_samples = self._relative_lane_vehicle_samples(pose, v)
            target_conflicts = {
                lane_idx: classify_lane_conflicts(
                    lane_idx,
                    target_samples,
                    front_distance=self._prepass_lane_fallback_front_distance,
                    side_distance=self._prepass_lane_fallback_side_distance,
                    rear_distance=self._prepass_lane_fallback_rear_distance,
                )
                for lane_idx in (0, 2)
            }
            new_target_lane_idx = self._propose_traffic_lane(
                new_target_lane_idx,
                target_passage,
                target_conflicts, opponent_vehicle_id, pose,
            )
            new_target_lane_idx = self._apply_l2_restricted_zone_policy(
                new_target_lane_idx,
                target_vehicle_id=opponent_vehicle_id,
                physical_passage=target_passage,
                conflicts_by_lane=target_conflicts,
                center_wp=center_wp_temp,
            )
            if new_target_lane_idx not in (0, 2):
                new_target_lane_idx = 1
            self.get_logger().warn(
                "[OvertakeTargetReevaluate] selected a fresh passing decision "
                f"for vehicle_id={opponent_vehicle_id} after replacing "
                f"vehicle_id={old_target_id}: lane=L{new_target_lane_idx}, "
                f"physical_passage={target_passage}, "
                f"conflicts={target_conflicts}"
            )

        # A slow vehicle may be a candidate while another overtake target is
        # still locked. Do not let that unrelated candidate reduce speed until
        # target acquisition is free or the switch is confirmed.
        slow_lead_speed_control_active = bool(
            slow_lead_prepare_active
            and (
                active_overtake_target_id is None
                or opponent_vehicle_id == active_overtake_target_id
                or target_switch_confirmed
            )
        )

        # An ordinary L0/L2 latch must not survive solely because its MPC is
        # still feasible.  Once the latched lead is ahead but outside the
        # metric overtake gate, release the hard outer-lane constraint in this
        # cycle.  Keep the target ID/lane as manoeuvre metadata so the existing
        # pass/Race-return state machines can finish without recreating a new
        # latch for the same vehicle.
        normal_outer_gate_state = self._latched_follow_target_state(
            pose, now_sec)
        normal_outer_gate_excluded = (
            recovery_active
            or self._post_reverse_full_width_recovery_active
            or self._mpc_safety_recovery_active
            or self._prepass_fallback_recovery_active
            or self._prepass_fallback_commit_pending
            or self._prepass_fallback_follow_active
            or self._follow_escape_active
            or self._parallel_abort_active
            or startup_overtake_suppressed
        )
        if (
            normal_outer_gate_state is not None
            and not normal_outer_gate_state.get("expired", False)
            and not normal_outer_gate_excluded
        ):
            self._update_outer_lane_progress_state(
                target_id=self._overtake.target_id,
                longitudinal=normal_outer_gate_state.get("longitudinal"),
                pose=pose,
                ego_speed=v,
                now_sec=now_sec,
            )
        else:
            # Recovery and other exclusive owners suspend this watchdog. A
            # fresh timer is armed only after ordinary outer-lane control and
            # lateral convergence resume.
            self._reset_outer_lane_progress()
        if (
            normal_outer_gate_state is not None
            and not normal_outer_gate_state.get("expired", False)
            and not normal_outer_gate_excluded
            and should_release_active_overtake_distance_gate(
                outer_lane_active=(
                    self._overtake.target_id is not None
                    and self._overtake.requested_lane in (0, 2)
                    and self._outer_lane_released_vehicle_id is None
                ),
                target_longitudinal=normal_outer_gate_state.get(
                    "longitudinal"),
                distance=normal_outer_gate_state.get("longitudinal"),
                max_distance=self._overtake_release_distance,
            )
        ):
            released_target_id = self._overtake.target_id
            released_lane_idx = self._overtake.requested_lane
            released_distance = float(
                normal_outer_gate_state["longitudinal"])
            self._outer_lane_released_vehicle_id = released_target_id
            self._forced_overtake_vehicle_id = None
            self._prepass_fallback_lane_idx = None
            self._prepass_fallback_blocked = False
            self._prepass_follow_last_retry_at = None
            self._prepass_fallback_recovery_stable_since = None
            self._prepass_fallback_recovery_started_at = None
            self._prepass_target_behind_since = None
            self._prepass_failed_lane_idx = None
            self._prepass_fallback_commit_success_since = None
            self._prepass_attempted_outer_lanes.clear()
            self._prepass_retry_after_reverse = False
            self._prepass_retry_lane_idx = None
            self._prepass_reverse_motion_started = False
            self._prepass_reverse_start_xy = None
            self._prepass_reverse_distance = 0.0
            self._close_obstacle_reverse_requested = False
            self._mpc.osqp_initialized = False
            self.get_logger().info(
                "[OvertakeDistanceGateRelease] target exited the ordinary "
                "arc-length overtake gate; releasing the outer-lane constraint and "
                "starting full-width L1 soft rejoin: "
                f"vehicle_id={released_target_id}, lane=L{released_lane_idx}, "
                f"distance={released_distance:.2f}m/"
                f"{self._overtake_release_distance:.2f}m"
            )

        if self._parallel_abort_active and not recovery_active:
            self._release_stationary_parallel_abort(pose, v)
        if self._parallel_abort_active and not recovery_active:
            abort_vehicle_active = (
                self._parallel_abort_vehicle_id
                in self._v2x_tracker.active_vehicle_ids()
            )
            abort_buf = (
                self._v2x_tracker._samples.get(
                    self._parallel_abort_vehicle_id)
                if abort_vehicle_active else None
            )
            abort_vehicle_safely_ahead = abort_buf is None
            if abort_buf:
                _, abort_x, abort_y = abort_buf[-1]
                abort_longitudinal = self._center_longitudinal_between(
                    pose.x, pose.y, abort_x, abort_y)
                abort_vehicle_safely_ahead = (
                    abort_longitudinal is not None
                    and abort_longitudinal
                    >= self._prepass_lane_fallback_front_distance
                )
            if abort_vehicle_safely_ahead:
                self.get_logger().info(
                    "[ParallelAbort] release: parallel vehicle is safely ahead "
                    f"or no longer tracked: vehicle_id="
                    f"{self._parallel_abort_vehicle_id}"
                )
                self._parallel_abort_active = False
                self._parallel_abort_vehicle_id = None
                self._parallel_abort_target_lane_idx = None
                self._parallel_timer_vehicle_id = None
                self._parallel_start_time = None

        # Evaluate the geographic hard-lane problem zone before any selector
        # can create a new L0/L2 latch or advance its shadow probe.  A fresh,
        # identified stopped/ultra-slow target is the only exception.  It may
        # enter the normal selector below, but still has to pass physical
        # passage, traffic, strict Shadow MPC and 20 m corridor-width gates.
        outer_lane_mpc_problem_zone = any(
            (
                start_wp <= center_wp_temp <= end_wp
                if start_wp <= end_wp
                else center_wp_temp >= start_wp or center_wp_temp <= end_wp
            )
            for start_wp, end_wp in self._outer_lane_mpc_problem_zones
        )
        # Latch the low-speed classification at 5 km/h and release it above
        # 6 km/h.  This must be computed here (rather than only in the final
        # corridor resolver) so the exception can create a new verified latch.
        # A missing target ID or velocity estimate can never enable it.
        if not outer_lane_mpc_problem_zone or opponent_vehicle_id is None:
            self._outer_lane_problem_slow_override_target_id = None
        elif (
            self._outer_lane_problem_slow_override_target_id
                == opponent_vehicle_id
        ):
            if (
                not opponent_velocity_valid
                or opponent_v_lead
                    > self._outer_lane_problem_override_release_speed
            ):
                self._outer_lane_problem_slow_override_target_id = None
        elif (
            opponent_velocity_valid
            and opponent_v_lead <= self._outer_lane_problem_override_speed
        ):
            self._outer_lane_problem_slow_override_target_id = (
                opponent_vehicle_id)
        outer_lane_problem_slow_override = (
            outer_lane_problem_slow_override_active(
                latched_target_id=(
                    self._outer_lane_problem_slow_override_target_id),
                opponent_vehicle_id=opponent_vehicle_id,
                velocity_valid=opponent_velocity_valid,
            )
        )
        prepass_selection_exclusive = (
            recovery_active
            or self._post_reverse_full_width_recovery_active
            or self._follow_escape_active
            or prepass_recovery_owns_lane_selection(
                self._prepass_fallback_recovery_active,
                self._prepass_fallback_commit_pending,
            )
        )
        exclusive_l1_rejoin = (
            not prepass_selection_exclusive
            and (
                self._center_lane_rejoin_active
                or self._l1_safety_reprobe_pending
                or self._l1_rejoin_backoff_active
                or (
                    self._l1_probe_active
                    and self._l1_probe_context == "rejoin"
                )
            )
        )
        # L1 rejoin normally owns lateral selection so an old passing-side
        # latch cannot immediately reappear. A newly detected nearby vehicle is
        # different: when an outer lane is physically passable and clear of
        # current/predicted V2X traffic, continuing to force L1 can remove a
        # valid passing opportunity. Allow that verified candidate to preempt
        # normal L1 rejoin regardless of the lead vehicle's stopped state.
        if (
            exclusive_l1_rejoin
            and (
                not outer_lane_mpc_problem_zone
                or outer_lane_problem_slow_override
            )
            and not self._follow_only
            and not startup_overtake_suppressed
            and not self._parallel_abort_active
            and l1_rejoin_preemption_target_relevant(
                opponent_ahead_detected=opponent_ahead_detected,
                lead_is_stationary=bool(opponent_velocity_valid and lead_is_stationary),
                lead_is_special_slow=bool(opponent_velocity_valid and lead_is_special_slow),
                opponent_vehicle_id=opponent_vehicle_id,
                opponent_arc_distance=opponent_arc_distance,
                maximum_distance=self._overtake_latch_max_distance)
            and self._mpc.infeasibility_counter == 0
            and self._mpc.current_prediction is not None
            and not self._mpc_safety_recovery_active
        ):
            preempt_passage, _ = self._vehicle_passage(
                opponent_vehicle_id, pose)
            preempt_samples = self._relative_lane_vehicle_samples(pose, v)
            preempt_conflicts = {
                lane_idx: classify_lane_conflicts(
                    lane_idx,
                    preempt_samples,
                    front_distance=self._prepass_lane_fallback_front_distance,
                    side_distance=self._prepass_lane_fallback_side_distance,
                    rear_distance=self._prepass_lane_fallback_rear_distance,
                )
                for lane_idx in (0, 2)
            }
            preempt_lane_idx = self._propose_traffic_lane(
                new_target_lane_idx,
                preempt_passage,
                preempt_conflicts, opponent_vehicle_id, pose,
            )
            preempt_lane_idx = self._apply_l2_restricted_zone_policy(
                preempt_lane_idx,
                target_vehicle_id=opponent_vehicle_id,
                physical_passage=preempt_passage,
                conflicts_by_lane=preempt_conflicts,
                center_wp=center_wp_temp,
            )
            if preempt_lane_idx in (0, 2):
                self._cancel_l1_rejoin_for_overtake()
                exclusive_l1_rejoin = False
                new_target_lane_idx = preempt_lane_idx
                self._mpc.osqp_initialized = False
                self.get_logger().warn(
                    "[L1RejoinOvertakePreempt] nearby forward vehicle has a "
                    "safe outer passing lane; cancelling L1 rejoin and "
                    f"selecting L{preempt_lane_idx}: vehicle_id="
                    f"{opponent_vehicle_id}, arc={opponent_arc_distance:.2f}m/"
                    f"{self._overtake_latch_max_distance:.2f}m, "
                    f"physical_passage={preempt_passage}, "
                    f"conflicts={preempt_conflicts}"
                )
        if recovery_active:
            # Freeze all lateral selection while shifting/reversing.  The
            # applied constraint is released to full width below, but no
            # latch, probe, or fallback decision may advance in this state.
            overtake_latch_started = False
            new_target_lane_idx = None
        elif self._prepass_fallback_recovery_active:
            # Prepass exclusively owns the lateral state from the moment the
            # failed outer lane is released until it selects a fallback or
            # switches to follow/reverse.  Do not let the ordinary selector
            # recreate L0/L2 latches during full-width recovery.
            overtake_latch_started = False
            new_target_lane_idx = None
        elif self._prepass_fallback_commit_pending:
            # Recovery has selected a concrete fallback.  Keep ordinary L1
            # rejoin and L0/L2 selection suspended until this lane has really
            # been applied to MPC and produced a feasible solution.
            overtake_latch_started = False
            new_target_lane_idx = self._prepass_fallback_commit_lane_idx
        elif self._follow_escape_active:
            # A stopped-follow escape owns lateral selection until a safe
            # forward probe commits or the existing reverse sequence starts.
            overtake_latch_started = False
            new_target_lane_idx = self._follow_escape_probe_lane_idx
        elif (
            outer_lane_mpc_problem_zone
            and not outer_lane_problem_slow_override
            and self._consecutive_overtake_handoff_target_id is None
        ):
            # Do not create a new hard L0/L2 owner in a section known to make
            # outer-lane MPC constraints collapse. Only an identified target
            # inside the 5/6 km/h hysteresis may reach the normal verified
            # selector below. The final zone policy releases any already-
            # existing ordinary latch to full width.
            self._reset_overtake_commit_probe()
            overtake_latch_started = False
            new_target_lane_idx = None
        elif self._parallel_abort_active:
            # Yield owns selection. Do not prepare a Shadow that release_target
            # below would discard before its execution.
            overtake_latch_started = False
            new_target_lane_idx = self._parallel_abort_target_lane_idx
        elif exclusive_l1_rejoin:
            # L1 rejoin exclusively owns lateral selection. Calling the outer
            # lane selector here would recreate an L0/L2 latch every cycle,
            # only for the block below to clear it again in the same cycle.
            overtake_latch_started = False
        elif startup_overtake_suppressed:
            # During the initial boost, keep the Center trajectory mode but do
            # not create an ordinary L0/L2 overtake latch.  This remains true
            # when hold_l0_during_boost is disabled for the full-width test.
            overtake_latch_started = False
            self._overtake.requested_lane = None
            self._outer_lane_released_vehicle_id = None
        elif self._follow_only:
            overtake_latch_started = False
            self._overtake.release_target()
            self._outer_lane_released_vehicle_id = None
            self._prepass_fallback_lane_idx = None
            self._prepass_fallback_blocked = False
            self._prepass_fallback_follow_active = False
            self._prepass_fallback_recovery_active = False
            self._prepass_fallback_commit_pending = False
            self._prepass_fallback_commit_lane_idx = None
        else:
            # Race->Center remains controlled by enter_center_wps.  Only the
            # creation of a *new* L0/L2 latch is delayed until the selected
            # opponent is inside the metric distance gate.  An already active
            # latch is preserved here and is released by the existing pass and
            # Prepass state machines.
            existing_outer_latch = self._overtake.requested_lane in (0, 2)
            local_slow_lead_commit_distance = (
                self._slow_lead_commit_distance_at(
                    center_wp_temp, lead_speed=opponent_v_lead,
                    lead_is_stationary=lead_is_stationary)
            )
            new_latch_distance = (
                local_slow_lead_commit_distance
                if lead_is_stationary or lead_is_special_slow
                else self._overtake_latch_max_distance
            )
            new_latch_within_distance = is_follow_retry_within_distance(
                opponent_arc_distance,
                new_latch_distance,
            )
            latch_candidate_vehicle_id = opponent_vehicle_id
            latch_candidate_lane_idx = new_target_lane_idx
            if (
                not existing_outer_latch
                and new_target_lane_idx in (0, 2)
                and not new_latch_within_distance
            ):
                early_shadow_ready = bool(
                    lead_is_special_slow
                    and 0.0 < float(opponent_arc_distance)
                    <= self._slow_lead_overtake_prepare_distance
                )
                if early_shadow_ready:
                    # Submit_0829 approached decisively as soon as its horizon
                    # check found a clear outer lane. Keep that useful timing,
                    # but replace the loose horizon result with the current
                    # strict Shadow checks. No hard lane is applied here.
                    self._prepare_overtake_commit_probe(
                        opponent_vehicle_id, new_target_lane_idx)
                else:
                    self._reset_overtake_commit_probe()
                # Stay on Center/L1 while the high-level opponent-ahead mode
                # is active. A verified early Shadow result may release speed
                # matching, but commitment still waits for the distance gate.
                latch_candidate_vehicle_id = None
                latch_candidate_lane_idx = 1
                self.get_logger().info(
                    "[OvertakeLatchDistanceHold] opponent detected but outside "
                    "new-latch gate; keeping L1: "
                    f"vehicle_id={opponent_vehicle_id}, "
                    f"arc={opponent_arc_distance:.2f}m/"
                    f"{new_latch_distance:.2f}m, "
                    f"early_shadow={early_shadow_ready}, cycles="
                    f"{self._overtake.probe.success_cycles}/"
                    f"{self._overtake_commit_probe_required_success_cycles}",
                    throttle_duration_sec=1.0,
                )
            elif (
                not existing_outer_latch
                and new_target_lane_idx in (0, 2)
            ):
                commit_gate = evaluate_overtake_commit_gate(
                    target_lane=new_target_lane_idx,
                    target_distance=opponent_arc_distance,
                    preview_curvatures=(
                        self._overtake_commit_curvature_preview(
                            center_wp_temp)
                    ),
                    minimum_distance=self._overtake_commit_min_distance,
                    maximum_distance=new_latch_distance,
                    outside_curvature_threshold=(
                        self._overtake_outside_curvature_threshold),
                )
                slow_lead_curve_override = bool(
                    lead_is_stationary or lead_is_special_slow)
                commit_allowed = bool(
                    commit_gate["allowed"]
                    or (
                        slow_lead_curve_override
                        and 0.0 < float(opponent_arc_distance)
                        <= float(new_latch_distance)
                    )
                )
                if (
                    commit_allowed
                    and slow_lead_curve_override
                    and not commit_gate["allowed"]
                ):
                    self.get_logger().info(
                        "[OvertakeCommitSlowLeadOverride] stopped/slow lead "
                        "has a verified clear corridor; allowing the outer "
                        "pass despite the ordinary distance/curve hold: "
                        f"vehicle_id={opponent_vehicle_id}, "
                        f"speed={opponent_v_lead:.2f}m/s, "
                        f"lane=L{new_target_lane_idx}, "
                        f"arc={opponent_arc_distance:.2f}m, "
                        f"outside_kappa="
                        f"{commit_gate['outside_curvature']:.3f}/"
                        f"{self._overtake_outside_curvature_threshold:.3f}1/m",
                        throttle_duration_sec=1.0,
                    )
                if not commit_allowed:
                    self._reset_overtake_commit_probe()
                    latch_candidate_vehicle_id = None
                    latch_candidate_lane_idx = 1
                    reasons = []
                    if not commit_gate["distance_ready"]:
                        reasons.append("distance_outside_commit_band")
                    if not commit_gate["curvature_ready"]:
                        reasons.append("outside_curve_too_tight")
                    self.get_logger().info(
                        "[OvertakeCommitHold] safe passage exists, but the "
                        "new outer-lane commitment is held until the simple "
                        "distance/curve gate is ready: "
                        f"vehicle_id={opponent_vehicle_id}, "
                        f"lane=L{new_target_lane_idx}, "
                        f"arc={opponent_arc_distance:.2f}m/"
                        f"[{self._overtake_commit_min_distance:.2f},"
                        f"{new_latch_distance:.2f}]m, "
                        f"outside_kappa="
                        f"{commit_gate['outside_curvature']:.3f}/"
                        f"{self._overtake_outside_curvature_threshold:.3f}1/m, "
                        f"reasons={reasons}",
                        throttle_duration_sec=1.0,
                    )
                elif not self._overtake_commit_probe_is_fresh(
                    opponent_vehicle_id,
                    new_target_lane_idx,
                    float(now.nanoseconds) / 1e9,
                ):
                    self._prepare_overtake_commit_probe(
                        opponent_vehicle_id, new_target_lane_idx)
                    latch_candidate_vehicle_id = None
                    latch_candidate_lane_idx = 1
                    self.get_logger().info(
                        "[OvertakeCommitProbeHold] ordinary gates passed; "
                        "keeping L1 until shadow MPC and 20m width checks "
                        "are confirmed: "
                        f"vehicle_id={opponent_vehicle_id}, "
                        f"lane=L{new_target_lane_idx}, cycles="
                        f"{self._overtake.probe.success_cycles}/"
                        f"{self._overtake_commit_probe_required_success_cycles}",
                        throttle_duration_sec=0.25)
            (
                new_target_lane_idx,
                self._overtake.target_id,
                self._overtake.requested_lane,
                overtake_latch_started,
            ) = select_latched_overtake_lane(
                # A metric slow-lead preemption must reach the latch writer
                # even when the waypoint-only detector remains false. The
                # candidate above is still withheld until strict Shadow passes.
                (opponent_ahead_detected or (
                    opponent_velocity_valid
                    and (lead_is_stationary or lead_is_special_slow)
                    and is_follow_retry_within_distance(
                        opponent_arc_distance, new_latch_distance)))
                and not self._parallel_abort_active,
                latch_candidate_vehicle_id,
                (
                    # ``_prepass_fallback_lane_idx == 1`` only records that
                    # the previous fallback successfully reached L1.  It must
                    # not permanently mask a newly verified L0/L2 corridor;
                    # overtake_latch_started clears that stale fallback state
                    # below.  Only the explicit follow state owns L1 here.
                    1 if self._prepass_fallback_follow_active
                    else latch_candidate_lane_idx
                ),
                self._overtake.target_id,
                self._overtake.requested_lane,
            )
        if (
            self._parallel_abort_active
            and not prepass_selection_exclusive
        ):
            # This state exclusively owns lateral selection until the parallel
            # vehicle has moved safely ahead. Do not let an outer-lane latch,
            # prepass fallback, or L1 rejoin overwrite it.
            new_target_lane_idx = self._parallel_abort_target_lane_idx
            if new_target_lane_idx not in (0, 1, 2):
                new_target_lane_idx = 1
            self._overtake.release_target()
            self._outer_lane_released_vehicle_id = None
            self._prepass_fallback_lane_idx = None
            self._prepass_fallback_blocked = False
            self._prepass_fallback_follow_active = False
            self._prepass_fallback_recovery_active = False
            self._prepass_fallback_commit_pending = False
            self._prepass_fallback_commit_lane_idx = None
            self._prepass_retry_after_reverse = False
            self._center_lane_rejoin_active = False
            self._center_lane_rejoin_constraint_released = False
            self._center_lane_rejoin_stable_since = None
            if not self._l1_safety_reprobe_pending:
                self._l1_probe_active = False
                self._l1_probe_context = None
                self._l1_probe_success_cycles = 0
                self._l1_probe_constraint_applied = False
        # Once the post-pass L1 rejoin starts, it owns the lateral state.
        # Drop every outer-lane latch so the old passing-side release cannot
        # compete with the L1 infeasibility release later in the same manoeuvre.
        if exclusive_l1_rejoin:
            had_overtake_latch = (
                self._overtake.target_id is not None
                or self._overtake.requested_lane in (0, 2)
                or self._prepass_fallback_lane_idx is not None
                or self._prepass_fallback_recovery_active
            )
            if (
                self._l1_probe_context != "fallback"
                and self._l1_safety_recovery_context != "fallback"
            ):
                self._overtake.release_target()
                self._outer_lane_released_vehicle_id = None
                self._prepass_fallback_lane_idx = None
                self._prepass_fallback_blocked = False
                self._prepass_fallback_recovery_active = False
                self._prepass_fallback_recovery_stable_since = None
            new_target_lane_idx = 1
            overtake_latch_started = False
            if had_overtake_latch:
                self.get_logger().info(
                    "[OvertakeLatch] released for exclusive L1 rejoin."
                )
        if overtake_latch_started:
            shadow_verified_latch = self._overtake_commit_probe_is_fresh(
                self._overtake.target_id,
                self._overtake.requested_lane,
                float(now.nanoseconds) / 1e9,
            )
            if shadow_verified_latch:
                self._overtake.verification.vehicle_id = (
                    self._overtake.target_id)
                self._overtake.verification.lane_idx = int(
                    self._overtake.requested_lane)
            elif not (
                self._overtake.verification.vehicle_id
                    == self._overtake.target_id
                and self._overtake.verification.lane_idx
                    == self._overtake.requested_lane
            ):
                self._clear_committed_shadow_verification()
            committed_shadow_retained = bool(
                self._overtake.verification.vehicle_id
                    == self._overtake.target_id
                and self._overtake.verification.lane_idx
                    == self._overtake.requested_lane
            )
            # The live committed-lane monitor owns safety from this point.
            # Stop spending a second solve on the pre-commit Shadow instance.
            self._reset_overtake_commit_probe()
            self._center_lane_rejoin_active = False
            self._center_lane_rejoin_constraint_released = False
            self._center_lane_rejoin_stable_since = None
            self._l1_probe_active = False
            self._l1_probe_context = None
            self._l1_probe_success_cycles = 0
            self._l1_probe_constraint_applied = False
            self._outer_lane_released_vehicle_id = None
            self._prepass_fallback_lane_idx = None
            self._prepass_fallback_blocked = False
            self._prepass_fallback_follow_active = False
            self._prepass_retry_after_reverse = False
            self._prepass_retry_lane_idx = None
            self._prepass_reverse_motion_started = False
            self._prepass_reverse_start_xy = None
            self._prepass_reverse_distance = 0.0
            self._follow_latched_cache = None
            self._prepass_fallback_recovery_active = False
            self._prepass_fallback_recovery_stable_since = None
            self._prepass_fallback_commit_pending = False
            self._prepass_fallback_commit_lane_idx = None
            lane_label = "L0" if self._overtake.requested_lane == 0 else "L2"
            self.get_logger().info(
                "[OvertakeLatch] fixed passing side: "
                f"vehicle_id={self._overtake.target_id}, "
                f"lane={lane_label}, "
                f"shadow_verified={committed_shadow_retained}"
            )

        current_time_sec = float(now.nanoseconds) / 1e9
        self._update_localization_consistency(current_time_sec)
        prepass_distance_gate_released_this_cycle = False

        retry_target_id = self._overtake.target_id
        retry_target_is_slow = False
        if retry_target_id is not None:
            retry_vx, retry_vy = self._v2x_tracker.velocity(retry_target_id)
            retry_target_is_slow = bool(
                self._v2x_tracker.has_velocity_estimate(retry_target_id)
                and math.hypot(retry_vx, retry_vy)
                    <= self._slow_lead_overtake_speed
            )
        follow_retry_sec = (
            self._slow_lead_overtake_retry_sec
            if retry_target_is_slow else self._prepass_follow_retry_sec
        )
        follow_retry_max_distance = (
            self._slow_lead_overtake_prepare_distance
            if retry_target_is_slow
            else self._prepass_follow_retry_max_distance
        )

        if (
            not self._follow_only
            and not recovery_active
            and not startup_overtake_suppressed
            and should_reevaluate_follow_overtake(
                self._prepass_fallback_follow_active,
                current_time_sec,
                self._prepass_follow_last_retry_at,
                follow_retry_sec,
            )
        ):
            self._prepass_follow_last_retry_at = current_time_sec
            follow_retry_state = self._latched_follow_target_state(
                pose, current_time_sec)
            if (
                follow_retry_state is not None
                and not follow_retry_state.get("expired", False)
                and is_follow_target_ahead(
                    follow_retry_state.get("longitudinal"))
            ):
                follow_retry_distance = float(
                    follow_retry_state.get("longitudinal", math.inf))
                follow_retry_commit_distance = self._slow_lead_commit_distance_at(
                    center_wp_temp, lead_speed=follow_retry_state.get("speed"))
                if not is_follow_retry_within_distance(
                    follow_retry_distance,
                    follow_retry_max_distance,
                ):
                    self.get_logger().info(
                        "[PrepassFollowRetryHold] periodic follow "
                        "re-evaluation is outside the overtake distance gate; "
                        f"keeping follow: vehicle_id="
                        f"{self._overtake.target_id}, "
                        f"target_arc={follow_retry_distance:.2f}m/"
                        f"{follow_retry_max_distance:.2f}m"
                    )
                elif (
                    retry_target_is_slow
                    and follow_retry_distance
                        > follow_retry_commit_distance
                ):
                    self.get_logger().info(
                        "[SlowLeadOvertakePrepare] evaluating at high "
                        "frequency while preserving the gap; outer-lane "
                        "commit waits for the distance gate: vehicle_id="
                        f"{retry_target_id}, arc="
                        f"{follow_retry_distance:.2f}m/"
                        f"{follow_retry_commit_distance:.2f}m, "
                        f"retry={follow_retry_sec:.2f}s",
                        throttle_duration_sec=1.0,
                    )
                else:
                    (
                        follow_retry_lane_idx,
                        _,
                        follow_retry_conflicts,
                    ) = self._select_prepass_retry_lane(
                        pose, v, self._overtake.requested_lane)
                    follow_retry_passage, _ = self._latched_target_passage(pose)
                    if follow_retry_lane_idx in (0, 2):
                        self._cancel_normal_l1_rejoin_for_prepass()
                        self._prepass_fallback_follow_active = False
                        self._prepass_fallback_blocked = False
                        self._prepass_fallback_recovery_active = True
                        self._prepass_fallback_recovery_stable_since = None
                        self._prepass_fallback_recovery_started_at = (
                            current_time_sec)
                        self._prepass_target_behind_since = None
                        # Make the newly verified lane the opposite-first choice
                        # when recovery becomes dynamically stable.
                        self._prepass_failed_lane_idx = (
                            2 if follow_retry_lane_idx == 0 else 0)
                        self._overtake.requested_lane = follow_retry_lane_idx
                        self._prepass_fallback_lane_idx = None
                        self._prepass_fallback_commit_pending = False
                        self._prepass_fallback_commit_lane_idx = None
                        self._prepass_fallback_commit_success_since = None
                        self._prepass_attempted_outer_lanes.clear()
                        self._mpc.infeasibility_counter = 0
                        self._mpc.osqp_initialized = False
                        self.get_logger().warn(
                            "[PrepassFollowRetry] periodic follow re-evaluation "
                            "found a clear outer lane inside the distance gate; "
                            "returning to full-width recovery before probing "
                            f"L{follow_retry_lane_idx}: vehicle_id="
                            f"{self._overtake.target_id}, "
                            f"target_arc={follow_retry_distance:.2f}m, "
                            f"physical_passage={follow_retry_passage}, "
                            f"conflicts={follow_retry_conflicts}"
                        )
                    else:
                        self.get_logger().info(
                            "[PrepassFollowRetryHold] periodic follow "
                            "re-evaluation found no clear outer lane; keeping "
                            f"follow: vehicle_id="
                            f"{self._overtake.target_id}, "
                            f"target_arc={follow_retry_distance:.2f}m, "
                            f"physical_passage={follow_retry_passage}, "
                            f"conflicts={follow_retry_conflicts}"
                        )

        if (
            (lead_is_stationary or lead_is_special_slow)
            and not recovery_active
            and not startup_overtake_suppressed
            and not self._follow_only
            and not self._prepass_fallback_follow_active
            and not self._parallel_abort_active
        ):
            if self._forced_overtake_vehicle_id != opponent_vehicle_id:
                action = (
                    "forcing immediate stopped-vehicle overtake"
                    if lead_is_stationary and (left_is_free or right_is_free)
                    else "prioritizing <=10km/h lead overtake with speed matching"
                    if left_is_free or right_is_free
                    else "no safe passing side; keeping stop/reverse fallback armed"
                    if lead_is_stationary
                    else "no safe passing side; keeping ordinary follow active"
                )
                log_tag = (
                    "[StoppedVehicle] " if lead_is_stationary
                    else "[SlowLeadOvertakeAssist] "
                )
                self.get_logger().warn(
                    f"{log_tag}vehicle_id={opponent_vehicle_id} "
                    f"speed={opponent_v_lead:.2f}m/s; {action}.",
                    throttle_duration_sec=1.0,
                )
            self._forced_overtake_vehicle_id = opponent_vehicle_id
        elif self._prepass_fallback_follow_active and not recovery_active:
            # A timed-out overtake is now ordinary longitudinal following,
            # including following a stopped lead. Do not re-arm forced
            # overtaking or its close-obstacle reverse fallback.
            self._forced_overtake_vehicle_id = None

        if (
            not recovery_active
            and self._forced_overtake_vehicle_id is not None
            and opponent_vehicle_id != self._forced_overtake_vehicle_id
        ):
            forced_id = self._forced_overtake_vehicle_id
            active_ids = self._v2x_tracker.active_vehicle_ids()
            forced_buf = self._v2x_tracker._samples.get(forced_id)
            forced_vehicle_passed = forced_id not in active_ids
            if forced_buf:
                _, forced_x, forced_y = forced_buf[-1]
                forced_lon = self._center_longitudinal_between(
                    pose.x, pose.y, forced_x, forced_y)
                forced_vehicle_passed = (
                    forced_lon is not None and forced_lon < -1.0)
            if forced_vehicle_passed:
                self.get_logger().info(
                    f"[StoppedVehicle] vehicle_id={forced_id} "
                    "forced overtake completed."
                )
                self._forced_overtake_vehicle_id = None

        prev_lane_idx = self._target_lane_idx

        latched_target_longitudinal = None
        if not self._prepass_fallback_recovery_active:
            self._prepass_dynamic_conflict_speed_limit = None
        latched_target_id = self._overtake.target_id
        if latched_target_id is not None and hasattr(self, '_v2x_tracker'):
            target_buf = self._v2x_tracker._samples.get(latched_target_id)
            if target_buf:
                _, target_x, target_y = target_buf[-1]
                latched_target_longitudinal = self._center_longitudinal_between(
                    pose.x, pose.y, target_x, target_y)

        # A latch prevents chatter, but it must not freeze a passing side after
        # the target (or another predicted vehicle) moves into that corridor.
        # D2 changes lanes while being approached, so waiting for five failed
        # MPC cycles lets the ego reach collision distance before Prepass
        # releases the lane. Re-evaluate the *latched* lane continuously and
        # enter full-width recovery immediately; the opposite side is applied
        # later only when it is independently verified.
        if (
            latched_target_id is not None
            and self._overtake.requested_lane in (0, 2)
            and (latched_target_longitudinal is None
                 or not math.isfinite(float(latched_target_longitudinal))
                 or latched_target_longitudinal >= -(
                     self._parallel_ego_half_length + self._parallel_vehicle_half_length
                     + self._parallel_safety_longitudinal_clearance))
            and not recovery_active
            and not self._prepass_fallback_recovery_active
            and not self._prepass_fallback_commit_pending
            and not self._prepass_fallback_follow_active
            and not self._parallel_abort_active
            and not self._l1_probe_active
            and not self._center_lane_rejoin_active
            and not self._center_lane_rejoin_constraint_released
            and not self._l1_safety_recovery_active
            and not self._l1_rejoin_backoff_active
        ):
            latched_lane_idx = int(self._overtake.requested_lane)
            opposite_lane_idx = 2 if latched_lane_idx == 0 else 0
            live_passage, _ = self._latched_target_passage(pose)
            live_samples = self._relative_lane_vehicle_samples(pose, v)
            live_conflicts = {
                lane_idx: classify_lane_conflicts(
                    lane_idx,
                    live_samples,
                    front_distance=self._prepass_lane_fallback_front_distance,
                    side_distance=self._prepass_lane_fallback_side_distance,
                    rear_distance=self._prepass_lane_fallback_rear_distance,
                )
                for lane_idx in (0, 2)
            }
            committed_conflicts = live_conflicts[latched_lane_idx]
            target_reported_in_corridor = any(
                latched_target_id in committed_conflicts.get(kind, [])
                for kind in ("front", "side", "rear")
            )
            non_target_corridor_conflicts = {
                kind: [
                    vehicle_id
                    for vehicle_id in committed_conflicts.get(kind, [])
                    if vehicle_id != latched_target_id
                ]
                for kind in ("front", "side", "rear")
            }
            # Immediately after Shadow MPC commits an outer lane, the live
            # prediction still belongs to L1/full-width while the tapered
            # Soft Transition is in progress.  The pass target naturally
            # collides with that old prediction, so using it here cancels the
            # verified manoeuvre before the outer corridor is ever applied.
            # Trust Shadow MPC for the target vehicle until the committed lane
            # is actually applied and the transition window has ended.  Live
            # width loss, body overlap and unsafe/unknown unrelated traffic
            # remain immediate release conditions below.
            committed_lane_prediction_established = bool(
                self._reference_path.is_overtaking
                and self._reference_path.target_lane_idx == latched_lane_idx
                and current_time_sec >= getattr(
                    self, "_constraint_transition_until", 0.0)
            )
            target_prediction_collision = (
                self._prediction_collision_with_vehicle(latched_target_id)
                if (
                    target_reported_in_corridor
                    and committed_lane_prediction_established
                )
                else False
            )
            if (
                target_reported_in_corridor
                and not committed_lane_prediction_established
            ):
                self.get_logger().info(
                    "[OvertakeSoftTransitionTargetHold] ignoring the live "
                    "L1/full-width prediction for the Shadow-verified pass "
                    "target until the committed outer-lane prediction is "
                    f"established: vehicle_id={latched_target_id}, "
                    f"lane=L{latched_lane_idx}",
                    throttle_duration_sec=0.5,
                )
            # Keep the post-commit check consistent with
            # L0ZoneFollowGeometryOverride.  Passage includes clearance for
            # overtaking the target, so it can become false while L0 remains a
            # valid lane in which to follow that same target.  Do not depend on
            # target_reported_in_corridor here: that classification can briefly
            # be empty on a curve even though _latched_target_passage() failed
            # because of the target.  Real lane-width loss and unrelated
            # unsafe/unknown traffic still release immediately, including rear traffic.
            l0_follow_geometry_override_active = bool(
                latched_lane_idx == 0
                and l0_restricted_follow_can_ignore_passage(
                    restriction_active=self._waypoint_in_configured_zones(
                        int(self._carN_center.wp_id),
                        self._l2_entry_restricted_zones,
                    ),
                    lane_has_vehicle_width=(
                        self._lane_horizon_has_vehicle_width(0)
                    ),
                    non_target_conflicts=non_target_corridor_conflicts,
                )
            )
            hold_decision = self._evaluate_committed_lane_hold(
                pose=pose, ego_speed=v, lane_idx=latched_lane_idx,
                target_id=latched_target_id, target_longitudinal=latched_target_longitudinal,
                live_passage=live_passage, conflicts=non_target_corridor_conflicts,
                now_sec=current_time_sec, legacy_target_hold=l0_follow_geometry_override_active)
            physical_passage_lost = hold_decision.physical_passage_lost
            latched_lane_unsafe = hold_decision.unsafe
            if target_prediction_collision is True:
                # The target is expected to overlap the ego's longitudinal
                # prediction while it is being passed.  Cancelling the
                # Shadow-verified corridor here turns a valid pass back into
                # slow following.  Emergency/parallel safety still owns the
                # speed. Width loss, body overlap and unsafe/unknown unrelated
                # traffic remain immediate corridor-release conditions above.
                self.get_logger().info(
                    "[OvertakeTargetPredictionSpeedOnly] committed target "
                    "intersects the live prediction; retaining the verified "
                    "passing lane and delegating separation to speed safety: "
                    f"vehicle_id={latched_target_id}, lane=L{latched_lane_idx}",
                    throttle_duration_sec=0.5,
                )
            opposite_lane_safe = bool(
                live_passage.get(opposite_lane_idx, False)
                and lane_conflicts_are_clear(
                    live_conflicts[opposite_lane_idx])
            )
            if latched_lane_unsafe:
                self._overtake.clear_hybrid()
                # The retained Shadow result only certifies the corridor at
                # commit time. Any subsequent geometry, traffic, or stalled
                # target-collision release invalidates that proof.
                self._clear_committed_shadow_verification()
                # Limit against the vehicle which actually entered the
                # corridor, not necessarily the latched overtake target. Rear
                # traffic requires a lateral release but must not make ego
                # brake. When the opposite side is already clear, retain a
                # useful closing-speed margin while changing lanes.
                blocking_vehicle_ids = set(
                    live_conflicts[latched_lane_idx].get("front", [])
                ) | set(
                    live_conflicts[latched_lane_idx].get("side", [])
                )
                blocking_speeds = []
                for blocking_vehicle_id in blocking_vehicle_ids:
                    blocker_vx, blocker_vy = self._v2x_tracker.velocity(
                        blocking_vehicle_id)
                    blocking_speeds.append(math.hypot(blocker_vx, blocker_vy))
                if blocking_speeds:
                    closing_margin = 1.5 if opposite_lane_safe else 0.5
                    self._prepass_dynamic_conflict_speed_limit = max(
                        max(blocking_speeds) + closing_margin,
                        1.5,
                    )
                else:
                    self._prepass_dynamic_conflict_speed_limit = None
                self._cancel_normal_l1_rejoin_for_prepass()
                self._prepass_failed_lane_idx = latched_lane_idx
                self._prepass_fallback_lane_idx = None
                self._prepass_fallback_blocked = False
                self._prepass_fallback_recovery_active = True
                self._prepass_fallback_recovery_stable_since = None
                self._prepass_fallback_recovery_started_at = current_time_sec
                self._prepass_target_behind_since = None
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_lane_idx = None
                self._prepass_fallback_commit_success_since = None
                self._prepass_attempted_outer_lanes.clear()
                if physical_passage_lost:
                    # A geometry/width loss is not permission to cross the
                    # track toward the other outer lane.  Keep full width now;
                    # subsequent recovery may select L1, but neither outer
                    # lane is a candidate for this recovery episode.
                    self._prepass_attempted_outer_lanes.update((0, 2))
                self._mpc.osqp_initialized = False
                self.get_logger().warn(
                    "[OvertakeLatchedLaneUnsafe] committed passing corridor "
                    "became unsafe; releasing it immediately to full width: "
                    f"vehicle_id={latched_target_id}, "
                    f"failed_lane=L{latched_lane_idx}, "
                    "candidate_lane="
                    f"{'none' if physical_passage_lost else ('L' + str(opposite_lane_idx) if opposite_lane_safe else 'none')}, "
                    f"physical_passage_lost={physical_passage_lost}, "
                    f"longitudinal={latched_target_longitudinal}m, "
                    f"physical_passage={live_passage}, "
                    f"conflicts={live_conflicts}"
                )
        else:
            self._overtake.passage_hold.reset()

        prepass_target_tracking_active = bool(
            self._prepass_fallback_recovery_active
            or self._prepass_fallback_commit_pending
            or self._prepass_fallback_lane_idx in (0, 1, 2)
        )
        target_is_confirmably_behind = bool(
            prepass_target_tracking_active
            and latched_target_longitudinal is not None
            and latched_target_longitudinal < 0.0
            and longitudinal_vehicle_clearance(
                latched_target_longitudinal,
                self._parallel_ego_half_length,
                self._parallel_vehicle_half_length,
            ) > self._parallel_critical_clearance
        )
        self._prepass_target_behind_since = update_continuous_condition_since(
            self._prepass_target_behind_since,
            now_sec=current_time_sec,
            condition=target_is_confirmably_behind,
        )
        if continuous_condition_confirmed(
            self._prepass_target_behind_since,
            now_sec=current_time_sec,
            confirm_sec=self._prepass_behind_release_confirm_sec,
        ):
            behind_elapsed = current_time_sec - self._prepass_target_behind_since
            released_target_id = self._overtake.target_id
            self._complete_overtake_target_behind(
                released_target_id,
                latched_target_longitudinal,
                source="prepass-behind-confirmation",
            )
            self.get_logger().info(
                "[PrepassBehindRelease] confirmed target behind ego; "
                "ending candidate search and entering soft L1 rejoin: "
                f"vehicle_id={released_target_id}, "
                f"longitudinal={latched_target_longitudinal:.2f}m, "
                f"confirm_sec={behind_elapsed:.2f}/"
                f"{self._prepass_behind_release_confirm_sec:.2f}"
            )

        prepass_fallback_triggered = (
            self._prepass_lane_fallback_enabled
            and not recovery_active
            and not startup_overtake_suppressed
            and opponent_ahead_detected
            and self._overtake.requested_lane in (0, 2)
            and self._prepass_fallback_lane_idx is None
            and not self._prepass_fallback_blocked
            and not self._prepass_fallback_follow_active
            and not self._prepass_fallback_recovery_active
            and latched_target_longitudinal is not None
            and latched_target_longitudinal >= 0.0
            and self._mpc.infeasibility_counter
                >= self._prepass_lane_fallback_infeasible_cycles
        )
        if prepass_fallback_triggered:
            self._cancel_normal_l1_rejoin_for_prepass()
            self._prepass_attempted_outer_lanes.clear()
            self._prepass_fallback_recovery_active = True
            self._prepass_fallback_recovery_stable_since = None
            self._prepass_fallback_recovery_started_at = current_time_sec
            self._prepass_target_behind_since = None
            self._prepass_failed_lane_idx = self._overtake.requested_lane
            self._prepass_fallback_commit_pending = False
            self._prepass_fallback_commit_lane_idx = None
            self.get_logger().warn(
                "[PrepassLaneFallbackRecovery] fixed lane became infeasible; "
                "releasing lane constraint to full width before selecting "
                f"a fallback: lane=L{self._overtake.requested_lane}, "
                f"mpc_infeasible={self._mpc.infeasibility_counter}"
            )

        if (
            self._prepass_fallback_recovery_active
            and not recovery_active
            and not self._post_reverse_full_width_recovery_active
        ):
            failed_lane_idx = (
                self._prepass_failed_lane_idx
                if self._prepass_failed_lane_idx in (0, 2)
                else self._overtake.requested_lane
            )
            if self._prepass_fallback_recovery_started_at is None:
                self._prepass_fallback_recovery_started_at = current_time_sec
            recovery_total_elapsed = (
                current_time_sec
                - self._prepass_fallback_recovery_started_at
            )
            (
                recovery_candidate_lane,
                latched_target_distance,
                passage_conflicts,
                recovery_physical_passage,
            ) = self._select_prepass_fallback_lane(
                pose,
                v,
                failed_lane_idx,
            )
            self._latch_prepass_soft_candidate(
                recovery_candidate_lane,
                current_time_sec,
            )
            if should_release_prepass_distance_gate(
                recovery_active=self._prepass_fallback_recovery_active,
                distance=(
                    latched_target_longitudinal
                    if latched_target_longitudinal is not None
                    and latched_target_longitudinal >= 0.0
                    else None
                ),
                max_distance=self._overtake_release_distance,
            ):
                released_target_id = self._overtake.target_id
                released_distance = float(latched_target_longitudinal)
                prepass_distance_gate_released_this_cycle = True
                self._release_lost_follow_target(
                    reason=(
                        "active Prepass target exited overtake distance gate: "
                        f"arc={released_distance:.2f}m, "
                        f"gate=<{self._overtake_release_distance:.2f}m"
                    )
                )
                self.get_logger().info(
                    "[PrepassDistanceGateRelease] target left the overtake "
                    "gate during full-width recovery; ending Prepass "
                    "immediately: "
                    f"vehicle_id={released_target_id}, "
                    f"target_arc={released_distance:.2f}m/"
                    f"{self._overtake_release_distance:.2f}m, "
                    f"elapsed={recovery_total_elapsed:.2f}s"
                )
            candidate_heading = self._candidate_lane_heading(
                recovery_candidate_lane, wp)
            candidate_heading_error = absolute_heading_difference(
                pose.theta, candidate_heading)
            candidate_lateral_speed = max(
                measured_lateral_speed,
                abs(float(v) * math.sin(candidate_heading_error)),
            )
            prepass_mpc_stable = (
                self._mpc.infeasibility_counter == 0
                and self._mpc.current_prediction is not None
                and not getattr(self._mpc, "used_prediction_fallback", False)
                and not self._mpc_safety_recovery_active
            )
            prepass_full_width_prediction_safe = bool(
                prepass_mpc_stable
                and (
                    self._overtake.target_id is None
                    or self._prediction_is_clear_of_vehicle(
                        self._overtake.target_id)
                )
            )
            prepass_heading_stable = (
                recovery_candidate_lane is not None
                and candidate_heading_error <= self._prepass_max_heading
            )
            prepass_dynamics_stable = (
                candidate_lateral_speed
                    <= self._center_lane_rejoin_max_lateral_speed
                and center_yaw_rate <= self._center_lane_rejoin_max_yaw_rate
            )

            (
                ordinary_timeout_expired,
                soft_guidance_watchdog_expired,
            ) = prepass_recovery_timeout_expired(
                elapsed=recovery_total_elapsed,
                recovery_timeout=(
                    self._prepass_lane_fallback_recovery_timeout_sec),
                soft_guidance_timeout=self._prepass_soft_guidance_timeout_sec,
                full_width_prediction_safe=prepass_full_width_prediction_safe,
            )
            if (
                self._prepass_fallback_recovery_active
                and (
                    ordinary_timeout_expired
                    or soft_guidance_watchdog_expired
                )
            ):
                self._prepass_fallback_recovery_active = False
                self._prepass_fallback_recovery_stable_since = None
                self._prepass_fallback_recovery_started_at = None
                timeout_target_id = self._overtake.target_id
                retry_lane_idx = recovery_candidate_lane
                timeout_physical_passage = recovery_physical_passage
                timeout_target_distance = latched_target_distance
                reverse_rear_clear = self._reverse_rear_is_clear(pose, v)
                timeout_follow_state = self._latched_follow_target_state(
                    pose, current_time_sec)
                timeout_target_longitudinal = (
                    timeout_follow_state.get("longitudinal")
                    if timeout_follow_state is not None
                    and not timeout_follow_state.get("expired", False)
                    else None
                )
                distance_within_gate = is_follow_retry_within_distance(
                    timeout_target_longitudinal,
                    self._prepass_follow_retry_max_distance,
                )
                candidate_outer_lanes = tuple(
                    lane_idx for lane_idx in passage_conflicts
                    if lane_idx in (0, 2)
                )
                physical_passage_available = any(
                    timeout_physical_passage.get(lane_idx, False)
                    for lane_idx in candidate_outer_lanes
                )
                traffic_clear = any(
                    timeout_physical_passage.get(lane_idx, False)
                    and lane_conflicts_are_clear(passage_conflicts[lane_idx])
                    for lane_idx in candidate_outer_lanes
                )
                timeout_reasons = classify_prepass_timeout_reasons(
                    mpc_stable=prepass_mpc_stable,
                    heading_stable=prepass_heading_stable,
                    dynamics_stable=prepass_dynamics_stable,
                    physical_passage_available=physical_passage_available,
                    traffic_clear=traffic_clear,
                    distance_within_gate=distance_within_gate,
                    target_behind=(
                        timeout_target_longitudinal is not None
                        and not is_follow_target_ahead(
                            timeout_target_longitudinal)
                    ),
                )

                if (
                    timeout_target_longitudinal is not None
                    and not is_follow_target_ahead(timeout_target_longitudinal)
                ):
                    self._release_lost_follow_target(
                        reason=(
                            "latched vehicle is behind ego "
                            f"(longitudinal={timeout_target_longitudinal:.2f}m)"
                        )
                    )
                    action = "latched target is behind; releasing without follow"
                elif not distance_within_gate:
                    self._release_lost_follow_target(
                        reason=(
                            "Prepass timeout target is outside overtake "
                            "distance gate: "
                            f"arc={timeout_target_longitudinal}, "
                            f"gate=<{self._prepass_follow_retry_max_distance:.2f}m"
                        )
                    )
                    action = (
                        "target is outside 10m overtake gate; releasing for "
                        "full-traffic re-evaluation"
                    )
                elif retry_lane_idx is None:
                    self._switch_prepass_to_follow(
                        "all fallback candidates are unsafe for the latched target"
                    )
                    action = "no feasible fallback candidate; switching to follow"
                elif retry_lane_idx == 1:
                    self._prepass_fallback_blocked = False
                    self._prepass_fallback_follow_active = False
                    self._prepass_fallback_lane_idx = None
                    self._prepass_fallback_commit_pending = False
                    self._prepass_fallback_commit_lane_idx = None
                    self._l1_probe_active = True
                    self._l1_probe_context = "fallback"
                    self._l1_probe_success_cycles = 0
                    self._l1_probe_constraint_applied = False
                    action = "outer lanes unavailable; probing L1"
                elif (
                    latched_target_distance is not None
                    and latched_target_distance
                        <= self._close_obstacle_reverse_distance
                    and reverse_rear_clear
                ):
                    self._prepass_fallback_blocked = True
                    self._prepass_fallback_follow_active = False
                    self._prepass_retry_after_reverse = True
                    self._prepass_retry_lane_idx = retry_lane_idx
                    self._prepass_reverse_motion_started = False
                    self._prepass_reverse_start_xy = None
                    self._prepass_reverse_distance = 0.0
                    action = (
                        f"passage L{retry_lane_idx} exists but gap is only "
                        f"{latched_target_distance:.2f}m; reversing before retry"
                    )
                elif (
                    latched_target_distance is None
                    or latched_target_distance
                        <= self._close_obstacle_reverse_distance
                ):
                    self._switch_prepass_to_follow(
                        "latched target is missing or rear corridor is occupied"
                    )
                    action = "passage exists but rear is occupied; switching to follow"
                else:
                    self._prepass_fallback_blocked = False
                    self._prepass_fallback_follow_active = False
                    self._prepass_fallback_lane_idx = retry_lane_idx
                    self._overtake.requested_lane = retry_lane_idx
                    self._prepass_attempted_outer_lanes.add(retry_lane_idx)
                    self._prepass_fallback_commit_pending = True
                    self._prepass_fallback_commit_lane_idx = retry_lane_idx
                    self._prepass_fallback_commit_success_since = None
                    self._mpc.infeasibility_counter = 0
                    self._mpc.osqp_initialized = False
                    action = f"passage L{retry_lane_idx} exists; retrying in place"

                self.get_logger().warn(
                    "[PrepassLaneFallbackTimeout] full-width recovery timed "
                    f"out; {action}: elapsed={recovery_total_elapsed:.2f}s, "
                    f"reasons={list(timeout_reasons)}, "
                    f"candidate_lane=L{recovery_candidate_lane}, "
                    f"candidate_heading_error="
                    f"{math.degrees(candidate_heading_error):.1f}deg/"
                    f"{math.degrees(self._prepass_max_heading):.1f}deg, "
                    f"mpc_stable={prepass_mpc_stable}, "
                    f"dynamics_stable={prepass_dynamics_stable}, "
                    f"vehicle_id={timeout_target_id}, "
                    f"target_distance={timeout_target_distance}, "
                    f"target_longitudinal={timeout_target_longitudinal}, "
                    f"physical_passage={timeout_physical_passage}, "
                    f"conflicts={passage_conflicts}, "
                    f"reverse_rear_clear={reverse_rear_clear}"
                )
                self._prepass_failed_lane_idx = None

            # Full-width recovery must be stable both numerically and
            # dynamically before committing to another narrow lane. Heading
            # and estimated lateral speed are evaluated against the selected
            # candidate lane geometry, not the Center reference heading.
            prepass_recovery_stable = (
                self._prepass_fallback_recovery_active
                and prepass_mpc_stable
                and (
                    not self._center_lane_rejoin_stability_enabled
                    or (
                        prepass_heading_stable
                        and prepass_dynamics_stable
                    )
                )
            )
            if prepass_recovery_stable:
                if self._prepass_fallback_recovery_stable_since is None:
                    self._prepass_fallback_recovery_stable_since = current_time_sec
                recovery_stable_elapsed = (
                    current_time_sec
                    - self._prepass_fallback_recovery_stable_since
                )
            else:
                self._prepass_fallback_recovery_stable_since = None
                recovery_stable_elapsed = 0.0
                if self._prepass_fallback_recovery_active:
                    self.get_logger().info(
                        "[PrepassLaneFallbackHold] keeping full width until "
                        "vehicle state is stable: "
                        f"candidate_lane=L{recovery_candidate_lane}, "
                        f"candidate_heading_error="
                        f"{math.degrees(candidate_heading_error):.1f}deg/"
                        f"{math.degrees(self._prepass_max_heading):.1f}, "
                        f"lateral_speed={candidate_lateral_speed:.2f}/"
                        f"{self._center_lane_rejoin_max_lateral_speed:.2f}m/s, "
                        f"yaw_rate={center_yaw_rate:.2f}/"
                        f"{self._center_lane_rejoin_max_yaw_rate:.2f}rad/s, "
                        f"mpc_infeasible={self._mpc.infeasibility_counter}",
                        throttle_duration_sec=1.0,
                    )

            if (
                self._prepass_fallback_recovery_active
                and recovery_stable_elapsed
                >= self._prepass_lane_fallback_recovery_stable_sec
            ):
                selected_fallback_lane = recovery_candidate_lane
                candidate_conflicts = passage_conflicts
                physical_passage = recovery_physical_passage

                self._prepass_fallback_recovery_active = False
                self._prepass_fallback_recovery_stable_since = None
                self._prepass_fallback_recovery_started_at = None
                if selected_fallback_lane is not None:
                    previous_lane = failed_lane_idx
                    if selected_fallback_lane == 1:
                        self._prepass_fallback_lane_idx = None
                        self._prepass_fallback_commit_pending = False
                        self._prepass_fallback_commit_lane_idx = None
                        self._l1_probe_active = True
                        self._l1_probe_context = "fallback"
                        self._l1_probe_success_cycles = 0
                        self._l1_probe_constraint_applied = False
                    else:
                        self._prepass_fallback_lane_idx = selected_fallback_lane
                        self._overtake.requested_lane = selected_fallback_lane
                        self._prepass_attempted_outer_lanes.add(
                            selected_fallback_lane)
                        self._prepass_fallback_commit_pending = True
                        self._prepass_fallback_commit_lane_idx = (
                            selected_fallback_lane)
                        self._prepass_fallback_commit_success_since = None
                    fallback_action = (
                        "probing"
                        if selected_fallback_lane == 1
                        else (
                            "re-probing"
                            if selected_fallback_lane == failed_lane_idx
                            else "switching"
                        )
                    )
                    self.get_logger().warn(
                        "[PrepassLaneFallback] full-width recovery stable; "
                        f"{fallback_action} "
                        f"L{previous_lane}->L{selected_fallback_lane}, "
                        f"stable_sec={recovery_stable_elapsed:.2f}, "
                        f"physical_passage={physical_passage}, "
                        f"conflicts={candidate_conflicts}"
                    )
                else:
                    self._switch_prepass_to_follow(
                        "all fallback lanes are unsafe after full-width recovery"
                    )
                    self.get_logger().warn(
                        "[PrepassLaneFallback] full-width recovery stable but "
                        "both fallback lanes are unsafe; switching to "
                        f"follow: physical_passage={physical_passage}, "
                        f"conflicts={candidate_conflicts}"
                    )
                self._prepass_failed_lane_idx = None

        # Check if vehicle is in or near a curve based on waypoint ranges (when following centerline)
        is_curve_locked = False
        if self._curve_lane_lock_enabled:
            # Check if currently following centerline
            if getattr(self, '_opponent_ahead_detected', False):
                is_in_curve = False
                N_wps = self._reference_path.n_waypoints
                reversed_wp = N_wps - 1 - wp
                for r in self._curve_lane_lock_wps:
                    if len(r) == 2:
                        start, end = r[0], r[1]
                        if start <= end:
                            if start <= reversed_wp <= end:
                                is_in_curve = True
                                break
                        else:
                            if reversed_wp >= start or reversed_wp <= end:
                                is_in_curve = True
                                break
                
                if is_in_curve:
                    is_curve_locked = True
                    if prev_lane_idx is None:
                        new_target_lane_idx = 1  # Force center lane (L1) to restrict corridor
                    else:
                        new_target_lane_idx = prev_lane_idx  # Stay in the current lane

        release_outer_lane_constraint = should_release_latched_overtake_lane(
            overtake_active=(
                opponent_ahead_detected
                and not self._center_lane_rejoin_active
                and not self._l1_probe_active
                and not self._prepass_fallback_follow_active
                and not self._prepass_fallback_recovery_active
            ),
            latched_vehicle_id=self._overtake.target_id,
            latched_lane_idx=self._overtake.requested_lane,
            target_longitudinal_distance=latched_target_longitudinal,
            infeasibility_counter=self._mpc.infeasibility_counter,
            behind_distance=self._outer_lane_release_behind_m,
            infeasible_cycles=self._outer_lane_release_infeasible_cycles,
        )
        if release_outer_lane_constraint:
            if self._outer_lane_released_vehicle_id != latched_target_id:
                lane_label = "L0" if self._overtake.requested_lane == 0 else "L2"
                self.get_logger().warn(
                    "[OvertakeLaneRelease] releasing outer-lane constraint "
                    f"while keeping Center: vehicle_id={latched_target_id}, "
                    f"lane={lane_label}, "
                    f"longitudinal={latched_target_longitudinal:.2f}m, "
                    f"mpc_infeasible={self._mpc.infeasibility_counter}"
                )
            self._outer_lane_released_vehicle_id = latched_target_id

        outer_lane_constraint_released = (
            self._outer_lane_released_vehicle_id is not None
            and self._outer_lane_released_vehicle_id
                == self._overtake.target_id
        )
        if outer_lane_constraint_released:
            # Keep the Center trajectory and target ID, but let MPC use the
            # full track width until the normal Race-return conditions pass.
            new_target_lane_idx = None

        if prepass_distance_gate_released_this_cycle:
            # Release the old outer-lane constraint in this same cycle; do not
            # let a lane request computed before the Prepass gate check survive
            # until the next control period.
            new_target_lane_idx = None
        elif self._prepass_fallback_follow_active:
            new_target_lane_idx = None
        elif l1_probe_failed_this_cycle:
            new_target_lane_idx = None
        elif self._l1_probe_active:
            new_target_lane_idx = 1
        elif self._prepass_fallback_recovery_active:
            new_target_lane_idx = None
        elif self._prepass_fallback_lane_idx is not None:
            new_target_lane_idx = self._prepass_fallback_lane_idx
        elif self._prepass_fallback_blocked:
            # Do not keep constraining the known-infeasible lane while the
            # longitudinal controller brings the vehicle to a stop.
            new_target_lane_idx = None

        release_center_lane_rejoin_constraint = (
            self._center_lane_rejoin_active
            and not self._center_lane_rejoin_constraint_released
            and self._target_lane_idx == 1
            and self._mpc.infeasibility_counter
                >= self._center_lane_rejoin_release_infeasible_cycles
        )
        if release_center_lane_rejoin_constraint:
            self._center_lane_rejoin_constraint_released = True
            self.get_logger().warn(
                "[CenterLaneRejoinRelease] releasing L1 constraint while "
                "keeping Center trajectory: "
                f"mpc_infeasible={self._mpc.infeasibility_counter}"
            )

        if (
            self._center_lane_rejoin_active
            and not self._center_lane_rejoin_constraint_released
        ):
            # This final override intentionally wins over the passing-side
            # latch, curve hold, and infeasibility release. Keep the Center
            # CSV while moving to L1 before switching back to Race.
            new_target_lane_idx = 1
        elif self._center_lane_rejoin_constraint_released:
            # Keep Center but expose the full drivable corridor. The release
            # remains latched until the normal heading-safe Race return.
            new_target_lane_idx = None

        if self._prepass_fallback_commit_pending:
            # A confirmed fallback outranks ordinary L1 rejoin and ordinary
            # outer-lane selection until an actually constrained MPC solve
            # succeeds.  The initial L0 boost below intentionally remains the
            # sole non-safety exception requested by the launch configuration.
            new_target_lane_idx = self._prepass_fallback_commit_lane_idx

        if self._parallel_abort_active and not prepass_selection_exclusive:
            # Final exclusive override: use the lane selected when Abort was
            # entered.  In particular, do not force L1 while the other vehicle
            # itself occupies L1.
            new_target_lane_idx = self._parallel_abort_target_lane_idx
            if new_target_lane_idx not in (0, 1, 2):
                new_target_lane_idx = 1

        if initial_start_lateral_hold_active and not self._parallel_abort_active:
            # During boost/minimum hold stay in L0. Afterwards remove the lane
            # constraint completely so tight corners can use the full track;
            # L1 is considered only after the post-boost stability gate passes.
            new_target_lane_idx = 0 if initial_start_l0_hold_active else None
            if not self._initial_start_lane_hold_logged:
                self._initial_start_lane_hold_logged = True
                self.get_logger().info(
                    "[InitialStartLaneHold] holding red right-side L0 during "
                    "initial turbo/maximum acceleration; deferring outer-lane "
                    "overtake selection."
                )
        elif startup_overtake_suppressed and not self._parallel_abort_active:
            # Full-width boost experiment: preserve the start-state exclusion
            # while deliberately applying no L0/L1/L2 hard lane constraint.
            new_target_lane_idx = None

        if self._prepass_fallback_recovery_active:
            # Final exclusive override.  Curve hold, L1 rejoin, parallel
            # selection, and initial boost must not narrow the corridor while
            # Prepass is validating recovery with the full track width.
            new_target_lane_idx = None

        if self._mpc_safety_recovery_active:
            # Solver recovery exclusively owns lateral selection. No
            # overtake/rejoin/start state may narrow the next MPC problem.
            new_target_lane_idx = None

        if self._post_reverse_full_width_recovery_active:
            # DRIVE is confirmed, but lateral selection remains frozen until
            # the solver has produced fresh full-width predictions for the
            # configured continuous-success interval.
            new_target_lane_idx = None

        if recovery_active:
            # Final override for the complete REVERSE/gear-transition window.
            # This also defeats initial boost, parallel handling, and L1 probe
            # requests that were computed earlier in the cycle.
            new_target_lane_idx = None

        if self._l1_safety_recovery_active:
            # This final override also wins over parallel/boost L1 requests.
            # The only path that may set L1 again is the controlled probe.
            new_target_lane_idx = None

        if self._l1_rejoin_backoff_active:
            # A failed normal rejoin probe owns lateral selection until the
            # time/progress/full-width recovery gate releases it.
            new_target_lane_idx = None

        if (
            self._follow_escape_active
            and not recovery_active
            and not self._mpc_safety_recovery_active
            and not self._post_reverse_full_width_recovery_active
        ):
            # This safety state may temporarily override an L1 recovery hold.
            # It applies only a lane that is being explicitly probed.
            new_target_lane_idx = self._follow_escape_probe_lane_idx

        # Final geographic guard. In this Center-WP zone L0 must never be part
        # of the applied corridor. L2 remains allowed; an L0 or full-width
        # request is replaced with L1. This override intentionally wins over
        # ordinary overtaking, fallback, boost and recovery ownership.
        l0_entry_prohibited_active = any(
            (
                start_wp <= center_wp_temp <= end_wp
                if start_wp <= end_wp
                else center_wp_temp >= start_wp or center_wp_temp <= end_wp
            )
            for start_wp, end_wp in self._l0_entry_prohibited_zones
        )
        outer_lane_mpc_problem_active = bool(
            outer_lane_mpc_problem_zone
            and not outer_lane_problem_slow_override
        )
        if outer_lane_mpc_problem_active:
            # This local map transition repeatedly produced an infeasible hard
            # outer-lane solve despite physical passage. Keep ordinary moving
            # traffic on full-width constraints with an L1-oriented reference.
            # A hard L1 constraint is not safe here either: a moving lead in
            # the prediction horizon can narrow L1 to zero and make MPC
            # infeasible. Stopped/ultra-slow targets are exempt so they can
            # still be passed after the normal passage and traffic checks.
            if new_target_lane_idx in (0, 1, 2):
                new_target_lane_idx = None
            if self._overtake.requested_lane in (0, 2):
                self._overtake.requested_lane = None
            if self._prepass_fallback_lane_idx in (0, 2):
                self._prepass_fallback_lane_idx = None
            if self._prepass_fallback_commit_lane_idx in (0, 2):
                self._prepass_fallback_commit_lane_idx = None
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_success_since = None
            # This zone owns corridor selection exclusively.  Cancel normal
            # L1 probe/rejoin state as well as outer-lane metadata so neither
            # can narrow the full-width request again later in this cycle or
            # immediately on the next cycle.
            self._center_lane_rejoin_active = False
            self._center_lane_rejoin_constraint_released = False
            self._center_lane_rejoin_stable_since = None
            self._l1_probe_active = False
            self._l1_probe_context = None
            self._l1_probe_success_cycles = 0
            self._l1_probe_constraint_applied = False
            self._l1_safety_recovery_active = False
            self._l1_safety_recovery_stable_since = None
            self._l1_safety_recovery_context = None
            self._l1_safety_reprobe_pending = False
            self._reset_l1_rejoin_backoff()
            self.get_logger().info(
                "[OuterLaneMPCProblemZone] using full-width constraints for "
                "ordinary moving traffic to avoid known outer/L1 hard-"
                "constraint failures: "
                f"center_wp={center_wp_temp}, lane=full_width",
                throttle_duration_sec=1.0)
        elif outer_lane_mpc_problem_zone and outer_lane_problem_slow_override:
            self.get_logger().info(
                "[OuterLaneMPCProblemZoneSlowOverride] stopped/ultra-slow "
                "target may still use a verified outer corridor: "
                f"center_wp={center_wp_temp}, vehicle_id="
                f"{opponent_vehicle_id}, speed={opponent_v_lead:.2f}m/s/"
                f"{self._outer_lane_problem_override_speed:.2f}m/s",
                throttle_duration_sec=1.0)
        if l0_entry_prohibited_active:
            if new_target_lane_idx == 0:
                new_target_lane_idx = 1
            elif (
                new_target_lane_idx is None
                and not outer_lane_mpc_problem_active
            ):
                # Ordinarily this geographic guard turns a full-width request
                # into L1. In the known MPC problem zone, full width is the
                # deliberate safe alternative to both outer and L1 hard bounds.
                new_target_lane_idx = 1
            if self._overtake.requested_lane == 0:
                self._overtake.requested_lane = None
            if self._prepass_fallback_lane_idx == 0:
                self._prepass_fallback_lane_idx = None
            if self._prepass_fallback_commit_lane_idx == 0:
                self._prepass_fallback_commit_lane_idx = None
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_success_since = None
            if self._follow_escape_probe_lane_idx == 0:
                self._follow_escape_probe_lane_idx = 1
            if self._parallel_abort_target_lane_idx == 0:
                self._parallel_abort_target_lane_idx = 1
            self.get_logger().info(
                "[L0EntryProhibited] forcing a non-L0 corridor: "
                f"center_wp={center_wp_temp}, lane=L{new_target_lane_idx}",
                throttle_duration_sec=1.0)

        # Final L2 geographic policy. Recovery/full-width owners always win;
        # this block only rewrites ordinary L2 requests and every stale owner
        # that could otherwise restore L2 on the next cycle.
        l2_entry_restricted_override_active = False
        if (
            opponent_ahead_detected
            and opponent_vehicle_id is not None
            and not recovery_active
            and not self._mpc_safety_recovery_active
            and not self._post_reverse_full_width_recovery_active
            and not self._prepass_fallback_recovery_active
            and not self._l1_safety_recovery_active
            and not self._l1_rejoin_backoff_active
            and not self._parallel_abort_active
            and not outer_lane_mpc_problem_active
            and self._waypoint_in_configured_zones(
                center_wp_temp, self._l2_entry_restricted_zones)
            and not self._l2_restricted_slow_override(opponent_vehicle_id)
            and (
                prev_lane_idx == 2
                or new_target_lane_idx == 2
                or self._overtake.requested_lane == 2
                or self._prepass_fallback_lane_idx == 2
                or self._prepass_fallback_commit_lane_idx == 2
            )
        ):
            restricted_passage, _ = self._vehicle_passage(
                opponent_vehicle_id, pose)
            restricted_samples = self._relative_lane_vehicle_samples(pose, v)
            restricted_conflicts = {
                lane_idx: classify_lane_conflicts(
                    lane_idx,
                    restricted_samples,
                    front_distance=self._prepass_lane_fallback_front_distance,
                    side_distance=self._prepass_lane_fallback_side_distance,
                    rear_distance=self._prepass_lane_fallback_rear_distance,
                )
                for lane_idx in (0, 1, 2)
            }
            restricted_lane_idx = self._apply_l2_restricted_zone_policy(
                2,
                target_vehicle_id=opponent_vehicle_id,
                physical_passage=restricted_passage,
                conflicts_by_lane=restricted_conflicts,
                center_wp=center_wp_temp,
            )
            new_target_lane_idx = restricted_lane_idx
            if self._overtake.requested_lane == 2:
                self._overtake.requested_lane = (
                    0 if restricted_lane_idx == 0 else None)
            if self._prepass_fallback_lane_idx == 2:
                self._prepass_fallback_lane_idx = (
                    0 if restricted_lane_idx == 0 else None)
            if self._prepass_fallback_commit_lane_idx == 2:
                self._prepass_fallback_commit_lane_idx = None
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_success_since = None
            if self._overtake.probe.lane_idx == 2:
                self._reset_overtake_commit_probe()
            self._clear_prepass_soft_guidance()
            self._mpc.osqp_initialized = False
            l2_entry_restricted_override_active = True

        preserve_hybrid_request = bool(
            self._hybrid_overtake_enabled
            and self._reference_path is self._reference_pathN_center
            and self._overtake.can_resume_hybrid(self._overtake.requested_lane)
            and new_target_lane_idx in (None, self._overtake.requested_lane)
            and not any((
                is_curve_locked, l0_entry_prohibited_active,
                outer_lane_mpc_problem_active, l2_entry_restricted_override_active,
                recovery_active, self._follow_escape_active,
                self._prepass_fallback_recovery_active, self._mpc_safety_recovery_active,
                self._post_reverse_full_width_recovery_active,
                self._prepass_fallback_follow_active, self._prepass_fallback_commit_pending,
                self._prepass_fallback_blocked, self._prepass_fallback_lane_idx is not None,
                self._l1_probe_active, self._center_lane_rejoin_active,
                self._center_lane_rejoin_constraint_released,
                self._l1_safety_recovery_active, self._l1_rejoin_backoff_active,
                self._parallel_abort_active, outer_lane_constraint_released,
                startup_overtake_suppressed, initial_start_lateral_hold_active,
                prepass_distance_gate_released_this_cycle, self._follow_only,
                self._overtake_completed_target_id == self._overtake.target_id,
            ))
        )
        if new_target_lane_idx != prev_lane_idx:
            can_change_lane = True
            
            # Check elapsed time since last lane change
            if self._last_lane_change_time is not None:
                elapsed = current_time_sec - self._last_lane_change_time
                if elapsed < self._lane_change_cooldown_sec:
                    can_change_lane = False  # Lock lane change

            if preserve_hybrid_request:
                can_change_lane = True

            # Curve lock override: force lane constraint immediately if in curve
            if is_curve_locked:
                can_change_lane = True
            if l0_entry_prohibited_active:
                # The geographic prohibition is a safety constraint and must
                # not wait for the ordinary lane-change cooldown.
                can_change_lane = True
            if outer_lane_mpc_problem_active:
                can_change_lane = True
            if l2_entry_restricted_override_active:
                # Geographic L2 removal must not wait for ordinary cooldown.
                can_change_lane = True
            if self._follow_escape_active:
                # Escape probes are safety decisions and must not be delayed
                # by the ordinary two-second lane-change cooldown.
                can_change_lane = True
            # A stationary lead must not remain trapped behind the normal
            # lane-change cooldown when a passing side is available.
            if (
                (lead_is_stationary or lead_is_special_slow)
                and new_target_lane_idx in (0, 2)
            ):
                can_change_lane = True
            # A new latch has already passed the horizon physical-width and
            # predicted-traffic checks and is itself the anti-chatter state.
            # Waiting for the generic cooldown here only delays execution of
            # a decision which has already been made safe and sticky.
            if overtake_latch_started and new_target_lane_idx in (0, 2):
                can_change_lane = True
            # Race return must release the old outer-lane constraint immediately.
            if not opponent_ahead_detected and new_target_lane_idx is None:
                can_change_lane = True
            # This is an infeasibility escape, not a new passing-side request.
            if outer_lane_constraint_released:
                can_change_lane = True
            if self._center_lane_rejoin_constraint_released:
                can_change_lane = True
            if self._l1_probe_active:
                can_change_lane = True
            if l1_probe_failed_this_cycle:
                can_change_lane = True
            if self._prepass_fallback_follow_active:
                can_change_lane = True
            if prepass_distance_gate_released_this_cycle:
                can_change_lane = True
            if self._parallel_abort_active:
                can_change_lane = True
            if initial_start_lateral_hold_active:
                # Apply the initial L0 hold immediately even if an overtake
                # candidate changed the lane shortly before motion was detected.
                can_change_lane = True
            if (
                self._prepass_fallback_recovery_active
                or self._prepass_fallback_blocked
            ):
                can_change_lane = True
            if self._mpc_safety_recovery_active:
                # Releasing an unsafe lane constraint must bypass the normal
                # lane-change cooldown.
                can_change_lane = True
            if self._l1_safety_recovery_active:
                can_change_lane = True
            if self._l1_rejoin_backoff_active:
                can_change_lane = True
            if is_prepass_fallback_lane_change(
                fallback_lane_idx=self._prepass_fallback_lane_idx,
                requested_lane_idx=new_target_lane_idx,
            ):
                # The fallback was selected only after full-width recovery and
                # a fresh conflict check.  Apply it immediately instead of
                # losing another two seconds to the ordinary lane cooldown.
                can_change_lane = True
            if can_change_lane:
                self._target_lane_idx = new_target_lane_idx
            else:
                # Keep the previous lane index to avoid chattering
                pass
        else:
            # No lane change request, keep active candidate
            self._target_lane_idx = new_target_lane_idx

        # Safety decisions must keep referring to the vehicle that started the
        # manoeuvre. A newly detected nearer vehicle must not silently replace
        # the latched target halfway through recovery.
        safety_target_id = self._overtake.target_id
        safety_passage, safety_target_distance = self._latched_target_passage(
            pose)
        safety_target_stationary = False
        safety_target_slow = False
        safety_target_speed = None
        if safety_target_id is not None:
            safety_vx, safety_vy = self._v2x_tracker.velocity(safety_target_id)
            safety_target_speed = math.hypot(safety_vx, safety_vy)
            safety_target_stationary = (
                self._v2x_tracker.has_velocity_estimate(safety_target_id)
                and safety_target_speed < self._stopped_lead_speed_threshold
            )
            safety_target_slow = (
                self._v2x_tracker.has_velocity_estimate(safety_target_id)
                and self._stopped_lead_speed_threshold <= safety_target_speed
                <= self._slow_lead_overtake_speed
            )
        tracked_stopped_lead = (
            safety_target_stationary
            and safety_target_id == self._forced_overtake_vehicle_id
        )
        forced_overtake_active, close_overtake_blocked = (
            evaluate_stopped_lead_overtake(
                tracked_stopped_lead=tracked_stopped_lead,
                distance=(
                    safety_target_distance
                    if safety_target_distance is not None else 99999.0),
                left_is_free=safety_passage.get(2, False),
                right_is_free=safety_passage.get(0, False),
                target_lane_idx=self._target_lane_idx,
                infeasibility_counter=self._mpc.infeasibility_counter,
                reverse_distance=self._close_obstacle_reverse_distance,
                infeasible_cycles=self._close_obstacle_infeasible_cycles,
            )
        )
        # A moving <=10 km/h lead gets the same priority over ordinary ACC as
        # a stopped lead, but must never inherit the stopped-vehicle reverse
        # request. Reverse remains exclusive to a genuinely stationary lead.
        tracked_slow_lead = bool(
            safety_target_slow
            and safety_target_id == self._forced_overtake_vehicle_id
        )
        if tracked_slow_lead:
            forced_overtake_active = bool(
                any(safety_passage.values())
                and self._target_lane_idx in (0, 2)
            )
            close_overtake_blocked = False
        if (
            tracked_stopped_lead
            and not any(safety_passage.values())
        ):
            close_overtake_blocked = False
            self._switch_prepass_to_follow(
                "latched stopped vehicle blocks both passing sides"
            )
        fallback_stop_requested = self._prepass_fallback_blocked
        self._close_obstacle_reverse_requested = bool(
            not startup_overtake_suppressed
            and not prestart_reverse_suppressed
            and (close_overtake_blocked or fallback_stop_requested)
        )

        full_width_recovery_requested = bool(
            self._prepass_fallback_recovery_active
            or self._mpc_safety_recovery_active
            or self._post_reverse_full_width_recovery_active
            or recovery_active
            or self._l1_safety_recovery_active
            or self._l1_rejoin_backoff_active
            or outer_lane_mpc_problem_active
        )
        preserve_hybrid_recovery = bool(
            full_width_recovery_requested
            and (self._prepass_fallback_recovery_active or self._mpc_safety_recovery_active)
            and self._overtake.can_resume_hybrid(self._overtake.requested_lane)
            and self._target_lane_idx in (None, self._overtake.requested_lane)
            and not any((
                recovery_active, self._post_reverse_full_width_recovery_active,
                l0_entry_prohibited_active, l2_entry_restricted_override_active,
                outer_lane_mpc_problem_active, self._parallel_abort_active,
                self._l1_probe_active, self._center_lane_rejoin_active,
                self._center_lane_rejoin_constraint_released,
                self._l1_safety_recovery_active, self._l1_rejoin_backoff_active,
                startup_overtake_suppressed, initial_start_lateral_hold_active,
                prepass_distance_gate_released_this_cycle,
                self._overtake_completed_target_id == self._overtake.target_id,
            ))
        )
        applied_lane_idx, hybrid_outer_transition, in_transition = self._apply_lane_decision(
            requested_lane=self._target_lane_idx,
            now_sec=current_time_sec,
            l0_prohibited=l0_entry_prohibited_active,
            full_width_recovery=full_width_recovery_requested,
            preserve_manoeuvre=preserve_hybrid_request or preserve_hybrid_recovery,
            allow_hybrid=(self._reference_path is self._reference_pathN_center
                          and not initial_start_boost_active
                          and not startup_overtake_suppressed
                          and not self._parallel_abort_active),
        )

        normal_l1_full_width_wait = (
            center_lane_rejoin_clear
            and not self._center_lane_rejoin_active
            and not self._l1_probe_active
        )
        rejoin_safety_full_width_wait = (
            self._l1_safety_recovery_active
            and (self._l1_safety_recovery_context or "rejoin") == "rejoin"
        )
        l1_probe_transition = (
            in_transition
            and self._l1_probe_active
            and self._l1_probe_context == "rejoin"
            and self._target_lane_idx == 1
        )
        soft_rejoin_enabled = (
            self._l1_soft_rejoin_enabled
            and self._reference_path is self._reference_pathN_center
            and (
                normal_l1_full_width_wait
                or rejoin_safety_full_width_wait
                or self._l1_rejoin_backoff_active
                or self._center_lane_rejoin_constraint_released
                or outer_lane_constraint_released
                or l1_probe_transition
            )
            and (
                self._reference_path.target_lane_idx is None
                and not self._reference_path.is_overtaking
            )
            and not self._mpc_safety_recovery_active
            and not self._post_reverse_full_width_recovery_active
            and not recovery_active
            and not self._prepass_fallback_recovery_active
            and not self._prepass_fallback_commit_pending
            and not self._prepass_fallback_follow_active
            and not self._follow_escape_active
            and not self._parallel_abort_active
            and not initial_start_lateral_hold_active
            and not initial_start_boost_active
            and not self._race_rejoin_handoff_active
            and not outer_lane_mpc_problem_active
        )
        self._update_l1_soft_rejoin_reference(
            enabled=soft_rejoin_enabled,
            now_sec=current_time_sec,
        )
        race_handoff_soft_enabled = (
            self._race_rejoin_handoff_active
            and self._race_rejoin_handoff_soft
            and self._reference_path is self._reference_pathN_center
            and self._reference_path.target_lane_idx is None
            and not self._reference_path.is_overtaking
            and not self._mpc_safety_recovery_active
            and not self._post_reverse_full_width_recovery_active
            and not recovery_active
            and not self._prepass_fallback_recovery_active
            and not self._prepass_fallback_commit_pending
            and not self._follow_escape_active
            and not self._parallel_abort_active
            and not outer_lane_mpc_problem_active
        )
        self._update_race_handoff_reference(
            enabled=race_handoff_soft_enabled,
            now_sec=current_time_sec)
        initial_start_soft_l0_enabled = (
            self._initial_start_soft_l0_enabled
            and initial_start_boost_active
            and not self._initial_start_hold_l0
            and self._reference_path is self._reference_pathN_center
            and self._reference_path.target_lane_idx is None
            and not self._reference_path.is_overtaking
            and not self._mpc_safety_recovery_active
            and not self._post_reverse_full_width_recovery_active
            and not recovery_active
            and not self._parallel_abort_active
        )
        self._update_initial_start_soft_l0_reference(
            enabled=initial_start_soft_l0_enabled,
            now_sec=current_time_sec,
        )
        prepass_soft_guidance_enabled = bool(
            self._prepass_fallback_recovery_active
            and self._prepass_soft_candidate_lane_idx in (0, 2)
            and self._reference_path is self._reference_pathN_center
            and applied_lane_idx is None
            and not self._mpc_safety_recovery_active
            and not self._post_reverse_full_width_recovery_active
            and not recovery_active
            and not self._parallel_abort_active
        )
        self._update_prepass_soft_reference(
            enabled=prepass_soft_guidance_enabled,
            lane_idx=self._prepass_soft_candidate_lane_idx,
            now_sec=current_time_sec,
        )
        overtake_soft_transition_enabled = bool(
            (hybrid_outer_transition if self._hybrid_overtake_enabled else in_transition)
            and self._target_lane_idx in (0, 2)
            and self._reference_path is self._reference_pathN_center
            and not full_width_recovery_requested
            and not initial_start_boost_active
            and not self._parallel_abort_active
        )
        if self._hybrid_overtake_enabled:
            self._update_overtake_transition_soft_reference(
                enabled=overtake_soft_transition_enabled,
                lane_idx=self._target_lane_idx, now_sec=current_time_sec,
                ego_speed=v, vehicle_id=self._overtake.target_id)
        else:
            self._mpcN_center.set_lane_transition_weights()
            self._update_legacy_overtake_transition_soft_reference(
                enabled=overtake_soft_transition_enabled,
                lane_idx=self._target_lane_idx, now_sec=current_time_sec,
                transition_end_sec=self._constraint_transition_until,
                transition_duration_sec=0.6)

        is_overtaking = self._reference_path.is_overtaking

        # 追従・追い越し、または対象のWaypoint区間（カーブなど慎重さが求められる箇所）はwp_id_offsetを1にする
        is_in_cautious_zone = (210 <= wp <= 243) or (261 <= wp <= 286)
        active_offset = 0 if (is_overtaking or is_in_cautious_zone) else self._default_wp_id_offset
        self._mpc.update_wp_id_offset(active_offset)

        base_prediction_fallback_limit = max(int(getattr(
            self._cfg.mpc, "max_prediction_fallback_cycles", 3)), 0)
        moving_target_prediction_clear = bool(
            is_overtaking
            and safety_target_id is not None
            and safety_target_id in self._v2x_tracker.active_vehicle_ids()
            and self._v2x_tracker.has_velocity_estimate(safety_target_id)
            and safety_target_speed is not None
            and safety_target_speed >= self._moving_lead_mpc_grace_speed
            and self._prediction_is_clear_of_vehicle(safety_target_id)
        )
        if moving_target_prediction_clear:
            active_prediction_fallback_limit = max(
                base_prediction_fallback_limit,
                self._moving_lead_mpc_grace_cycles,
            )
        elif safety_target_slow and is_overtaking:
            active_prediction_fallback_limit = (
                self._slow_lead_overtake_infeasible_cycles)
        else:
            active_prediction_fallback_limit = base_prediction_fallback_limit
        self._mpc.max_prediction_fallback_cycles = (
            active_prediction_fallback_limit)
        
        # Objective-only local offsets never widen hard lane/course bounds.
        # Clear them on Race MPC so Center-specific state cannot leak across a
        # trajectory switch.
        if self._mpc is self._mpcN_center:
            self._update_l2_target_objective_offsets(
                self._mpc, self._carN_center.wp_id)
            self._update_full_width_l1_objective_offsets(
                self._mpc, self._carN_center.wp_id)
            self._update_full_width_l0_objective_offsets(
                self._mpc, self._carN_center.wp_id)
        else:
            self._mpc.set_target_lane_lateral_offsets()
            self._mpc.set_full_width_l1_offset_limits()
            self._mpc.set_full_width_l0_offset_limits()

        if (self._follow_escape_active and not self._follow_escape_forward_active
                and self._follow_escape_probe_lane_idx in (0,2)
                and self._reference_path.target_lane_idx == self._follow_escape_probe_lane_idx
                and not self._prepass_fallback_recovery_active
                and not self._mpc_safety_recovery_active
                and not self._post_reverse_full_width_recovery_active
                and not recovery_active and not self._parallel_abort_active
                and not self._collision_evidence_hold):
            self._mpc.update_v_max(self._follow_escape_creep_speed)
            self._reference_path.set_v_ref(
                [self._follow_escape_creep_speed]*len(self._reference_path.waypoints))

        # MPCの実行
        with self._stats.time_block("control"):
            u, max_delta = self._mpc.get_control()
        self._live_prediction_context = (
            self._mpc, self._mpc.current_prediction,
            self._reference_path.target_lane_idx, self._reference_path,
            self._v2x_tracker)

        pure_pursuit_safe_this_cycle = False
        if self._steering_fallback_enabled:
            solution_valid = bool(
                not self._mpc.recovery_requested
                and self._mpc.infeasibility_counter == 0
                and self._mpc.current_prediction is not None
                and not self._mpc.used_prediction_fallback
                and not self._mpc.time_budget_exceeded)
            solution_accurate = bool(
                solution_valid
                and getattr(self._mpc, "last_solution_accurate", False))
            # Two-state controller ownership. One invalid MPC cycle enters the
            # fallback state; fallback keeps ownership until enough consecutive
            # accurate MPC solutions have been observed.
            if not solution_valid and not self._steering_fallback_armed:
                self._steering_fallback_armed = True
                self._steering_fallback_success_cycles = 0
                self.get_logger().warn(
                    "[ActivePathSteeringFallbackEnter] MPC output became "
                    "invalid; fallback now owns steering")

            if self._steering_fallback_armed:
                if solution_accurate:
                    self._steering_fallback_success_cycles += 1
                else:
                    self._steering_fallback_success_cycles = 0

                if self._steering_fallback_success_cycles >= (
                    self._steering_fallback_success_required
                ):
                    self._steering_fallback_armed = False
                    self._steering_fallback_success_cycles = 0
                    self.get_logger().info(
                        "[ActivePathSteeringFallbackRelease] MPC recovered; "
                        "normal control now owns steering")
                else:
                    # Even a temporarily valid MPC output cannot take control
                    # while the fallback state is latched.
                    fallback_delta, lookahead, target_wp, reason = (
                        self._active_path_pure_pursuit_feedback(
                            predicted_pose, v))
                    controller = "pure_pursuit"
                    if fallback_delta is not None:
                        pp_safe, validation_reason = (
                            self._pure_pursuit_feedback_is_safe(
                                predicted_pose, v, fallback_delta))
                        if not pp_safe:
                            fallback_delta = None
                            reason = validation_reason
                    if fallback_delta is None:
                        fallback_delta = self._legacy_active_path_feedback()
                        controller = "legacy_feedback"
                        u[0] = min(
                            float(u[0]), self._steering_fallback_speed)
                    else:
                        # Safety and emergency-brake processing later in this
                        # cycle may still reduce or stop this request.
                        pure_pursuit_safe_this_cycle = True
                        if (
                            moving_target_prediction_clear
                            and self._mpc.used_prediction_fallback
                        ):
                            # Preserve the stored MPC speed instead of jumping
                            # to the generic fallback speed while a normally
                            # moving opponent is clearing the temporary
                            # constraint conflict.
                            u[0] = min(
                                max(float(u[0]), 0.0),
                                self._steering_fallback_speed,
                            )
                        else:
                            u[0] = self._steering_fallback_speed
                    u[1] = fallback_delta
                    self._mpc.previous_steering = fallback_delta
                    self.get_logger().warn(
                        "[ActivePathSteeringFallback] fallback owns control; "
                        f"controller={controller}, mpc_valid={solution_valid}, "
                        "mpc_recovery_cycles="
                        f"{self._steering_fallback_success_cycles}/"
                        f"{self._steering_fallback_success_required}, "
                        f"pp_reason={reason}, lookahead={lookahead:.2f}m, "
                        f"target_wp={target_wp}, speed={float(u[0]):.2f}m/s, "
                        f"steering={fallback_delta:+.3f}rad",
                        throttle_duration_sec=0.5)

        # Race復帰中だけ裏でRace MPCを解く。追越し車線の選択には使わない。
        self._run_race_rejoin_probe(
            predicted_pose,
            recovery_active=recovery_active,
            now_sec=current_time_sec,
            heading_ok=race_rejoin_probe_heading_ok,
            heading_released=race_rejoin_probe_heading_released)
        self._run_overtake_commit_probe(
            predicted_pose,
            recovery_active=recovery_active)

        if self._mpc.used_prediction_fallback:
            self._mpc_prediction_fallback_cycles += 1
        elif self._mpc.infeasibility_counter == 0:
            self._mpc_prediction_fallback_cycles = 0

        moving_target_grace_active = bool(
            moving_target_prediction_clear
            and pure_pursuit_safe_this_cycle
            and self._prediction_is_clear_of_vehicle(safety_target_id)
        )
        if moving_target_prediction_clear and not moving_target_grace_active:
            # The extended grace is conditional, not a blind stale-command
            # hold.  Once the opponent prediction or the independently
            # checked Pure Pursuit path becomes unsafe, return immediately to
            # the ordinary short fallback budget.
            active_prediction_fallback_limit = (
                base_prediction_fallback_limit)

        max_fallback_cycles = active_prediction_fallback_limit
        if self._mpc_prediction_fallback_cycles > max_fallback_cycles:
            # Keep the limit valid even when Race/Center switches between two
            # MPC instances during one failure sequence.
            self._mpc.current_prediction = None
            self._mpc.current_control = np.zeros_like(
                self._mpc.current_control)
            self._mpc.used_prediction_fallback = False
            self._mpc.recovery_requested = True
            self._mpc.failure_reason = (
                "prediction fallback exceeded controller-wide limit "
                f"({max_fallback_cycles} cycles)")
        elif (
            self._mpc.used_prediction_fallback
            and moving_target_grace_active
        ):
            self.get_logger().info(
                "[MovingLeadMPCGrace] moving target and stored prediction "
                "remain clear; continuing with safe-path steering while "
                "waiting for MPC to recover: "
                f"vehicle_id={safety_target_id}, "
                f"target_speed={float(safety_target_speed):.2f}m/s, "
                f"fallback_cycles={self._mpc_prediction_fallback_cycles}/"
                f"{max_fallback_cycles}",
                throttle_duration_sec=0.5,
            )

        applied_lane_idx = (
            self._reference_path.target_lane_idx
            if self._reference_path.is_overtaking else None
        )
        outer_bounds_collapsed = False
        outer_collapse_detail = None
        if self._mpc.recovery_requested and applied_lane_idx in (0, 2):
            outer_bounds_collapsed, outer_collapse_detail = (
                self._outer_lane_constraint_is_collapsed(applied_lane_idx)
            )
            if outer_bounds_collapsed:
                self.get_logger().error(
                    "[OuterLaneConstraintCollapse] latest constrained solve "
                    "contains an invalid/zero-width outer corridor; bypassing "
                    "the normal failure confirmation and releasing to full "
                    "width immediately: "
                    f"lane=L{applied_lane_idx}, "
                    f"i={outer_collapse_detail['index']}, "
                    f"wp={outer_collapse_detail['wp']}, "
                    f"bounds=[{outer_collapse_detail['lower']:.3f},"
                    f"{outer_collapse_detail['upper']:.3f}], "
                    f"width={outer_collapse_detail['width']:.3f}m, "
                    f"reason={self._mpc.failure_reason}"
                )
        race_to_center_failure_grace_active = bool(
            self._reference_path is self._reference_pathN_center
            and self._trajectory_last_switch_time is not None
            and current_time_sec - self._trajectory_last_switch_time
                < self._race_to_center_failure_grace_sec
        )
        ignore_outer_failure_during_handoff = bool(
            not outer_bounds_collapsed
            and
            race_to_center_failure_grace_active
            and applied_lane_idx in (0, 2)
            and pure_pursuit_safe_this_cycle
        )
        if (
            self._mpc.recovery_requested
            and not ignore_outer_failure_during_handoff
        ):
            self._mpc_recovery_request_cycles += 1
        else:
            self._mpc_recovery_request_cycles = 0
        pp_safe_fast_recovery_active = bool(
            pure_pursuit_safe_this_cycle
            and not safety_target_slow
            and self._waypoint_in_configured_zones(
                center_wp_temp, self._pp_safe_recovery_fast_zones)
        )
        recovery_confirm_cycles = (
            self._mpc_pp_safe_zone_recovery_confirm_cycles
            if pp_safe_fast_recovery_active
            else self._mpc_pp_safe_recovery_confirm_cycles
            if pure_pursuit_safe_this_cycle
            else self._mpc_recovery_confirm_cycles
        )
        recovery_request_confirmed = bool(
            self._mpc.recovery_requested
            and (
                outer_bounds_collapsed
                or self._mpc_recovery_request_cycles >= recovery_confirm_cycles
            )
        )
        outer_lane_recovery_confirmed = bool(
            recovery_request_confirmed
            and (
                outer_bounds_collapsed
                or not race_to_center_failure_grace_active
            )
        )
        if self._prepass_fallback_commit_pending:
            commit_lane_applied = (
                applied_lane_idx == self._prepass_fallback_commit_lane_idx)
            commit_solution_feasible = (
                not recovery_active
                and not self._mpc.recovery_requested
                and self._mpc.infeasibility_counter == 0
                and self._mpc.current_prediction is not None
            )
            self._prepass_fallback_commit_success_since = (
                update_fallback_commit_success_since(
                    self._prepass_fallback_commit_success_since,
                    now_sec=current_time_sec,
                    lane_applied=commit_lane_applied,
                    feasible_solution=commit_solution_feasible,
                )
            )

        commit_success_elapsed = (
            current_time_sec - self._prepass_fallback_commit_success_since
            if self._prepass_fallback_commit_success_since is not None
            else 0.0
        )

        if (
            self._prepass_fallback_commit_pending
            and commit_success_elapsed
                >= self._prepass_commit_required_success_sec
        ):
            committed_lane_idx = self._prepass_fallback_commit_lane_idx
            self._prepass_fallback_commit_pending = False
            self._prepass_fallback_commit_lane_idx = None
            self._prepass_fallback_commit_success_since = None
            self._prepass_attempted_outer_lanes.clear()
            self.get_logger().info(
                "[PrepassFallbackCommit] selected fallback lane was applied "
                "and solved successfully for the required duration; releasing "
                f"exclusive ownership: lane=L{committed_lane_idx}, "
                f"success_sec={commit_success_elapsed:.2f}"
            )

        # Capture every failed outer-lane solve, even if Prepass cannot start
        # (for example because its opponent latch disappeared in this cycle).
        # The MPC snapshot still describes the failed constrained solve.
        if (
            not recovery_active
            and outer_lane_recovery_confirmed
            and applied_lane_idx in (0, 2)
        ):
            physical_passage, _ = self._latched_target_passage(pose)
            if physical_passage.get(int(applied_lane_idx), False):
                self.get_logger().warn(
                    "[PassageMPCMismatch] physical passage was true but the "
                    "outer-lane MPC failed: "
                    f"vehicle_id={self._overtake.target_id}, "
                    f"lane=L{applied_lane_idx}, wp={center_wp_temp}, "
                    f"mpc_reason={self._mpc.failure_reason}, "
                    f"infeasible={self._mpc.infeasibility_counter}"
                )
            self._log_lane_constraint_diagnostics(
                lane_idx=int(applied_lane_idx),
                context="safety_trigger:outer_lane",
                failed_wp=center_wp_temp,
                reason=(
                    self._mpc.failure_reason
                    or "outer-lane SafetyRecovery"
                ),
            )

        if not recovery_active and should_start_l1_recovery_from_safety(
            recovery_requested=recovery_request_confirmed,
            applied_lane_idx=applied_lane_idx,
            l1_recovery_pending=self._l1_safety_reprobe_pending,
        ):
            failed_context = (
                self._l1_probe_context
                or (
                    "fallback"
                    if self._prepass_fallback_lane_idx == 1
                    else "rejoin"
                )
            )
            self._log_l1_constraint_diagnostics(
                context=f"safety_trigger:{failed_context}",
                failed_wp=center_wp_temp,
                reason=(self._mpc.failure_reason or "L1 SafetyRecovery"),
            )
            self._l1_safety_recovery_active = True
            self._l1_safety_recovery_stable_since = None
            self._l1_safety_recovery_context = failed_context
            self._l1_safety_reprobe_pending = True
            self._center_lane_rejoin_active = False
            self._center_lane_rejoin_constraint_released = False
            self._center_lane_rejoin_stable_since = None
            self._l1_probe_active = False
            self._l1_probe_context = None
            self._l1_probe_success_cycles = 0
            self._l1_probe_constraint_applied = False
            if failed_context == "fallback":
                self._prepass_fallback_lane_idx = None
                self._prepass_fallback_recovery_active = False
                self._prepass_fallback_recovery_stable_since = None
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_lane_idx = None
            self.get_logger().warn(
                "[L1SafetyRecoveryTrigger] applied L1 MPC failure recorded; "
                "holding full width until stable before re-probe: "
                f"context={failed_context}, "
                f"mpc_infeasible={self._mpc.infeasibility_counter}, "
                f"reason={self._mpc.failure_reason}"
            )

        if (
            not recovery_active
            and not startup_overtake_suppressed
            and should_start_prepass_recovery_from_safety(
                recovery_requested=outer_lane_recovery_confirmed,
                fallback_enabled=self._prepass_lane_fallback_enabled,
                applied_lane_idx=applied_lane_idx,
                latched_vehicle_id=self._overtake.target_id,
                opponent_ahead=opponent_ahead_detected,
                fallback_recovery_active=self._prepass_fallback_recovery_active,
                fallback_follow_active=self._prepass_fallback_follow_active,
            )
        ):
            continuing_failed_fallback = bool(
                self._prepass_fallback_commit_pending
                and applied_lane_idx == self._prepass_fallback_commit_lane_idx
            )
            if not continuing_failed_fallback:
                self._prepass_attempted_outer_lanes.clear()
            self._cancel_normal_l1_rejoin_for_prepass()
            self._prepass_failed_lane_idx = applied_lane_idx
            self._overtake.requested_lane = applied_lane_idx
            self._prepass_fallback_lane_idx = None
            self._prepass_fallback_blocked = False
            self._prepass_fallback_recovery_active = True
            self._prepass_fallback_recovery_stable_since = None
            self._prepass_fallback_recovery_started_at = current_time_sec
            self._prepass_target_behind_since = None
            self._prepass_fallback_commit_pending = False
            self._prepass_fallback_commit_lane_idx = None
            self.get_logger().warn(
                "[PrepassLaneFallbackSafetyTrigger] outer-lane MPC failure "
                "starts full-width fallback recovery immediately: "
                f"vehicle_id={self._overtake.target_id}, "
                f"failed_lane=L{applied_lane_idx}, "
                f"mpc_infeasible={self._mpc.infeasibility_counter}, "
                f"reason={self._mpc.failure_reason}"
            )

        if recovery_request_confirmed:
            if not self._mpc_safety_recovery_active:
                # Full-width MPC recovery preempts FollowEscape.  Keep the
                # target latch, but never carry a candidate lane or consecutive
                # shadow-success count across the recovery boundary.
                self._invalidate_follow_escape_probe_for_recovery(
                    self._mpc.failure_reason or "MPC SafetyRecovery"
                )
                # Start the immobility observation window at SafetyRecovery
                # entry.  Old GNSS samples from the preceding stop must not
                # make reverse eligible immediately.
                self._gnss_history = []
                self._stuck_since = None
                self.get_logger().error(
                    "[MPCSafetyRecovery] discarding stale prediction and "
                    "switching to full-width recovery: "
                    f"reason={self._mpc.failure_reason}, "
                    f"failed_cycles={self._mpc.infeasibility_counter}, "
                    f"confirmed_cycles={self._mpc_recovery_request_cycles}/"
                    f"{recovery_confirm_cycles}, "
                    f"time_budget_exceeded="
                    f"{self._mpc.time_budget_exceeded}"
                )
            self._mpc_safety_recovery_active = True
            self._mpc_safety_recovery_success_cycles = 0
            u[0] = 0.0
        elif self._mpc_safety_recovery_active:
            # Ordinary MPC recovery never changed gear and must not depend on a
            # GearReport topic.  Only post-reverse recovery requires DRIVE to be
            # confirmed (directly or by the stopped-command fallback).
            recovery_gear_ready = bool(
                not self._post_reverse_full_width_recovery_active
                or self._current_gear_is_drive()
            )
            if should_count_mpc_recovery_success(
                stuck_recovery_active=recovery_active,
                gear_is_drive=recovery_gear_ready,
                infeasibility_counter=self._mpc.infeasibility_counter,
                has_current_prediction=(
                    self._mpc.current_prediction is not None),
                used_prediction_fallback=self._mpc.used_prediction_fallback,
            ):
                self._mpc_safety_recovery_success_cycles += 1
                if (
                    self._mpc_safety_recovery_success_cycles
                    >= self._mpc_safety_recovery_success_required_cycles
                ):
                    self._mpc_safety_recovery_active = False
                    self._mpc_safety_recovery_success_cycles = 0
                    post_reverse_recovery_completed = (
                        self._post_reverse_full_width_recovery_active
                    )
                    self._post_reverse_full_width_recovery_active = False
                    if (
                        post_reverse_recovery_completed
                        and self._prepass_fallback_recovery_active
                    ):
                        # Start Prepass timing only now. Its first action is a
                        # fresh physical-passage/V2X evaluation; no lane was
                        # selected during gear or MPC recovery.
                        self._prepass_fallback_recovery_started_at = (
                            current_time_sec)
                        self._prepass_fallback_recovery_stable_since = None
                    self.get_logger().info(
                        "[MPCSafetyRecovery] full-width MPC recovered for "
                        f"{self._mpc_safety_recovery_success_sec:.2f}s "
                        "continuously; normal lane selection resumed."
                    )
                    if post_reverse_recovery_completed:
                        self.get_logger().info(
                            "[PostReverseFullWidthRecovery] complete; "
                            "Prepass/lane selection may resume."
                        )
            else:
                self._mpc_safety_recovery_success_cycles = 0

        slow_pass_release_ids = (self._slow_pass_spacing_release_ids(pose)
                                 if self.USE_OBSTACLE_AVOIDANCE else set())
        if slow_pass_release_ids:
            self.get_logger().info(
                f"[SlowPassSpacingRelease] lane=L{self._reference_path.target_lane_idx}, "
                f"vehicles={sorted(slow_pass_release_ids)}; fresh swept transition clear",
                throttle_duration_sec=0.5)
        elif self.USE_OBSTACLE_AVOIDANCE and self._overtake.target_id is not None:
            self.get_logger().info(
                f"[SlowPassSpacingHold] target={self._overtake.target_id}, "
                f"reason={self._slow_pass_release_reason}", throttle_duration_sec=1.0)
        forced_overtake_prediction_clear = (
            forced_overtake_active
            and not in_transition
            and self._overtake_prediction_is_clear(
                self._forced_overtake_vehicle_id)
        )
        slow_lead_outer_lane_idx = self._reference_path.target_lane_idx
        slow_lead_outer_lane_error = (
            self._lane_lateral_error(
                pose.x, pose.y, int(slow_lead_outer_lane_idx))
            if slow_lead_outer_lane_idx in (0, 2)
            else math.inf
        )
        slow_lead_lateral_move_established = bool(
            slow_lead_speed_control_active
            and self._reference_path.is_overtaking
            and slow_lead_outer_lane_idx in (0, 2)
            and not in_transition
            and slow_lead_outer_lane_error
                <= self._slow_lead_speed_match_release_lateral_error
        )
        slow_lead_speed_match_required = bool(
            not forced_overtake_prediction_clear
            and opponent_vehicle_id not in slow_pass_release_ids
            and not slow_lead_lateral_move_established
        )
        confirmed_precommit_lane_idx = (
            int(self._overtake.probe.lane_idx)
            if (
                self._overtake.probe.lane_idx in (0, 2)
                and self._overtake_commit_probe_is_fresh(
                    opponent_vehicle_id,
                    self._overtake.probe.lane_idx,
                    current_time_sec,
                )
            )
            else None
        )
        committed_shadow_lane_idx = (
            int(self._overtake.verification.lane_idx)
            if (
                self._overtake.verification.lane_idx in (0, 2)
                and self._overtake.verification.vehicle_id == opponent_vehicle_id
                and self._overtake.target_id == opponent_vehicle_id
                and self._overtake.requested_lane
                    == self._overtake.verification.lane_idx
            )
            else None
        )
        verified_speed_release_lane_idx = (
            confirmed_precommit_lane_idx
            if confirmed_precommit_lane_idx in (0, 2)
            else committed_shadow_lane_idx
        )
        confirmed_precommit_lane_still_clear = bool(
            (verified_speed_release_lane_idx == 2 and left_is_free)
            or (verified_speed_release_lane_idx == 0 and right_is_free)
        )
        precommit_safe_outer_corridor = bool(
            (slow_lead_speed_control_active or forced_overtake_active)
            and (
                not forced_overtake_active
                or opponent_vehicle_id == self._forced_overtake_vehicle_id
            )
            # A horizon passage result alone is not enough to release speed
            # matching. Require two consecutive strict Shadow MPC successes
            # for this exact vehicle and the same lane that is still clear.
            and confirmed_precommit_lane_still_clear
            and not self._prepass_fallback_recovery_active
            and not self._mpc_safety_recovery_active
            and not recovery_active
            and not self._parallel_abort_active
            and not outer_lane_mpc_problem_active
        )
        if precommit_safe_outer_corridor:
            # Strict Shadow MPC has already verified physical width, V2X
            # traffic, solution accuracy, relaxation and the forward corridor
            # for this exact lane. Do not crawl at lead_speed + 0.5 m/s merely
            # because applying the verified lane is still pending.
            # This intentionally does not depend on the currently applied
            # lane: before commitment it is normally L1/full-width.
            # ParallelSafety, EmergencyBrake and MPC recovery remain able to
            # impose stricter limits independently.
            slow_lead_speed_match_required = False
            safe_lane_idx = verified_speed_release_lane_idx
            self.get_logger().info(
                "[SlowLeadPrecommitSpeedRelease] safe outer corridor is "
                "Shadow-confirmed for the same target/lane; keeping normal "
                "approach speed through "
                f"lane commit: vehicle_id={opponent_vehicle_id}, "
                f"lane=L{safe_lane_idx}, distance={opponent_distance:.2f}m",
                throttle_duration_sec=1.0,
            )
        if slow_lead_lateral_move_established:
            self.get_logger().info(
                "[SlowLeadSpeedMatchRelease] outer-lane lateral movement is "
                "established; releasing lead-speed matching while ordinary "
                "ParallelSafety/EmergencyBrake remain active: "
                f"vehicle_id={opponent_vehicle_id}, "
                f"lane=L{slow_lead_outer_lane_idx}, "
                f"lateral_error={slow_lead_outer_lane_error:.2f}/"
                f"{self._slow_lead_speed_match_release_lateral_error:.2f}m",
                throttle_duration_sec=1.0,
            )

        # ref_vel_configuratorがある場合はその値を基準に、なければv_maxを基準にする
        if self._ref_vel_configulator is not None:
            ref_vel_mps = self._ref_vel_configulator.get_ref_vel(self._mpc.model.wp_id)
            ref_vel_kmph = min(
                kmh_to_m_per_sec(ref_vel_mps),
                self._mpc_cfg.v_max)
        else:
            ref_vel_kmph = self._mpc_cfg.v_max

        if (
            self._prepass_fallback_recovery_active
            and self._prepass_dynamic_conflict_speed_limit is not None
        ):
            # Do not continue closing at race speed while lateral ownership is
            # released and the opposite lane is being validated. A small
            # margin preserves forward controllability without ramming the
            # lane-changing target before the next constrained solve.
            safe_transition_speed = float(
                self._prepass_dynamic_conflict_speed_limit)
            ref_vel_kmph = min(ref_vel_kmph, safe_transition_speed)
            self.get_logger().warn(
                "[OvertakeLatchedLaneUnsafeSpeed] limiting approach speed "
                "during full-width side re-evaluation: "
                f"target={latched_target_id}, "
                f"speed_limit={ref_vel_kmph:.2f}m/s",
                throttle_duration_sec=1.0,
            )

        emergency_brake_active = False
        emergency_brake_vehicle_id = None
        emergency_brake_vehicle_distance = math.inf
        follow_restart_active = False
        follow_restart_target = 0.0
        follow_target_expired = False
        follow_control_active = False
        intentional_follow_stop_active = False
        slow_lead_active_speed_margin = self._slow_lead_overtake_speed_margin
        hybrid_escape_target = self._overtake.hybrid.vehicle_id
        hybrid_escape_speed = (self._hybrid_escape_speed(pose, v)
                               if self.USE_OBSTACLE_AVOIDANCE else 0.0)
        hybrid_escape_speeds = {hybrid_escape_target: hybrid_escape_speed}
        if self.USE_OBSTACLE_AVOIDANCE:
            for member in self._stationary_lane_group(pose, v, self._overtake.hybrid.lane_idx):
                hybrid_escape_speeds[member] = self._hybrid_escape_speed(pose, v, member)
            for member in slow_pass_release_ids:
                hybrid_escape_speeds[member] = max(
                    hybrid_escape_speeds.get(member,0.), self._hybrid_escape_creep_speed)
            hybrid_escape_speed = max(hybrid_escape_speeds.values(), default=0.0)
            # --- ACC spacing control (車間距離維持制御) ---
            # 前方車両がいて、かつ自車の走行ライン上（横方向の差が 1.2m 未満）に他車が位置する場合に
            # 追従状態とみなして、設定された距離内で車間制御を有効化する。
            # 横方向の差が設定値以上の場合は、別車線の車両として扱う。
            acc_vehicle_id = opponent_vehicle_id
            acc_distance = opponent_distance
            acc_lead_speed = opponent_v_lead
            acc_offset = opponent_offset
            acc_velocity_valid = opponent_velocity_valid
            latched_follow_state = None
            if self._prepass_fallback_follow_active:
                latched_follow_state = self._latched_follow_target_state(
                    pose, current_time_sec)
                if (
                    latched_follow_state is None
                    or latched_follow_state.get("expired", False)
                ):
                    follow_target_expired = True
                elif not is_follow_target_ahead(
                    latched_follow_state.get("longitudinal")
                ):
                    behind_longitudinal = latched_follow_state["longitudinal"]
                    self._release_lost_follow_target(
                        reason=(
                            "latched vehicle moved behind ego "
                            f"(longitudinal={behind_longitudinal:.2f}m)"
                        )
                    )
                    latched_follow_state = None
                else:
                    acc_vehicle_id = latched_follow_state["vehicle_id"]
                    acc_distance = latched_follow_state["distance"]
                    acc_lead_speed = latched_follow_state["speed"]
                    acc_offset = latched_follow_state["offset"]
                    acc_velocity_valid = latched_follow_state["velocity_valid"]
                    if latched_follow_state.get("stale", False):
                        self.get_logger().warn(
                            "[FollowTargetHold] latched target temporarily "
                            f"missing; vehicle_id={acc_vehicle_id}, "
                            f"cached_distance={acc_distance:.2f}m, "
                            f"speed_cap={self._follow_target_lost_max_speed:.2f}m/s",
                            throttle_duration_sec=0.5,
                        )

            e_y = self._car.spatial_state.e_y
            lat_dist = abs(acc_offset - e_y)

            # --- Standard Follow (Same Lane) ---
            if (
                (opponent_ahead is not None or latched_follow_state is not None)
                and (
                    lat_dist < self._follow_lateral_distance
                    or self._prepass_fallback_follow_active
                )
                and not forced_overtake_active
                and acc_vehicle_id not in slow_pass_release_ids
                and (
                    not initial_start_boost_active
                    or self._follow_only
                )
            ):
                if follow_target_expired:
                    ref_vel_kmph = 0.0
                    self._release_lost_follow_target()
                elif acc_distance < self._follow_engage_distance:
                    follow_control_active = True
                    v_ref_acc = (
                        acc_lead_speed
                        + self._follow_spacing_kp * (
                            acc_distance - self._follow_desired_distance
                        )
                    )
                    v_ref_acc = max(0.0, v_ref_acc)  # 後退は禁止のため下限は0

                    if (
                        latched_follow_state is not None
                        and latched_follow_state.get("stale", False)
                    ):
                        v_ref_acc = min(
                            v_ref_acc, self._follow_target_lost_max_speed)

                    if hybrid_escape_speeds.get(acc_vehicle_id, 0.0) > 0.0:
                        v_ref_acc = max(v_ref_acc, hybrid_escape_speeds[acc_vehicle_id])
                    ref_vel_kmph = min(ref_vel_kmph, v_ref_acc)

                    ego_is_stopped = (
                        abs(v) < self._follow_restart_ego_stopped_speed)
                    startup_restart_waiting = (
                        self._grounded_start_boost_eligible is not None
                        and not self._has_moved_once
                    )
                    restart_min_gap = startup_follow_restart_gap(
                        startup_waiting=startup_restart_waiting,
                        normal_min_gap=self._follow_restart_min_gap,
                        startup_min_gap=self._follow_restart_start_min_gap,
                    )
                    lead_is_stopped_for_follow = (
                        acc_velocity_valid
                        and acc_lead_speed
                            < self._stopped_lead_speed_threshold
                    )
                    intentional_follow_stop_active = (
                        self._follow_only
                        and lead_is_stopped_for_follow
                        and acc_distance <= self._follow_desired_distance
                    )
                    if intentional_follow_stop_active:
                        ref_vel_kmph = 0.0
                        follow_restart_active = False
                        self.get_logger().info(
                            "[FollowOnlyStop] stopped lead reached the target "
                            f"gap: vehicle_id={acc_vehicle_id}, "
                            f"distance={acc_distance:.2f}m/"
                            f"{self._follow_desired_distance:.2f}m; "
                            "holding speed at zero.",
                            throttle_duration_sec=1.0,
                        )
                    if ego_is_stopped and lead_is_stopped_for_follow:
                        self._follow_stopped_vehicle_id = acc_vehicle_id
                        self._follow_restart_until = 0.0
                    elif (
                        ego_is_stopped
                        and not (
                            latched_follow_state is not None
                            and latched_follow_state.get("stale", False)
                        )
                        and acc_vehicle_id
                            == self._follow_stopped_vehicle_id
                        and acc_velocity_valid
                        and acc_lead_speed
                            >= self._follow_restart_lead_moving_speed
                        and acc_distance >= restart_min_gap
                    ):
                        self._follow_restart_until = (
                            current_time_sec + self._follow_restart_duration)
                        self._follow_stopped_vehicle_id = None
                        self.get_logger().info(
                            "[FollowRestart] lead started moving: "
                            f"vehicle_id={acc_vehicle_id}, "
                            f"lead_speed={acc_lead_speed:.2f}m/s, "
                            f"gap={acc_distance:.2f}m/"
                            f"{restart_min_gap:.2f}m, "
                            f"startup={startup_restart_waiting}",
                            throttle_duration_sec=1.0,
                        )

                    follow_restart_active = (
                        current_time_sec < self._follow_restart_until
                        and not (
                            latched_follow_state is not None
                            and latched_follow_state.get("stale", False)
                        )
                        and acc_distance >= restart_min_gap
                    )
                    if follow_restart_active:
                        follow_restart_target = min(
                            self._follow_restart_max_speed,
                            acc_lead_speed
                                + self._follow_restart_speed_margin,
                        )
                        ref_vel_kmph = max(
                            ref_vel_kmph, follow_restart_target)

            # --- Emergency Proximity Brake (waypoint-independent) ---
            # opponent_ahead (waypoint差ベースの検出) に依存せず、全V2X車両を直接スキャンする。
            # wp_diff=0 の場合など waypoint 検出をすり抜けても必ずブレーキがかかる。
            # Front/rear is determined on the Center arc. Ego yaw projection
            # changes sign in corners and must not suppress a relevant car.
            EMERGENCY_BRAKE_DIST = self._moving_emergency_preview_distance
            IMMEDIATE_EMERGENCY_BRAKE_DIST = 6.0
            emergency_stopped_blocker_id = None
            emergency_stopped_blocker_dist = float("inf")
            if hasattr(self, '_v2x_tracker'):
                active_vehicle_ids = self._v2x_tracker.active_vehicle_ids()
                active_vehicle_id_set = set(active_vehicle_ids)
                self._moving_vehicle_brake_bypass_since = {
                    vehicle_id: since
                    for vehicle_id, since
                    in self._moving_vehicle_brake_bypass_since.items()
                    if vehicle_id in active_vehicle_id_set
                }
                self._center_path_collision_hazard_until = {
                    vehicle_id: until
                    for vehicle_id, until
                    in self._center_path_collision_hazard_until.items()
                    if vehicle_id in active_vehicle_id_set
                    and until > current_time_sec
                }
                moving_bypass_candidates = set()
                # Pre-confirm moving traffic and arm one simple dynamic-gap
                # latch before the old fixed 6 m emergency point.
                for vehicle_id in active_vehicle_ids:
                    preview_buf = self._v2x_tracker._samples.get(vehicle_id)
                    if not preview_buf:
                        continue
                    _, preview_x, preview_y = preview_buf[-1]
                    preview_distance = math.hypot(
                        preview_x - pose.x, preview_y - pose.y)
                    preview_longitudinal = self._center_longitudinal_between(
                        pose.x, pose.y, preview_x, preview_y)
                    preview_is_relevant = bool(
                        preview_longitudinal is not None
                        and 0.0 < preview_longitudinal
                        <= self._moving_emergency_preview_distance
                        and preview_distance
                        <= self._moving_emergency_preview_distance
                    )
                    center_path_prediction = (
                        self._center_path_collision_prediction(vehicle_id)
                        if preview_is_relevant else None
                    )
                    if (
                        center_path_prediction is not None
                        and center_path_prediction.get("collision", False)
                    ):
                        self._center_path_collision_hazard_until[vehicle_id] = (
                            current_time_sec
                            + self._center_path_collision_hazard_hold_sec
                        )
                        self.get_logger().warn(
                            "[CenterPathCollisionPrediction] opponent motion "
                            "along Center intersects the ego MPC envelope; "
                            "latching gentle speed matching: "
                            f"vehicle_id={vehicle_id}, ttc="
                            f"{center_path_prediction['time']:.2f}s, "
                            f"hold_sec="
                            f"{self._center_path_collision_hazard_hold_sec:.2f}",
                            throttle_duration_sec=0.5,
                        )
                    if preview_is_relevant:
                        _, preview_envelope = (
                            self._current_center_envelopes_are_separated(
                                pose, vehicle_id)
                        )
                        ego_frenet = self._center_frenet(pose.x, pose.y)
                        opponent_frenet = self._center_frenet(
                            preview_x, preview_y)
                        center_lateral_clearance = None
                        if ego_frenet is not None and opponent_frenet is not None:
                            center_lateral_clearance = lateral_vehicle_clearance(
                                abs(opponent_frenet[1] - ego_frenet[1]),
                                float(self._cfg.bicycle_model.width),
                                self._v2x_parallel_vehicle_half_width,
                            )
                        preview_vx, preview_vy = self._v2x_tracker.velocity(
                            vehicle_id)
                        preview_speed = math.hypot(preview_vx, preview_vy)
                        closing_speed = max(abs(float(v)) - preview_speed, 0.0)
                        required_gap = (
                            self._moving_emergency_desired_distance
                            + self._moving_emergency_reaction_sec * closing_speed
                            + closing_speed ** 2
                            / (2.0
                               * self._moving_emergency_available_deceleration)
                        )
                        dynamic_gap_hazard = bool(
                            preview_envelope is not None
                            and center_lateral_clearance is not None
                            and center_lateral_clearance
                                <= self._parallel_warning_clearance
                            and preview_envelope["arc_gap"] <= required_gap
                        )
                        if dynamic_gap_hazard:
                            self._center_path_collision_hazard_until[
                                vehicle_id] = (
                                    current_time_sec
                                    + self._center_path_collision_hazard_hold_sec
                                )
                            self.get_logger().warn(
                                "[DynamicGapHazard] Center body gap is below "
                                "the braking-distance requirement; holding one "
                                "continuous speed limit: "
                                f"vehicle_id={vehicle_id}, gap="
                                f"{preview_envelope['arc_gap']:.2f}m/"
                                f"{required_gap:.2f}m, closing="
                                f"{closing_speed:.2f}m/s",
                                throttle_duration_sec=0.5,
                            )
                    if (
                        preview_is_relevant
                        and vehicle_id not in (
                            self._center_path_collision_hazard_until)
                        and self._moving_vehicle_will_clear_after_brief_conflict(
                            vehicle_id, pose, v)
                    ):
                        moving_bypass_candidates.add(vehicle_id)
                        self._moving_vehicle_brake_bypass_since.setdefault(
                            vehicle_id, current_time_sec)
                    else:
                        self._moving_vehicle_brake_bypass_since.pop(
                            vehicle_id, None)
                for vid in active_vehicle_ids:
                    if vid in slow_pass_release_ids:
                        self._center_path_collision_hazard_until.pop(vid,None)
                        continue
                    buf = self._v2x_tracker._samples.get(vid)
                    if buf:
                        _, opp_x, opp_y = buf[-1]
                        dx = opp_x - pose.x
                        dy = opp_y - pose.y
                        dist = math.hypot(dx, dy)
                        if dist < EMERGENCY_BRAKE_DIST and dist > 0.01:
                            if (
                                dist >= IMMEDIATE_EMERGENCY_BRAKE_DIST
                                and self._center_path_collision_hazard_until.get(
                                    vid, 0.0) <= current_time_sec
                            ):
                                # The wider scan exists only to act on an
                                # already established dynamic/predicted risk.
                                # Ordinary traffic keeps the original 6 m
                                # proximity behavior.
                                continue
                            center_longitudinal = self._center_longitudinal_between(
                                pose.x, pose.y, opp_x, opp_y)
                            if (
                                center_longitudinal is not None
                                and 0.0 < center_longitudinal
                                <= EMERGENCY_BRAKE_DIST
                            ):
                                fwd_dot = center_longitudinal / dist
                                opp_wp_id = self._car.get_closest_waypoint(opp_x, opp_y)
                                opp_wp = self._reference_path.get_waypoint(opp_wp_id)
                                if opp_wp.normal_angle is not None:
                                    nx = -math.cos(opp_wp.normal_angle)
                                    ny = -math.sin(opp_wp.normal_angle)
                                else:
                                    normal_angle = opp_wp.psi + math.pi / 2.0
                                    nx = math.cos(normal_angle)
                                    ny = math.sin(normal_angle)

                                ego_lateral_offset = (
                                    (pose.x - opp_wp.x) * nx
                                    + (pose.y - opp_wp.y) * ny
                                )
                                opp_lateral_offset = (
                                    (opp_x - opp_wp.x) * nx
                                    + (opp_y - opp_wp.y) * ny
                                )
                                lateral_dist = abs(
                                    opp_lateral_offset - ego_lateral_offset)
                                lanes = self._reference_path.get_lane_bounds(opp_wp_id)

                                def lane_index(offset):
                                    for index, (ub_lane, lb_lane) in enumerate(lanes):
                                        if lb_lane <= offset <= ub_lane:
                                            return index
                                    return None

                                ego_lane_idx = lane_index(ego_lateral_offset)
                                opp_lane_idx = lane_index(opp_lateral_offset)
                                same_lane = (
                                    ego_lane_idx is not None
                                    and ego_lane_idx == opp_lane_idx
                                )
                                lateral_clearance = lateral_vehicle_clearance(
                                    lateral_dist,
                                    float(self._cfg.bicycle_model.width),
                                    self._v2x_parallel_vehicle_half_width,
                                )
                                lateral_envelopes_conflict = bool(
                                    lateral_clearance
                                    <= self._parallel_critical_clearance
                                )
                                center_path_hazard_active = bool(
                                    self._center_path_collision_hazard_until.get(
                                        vid, 0.0) > current_time_sec
                                )

                                # Lane labels overlap near boundaries and are
                                # not sufficient evidence for braking.
                                if (
                                    not lateral_envelopes_conflict
                                    and not center_path_hazard_active
                                ):
                                    self.get_logger().info(
                                        "[EmergencyBrakeSkip] "
                                        f"vehicle_id={vid} ego_lane={ego_lane_idx} "
                                        f"opp_lane={opp_lane_idx} "
                                        f"lateral={lateral_dist:.2f}m, "
                                        f"clearance={lateral_clearance:+.2f}m",
                                        throttle_duration_sec=1.0,
                                    )
                                    continue

                                opp_vx, opp_vy = self._v2x_tracker.velocity(vid)
                                opp_spd = math.hypot(opp_vx, opp_vy)
                                (
                                    current_envelopes_separated,
                                    current_envelope_state,
                                ) = self._current_center_envelopes_are_separated(
                                    pose, vid)
                                if (
                                    forced_overtake_prediction_clear
                                    and vid == self._forced_overtake_vehicle_id
                                    and current_envelopes_separated
                                ):
                                    # Bypass braking only when this cycle has a
                                    # valid outer-lane-constrained prediction
                                    # and the current corner-aware envelopes
                                    # both stay clear of the stopped target.
                                    continue
                                fresh_mpc_prediction = bool(
                                    self._mpc.current_prediction is not None
                                    and self._mpc.infeasibility_counter == 0
                                    and not self._mpc.used_prediction_fallback
                                    and not self._mpc.recovery_requested
                                )
                                if (
                                    center_path_hazard_active
                                    and self._fresh_outer_prediction_releases_center_stop(
                                        vid, current_envelope_state)
                                ):
                                    self._center_path_collision_hazard_until.pop(vid, None)
                                    self.get_logger().info(
                                        f"[FreshOuterEmergencyRelease] vehicle_id={vid}, "
                                        f"target={self._overtake.target_id}, "
                                        f"lane=L{self._reference_path.target_lane_idx}; "
                                        "lateral separation and both fresh moving predictions are clear",
                                        throttle_duration_sec=0.5)
                                    continue
                                if (
                                    fresh_mpc_prediction
                                    and current_envelopes_separated
                                    and not center_path_hazard_active
                                    and self._prediction_is_clear_of_vehicle(vid)
                                ):
                                    self.get_logger().info(
                                        "[EmergencyBrakePredictionSkip] fresh "
                                        "MPC prediction passes the vehicle "
                                        f"safely: vehicle_id={vid}, "
                                        f"center_arc="
                                        f"{current_envelope_state['arc_delta']:+.2f}m, "
                                        f"arc_gap="
                                        f"{current_envelope_state['arc_gap']:+.2f}m, "
                                        f"body_lateral_gap="
                                        f"{current_envelope_state['lateral_gap']:+.2f}m, "
                                        f"rect_overlap="
                                        f"{current_envelope_state['rectangles_overlap']}, "
                                        f"overlap_kind={current_envelope_state.get('overlap_kind','unknown')}, "
                                        f"yaw_known={current_envelope_state.get('yaw_known',False)}",
                                        throttle_duration_sec=1.0,
                                    )
                                    continue
                                if (
                                    fresh_mpc_prediction
                                    and not current_envelopes_separated
                                    and current_envelope_state is not None
                                ):
                                    self.get_logger().warn(
                                        "[EmergencyBrakePredictionCurrentEnvelopeBlock] "
                                        "future MPC clearance cannot override the "
                                        "current corner-aware body envelope: "
                                        f"vehicle_id={vid}, center_arc="
                                        f"{current_envelope_state['arc_delta']:+.2f}m, "
                                        f"arc_gap="
                                        f"{current_envelope_state['arc_gap']:+.2f}m, "
                                        f"body_lateral_gap="
                                        f"{current_envelope_state['lateral_gap']:+.2f}m, "
                                        f"rect_overlap="
                                        f"{current_envelope_state['rectangles_overlap']}, "
                                        f"overlap_kind={current_envelope_state.get('overlap_kind','unknown')}, "
                                        f"yaw_known={current_envelope_state.get('yaw_known',False)}",
                                        throttle_duration_sec=1.0,
                                    )
                                moving_vehicle_will_clear = (
                                    fresh_mpc_prediction
                                    and current_envelopes_separated
                                    and not center_path_hazard_active
                                    and self._moving_vehicle_will_clear_after_brief_conflict(
                                        vid, pose, v)
                                )
                                moving_clear_since = (
                                    self._moving_vehicle_brake_bypass_since.get(
                                        vid)
                                )
                                if moving_vehicle_will_clear:
                                    moving_bypass_candidates.add(vid)
                                    if moving_clear_since is None:
                                        moving_clear_since = current_time_sec
                                        self._moving_vehicle_brake_bypass_since[
                                            vid] = moving_clear_since
                                    moving_clear_confirmed = (
                                        current_time_sec - moving_clear_since
                                        >= self._moving_vehicle_brake_bypass_confirm_sec
                                    )
                                else:
                                    self._moving_vehicle_brake_bypass_since.pop(
                                        vid, None)
                                    moving_clear_confirmed = False
                                if moving_clear_confirmed:
                                    self.get_logger().info(
                                        "[EmergencyBrakeMovingVehicleSkip] "
                                        "fast vehicle remains clear after the "
                                        "brief conflict window: "
                                        f"vehicle_id={vid}, speed={opp_spd:.2f}m/s, "
                                        f"confirm_sec="
                                        f"{self._moving_vehicle_brake_bypass_confirm_sec:.2f}, "
                                        f"horizon_sec="
                                        f"{self._moving_vehicle_brake_bypass_horizon_sec:.2f}",
                                        throttle_duration_sec=1.0,
                                    )
                                    continue
                                moving_target = bool(
                                    self._v2x_tracker.has_velocity_estimate(vid)
                                    and opp_spd
                                    >= self._stopped_lead_speed_threshold
                                )
                                arc_vehicle_gap = (
                                    current_envelope_state["arc_gap"]
                                    if current_envelope_state is not None
                                    else longitudinal_vehicle_clearance(
                                        center_longitudinal,
                                        self._parallel_ego_half_length,
                                        self._parallel_vehicle_half_length,
                                    )
                                )
                                committed_shadow_lane = (
                                    int(self._overtake.verification.lane_idx)
                                    if self._overtake.verification.lane_idx in (0, 2)
                                    else None
                                )
                                fresh_probe_lane = (
                                    int(self._overtake.probe.lane_idx)
                                    if (
                                        self._overtake.probe.lane_idx in (0, 2)
                                        and self._overtake_commit_probe_is_fresh(
                                            vid,
                                            self._overtake.probe.lane_idx,
                                            current_time_sec,
                                        )
                                    )
                                    else None
                                )
                                verified_outer_lane = (
                                    committed_shadow_lane
                                    if committed_shadow_lane in (0, 2)
                                    else fresh_probe_lane
                                )
                                committed_outer_prediction_clear = bool(
                                    committed_shadow_lane in (0, 2)
                                    and committed_shadow_lane
                                        == self._overtake.requested_lane
                                    and self._overtake.committed
                                    and fresh_mpc_prediction
                                    and self._prediction_is_clear_of_vehicle(vid)
                                )
                                precommit_outer_prediction_clear = bool(
                                    fresh_probe_lane in (0, 2)
                                    and not self._prepass_fallback_recovery_active
                                    and not self._mpc_safety_recovery_active
                                )
                                outer_bypass_target_matches = (
                                    outer_prediction_bypass_target_matches(
                                        vehicle_id=vid,
                                        latched_target_id=(
                                            self._overtake.target_id),
                                        handoff_target_id=(
                                            self._consecutive_overtake_handoff_target_id),
                                        handoff_lane_idx=(
                                            self._consecutive_overtake_handoff_lane_idx),
                                        verified_outer_lane=(
                                            verified_outer_lane),
                                    )
                                )
                                shadow_outer_prediction_clear = bool(
                                    center_path_hazard_active
                                    and outer_bypass_target_matches
                                    and current_envelopes_separated
                                    and (
                                        committed_outer_prediction_clear
                                        or precommit_outer_prediction_clear
                                    )
                                )
                                if shadow_outer_prediction_clear:
                                    # Center-path prediction assumes ego stays
                                    # on Center and can therefore keep matching
                                    # a slow lead after a strict outer pass is
                                    # already committed. Ignore only that stale
                                    # Center hazard when the live outer-lane MPC
                                    # and the current rotated body envelopes are
                                    # both clear. A genuinely short dynamic gap
                                    # remains protected by the normal emergency
                                    # branch below.
                                    closing_speed = max(
                                        abs(float(v)) - opp_spd, 0.0)
                                    required_dynamic_gap = (
                                        self._moving_emergency_desired_distance
                                        + self._moving_emergency_reaction_sec
                                            * closing_speed
                                        + closing_speed ** 2
                                        / (2.0
                                           * self._moving_emergency_available_deceleration)
                                    )
                                    if arc_vehicle_gap > required_dynamic_gap:
                                        center_path_hazard_active = False
                                        self.get_logger().info(
                                            "[CenterPathPredictionOuterBypass] "
                                            "strict Shadow-verified outer MPC "
                                            "passes the active/handoff target and the current "
                                            "dynamic gap is safe; suppressing "
                                            "Center-only speed matching: "
                                            f"vehicle_id={vid}, lane=L"
                                            f"{verified_outer_lane}, gap="
                                            f"{arc_vehicle_gap:.2f}m/"
                                            f"{required_dynamic_gap:.2f}m",
                                            throttle_duration_sec=0.5,
                                        )
                                moving_speed_match = bool(
                                    moving_target
                                    and (
                                        center_path_hazard_active
                                        or arc_vehicle_gap
                                        > self._moving_emergency_critical_distance
                                    )
                                )
                                if moving_speed_match:
                                    # A moving lead should normally be matched,
                                    # not treated like a stopped wall. The old
                                    # 5.5 m / Kp=1.5 rule could command several
                                    # m/s below the lead and immediately lose
                                    # the draft even when relative speed was
                                    # already small.
                                    if center_path_hazard_active:
                                        closing_speed = max(
                                            abs(float(v)) - opp_spd, 0.0)
                                        usable_gap = max(
                                            arc_vehicle_gap
                                            - self._moving_emergency_desired_distance
                                            - self._moving_emergency_reaction_sec
                                                * closing_speed,
                                            0.0,
                                        )
                                        v_ref_emg = math.sqrt(
                                            opp_spd ** 2
                                            + 2.0
                                            * self._moving_emergency_available_deceleration
                                            * usable_gap
                                        )
                                    else:
                                        v_ref_emg = (
                                            opp_spd
                                            + self._moving_emergency_spacing_kp
                                            * (
                                                arc_vehicle_gap
                                                - self._moving_emergency_desired_distance
                                            )
                                        )
                                        v_ref_emg = max(
                                            opp_spd
                                            - self._moving_emergency_max_speed_deficit,
                                            v_ref_emg,
                                        )
                                    emergency_mode = (
                                        "center_path_prediction_match"
                                        if center_path_hazard_active
                                        else "moving_speed_match"
                                    )
                                else:
                                    d_target_emg = 5.5
                                    K_p_emg = 1.5
                                    v_ref_emg = (
                                        opp_spd
                                        + K_p_emg
                                        * (arc_vehicle_gap - d_target_emg)
                                    )
                                    emergency_mode = "critical_or_stopped"
                                stopped_vehicle_too_close = (
                                    opp_spd < self._stopped_lead_speed_threshold
                                    and arc_vehicle_gap
                                    <= self._close_obstacle_reverse_distance
                                )
                                if (
                                    stopped_vehicle_too_close
                                    and not (hybrid_escape_speeds.get(vid, 0.0) > 0.0)
                                    and same_lane
                                    and self._v2x_tracker.has_velocity_estimate(vid)
                                    and arc_vehicle_gap
                                    < emergency_stopped_blocker_dist
                                ):
                                    emergency_stopped_blocker_id = vid
                                    emergency_stopped_blocker_dist = arc_vehicle_gap
                                min_emergency_speed = (
                                    0.0 if stopped_vehicle_too_close else 0.5)
                                v_ref_emg = max(min_emergency_speed, v_ref_emg)
                                committed_lane_idx = (
                                    int(self._overtake.requested_lane)
                                    if self._overtake.requested_lane in (0, 2)
                                    else None
                                )
                                committed_target_matches = bool(
                                    vid == self._overtake.target_id
                                    and vid == self._overtake.verification.vehicle_id
                                    and committed_lane_idx
                                        == self._overtake.verification.lane_idx
                                )
                                commit_creep_conflicts = {
                                    "front": [], "side": [], "rear": []}
                                commit_creep_passable = False
                                if committed_target_matches:
                                    live_passage, _ = self._latched_target_passage(
                                        pose)
                                    commit_creep_passable = bool(
                                        live_passage.get(
                                            committed_lane_idx, False))
                                    commit_creep_conflicts = classify_lane_conflicts(
                                        committed_lane_idx,
                                        self._relative_lane_vehicle_samples(
                                            pose, v),
                                        front_distance=(
                                            self._prepass_lane_fallback_front_distance),
                                        side_distance=(
                                            self._prepass_lane_fallback_side_distance),
                                        rear_distance=(
                                            self._prepass_lane_fallback_rear_distance),
                                    )
                                strict_commit_creep = (
                                    strict_shadow_slow_commit_creep_allowed(
                                        target_matches=committed_target_matches,
                                        shadow_verified=bool(
                                            self._overtake.verification.vehicle_id
                                                is not None),
                                        committed_outer_lane=bool(
                                            self._overtake.committed
                                            and committed_lane_idx in (0, 2)
                                            and self._reference_path.is_overtaking),
                                        target_is_slow=bool(
                                            self._v2x_tracker.has_velocity_estimate(
                                                vid)
                                            and opp_spd
                                                <= self._strict_shadow_commit_creep_max_target_speed),
                                        current_envelopes_separated=bool(
                                            current_envelopes_separated),
                                        candidate_passable=(
                                            commit_creep_passable),
                                        candidate_conflicts=(
                                            commit_creep_conflicts),
                                    )
                                )
                                if strict_commit_creep:
                                    # Break the zero-speed/lateral-no-progress
                                    # loop without bypassing collision safety.
                                    # Later ParallelSafety and recovery layers
                                    # may still impose a stricter command.
                                    v_ref_emg = max(
                                        v_ref_emg,
                                        self._strict_shadow_stopped_commit_creep_speed,
                                    )
                                    emergency_mode = (
                                        "strict_shadow_stopped_commit_creep")
                                    self.get_logger().warn(
                                        "[EmergencyBrakeCommitCreep] strict "
                                        "Shadow-verified slow-car pass remains "
                                        "physically and dynamically clear; replacing "
                                        "the emergency full stop with low-speed "
                                        f"creep: vehicle_id={vid}, lane=L"
                                        f"{committed_lane_idx}, speed="
                                        f"{min(ref_vel_kmph, v_ref_emg):.2f}m/s, conflicts="
                                        f"{commit_creep_conflicts}",
                                        throttle_duration_sec=0.5,
                                    )
                                if hybrid_escape_speeds.get(vid, 0.0) > 0.0:
                                    v_ref_emg = max(v_ref_emg, hybrid_escape_speeds[vid])
                                    if not strict_commit_creep:
                                        emergency_mode = "hybrid_lateral_escape_creep"
                                # Apply only this vehicle's adjusted cap. A preceding
                                # vehicle's stop or ACC cap must survive iteration order.
                                ref_vel_kmph = min(ref_vel_kmph, v_ref_emg)
                                emergency_brake_active = True
                                if dist < emergency_brake_vehicle_distance:
                                    emergency_brake_vehicle_distance = dist
                                    emergency_brake_vehicle_id = vid
                                self.get_logger().warn(
                                    f"[EmergencyBrake] vehicle_id={vid} "
                                    f"forward obstacle at center_arc="
                                    f"{center_longitudinal:.2f}m, "
                                    f"body_gap={arc_vehicle_gap:+.2f}m "
                                    f"(dot={fwd_dot:.2f}, lateral={lateral_dist:.2f}m, "
                                    f"clearance={lateral_clearance:+.2f}m, "
                                    f"ego_lane={ego_lane_idx}, opp_lane={opp_lane_idx}). "
                                    f"opp_speed={opp_spd:.2f}m/s, "
                                    f"mode={emergency_mode}. "
                                    f"Speed → {ref_vel_kmph:.2f}m/s",
                                    throttle_duration_sec=0.5
                                )

                self._moving_vehicle_brake_bypass_since = {
                    vehicle_id: since
                    for vehicle_id, since
                    in self._moving_vehicle_brake_bypass_since.items()
                    if vehicle_id in moving_bypass_candidates
                }

            if (
                emergency_stopped_blocker_id is not None
                and not startup_overtake_suppressed
                and not prestart_reverse_suppressed
            ):
                self._arm_emergency_blocker_recovery(
                    emergency_stopped_blocker_id, pose, v)

            if (
                not startup_overtake_suppressed
                and not prestart_reverse_suppressed
            ):
                self._update_follow_deadlock_escape(
                    now_sec=current_time_sec,
                    pose=pose,
                    ego_speed=v,
                    follow_active=follow_control_active,
                    target_id=(acc_vehicle_id if follow_control_active else None),
                    lead_speed=(
                        acc_lead_speed if follow_control_active else 99999.0),
                    forward_command=max(float(u[0]), 0.0),
                    emergency_brake_active=emergency_brake_active,
                    emergency_brake_vehicle_id=emergency_brake_vehicle_id,
                )

            # --- Post-Overtake Cooldown: 追い越し後クールダウン中の後方車両監視 ---
            # Center→Race に切り替わった直後は、後方の近接車との衝突リスクが高い。
            # Race軌道がコーナーインを攻めて後方の相手と交差しないよう、
            # クールダウン期間中は後方の近接車との距離に応じて速度を抑制する。
            RACE_RETURN_COOLDOWN_SEC = 3.0  # [s] 追い越し完了後に速度抑制する時間

            # Center→Race 切り替えを検出してクールダウンタイマーを開始
            if trajectory_switched and not opponent_ahead_detected:
                self._race_return_time = current_time_sec
                self.get_logger().info(
                    "[PostOvertake] Cooldown started after returning to Race "
                    f"trajectory: vehicle_id={self._post_overtake_vehicle_id}."
                )

            in_post_overtake_cooldown = (
                self._race_return_time is not None and
                current_time_sec - self._race_return_time < RACE_RETURN_COOLDOWN_SEC
            )

            post_vehicle_id = self._post_overtake_vehicle_id
            if (
                in_post_overtake_cooldown
                and post_vehicle_id is not None
                and hasattr(self, '_v2x_tracker')
            ):
                buf = self._v2x_tracker._samples.get(post_vehicle_id)
                if buf:
                    _, opp_x, opp_y = buf[-1]
                    longitudinal = self._center_longitudinal_between(
                        pose.x, pose.y, opp_x, opp_y)
                    # Negative arc distance means the latched vehicle is behind.
                    if longitudinal is not None and longitudinal < 0.0:
                        behind_dist = -longitudinal
                        if behind_dist < 12.0:
                            opp_vx, opp_vy = self._v2x_tracker.velocity(
                                post_vehicle_id)
                            opp_speed = math.hypot(opp_vx, opp_vy)
                            d_behind_target = 8.0
                            K_p_behind = 0.8
                            v_ref_behind = (
                                opp_speed
                                + K_p_behind * (behind_dist - d_behind_target)
                            )
                            v_ref_behind = max(2.0, v_ref_behind)
                            ref_vel_kmph = min(ref_vel_kmph, v_ref_behind)
                            if self._loop % int(self._mpc_cfg.control_rate) == 0:
                                self.get_logger().info(
                                    "[PostOvertake] "
                                    f"vehicle_id={post_vehicle_id}, "
                                    f"longitudinal={longitudinal:.1f}m, "
                                    f"limiting speed to {ref_vel_kmph:.2f}m/s",
                                    throttle_duration_sec=1.0,
                                )
            elif self._race_return_time is not None and not in_post_overtake_cooldown:
                self._race_return_time = None
                self._post_overtake_vehicle_id = None

            # --- Parallel Running Safety Control (並走接近制御) ---
            # Safety uses a wider rear window and always remains active.
            # Abort uses a narrower window and may be suppressed independently.
            parallel_safety_candidate = None
            parallel_abort_candidate = None
            ego_center_frenet = self._center_frenet(pose.x, pose.y)
            ego_parallel_lane_idx = (
                None if ego_center_frenet is None
                else self._center_lane_index_for_offset(
                    pose.x, pose.y, ego_center_frenet[1])
            )
            if (
                hasattr(self, '_v2x_tracker')
                and ego_center_frenet is not None
            ):
                ego_center_s, ego_center_lateral = ego_center_frenet
                for vid in self._v2x_tracker.active_vehicle_ids():
                    buf = self._v2x_tracker._samples.get(vid)
                    if buf:
                        _, opp_x, opp_y = buf[-1]
                        opp_center_frenet = self._center_frenet(opp_x, opp_y)
                        if opp_center_frenet is None:
                            continue
                        opp_center_s, opp_center_lateral = opp_center_frenet
                        parallel_arc_delta = signed_closed_path_arc_distance(
                            ego_center_s,
                            opp_center_s,
                            self._center_arc_total_length,
                        )
                        if parallel_arc_delta is None:
                            continue
                        # Front/rear filtering must use the same Center arc as
                        # the final parallel-envelope test. Ego-heading
                        # projection changes sign in corners and previously
                        # discarded relevant vehicles before this test.
                        if (
                            parallel_arc_delta
                                < -max(
                                    self._parallel_safety_arc_behind,
                                    self._parallel_abort_arc_behind,
                                )
                            or parallel_arc_delta > max(
                                self._parallel_safety_arc_ahead,
                                self._parallel_abort_arc_ahead,
                            )
                        ):
                            continue
                        lateral_center_distance = abs(
                            opp_center_lateral - ego_center_lateral)
                        lateral_clearance = lateral_vehicle_clearance(
                            lateral_center_distance,
                            float(self._cfg.bicycle_model.width),
                            self._v2x_parallel_vehicle_half_width,
                        )
                        longitudinal_clearance = (
                            longitudinal_vehicle_clearance(
                                parallel_arc_delta,
                                self._parallel_ego_half_length,
                                self._parallel_vehicle_half_length,
                            )
                        )
                        other_lane_idx = self._center_lane_index_for_offset(
                            opp_x, opp_y, opp_center_lateral)
                        arc_distance = parallel_arc_delta
                        longitudinal_d = parallel_arc_delta
                        candidate = {
                            "vehicle_id": vid,
                            "lane_idx": other_lane_idx,
                            "lateral_center": lateral_center_distance,
                            "clearance": lateral_clearance,
                            "longitudinal_clearance": (
                                longitudinal_clearance),
                            "longitudinal": longitudinal_d,
                            "arc": arc_distance,
                        }

                        safety_arc_ok = (
                            parallel_arc_delta is None
                            or -self._parallel_safety_arc_behind
                                <= parallel_arc_delta
                                <= self._parallel_safety_arc_ahead
                        )
                        if (
                            self._parallel_safety_enabled
                            and
                            safety_arc_ok
                            and is_parallel_vehicle(
                                ego_lane_idx=ego_parallel_lane_idx,
                                other_lane_idx=other_lane_idx,
                                lateral_clearance=lateral_clearance,
                                longitudinal_clearance=(
                                    longitudinal_clearance),
                                longitudinal_distance=longitudinal_d,
                                maximum_lateral_clearance=(
                                    self._parallel_warning_clearance),
                                maximum_longitudinal_clearance=(
                                    self._parallel_safety_longitudinal_clearance
                                ),
                                minimum_longitudinal_distance=(
                                    -self._parallel_safety_lon_behind),
                                maximum_longitudinal_distance=(
                                    self._parallel_safety_lon_ahead),
                            )
                            and (
                                parallel_safety_candidate is None
                                or lateral_clearance
                                    < parallel_safety_candidate["clearance"]
                            )
                        ):
                            parallel_safety_candidate = candidate

                        abort_arc_ok = (
                            parallel_arc_delta is None
                            or -self._parallel_abort_arc_behind
                                <= parallel_arc_delta
                                <= self._parallel_abort_arc_ahead
                        )
                        if (
                            vid != self._forced_overtake_vehicle_id
                            and abort_arc_ok
                            and is_parallel_vehicle(
                                ego_lane_idx=ego_parallel_lane_idx,
                                other_lane_idx=other_lane_idx,
                                lateral_clearance=lateral_clearance,
                                longitudinal_clearance=(
                                    longitudinal_clearance),
                                longitudinal_distance=longitudinal_d,
                                maximum_lateral_clearance=(
                                    self._parallel_warning_clearance),
                                maximum_longitudinal_clearance=(
                                    self._parallel_abort_longitudinal_clearance
                                ),
                                minimum_longitudinal_distance=(
                                    -self._parallel_abort_lon_behind),
                                maximum_longitudinal_distance=(
                                    self._parallel_abort_lon_ahead),
                            )
                            and (
                                parallel_abort_candidate is None
                                or lateral_clearance
                                    < parallel_abort_candidate["clearance"]
                            )
                        ):
                            parallel_abort_candidate = candidate

            parallel_abort_suppressed = (
                self._stuck_recovery_until is not None
                or self._post_reverse_full_width_recovery_active
                or self._mpc_safety_recovery_active
                or self._prepass_fallback_recovery_active
                or forced_overtake_active
            )

            # Slowdown is independent from Abort suppression, including while
            # forcibly overtaking a stopped vehicle.
            if parallel_safety_candidate is not None:
                safety = parallel_safety_candidate
                clearance = safety["clearance"]
                safety_vehicle_id = safety["vehicle_id"]
                fresh_overtake_prediction_clear = bool(
                    safety_vehicle_id == self._overtake.target_id
                    and self._reference_path.is_overtaking
                    and self._reference_path.target_lane_idx in (0, 2)
                    and clearance > 0.0
                    and not self._mpc.recovery_requested
                    and self._mpc.infeasibility_counter == 0
                    and self._mpc.current_prediction is not None
                    and not self._mpc.used_prediction_fallback
                    and self._prediction_is_clear_of_vehicle(
                        safety_vehicle_id)
                )
                overtake_speed_floor = 0.0
                if fresh_overtake_prediction_clear:
                    opp_vx, opp_vy = self._v2x_tracker.velocity(
                        safety_vehicle_id)
                    overtake_speed_floor = (
                        math.hypot(opp_vx, opp_vy)
                        + self._parallel_overtake_speed_margin
                    )
                moving_parallel_speed_floor = 0.0
                if (
                    self._v2x_tracker.has_velocity_estimate(safety_vehicle_id)
                    and safety["longitudinal_clearance"] > 0.0
                ):
                    opp_vx, opp_vy = self._v2x_tracker.velocity(
                        safety_vehicle_id)
                    parallel_opp_speed = math.hypot(opp_vx, opp_vy)
                    if parallel_opp_speed >= self._stopped_lead_speed_threshold:
                        # The vehicles are laterally close but their
                        # longitudinal envelopes have not overlapped. Matching
                        # a moving opponent is sufficient to stop further
                        # closure; dropping far below its speed only loses the
                        # pass. No floor is used after longitudinal overlap.
                        moving_parallel_speed_floor = max(
                            parallel_opp_speed
                            - self._moving_emergency_max_speed_deficit,
                            0.0,
                        )
                parallel_speed_floor = max(
                    overtake_speed_floor, moving_parallel_speed_floor)
                overtake_relaxation_note = (
                    f", moving_floor={parallel_speed_floor:.2f}m/s"
                    if parallel_speed_floor > 0.0 else ""
                )
                if clearance <= self._parallel_critical_clearance:
                    ratio = float(np.clip(
                        max(clearance, 0.0)
                        / max(self._parallel_critical_clearance, 1e-6),
                        0.0, 1.0))
                    v_ref_parallel = ref_vel_kmph * (0.4 + 0.3 * ratio)
                    if parallel_speed_floor > 0.0:
                        # A positive lateral envelope gap plus a fresh,
                        # collision-free MPC pass permits finishing the pass.
                        # Never apply this floor to actual envelope overlap.
                        v_ref_parallel = max(
                            v_ref_parallel, parallel_speed_floor)
                    ref_vel_kmph = min(ref_vel_kmph, v_ref_parallel)
                    self.get_logger().warn(
                        f"[ParallelSafety] CRITICAL: vehicle_id={safety['vehicle_id']} "
                        f"lane=L{safety['lane_idx']} "
                        f"lat={safety['lateral_center']:.2f}m, "
                        f"clearance={clearance:+.2f}m, "
                        f"lon_clearance="
                        f"{safety['longitudinal_clearance']:+.2f}m, "
                        f"lon={safety['longitudinal']:+.2f}m, "
                        f"arc={safety['arc']:+.2f}m, "
                        f"speed limited to {ref_vel_kmph:.2f}m/s"
                        f"{overtake_relaxation_note}",
                        throttle_duration_sec=1.0
                    )
                else:
                    ratio = (
                        (clearance - self._parallel_critical_clearance)
                        / max(
                            self._parallel_warning_clearance
                            - self._parallel_critical_clearance,
                            1e-6,
                        )
                    )
                    v_ref_parallel = ref_vel_kmph * (
                        0.7 + 0.3 * float(np.clip(ratio, 0.0, 1.0)))
                    if parallel_speed_floor > 0.0:
                        v_ref_parallel = max(
                            v_ref_parallel, parallel_speed_floor)
                    ref_vel_kmph = min(ref_vel_kmph, v_ref_parallel)
                    self.get_logger().info(
                        f"[ParallelSafety] WARNING: vehicle_id={safety['vehicle_id']} "
                        f"lane=L{safety['lane_idx']} "
                        f"lat={safety['lateral_center']:.2f}m, "
                        f"clearance={clearance:+.2f}m, "
                        f"lon_clearance="
                        f"{safety['longitudinal_clearance']:+.2f}m, "
                        f"lon={safety['longitudinal']:+.2f}m, "
                        f"arc={safety['arc']:+.2f}m, "
                        f"speed limited to {ref_vel_kmph:.2f}m/s"
                        f"{overtake_relaxation_note}",
                        throttle_duration_sec=1.0
                    )

            abort_vehicle_id = (
                None if parallel_abort_candidate is None
                else parallel_abort_candidate["vehicle_id"]
            )
            if abort_vehicle_id is None:
                self._parallel_timer_vehicle_id = None
                self._parallel_start_time = None
                parallel_duration = 0.0
            elif abort_vehicle_id != self._parallel_timer_vehicle_id:
                if self._parallel_timer_vehicle_id is not None:
                    self.get_logger().info(
                        "[ParallelAbortTimer] target changed; resetting timer: "
                        f"old={self._parallel_timer_vehicle_id}, "
                        f"new={abort_vehicle_id}"
                    )
                self._parallel_timer_vehicle_id = abort_vehicle_id
                self._parallel_start_time = current_time_sec
                parallel_duration = 0.0
            elif parallel_abort_suppressed:
                # Recovery owns motion; no Abort duration is accumulated.
                self._parallel_start_time = current_time_sec
                parallel_duration = 0.0
            else:
                if self._parallel_start_time is None:
                    self._parallel_start_time = current_time_sec
                parallel_duration = current_time_sec - self._parallel_start_time

            stationary_parallel_group = (
                self._stationary_lane_group(pose, v, self._target_lane_idx)
                if (parallel_abort_candidate is not None
                    and parallel_duration > self._parallel_abort_sec) else {})
            if abort_vehicle_id in stationary_parallel_group:
                self.get_logger().info(
                    f"[StationaryParallelContinue] group={tuple(stationary_parallel_group)}, "
                    f"lane=L{self._target_lane_idx}; retaining pass instead of stationary yield",
                    throttle_duration_sec=1.0)
                self._parallel_start_time = current_time_sec
                parallel_duration = 0.0
            if (
                parallel_abort_candidate is not None
                and parallel_duration > self._parallel_abort_sec
                and self._target_lane_idx in (0, 2)
                and not parallel_abort_suppressed
                and not self._parallel_abort_active
            ):
                abort = parallel_abort_candidate
                abort_target_lane_idx = select_parallel_abort_lane(
                    self._target_lane_idx, abort["lane_idx"])
                self.get_logger().warn(
                    "[ParallelAbort] entering exclusive yield: "
                    f"vehicle_id={abort['vehicle_id']} "
                    f"opponent_lane=L{abort['lane_idx']} "
                    f"target_lane=L{abort_target_lane_idx} "
                    f"lat={abort['lateral_center']:.2f}m, "
                    f"clearance={abort['clearance']:+.2f}m, "
                    f"lon_clearance="
                    f"{abort['longitudinal_clearance']:+.2f}m, "
                    f"lon={abort['longitudinal']:+.2f}m, "
                    f"arc={abort['arc']:+.2f}m, "
                    f"after {parallel_duration:.1f}s parallel running. "
                    "Stopping before applying the selected lane on the "
                    "next cycle.",
                    throttle_duration_sec=1.0
                )
                self._parallel_abort_previous_lane = self._target_lane_idx
                self._parallel_abort_active = True
                self._parallel_abort_vehicle_id = abort["vehicle_id"]
                self._parallel_abort_target_lane_idx = abort_target_lane_idx
                self._overtake.release_target()
                self._forced_overtake_vehicle_id = None
                self._outer_lane_released_vehicle_id = None
                self._prepass_fallback_lane_idx = None
                self._prepass_fallback_blocked = False
                self._prepass_fallback_recovery_active = False
                self._prepass_fallback_commit_pending = False
                self._prepass_fallback_commit_lane_idx = None
                self._prepass_retry_after_reverse = False
                ref_vel_kmph = 0.0
                self._parallel_start_time = None
                self._parallel_timer_vehicle_id = None

            slow_lead_active_speed_margin = (
                self._slow_lead_overtake_committed_speed_margin
                if self._reference_path.target_lane_idx in (0, 2)
                and self._reference_path.is_overtaking
                else rolling_precommit_speed_margin(
                    base_margin=self._slow_lead_overtake_speed_margin,
                    far_bonus=self._slow_lead_overtake_far_speed_bonus,
                    target_distance=opponent_distance,
                    commit_distance=self._slow_lead_overtake_speed_margin_fade_distance,
                    prepare_distance=self._slow_lead_overtake_prepare_distance)
            )
            if (
                slow_lead_speed_control_active
                and slow_lead_speed_match_required
            ):
                # Keep momentum at the prepare gate and taper the extra
                # closing margin toward the configured near-distance gate.
                ref_vel_kmph = min(
                    ref_vel_kmph,
                    opponent_v_lead + slow_lead_active_speed_margin,
                )
                self.get_logger().info(
                    "[SlowLeadOvertakePrepare] matching speed before outer "
                    "lane commit: vehicle_id="
                    f"{opponent_vehicle_id}, distance={opponent_distance:.2f}m, "
                    f"lead_speed={opponent_v_lead:.2f}m/s, "
                    f"limit={ref_vel_kmph:.2f}m/s",
                    throttle_duration_sec=1.0,
                )

            if forced_overtake_active:
                ref_vel_kmph = min(
                    ref_vel_kmph, self._forced_overtake_speed)
                if (
                    tracked_slow_lead
                    and safety_target_speed is not None
                    and slow_lead_speed_match_required
                ):
                    # Preserve the gap while the outer-lane constraint is
                    # transitioning/settling. Once the predicted pass is clear,
                    # the ordinary forced-overtake speed cap takes over.
                    ref_vel_kmph = min(
                        ref_vel_kmph,
                        safety_target_speed
                        + slow_lead_active_speed_margin,
                    )
                    self.get_logger().info(
                        "[SlowLeadOvertakeSpeedMatch] limiting approach speed "
                        "until the outer-lane prediction is clear: vehicle_id="
                        f"{safety_target_id}, lead_speed="
                        f"{safety_target_speed:.2f}m/s, limit="
                        f"{ref_vel_kmph:.2f}m/s",
                        throttle_duration_sec=1.0,
                    )

            if close_overtake_blocked or fallback_stop_requested:
                ref_vel_kmph = 0.0
                self.get_logger().warn(
                    "[OvertakeBlocked] "
                    f"vehicle_id={self._overtake.target_id} "
                    f"distance={opponent_distance:.2f}m, "
                    f"mpc_infeasible={self._mpc.infeasibility_counter}; "
                    "holding speed at zero and waiting for reverse recovery.",
                    throttle_duration_sec=1.0,
                )

        if (
            initial_start_boost_active
            and not follow_control_active
            and not emergency_brake_active
            and not self._close_obstacle_reverse_requested
        ):
            ref_vel_kmph = self._mpc_cfg.v_max
        if self._mpc_safety_recovery_active:
            # Apply after every longitudinal override so recovery cannot be
            # accelerated by start boost or follow restart.
            ref_vel_kmph = min(
                ref_vel_kmph,
                float(getattr(
                    self._cfg.mpc, "safety_recovery_speed", 1.0)))
        if (
            self._follow_escape_active
            and self._follow_escape_probe_lane_idx is not None
            and not self._mpc_safety_recovery_active
            and self._stuck_recovery_until is None
            and not emergency_brake_active
            and self._follow_escape_lane_traffic_is_clear(
                pose, v, self._follow_escape_probe_lane_idx)
        ):
            # Feed a low-speed reference into the next constrained MPC solve.
            # The actual command remains zero until three safe solves commit.
            ref_vel_kmph = max(
                ref_vel_kmph, self._follow_escape_creep_speed)
        if (
            slow_lead_speed_control_active
            and slow_lead_speed_match_required
        ):
            # Re-apply after start/restart/recovery overrides. Other safety
            # limiters may still command a lower speed or a complete stop.
            ref_vel_kmph = min(
                ref_vel_kmph,
                opponent_v_lead + slow_lead_active_speed_margin,
            )
        self._mpc.update_v_max(ref_vel_kmph)
        v_ref: List[float] = [ref_vel_kmph] * len(self._reference_path.waypoints)
        self._reference_path.set_v_ref(v_ref)

        if (
            hybrid_escape_speed > 0.0
            and hybrid_escape_target == self._overtake.hybrid.vehicle_id
            and not self._overtake.hybrid.paused
            and not self._overtake.hybrid.completed
            and not self._mpc_safety_recovery_active
            and not self._prepass_fallback_recovery_active
            and not self._parallel_abort_active
            and not self._close_obstacle_reverse_requested
            and self._stuck_recovery_until is None
            and not self._follow_escape_active
            and not intentional_follow_stop_active
            and not close_overtake_blocked and not fallback_stop_requested
            and (pure_pursuit_safe_this_cycle or (
                self._mpc.last_solution_accurate
                and not self._mpc.used_prediction_fallback
                and not self._mpc.recovery_requested))
        ):
            # The previous zero-speed MPC can still command zero. All other
            # reference limits have now run; only request bounded forward creep.
            creep_command = max(0.0, min(hybrid_escape_speed, ref_vel_kmph))
            u[0] = max(u[0], creep_command)
            if creep_command > 0.0:
                self.get_logger().info(
                    f"[EmergencyBrakeHybridEscapeCreep] vehicle_id={hybrid_escape_target}, "
                    f"permitted_ids={tuple(vid for vid, speed in hybrid_escape_speeds.items() if speed > 0.0)}, "
                    f"lane=L{self._overtake.hybrid.lane_idx}, speed={creep_command:.2f}m/s",
                    throttle_duration_sec=0.5)

        if follow_restart_active:
            # Respect any stricter safety limiter applied after the ACC block
            # (parallel traffic, stopped-obstacle hold, etc.).
            follow_restart_target = min(
                follow_restart_target, ref_vel_kmph)
            follow_restart_active = (
                follow_restart_target > abs(v) + 0.1)
            if follow_restart_active:
                u[0] = max(u[0], follow_restart_target)
        if (
            initial_start_boost_active
            and not follow_control_active
            and not emergency_brake_active
            and not self._close_obstacle_reverse_requested
        ):
            u[0] = max(u[0], ref_vel_kmph)
        if (
            slow_lead_speed_control_active
            and slow_lead_speed_match_required
        ):
            # The MPC/fallback command was calculated before this cycle's
            # low-speed classification, so enforce the new cap immediately.
            u[0] = min(u[0], ref_vel_kmph)
        if emergency_brake_active or self._close_obstacle_reverse_requested:
            u[0] = min(u[0], ref_vel_kmph)
        if intentional_follow_stop_active:
            # The MPC solve above still contains the previous cycle's speed
            # limit, so apply the newly detected stop immediately as well.
            u[0] = 0.0
        if self._mpc_safety_recovery_active:
            # The trigger cycle remains stopped. Valid full-width solves may
            # creep at the configured recovery speed on later cycles.
            if self._mpc.recovery_requested:
                u[0] = 0.0
            else:
                u[0] = min(u[0], ref_vel_kmph)
            # A dynamics/rate-infeasible MPC can remain trapped indefinitely
            # even though the independently predicted Pure Pursuit path is
            # collision-free. Use that already validated steering at a low
            # creep speed to restore a solvable pose before choosing reverse.
            if (
                pure_pursuit_safe_this_cycle
                and not emergency_brake_active
                and not self._close_obstacle_reverse_requested
                and not intentional_follow_stop_active
                and not close_overtake_blocked
                and not fallback_stop_requested
            ):
                recovery_creep_speed = (
                    self._mpc_safety_recovery_pp_fast_creep_speed
                    if pp_safe_fast_recovery_active
                    else self._mpc_safety_recovery_pp_creep_speed
                )
                u[0] = min(
                    recovery_creep_speed,
                    self._steering_fallback_speed,
                )
                self.get_logger().info(
                    "[MPCSafetyRecoveryForwardCreep] MPC is unavailable but "
                    "the predicted Pure Pursuit path is safe; moving forward "
                    f"at {float(u[0]):.2f}m/s to recover a solvable pose; "
                    f"fast_zone={pp_safe_fast_recovery_active}.",
                    throttle_duration_sec=1.0,
                )
        if self._follow_escape_active:
            if (
                self._follow_escape_forward_active
                and not emergency_brake_active
                and self._prediction_is_clear_of_vehicle(
                    self._follow_escape_target_id)
                and self._follow_escape_lane_traffic_is_clear(
                    pose, v, self._follow_escape_probe_lane_idx)
            ):
                u[0] = max(
                    u[0],
                    min(self._follow_escape_creep_speed, ref_vel_kmph),
                )
            else:
                # Probe and rear-blocked states are observation-only.
                u[0] = 0.0

        if self._collision_evidence_hold:
            # Do not let restart, Hybrid creep or another vehicle override missing evidence.
            u[0] = 0.0

        # 停止命令がコマンドで入力させたら減速させる
        if not self._enable_control:
            last_v_cmd = self._last_u[0]
            if last_v_cmd < 0.5:
                u[0] = 0.0
            else:
                decel_v = last_v_cmd + self._mpc_cfg.a_min * dt
                u[0] = np.clip(decel_v, 0.0, self._mpc_cfg.v_max)
       
        if len(u) == 0:
            self.get_logger().error("No control signal", throttle_duration_sec=1)
            u = [0.0, 0.0]

        self._intentional_follow_stop_active = (
            intentional_follow_stop_active
            and not self._follow_escape_active
        )
        recovering_from_stuck = self._apply_stuck_recovery(now, u, v, pose)

        committed_overtake_acceleration_active = bool(
            self._committed_overtake_acceleration_boost_enabled
            and self._overtake.committed
            and self._overtake.verification.vehicle_id
                == self._overtake.target_id
            and self._overtake.verification.lane_idx in (0, 2)
            and self._reference_path.is_overtaking
            and self._reference_path.target_lane_idx
                == self._overtake.verification.lane_idx
            and not slow_lead_speed_match_required
            and not emergency_brake_active
            and not self._close_obstacle_reverse_requested
            and not intentional_follow_stop_active
            and not self._parallel_abort_active
            and not self._mpc_safety_recovery_active
            and not self._follow_escape_active
            and not recovering_from_stuck
            and self._enable_control
            and self._mpc.infeasibility_counter == 0
            and not self._mpc.used_prediction_fallback
            and not self._mpc.recovery_requested
            and ref_vel_kmph - abs(v)
                >= self._committed_overtake_acceleration_min_speed_error
        )
        if committed_overtake_acceleration_active:
            # The solve at the commit edge can still contain the preceding
            # cycle's lead-speed cap. The same-target/same-lane strict Shadow
            # proof is retained across Soft Transition, so immediately expose
            # the already safety-limited reference speed to longitudinal
            # control. Emergency/Parallel/MPC recovery above always vetoes it.
            u[0] = max(float(u[0]), float(ref_vel_kmph))
            self.get_logger().info(
                "[CommittedOvertakeAcceleration] strict Shadow lane is "
                "committed and no safety limiter is active; applying maximum "
                "acceleration without the ordinary LPF delay: "
                f"vehicle_id={self._overtake.target_id}, "
                f"lane=L{self._overtake.verification.lane_idx}, "
                f"speed={abs(v):.2f}m/s, target={ref_vel_kmph:.2f}m/s",
                throttle_duration_sec=0.5,
            )

        acc = 0.
        bug_acc_enabled = False


        #boostモードがONのとき
        if recovering_from_stuck:
            bug_acc_enabled = False
            straight_reentry_motion_active = bool(
                self._straight_reentry_active
                and not self._straight_reentry_returning_drive
                and float(u[0]) > 0.0
            )
            if straight_reentry_motion_active:
                # StraightReentry uses a positive speed command in both DRIVE
                # and REVERSE (the selected gear determines the direction).
                # Do not pair that command with the shift-hold brake below;
                # doing so leaves AWSIM at zero speed until every reentry
                # attempt times out.
                acc = np.clip(
                    self.KP * (float(u[0]) - abs(v)),
                    0.0,
                    self._mpc_cfg.a_max,
                )
            elif not self._stuck_reverse_drive_active:
                acc = -8.0
            elif self._stuck_reverse_command_mode in ("negative_speed_positive_accel", "awsim_reverse_button"):
                if self._stuck_reverse_acceleration_positive:
                    acc = abs(self._stuck_reverse_acceleration)
                else:
                    acc = -abs(self._stuck_reverse_acceleration)
            else:
                acc = self._stuck_reverse_acceleration
            self.get_logger().info(
                f"[StuckRecovery] cmd speed={u[0]:.2f} acc={acc:.2f} "
                f"phase={'straight_reentry' if straight_reentry_motion_active else 'reverse_or_shift'} "
                f"actuation=({self._stuck_actuation_accel_cmd:.2f},"
                f"{self._stuck_actuation_brake_cmd:.2f}) "
                f"gear={getattr(self._gear_report, 'report', None)} "
                f"mode={getattr(self._control_mode_report, 'mode', None)} "
                f"vel={getattr(self._velocity_report, 'longitudinal_velocity', None)} "
                f"state={self._awsim_state}",
                throttle_duration_sec=1.0,
            )
            self._pred_marker_color = YELLOW
        elif (
            emergency_brake_active
            or self._close_obstacle_reverse_requested
            or intentional_follow_stop_active
        ):
            bug_acc_enabled = False
            acc = np.clip(
                self.KP * (u[0] - abs(v)),
                self._mpc_cfg.a_min,
                self._mpc_cfg.a_max,
            )
            self._pred_marker_color = RED
        elif follow_restart_active:
            bug_acc_enabled = False
            acc = max(
                self._follow_restart_acceleration,
                self.KP * (u[0] - abs(v)),
            )
            acc = np.clip(acc, 0.0, self._mpc_cfg.a_max)
            self._pred_marker_color = CYAN
        elif initial_start_boost_active and not follow_control_active:
            bug_acc_enabled = False
            acc = self._mpc_cfg.a_max
            self._pred_marker_color = CYAN
        elif committed_overtake_acceleration_active:
            bug_acc_enabled = False
            acc = self._mpc_cfg.a_max
            self._pred_marker_color = CYAN
        elif self.USE_BUG_ACC:
            def deg2rad(deg):
                return deg * np.pi / 180.0

            forced_speed_error = u[0] - abs(v)
            if (
                forced_overtake_active
                and forced_speed_error <= 0.3
            ):
                bug_acc_enabled = False
                acc = np.clip(
                    self.KP * forced_speed_error,
                    self._mpc_cfg.a_min,
                    self._mpc_cfg.a_max,
                )
                self._pred_marker_color = YELLOW
            elif abs(v) > kmh_to_m_per_sec(44.0) or \
             (abs(v) > kmh_to_m_per_sec(38.0) and abs(max_delta) > deg2rad(12.0)):
                bug_acc_enabled = False
                acc = self._mpc_cfg.a_min / 3.0 * 2.0
                self._pred_marker_color = RED
            elif abs(v) > kmh_to_m_per_sec(41.0) or abs(u[1]) > deg2rad(10.0):
                bug_acc_enabled = False
                acc = self._mpc_cfg.a_max
                self._pred_marker_color = YELLOW
            else:
                bug_acc_enabled = True
                acc = 500.0
                self._pred_marker_color = CYAN
        else:
            acc =  self.KP * (u[0] - v)
            acc = np.clip(acc, self._mpc_cfg.a_min, self._mpc_cfg.a_max)

        # 加速度と操舵角の平滑化
        if not recovering_from_stuck:
            # Safety braking must not be blended with a previous boost value.
            if not (
                emergency_brake_active
                or self._close_obstacle_reverse_requested
                or follow_restart_active
                or initial_start_boost_active
                or committed_overtake_acceleration_active
            ):
                acc = self._last_acc + (acc - self._last_acc) * self._mpc_cfg.accel_low_pass_gain
            u[1] = self._last_u[1] + (u[1] - self._last_u[1]) * self._mpc_cfg.steer_low_pass_gain

        self._last_acc = acc
        self._last_u[0] = u[0]
        self._last_u[1] = u[1]

        # update car state (use v for feedback actual speed)
        self._car.drive([v, u[1]])

        # Publish control command.  Keep this active during stuck recovery because
        # the known-working teleop path drives AWSIM through control_cmd directly.
        self._publish_control_command(now, u, acc, bug_acc_enabled)

        # Log states
        self._sim_logger.log(self._car, u, t)
        self._sim_logger.plot_animation(t, self._loop, self._current_laps, self._lap_times, is_colliding, u, self._mpc, self._car)

        # 約 0.25 秒ごとに予測結果を表示
        if self._loop % (self._mpc_cfg.control_rate // 4) == 0:
            if self._mpc.current_prediction is not None:
                self._publish_mpc_pred_marker(
                    self._mpc.current_prediction[0],
                    self._mpc.current_prediction[1],
                )
            else:
                self._clear_mpc_pred_markers()

        # 約 1 秒ごとに車線境界を表示
        if self._loop % int(self._mpc_cfg.control_rate) == 0:
            self._publish_lane_markers(self._reference_path)

    def run(self) -> None:
        self._wait_until_clock_received()
        self._wait_until_odom_received()
        self._wait_until_gnss_received()
        self._wait_until_trajectory_received()
        self._wait_until_path_constraints_received()

        # initialize car states
        pose = self.get_ego_pose()
        self._car.update_states(pose.x, pose.y, pose.theta)
        self._car.update_reference_path(self._car.reference_path)

        if self._ref_vel_configulator is None:
            self._publish_ref_path_marker(self._car.reference_path)

        self._pred_marker_color = CYAN

        # initialize control states
        self._control_rate = self.create_rate(self._mpc_cfg.control_rate)
        self._sim_logger = SimulationLogger(
            self.get_logger(),
            self._car.temporal_state.x, self._car.temporal_state.y, self._cfg.sim_logger.animation_enabled, self.SHOW_PLOT_ANIMATION, self.PLOT_RESULTS, self.ANIMATION_INTERVAL) # type: ignore

        self._loop = 0
        self._last_acc = 0.0
        self._last_u = np.array([0.0, 0.0])
        self._t_start = self.get_clock().now()
        self._last_t = self._t_start

        self.get_logger().info("----------------------")
        self.get_logger().info("START!")
        self.get_logger().info("----------------------")

        while rclpy.ok() and (not self._sim_logger.stop_requested()):
            self._control()

    def stop(self):
        # Wait for stopping
        self.get_logger().warn("----------------------")
        self.get_logger().warn("Stopping...")
        self.get_logger().warn("----------------------")
        timeout_time = self.get_clock().now() + rclpy.time.Duration(seconds=5)
        while self._odom.twist.twist.linear.x > 0.1 and self.get_clock().now() < timeout_time:
            self._enable_control = False
            self._control()

        # Publish zero command to stop the car completely
        zero_cmd = self._create_ackerman_control_command(self.get_clock().now(), [0.0, 0.0], 0.0, False)
        self._command_pub.publish(zero_cmd)

        self.get_logger().warn(">> Stop Completed!")

        # show results
        self._sim_logger.show_results(self._current_laps, self._lap_times, self._car)

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path
