#!/usr/bin/env python3

import yaml
import math
from typing import List, Tuple, Optional, NamedTuple
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import numpy as np
import copy
import os
import shutil
from datetime import datetime

# ROS 2
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty, Bool, Float32MultiArray, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose2D, Point, Vector3, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA

from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_planning_msgs.msg import Trajectory
from v2x_msgs.msg import V2XVehiclePositionArray
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    V2XVehicleTracker,
    predictions_to_obstacles,
)

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.utils import load_waypoints, kmh_to_m_per_sec, load_ref_path

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists
from multi_purpose_mpc_ros.simulation_logger import SimulationLogger
from multi_purpose_mpc_ros.obstacle_manager import ObstacleManager
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
        self._odom: Optional[Odometry] = None
        self._gnss_pose: Optional[PoseWithCovarianceStamped] = None
        self._enable_control = True
        self._initialize()
        self._setup_parameters_callback()
        self._setup_pub_sub()

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

                elif param.name == "accel_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.accel_low_pass_gain = param.value
                    self.get_logger().warn(f"accel_low_pass_gain was updated to '{param.value}'")

                elif param.name == "steer_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.steer_low_pass_gain = param.value
                    self.get_logger().warn(f"steer_low_pass_gain was updated to '{param.value}'")

                elif param.name == "wp_id_offset" and param.type_ == Parameter.Type.INTEGER:
                    mpc_cfg.wp_id_offset = param.value
                    self._mpc.update_wp_id_offset(param.value)
                    self.get_logger().warn(f"wp_id_offset was updated to '{param.value}'")


            return SetParametersResult(successful=True)

        declatre_parameters()
        self.add_on_set_parameters_callback(param_cb)

    def _initialize(self) -> None:
        self._map_z = 0.02

        def create_map() -> Map:
            return Map(self.in_pkg_share(self._cfg.map.yaml_path)) # type: ignore

        def create_ref_path(map: Map) -> ReferencePath:
            cfg_ref_path = self._cfg.reference_path # type: ignore
            is_ref_path_given = cfg_ref_path.csv_path != "" # type: ignore
            if is_ref_path_given:
                print("Using given reference path")
                wp_x, wp_y, _, _ = load_ref_path(self.in_pkg_share(self._cfg.reference_path.csv_path)) # type: ignore
                ref_path = ReferencePath(
                    map, wp_x, wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)
                
                # 必要に応じて境界ファイルのロード処理をここに追記
                return ref_path
            else:
                print("Using waypoints to create reference path")
                wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path)) # type: ignore
                return ReferencePath(
                    map, wp_x, wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)

        def create_obstacles() -> List[Obstacle]:
            use_csv_obstacles = self._cfg.obstacles.csv_path != "" # type: ignore
            if use_csv_obstacles:
                obstacles_file_path = self.in_pkg_share(self._cfg.obstacles.csv_path) # type: ignore
                obs_x, obs_y = load_waypoints(obstacles_file_path)
                obstacles = []
                for cx, cy in zip(obs_x, obs_y):
                    obstacles.append(Obstacle(cx=cx, cy=cy, radius=self._cfg.obstacles.radius)) # type: ignore
                self._obstacle_manager = ObstacleManager(self._map, obstacles)
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

        def create_mpc(car: BicycleModel, N_horizon: int, R_matrix=None) -> Tuple[MPCConfig, MPC]:
            cfg_mpc = self._cfg.mpc # type: ignore
            mpc_R = R_matrix if R_matrix is not None else cfg_mpc.R

            mpc_cfg = MPCConfig(
                N_horizon,
                sparse.diags(cfg_mpc.Q),
                sparse.diags(mpc_R),
                sparse.diags(cfg_mpc.QN),
                kmh_to_m_per_sec(self.BUG_VEL if self.USE_BUG_ACC else cfg_mpc.v_max),
                cfg_mpc.a_min,
                cfg_mpc.a_max,
                cfg_mpc.ay_max,
                np.deg2rad(cfg_mpc.delta_max_deg),
                cfg_mpc.steer_rate_max,
                cfg_mpc.control_rate,
                cfg_mpc.steering_tire_angle_gain_var,
                cfg_mpc.accel_low_pass_gain,
                cfg_mpc.steer_low_pass_gain,
                cfg_mpc.wp_id_offset,
                cfg_mpc.use_max_kappa_pred)

            state_constraints = {"xmin": np.array([-np.inf, -np.inf, -np.inf]), "xmax": np.array([np.inf, np.inf, np.inf])}
            input_constraints = {
                "umin": np.array([0.0, -np.tan(mpc_cfg.delta_max) / car.length]),
                "umax": np.array([mpc_cfg.v_max, np.tan(mpc_cfg.delta_max) / car.length])}

            scaled_steer_rate_max = mpc_cfg.steer_rate_max / mpc_cfg.steering_tire_angle_gain_var

            mpc = MPC(
                car, N_horizon, mpc_cfg.Q, mpc_cfg.R, mpc_cfg.QN,
                state_constraints, input_constraints, mpc_cfg.ay_max,
                scaled_steer_rate_max, mpc_cfg.wp_id_offset,
                self.USE_OBSTACLE_AVOIDANCE,
                self._cfg.reference_path.use_path_constraints_topic,
                mpc_cfg.use_max_kappa_pred)

            return mpc_cfg, mpc

        def compute_speed_profile(car: BicycleModel, mpc_config: MPCConfig) -> None:
            speed_profile_constraints = {
                "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
                "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}
            car.reference_path.compute_speed_profile(speed_profile_constraints)

        self._map = create_map()
        
        # 2系統の参照経路と車両モデルを初期化
        self._reference_pathN = create_ref_path(self._map)
        self._reference_path10 = create_ref_path(self._map)
        self._carN = create_car(self._reference_pathN)
        self._car10 = create_car(self._reference_path10)

        # 標準(N)と特定区間(9ステップ/R10重み)のMPCを構築
        self._mpc_cfg, self._mpcN = create_mpc(self._carN, self._cfg.mpc.N)
        _, self._mpc10 = create_mpc(self._car10, 9, getattr(self._cfg.mpc, 'R10', self._cfg.mpc.R))
        self._mpc10.update_wp_id_offset(1)

        # 動的切り替え用のポインタ初期化
        self._car = self._carN
        self._reference_path = self._reference_pathN
        self._mpc = self._mpcN

        compute_speed_profile(self._carN, self._mpc_cfg)
        compute_speed_profile(self._car10, self._mpc_cfg)

        def create_ref_vel_configulator() -> Optional[ReferenceVelocityConfigulator]:
            if self._ref_vel_config_path is None:
                return None
            return ReferenceVelocityConfigulator(self, self._config_path, self._ref_vel_config_path)

        self._ref_vel_configulator: Optional[ReferenceVelocityConfigulator] = create_ref_vel_configulator()
        self._trajectory: Optional[Trajectory] = None
        self._path_constraints = None

        if self.USE_OBSTACLE_AVOIDANCE:
            self._static_obstacles: List[Obstacle] = create_obstacles()
            self._dynamic_obstacles: List[Obstacle] = []
            self._obstacles_updated = bool(self._static_obstacles)
            v2x_cfg = self._cfg.v2x_obstacle_avoidance  # type: ignore
            self._v2x_tracker = V2XVehicleTracker(
                v_max_safety=float(v2x_cfg.v_max_safety),
                position_jump_threshold=float(v2x_cfg.position_jump_threshold),
                warn_callback=self.get_logger().warn,
            )
            self._v2x_vehicle_radius = float(v2x_cfg.vehicle_radius)
            mpc_N = int(self._cfg.mpc.N)  # type: ignore
            t_horizon = mpc_N / float(self._cfg.mpc.control_rate)  # type: ignore
            self._v2x_t_samples = [k * t_horizon / max(mpc_N - 1, 1) for k in range(mpc_N)]
            ref_max_width = float(self._cfg.reference_path.max_width)  # type: ignore
            self._v2x_corridor_threshold_sq = (ref_max_width / 2.0 + self._v2x_vehicle_radius + 0.5) ** 2
            wps = self._reference_path.waypoints
            self._waypoint_xy = np.asarray([(wp.x, wp.y) for wp in wps], dtype=np.float64)

        self._current_laps = 1
        self._last_lap_time = 0.0
        self._lap_times = [None] * (self.MAX_LAPS + 1)
        self._loop = 0
        self._last_condition = None
        self._last_colliding_time = None
        self._last_v2x_time = None
        self._stats = ExecutionStats(self.get_logger(), window_size=50, record_count_threshold=1000)

        if self._cfg.common.save_config:
            self._save_config()


        # Obstacles
        if self.USE_OBSTACLE_AVOIDANCE:
            self._static_obstacles: List[Obstacle] = create_obstacles()
            self._dynamic_obstacles: List[Obstacle] = []
            self._obstacles_updated = bool(self._static_obstacles)
            v2x_cfg = self._cfg.v2x_obstacle_avoidance  # type: ignore
            self._v2x_tracker = V2XVehicleTracker(
                v_max_safety=float(v2x_cfg.v_max_safety),
                position_jump_threshold=float(v2x_cfg.position_jump_threshold),
                warn_callback=self.get_logger().warn,
            )
            self._v2x_vehicle_radius = float(v2x_cfg.vehicle_radius)
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

        # loop counter (initialized early to avoid race conditions with V2X callbacks)
        self._loop = 0

        # condition
        self._last_condition = None
        self._last_colliding_time = None

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

        # ★ここを追加：AIの追い越し判断を可視化するための新規パブリッシャー
        self._overtake_vis_pub = self.create_publisher(MarkerArray, "/mpc/overtake_vis", 1)

        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # NOTE:評価環境での可視化のためにダミーのトピック名を使用
        self._ref_path_pub = self.create_publisher(
            MarkerArray, "/mpc/ref_path", latching_qos)
        self._ref_path_pub_dummy = self.create_publisher(
            MarkerArray, "/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner/debug/bound", latching_qos)

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
            self._condition_sub = self.create_subscription(
                Int32, "/aichallenge/pitstop/condition", self._condition_callback, 1)

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
        if not self.USE_BUG_ACC:
            self._command_raw_pub.publish(cmd)

        # compensate steering angle for the real vehicle
        # AWSIMにおいても後段のactuation_cmd_converter でgainを考慮した指令を生成するため、実機/sim問わず
        # gain を掛ける
        if self.USE_BUG_ACC:
            cmd.command.lateral.steering_tire_angle *= self._mpc_cfg.steering_tire_angle_gain_var
        else:
            cmd.lateral.steering_tire_angle *= self._mpc_cfg.steering_tire_angle_gain_var
        self._command_pub.publish(cmd)


    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

    def _gnss_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._gnss_pose = msg

    @property
    def _is_currently_overtaking(self) -> bool:
        if not hasattr(self, '_target_lane_idx'):
            return False
        return self._target_lane_idx in [0, 2]

    def _control_mode_request_callback(self, msg):
        if msg.data and not self._enable_control:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _path_constraints_callback(self, msg: PathConstraints):
        self._reference_path.set_path_constraints(
            msg.upper_bounds, msg.lower_bounds, msg.rows, msg.cols)

    def _get_opponent_position_and_id(self) -> Optional[Tuple[Tuple[float, float], str]]:
        if not self.USE_OBSTACLE_AVOIDANCE:
            return None
        import os
        domain_id = os.environ.get("ROS_DOMAIN_ID", "1")
        host_id = f"d{domain_id}"
        for vid in self._v2x_tracker.active_vehicle_ids():
            if vid != host_id:
                buf = self._v2x_tracker._samples.get(vid)
                if buf:
                    _, x, y = buf[-1]
                    return (x, y), vid
        return None

    def _v2x_callback(self, msg: V2XVehiclePositionArray) -> None:
        self._last_v2x_time = self.get_clock().now()
        now_sec = self._last_v2x_time.nanoseconds / 1e9

        valid_vehicles = []
        for v in msg.vehicles:
            t = float(v.header.stamp.sec) + float(v.header.stamp.nanosec) * 1e-9
            # タイムスタンプが現在時刻から1.0秒以内のものだけを有効データとする（シミュレーションのリセット対策およびゴースト対策）
            if abs(now_sec - t) < 1.0:
                valid_vehicles.append(v)
            else:
                self.get_logger().warn(
                    f"Ignoring stale V2X message for vehicle {v.vehicle_id} (age: {now_sec - t:.2f}s)",
                    throttle_duration_sec=2
                )

        class FilteredMsg:
            def __init__(self, vehicles):
                self.vehicles = vehicles

        self._v2x_tracker.update(FilteredMsg(valid_vehicles))

        # スタート時のスタックを防ぐため、ごく初期（30ループ未満または低速時）のみ他車を無視する
        v_host = self._odom.twist.twist.linear.x if self._odom is not None else 0.0
        if self._loop < 30 or v_host < 1.5:
            self._dynamic_obstacles = []
        else:
            predictions = self._v2x_tracker.predict_all(self._v2x_t_samples)
            # Filter out the host vehicle's own predictions
            import os
            domain_id = os.environ.get("ROS_DOMAIN_ID", "1")
            host_id = f"d{domain_id}"
            predictions = {vid: pts for vid, pts in predictions.items() if vid != host_id}

            self._dynamic_obstacles = predictions_to_obstacles(
                predictions, self._v2x_vehicle_radius)
        self._obstacles_updated = True

    def _filter_obstacles_to_corridor(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        if not obstacles or self._waypoint_xy.size == 0:
            return obstacles
        thr_sq = self._v2x_corridor_threshold_sq
        wps = self._waypoint_xy
        kept: List[Obstacle] = []
        for ob in obstacles:
            dxy = wps - np.array([ob.cx, ob.cy], dtype=np.float64)
            if np.min(np.einsum('ij,ij->i', dxy, dxy)) <= thr_sq:
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
        # section = int(msg.data[3])

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
            self._last_colliding_time = self.get_clock().now()
            self.get_logger().warning(f"Collision detected!")
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
        self._wait_until_message_received(lambda: self._gnss_pose, 'gnss', timeout)

    def _get_current_pose(self) -> Pose2D:
        pose = odom_to_pose_2d(self._odom) # type: ignore
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

    # ==========================================
    # 追い越し可視化用マーカーパブリッシュ関数 (テキストなし版)
    # ==========================================
    def _publish_overtake_visualization(self, host_pose, opp_x, opp_y, pass_px, pass_py, decision: str):
        now_msg = self.get_clock().now().to_msg()
        markers = MarkerArray()
        
        host_wp = self._car.get_closest_waypoint(host_pose.x, host_pose.y)
        opp_wp = self._car.get_closest_waypoint(opp_x, opp_y)
        N_total = self._reference_path.n_waypoints
        
        # 1. 相手とのつながり (透明な太い色付き帯)
        m_conn = Marker()
        m_conn.header.frame_id = "map"
        m_conn.header.stamp = now_msg
        m_conn.ns = "overtake_connection"
        m_conn.id = 0
        m_conn.type = Marker.LINE_STRIP
        m_conn.action = Marker.ADD
        m_conn.pose.orientation.w = 1.0
        m_conn.scale.x = 2.0  # 2m幅の太い帯
        
        if "FOLLOW" in decision:
            m_conn.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3) # 抜けない時は赤
        else:
            m_conn.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3) # 抜ける時は緑
            
        # 自車から相手車両のウェイポイントまでをコースに沿って繋ぐ
        wp_diff = (opp_wp - host_wp) % N_total
        m_conn.points.append(Point(x=host_pose.x, y=host_pose.y, z=self._map_z))
        for i in range(1, wp_diff):
            curr_idx = (host_wp + i) % N_total
            wp_pt = self._reference_path.get_waypoint(curr_idx)
            m_conn.points.append(Point(x=wp_pt.x, y=wp_pt.y, z=self._map_z))
        m_conn.points.append(Point(x=opp_x, y=opp_y, z=self._map_z))
        markers.markers.append(m_conn)

        # 2. 追い越し目標ライン (シアンの線)
        m_path = Marker()
        m_path.header.frame_id = "map"
        m_path.header.stamp = now_msg
        m_path.ns = "overtake_target"
        m_path.id = 1
        m_path.type = Marker.LINE_STRIP
        m_path.action = Marker.ADD
        m_path.pose.orientation.w = 1.0
        m_path.scale.x = 0.4
        m_path.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8) # シアン
        
        m_path.points.append(Point(x=host_pose.x, y=host_pose.y, z=self._map_z))
        m_path.points.append(Point(x=pass_px, y=pass_py, z=self._map_z))
        markers.markers.append(m_path)

        self._overtake_vis_pub.publish(markers)

    def _clear_overtake_visualization(self):
        markers = MarkerArray()
        for i in range(2):
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "overtake_connection" if i==0 else "overtake_target"
            m.id = i
            m.action = Marker.DELETE
            markers.markers.append(m)
        self._overtake_vis_pub.publish(markers)
  
    def _control(self):
        now = self.get_clock().now()
        # V2Xタイムアウト処理：0.5秒以上V2Xデータを受信しなかった場合、動的障害物をクリアする
        if self.USE_OBSTACLE_AVOIDANCE:
            if self._last_v2x_time is not None:
                elapsed_v2x = (now - self._last_v2x_time).nanoseconds / 1e9
                if elapsed_v2x > 0.5:
                    self._dynamic_obstacles = []
                    self._obstacles_updated = True
                    self.get_logger().warn("V2X message timeout (>0.5s) - clearing dynamic obstacles", throttle_duration_sec=2)

        t = (now - self._t_start).nanoseconds / 1e9
        dt = (now - self._last_t).nanoseconds / 1e9

        self._last_t = now
        self._loop += 1

        if self.use_stats:
            self._stats.record()

        self._control_rate.sleep()

        if self._loop % 100 == 0:
            if self._cfg.reference_path.update_by_topic: # type: ignore
                new_referece_path = self._create_reference_path_from_autoware_trajectory(self._trajectory)
                if new_referece_path is not None:
                    self._car.reference_path = new_referece_path
                    self._car.update_reference_path(self._car.reference_path)

        if self.USE_OBSTACLE_AVOIDANCE and self._obstacles_updated:
            self._obstacles_updated = False
            self._map.reset_map()
            filtered_dynamic = self._filter_obstacles_to_corridor(self._dynamic_obstacles)
            self._map.add_obstacles(self._static_obstacles + filtered_dynamic)
            self._reference_path.reset_dynamic_constraints()

        is_colliding = False
        if self._last_colliding_time is not None:
            elapsed_from_last_colliding = (now - self._last_colliding_time).nanoseconds / 1e9
            if elapsed_from_last_colliding < 5.0:
                is_colliding = True

        pose = self._get_current_pose()
        v = self._odom.twist.twist.linear.x

        # 1. 現在のウェイポイントの取得と、コースに合わせたモデルの動的切り替え
        self._car.update_states(pose.x, pose.y, pose.theta)
        self._car.get_current_waypoint()
        wp = self._car.wp_id

        if (215 <= wp <= 245) or (260 <= wp <= 300) or (320 <= wp <= 340):
            self._mpc = self._mpc10
            self._car = self._car10
            self._reference_path = self._reference_path10
        else:
            self._mpc = self._mpcN
            self._car = self._carN
            self._reference_path = self._reference_pathN

        # ポインタ切り替え後に再度状態を同期
        self._car.update_states(pose.x, pose.y, pose.theta)
        self._mpc.previous_steering = self._last_u[1]

        
        # =========================================================================
        # 2. --- Overtaking Space Calculation & Lane Selection Logic ---
        # =========================================================================
        if not hasattr(self, '_target_lane_idx'):
            self._target_lane_idx = None
        if not hasattr(self, '_last_lane_change_time'):
            self._last_lane_change_time = None

        new_target_lane_idx = self._target_lane_idx
        opponent_ahead = None
        opponent_offset = 0.0
        opponent_distance = 99999.0
        opponent_v_lead = 0.0
        min_wp_diff = 99999
        opp_id = None
        N_total = self._reference_path.n_waypoints

        if self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker'):
            opp_pos_info = self._get_opponent_position_and_id()
            if opp_pos_info is not None:
                opp_pos, opp_id = opp_pos_info
                opp_x, opp_y = opp_pos
                opp_wp_id = self._car.get_closest_waypoint(opp_x, opp_y)
                
                wp_diff = (opp_wp_id - wp) % N_total
                
                # 相手が約15m(wp差25)以内の前方にいるかチェック
                if 0 < wp_diff < 25: 
                    if wp_diff < min_wp_diff:
                        min_wp_diff = wp_diff
                        opponent_ahead = opp_wp_id
                        
                        opp_wp_obj = self._reference_path.get_waypoint(opp_wp_id)
                        angle_ub = opp_wp_obj.psi + math.pi / 2.0
                        dx_opp = opp_x - opp_wp_obj.x
                        dy_opp = opp_y - opp_wp_obj.y
                        opponent_offset = dx_opp * math.cos(angle_ub) + dy_opp * math.sin(angle_ub)
                        opponent_distance = math.hypot(opp_x - pose.x, opp_y - pose.y)
                        opp_vel_xy = self._v2x_tracker.velocity(opp_id)
                        opponent_v_lead = math.hypot(opp_vel_xy[0], opp_vel_xy[1])

        # =========================================================================
        # 3. --- Lane selection based on hysteresis ---
        # =========================================================================
        new_target_lane_idx = self._target_lane_idx
        prev_lane_idx = self._target_lane_idx

        if opponent_ahead is not None:
            opp_wp_obj = self._reference_path.get_waypoint(opponent_ahead)
            angle_ub = opp_wp_obj.psi + math.pi / 2.0
            
            # V2X車両半径を用いて相手がコース上で占有している幅を計算
            opp_left_edge = opponent_offset + self._v2x_vehicle_radius
            opp_right_edge = opponent_offset - self._v2x_vehicle_radius
            
            # コース全体の幅(ub=左端, lb=右端)から相手の幅を引き、残りの通過可能スペースを計算
            space_left = opp_wp_obj.ub - opp_left_edge
            space_right = opp_right_edge - opp_wp_obj.lb
            
            # 安全に通過するために必要な幅 (自車幅 + 余裕0.5m)
            required_space = self._car.width + 0.3 
            
            # 優先車線の決定 (ヒステリシスを設けて左右スペースの大きさが拮抗したときのチャタリングを防ぐ)
            if space_left > space_right + 0.3:
                preferred_lane = 2 # 左車線
                preferred_space = space_left
                alt_lane = 0
                alt_space = space_right
            elif space_right > space_left + 0.3:
                preferred_lane = 0 # 右車線
                preferred_space = space_right
                alt_lane = 2
                alt_space = space_left
            else:
                # 左右のスペースに差がない場合、現在の車線を維持
                if prev_lane_idx == 0:
                    preferred_lane = 0
                    preferred_space = space_right
                    alt_lane = 2
                    alt_space = space_left
                elif prev_lane_idx == 2:
                    preferred_lane = 2
                    preferred_space = space_left
                    alt_lane = 0
                    alt_space = space_right
                else:
                    if space_left >= space_right:
                        preferred_lane = 2
                        preferred_space = space_left
                        alt_lane = 0
                        alt_space = space_right
                    else:
                        preferred_lane = 0
                        preferred_space = space_right
                        alt_lane = 2
                        alt_space = space_left

            # 通過可能スペースに基づいて実際の車線インデックスを決定
            if preferred_space > required_space:
                new_target_lane_idx = preferred_lane
                decision_str = "OVERTAKE LEFT" if preferred_lane == 2 else "OVERTAKE RIGHT"
                target_offset = (opp_wp_obj.ub - self._car.width/2 - 0.2) if preferred_lane == 2 else (opp_wp_obj.lb + self._car.width/2 + 0.2)
                pass_px = opp_wp_obj.x + target_offset * math.cos(angle_ub)
                pass_py = opp_wp_obj.y + target_offset * math.sin(angle_ub)
            elif alt_space > required_space:
                new_target_lane_idx = alt_lane
                decision_str = "OVERTAKE LEFT" if alt_lane == 2 else "OVERTAKE RIGHT"
                target_offset = opp_wp_obj.ub - self._car.width/2 if alt_lane == 2 else opp_wp_obj.lb + self._car.width/2
                pass_px = opp_wp_obj.x + target_offset * math.cos(angle_ub)
                pass_py = opp_wp_obj.y + target_offset * math.sin(angle_ub)
            else:
                new_target_lane_idx = 1 # 追従
                decision_str = "FOLLOW (Blocked)"
                pass_px = opp_x
                pass_py = opp_y
            
            # RVizに可視化情報をパブリッシュ
            self._publish_overtake_visualization(pose, opp_x, opp_y, pass_px, pass_py, decision_str)
            
        else:
            new_target_lane_idx = None
            self._clear_overtake_visualization()

        # 最初のスタート時（Lap 1 かつ 25 <= wp < 50）かつ近くに相手がいる場合は、強制的にセンター車線追従
        is_start_grid = (self._current_laps == 1 and 25 <= wp < 50)
        opp_pos_info = self._get_opponent_position_and_id() if (self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker')) else None
        
        if is_start_grid and opp_pos_info is not None:
            opp_pos, opp_id = opp_pos_info
            opp_x, opp_y = opp_pos
            opp_dist = math.hypot(opp_x - pose.x, opp_y - pose.y)
            if opp_dist < 6.0:
                new_target_lane_idx = 1
                decision_str = "START FOLLOW"
                # RVizに可視化情報をパブリッシュ
                self._publish_overtake_visualization(pose, opp_x, opp_y, opp_x, opp_y, decision_str)

        # =========================================================================
        # チャタリング防止 (短時間ロック - 0.3s)
        # =========================================================================
        current_time_sec = float(now.nanoseconds) / 1e9
        if new_target_lane_idx != prev_lane_idx:
            can_change_lane = True
            if self._last_lane_change_time is not None:
                elapsed = current_time_sec - self._last_lane_change_time
                if elapsed < 0.3: # 0.3秒ロックに変更
                    can_change_lane = False

            if can_change_lane:
                self._target_lane_idx = new_target_lane_idx
                self._last_lane_change_time = current_time_sec
                if new_target_lane_idx is not None:
                    target_str = "right (L0)" if new_target_lane_idx == 0 else "left (L2)" if new_target_lane_idx == 2 else "center (L1)"
                    self.get_logger().info(f"[LaneChange] Switching to lane {target_str} (lock for 0.3s)", throttle_duration_sec=1.0)
                else:
                    self.get_logger().info("[LaneChange] Switching back to free driving", throttle_duration_sec=1.0)
        else:
            self._target_lane_idx = new_target_lane_idx

        # パスへの車線反映
        self._reference_path.target_lane_idx = self._target_lane_idx
        self._reference_pathN.target_lane_idx = self._target_lane_idx
        self._reference_path10.target_lane_idx = self._target_lane_idx


        # 3. --- MPCの実行とエラーハンドリング（最終防衛ブレーキ） ---
        try:
            with self._stats.time_block("control"):
                u, max_delta = self._mpc.get_control()
        except (TypeError, ValueError):
            self.get_logger().error("🚨 MPC Solver Failed! Applying emergency safe deceleration.")
            emergency_v = max(0.0, v + self._mpc_cfg.a_min * dt) # 安全に減速
            u = np.array([emergency_v, self._last_u[1]]) # 前回の舵角を維持
            max_delta = np.abs(self._last_u[1])

        if self._ref_vel_configulator is not None:
            ref_vel_mps = self._ref_vel_configulator.get_ref_vel(self._mpc.model.wp_id)
            ref_vel_kmph = min(kmh_to_m_per_sec(ref_vel_mps), self._mpc_cfg.v_max)
            
            # ACC 車間距離制御の統合
            e_y = self._car.spatial_state.e_y
            lat_dist = abs(opponent_offset - e_y)
            if opponent_ahead is not None and lat_dist < 1.0 and not self._is_currently_overtaking:
                if opponent_distance < 12.0:
                    d_target = 7.5
                    K_p = 1.2
                    v_ref_acc = opponent_v_lead + K_p * (opponent_distance - d_target)
                    ref_vel_kmph = min(ref_vel_kmph, max(0.0, v_ref_acc))

            self._mpc.update_v_max(ref_vel_kmph)
            v_ref: List[float] = [ref_vel_kmph] * len(self._reference_path.waypoints)
            self._reference_path.set_v_ref(v_ref)

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

        # 4. --- Overtake / Follow Mode decision (Simplified) ---
        bug_acc_enabled = False
        
        bug_acc_enabled = False
        
        if self._is_currently_overtaking:
            # 🟢 追い越しを実行（ライン変更）している時は「黄色」
            self._pred_marker_color = YELLOW
        else:
            e_y = self._car.spatial_state.e_y
            lat_dist = abs(opponent_offset - e_y) if opponent_ahead is not None else np.inf
            
            if opponent_ahead is not None and lat_dist < 1.0 and opponent_distance < 12.0:
                # 🔴 完全に前をブロックされて追従・減速している時は「赤色」
                u[0] = min(u[0], opponent_v_lead)
                self._pred_marker_color = RED
            else:
                # 🔵 前方に誰もいない、または安全な通常走行時は「シアン」
                self._pred_marker_color = CYAN

        # 最初のスタート時（Lap 1 かつ wp < 20）かつ近くに相手がいる場合は、速度を抑えて後方に回り込む
        if is_start_grid and opp_pos_info is not None:
            opp_pos, opp_id = opp_pos_info
            opp_x, opp_y = opp_pos
            opp_dist = math.hypot(opp_x - pose.x, opp_y - pose.y)
            if opp_dist < 6.0:
                opp_vel_xy = self._v2x_tracker.velocity(opp_id)
                opp_speed = math.hypot(opp_vel_xy[0], opp_vel_xy[1])
                # 相手の速度 - 1.5m/s (時速5.4km程度遅く走る) に制限し、最低でも時速10kmで進む
                u[0] = min(u[0], max(10.0 / 3.6, opp_speed - 1.5))
                self._pred_marker_color = RED
        

        # 5. --- 加速出力計算 (標準PIDベース) ---
        acc = self.KP * (u[0] - v)
        acc = np.clip(acc, self._mpc_cfg.a_min, self._mpc_cfg.a_max)

        # ローパスフィルタ適用と指令送信
        acc = self._last_acc + (acc - self._last_acc) * self._mpc_cfg.accel_low_pass_gain
        u[1] = self._last_u[1] + (u[1] - self._last_u[1]) * self._mpc_cfg.steer_low_pass_gain

        self._last_acc = acc
        self._last_u[0] = u[0]
        self._last_u[1] = u[1]

        self._car.drive([v, u[1]])
        self._publish_control_command(now, u, acc, bug_acc_enabled)
        self._sim_logger.log(self._car, u, t)
        self._sim_logger.plot_animation(t, self._loop, self._current_laps, self._lap_times, is_colliding, u, self._mpc, self._car)

        if (self._mpc.current_prediction is not None) and (self._loop % (self._mpc_cfg.control_rate // 4) == 0):
            self._publish_mpc_pred_marker(self._mpc.current_prediction[0], self._mpc.current_prediction[1])

    def run(self) -> None:
        self._wait_until_clock_received()
        self._wait_until_odom_received()
        self._wait_until_gnss_received()
        self._wait_until_trajectory_received()
        self._wait_until_path_constraints_received()

        # initialize car states
        pose = self._get_current_pose()
        self._car.update_states(pose.x, pose.y, pose.theta)
        self._car.update_reference_path(self._car.reference_path)

        if self._ref_vel_configulator is None:
            self._publish_ref_path_marker(self._car.reference_path)

        self._pred_marker_color = CYAN

        # for i in range(10):
        #     self._obstacle_manager.push_next_obstacle()

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
            self._control_rate.sleep()

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
