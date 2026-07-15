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
from collections import deque

from std_msgs.msg import Empty, Bool, Float32MultiArray, Int32, String
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose2D, Point, Vector3, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA

from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

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
        self.declare_parameter("use_rviz_visualization", True)

        # get parameters
        self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.USE_BUG_ACC = self.get_parameter("use_boost_acceleration").get_parameter_value().bool_value
        self.USE_OBSTACLE_AVOIDANCE = self.get_parameter("use_obstacle_avoidance").get_parameter_value().bool_value
        self.use_stats = self.get_parameter("use_stats").get_parameter_value().bool_value
        self._config_path = config_path
        self._ref_vel_config_path: Optional[str] = ref_vel_config_path
        self._cfg = self._load_config()
        
        # Determine if RViz visualization should be active
        config_val = True
        try:
            config_val = self._cfg.common.use_rviz_visualization # type: ignore
        except AttributeError:
            pass
        param_val = self.get_parameter("use_rviz_visualization").get_parameter_value().bool_value
        self._rviz_active = param_val and config_val
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

        # Stuck Recovery configuration
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
            depth=1
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

        # Stuck Recovery publishers and subscribers
        self._awsim_control_mode_request_pub = self.create_publisher(
            Bool, "/awsim/control_mode_request_topic", 1)
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
        self._joy_cmd_pub = self.create_publisher(Joy, "/racing_kart/joy", 1)
        self._joy_cmd_pub_plain = self.create_publisher(Joy, "/joy", 1)

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
        if self.use_sim_time:
            self._awsim_state_sub = self.create_subscription(
                String, "/awsim/state", self._awsim_state_callback, 1)

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


    def _configure_stuck_recovery(self) -> None:
        cfg = getattr(self._cfg, "stuck_recovery", None)

        def get_cfg(name: str, default):
            return getattr(cfg, name, default) if cfg is not None else default

        self._stuck_recovery_enabled = bool(get_cfg("enabled", True))
        self._stuck_speed_threshold = float(get_cfg("speed_threshold", 0.15))
        self._stuck_forward_cmd_threshold = float(get_cfg("forward_cmd_threshold", 0.8))
        self._stuck_time_threshold = float(get_cfg("stuck_time_threshold", 2.0))
        self._stuck_gnss_distance_threshold = float(get_cfg("gnss_distance_threshold", 0.3))
        self._stuck_reverse_duration = float(get_cfg("reverse_duration", 3.0))
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
        self._stuck_use_actuation_cmd = bool(get_cfg("use_actuation_cmd", True))
        self._stuck_actuation_accel_cmd = abs(float(get_cfg("actuation_accel_cmd", 1.0)))
        self._stuck_actuation_brake_cmd = abs(float(get_cfg("actuation_brake_cmd", 0.0)))
        self._stuck_use_joy_cmd = bool(get_cfg("use_joy_cmd", True))
        self._stuck_joy_speed_axis = int(get_cfg("joy_speed_axis", 1))
        self._stuck_joy_steer_axis = int(get_cfg("joy_steer_axis", 3))
        self._stuck_joy_reverse_value = float(get_cfg("joy_reverse_value", -1.0))
        self._stuck_joy_steer_value = float(get_cfg("joy_steer_value", 0.0))
        self._stuck_joy_axes_size = int(get_cfg("joy_axes_size", 8))
        self._stuck_joy_buttons_size = int(get_cfg("joy_buttons_size", 13))
        self._stuck_joy_hold_buttons = [
            int(value)
            for value in str(get_cfg("joy_hold_buttons", "2")).split(",")
            if value.strip()
        ]
        self._gear_reverse_reports = {
            int(value)
            for value in str(get_cfg("reverse_gear_reports", "20")).split(",")
            if value.strip()
        }
        self._stuck_reverse_gear_command_override = get_cfg("reverse_gear_command", None)
        self._stuck_drive_gear_command_override = get_cfg("drive_gear_command", None)
        self._stuck_pre_reverse_gear_command_override = get_cfg("pre_reverse_gear_command", None)
        self._stuck_control_mode_requested = False
        self._last_stuck_gear_command = None
        self._stuck_reverse_drive_after = None
        self._stuck_reverse_drive_active = False
        self._stuck_recovery_started_at = None
        self._stuck_pre_reverse_until = None
        self._gear_report = None
        self._control_mode_report = None
        self._velocity_report = None
        self._awsim_state = None
        self._actuation_cmd_pub = None
        self._joy_cmd_pub = None
        self._joy_cmd_pub_plain = None
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
        self._last_control_mode = None
        self._autonomous_entered_at = None
        self._has_moved_once = False
        self._stuck_pre_drive_until = None
        self._stuck_wait_for_drive = False

        if self._stuck_recovery_enabled:
            self.get_logger().info(
                "[StuckRecovery] enabled: "
                f"speed<{self._stuck_speed_threshold:.2f}m/s for "
                f"{self._stuck_time_threshold:.1f}s -> reverse "
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
                f"source={__file__}"
            )

    def _apply_stuck_reverse_command(self, u) -> None:
        if self._stuck_reverse_command_mode in ("teleop", "awsim_reverse_button"):
            u[0] = self._stuck_forward_reverse_speed
        elif self._stuck_reverse_command_mode == "negative_speed_positive_accel":
            u[0] = -abs(self._stuck_reverse_speed)
        else:
            u[0] = -abs(self._stuck_reverse_speed)
        u[1] *= self._stuck_reverse_steering_scale

    def _publish_gear_command(self, now, command: int) -> None:
        if not self._stuck_send_gear_command or GearCommand is None or self._gear_cmd_pub is None:
            return
        if self._last_stuck_gear_command == command:
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

    def _publish_stuck_joy_command(self, now, joy_value: float = None) -> None:
        if not self._stuck_use_joy_cmd or self._joy_cmd_pub is None:
            return
        axes_size = max(
            self._stuck_joy_axes_size,
            self._stuck_joy_speed_axis + 1,
            self._stuck_joy_steer_axis + 1,
        )
        msg = Joy()
        msg.header.stamp = now.to_msg()
        msg.axes = [0.0] * axes_size
        msg.buttons = [0] * self._stuck_joy_buttons_size
        val = self._stuck_joy_reverse_value if joy_value is None else joy_value
        msg.axes[self._stuck_joy_speed_axis] = val
        msg.axes[self._stuck_joy_steer_axis] = self._stuck_joy_steer_value
        for button_index in self._stuck_joy_hold_buttons:
            if 0 <= button_index < len(msg.buttons):
                msg.buttons[button_index] = 1
        self._joy_cmd_pub.publish(msg)
        if self._joy_cmd_pub_plain is not None:
            self._joy_cmd_pub_plain.publish(msg)

    def _current_gear_is_reverse(self) -> bool:
        if self._gear_report is None:
            return False
        return int(getattr(self._gear_report, "report", -1)) in self._gear_reverse_reports

    def _gear_status_callback(self, msg) -> None:
        self._gear_report = msg

    def _control_mode_status_callback(self, msg) -> None:
        self._control_mode_report = msg

    def _velocity_status_callback(self, msg) -> None:
        self._velocity_report = msg

    def _awsim_state_callback(self, msg) -> None:
        self._awsim_state = getattr(msg, "data", None)

    def _request_awsim_control_mode_for_recovery(self) -> None:
        if not self._stuck_request_control_mode:
            return
        msg = Bool()
        msg.data = True
        self._awsim_control_mode_request_pub.publish(msg)
        self._stuck_control_mode_requested = True
        self.get_logger().info(
            "[StuckRecovery] requested AWSIM control mode (data=True).",
            throttle_duration_sec=1.0,
        )

    def _apply_stuck_recovery(self, now, u, actual_speed: float) -> bool:
        if not self._stuck_recovery_enabled:
            return False

        now_sec = float(now.nanoseconds) / 1e9

        if self._stuck_recovery_until is not None:
            # 1. タイムアウト判定
            if now_sec >= self._stuck_recovery_until:
                if self._stuck_reverse_drive_active:
                    self._stuck_reverse_drive_active = False
                    self._stuck_pre_drive_until = now_sec + 0.0
                    self._stuck_recovery_until = now_sec + 5.0
                    self._stuck_wait_for_drive = False
                    self.get_logger().info("[StuckRecovery] Reverse drive finished. Starting drive transition...")
                else:
                    self._stuck_recovery_until = None

            if self._stuck_recovery_until is not None:
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
                    gear_status_known = self._gear_report is not None
                    gear_is_drive = gear_status_known and getattr(self._gear_report, 'report', None) == self._gear_drive_command

                    if pre_driving:
                        self._publish_gear_command(now, self._gear_reverse_command)
                    else:
                        self._stuck_pre_drive_until = None
                        self._stuck_wait_for_drive = True
                        self._publish_gear_command(now, self._gear_drive_command)

                    waiting_for_drive = (
                        pre_driving or (self._stuck_send_gear_command and gear_status_known and not gear_is_drive)
                    )

                    if waiting_for_drive:
                        if pre_driving:
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
                        return True
                    else:
                        self._stuck_recovery_until = None

                else:
                    gear_status_known = self._gear_report is not None
                    gear_is_reverse = self._current_gear_is_reverse()
                    if (
                        gear_is_reverse
                        and self._stuck_recovery_started_at is None
                    ):
                        self._stuck_recovery_started_at = now_sec
                        self._stuck_recovery_until = now_sec + self._stuck_reverse_duration
                        self.get_logger().info(
                            "[StuckRecovery] AWSIM gear is REVERSE; starting reverse drive "
                            f"for {self._stuck_reverse_duration:.1f}s."
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
                        self._publish_gear_command(now, self._gear_reverse_command)
                    waiting_for_reverse = (
                        (pre_shifting or self._stuck_wait_for_reverse_gear)
                        and (
                            pre_shifting
                            or (gear_status_known and not gear_is_reverse)
                            or (
                                not gear_status_known
                                and self._stuck_reverse_drive_after is not None
                                and now_sec < self._stuck_reverse_drive_after
                            )
                        )
                    )
                    if waiting_for_reverse:
                        u[0] = 0.0
                        u[1] = 0.0
                        self._stuck_reverse_drive_active = False
                        if gear_status_known:
                            self.get_logger().warn(
                                "[StuckRecovery] pre-shift before reverse "
                                f"(command={self._gear_pre_reverse_command}, "
                                f"current={getattr(self._gear_report, 'report', None)})."
                                if pre_shifting
                                else "[StuckRecovery] waiting for AWSIM gear to become REVERSE "
                                f"(current={getattr(self._gear_report, 'report', None)}).",
                                throttle_duration_sec=1.0,
                             )
                    else:
                        self._apply_stuck_reverse_command(u)
                        self._publish_stuck_actuation_command(now, self._stuck_actuation_accel_cmd, 0.0, u[1])
                        self._stuck_reverse_drive_active = True
                    return True

            self._stuck_recovery_until = None
            self._stuck_cooldown_until = now_sec + self._stuck_cooldown
            self._stuck_since = None
            self._stuck_control_mode_requested = False
            self._last_stuck_gear_command = None
            self._stuck_reverse_drive_after = None
            self._stuck_reverse_drive_active = False
            self._stuck_recovery_started_at = None
            self._stuck_pre_reverse_until = None
            self._stuck_pre_drive_until = None
            self._stuck_wait_for_drive = False
            self._publish_gear_command(now, self._gear_drive_command)
            if self._stuck_request_control_mode:
                msg = Bool()
                msg.data = True
                self._awsim_control_mode_request_pub.publish(msg)
            self.get_logger().info(
                "[StuckRecovery] reverse finished; returning to MPC control "
                f"(gear={getattr(self._gear_report, 'report', None)})."
            )
            return False

        in_cooldown = (
            self._stuck_cooldown_until is not None
            and now_sec < self._stuck_cooldown_until
        )
        if in_cooldown:
            return False

        if abs(actual_speed) > 1.0:
            self._has_moved_once = True
        elif abs(actual_speed) < 0.1 and self._car.wp_id < 10:
            self._has_moved_once = False

        if not self._has_moved_once:
            self._stuck_since = None
            return False

        if abs(actual_speed) < self._stuck_speed_threshold and u[0] > self._stuck_forward_cmd_threshold:
            if self._stuck_since is None:
                self._stuck_since = now_sec
            elif now_sec - self._stuck_since >= self._stuck_time_threshold:
                self._stuck_recovery_until = now_sec + self._stuck_max_shift_wait
                self._stuck_reverse_drive_after = now_sec + self._stuck_gear_shift_delay
                self._stuck_pre_reverse_until = (
                    now_sec + self._stuck_pre_reverse_duration
                    if self._stuck_pre_reverse_duration > 0.0
                    else None
                )
                self._stuck_recovery_started_at = None
                self._last_stuck_gear_command = None
                self._request_awsim_control_mode_for_recovery()
                if self._stuck_pre_reverse_until is not None:
                    self._publish_gear_command(now, self._gear_pre_reverse_command)
                else:
                    self._publish_gear_command(now, self._gear_reverse_command)
                u[0] = 0.0
                u[1] = 0.0
                self._stuck_reverse_drive_active = False
                self.get_logger().warn(
                    f"[StuckRecovery] vehicle seems stuck; commanding reverse "
                    f"({self._stuck_reverse_command_mode}, "
                    f"gear={getattr(self._gear_report, 'report', None)}).",
                    throttle_duration_sec=0.5,
                )
                return True
        else:
            self._stuck_since = None

        return False


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
        if not getattr(self, "_rviz_active", True):
            return
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
        if not getattr(self, "_rviz_active", True):
            return
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
    # 追い越し可視化用マーカー関数（コースに沿った追い越し・復帰ライン版）
    # ==========================================
    def _publish_overtake_visualization(self, host_pose, opp_x, opp_y, pass_px, pass_py, decision: str):
        if not getattr(self, "_rviz_active", True):
            return
        if getattr(self, "_is_tight_curve", False):
            decision += " [TIGHT CURVE LIMIT ACTIVE]"
        now_msg = self.get_clock().now().to_msg()
        markers = MarkerArray()
        
        host_wp = self._car.get_closest_waypoint(host_pose.x, host_pose.y)
        N_total = self._reference_path.n_waypoints
        
        # 1. まずは広範囲（または全域）で最も「2D直線距離」が近いWPをラフに探す
        nearest_wp_idx = None
        min_dist_2d = float('inf')
        
        for idx in range(N_total):
            wp_pos = self._reference_path.get_waypoint(idx)
            dist = math.hypot(opp_x - wp_pos.x, opp_y - wp_pos.y)
            if dist < min_dist_2d:
                min_dist_2d = dist
                nearest_wp_idx = idx

        # 2. 【ヘアピン判定】もし見つかった最寄りWPが、自車から「インデックス上」は遥か遠くにあるのに、物理距離が超近い場合
        wp_diff_temp = (nearest_wp_idx - host_wp) % N_total
        if wp_diff_temp > N_total / 2:
            wp_diff_temp -= N_total

        # 総Waypoint数(N_total)の40%以上離れている場合のみに限定する（スタート直後の僅かなズレでの誤発動を完全に防ぐ）
        hairpin_threshold = int(N_total * 0.4)

        if abs(wp_diff_temp) > hairpin_threshold and min_dist_2d < 6.0:
            local_window = [(host_wp + offset) % N_total for offset in range(-25, 60)]
            min_dist_2d = float('inf')
            for idx in local_window:
                wp_pos = self._reference_path.get_waypoint(idx)
                dist = math.hypot(opp_x - wp_pos.x, opp_y - wp_pos.y)
                if dist < min_dist_2d:
                    min_dist_2d = dist
                    nearest_wp_idx = idx
        opp_wp = nearest_wp_idx
        
        # 1. 相手とのつながり (透明な太い赤/緑のロックオン帯)
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
            m_conn.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.2) # 追従時は赤
        else:
            m_conn.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.2) # 抜ける時は緑
            
        wp_diff = (opp_wp - host_wp) % N_total
        m_conn.points.append(Point(x=host_pose.x, y=host_pose.y, z=self._map_z))
        for i in range(1, wp_diff):
            curr_idx = (host_wp + i) % N_total
            wp_pt = self._reference_path.get_waypoint(curr_idx)
            m_conn.points.append(Point(x=wp_pt.x, y=wp_pt.y, z=self._map_z))
        m_conn.points.append(Point(x=opp_x, y=opp_y, z=self._map_z))
        markers.markers.append(m_conn)

        # 2. 🌟 劇的進化：コースに沿った「追い越し・復帰ライン」 (シアンの太い帯)
        m_path = Marker()
        m_path.header.frame_id = "map"
        m_path.header.stamp = now_msg
        m_path.ns = "overtake_target"
        m_path.id = 1
        m_path.type = Marker.LINE_STRIP
        m_path.action = Marker.ADD
        m_path.pose.orientation.w = 1.0
        m_path.scale.x = 0.6  # 0.4から0.6に太くして見やすく
        m_path.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.5) # シアンの半透明帯
        
        m_path.points.append(Point(x=host_pose.x, y=host_pose.y, z=self._map_z))
        
        # 相手のさらに先（15ウェイポイント先：約9m先）まで未来の予測線をコースに沿って計算
        # これにより「避けて、抜かして、元のレーンに戻る」までのS字の帯が作られます
        preview_wps = wp_diff + 15 
        
        for i in range(1, preview_wps + 1):
            curr_idx = (host_wp + i) % N_total
            wp_pt = self._reference_path.get_waypoint(curr_idx)
            angle_ub = wp_pt.psi + math.pi / 2.0
            
            # 自車から相手の手前までは、徐々にターゲット車線（L0 or L2）に向かって滑らかにオフセットを広げる
            if i <= wp_diff:
                # 相手の真横（wp_diff）に達した時に最大の回避幅になるよう線形補間
                blend_ratio = float(i) / float(wp_diff)
                if self._target_lane_idx == 2:   # 左から抜く
                    t_offset = (wp_pt.ub - self._car.width/2 - 0.2) * blend_ratio
                elif self._target_lane_idx == 0: # 右から抜く
                    t_offset = (wp_pt.lb + self._car.width/2 + 0.2) * blend_ratio
                else:
                    t_offset = 0.0
            else:
                # 相手を抜かした後のセクション（復帰フェーズ）：徐々に中央車線（0.0）に戻るように減衰させる
                remain_steps = i - wp_diff
                blend_ratio = max(0.0, 1.0 - (float(remain_steps) / 15.0))
                if self._target_lane_idx == 2:
                    t_offset = (wp_pt.ub - self._car.width/2 - 0.2) * blend_ratio
                elif self._target_lane_idx == 0:
                    t_offset = (wp_pt.lb + self._car.width/2 + 0.2) * blend_ratio
                else:
                    t_offset = 0.0
                    
            px = wp_pt.x + t_offset * math.cos(angle_ub)
            py = wp_pt.y + t_offset * math.sin(angle_ub)
            m_path.points.append(Point(x=px, y=py, z=self._map_z))
            
        markers.markers.append(m_path)
        self._overtake_vis_pub.publish(markers)

    def _clear_overtake_visualization(self):
        if not getattr(self, "_rviz_active", True):
            return
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

        # 🌟【制御遅延補償】Autowareから実車への送信タイムラグ（0.15秒）に対応するため、0.15秒先の未来位置を予測
        delay_time = 0.15  # 秒
        L = self._mpcN.model.length  # デフォルトモデルのホイールベースを使用
        psi_pred = pose.theta + (v / L) * math.tan(self._last_u[1]) * delay_time
        psi_pred = (psi_pred + math.pi) % (2 * math.pi) - math.pi
        x_pred = pose.x + v * math.cos(pose.theta) * delay_time
        y_pred = pose.y + v * math.sin(pose.theta) * delay_time

        # 1. 予測された未来の状態の取得と、コースに合わせたモデルの動的切り替え
        self._car.update_states(x_pred, y_pred, psi_pred)
        self._car.get_current_waypoint()
        wp = self._car.wp_id

        if (175 <= wp <= 245) or (260 <= wp <= 300) or (320 <= wp <= 340):
            self._mpc = self._mpc10
            self._car = self._car10
            self._reference_path = self._reference_path10
        else:
            self._mpc = self._mpcN
            self._car = self._carN
            self._reference_path = self._reference_pathN

        # ポインタ切り替え後に再度予測状態を同期
        self._car.update_states(x_pred, y_pred, psi_pred)
        self._mpc.previous_steering = self._last_u[1]

        # 🏎️【最終リファイン】ヘアピン旋回中のステア飽和・MPC破綻防止ロジック
        current_wp = self._reference_path.get_waypoint(wp)
        is_tight_curve = False
        if hasattr(current_wp, 'kappa') and abs(current_wp.kappa) > 0.15:
            is_tight_curve = True
        elif 210 <= wp <= 245:
            is_tight_curve = True

        self._is_tight_curve = is_tight_curve
        max_avoid = 1.0 if is_tight_curve else 1.4
        self._carN.max_avoid_offset = max_avoid
        self._car10.max_avoid_offset = max_avoid

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
        self._absolute_opponent_offset = None
        opponent_v_lead = 0.0
        min_wp_diff = 99999
        opp_id = None
        space_left = 0.0
        space_right = 0.0
        N_total = self._reference_path.n_waypoints

        # 🌟近距離ロック中であっても他車の位置/速度情報は常に最新に更新する
        if self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker'):
            opp_pos_info = self._get_opponent_position_and_id()
            if opp_pos_info is not None:
                opp_pos, opp_id = opp_pos_info
                opp_x, opp_y = opp_pos
                
                # 1. まずは広範囲（または全域）で最も「2D直線距離」が近いWPをラフに探す
                nearest_wp_idx = None
                min_dist_2d = float('inf')
                
                for idx in range(N_total):
                    wp_pos = self._reference_path.get_waypoint(idx)
                    dist = math.hypot(opp_x - wp_pos.x, opp_y - wp_pos.y)
                    if dist < min_dist_2d:
                        min_dist_2d = dist
                        nearest_wp_idx = idx

                # 2. 【ヘアピン判定】もし見つかった最寄りWPが、自車から「インデックス上」は遥か遠くにあるのに、物理距離が超近い場合
                wp_diff_temp = (nearest_wp_idx - wp) % N_total
                if wp_diff_temp > N_total / 2:
                    wp_diff_temp -= N_total

                # 総Waypoint数(N_total)の40%以上離れている場合のみに限定する（スタート直後の僅かなズレでの誤発動を完全に防ぐ）
                hairpin_threshold = int(N_total * 0.4)

                if abs(wp_diff_temp) > hairpin_threshold and min_dist_2d < 6.0:
                    local_window = [(wp + offset) % N_total for offset in range(-25, 60)]
                    min_dist_2d = float('inf')
                    for idx in local_window:
                        wp_pos = self._reference_path.get_waypoint(idx)
                        dist = math.hypot(opp_x - wp_pos.x, opp_y - wp_pos.y)
                        if dist < min_dist_2d:
                            min_dist_2d = dist
                            nearest_wp_idx = idx
                    self.get_logger().warning(f"🚨 Hairpin ghost projection detected! Activated Local Window for Search.")
                
                opp_wp_id = nearest_wp_idx
                
                wp_diff = (opp_wp_id - wp) % N_total
                
                if 0 < wp_diff < 35: 
                    if wp_diff < min_wp_diff:
                        min_wp_diff = wp_diff
                        opponent_ahead = opp_wp_id
                        
                        opp_wp_obj = self._reference_path.get_waypoint(opponent_ahead)
                        angle_ub = opp_wp_obj.psi + math.pi / 2.0
                        dx_opp = opp_x - opp_wp_obj.x
                        dy_opp = opp_y - opp_wp_obj.y
                        
                        # 相手のWaypoint（レースライン）に対する相対的な左右のズレ [m] (左が正, 右が負)
                        opponent_offset = -(dx_opp * math.cos(angle_ub) + dy_opp * math.sin(angle_ub))
                        
                        # レースラインが道路中央からどれだけズレているかを計算
                        # (ub:左側の道路幅(正), lb:右側の道路幅(負)。中央なら ub = -lb なので center_offset = 0)
                        if opp_wp_obj.ub is not None and opp_wp_obj.lb is not None:
                            center_offset = (opp_wp_obj.ub + opp_wp_obj.lb) / 2.0
                        else:
                            center_offset = 0.0
                            
                        # 道路中央線に対する絶対的な左右のズレ [m] (左が正, 右が負)
                        self._absolute_opponent_offset = opponent_offset - center_offset
                        self.get_logger().info(
                            f"[DEBUG_OPP] opp_x={opp_x:.3f} opp_y={opp_y:.3f} "
                            f"wp_x={opp_wp_obj.x:.3f} wp_y={opp_wp_obj.y:.3f} psi={opp_wp_obj.psi:.3f} "
                            f"dx={dx_opp:.3f} dy={dy_opp:.3f} "
                            f"opp_offset={opponent_offset:.3f} center_offset={center_offset:.3f} "
                            f"abs_offset={self._absolute_opponent_offset:.3f}",
                            throttle_duration_sec=0.1
                        )
                        opponent_distance = math.hypot(opp_x - pose.x, opp_y - pose.y)
                        opp_vel_xy = self._v2x_tracker.velocity(opp_id)
                        opponent_v_lead = math.hypot(opp_vel_xy[0], opp_vel_xy[1])

        # =========================================================================
        # 3. --- Lane selection based on hysteresis & Near Lock ---
        # =========================================================================
        prev_lane_idx = self._target_lane_idx
        decision_str = "FREE DRIVING"
        pass_px, pass_py = pose.x, pose.y

        # 近距離ロック条件の判定（相手が前方10wp以内に接近しており、すでに追従以外を選択している場合）
        is_near_lock = (opponent_ahead is not None) and (min_wp_diff <= 10) and (prev_lane_idx in [0, 2]) and (not is_colliding)

        if is_near_lock:
            # 🚨 相手に近すぎるため車線インデックスをフリーズ
            new_target_lane_idx = prev_lane_idx
            decision_str = "LOCK CURRENT ROUTE (Near)"
            
            opp_wp_obj = self._reference_path.get_waypoint(opponent_ahead)
            angle_ub = opp_wp_obj.psi + math.pi / 2.0
            target_offset = 0.0
            if new_target_lane_idx == 2:
                target_offset = (opp_wp_obj.ub - self._car.width/2 - 0.15)
            elif new_target_lane_idx == 0:
                target_offset = (opp_wp_obj.lb + self._car.width/2 + 0.15)
            
            pass_px = opp_wp_obj.x + target_offset * math.cos(angle_ub)
            pass_py = opp_wp_obj.y + target_offset * math.sin(angle_ub)
            self._publish_overtake_visualization(pose, opp_x, opp_y, pass_px, pass_py, decision_str)   

        elif opponent_ahead is not None:
            # 30wpより遠いときは中央(L1)を維持してドラフティング
            if min_wp_diff > 30:
                new_target_lane_idx = 1
                decision_str = "FOLLOW (Tailing)"
                pass_px = opp_x
                pass_py = opp_y
                self._publish_overtake_visualization(pose, opp_x, opp_y, pass_px, pass_py, decision_str)
            else:
                # 本格的な追い越し判断フェーズ
                opp_wp_obj = self._reference_path.get_waypoint(opponent_ahead)
                angle_ub = opp_wp_obj.psi + math.pi / 2.0
                
                opp_left_edge = opponent_offset + self._v2x_vehicle_radius
                opp_right_edge = opponent_offset - self._v2x_vehicle_radius
                
                space_left = opp_wp_obj.ub - opp_left_edge
                space_right = opp_right_edge - opp_wp_obj.lb
                required_space = self._car.width + 0.3 
                
                # 🌟【チャタリング完全絶滅版】現在選択中の車線への強力な未練ヒステリシス
                # 左右のスペース差の基準（0.3m）を、現在選んでいる車線に応じて動的に引き上げる
                # これにより、一瞬の座標のブレで右左がパタパタひっくり返るのを完全に防ぎます
                hys_margin = 0.6  # 🌟 通常時は 0.6m

                # -----------------------------------------------------------------
                # 🌟【追加修正】接近戦でのサンドイッチ衝突を防ぐ特効薬
                # 相手が15m以内に接近しているときは、意地を張るマージン（0.6m）を
                # 強制的に 0.1m まで引き下げ、壁の危険を察知したら瞬時に逆車線へ逃げられるようにする！
                # -----------------------------------------------------------------
                if opponent_distance < 15.0:
                    hys_margin = 0.1
                # -----------------------------------------------------------------
                
                # 現在すでに左(2)か右(0)を選んでいるなら、その車線側に下駄を履かせる
                left_bonus = 0.3 if prev_lane_idx == 2 else 0.0
                right_bonus = 0.3 if prev_lane_idx == 0 else 0.0

                if (space_left + left_bonus) > (space_right + right_bonus) + hys_margin:
                    preferred_lane = 2 # 左車線
                    preferred_space = space_left
                    alt_lane = 0
                    alt_space = space_right
                elif (space_right + right_bonus) > (space_left + left_bonus) + hys_margin:
                    preferred_lane = 0 # 右車線
                    preferred_space = space_right
                    alt_lane = 2
                    alt_space = space_left
                else:
                    # 左右のスペースに圧倒的な差がない（拮抗している）場合は、前回の意思決定を「絶対維持」
                    if prev_lane_idx in [0, 2]:
                        preferred_lane = prev_lane_idx
                        preferred_space = space_left if prev_lane_idx == 2 else space_right
                        alt_lane = 0 if prev_lane_idx == 2 else 2
                        alt_space = space_right if prev_lane_idx == 2 else space_left
                    else:
                        # 完全にニュートラルな状態からの初期選択
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

                if preferred_space > required_space:
                    new_target_lane_idx = preferred_lane
                    decision_str = "OVERTAKE LEFT" if preferred_lane == 2 else "OVERTAKE RIGHT"
                    target_offset = (opp_wp_obj.ub - self._car.width/2 - 0.15) if preferred_lane == 2 else (opp_wp_obj.lb + self._car.width/2 + 0.15)
                    pass_px = opp_wp_obj.x + target_offset * math.cos(angle_ub)
                    pass_py = opp_wp_obj.y + target_offset * math.sin(angle_ub)
                elif alt_space > required_space:
                    new_target_lane_idx = alt_lane
                    decision_str = "OVERTAKE LEFT" if alt_lane == 2 else "OVERTAKE RIGHT"
                    target_offset = (opp_wp_obj.ub - self._car.width/2 - 0.15) if alt_lane == 2 else (opp_wp_obj.lb + self._car.width/2 + 0.15)
                    pass_px = opp_wp_obj.x + target_offset * math.cos(angle_ub)
                    pass_py = opp_wp_obj.y + target_offset * math.sin(angle_ub)
                else:
                    new_target_lane_idx = 1
                    decision_str = "FOLLOW (Blocked)"
                    pass_px = opp_x
                    pass_py = opp_y
                
                self.get_logger().info(
                    f"[DEBUG_SPACE] space_left={space_left:.3f} space_right={space_right:.3f} "
                    f"preferred_lane={preferred_lane} preferred_space={preferred_space:.3f} "
                    f"required_space={required_space:.3f} new_target={new_target_lane_idx} decision={decision_str}",
                    throttle_duration_sec=0.1
                )
                
                # =========================================================================
                # 🌟【大修正】他車の左右位置を自車基準ではなく「コース基準」で絶対判定するオーバーライド
                # =========================================================================
                if self._absolute_opponent_offset is not None and new_target_lane_idx != 1:
                    # 3. コース基準での絶対的な車線判定
                    if self._absolute_opponent_offset > 0.4:
                        opp_actual_lane = 2  # 相手は絶対に「左車線」にいる
                    elif self._absolute_opponent_offset < -0.4:
                        opp_actual_lane = 0  # 相手は絶対に「右車線」にいる
                    else:
                        opp_actual_lane = 1  # 相手は「中央車線」にいる

                    # 4. 🌟【オーバーライド】相手が左(2)にいるなら、自車は「右回避(L0)」しか選べないように強制ロック！
                    if opp_actual_lane == 2:
                        new_target_lane_idx = 0  # ➔ 右回避(L0)に強制指定！
                        decision_str = "EMERGENCY OVERRIDE: TARGET RIGHT (Opponent is Left)"
                        target_offset = (opp_wp_obj.lb + self._car.width/2 + 0.15)
                        pass_px = opp_wp_obj.x + target_offset * math.cos(angle_ub)
                        pass_py = opp_wp_obj.y + target_offset * math.sin(angle_ub)
                    
                    # 相手が右(0)にいるなら、自車は「左回避(L2)」しか選べないように強制ロック！
                    elif opp_actual_lane == 0:
                        new_target_lane_idx = 2  # ➔ 左回避(L2)に強制指定！
                        decision_str = "EMERGENCY OVERRIDE: TARGET LEFT (Opponent is Right)"
                        target_offset = (opp_wp_obj.ub - self._car.width/2 - 0.15)
                        pass_px = opp_wp_obj.x + target_offset * math.cos(angle_ub)
                        pass_py = opp_wp_obj.y + target_offset * math.sin(angle_ub)

                    self.get_logger().info(
                        f"[DEBUG_OVERRIDE] opp_actual_lane={opp_actual_lane} "
                        f"new_target={new_target_lane_idx} decision={decision_str}",
                        throttle_duration_sec=0.1
                    )

                self._publish_overtake_visualization(pose, opp_x, opp_y, pass_px, pass_py, decision_str)

                # 🌟【重要】選択したレーンが死んでいるかチェックする再評価ロジック
                if self._target_lane_idx in [0, 2]: # 現在すでに回避モードなら
                    # 相手と物理的に接触しそうなほど近いか？（例：2.5m以内）
                    if opponent_distance < 2.5:
                        # 相手が事故って止まっている等で、現在選んでいるレーンが物理的にブロックされているかチェック
                        is_blocked = (self._target_lane_idx == 2 and space_left < required_space) or \
                                     (self._target_lane_idx == 0 and space_right < required_space)
                        
                        if is_blocked:
                            # 逆側の車線が空いていれば、即座にレーン変更を許可（ロックを無視する）
                            new_target_lane_idx = alt_lane if (alt_space > required_space) else self._target_lane_idx
                            if new_target_lane_idx != prev_lane_idx:
                                 self.get_logger().info(f"🚨 Path Blocked! Forcing emergency lane swap to {new_target_lane_idx}")
                                 self._last_lane_change_time = 0.0 # 強制的にロックを解除
            
        else:
            new_target_lane_idx = None
            self._clear_overtake_visualization()

        # スタートグリッド保護処理
        is_start_grid = (self._current_laps == 1 and 25 <= wp < 50)
        if is_start_grid and opp_id is not None:
            opp_dist = math.hypot(opp_x - pose.x, opp_y - pose.y)
            if opp_dist < 6.0:
                new_target_lane_idx = 1
                decision_str = "START FOLLOW"
                self._publish_overtake_visualization(pose, opp_x, opp_y, opp_x, opp_y, decision_str)

        # =========================================================================
        # 🌟【微調整】難所（N=9区間）の手前での追い越し強制キャンセルガード
        # =========================================================================
        if (160 <= wp < 195) and (opponent_ahead is not None) and (min_wp_diff < 35):
            new_target_lane_idx = 1
            decision_str = "FORCE FOLLOW (Approach to N=9 Danger Zone)"
            pass_px = opp_x
            pass_py = opp_y
            self._publish_overtake_visualization(pose, opp_x, opp_y, pass_px, pass_py, decision_str)

        # =========================================================================
        # チャタリング防止 (1.0s タイマーロック)
        # =========================================================================
        current_time_sec = float(now.nanoseconds) / 1e9

        if is_near_lock:
            self._target_lane_idx = new_target_lane_idx
        elif new_target_lane_idx != prev_lane_idx:
            can_change_lane = True
            if self._last_lane_change_time is not None:
                elapsed = current_time_sec - self._last_lane_change_time
                if elapsed < 0.6: 
                    can_change_lane = False

            if can_change_lane:
                self._target_lane_idx = new_target_lane_idx
                self._last_lane_change_time = current_time_sec
                if new_target_lane_idx is not None:
                    target_str = "right (L0)" if new_target_lane_idx == 0 else "left (L2)" if new_target_lane_idx == 2 else "center (L1)"
                    self.get_logger().info(f"[LaneChange] Switching to lane {target_str}", throttle_duration_sec=1.0)
                else:
                    self.get_logger().info("[LaneChange] Switching back to free driving", throttle_duration_sec=1.0)
        else:
            self._target_lane_idx = new_target_lane_idx

        # パスへの車線反映
        self._reference_path.target_lane_idx = self._target_lane_idx
        self._reference_pathN.target_lane_idx = self._target_lane_idx
        self._reference_path10.target_lane_idx = self._target_lane_idx

        # Determine safety margin for MPC solver
        if opponent_ahead is not None and opponent_distance < 12.0:
            # When an opponent is nearby (within 12m), use a slightly relaxed but safe margin.
            # 0.85 * 0.775m = 0.66m (provides about 58cm of physical buffer).
            solver_margin = self._mpc.model.safety_margin * 0.85
        else:
            solver_margin = self._mpc.model.safety_margin

        # 3. --- MPCの実行とエラーハンドリング ---
        try:
            with self._stats.time_block("control"):
                u, max_delta = self._mpc.get_control(solver_margin)
        except (TypeError, ValueError) as e:
            import traceback
            self.get_logger().error(f"🚨 MPC Solver Failed! Applying Pure Pursuit recovery steering. Error: {e}")
            self.get_logger().error(traceback.format_exc())
            
            # Pure Pursuit recovery steering fallback
            try:
                # 1. Get closest waypoint index on the reference path
                wp_id = self._mpc.model.get_closest_waypoint(pose.x, pose.y)
                # 2. Look ahead by 15 waypoints (approx. 9 meters)
                lookahead_wp_id = (wp_id + 15) % len(self._reference_path.waypoints)
                lookahead_wp = self._reference_path.waypoints[lookahead_wp_id]
                
                # 3. Compute relative angle alpha to the lookahead point
                dx = lookahead_wp.x - pose.x
                dy = lookahead_wp.y - pose.y
                yaw = pose.theta
                
                target_angle = math.atan2(dy, dx)
                alpha = target_angle - yaw
                alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
                
                # 4. Pure Pursuit formula: delta = atan2(2 * L * sin(alpha), lookahead_distance)
                L = self._mpc.model.length
                lookahead_dist = math.hypot(dx, dy)
                if lookahead_dist > 0.5:
                    pure_pursuit_delta = math.atan2(2.0 * L * math.sin(alpha), lookahead_dist)
                else:
                    pure_pursuit_delta = self._last_u[1]
                    
                # Limit the rate change of steer angle
                max_delta_change = self._mpc_cfg.steer_rate_max * dt
                delta = np.clip(
                    pure_pursuit_delta,
                    self._last_u[1] - max_delta_change,
                    self._last_u[1] + max_delta_change
                )
                delta = np.clip(delta, -self._mpc_cfg.delta_max, self._mpc_cfg.delta_max)
            except Exception as ex:
                self.get_logger().error(f"🚨 Pure Pursuit fallback failed: {ex}")
                delta = self._last_u[1]

            emergency_v = max(0.0, v + self._mpc_cfg.a_min * dt)
            u = np.array([emergency_v, delta])
            max_delta = np.abs(delta)

        # 速度計画の上書き処理
        is_overtaking_state = self._target_lane_idx in [0, 2]
        if self._ref_vel_configulator is not None:
            ref_vel_mps = self._ref_vel_configulator.get_ref_vel(self._mpc.model.wp_id)
            ref_vel_kmph = min(kmh_to_m_per_sec(ref_vel_mps), self._mpc_cfg.v_max)
            
            e_y = self._car.spatial_state.e_y
            lat_dist = abs(opponent_offset - e_y)
            # 追従減速（ACC）と追い越し最終防衛リミッター
            if opponent_ahead is not None:
                if not is_overtaking_state:
                    # ① 通常走行・追従時の標準ACC
                    if lat_dist < 1.0 and opponent_distance < 12.0:
                        d_target = 6.5
                        K_p = 1.2
                        v_ref_acc = opponent_v_lead + K_p * (opponent_distance - d_target)
                        ref_vel_kmph = min(ref_vel_kmph, max(0.0, v_ref_acc))
                else:
                    # ② 追い越しライン走行中の最終防衛リミッター
                    if opponent_distance < 5.0 and lat_dist < 0.6:
                        safe_overtake_vel = opponent_v_lead + (5.0 / 3.6)
                        ref_vel_kmph = min(ref_vel_kmph, max(0.0, safe_overtake_vel))

                    # 🌟【スタック完全脱出版】挟み込み・壁激突を防止しつつ、デッドロックを回避する
                    # 追い越し中（L0 or L2）かつ、相手との距離が10m以内に近づいている状況で
                    if opponent_distance < 10.0:
                        my_target_lane = self._target_lane_idx if self._target_lane_idx is not None else 1
                        avail_width = space_left if my_target_lane == 2 else (space_right if my_target_lane == 0 else (space_left + space_right))
                        # 自分が進もうとしているレーンの有効幅が、実車幅（1.50m）未満に潰れている場合
                        if avail_width < 1.50:
                            # 相手の速度をベースに退避速度を計算
                            safe_abort_vel = opponent_v_lead - (5.0 / 3.6)
                            
                            # 相手が停止(0km/h)している場合、目標速度が0になってフリーズするのを防ぐため、
                            # 最低でも「時速 7.0km/h (約1.94m/s)」のツッコミ速度を絶対保証する！
                            min_abort_vel_mps = 7.0 / 3.6
                            ref_vel_kmph = min(ref_vel_kmph, max(min_abort_vel_mps, safe_abort_vel))
                            
                            # 車線選択のホールドを強制解除し、中央（L1）に戻して仕切り直す
                            self._target_lane_idx = 1
                            self._last_lane_change_time = 0.0 # 即時変更を許可

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

        # 4. --- マーカーカラー決定ロジック ---
        bug_acc_enabled = False
        
        if is_overtaking_state:
            # 追い越し中
            if opponent_ahead is not None and opponent_distance < 5.0 and lat_dist < 1.1:
                # 真後ろで詰まっている時は安全のため 相手速度 + 5km/h に制限
                max_allowed_vel = opponent_v_lead + (5.0 / 3.6)
                u[0] = min(u[0], max_allowed_vel)
                self._pred_marker_color = RED
            else:
                self._pred_marker_color = YELLOW  # 🟢 追い越しライン爆走中は黄色
        else:
            # 追従中
            if opponent_ahead is not None and lat_dist < 1.0 and opponent_distance < 12.0:
                # 通常追従時は相手の速度以下に制限して追従する
                u[0] = min(u[0], opponent_v_lead)
                self._pred_marker_color = RED    # 🔴 前が詰まって追従中は赤
            else:
                self._pred_marker_color = CYAN   # 🔵 通常の単独レコードライン走行はシアン

        # スタートグリッド時の低速制限
        if is_start_grid and opp_id is not None:
            opp_dist = math.hypot(opp_x - pose.x, opp_y - pose.y)
            if opp_dist < 6.0:
                opp_vel_xy = self._v2x_tracker.velocity(opp_id)
                opp_speed = math.hypot(opp_vel_xy[0], opp_vel_xy[1])
                u[0] = min(u[0], max(10.0 / 3.6, opp_speed - 1.5))
                self._pred_marker_color = RED

        # 5. --- 加速出力計算 (標準PIDベース) ---
        recovering_from_stuck = self._apply_stuck_recovery(now, u, v)

        if recovering_from_stuck:
            bug_acc_enabled = False
            if not self._stuck_reverse_drive_active:
                acc = -8.0
            elif self._stuck_reverse_command_mode in ("negative_speed_positive_accel", "awsim_reverse_button"):
                if self._stuck_reverse_acceleration_positive:
                    acc = abs(self._stuck_reverse_acceleration)
                else:
                    acc = -abs(self._stuck_reverse_acceleration)
            else:
                acc = self._stuck_reverse_acceleration
            self.get_logger().info(
                f"[StuckRecovery] reverse cmd speed={u[0]:.2f} acc={acc:.2f} "
                f"actuation=({self._stuck_actuation_accel_cmd:.2f},"
                f"{self._stuck_actuation_brake_cmd:.2f}) "
                f"gear={getattr(self._gear_report, 'report', None)} "
                f"mode={getattr(self._control_mode_report, 'mode', None)} "
                f"vel={getattr(self._velocity_report, 'longitudinal_velocity', None)} "
                f"state={self._awsim_state}",
                throttle_duration_sec=1.0,
            )
            self._pred_marker_color = YELLOW
        else:
            acc = self.KP * (u[0] - v)
            acc = np.clip(acc, self._mpc_cfg.a_min, self._mpc_cfg.a_max)
            acc = self._last_acc + (acc - self._last_acc) * self._mpc_cfg.accel_low_pass_gain
            u[1] = self._last_u[1] + (u[1] - self._last_u[1]) * self._mpc_cfg.steer_low_pass_gain

        # =========================================================================
        # 🌟【ハルキ専用】混走バトル・コックピットログ（1行集約デバッグ）
        # =========================================================================
        if opponent_ahead is not None:
            # 1. 相手の車線インデックスを文字化 (L0:右, L1:中, L2:左)
            # コース絶対基準のオフセットがあればそちらを優先してログ判定する
            abs_offset = getattr(self, "_absolute_opponent_offset", None)
            if abs_offset is not None:
                if abs_offset > 0.4:
                    opp_lane_str = "L2(左)"
                elif abs_offset < -0.4:
                    opp_lane_str = "L0(右)"
                else:
                    opp_lane_str = "L1(中)"
            else:
                if opponent_offset > 0.5:
                    opp_lane_str = "L2(左)"
                elif opponent_offset < -0.5:
                    opp_lane_str = "L0(右)"
                else:
                    opp_lane_str = "L1(中)"

            # 2. 自分が選択したターゲットコースを文字化
            my_target_lane = self._target_lane_idx if self._target_lane_idx is not None else 1
            my_lane_str = f"L{my_target_lane}"

            # 3. 自分が抜ける残り幅（幾何学的コリドーの最小隙間）
            # space_left / space_right から、選択した車線側の有効幅を代入
            avail_width = space_left if my_target_lane == 2 else (space_right if my_target_lane == 0 else (space_left + space_right))

            # 4. 大きく回避ステアが動いているかのインジケータ生成
            # delta_deg: 現在のタイヤ角（度数法）。プラスが左、マイナスが右
            delta_deg = math.degrees(u[1])
            
            if delta_deg > 5.0:
                steer_dir_sign = f"{delta_deg:5.1f}° [ 🟢<<< 左回避 ]"
            elif delta_deg < -5.0:
                steer_dir_sign = f"{delta_deg:5.1f}° [ 🟢>>> 右回避 ]"
            else:
                steer_dir_sign = f"{delta_deg:5.1f}° [  ｜  直進傾向 ]"

            # 5. 条件に応じたアイコン変更（5m未満の超接近戦は警告アラートにする）
            status_icon = "🚨" if opponent_distance < 5.0 else "🏎️"

            # 6. ロギング実行（知りたい5項目をすべて1行に凝縮）
            self.get_logger().info(
                f"{status_icon}[wp:{wp:3d}] "
                f"相手距離:{opponent_distance:4.1f}m ({opp_lane_str}) "
                f"➔ 自車選択:{my_lane_str} (有効幅:{avail_width:3.1f}m) "
                f"| ステア:{steer_dir_sign}",
                throttle_duration_sec=0.1
            )

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
