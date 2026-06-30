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
from geometry_msgs.msg import Quaternion, Pose2D, Point, Vector3
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
                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)

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
                mpc_cfg.use_max_kappa_pred)


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

        self._reference_pathN = create_ref_path(self._map)
        self._reference_path10 = create_ref_path(self._map)

        self._carN = create_car(self._reference_pathN)
        self._car10 = create_car(self._reference_path10)

        self._mpc_cfg, self._mpcN = create_mpc(self._carN,self._cfg.mpc.N)
        _, self._mpc10 = create_mpc(self._car10, 9,self._cfg.mpc.R10)
        
        self._car = self._carN
        self._reference_path = self._reference_pathN
        self._mpc = self._mpcN
        
        compute_speed_profile(self._carN, self._mpc_cfg)
        compute_speed_profile(self._car10, self._mpc_cfg)

        self._ref_vel_configulator: Optional[ReferenceVelocityConfigulator] = create_ref_vel_configulator()

        self._trajectory: Optional[Trajectory] = None
        self._path_constraints = None

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

        # compensate steering angle for the real vehicle
        # AWSIMにおいても後段のactuation_cmd_converter でgainを考慮した指令を生成するため、実機/sim問わず
        # gain を掛ける
        cmd.lateral.steering_tire_angle *= self._mpc_cfg.steering_tire_angle_gain_var
        self._command_pub.publish(cmd)


    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

    def _control_mode_request_callback(self, msg):
        if msg.data and not self._enable_control:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _path_constraints_callback(self, msg: PathConstraints):
        self._reference_path.set_path_constraints(
            msg.upper_bounds, msg.lower_bounds, msg.rows, msg.cols)

    def _v2x_callback(self, msg: V2XVehiclePositionArray) -> None:
        self._v2x_tracker.update(msg)
        predictions = self._v2x_tracker.predict_all(self._v2x_t_samples)
        self._dynamic_obstacles = predictions_to_obstacles(
            predictions, self._v2x_vehicle_radius)
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
            # reference_path の境界線更新メソッドを呼び出す
            self._reference_path.update_boundaries_from_markers(
                np.array(left_pts), np.array(right_pts)
            )
            # 1回取得できれば十分なので、このサブスクライバを破棄する
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

                # 世界座標へ変換 (ub方向 = pi/2 + psi)
                angle_ub = math.fmod(math.pi / 2.0 + wp.psi + math.pi, 2 * math.pi) - math.pi
                curr_left_x = wp.x + ub_l * math.cos(angle_ub)
                curr_left_y = wp.y + ub_l * math.sin(angle_ub)
                curr_right_x = wp.x + lb_l * math.cos(angle_ub)
                curr_right_y = wp.y + lb_l * math.sin(angle_ub)

                next_angle_ub = math.fmod(math.pi / 2.0 + next_wp.psi + math.pi, 2 * math.pi) - math.pi
                next_left_x = next_wp.x + next_ub_l * math.cos(next_angle_ub)
                next_left_y = next_wp.y + next_ub_l * math.sin(next_angle_ub)
                next_right_x = next_wp.x + next_lb_l * math.cos(next_angle_ub)
                next_right_y = next_wp.y + next_lb_l * math.sin(next_angle_ub)

                # 頂点データ
                p_curr_left = Point(x=curr_left_x, y=curr_left_y, z=self._map_z)
                p_curr_right = Point(x=curr_right_x, y=curr_right_y, z=self._map_z)
                p_next_left = Point(x=next_left_x, y=next_left_y, z=self._map_z)
                p_next_right = Point(x=next_right_x, y=next_right_y, z=self._map_z)

                '''
                # 追い越しゾーンの場合は車線ごとの色、それ以外はグレー
                if has_overtake and ref_path.overtake_zone[wp_id]:
                # 全区間追い越し許可にしているため、常に車線ごとの色にする
                color_curr = lane_colors[lane_idx]
                color_next = lane_colors[lane_idx]
                else:
                    color_next = grey
                '''

                # 205 <= wp <= 245 の範囲では追い越し不可（グレー）にする
                if 205 <= wp_id <= 245:
                    color_curr = grey
                else:
                    color_curr = lane_colors[lane_idx]

                if 205 <= next_wp_id <= 245:
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

    def _control(self):
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

        if self._loop % 100 == 0:
            # update reference path
            if self._cfg.reference_path.update_by_topic: # type: ignore
                new_referece_path = self._create_reference_path_from_autoware_trajectory(self._trajectory)
                if new_referece_path is not None:
                    self._car.reference_path = new_referece_path
                    self._car.update_reference_path(self._car.reference_path)

            def plot_reference_path(car):
                import matplotlib.pyplot as plt
                import sys
                fig, ax = plt.subplots(1, 1)
                car.reference_path.show(ax)
                plt.show()
                sys.exit(1)
            # plot_reference_path(self._car)

        #メイン更新処理
        if self.USE_OBSTACLE_AVOIDANCE and self._obstacles_updated:
            self._obstacles_updated = False
            self._map.reset_map() #毎周期障害物を消して作り直している
            filtered_dynamic = self._filter_obstacles_to_corridor(self._dynamic_obstacles) #動的障害物フィルタリング.遠い車は消す
            self._map.add_obstacles(self._static_obstacles + filtered_dynamic) #障害物をマップに追加
            self._reference_path.reset_dynamic_constraints() #動的制約をリセット

        #可視化用衝突判定
        is_colliding = False 
        if self._last_colliding_time is not None:
            elapsed_from_last_colliding = (now - self._last_colliding_time).nanoseconds / 1e9
            if elapsed_from_last_colliding < 5.0:
                is_colliding = True
        
        #オドメトリ(x,y,yaw,v)取得
        pose = odom_to_pose_2d(self._odom) # type: ignore
        v = self._odom.twist.twist.linear.x

        #車両モデル更新
        self._car.update_states(pose.x, pose.y, pose.theta)

        #MPCの切り替え処理
        self._car.get_current_waypoint()
        wp = self._car.wp_id
        if (215 <= wp <= 245) or (280 <= wp <= 305):
            self._mpc = self._mpc10
            self._car = self._car10
            self._reference_path = self._reference_path10
            #print("using mpc10")
        else:
            self._mpc = self._mpcN
            self._car = self._carN
            self._reference_path = self._reference_pathN

        pose = odom_to_pose_2d(self._odom)
        self._mpc.previous_steering = self._last_u[1]

        self._car.update_states(pose.x, pose.y, pose.theta)
        self._car.get_current_waypoint()
        wp = self._car.wp_id

        # --- Overtaking Lane Selection Logic ---
        if not hasattr(self, '_target_lane_idx'):
            self._target_lane_idx = None

        N_total = self._reference_path.n_waypoints
        opponent_ahead = None
        opponent_offset = 0.0
        opponent_center = 0.0
        opponent_distance = 99999.0 #前方車両との距離
        opponent_v_lead = 0.0 #前方車両の速度
        min_wp_diff = 99999

        if self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker'):
            for vid in self._v2x_tracker.active_vehicle_ids():
                buf = self._v2x_tracker._samples.get(vid)
                if buf:
                    _, opp_x, opp_y = buf[-1]
                    opp_wp_id = self._car.get_closest_waypoint(opp_x, opp_y)
                    
                    wp_diff = (opp_wp_id - wp) % N_total
                    if 0 < wp_diff < 35:  # within ~21 meters
                        if wp_diff < min_wp_diff:
                            min_wp_diff = wp_diff
                            opponent_ahead = opp_wp_id
                            
                            # 前方車両の横オフセットを算出
                            opp_wp = self._reference_path.get_waypoint(opp_wp_id)
                            angle_ub = opp_wp.psi + math.pi / 2.0
                            dx = opp_x - opp_wp.x
                            dy = opp_y - opp_wp.y
                            opponent_offset = dx * math.cos(angle_ub) + dy * math.sin(angle_ub)
                            opponent_center = (opp_wp.ub + opp_wp.lb) / 2.0

                            # 車間距離（Euclidean距離）と前方車両の速度を取得
                            opponent_distance = math.hypot(opp_x - pose.x, opp_y - pose.y)
                            opp_vx, opp_vy = self._v2x_tracker.velocity(vid)
                            opponent_v_lead = math.hypot(opp_vx, opp_vy)

        # 205 <= wp <= 245 の範囲では追い越しをしないように設定
        if 205 <= wp <= 245:
            is_overtake_zone = False
        else:
            is_overtake_zone = True

        if opponent_ahead is not None:
            if is_overtake_zone:
                # 3車線用の選択ロジック:
                if opponent_offset > 0.0:
                    # 前方車両が左側にいる -> 右車線 (L0) を走行して追い越し
                    self._target_lane_idx = 0
                    target_str = "right (L0)"
                else:
                    # 前方車両が右側にいる -> 左車線 (L2) を走行して追い越し
                    self._target_lane_idx = 2
                    target_str = "left (L2)"
                self.get_logger().info(
                    f"[Overtake] Opponent ahead at wp {opponent_ahead}. Offset: {opponent_offset:.2f}. "
                    f"Overtaking via {target_str}.",
                    throttle_duration_sec=1.0
                )
            else:
                # 追い越し不可エリア -> 中央車線 (L1) を走行して追従
                self._target_lane_idx = 1
                self.get_logger().info(
                    f"[Follow] Opponent ahead at wp {opponent_ahead} in non-overtake zone. Following via center lane (L1).",
                    throttle_duration_sec=1.0
                )
        else:
            self._target_lane_idx = None

        # Apply target lane
        self._reference_path.target_lane_idx = self._target_lane_idx
        self._reference_pathN.target_lane_idx = self._target_lane_idx
        self._reference_path10.target_lane_idx = self._target_lane_idx
        
        # MPCの実行
        with self._stats.time_block("control"):
            u, max_delta = self._mpc.get_control()

        if self._ref_vel_configulator is not None:
            ref_vel_mps = self._ref_vel_configulator.get_ref_vel(self._mpc.model.wp_id)
            ref_vel_kmph = min(
                kmh_to_m_per_sec(ref_vel_mps),
                self._mpc_cfg.v_max)
            
            # --- ACC spacing control (車間距離維持制御) ---
            # 前方車両がいて、かつ自車の走行ライン上（横方向の差が 1.5m 未満）に他車が位置する場合に
            # 追従状態とみなして車間制御（5m〜10m）を有効化する。
            # 横方向の差が 1.5m 以上の場合は、別車線での追い越し中とみなして加速を許可する。
            e_y = self._car.spatial_state.e_y
            lat_dist = abs(opponent_offset - e_y)
            if opponent_ahead is not None and lat_dist < 1.5:
                if opponent_distance < 15.0:
                    d_target = 7.5  # 目標車間距離 (5m 〜 10m の中央値 7.5m)
                    K_p = 1.2       # 比例ゲイン
                    v_ref_acc = opponent_v_lead + K_p * (opponent_distance - d_target)
                    v_ref_acc = max(0.0, v_ref_acc)  # 後退は禁止のため下限は0
                    
                    ref_vel_kmph = min(ref_vel_kmph, v_ref_acc)
                    
                    if self._loop % int(self._mpc_cfg.control_rate) == 0:
                        self.get_logger().info(
                            f"[ACC] Distance to opp: {opponent_distance:.2f}m, Opp speed: {opponent_v_lead:.2f}m/s. "
                            f"Target speed limited to {ref_vel_kmph:.2f}m/s to maintain distance.",
                            throttle_duration_sec=1.0
                        )

            self._mpc.update_v_max(ref_vel_kmph)
            v_ref: List[float] = [ref_vel_kmph] * len(self._reference_path.waypoints)
            self._reference_path.set_v_ref(v_ref)

        # 停止命令がコマンドで入力させたら減速させる
        if not self._enable_control:
            last_v_cmd = self._last_u[0]
            if last_v_cmd < 0.5:
                u[0] = 0.0
            else:
                decel_v = last_v_cmd + self._mpc_cfg.a_min * dt
                u[0] = np.clip(decel_v, 0.0, self._mpc_cfg.v_max)

        #u[0] = 12.5         
        if len(u) == 0:
            self.get_logger().error("No control signal", throttle_duration_sec=1)
            u = [0.0, 0.0]
            # continue

        acc = 0.
        bug_acc_enabled = False


        #boostモードがONのとき
        if self.USE_BUG_ACC:
            def deg2rad(deg):
                return deg * np.pi / 180.0

            if abs(v) > kmh_to_m_per_sec(44.0) or \
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
        acc = self._last_acc + (acc - self._last_acc) * self._mpc_cfg.accel_low_pass_gain
        u[1] = self._last_u[1] + (u[1] - self._last_u[1]) * self._mpc_cfg.steer_low_pass_gain

        self._last_acc = acc
        self._last_u[0] = u[0]
        self._last_u[1] = u[1]

        # update car state (use v for feedback actual speed)
        self._car.drive([v, u[1]])

        # Publish control command
        self._publish_control_command(now, u, acc, bug_acc_enabled)

        # Log states
        self._sim_logger.log(self._car, u, t)
        self._sim_logger.plot_animation(t, self._loop, self._current_laps, self._lap_times, is_colliding, u, self._mpc, self._car)

        # 約 0.25 秒ごとに予測結果を表示
        if (self._mpc.current_prediction is not None) and (self._loop % (self._mpc_cfg.control_rate // 4) == 0):
            self._publish_mpc_pred_marker(self._mpc.current_prediction[0], self._mpc.current_prediction[1]) # type: ignore

        # 約 1 秒ごとに車線境界を表示
        if self._loop % int(self._mpc_cfg.control_rate) == 0:
            self._publish_lane_markers(self._reference_path)

    def run(self) -> None:
        self._wait_until_clock_received()
        self._wait_until_odom_received()
        self._wait_until_trajectory_received()
        self._wait_until_path_constraints_received()

        # initialize car states
        pose = odom_to_pose_2d(self._odom) # type: ignore
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
