#!/usr/bin/env python3

import yaml
import math
import time
from typing import List, Tuple, Optional, NamedTuple
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import numpy as np
import copy
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
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

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
    evaluate_stopped_lead_overtake,
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
                mpc_cfg.understeer_coeff)


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
        #self._reference_path10_center = create_ref_path(self._map, custom_csv_path=center_csv)
        self._carN_center = create_car(self._reference_pathN_center)
        #self._car10_center = create_car(self._reference_path10_center)
        self._mpc_cfg_center, self._mpcN_center = create_mpc(self._carN_center, self._cfg.mpc.N)
        #_, self._mpc10_center = create_mpc(self._car10_center, 9, self._cfg.mpc.R10)
        compute_speed_profile(self._carN_center, self._mpc_cfg_center)
        #compute_speed_profile(self._car10_center, self._mpc_cfg_center)

        # Lane lock on curves configuration
        self._curve_lane_lock_enabled = bool(getattr(cfg_ref_path, "curve_lane_lock_enabled", True))
        self._curve_lane_lock_kappa_threshold = float(getattr(cfg_ref_path, "curve_lane_lock_kappa_threshold", 0.05))
        self._curve_lane_lock_lookahead_wps = int(getattr(cfg_ref_path, "curve_lane_lock_lookahead_wps", 15))
        self._curve_lane_lock_lookbehind_wps = int(getattr(cfg_ref_path, "curve_lane_lock_lookbehind_wps", 5))
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
        self._path_constraints = None
        self._last_lane_change_time = None
        self._target_lane_idx = None

        switch_cfg = getattr(self._cfg, "trajectory_switch", None)
        self._trajectory_enter_center_wps = int(
            getattr(switch_cfg, "enter_center_wps", 25))
        self._trajectory_exit_center_wps = int(
            getattr(switch_cfg, "exit_center_wps", 35))
        self._trajectory_behind_release_wps = int(
            getattr(switch_cfg, "behind_release_wps", 12))
        self._trajectory_switch_min_hold = float(
            getattr(switch_cfg, "min_hold_sec", 1.0))
        self._trajectory_exit_confirm = float(
            getattr(switch_cfg, "exit_confirm_sec", 0.5))
        self._trajectory_last_switch_time = None
        self._trajectory_clear_since = None
        self._trajectory_switch_reason = "initial"
        self._trajectory_vehicle_id = None
        self._prev_opponent_ahead_detected = False

        overtake_cfg = getattr(self._cfg, "stopped_vehicle_overtake", None)
        self._stopped_lead_speed_threshold = float(
            getattr(overtake_cfg, "speed_threshold", 0.3))
        self._forced_overtake_speed = float(
            getattr(overtake_cfg, "max_speed", 10.0))
        self._close_obstacle_reverse_distance = float(
            getattr(overtake_cfg, "reverse_distance", 3.0))
        self._close_obstacle_infeasible_cycles = int(
            getattr(overtake_cfg, "infeasible_cycles", 5))
        self._forced_overtake_vehicle_id = None
        self._close_obstacle_reverse_requested = False

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
        self._awsim_command_speed = 0.0

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
        self._stuck_collision_window = float(get_cfg("collision_window", 20.0))
        self._stuck_collision_count_threshold = int(get_cfg("collision_count_threshold", 3))
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
        self._gnss_history = []
        self._collision_times = []

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
            self._awsim_admin_status_sub = self.create_subscription(
                Float32MultiArray, "/admin/awsim/status", self._awsim_admin_status_callback, 1)
            self._awsim_state_sub = self.create_subscription(
                String, "/awsim/state", self._awsim_state_callback, 1)
            self._condition_sub = self.create_subscription(
                Int32, "/aichallenge/pitstop/condition", self._condition_callback, 1)

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
                    # 後退駆動が終わったので、前進復帰シーケンスを開始する
                    self._stuck_reverse_drive_active = False
                    self._stuck_pre_drive_until = now_sec + 0.0
                    self._stuck_recovery_until = now_sec + 5.0
                    self._stuck_wait_for_drive = False
                    self.get_logger().info("[StuckRecovery] Reverse drive finished. Starting drive transition...")
                else:
                    # トルク抜きやシフト待ちでタイムアウトした場合、あるいは前進復帰中のタイムアウト
                    self._stuck_recovery_until = None

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
                        self._stuck_recovery_until = None  # シフト完了につき正常終了へ

                # 後退（REVERSE）リカバリーシーケンス
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
                        if pre_shifting:
                            u[0] = 0.0
                        else:
                            u[0] = abs(self._stuck_forward_reverse_speed)
                        u[1] = 0.0
                        self._stuck_reverse_drive_active = False
                        #if gear_status_known:
                            #self.get_logger().warn(
                            #    "[StuckRecovery] pre-shift before reverse "
                            #    f"(command={self._gear_pre_reverse_command}, "
                            #    f"current={getattr(self._gear_report, 'report', None)})."
                            #    if pre_shifting
                            #    else "[StuckRecovery] waiting for AWSIM gear to become REVERSE "
                            #    f"(current={getattr(self._gear_report, 'report', None)}).",
                            #    throttle_duration_sec=1.0,
                            # )
                    else:
                        self._apply_stuck_reverse_command(u)
                        self._publish_stuck_actuation_command(now, self._stuck_actuation_accel_cmd, 0.0, u[1])
                        self._stuck_reverse_drive_active = True
                    return True

            # 3. 正常終了・タイムアウト終了後のリセット処理
            self._stuck_recovery_until = None
            self._stuck_cooldown_until = now_sec + self._stuck_cooldown
            self._stuck_since = None
            self._gnss_history = []
            self._stuck_control_mode_requested = False
            self._last_stuck_gear_command = None
            self._stuck_reverse_drive_after = None
            self._stuck_reverse_drive_active = False
            self._stuck_recovery_started_at = None
            self._stuck_pre_reverse_until = None
            self._stuck_pre_drive_until = None
            self._stuck_wait_for_drive = False
            self._publish_gear_command(now, self._gear_drive_command)
            # 自動運転モードを明示的にONにする
            if self._stuck_request_control_mode:
                msg = Bool()
                msg.data = True
                self._awsim_control_mode_request_pub.publish(msg)
            #self.get_logger().info(
            #    "[StuckRecovery] reverse finished; returning to MPC control "
            #    f"(gear={getattr(self._gear_report, 'report', None)})."
            #)
            return False

        # 4. 通常時の判定（誤爆防止マスク＆スタック検知）
        in_cooldown = (
            self._stuck_cooldown_until is not None
            and now_sec < self._stuck_cooldown_until
        )
        if in_cooldown:
            return False

        # 動き出した実績の管理（シミュレータ起動・リセット時の誤爆防止）
        if abs(actual_speed) > 1.0:
            self._has_moved_once = True
        elif abs(actual_speed) < 0.1 and self._car.wp_id < 10:
            self._has_moved_once = False

        if not self._has_moved_once:
            self._stuck_since = None
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
        if is_stuck_state and (
            u[0] > self._stuck_forward_cmd_threshold
            or should_recover_from_close_obstacle
        ):
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
                self._close_obstacle_reverse_requested = False
                
                # 詳細な検知理由をログ出力
                reason_speed = f"speed({actual_speed:.2f}) < threshold({self._stuck_speed_threshold:.2f})"
                reason_gnss = f"gnss_dist({gnss_moved_dist:.3f}) < threshold({self._stuck_gnss_distance_threshold:.2f})" if gnss_moved_dist is not None else "gnss_dist=None"
                #self.get_logger().warn(
                #    f"[StuckRecovery] vehicle seems stuck ({reason_speed} or {reason_gnss}); commanding reverse "
                #    f"({self._stuck_reverse_command_mode}, "
                #    f"gear={getattr(self._gear_report, 'report', None)}).",
                #    throttle_duration_sec=0.5,
                #)
                return True
        else:
            self._stuck_since = None

        return False


    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

    def _gnss_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._gnss_pose = msg

    def _control_mode_request_callback(self, msg):
        if msg.data and not self._enable_control:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _path_constraints_callback(self, msg: PathConstraints):
        self._reference_path.set_path_constraints(
            msg.upper_bounds, msg.lower_bounds, msg.rows, msg.cols)

    def _v2x_callback(self, msg: V2XVehiclePositionArray) -> None:
        # If obstacle avoidance is disabled, clear tracker and bypass V2X processing entirely.
        if not self.USE_OBSTACLE_AVOIDANCE:
            if hasattr(self, '_v2x_tracker'):
                self._v2x_tracker._active = []
            return

        # Create a new list excluding the ego vehicle
        filtered_vehicles = []
        for v in msg.vehicles:
            if hasattr(self, '_ego_vehicle_id') and v.vehicle_id == self._ego_vehicle_id:
                continue
            filtered_vehicles.append(v)
        
        # Override msg.vehicles with the filtered list
        msg.vehicles = filtered_vehicles

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
            left_arr = np.array(left_pts)
            right_arr = np.array(right_pts)
            # Race / Center 両軌道に境界線を反映（切り替え後も正しく動作するよう両方更新）
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
        self._update_awsim_command_speed(msg)
        # section = int(msg.data[3])

        if self._current_laps is None:
            self._current_laps = 1 if laps == 0 else laps

        if laps > self._current_laps:
            self.get_logger().info(f'\033[32mLap {self._current_laps} completed! Lap time: {self._last_lap_time} s\033[0m')
            self._lap_times[self._current_laps] = self._last_lap_time
            self._current_laps = laps

        self._last_lap_time = lap_time

    def _awsim_admin_status_callback(self, msg):
        self._update_awsim_command_speed(msg)

    def _update_awsim_command_speed(self, msg):
        if len(msg.data) >= 2:
            self._awsim_command_speed = float(msg.data[1])

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

    def _switch_mpc(self, new_mpc, new_car, new_ref_path):
        if self._mpc != new_mpc:
            old_control = self._mpc.current_control
            new_N = new_mpc.N
            old_N = self._mpc.N
            nu = 2
            
            new_control = np.zeros(nu * new_N)
            if old_control is not None and len(old_control) > 0:
                steps_to_copy = min(new_N, old_N)
                new_control[:steps_to_copy * nu] = old_control[:steps_to_copy * nu]
                if new_N > old_N:
                    last_v = old_control[-2]
                    last_delta = old_control[-1]
                    for i in range(old_N, new_N):
                        new_control[i*nu : (i+1)*nu] = [last_v, last_delta]
            
            new_mpc.current_control = new_control
            new_mpc.previous_steering = self._mpc.previous_steering
            new_mpc.infeasibility_counter = self._mpc.infeasibility_counter
            
            self._mpc = new_mpc
            self._car = new_car
            self._reference_path = new_ref_path

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
        pose = self.get_ego_pose()
        v = self._odom.twist.twist.linear.x
        
        # --- Dynamic Trajectory Switching (Race ↔ Center) ---
        opponent_ahead_detected = getattr(self, '_opponent_ahead_detected', False)
        
        # Estimate closest opponent distance in waypoint steps (signed)
        # temp_car は Race 軌道のウェイポイント参照専用。
        # update_states の代わりに _wp_xy キャッシュを使って直接 wp_id を取得し
        # BicycleModel の get_closest_waypoint 呼び出し（全点スキャン）を1回に抑える。
        wp_temp = self._carN_race.get_closest_waypoint(pose.x, pose.y)
        N_total_temp = self._reference_pathN_race.n_waypoints

        closest_opp_ahead = 99999
        closest_opp_behind = 99999
        closest_opp_ahead_id = None
        closest_opp_behind_id = None
        if self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker'):
            for vid in self._v2x_tracker.active_vehicle_ids():
                buf = self._v2x_tracker._samples.get(vid)
                if buf:
                    _, opp_x, opp_y = buf[-1]
                    # Euclidean 距離で事前フィルタ（過度に遠い車は全点スキャンをスキップ）
                    if math.hypot(opp_x - pose.x, opp_y - pose.y) > 35.0:
                        continue
                    opp_wp_id = self._carN_race.get_closest_waypoint(opp_x, opp_y)
                    # 符号付きインデックス差（-N_total/2 〜 +N_total/2）
                    wp_diff = (opp_wp_id - wp_temp + N_total_temp // 2) % N_total_temp - N_total_temp // 2
                    
                    if wp_diff >= 0:
                        if wp_diff < closest_opp_ahead:
                            closest_opp_ahead = wp_diff
                            closest_opp_ahead_id = vid
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
        forced_overtake_pending = self._forced_overtake_vehicle_id is not None

        if recovery_active:
            self._trajectory_clear_since = None
            self._trajectory_switch_reason = "stuck_recovery_hold"
        elif opponent_ahead_detected:
            opponent_is_clear = (
                closest_opp_ahead > self._trajectory_exit_center_wps
                and closest_opp_behind > self._trajectory_behind_release_wps
            )
            if opponent_is_clear and not forced_overtake_pending:
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
        else:
            self._trajectory_clear_since = None
            if (
                closest_opp_ahead < self._trajectory_enter_center_wps
                and hold_elapsed
            ):
                opponent_ahead_detected = True
                self._trajectory_switch_reason = "opponent_ahead"

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

        self._opponent_ahead_detected = opponent_ahead_detected
        self._print_obstacle_detected = opponent_ahead_detected

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
            self._trajectory_last_switch_time = now_sec
            label = "Race→Center" if opponent_ahead_detected else "Center→Race"
            center_wp_at_switch = self._carN_center.get_closest_waypoint(
                pose.x, pose.y)
            if opponent_ahead_detected and closest_opp_ahead_id is not None:
                self._trajectory_vehicle_id = closest_opp_ahead_id
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
            self._clear_mpc_pred_markers()
            if not opponent_ahead_detected:
                self._trajectory_vehicle_id = None

        # --- 重要: self._car / self._mpc / self._reference_path を _carN / _mpcN / _reference_pathN に同期 ---
        # self._carN / _mpcN / _reference_pathN は切り替えブロックで更新されているが、
        # 実際に制御計算に使われるのは self._car / _mpc / _reference_path なので必ず反映する。
        self._car = self._carN
        self._mpc = self._mpcN
        self._reference_path = self._reference_pathN

        # Solve from the state at which this cycle's command reaches the actuator.
        predicted_pose = self._predict_pose_after_steering_delay(
            pose, v, float(now.nanoseconds) / 1e9)
        self._car.update_states(
            predicted_pose.x, predicted_pose.y, predicted_pose.theta)
        wp = self._car.wp_id  # update_states 内で get_closest_waypoint が実行済み


        # --- Overtaking Lane Selection Logic ---
        if not hasattr(self, '_target_lane_idx'):
            self._target_lane_idx = None

        N_total = self._reference_path.n_waypoints
        opponent_ahead = None
        opponent_offset = 0.0
        opponent_center = 0.0
        opponent_distance = 99999.0 #前方車両との距離
        opponent_v_lead = 0.0 #前方車両の速度
        opponent_vehicle_id = None
        opponent_velocity_valid = False
        min_wp_diff = 99999

        # --- Parallel Running Detection (並走検出) ---
        parallel_lat_dist = 99999.0   # 最も近い並走車との横距離
        parallel_lon_dist = 99999.0   # その車との縦距離 (Euclidean)
        if not hasattr(self, '_parallel_start_time'):
            self._parallel_start_time = None

        if self.USE_OBSTACLE_AVOIDANCE and hasattr(self, '_v2x_tracker'):
            for vid in self._v2x_tracker.active_vehicle_ids():
                buf = self._v2x_tracker._samples.get(vid)
                if buf:
                    _, opp_x, opp_y = buf[-1]
                    opp_wp_id = self._car.get_closest_waypoint(opp_x, opp_y)
                    
                    wp_diff = (opp_wp_id - wp) % N_total
                    rel_forward = (
                        (opp_x - pose.x) * math.cos(pose.theta)
                        + (opp_y - pose.y) * math.sin(pose.theta)
                    )
                    same_wp_ahead = (wp_diff == 0 and rel_forward > 0.0)
                    if same_wp_ahead or 0 < wp_diff < 40:
                        if wp_diff < min_wp_diff:
                            min_wp_diff = wp_diff
                            opponent_ahead = opp_wp_id
                            opponent_vehicle_id = vid
                            
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
                            opponent_velocity_valid = (
                                self._v2x_tracker.has_velocity_estimate(vid))

        # 常に追い越しを許可する
        is_overtake_zone = True

        # Calculate candidate target lane based on opponent position
        new_target_lane_idx = self._target_lane_idx  # Keep current active lane by default
        left_is_free = False
        right_is_free = False

        if opponent_ahead is not None:
            if is_overtake_zone:
                opp_wp = self._reference_path.get_waypoint(opponent_ahead)
                # 追い越しに必要な最小の横方向スペース [m] (自車幅 + 相手車幅半分 + マージン)
                MIN_OVERTAKE_SPACE = 2.1  
                
                # 相手の位置におけるコース左右境界までの残りスペース
                space_left = opp_wp.ub - opponent_offset
                space_right = opponent_offset - opp_wp.lb
                
                left_is_free = (space_left >= MIN_OVERTAKE_SPACE)
                right_is_free = (space_right >= MIN_OVERTAKE_SPACE)
                
                if left_is_free and right_is_free:
                    # 両方空いている場合は、より広いスペースがある方を選択
                    if space_left > space_right:
                        new_target_lane_idx = 2  # 左から追い越し (L2)
                    else:
                        new_target_lane_idx = 0  # 右から追い越し (L0)
                elif left_is_free:
                    new_target_lane_idx = 2      # 左からのみ追い越し可能 (L2)
                elif right_is_free:
                    new_target_lane_idx = 0      # 右からのみ追い越し可能 (L0)
                else:
                    # どちらもスペースが狭すぎる場合は無理せず中央(L1)で追従
                    new_target_lane_idx = 1
            else:
                # 追い越し不可エリア -> 中央車線 (L1) を走行して追従
                new_target_lane_idx = 1
        elif not opponent_ahead_detected:
            # 追い越しモード自体が終了した場合はターゲット車線をクリア
            new_target_lane_idx = None

        current_time_sec = float(now.nanoseconds) / 1e9

        lead_is_stationary = (
            opponent_vehicle_id is not None
            and opponent_velocity_valid
            and opponent_v_lead < self._stopped_lead_speed_threshold
        )
        if lead_is_stationary:
            if self._forced_overtake_vehicle_id != opponent_vehicle_id:
                action = (
                    "forcing immediate low-speed overtake"
                    if left_is_free or right_is_free
                    else "no safe passing side; keeping stop/reverse fallback armed"
                )
                self.get_logger().warn(
                    "[StoppedVehicle] "
                    f"vehicle_id={opponent_vehicle_id} "
                    f"speed={opponent_v_lead:.2f}m/s; {action}.",
                    throttle_duration_sec=1.0,
                )
            self._forced_overtake_vehicle_id = opponent_vehicle_id

        if (
            self._forced_overtake_vehicle_id is not None
            and opponent_vehicle_id != self._forced_overtake_vehicle_id
        ):
            forced_id = self._forced_overtake_vehicle_id
            active_ids = self._v2x_tracker.active_vehicle_ids()
            forced_buf = self._v2x_tracker._samples.get(forced_id)
            forced_vehicle_passed = forced_id not in active_ids
            if forced_buf:
                _, forced_x, forced_y = forced_buf[-1]
                forced_lon = (
                    (forced_x - pose.x) * math.cos(pose.theta)
                    + (forced_y - pose.y) * math.sin(pose.theta)
                )
                forced_vehicle_passed = forced_lon < -1.0
            if forced_vehicle_passed:
                self.get_logger().info(
                    f"[StoppedVehicle] vehicle_id={forced_id} "
                    "forced overtake completed."
                )
                self._forced_overtake_vehicle_id = None

        prev_lane_idx = self._target_lane_idx

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

        if new_target_lane_idx != prev_lane_idx:
            can_change_lane = True
            
            # Check elapsed time since last lane change
            if self._last_lane_change_time is not None:
                elapsed = current_time_sec - self._last_lane_change_time
                if elapsed < 2.5:
                    can_change_lane = False  # Lock lane change

            # Curve lock override: force lane constraint immediately if in curve
            if is_curve_locked:
                can_change_lane = True
            # A stationary lead must not remain trapped behind the normal
            # lane-change cooldown when a passing side is available.
            if lead_is_stationary and new_target_lane_idx in (0, 2):
                can_change_lane = True

            if can_change_lane:
                self._target_lane_idx = new_target_lane_idx
                # Only update the lane change timestamp if it was an actual lane change request
                # and not just forcing center lane from None
                if not (prev_lane_idx is None and new_target_lane_idx == 1):
                    self._last_lane_change_time = current_time_sec
                if new_target_lane_idx is not None:
                    target_str = "right (L0)" if new_target_lane_idx == 0 else "left (L2)" if new_target_lane_idx == 2 else "center (L1)"
                    self.get_logger().info(
                        f"[LaneChange] Switching to lane {target_str} (lock for 5s)",
                        throttle_duration_sec=1.0
                    )
                else:
                    self.get_logger().info(
                        "[LaneChange] Switching back to free driving (lock for 5s)",
                        throttle_duration_sec=1.0
                    )
            else:
                # Keep the previous lane index to avoid chattering
                pass
        else:
            # No lane change request, keep active candidate
            self._target_lane_idx = new_target_lane_idx

        tracked_stopped_lead = (
            lead_is_stationary
            and opponent_vehicle_id is not None
            and opponent_vehicle_id == self._forced_overtake_vehicle_id
        )
        forced_overtake_active, close_overtake_blocked = (
            evaluate_stopped_lead_overtake(
                tracked_stopped_lead=tracked_stopped_lead,
                distance=opponent_distance,
                left_is_free=left_is_free,
                right_is_free=right_is_free,
                target_lane_idx=self._target_lane_idx,
                infeasibility_counter=self._mpc.infeasibility_counter,
                reverse_distance=self._close_obstacle_reverse_distance,
                infeasible_cycles=self._close_obstacle_infeasible_cycles,
            )
        )
        self._close_obstacle_reverse_requested = close_overtake_blocked

        # Apply target lane
        # --- 制約切り替え安定化ウィンドウ ---
        # target_lane_idx が変わった瞬間に制約を即時切り替えると OSQP が infeasible になりやすい。
        # フラグ変更後 CONSTRAINT_TRANSITION_SEC 秒間は full-width (is_overtaking=False / target=None) で走り、
        # その後に絞り込んだ車線制約を適用する。
        CONSTRAINT_TRANSITION_SEC = 0.6  # [s] 安定化ウィンドウ幅
        if not hasattr(self, '_constraint_transition_until'):
            self._constraint_transition_until = 0.0
        if not hasattr(self, '_prev_applied_lane_idx'):
            self._prev_applied_lane_idx = self._target_lane_idx

        if self._target_lane_idx != self._prev_applied_lane_idx:
            # 車線が変わった → 安定化ウィンドウを開始
            self._constraint_transition_until = current_time_sec + CONSTRAINT_TRANSITION_SEC
            self._prev_applied_lane_idx = self._target_lane_idx

        in_transition = (current_time_sec < self._constraint_transition_until)

        if in_transition:
            # 安定化中: 制約はフル幅（通常走行扱い）
            self._reference_path.target_lane_idx  = None
            self._reference_pathN.target_lane_idx = None
            self._reference_path.is_overtaking    = False
            self._reference_pathN.is_overtaking   = False
        else:
            # 安定化終了: 本来の車線制約を適用
            self._reference_path.target_lane_idx  = self._target_lane_idx
            self._reference_pathN.target_lane_idx = self._target_lane_idx
            is_overtaking = (self._target_lane_idx is not None)
            self._reference_path.is_overtaking    = is_overtaking
            self._reference_pathN.is_overtaking   = is_overtaking

        is_overtaking = self._reference_path.is_overtaking

        # 追従・追い越し、または対象のWaypoint区間（カーブなど慎重さが求められる箇所）はwp_id_offsetを1にする
        is_in_cautious_zone = (210 <= wp <= 243) or (261 <= wp <= 286)
        active_offset = 0 if (is_overtaking or is_in_cautious_zone) else self._default_wp_id_offset
        self._mpc.update_wp_id_offset(active_offset)
        
        # MPCの実行
        with self._stats.time_block("control"):
            u, max_delta = self._mpc.get_control()

        # ref_vel_configuratorがある場合はその値を基準に、なければv_maxを基準にする
        if self._ref_vel_configulator is not None:
            ref_vel_mps = self._ref_vel_configulator.get_ref_vel(self._mpc.model.wp_id)
            ref_vel_kmph = min(
                kmh_to_m_per_sec(ref_vel_mps),
                self._mpc_cfg.v_max)
        else:
            ref_vel_kmph = self._mpc_cfg.v_max

        emergency_brake_active = False
        if self.USE_OBSTACLE_AVOIDANCE:
            # --- ACC spacing control (車間距離維持制御) ---
            # 前方車両がいて、かつ自車の走行ライン上（横方向の差が 1.2m 未満）に他車が位置する場合に
            # 追従状態とみなして車間制御（5m〜10m）を有効化する。
            # 横方向の差が 1.2m 以上の場合は、別車線での追い越し中とみなして加速を許可する。
            e_y = self._car.spatial_state.e_y
            lat_dist = abs(opponent_offset - e_y)

            # --- Standard Follow (Same Lane) ---
            if (
                opponent_ahead is not None
                and lat_dist < 1.2
                and not forced_overtake_active
            ):
                if opponent_distance < 15.0:
                    d_target = 8.0 # 目標車間距離 (5m 〜 10m の中央値 7.5m)
                    K_p = 1.2       # 比例ゲイン
                    v_ref_acc = opponent_v_lead + K_p * (opponent_distance - d_target)
                    v_ref_acc = max(0.0, v_ref_acc)  # 後退は禁止のため下限は0

                    ref_vel_kmph = min(ref_vel_kmph, v_ref_acc)

                    #if self._loop % int(self._mpc_cfg.control_rate) == 0:
                        #self.get_logger().info(
                        #    f"[ACC] Distance to opp: {opponent_distance:.2f}m, Opp speed: {opponent_v_lead:.2f}m/s. "
                        #    f"Target speed limited to {ref_vel_kmph:.2f}m/s to maintain distance.",
                        #    throttle_duration_sec=1.0
                        #)

            # --- Emergency Proximity Brake (waypoint-independent) ---
            # opponent_ahead (waypoint差ベースの検出) に依存せず、全V2X車両を直接スキャンする。
            # wp_diff=0 の場合など waypoint 検出をすり抜けても必ずブレーキがかかる。
            # 自車ヨー角を用いて「前方向」かどうかを判定する。
            EMERGENCY_BRAKE_DIST  = 6.0  # [m] この距離以内で前方に車がいたら緊急ブレーキ
            EMERGENCY_BRAKE_ANGLE = 60.0 # [deg] 前方判定の角度半幅（進行方向±この角度以内）
            EMERGENCY_BRAKE_LATERAL_DIST = 1.2 # [m] 車線境界付近を含む横接近判定
            if hasattr(self, '_v2x_tracker'):
                ego_yaw = pose.theta  # 自車ヨー角 [rad]
                cos_thresh = math.cos(math.radians(EMERGENCY_BRAKE_ANGLE))
                for vid in self._v2x_tracker.active_vehicle_ids():
                    buf = self._v2x_tracker._samples.get(vid)
                    if buf:
                        _, opp_x, opp_y = buf[-1]
                        dx = opp_x - pose.x
                        dy = opp_y - pose.y
                        dist = math.hypot(dx, dy)
                        if dist < EMERGENCY_BRAKE_DIST and dist > 0.01:
                            # 自車前方向ベクトルとの内積で「前方」を判定
                            fwd_dot = (dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)) / dist
                            if fwd_dot > cos_thresh:
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
                                laterally_close = (
                                    lateral_dist < EMERGENCY_BRAKE_LATERAL_DIST)

                                if not (same_lane or laterally_close):
                                    self.get_logger().info(
                                        "[EmergencyBrakeSkip] "
                                        f"vehicle_id={vid} ego_lane={ego_lane_idx} "
                                        f"opp_lane={opp_lane_idx} "
                                        f"lateral={lateral_dist:.2f}m",
                                        throttle_duration_sec=1.0,
                                    )
                                    continue

                                opp_vx, opp_vy = self._v2x_tracker.velocity(vid)
                                opp_spd = math.hypot(opp_vx, opp_vy)
                                if (
                                    forced_overtake_active
                                    and vid == self._forced_overtake_vehicle_id
                                ):
                                    # The accepted outer-lane MPC path is the
                                    # safety controller for this stopped target.
                                    continue
                                d_target_emg = 5.5
                                K_p_emg = 1.5
                                v_ref_emg = opp_spd + K_p_emg * (dist - d_target_emg)
                                stopped_vehicle_too_close = (
                                    opp_spd < self._stopped_lead_speed_threshold
                                    and dist <= self._close_obstacle_reverse_distance
                                )
                                min_emergency_speed = (
                                    0.0 if stopped_vehicle_too_close else 0.5)
                                v_ref_emg = max(min_emergency_speed, v_ref_emg)
                                ref_vel_kmph = min(ref_vel_kmph, v_ref_emg)
                                emergency_brake_active = True
                                self.get_logger().warn(
                                    f"[EmergencyBrake] vehicle_id={vid} "
                                    f"forward obstacle at {dist:.2f}m "
                                    f"(dot={fwd_dot:.2f}, lateral={lateral_dist:.2f}m, "
                                    f"ego_lane={ego_lane_idx}, opp_lane={opp_lane_idx}). "
                                    f"Speed → {ref_vel_kmph:.2f}m/s",
                                    throttle_duration_sec=0.5
                                )

            # --- Post-Overtake Cooldown: 追い越し後クールダウン中の後方車両監視 ---
            # Center→Race に切り替わった直後は、後方の近接車との衝突リスクが高い。
            # Race軌道がコーナーインを攻めて後方の相手と交差しないよう、
            # クールダウン期間中は後方の近接車との距離に応じて速度を抑制する。
            RACE_RETURN_COOLDOWN_SEC = 3.0  # [s] 追い越し完了後に速度抑制する時間
            if not hasattr(self, '_race_return_time'):
                self._race_return_time = None

            # Center→Race 切り替えを検出してクールダウンタイマーを開始
            if trajectory_switched and not opponent_ahead_detected:
                self._race_return_time = current_time_sec
                self.get_logger().info("[PostOvertake] Cooldown started after returning to Race trajectory.")

            in_post_overtake_cooldown = (
                self._race_return_time is not None and
                current_time_sec - self._race_return_time < RACE_RETURN_COOLDOWN_SEC
            )

            if in_post_overtake_cooldown and hasattr(self, '_v2x_tracker'):
                for vid in self._v2x_tracker.active_vehicle_ids():
                    buf = self._v2x_tracker._samples.get(vid)
                    if buf:
                        _, opp_x, opp_y = buf[-1]
                        behind_dist = math.hypot(opp_x - pose.x, opp_y - pose.y)
                        if behind_dist < 12.0:  # 後方12m以内に相手がいる場合のみ速度制限
                            opp_vx, opp_vy = self._v2x_tracker.velocity(vid)
                            opp_speed = math.hypot(opp_vx, opp_vy)
                            # 後方車両との距離が縮まらないよう、相手速度を上限とする
                            d_behind_target = 8.0  # 目標後方間隔
                            K_p_behind = 0.8
                            v_ref_behind = opp_speed + K_p_behind * (behind_dist - d_behind_target)
                            v_ref_behind = max(2.0, v_ref_behind)
                            ref_vel_kmph = min(ref_vel_kmph, v_ref_behind)
                            if self._loop % int(self._mpc_cfg.control_rate) == 0:
                                self.get_logger().info(
                                    f"[PostOvertake] Behind opp {behind_dist:.1f}m, "
                                    f"limiting speed to {ref_vel_kmph:.2f}m/s",
                                    throttle_duration_sec=1.0
                                )

            # --- Parallel Running Safety Control (並走接近制御) ---
            # 並走（横距離が小さく縦距離も小さい）の場合、速度を落として衝突を回避する
            # 並走が続く場合は追い越しを中断して中央車線に戻す
            LAT_WARN_THRESH  = 2.0   # [m] 警戒ゾーン開始 (並走接近を検出)
            LAT_CRIT_THRESH  = 1.4   # [m] 臨界ゾーン (強制減速)
            LON_PARALLEL_MAX = 4.5   # [m] この縦距離以内を「並走」と判定
            PARALLEL_ABORT_SEC = 4.0 # [s] 並走がこの時間以上続いたら追い越し中断

            parallel_vehicle_id = None
            if hasattr(self, '_v2x_tracker'):
                for vid in self._v2x_tracker.active_vehicle_ids():
                    buf = self._v2x_tracker._samples.get(vid)
                    if buf:
                        _, opp_x, opp_y = buf[-1]
                        dx = opp_x - pose.x
                        dy = opp_y - pose.y
                        lon_d = abs(
                            dx * math.cos(pose.theta)
                            + dy * math.sin(pose.theta)
                        )
                        # 進行方向の前後距離が近い場合だけ横距離を計算
                        if lon_d < LON_PARALLEL_MAX:
                            opp_wp_id = self._car.get_closest_waypoint(opp_x, opp_y)
                            opp_wp = self._reference_path.get_waypoint(opp_wp_id)
                            if opp_wp.normal_angle is not None:
                                nx = -math.cos(opp_wp.normal_angle)
                                ny = -math.sin(opp_wp.normal_angle)
                            else:
                                a = opp_wp.psi + math.pi / 2.0
                                nx = math.cos(a)
                                ny = math.sin(a)
                            lat_d = abs(dx * nx + dy * ny)
                            if lat_d < parallel_lat_dist:
                                parallel_lat_dist = lat_d
                                parallel_lon_dist = lon_d
                                parallel_vehicle_id = vid

            is_parallel = (parallel_lat_dist < LAT_WARN_THRESH and parallel_lon_dist < LON_PARALLEL_MAX)
            forced_target_is_parallel = (
                forced_overtake_active
                and parallel_vehicle_id == self._forced_overtake_vehicle_id
            )

            if is_parallel and not forced_target_is_parallel:
                if self._parallel_start_time is None:
                    self._parallel_start_time = current_time_sec
                parallel_duration = current_time_sec - self._parallel_start_time

                # 臨界ゾーン: 強制減速
                if parallel_lat_dist < LAT_CRIT_THRESH:
                    # 横距離が近いほど強く減速（目標速度を直接スケール）
                    ratio = max(0.0, (parallel_lat_dist - 0.5) / (LAT_CRIT_THRESH - 0.5))
                    v_ref_parallel = ref_vel_kmph * (0.4 + 0.6 * ratio)  # 最大60%減速
                    ref_vel_kmph = min(ref_vel_kmph, v_ref_parallel)
                    self.get_logger().warn(
                        f"[ParallelSafety] CRITICAL: vehicle_id={parallel_vehicle_id} "
                        f"lat={parallel_lat_dist:.2f}m, "
                        f"lon={parallel_lon_dist:.2f}m, "
                        f"speed limited to {ref_vel_kmph:.2f}m/s",
                        throttle_duration_sec=1.0
                    )
                # 警戒ゾーン: 緩やかに減速
                elif parallel_lat_dist < LAT_WARN_THRESH:
                    ratio = (parallel_lat_dist - LAT_CRIT_THRESH) / (LAT_WARN_THRESH - LAT_CRIT_THRESH)
                    v_ref_parallel = ref_vel_kmph * (0.7 + 0.3 * ratio)  # 最大30%減速
                    ref_vel_kmph = min(ref_vel_kmph, v_ref_parallel)
                    self.get_logger().info(
                        f"[ParallelSafety] WARNING: vehicle_id={parallel_vehicle_id} "
                        f"lat={parallel_lat_dist:.2f}m, "
                        f"lon={parallel_lon_dist:.2f}m, "
                        f"speed limited to {ref_vel_kmph:.2f}m/s (duration={parallel_duration:.1f}s)",
                        throttle_duration_sec=1.0
                    )

                # 並走中断: 長時間並走が続いたら中央車線に戻して相手に先行させる
                if parallel_duration > PARALLEL_ABORT_SEC and self._target_lane_idx is not None:
                    '''
                    self.get_logger().warn(
                        f"[ParallelSafety] ABORT: vehicle_id={parallel_vehicle_id} "
                        f"lat={parallel_lat_dist:.2f}m, "
                        f"lon={parallel_lon_dist:.2f}m, "
                        f"after {parallel_duration:.1f}s parallel running. "
                        f"Returning to center lane.",
                        throttle_duration_sec=1.0
                    )
                    '''
                    self._target_lane_idx = 1  # 中央車線へ退避
                    self._last_lane_change_time = current_time_sec
                    self._parallel_start_time = None
            else:
                self._parallel_start_time = None

            if forced_overtake_active:
                ref_vel_kmph = min(
                    ref_vel_kmph, self._forced_overtake_speed)

            if close_overtake_blocked:
                ref_vel_kmph = 0.0
                self.get_logger().warn(
                    "[StoppedVehicle] "
                    f"vehicle_id={self._forced_overtake_vehicle_id} "
                    f"distance={opponent_distance:.2f}m, "
                    f"mpc_infeasible={self._mpc.infeasibility_counter}; "
                    "holding speed at zero and waiting for reverse recovery.",
                    throttle_duration_sec=1.0,
                )

        self._mpc.update_v_max(ref_vel_kmph)
        v_ref: List[float] = [ref_vel_kmph] * len(self._reference_path.waypoints)
        self._reference_path.set_v_ref(v_ref)

        if emergency_brake_active or self._close_obstacle_reverse_requested:
            u[0] = min(u[0], ref_vel_kmph)

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
            # continue

        recovering_from_stuck = self._apply_stuck_recovery(now, u, v)

        acc = 0.
        bug_acc_enabled = False


        #boostモードがONのとき
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
        elif emergency_brake_active or self._close_obstacle_reverse_requested:
            bug_acc_enabled = False
            acc = np.clip(
                self.KP * (u[0] - abs(v)),
                self._mpc_cfg.a_min,
                self._mpc_cfg.a_max,
            )
            self._pred_marker_color = RED
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
