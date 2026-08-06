#include "simple_pure_pursuit/simple_pure_pursuit.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>

#include <algorithm>

namespace simple_pure_pursuit
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;

SimplePurePursuit::SimplePurePursuit()
: Node("simple_pure_pursuit"),
  // initialize parameters
  wheel_base_(declare_parameter<float>("wheel_base", 2.14)),
  lookahead_gain_(declare_parameter<float>("lookahead_gain", 1.0)),
  lookahead_min_distance_(declare_parameter<float>("lookahead_min_distance", 1.0)),
  speed_proportional_gain_(declare_parameter<float>("speed_proportional_gain", 1.0)),
  use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),
  external_target_vel_(declare_parameter<float>("external_target_vel", 0.0)),
  steering_tire_angle_gain_(declare_parameter<float>("steering_tire_angle_gain", 1.0)),
  gnss_timeout_sec_(declare_parameter<double>("gnss_timeout_sec", 0.5)),
  max_gnss_position_covariance_(
    declare_parameter<double>("max_gnss_position_covariance", 1.0))
{
  pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);
  pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);
  pub_lookahead_point_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1);

  const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
  const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().reliable();
  sub_kinematics_ = create_subscription<Odometry>(
    "input/kinematics", bv_qos, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
  sub_gnss_pose_ = create_subscription<PoseWithCovarianceStamped>(
    "input/gnss_pose", reliable_qos,
    [this](const PoseWithCovarianceStamped::SharedPtr msg) { gnss_pose_ = msg; });
  sub_trajectory_ = create_subscription<Trajectory>(
    "input/trajectory", bv_qos, [this](const Trajectory::SharedPtr msg) { trajectory_ = msg; });

  using namespace std::literals::chrono_literals;
  timer_ = create_wall_timer(10ms, std::bind(&SimplePurePursuit::onTimer, this));
}

AckermannControlCommand zeroAckermannControlCommand(rclcpp::Time stamp)
{
  AckermannControlCommand cmd;
  cmd.stamp = stamp;
  cmd.longitudinal.stamp = stamp;
  cmd.longitudinal.speed = 0.0;
  cmd.longitudinal.acceleration = 0.0;
  cmd.lateral.stamp = stamp;
  cmd.lateral.steering_tire_angle = 0.0;
  return cmd;
}

void SimplePurePursuit::onTimer()
{
  // check data
  if (!subscribeMessageAvailable()) {
    return;
  }

  const auto & vehicle_pose = gnss_pose_->pose.pose;
  size_t closet_traj_point_idx = findNearestIndex(trajectory_->points, vehicle_pose.position);

  // publish zero command
  AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());

  // get closest trajectory point from current position
  TrajectoryPoint closet_traj_point = trajectory_->points.at(closet_traj_point_idx);

  // calc longitudinal speed and acceleration
  double target_longitudinal_vel = closet_traj_point.longitudinal_velocity_mps;
  if (use_external_target_vel_) {
    target_longitudinal_vel = std::min(target_longitudinal_vel, static_cast<double>(external_target_vel_));
  }
  double current_longitudinal_vel = odometry_->twist.twist.linear.x;

  cmd.longitudinal.speed = target_longitudinal_vel;
  // Keep the command below AWSIM's effective acceleration limit and avoid the
  // excessive-acceleration penalty (triggered above +3.0 m/s^2).
  constexpr double max_acceleration = 1.3;
  constexpr double max_deceleration = 3.0;
  cmd.longitudinal.acceleration = std::clamp(
    speed_proportional_gain_ * (target_longitudinal_vel - current_longitudinal_vel),
    -max_deceleration, max_acceleration);

  // calc lateral control
  //// calc lookahead distance
  double lookahead_distance = lookahead_gain_ * target_longitudinal_vel + lookahead_min_distance_;
  //// calc center coordinate of rear wheel
  // GNSS supplies the map position, but its heading is derived from successive
  // positions and is unreliable while stopped. Use the initialized EKF heading
  // from odometry so Pure Pursuit has a valid map-frame yaw from startup.
  const double vehicle_yaw = tf2::getYaw(odometry_->pose.pose.orientation);
  double rear_x = vehicle_pose.position.x - wheel_base_ / 2.0 * std::cos(vehicle_yaw);
  double rear_y = vehicle_pose.position.y - wheel_base_ / 2.0 * std::sin(vehicle_yaw);
  //// search lookahead point (closed loop)
  size_t lookahead_point_idx = closet_traj_point_idx;
  for (size_t i = 0; i < trajectory_->points.size(); ++i) {
    size_t idx = (closet_traj_point_idx + i) % trajectory_->points.size();
    const auto & point = trajectory_->points.at(idx);
    if (std::hypot(point.pose.position.x - rear_x, point.pose.position.y - rear_y) >= lookahead_distance) {
      lookahead_point_idx = idx;
      break;
    }
  }
  double lookahead_point_x = trajectory_->points.at(lookahead_point_idx).pose.position.x;
  double lookahead_point_y = trajectory_->points.at(lookahead_point_idx).pose.position.y;

  geometry_msgs::msg::PointStamped lookahead_point_msg;
  lookahead_point_msg.header.stamp = get_clock()->now();
  lookahead_point_msg.header.frame_id = "map";
  lookahead_point_msg.point.x = lookahead_point_x;
  lookahead_point_msg.point.y = lookahead_point_y;
  lookahead_point_msg.point.z = closet_traj_point.pose.position.z;
  pub_lookahead_point_->publish(lookahead_point_msg);

  // calc steering angle for lateral control
  double alpha = std::atan2(lookahead_point_y - rear_y, lookahead_point_x - rear_x) -
                 vehicle_yaw;
  cmd.lateral.steering_tire_angle =
    steering_tire_angle_gain_ * std::atan2(2.0 * wheel_base_ * std::sin(alpha), lookahead_distance);

  pub_cmd_->publish(cmd);
  cmd.lateral.steering_tire_angle /=  steering_tire_angle_gain_;
  pub_raw_cmd_->publish(cmd);
}

bool SimplePurePursuit::subscribeMessageAvailable()
{
  if (!odometry_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "odometry is not available");
    return false;
  }
  if (!gnss_pose_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "GNSS pose is not available");
    return false;
  }
  const double gnss_age = (get_clock()->now() - gnss_pose_->header.stamp).seconds();
  if (gnss_age > gnss_timeout_sec_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000 /*ms*/, "GNSS pose is stale (age: %.3f s)", gnss_age);
    return false;
  }
  const auto & covariance = gnss_pose_->pose.covariance;
  if (covariance[0] > max_gnss_position_covariance_ ||
      covariance[7] > max_gnss_position_covariance_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000 /*ms*/,
      "GNSS position covariance is too large (x: %.3f, y: %.3f, limit: %.3f)",
      covariance[0], covariance[7], max_gnss_position_covariance_);
    return false;
  }
  if (!trajectory_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "trajectory is not available");
    return false;
  }
  if (trajectory_->points.empty()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/,  "trajectory points is empty");
      return false;
    }
  return true;
}
}  // namespace simple_pure_pursuit

int main(int argc, char const * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_pure_pursuit::SimplePurePursuit>());
  rclcpp::shutdown();
  return 0;
}
