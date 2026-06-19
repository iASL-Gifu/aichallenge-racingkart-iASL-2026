#include "simple_pure_pursuit/simple_pure_pursuit.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <cmath>

namespace simple_pure_pursuit
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;

SimplePurePursuit::SimplePurePursuit()
: Node("simple_pure_pursuit"),
  // initialize parameters
  wheel_base_(declare_parameter<float>("wheel_base", 2.14)),
  lookahead_distance_1_(declare_parameter<std::vector<double>>(
      "lookahead_distance_1", std::vector<double>{4.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45})),
  lookahead_distance_2_(declare_parameter<std::vector<double>>(
      "lookahead_distance_2", std::vector<double>{1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45})),
  steering_angle_to_lookahead_ratio_(declare_parameter<double>("steering_angle_to_lookahead_ratio", 2.0)),
  speed_proportional_gain_1_(declare_parameter<float>("speed_proportional_gain_1", 1.0)),
  speed_proportional_gain_2_(declare_parameter<float>("speed_proportional_gain_2", 1.0)),
  high_acceleration_speed_threshold_(declare_parameter<float>("high_acceleration_speed_threshold", 5.0)),
  adaptive_gain_steering_threshold_(declare_parameter<double>("adaptive_gain_steering_threshold", 0.15)),
  adaptive_gain_lookahead_threshold_(declare_parameter<double>("adaptive_gain_lookahead_threshold", 5.0)),
  adaptive_gain_reduction_ratio_(declare_parameter<double>("adaptive_gain_reduction_ratio", 0.5)),
  use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),
  external_target_vel_(declare_parameter<float>("external_target_vel", 0.0)),
  steering_tire_angle_gain_(declare_parameter<std::vector<double>>(
      "steering_tire_angle_gain", std::vector<double>{1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5}))
{
  pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);
  pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);
  pub_lookahead_point_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1);

  const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
  sub_kinematics_ = create_subscription<Odometry>(
    "input/kinematics", bv_qos, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
    
  // 【修正箇所】余計な.clear()を削除し、スマートポインタの代入のみに変更
  sub_trajectory_ = create_subscription<Trajectory>(
    "input/trajectory", bv_qos, [this](const Trajectory::SharedPtr msg) {
      trajectory_ = msg; 
    });
    
  sub_status_ = create_subscription<std_msgs::msg::Float32MultiArray>(
    "/awsim/status", rclcpp::QoS{1}.best_effort(),
    std::bind(&SimplePurePursuit::statusCallback, this, std::placeholders::_1));
  sub_steering_ = create_subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>(
    "/vehicle/status/steering_status", rclcpp::QoS{1},
    std::bind(&SimplePurePursuit::steeringCallback, this, std::placeholders::_1));
  pub_lookahead_ = create_publisher<std_msgs::msg::Float64>("output/lookahead_distance", 1);

  using namespace std::literals::chrono_literals;
  timer_ =
    rclcpp::create_timer(this, get_clock(), 10ms, std::bind(&SimplePurePursuit::onTimer, this));
}

void SimplePurePursuit::steeringCallback(const autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr msg)
{
  current_steering_angle_ = msg->steering_tire_angle;
}

void SimplePurePursuit::statusCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 4) return;
  const int lap = static_cast<int>(msg->data[1]);
  const int section = static_cast<int>(msg->data[3]);

  if (current_lap_ == -1) {
    cumulative_laps_time_ = 0.0;
  } else if (current_lap_ != lap) {
    double this_lap_time = msg->data[2] - cumulative_laps_time_;
    RCLCPP_INFO(get_logger(), "\033[32mLap %d completed! Lap time: %.3f s\033[0m", current_lap_, this_lap_time);
    cumulative_laps_time_ = msg->data[2];
  }

  current_lap_ = lap;
  current_section_ = section;

  // ラップ数に応じた設定
  switch (lap) {
    case 0:
      lookahead_distance = lookahead_distance_1_[0];
      break;
    case 1:
      lookahead_distance = lookahead_distance_1_[section];
      break;
    default:
      lookahead_distance = lookahead_distance_2_[section - 1];
      break;
  }

  steering_tire_angle_gain = steering_tire_angle_gain_[section];
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

  // 1. statusCallbackで設定されたベースのlookahead_distanceを取得
  double base_lookahead_distance = lookahead_distance;
  
  // 2. 現在のステアリング角度（絶対値）に基づいてlookahead_distanceを補正
  //    ステアリングが切れているほど、lookahead_distanceを短くする
  double lookahead_reduction = steering_angle_to_lookahead_ratio_ * std::abs(current_steering_angle_);
  double adjusted_lookahead_distance = base_lookahead_distance - lookahead_reduction;
  RCLCPP_INFO(this->get_logger(),
  "lap: %d, section: %d, base: %.2f, adjusted: %.2f, steering: %.3f",
  current_lap_,
  current_section_,
  base_lookahead_distance,
  adjusted_lookahead_distance,
  current_steering_angle_);

  // 3. 計算されたlookahead_distanceが最小値を下回らないようにクランプする
  adjusted_lookahead_distance = std::max(adjusted_lookahead_distance, 3.0);

  size_t closet_traj_point_idx =
    findNearestIndex(trajectory_->points, odometry_->pose.pose.position);

  // publish zero command
  AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());

  // get closest trajectory point from current position
  TrajectoryPoint closet_traj_point = trajectory_->points.at(closet_traj_point_idx);

  // calc longitudinal speed and acceleration
  double target_longitudinal_vel =
    use_external_target_vel_ ? external_target_vel_ : closet_traj_point.longitudinal_velocity_mps;
  double current_longitudinal_vel = odometry_->twist.twist.linear.x;

  double active_speed_proportional_gain = speed_proportional_gain_2_;
  if (current_lap_ == 0 && current_section_ == 0) {
    if (current_longitudinal_vel < high_acceleration_speed_threshold_) {
      active_speed_proportional_gain = speed_proportional_gain_1_;
    }
  }

  // ゼロ除算対策を追加した滑らかなゲイン減衰
  double steering_ratio = 0.0;
  if (adaptive_gain_steering_threshold_ > 1e-5) {
    steering_ratio = std::abs(current_steering_angle_) / adaptive_gain_steering_threshold_;
  }
  steering_ratio = std::clamp(steering_ratio, 0.0, 1.0);

  double lookahead_ratio = 0.0;
  if (adjusted_lookahead_distance > 1e-5) {
    lookahead_ratio = adaptive_gain_lookahead_threshold_ / adjusted_lookahead_distance;
  }
  lookahead_ratio = std::clamp(lookahead_ratio, 0.0, 1.0);

  double reduction = std::max(steering_ratio, lookahead_ratio);
  active_speed_proportional_gain *= (1.0 - reduction * (1.0 - adaptive_gain_reduction_ratio_));

  cmd.longitudinal.speed = target_longitudinal_vel;
  cmd.longitudinal.acceleration =
    active_speed_proportional_gain * (target_longitudinal_vel - current_longitudinal_vel);

  // calc lateral control
  // calc center coordinate of rear wheel
  double rear_x = odometry_->pose.pose.position.x -
                  wheel_base_ / 2.0 * std::cos(odometry_->pose.pose.orientation.z);
  double rear_y = odometry_->pose.pose.position.y -
                  wheel_base_ / 2.0 * std::sin(odometry_->pose.pose.orientation.z);
                  
  // search lookahead point (std::rotateを廃止した安全なループ探索)
  auto lookahead_point_itr = trajectory_->points.end();
  for (size_t i = closet_traj_point_idx; i < trajectory_->points.size(); ++i) {
    const auto & point = trajectory_->points.at(i);
    if (std::hypot(point.pose.position.x - rear_x, point.pose.position.y - rear_y) >= adjusted_lookahead_distance) {
      lookahead_point_itr = trajectory_->points.begin() + i;
      break;
    }
  }

  // もし条件を満たす点が見つからなかった場合（経路の終端付近など）の安全対策
  if (lookahead_point_itr == trajectory_->points.end()) {
    lookahead_point_itr = std::prev(trajectory_->points.end()); // 最後の点を使用する
  }
  
  double lookahead_point_x = lookahead_point_itr->pose.position.x;
  double lookahead_point_y = lookahead_point_itr->pose.position.y;

  geometry_msgs::msg::PointStamped lookahead_point_msg;
  lookahead_point_msg.header.stamp = get_clock()->now();
  lookahead_point_msg.header.frame_id = "map";
  lookahead_point_msg.point.x = lookahead_point_x;
  lookahead_point_msg.point.y = lookahead_point_y;
  lookahead_point_msg.point.z = closet_traj_point.pose.position.z;
  pub_lookahead_point_->publish(lookahead_point_msg);

  // calc steering angle for lateral control
  double alpha = std::atan2(lookahead_point_y - rear_y, lookahead_point_x - rear_x) -
                 tf2::getYaw(odometry_->pose.pose.orientation);
  cmd.lateral.steering_tire_angle =
    steering_tire_angle_gain * std::atan2(2.0 * wheel_base_ * std::sin(alpha), adjusted_lookahead_distance);

  pub_cmd_->publish(cmd);
  cmd.lateral.steering_tire_angle /=  steering_tire_angle_gain;
  pub_raw_cmd_->publish(cmd);

  auto lookahead_msg = std_msgs::msg::Float64();
  lookahead_msg.data = adjusted_lookahead_distance;
  pub_lookahead_->publish(lookahead_msg);
}

bool SimplePurePursuit::subscribeMessageAvailable()
{
  if (!odometry_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "odometry is not available");
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