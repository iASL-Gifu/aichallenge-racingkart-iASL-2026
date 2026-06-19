#ifndef SIMPLE_PURE_PURSUIT_HPP_
#define SIMPLE_PURE_PURSUIT_HPP_

#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory_point.hpp>
#include <autoware_auto_vehicle_msgs/msg/steering_report.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace simple_pure_pursuit {

using autoware_auto_control_msgs::msg::AckermannControlCommand;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PointStamped;
using geometry_msgs::msg::Twist;
using nav_msgs::msg::Odometry;

class SimplePurePursuit : public rclcpp::Node {
 public:
  explicit SimplePurePursuit();

 private:
  // subscribers
  rclcpp::Subscription<Odometry>::SharedPtr sub_kinematics_;
  rclcpp::Subscription<Trajectory>::SharedPtr sub_trajectory_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_status_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>::SharedPtr sub_steering_;
  
  // publishers
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_cmd_;
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_raw_cmd_;
  rclcpp::Publisher<PointStamped>::SharedPtr pub_lookahead_point_;  
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_lookahead_;

  // timer
  rclcpp::TimerBase::SharedPtr timer_;

  // updated by subscribers
  Trajectory::SharedPtr trajectory_;
  Odometry::SharedPtr odometry_;
  
  // State variables
  double current_steering_angle_{0.0};
  double lookahead_distance{5.5};
  double steering_tire_angle_gain{1.0};
  int current_lap_{-1};         // 初回判定に使うため-1でOK
  int current_section_{0};      // 配列のインデックスに使うため0が安全
  double cumulative_laps_time_{0.0};

  // pure pursuit parameters
  const float wheel_base_;      // cpp側でdeclare_parameter<float>となっているため型を合わせました
  const std::vector<double> lookahead_distance_1_;
  const std::vector<double> lookahead_distance_2_;
  const double steering_angle_to_lookahead_ratio_;
  const float speed_proportional_gain_1_;
  const float speed_proportional_gain_2_;
  const float high_acceleration_speed_threshold_;
  const double adaptive_gain_steering_threshold_;
  const double adaptive_gain_lookahead_threshold_;
  const double adaptive_gain_reduction_ratio_;
  const bool use_external_target_vel_;
  const float external_target_vel_;
  const std::vector<double> steering_tire_angle_gain_;

  // Callbacks and core methods
  void onTimer();
  bool subscribeMessageAvailable();
  void steeringCallback(const autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr msg);
  void statusCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
};

}  // namespace simple_pure_pursuit

#endif  // SIMPLE_PURE_PURSUIT_HPP_