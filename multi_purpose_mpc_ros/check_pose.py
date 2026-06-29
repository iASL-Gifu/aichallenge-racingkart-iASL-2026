import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class PoseCheck(Node):
    def __init__(self):
        super().__init__('pose_check')
        self.sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.callback,
            10
        )
        self.count = 0

    def callback(self, msg):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear.x
        print(f"Pose: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f} | Speed={vel:.2f} m/s")
        self.count += 1
        if self.count >= 5:
            raise SystemExit

def main():
    rclpy.init()
    node = PoseCheck()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
