import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import sys

class MarkerDetailChecker(Node):
    def __init__(self):
        super().__init__('marker_detail_checker')
        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.sub = self.create_subscription(
            MarkerArray,
            '/map/vector_map_marker',
            self.callback,
            qos_profile
        )
        self.received = False

    def callback(self, msg):
        for m in msg.markers:
            if m.ns in ['left_lane_bound', 'right_lane_bound']:
                print(f"--- Namespace: {m.ns} (ID: {m.id}) ---")
                print(f"  Frame ID: {m.header.frame_id}")
                print(f"  Type: {m.type} (TRIANGLE_LIST is 11, LINE_STRIP is 4)")
                print(f"  Number of points: {len(m.points)}")
                print(f"  Number of colors: {len(m.colors)}")
                if len(m.points) > 0:
                    print("  First 5 points:")
                    for idx, p in enumerate(m.points[:5]):
                        print(f"    [{idx}] x={p.x:.6f}, y={p.y:.6f}, z={p.z:.6f}")
        self.received = True
        sys.exit(0)

def main():
    rclpy.init()
    node = DetailChecker = MarkerDetailChecker()
    import threading
    def timeout():
        import time
        time.sleep(10.0)
        if not node.received:
            print("Timeout waiting for marker array")
            sys.exit(1)
    
    t = threading.Thread(target=timeout)
    t.daemon = True
    t.start()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
