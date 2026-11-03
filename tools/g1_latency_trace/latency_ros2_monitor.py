"""ROS 2 Foxy/Humble monitor for latency_trace.py collect --ros.

Telemetry only. Never subscribes or publishes robot command topics.
"""
import collections
import json

from latency_trace import analyze, print_summary


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    rclpy.init()

    class Monitor(Node):
        def __init__(self):
            super().__init__('g1_latency_monitor')
            self.events = collections.deque(maxlen=10000)
            self.subscription = self.create_subscription(
                String, '/g1_latency/events', self.receive,
                QoSProfile(depth=2048, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.timer = self.create_timer(5.0, self.report)

        def receive(self, msg):
            try:
                e = json.loads(msg.data)
                if e.get('v') == 1 and all(k in e for k in ('clock','trace_id','stage','event_id')):
                    self.events.append(e)
            except (ValueError, TypeError):
                self.get_logger().warning('Ignoring invalid latency telemetry')

        def report(self):
            print_summary(analyze(list(self.events)))

    node = Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
