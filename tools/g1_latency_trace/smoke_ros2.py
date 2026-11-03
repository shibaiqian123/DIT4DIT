"""Localhost-only ROS 2/UDP integration test; no robot endpoints."""
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time

os.environ['ROS_DOMAIN_ID'] = '223'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['LATENCY_TRACE_HOST'] = 'test_host'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    os.environ['LATENCY_TRACE_UDP'] = f'127.0.0.1:{port}'
    import latency_trace as trace
    rclpy.init(args=[])
    node = Node('latency_smoke_subscriber')
    received = []
    subscription = node.create_subscription(String, '/g1_latency/events',
        lambda msg: received.append(json.loads(msg.data)),
        QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)/'trace.jsonl'
        proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('latency_trace.py')),
                                 'collect', '--ros', '--port', str(port), '--output', str(output)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if not select.select([proc.stdout], [], [], 15)[0]:
                raise RuntimeError('Collector startup timeout')
            line = proc.stdout.readline()
            if 'Recording local trace events:' not in line:
                raise RuntimeError('Collector failed: '+line)
            deadline = time.monotonic()+10
            while not received and time.monotonic()<deadline:
                trace.mark('smoke-request', 'request_send_start')
                rclpy.spin_once(node, timeout_sec=.1)
            assert received and received[0]['trace_id']=='smoke-request'
            assert json.loads(output.read_text().splitlines()[0])['stage']=='request_send_start'
            print('PASS: SDK -> local UDP -> JSONL -> ROS 2 topic -> subscriber')
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.communicate()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
