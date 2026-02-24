#!/usr/bin/env python3
import threading
import time
import math
import sys

import numpy as np
np.float = float  

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, QuaternionStamped, TransformStamped
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import TransformBroadcaster

from pymavlink import mavutil

class GimbalController(Node):
    def __init__(self):
        super().__init__('gimbal_controller')

        self.declare_parameter('serial_port', '/dev/ttyTHS1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('mavlink_url', '')
        port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud = self.get_parameter('baudrate').get_parameter_value().integer_value
        mavlink_url = self.get_parameter('mavlink_url').get_parameter_value().string_value

        endpoint = mavlink_url if mavlink_url else port
        if mavlink_url:
            self.mav = mavutil.mavlink_connection(endpoint)
            self.get_logger().info(f"Waiting for heartbeat on {endpoint}...")
        else:
            self.mav = mavutil.mavlink_connection(endpoint, baud=baud)
            self.get_logger().info(f"Waiting for heartbeat on {endpoint}@{baud}...")
        self.mav.wait_heartbeat()
        self.get_logger().info("Heartbeat received")

        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_MOUNT_CONFIGURE,
            0,
            mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING,
            0, 0, 0, 0, 0, 0
        )

        self.ref_q = [0.0, 0.0, 0.0, 1.0]

        self.current_rpy = [0.0, 0.0, 0.0]

        self.create_subscription(
            QuaternionStamped,
            'gimbal_reference',
            self.reference_callback,
            10
        )

        self.q_pub = self.create_publisher(QuaternionStamped, 'gimbal_orientation', 10)
        self.tf_br = TransformBroadcaster(self)

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self._stop_event = threading.Event()
        t = threading.Thread(target=self._mavlink_reader, daemon=True)
        t.start()

    def reference_callback(self, msg: Quaternion):
        self.ref_q = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]

    def timer_callback(self):
        roll, pitch, yaw = euler_from_quaternion(self.ref_q)

        deg_roll  = math.degrees(roll)
        deg_pitch = math.degrees(pitch)
        deg_yaw   = math.degrees(yaw)
        # deg_yaw   = 30.0

        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
            0,
            deg_pitch,    # pitch
            deg_roll,     # roll
            deg_yaw,      # yaw
            0, 0, 0,      # unused
            mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING
        )
        self.get_logger().debug(
            f"Sent MOUNT_CONTROL pitch={deg_pitch:.1f}, roll={deg_roll:.1f}, yaw={deg_yaw:.1f}"
        )

        r, p, y = self.current_rpy
        q = quaternion_from_euler(
            math.radians(r),
            math.radians(p),
            math.radians(y)
        )
        qt = QuaternionStamped()
        qt.header.stamp = self.get_clock().now().to_msg()
        qt.header.frame_id = 'base_link'
        qt.quaternion.x, qt.quaternion.y, qt.quaternion.z, qt.quaternion.w = q

        self.q_pub.publish(qt)

        # Broadcast tf: base_link → ai_cam
        # t = TransformStamped()
        # t.header.stamp = qt.header.stamp
        # t.header.frame_id = 'base_link'
        # t.child_frame_id = 'ai_cam'
        # t.transform.translation.x = 0.0
        # t.transform.translation.y = 0.0
        # t.transform.translation.z = 0.0
        # t.transform.rotation = qt.quaternion
        # self.tf_br.sendTransform(t)

    def _mavlink_reader(self):
        while not self._stop_event.is_set():
            msg = self.mav.recv_match(type='MOUNT_ORIENTATION', blocking=True, timeout=1)
            if msg:
                self.current_rpy = [msg.roll, msg.pitch, msg.yaw]

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GimbalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
