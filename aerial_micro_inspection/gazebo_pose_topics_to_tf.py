#!/usr/bin/env python3
"""Publish ROS TF tree from bridged Gazebo pose topics.

FRAME_TREE_MAP documents the exact mapping from Gazebo frame/link/sensor names
(as carried in bridged TransformStamped frame ids) to ROS TF frame ids.
"""

import math
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


# Gazebo frame or entity name -> ROS TF frame name.
# This is the canonical tree mapping used by this node.
FRAME_TREE_MAP: Dict[str, str] = {
    "tree_car_world": "tree_car_world",
    "x500_inspection_0": "uav_gt",
    "x500_inspection_0/x500_inspection_0": "gimbal_root_gt",
    "x500_inspection_0/x500_inspection_0/cgo3_mount_link": "gimbal_mount_gt",
    "x500_inspection_0/x500_inspection_0/cgo3_vertical_arm_link": "gimbal_yaw_link_gt",
    "x500_inspection_0/x500_inspection_0/cgo3_horizontal_arm_link": "gimbal_roll_link_gt",
    "x500_inspection_0/x500_inspection_0/camera_link": "gimbal_camera_link_gt",
    "x500_inspection_0/x500_inspection_0/camera_link/camera": "inspection_cam_sensor_gt",
    "x500_inspection_0/x500_inspection_0/camera_link/camera_imu": "inspection_cam_imu_gt",
    "x500_inspection_0/inspection_stereo": "nav_rig_gt",
    "x500_inspection_0/inspection_stereo/camera_link": "nav_camera_link_gt",
    "x500_inspection_0/inspection_stereo/camera_link/IMX214": "nav_rgb_sensor_gt",
    "x500_inspection_0/inspection_stereo/camera_link/StereoOV7251": "nav_depth_sensor_gt",
    "x500_inspection_0/inspection_stereo/camera_link/SemanticGT": "nav_seg_sensor_gt",
}

# ROS topics produced by ros_gz_bridge from the matching Gazebo pose topics.
POSE_TOPICS: List[str] = [
    "/model/x500_inspection_0/pose",
    "/model/x500_inspection_0/model/x500_inspection_0/pose",
    "/model/x500_inspection_0/model/inspection_stereo/pose",
]

# parent_sensor_frame -> optical_child_frame
OPTICAL_FRAME_MAP: Dict[str, str] = {
    "inspection_cam_sensor_gt": "inspection_cam_optical_gt",
    "nav_rgb_sensor_gt": "nav_rgb_optical_gt",
    "nav_depth_sensor_gt": "nav_depth_optical_gt",
    "nav_seg_sensor_gt": "nav_seg_optical_gt",
}

# Sensor frame -> source TF child frame name whose timestamp should be reused.
SENSOR_TO_GZ_CHILD: Dict[str, str] = {
    "inspection_cam_sensor_gt": "x500_inspection_0/x500_inspection_0/camera_link/camera",
    "nav_rgb_sensor_gt": "x500_inspection_0/inspection_stereo/camera_link/IMX214",
    "nav_depth_sensor_gt": "x500_inspection_0/inspection_stereo/camera_link/StereoOV7251",
    "nav_seg_sensor_gt": "x500_inspection_0/inspection_stereo/camera_link/SemanticGT",
}


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class GazeboPoseTopicsToTf(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_pose_topics_to_tf")

        self.declare_parameter("optical_rpy", [-1.57079632679, 0.0, -1.57079632679])
        self.declare_parameter("publish_unmapped_frames", False)

        self.publish_unmapped = bool(self.get_parameter("publish_unmapped_frames").value)

        optical_rpy = self.get_parameter("optical_rpy").value
        self.q_sensor_to_optical = _quat_from_rpy(
            float(optical_rpy[0]), float(optical_rpy[1]), float(optical_rpy[2])
        )

        self.tf_br = TransformBroadcaster(self)

        self._latest_by_child: Dict[str, TransformStamped] = {}

        for topic in POSE_TOPICS:
            self.create_subscription(TransformStamped, topic, self._make_cb(topic), 10)
            self.get_logger().info(f"Subscribed to bridged pose topic: {topic}")

        self._pub_timer = self.create_timer(0.03, self._publish)

        self.get_logger().info("gazebo_pose_topics_to_tf started")

    def _make_cb(self, _topic: str):
        def _cb(msg: TransformStamped) -> None:
            if not msg.child_frame_id:
                return
            self._latest_by_child[msg.child_frame_id] = msg

        return _cb

    def _map_frame(self, gz_name: str) -> Optional[str]:
        mapped = FRAME_TREE_MAP.get(gz_name)
        if mapped is None:
            mapped = FRAME_TREE_MAP.get(gz_name.replace("::", "/"))
        if mapped is not None:
            return mapped
        if self.publish_unmapped:
            return gz_name.replace("::", "/")
        return None

    def _send_tf(
        self,
        parent: str,
        child: str,
        tx: float,
        ty: float,
        tz: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        stamp,
    ) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = tz
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_br.sendTransform(t)

    def _publish(self) -> None:
        blocks = list(self._latest_by_child.values())

        published_sensor_frames = set()
        stamp_by_sensor_frame = {}

        for p in blocks:
            
            parent = self._map_frame(p.header.frame_id)
            child = self._map_frame(p.child_frame_id)
            if parent is None or child is None:
                continue

            self._send_tf(
                parent,
                child,
                p.transform.translation.x,
                p.transform.translation.y,
                p.transform.translation.z,
                p.transform.rotation.x,
                p.transform.rotation.y,
                p.transform.rotation.z,
                p.transform.rotation.w,
                p.header.stamp,
            )
            published_sensor_frames.add(child)

            # Keep the source stamp for optical frame children.
            for sensor_frame, gz_child in SENSOR_TO_GZ_CHILD.items():
                child_norm = p.child_frame_id.replace("::", "/")
                if p.child_frame_id == gz_child or child_norm == gz_child:
                    stamp_by_sensor_frame[sensor_frame] = p.header.stamp
                    break

        # Publish optical frames relative to camera sensor frames.
        for sensor_frame, optical_frame in OPTICAL_FRAME_MAP.items():
            if sensor_frame not in published_sensor_frames:
                continue
            stamp = stamp_by_sensor_frame.get(sensor_frame, self.get_clock().now().to_msg())
            self._send_tf(
                sensor_frame,
                optical_frame,
                0.0,
                0.0,
                0.0,
                self.q_sensor_to_optical[0],
                self.q_sensor_to_optical[1],
                self.q_sensor_to_optical[2],
                self.q_sensor_to_optical[3],
                stamp,
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GazeboPoseTopicsToTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
