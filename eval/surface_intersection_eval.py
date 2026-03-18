#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional
import struct

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.clock import ClockType

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Float32, Float32MultiArray
from std_msgs.msg import Empty
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_matrix


@dataclass
class CachedImage:
    msg: Image
    data: np.ndarray


class SurfaceIntersectionEvaluator(Node):
    def __init__(self) -> None:
        super().__init__('surface_intersection_evaluator')

        # -------------------------------
        # In-code configuration parameters
        # -------------------------------
        self.segmentation_topic = '/nav_cam/segmentation/labels_map'
        self.depth_topic = '/nav_cam/depth_image/image_raw'
        self.rgb_topic = '/nav_cam/rgb_image/image_raw'
        self.nav_camera_info_topic = '/nav_cam/depth_image/camera_info'
        self.inspection_camera_info_topic = '/inspection_cam/rgb_image/camera_info'
        self.sweep_reset_topic = '/gimbal_tracker/sweep_reset'
        self.sweep_windows_topic = '/gimbal_tracker/sweep_windows_px'

        self.world_frame = 'tree_car_world'
        self.nav_depth_frame = 'nav_depth_optical_gt'
        self.gimbal_setpoint_frame = self.nav_depth_frame
        self.inspection_frame = 'inspection_cam_optical_gt'
        self.uav_frame = 'uav_gt'
        self.use_latest_tf = True
        self.target_label_id = 5
        self.sync_tolerance_sec = 0.08

        self.min_depth_m = 0.2
        self.max_depth_m = 30.0
        self.downsample_step = 2
        self.setpoint_path_inward_offset_m = 0.2
        self.setpoint_depth_jump_threshold_m = 0.8

        self.ray_match_max_distance_m = 0.50
        self.trajectory_min_separation_m = 0.01
        self.gimbal_traj_inward_offset_m = 0.25
        # Backward-compatible alias used by older code/comments.
        self.intersection_closer_to_cam_m = 0.05

        self.surface_cloud_topic = '/eval/target_surface_cloud'
        self.visited_surface_cloud_topic = '/eval/target_surface_visited_cloud'
        self.intersection_path_topic = '/eval/inspection_axis_intersection_path'
        self.uav_path_topic = '/eval/uav_path'
        self.gimbal_setpoint_path_topic = '/eval/gimbal_setpoint_path'
        self.coverage_ratio_topic = '/eval/target_surface_coverage_ratio'

        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos_sensor = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Image, self.segmentation_topic, self._seg_cb, qos_sensor)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, qos_sensor)
        self.create_subscription(Image, self.rgb_topic, self._rgb_cb, qos_sensor)
        self.create_subscription(CameraInfo, self.nav_camera_info_topic, self._nav_info_cb, qos_sensor)
        self.create_subscription(CameraInfo, self.inspection_camera_info_topic, self._inspection_info_cb, qos_sensor)
        self.create_subscription(Empty, self.sweep_reset_topic, self._reset_cb, 10)
        self.create_subscription(Float32MultiArray, self.sweep_windows_topic, self._sweep_windows_cb, 10)

        self.surface_cloud_pub = self.create_publisher(PointCloud2, self.surface_cloud_topic, 10)
        self.visited_surface_cloud_pub = self.create_publisher(PointCloud2, self.visited_surface_cloud_topic, 10)
        self.intersection_path_pub = self.create_publisher(Path, self.intersection_path_topic, 10)
        self.uav_path_pub = self.create_publisher(Path, self.uav_path_topic, 10)
        self.gimbal_setpoint_path_pub = self.create_publisher(Path, self.gimbal_setpoint_path_topic, 10)
        self.coverage_ratio_pub = self.create_publisher(Float32, self.coverage_ratio_topic, 10)

        self.nav_cam_info: Optional[CameraInfo] = None
        self.inspection_cam_info: Optional[CameraInfo] = None
        self.latest_seg: Optional[CachedImage] = None
        self.latest_depth: Optional[CachedImage] = None
        self.latest_rgb: Optional[CachedImage] = None

        self.latest_surface_points_world = np.empty((0, 3), dtype=np.float32)
        self.latest_surface_colors = np.empty((0, 3), dtype=np.uint8)
        self.frozen_surface_points_world = np.empty((0, 3), dtype=np.float32)
        self.frozen_surface_colors = np.empty((0, 3), dtype=np.uint8)
        self.visited_mask = np.zeros((0,), dtype=bool)
        self.waiting_for_sweep_start_cloud = True
        self.intersection_history = []
        self.last_processed_ns = -1
        self.pending_sweep_windows_px = []

        self.intersection_path = Path()
        self.intersection_path.header.frame_id = self.world_frame
        self.uav_path = Path()
        self.uav_path.header.frame_id = self.world_frame
        self.gimbal_setpoint_path = Path()
        self.gimbal_setpoint_path.header.frame_id = self.gimbal_setpoint_frame

        self.get_logger().info('SurfaceIntersectionEvaluator started')
        self.get_logger().info(f'segmentation_topic={self.segmentation_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')
        self.get_logger().info(f'rgb_topic={self.rgb_topic}')
        self.get_logger().info(f'nav_camera_info_topic={self.nav_camera_info_topic}')
        self.get_logger().info(f'inspection_camera_info_topic={self.inspection_camera_info_topic}')
        self.get_logger().info(f'sweep_reset_topic={self.sweep_reset_topic}')
        self.get_logger().info(f'sweep_windows_topic={self.sweep_windows_topic}')
        self.get_logger().info(f'world_frame={self.world_frame}')
        self.get_logger().info(f'gimbal_setpoint_frame={self.gimbal_setpoint_frame}')
        self.get_logger().info(f'nav_depth_frame={self.nav_depth_frame}')
        self.get_logger().info(f'inspection_frame={self.inspection_frame}')
        self.get_logger().info(f'uav_frame={self.uav_frame}')
        self.get_logger().info(f'use_latest_tf={self.use_latest_tf}')
        self.get_logger().info(f'gimbal_setpoint_path_topic={self.gimbal_setpoint_path_topic}')
        self.get_logger().info(f'visited_surface_cloud_topic={self.visited_surface_cloud_topic}')
        self.get_logger().info(f'coverage_ratio_topic={self.coverage_ratio_topic}')

    @staticmethod
    def _stamp_to_sec(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _stamp_to_ns(msg: Image) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _nav_info_cb(self, msg: CameraInfo) -> None:
        self.nav_cam_info = msg

    def _inspection_info_cb(self, msg: CameraInfo) -> None:
        self.inspection_cam_info = msg

    def _apply_reset_now(self) -> None:
        # Report final coverage for the finished sweep before clearing state.
        final_ratio = self._compute_coverage_ratio()
        if self.frozen_surface_points_world.shape[0] > 0:
            self.get_logger().info(f'Final sweep coverage ratio: {final_ratio:.4f}')

        self.intersection_history.clear()
        self.intersection_path.poses.clear()
        self.uav_path.poses.clear()
        self.gimbal_setpoint_path.poses.clear()
        self.pending_sweep_windows_px = []
        self.frozen_surface_points_world = np.empty((0, 3), dtype=np.float32)
        self.frozen_surface_colors = np.empty((0, 3), dtype=np.uint8)
        self.visited_mask = np.zeros((0,), dtype=bool)
        self.waiting_for_sweep_start_cloud = True

        now = self.get_clock().now().to_msg()
        self.intersection_path.header.stamp = now
        self.uav_path.header.stamp = now
        self.gimbal_setpoint_path.header.stamp = now
        self.intersection_path_pub.publish(self.intersection_path)
        self.uav_path_pub.publish(self.uav_path)
        self.gimbal_setpoint_path_pub.publish(self.gimbal_setpoint_path)
        self._publish_colored_cloud(
            self.surface_cloud_pub,
            self.frozen_surface_points_world,
            self.frozen_surface_colors,
            now,
        )
        self._publish_colored_cloud(
            self.visited_surface_cloud_pub,
            self.frozen_surface_points_world,
            self.frozen_surface_colors,
            now,
        )
        self.coverage_ratio_pub.publish(Float32(data=0.0))

    def _rgb_cb(self, msg: Image) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = CachedImage(msg=msg, data=np.asarray(rgb))
            self._try_process()
        except Exception as exc:
            self.get_logger().warning(f'Failed to convert rgb image: {exc}')

    def _reset_cb(self, _msg: Empty) -> None:
        self._apply_reset_now()
        self.get_logger().info('Received sweep reset signal -> cleared intersection and UAV paths')

    def _sweep_windows_cb(self, msg: Float32MultiArray) -> None:
        values = list(msg.data)
        if len(values) < 4:
            self.pending_sweep_windows_px = []
            return

        windows = []
        usable = len(values) - (len(values) % 4)
        for i in range(0, usable, 4):
            x0 = float(values[i])
            y0 = float(values[i + 1])
            x1 = float(values[i + 2])
            y1 = float(values[i + 3])
            windows.append((x0, y0, x1, y1))

        self.pending_sweep_windows_px = windows
        stamp = self.latest_depth.msg.header.stamp if self.latest_depth is not None else self.get_clock().now().to_msg()
        self._publish_gimbal_setpoint_path_from_windows(stamp)

    def _publish_gimbal_setpoint_path_from_windows(self, stamp) -> None:
        if self.nav_cam_info is None or self.latest_depth is None:
            return
        if len(self.pending_sweep_windows_px) == 0:
            return

        depth = self.latest_depth.data
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        h, w = depth_m.shape[:2]
        k = self.nav_cam_info.k
        fx, fy = float(k[0]), float(k[4])
        cx0, cy0 = float(k[2]), float(k[5])

        # First pass: estimate per-window center depth in window order.
        centers = []
        depths = []
        for x0, y0, x1, y1 in self.pending_sweep_windows_px:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            centers.append((cx, cy))

            ix = int(round(cx))
            iy = int(round(cy))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                depths.append(None)
                continue

            px0 = max(0, ix - 2)
            px1 = min(w, ix + 3)
            py0 = max(0, iy - 2)
            py1 = min(h, iy + 3)
            vals = depth_m[py0:py1, px0:px1]
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals > self.min_depth_m) & (vals < self.max_depth_m)]
            if vals.size == 0:
                depths.append(None)
                continue

            depths.append(float(np.mean(vals)))

        # Smooth abrupt per-window depth jumps using neighboring windows.
        smoothed_depths = list(depths)
        for i in range(1, len(depths) - 1):
            z = depths[i]
            z_prev = depths[i - 1]
            z_next = depths[i + 1]
            if z is None or z_prev is None or z_next is None:
                continue
            if (
                abs(z - z_prev) > self.setpoint_depth_jump_threshold_m
                and abs(z - z_next) > self.setpoint_depth_jump_threshold_m
            ):
                smoothed_depths[i] = 0.5 * (z_prev + z_next)

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.gimbal_setpoint_frame

        for (cx, cy), z_raw in zip(centers, smoothed_depths):
            if z_raw is None:
                continue

            z = max(self.min_depth_m, z_raw)
            x = (cx - cx0) * z / fx
            y = (cy - cy0) * z / fy

            # Pull path points slightly toward camera to avoid z-fighting/occlusion in RViz.
            z_vis = max(self.min_depth_m, z - self.setpoint_path_inward_offset_m)
            scale = z_vis / z if z > 1e-6 else 1.0
            x_vis = x * scale
            y_vis = y * scale

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.gimbal_setpoint_frame
            pose.pose.position.x = x_vis
            pose.pose.position.y = y_vis
            pose.pose.position.z = z_vis
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.gimbal_setpoint_path = path
        self.gimbal_setpoint_path_pub.publish(self.gimbal_setpoint_path)

    def _seg_cb(self, msg: Image) -> None:
        try:
            seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            seg = np.asarray(seg)
            if seg.ndim == 3:
                seg = seg[..., 0]
            self.latest_seg = CachedImage(msg=msg, data=seg)
            self._try_process()
        except Exception as exc:
            self.get_logger().warning(f'Failed to convert segmentation image: {exc}')

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = np.asarray(depth)
            self.latest_depth = CachedImage(msg=msg, data=depth)
            self._try_process()
        except Exception as exc:
            self.get_logger().warning(f'Failed to convert depth image: {exc}')

    def _try_process(self) -> None:
        if self.nav_cam_info is None or self.inspection_cam_info is None:
            return
        if self.latest_seg is None or self.latest_depth is None or self.latest_rgb is None:
            return

        seg_msg = self.latest_seg.msg
        depth_msg = self.latest_depth.msg
        rgb_msg = self.latest_rgb.msg

        dt = abs(self._stamp_to_sec(seg_msg) - self._stamp_to_sec(depth_msg))
        if dt > self.sync_tolerance_sec:
            return

        dt_rgb = abs(self._stamp_to_sec(seg_msg) - self._stamp_to_sec(rgb_msg))
        if dt_rgb > self.sync_tolerance_sec:
            return

        current_ns = self._stamp_to_ns(seg_msg)
        if current_ns == self.last_processed_ns:
            return
        self.last_processed_ns = current_ns

        points_world, colors = self._reconstruct_surface_world(
            self.latest_seg.data,
            self.latest_depth.data,
            self.latest_rgb.data,
            seg_msg,
        )
        self.latest_surface_points_world = points_world
        self.latest_surface_colors = colors

        if self.waiting_for_sweep_start_cloud and points_world.size > 0:
            # Latch first valid surface cloud after each sweep reset.
            self.frozen_surface_points_world = points_world.copy()
            self.frozen_surface_colors = colors.copy()
            self.visited_mask = np.zeros((self.frozen_surface_points_world.shape[0],), dtype=bool)
            self.waiting_for_sweep_start_cloud = False
            self.get_logger().info(
                f'Latched sweep-start car cloud with {self.frozen_surface_points_world.shape[0]} points')

        pub_points = self.frozen_surface_points_world
        pub_colors = self.frozen_surface_colors
        if pub_points.size == 0 and points_world.size > 0:
            pub_points = points_world
            pub_colors = colors

        self._publish_colored_cloud(self.surface_cloud_pub, pub_points, pub_colors, seg_msg.header.stamp)

        self._update_uav_path(seg_msg)
        self._update_visited_from_fov(seg_msg)

        intersection = self._compute_optical_axis_intersection(seg_msg)
        if intersection is not None:
            self._update_trajectory(intersection)
            self._publish_intersection_path(seg_msg)
            self.uav_path_pub.publish(self.uav_path)

        self._publish_gimbal_setpoint_path_from_windows(seg_msg.header.stamp)

    def _reconstruct_surface_world(
        self,
        seg: np.ndarray,
        depth: np.ndarray,
        rgb: np.ndarray,
        stamp_source: Image,
    ) -> tuple[np.ndarray, np.ndarray]:
        if seg.shape[:2] != depth.shape[:2]:
            self.get_logger().warning(
                f'Shape mismatch: seg={seg.shape} depth={depth.shape}; skipping frame')
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        if rgb.shape[:2] != depth.shape[:2]:
            self.get_logger().warning(
                f'Shape mismatch: rgb={rgb.shape} depth={depth.shape}; skipping frame')
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        mask = (seg.astype(np.int64) == int(self.target_label_id))
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        valid = mask & np.isfinite(depth_m) & (depth_m > self.min_depth_m) & (depth_m < self.max_depth_m)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        v, u = np.where(valid)
        if self.downsample_step > 1:
            keep = np.arange(0, len(u), self.downsample_step)
            u = u[keep]
            v = v[keep]

        z = depth_m[v, u]

        k = self.nav_cam_info.k
        fx, fy = float(k[0]), float(k[4])
        cx, cy = float(k[2]), float(k[5])

        x = (u.astype(np.float32) - cx) * z / fx
        y = (v.astype(np.float32) - cy) * z / fy
        pts_nav = np.stack([x, y, z], axis=1).astype(np.float32)

        colors = rgb[v, u, :3].astype(np.uint8)

        nav_frame = self.nav_depth_frame
        if not nav_frame:
            self.get_logger().warning('No nav camera frame_id available; cannot transform to world')
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        transform = self._lookup_transform(self.world_frame, nav_frame, stamp_source)
        if transform is None:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        return self._transform_points(pts_nav, transform), colors

    def _update_uav_path(self, stamp_source: Image) -> None:
        tf_world_uav = self._lookup_transform(self.world_frame, self.uav_frame, stamp_source)
        if tf_world_uav is None:
            return

        pose = PoseStamped()
        pose.header.stamp = stamp_source.header.stamp
        pose.header.frame_id = self.world_frame
        pose.pose.position.x = tf_world_uav.transform.translation.x
        pose.pose.position.y = tf_world_uav.transform.translation.y
        pose.pose.position.z = tf_world_uav.transform.translation.z
        pose.pose.orientation = tf_world_uav.transform.rotation
        self.uav_path.poses.append(pose)
        self.uav_path.header.stamp = stamp_source.header.stamp

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp_source: Image):
        if self.use_latest_tf:
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME),
                    timeout=Duration(seconds=0.15),
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                self.get_logger().warning(
                    f'TF lookup failed (latest mode): {target_frame} <- {source_frame}: {exc}')
                return None

        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp_source.header.stamp,
                timeout=Duration(seconds=0.15),
            )
        except ExtrapolationException:
            # If requested stamp is outside buffer (past/future), use latest available transform.
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(seconds=0, nanoseconds=0, clock_type=ClockType.ROS_TIME),
                    timeout=Duration(seconds=0.15),
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                self.get_logger().warning(
                    f'TF lookup failed (latest fallback): {target_frame} <- {source_frame}: {exc}')
                return None
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self.get_logger().warning(
                f'TF lookup failed: {target_frame} <- {source_frame}: {exc}')
            return None

    @staticmethod
    def _transform_points(points: np.ndarray, tf_msg) -> np.ndarray:
        q = tf_msg.transform.rotation
        t = tf_msg.transform.translation

        mat = quaternion_matrix([q.x, q.y, q.z, q.w])
        rot = mat[:3, :3].astype(np.float32)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return (points @ rot.T) + trans

    def _compute_optical_axis_intersection(self, stamp_source: Image) -> Optional[np.ndarray]:
        points_src = self.frozen_surface_points_world
        if points_src.shape[0] == 0:
            points_src = self.latest_surface_points_world

        if points_src.shape[0] == 0:
            return None

        inspection_frame = self.inspection_frame
        if not inspection_frame:
            return None

        tf_world_inspection = self._lookup_transform(self.world_frame, inspection_frame, stamp_source)
        if tf_world_inspection is None:
            return None

        q = tf_world_inspection.transform.rotation
        t = tf_world_inspection.transform.translation

        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        axis_world = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm < 1e-8:
            return None
        axis_world = axis_world / axis_norm

        origin = np.array([t.x, t.y, t.z], dtype=np.float32)
        pts = points_src

        rel = pts - origin
        proj = rel @ axis_world
        forward = proj > 0.0
        if not np.any(forward):
            return None

        rel_f = rel[forward]
        proj_f = proj[forward]
        pts_f = pts[forward]

        closest_on_ray = np.outer(proj_f, axis_world)
        perp_vec = rel_f - closest_on_ray
        perp_dist = np.linalg.norm(perp_vec, axis=1)

        idx = int(np.argmin(perp_dist))
        if float(perp_dist[idx]) > self.ray_match_max_distance_m:
            return None

        intersection_world = pts_f[idx].astype(np.float32)
        to_cam = origin - intersection_world
        dist = float(np.linalg.norm(to_cam))
        if dist > 1e-8:
            shift = min(self.gimbal_traj_inward_offset_m, dist)
            intersection_world = intersection_world + (to_cam / dist) * shift

        return intersection_world

    def _compute_coverage_ratio(self) -> float:
        total = int(self.frozen_surface_points_world.shape[0])
        if total == 0:
            return 0.0
        visited = int(np.count_nonzero(self.visited_mask))
        return float(visited) / float(total)

    def _update_visited_from_fov(self, stamp_source: Image) -> None:
        points = self.frozen_surface_points_world
        colors = self.frozen_surface_colors
        if points.shape[0] == 0:
            self._publish_colored_cloud(self.visited_surface_cloud_pub, points, colors, stamp_source.header.stamp)
            self.coverage_ratio_pub.publish(Float32(data=0.0))
            return

        if self.visited_mask.shape[0] != points.shape[0]:
            self.visited_mask = np.zeros((points.shape[0],), dtype=bool)

        if self.inspection_cam_info is None:
            return

        tf_inspection_world = self._lookup_transform(self.inspection_frame, self.world_frame, stamp_source)
        if tf_inspection_world is None:
            return

        pts_cam = self._transform_points(points, tf_inspection_world)
        z = pts_cam[:, 2]
        valid_depth = z > 1e-6
        safe_z = np.where(valid_depth, z, 1.0)

        k = self.inspection_cam_info.k
        fx, fy = float(k[0]), float(k[4])
        cx, cy = float(k[2]), float(k[5])
        width = int(self.inspection_cam_info.width)
        height = int(self.inspection_cam_info.height)

        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        u = fx * (x / safe_z) + cx
        v = fy * (y / safe_z) + cy

        in_image = (
            valid_depth
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0.0)
            & (u < float(width))
            & (v >= 0.0)
            & (v < float(height))
        )

        self.visited_mask |= in_image
        visited_pts = points[self.visited_mask]
        visited_colors = colors[self.visited_mask]
        self._publish_colored_cloud(
            self.visited_surface_cloud_pub,
            visited_pts,
            visited_colors,
            stamp_source.header.stamp,
        )

        ratio = self._compute_coverage_ratio()
        self.coverage_ratio_pub.publish(Float32(data=float(ratio)))

    def _update_trajectory(self, point: np.ndarray) -> None:
        if len(self.intersection_history) == 0:
            self.intersection_history.append(point)
            return

        last = self.intersection_history[-1]
        if float(np.linalg.norm(point - last)) >= self.trajectory_min_separation_m:
            self.intersection_history.append(point)

    def _publish_intersection_path(self, stamp_source: Image) -> None:
        self.intersection_path.poses.clear()
        for p in self.intersection_history:
            pose = PoseStamped()
            pose.header.stamp = stamp_source.header.stamp
            pose.header.frame_id = self.world_frame
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.position.z = float(p[2])
            pose.pose.orientation.w = 1.0
            self.intersection_path.poses.append(pose)

        self.intersection_path.header.stamp = stamp_source.header.stamp
        self.intersection_path_pub.publish(self.intersection_path)

    def _publish_cloud(self, pub, points_xyz: np.ndarray, stamp) -> None:
        header = Header()
        header.stamp = stamp
        header.frame_id = self.world_frame

        if points_xyz.shape[0] == 0:
            cloud = point_cloud2.create_cloud_xyz32(header, [])
        else:
            cloud = point_cloud2.create_cloud_xyz32(header, points_xyz.tolist())
        pub.publish(cloud)

    def _publish_colored_cloud(self, pub, points_xyz: np.ndarray, colors_bgr: np.ndarray, stamp) -> None:
        header = Header()
        header.stamp = stamp
        header.frame_id = self.world_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        if points_xyz.shape[0] == 0:
            cloud = point_cloud2.create_cloud(header, fields, [])
            pub.publish(cloud)
            return

        pts = []
        for p, c in zip(points_xyz, colors_bgr):
            b, g, r = int(c[0]), int(c[1]), int(c[2])
            rgb_uint32 = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            pts.append((float(p[0]), float(p[1]), float(p[2]), rgb_uint32))

        cloud = point_cloud2.create_cloud(header, fields, pts)
        pub.publish(cloud)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SurfaceIntersectionEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
