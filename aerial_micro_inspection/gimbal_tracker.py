#!/usr/bin/env python3
import sys
import cv2
import time
import numpy as np
import threading
import yaml
import os
from typing import Optional
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf_transformations import quaternion_from_euler
from sensor_msgs.msg import Image
from geometry_msgs.msg import QuaternionStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Empty
from aerial_micro_inspection_interfaces.msg import ObjectDetectionResult
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


from aerial_micro_inspection.utils import load_yaml, quaternion_to_rotmat

## TODO: 
##       Publish the tf for all frames
##       Fix launch file
##       Fix the wrong scale of position in the calibration
##       Increase the accuracy of calibration with iterative methods


class GimbalTracker(Node):
    def __init__(self):
        super().__init__('gimbal_tracker')
        self.bridge = CvBridge()

        # ROS Parameters
        self.declare_parameter('mission_config_file', '')
        self.declare_parameter('configs_dir', '')
        param_file = self.get_parameter('mission_config_file').get_parameter_value().string_value
        with open(param_file, 'r') as f:
            self.config = yaml.safe_load(f)
        configs_dir  = self.get_parameter('configs_dir').value

        output_cfg = self.config.get('output', {})
        input_cfg = self.config.get('input', {})
        pipeline_cfg = self.config.get('pipeline', {})

        det_t  = output_cfg.get('surface_segmentation_topic', output_cfg.get('detection_topic'))
        pln_img_t  = output_cfg.get('gimbal_planning_img_topic')
        nav_img_t  = input_cfg.get('nav_rgb_topic', input_cfg.get('nav_image_topic'))
        ai_img_t   = input_cfg.get('inspection_topic', input_cfg.get('ai_image_topic'))
        depth_t    = input_cfg.get('nav_depth_topic', input_cfg.get('depth_image_topic'))
        states_t   = input_cfg.get('drone_states_topic')
        gimbal_t   = input_cfg.get('gimbal_orientation_topic')
        des_ori_t   = output_cfg.get('gimbal_setpoint_topic', output_cfg.get('gimbal_setpoint'))
        calib_p = os.path.join(configs_dir, input_cfg.get('pan_tilt_calibration'))
        self.planning_mode = pipeline_cfg.get('planning_mode', self.config.get('planning', {}).get('mode', 'sweeping'))
        self.sweeping_overlap = float(pipeline_cfg.get('sweeping_overlap', 0.5))
        self.min_cell_occupancy = float(pipeline_cfg.get('min_cell_occupancy', 0.3))
        self.enable_nav_tracker_mode = bool(pipeline_cfg.get('enable_nav_tracker_mode', False))
        self.tracker_type = str(pipeline_cfg.get('tracker_type', 'medianflow')).lower()
        self.wp_hold_time = float(pipeline_cfg.get('waypoint_hold_time', 2.0))
        self.vis_enabled = self.config['visualization']
        self.img_type = self.config['input']['img_type']
        runtime_cfg = self.config.get('runtime', {})
        self.downsample_visualization = bool(runtime_cfg.get('downsample_visualization', False))
        self.vis_output_width = int(runtime_cfg.get('visualization_width', 240))
        self.vis_output_height = int(runtime_cfg.get('visualization_height', 160))
        self.rt_correction = False

        # Load calibration
        calib = load_yaml(calib_p)
        if not calib:
            self.get_logger().error(f"Cannot load calibration: {calib_p}")
            sys.exit(1)
        # intrinsics for cameras based on what they were during calibration
        self.nav_K = np.array(calib['nav_camera']['camera_matrix']).reshape(3,3)
        self.nav_D = np.array(calib['nav_camera']['distortion'])
        self.ai_K = np.array(calib['ai_camera']['camera_matrix']).reshape(3,3)
        self.ai_D = np.array(calib['ai_camera']['distortion'])
        self.ai_img_shape = (calib['ai_camera']['image_height'], calib['ai_camera']['image_width'])
        # gimbal-base to nav transform (the main output of calibration)
        T_gb_nav = np.array(calib['average_transform_gimbalbase_to_nav']).reshape(4,4)
        self.yaw_offset = float(calib['offset_yaw_deg'])
        self.pitch_offset = float(calib['offset_pitch_deg'])
        # store inverse for mapping nav to gimbal_base
        self.T_nav_gb = np.linalg.inv(T_gb_nav)
        self.nav_h = int(calib['nav_camera']['image_height'])
        self.nav_w = int(calib['nav_camera']['image_width'])

        self.det_subscription = self.create_subscription(
            ObjectDetectionResult,
            det_t,
            self.cb_detection_only,
            10
        )

        sub_nav = Subscriber(self, Image, nav_img_t)
        sub_depth = Subscriber(self, Image, depth_t)
        self.nav_depth_sync = ApproximateTimeSynchronizer(
            [sub_nav, sub_depth],
            queue_size=30,
            slop=0.1
        )
        self.nav_depth_sync.registerCallback(self.cb_nav_depth)

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/gimbal_adjustment',
            self.gimbal_adjusments_callback,
            10  
        )

        self.gimbal_publisher = self.create_publisher(QuaternionStamped, des_ori_t, 10)
        self.image_publisher = self.create_publisher(Image, pln_img_t, 10)
        self.sweep_reset_pub = self.create_publisher(Empty, '/gimbal_tracker/sweep_reset', 10)
        self.sweep_windows_pub = self.create_publisher(Float32MultiArray, '/gimbal_tracker/sweep_windows_px', 10)

        self.executing_trajectory = False
        self.mask_latest = None
        self.wins = []
        self.det_stamp = None

        self.state_lock = threading.Lock()
        self.base_wins = []
        self.current_wp_idx = 0
        self.active_wp_start_time: Optional[float] = None
        self.allow_new_plan = True
        self.pending_tracker_init_bbox = None
        self.latest_tracked_bbox = None
        self.tracker_ref_center = None
        self.tracker_displacement = (0, 0)
        self.cv_tracker = None
        self.latest_det_msg = None
        self.sweep_reset_deadline_wall: Optional[float] = None

        if self.enable_nav_tracker_mode:
            self.get_logger().info('Tracker-driven nav mode enabled.')
        else:
            self.get_logger().info('Tracker-driven nav mode disabled: fixed sweep plan with no motion-feedback updates.')

        self.get_logger().info(f'GimbalTracker ready with {calib_p} config file.')


    def gimbal_adjusments_callback(self, msg):
        self.pitch_offset += msg.data[0]
        self.yaw_offset += msg.data[1]

    def _cancel_sweep_reset_timer(self):
        self.sweep_reset_deadline_wall = None

    def _schedule_sweep_reset(self):
        self._cancel_sweep_reset_timer()

        delay = max(0.0, float(self.wp_hold_time))
        if delay <= 0.0:
            self.sweep_reset_pub.publish(Empty())
            return
        self.sweep_reset_deadline_wall = time.time() + delay

    def _publish_sweep_windows_px(self, wins):
        msg = Float32MultiArray()
        data = []
        for x0, y0, x1, y1 in wins:
            data.extend([float(x0), float(y0), float(x1), float(y1)])
        msg.data = data
        self.sweep_windows_pub.publish(msg)
        
    def cb_detection_only(self, det_msg):
        if self.planning_mode != "sweeping":
            with self.state_lock:
                self.latest_det_msg = det_msg
            return

        with self.state_lock:
            if not self.allow_new_plan:
                return

        if not (det_msg.img_w == self.nav_w and det_msg.img_h == self.nav_h):
            raise ValueError('The detection image size should match the calibrated NAV image size.')

        win = self.get_ai_image_size_on_nav_image(self.ai_img_shape)
        mask = self.bridge.imgmsg_to_cv2(det_msg.mask, 'mono8')
        wins, _ = self.sweep(
            depth=None,
            mask=mask,
            window_size=win,
            overlap=self.sweeping_overlap,
            threshold=self.min_cell_occupancy,
            compute_points=False
        )

        if len(wins) == 0:
            return

        init_bbox = None
        if self.enable_nav_tracker_mode:
            init_bbox = self._sanitize_xywh((det_msg.x, det_msg.y, det_msg.w, det_msg.h))
            if init_bbox[2] <= 1 or init_bbox[3] <= 1:
                self.get_logger().info('Detection bbox too small for tracker initialization.')
                return

        with self.state_lock:
            self.base_wins = wins
            self.wins = list(wins)
            self.mask_latest = mask.copy()
            self.current_wp_idx = 0
            self.active_wp_start_time = None
            self.det_stamp = det_msg.header.stamp
            self.pending_tracker_init_bbox = init_bbox
            self.latest_tracked_bbox = None
            self.tracker_ref_center = None
            self.tracker_displacement = (0, 0)
            self.cv_tracker = None
            self.executing_trajectory = True
            self.allow_new_plan = False
            self.active_wp_start_time = None

        self._publish_sweep_windows_px(wins)

        # Emit sweep start reset after a fixed delay equal to waypoint_hold_time.
        self._schedule_sweep_reset()

    def cb_nav_depth(self, nav_msg, depth_msg):
        nav_img = self.bridge.imgmsg_to_cv2(nav_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        if not (depth.shape[1] == self.nav_w and depth.shape[0] == self.nav_h and
                nav_img.shape[1] == self.nav_w and nav_img.shape[0] == self.nav_h):
            raise ValueError('The NAV RGB and depth image sizes should match the calibrated NAV image size.')

        if self.planning_mode != "sweeping":
            with self.state_lock:
                det_msg = self.latest_det_msg

            if det_msg is None:
                return

            if self.planning_mode == "pitch_yaw_lock":
                p_gb = self.point_towards_box(depth, (det_msg.x, det_msg.y, det_msg.w, det_msg.h))
                if p_gb is None:
                    return
                self.publish_orientation(p_gb, nav_msg.header.stamp)
                return

            if self.planning_mode == "pitch_lock":
                win_w, _ = self.get_ai_image_size_on_nav_image(self.ai_img_shape)
                win_w = 2 * win_w
                mask = self.bridge.imgmsg_to_cv2(det_msg.mask, 'mono8')
                box = self.largest_center_strip_bbox(mask, win_w)
                if box is None:
                    return
                p_gb = self.point_towards_box(depth, box)
                if p_gb is None:
                    return
                self.publish_orientation(p_gb, nav_msg.header.stamp)
            return

        with self.state_lock:
            if not self.executing_trajectory:
                return

            tracked_bbox = None
            if self.enable_nav_tracker_mode:
                if self.cv_tracker is None:
                    if self.pending_tracker_init_bbox is None:
                        return
                    tracker = self._create_opencv_tracker()
                    if tracker is None:
                        self._reset_active_plan('No OpenCV tracker backend available.')
                        return

                    init_bbox = tuple(int(v) for v in self.pending_tracker_init_bbox)
                    self.get_logger().info(f'Initializing tracker with bbox: {init_bbox}')
                    ok = tracker.init(nav_img, init_bbox)

                    if not ok:
                        self._reset_active_plan('Failed to initialize visual tracker on NAV image.')
                        return
                    self.cv_tracker = tracker
                    x, y, w, h = init_bbox
                    self.tracker_ref_center = (x + 0.5 * w, y + 0.5 * h)
                    self.latest_tracked_bbox = init_bbox

                ok, tracked_bbox = self.cv_tracker.update(nav_img)
                if not ok:
                    self._reset_active_plan('Tracker lost target; waiting for a new detection plan.')
                    return

                tracked_bbox = self._sanitize_xywh(tracked_bbox)
                self.latest_tracked_bbox = tracked_bbox

                cx = tracked_bbox[0] + 0.5 * tracked_bbox[2]
                cy = tracked_bbox[1] + 0.5 * tracked_bbox[3]
                dx = int(round(cx - self.tracker_ref_center[0]))
                dy = int(round(cy - self.tracker_ref_center[1]))
            else:
                dx, dy = 0, 0

            self.tracker_displacement = (dx, dy)

            self.wins = self._shift_wins(self.base_wins, dx, dy, self.nav_w, self.nav_h)
            if len(self.wins) == 0:
                self._reset_active_plan('No valid shifted sweep windows remain.')
                return

            if self.current_wp_idx >= len(self.wins):
                self._reset_active_plan('Finished all sweep windows.')
                return

            now = time.time()

            if self.sweep_reset_deadline_wall is not None and now >= self.sweep_reset_deadline_wall:
                self.sweep_reset_pub.publish(Empty())
                self.get_logger().info(
                    f'Sweep reset emitted after startup delay={self.wp_hold_time:.2f}s (waypoint_hold_time)')
                self.sweep_reset_deadline_wall = None

            if self.active_wp_start_time is None:
                self.active_wp_start_time = now

            x0, y0, x1, y1 = self.wins[self.current_wp_idx]
            p_gb = self.point_towards_box(depth, (x0, y0, x1 - x0, y1 - y0))

            self._publish_sweep_visualization(self.current_wp_idx, tracked_bbox=tracked_bbox)

            if p_gb is not None:
                self.publish_orientation(p_gb, nav_msg.header.stamp)

            if self.active_wp_start_time is None:
                self.active_wp_start_time = now

            if (now - self.active_wp_start_time) >= self.wp_hold_time:
                self.current_wp_idx += 1
                self.active_wp_start_time = now
                if self.current_wp_idx >= len(self.wins):
                    self._reset_active_plan('Finished all sweep windows.')

    def _create_opencv_tracker(self):
        tracker_builders = {
            'csrt': ['TrackerCSRT_create'],
            'kcf': ['TrackerKCF_create'],
            'mil': ['TrackerMIL_create'],
            'mosse': ['TrackerMOSSE_create'],
            'medianflow': ['TrackerMedianFlow_create'],
            'boosting': ['TrackerBoosting_create'],
            'tld': ['TrackerTLD_create'],
        }

        preferred = self.tracker_type if self.tracker_type in tracker_builders else 'csrt'
        search_order = [preferred, 'csrt', 'kcf', 'mosse', 'medianflow', 'boosting', 'tld']

        seen = set()
        for tracker_name in search_order:
            if tracker_name in seen:
                continue
            seen.add(tracker_name)

            for ctor_name in tracker_builders[tracker_name]:
                ctor = getattr(cv2, ctor_name, None)
                if ctor is not None:
                    self.get_logger().info(f'Using OpenCV tracker backend: {tracker_name}')
                    return ctor()

                legacy = getattr(cv2, 'legacy', None)
                if legacy is not None:
                    legacy_ctor = getattr(legacy, ctor_name, None)
                    if legacy_ctor is not None:
                        self.get_logger().info(f'Using OpenCV tracker backend: {tracker_name} (legacy)')
                        return legacy_ctor()

        return None

    def _sanitize_xywh(self, bbox):
        x, y, w, h = [int(round(v)) for v in bbox]
        x = max(0, min(x, self.nav_w - 1))
        y = max(0, min(y, self.nav_h - 1))
        w = max(0, min(w, self.nav_w - x))
        h = max(0, min(h, self.nav_h - y))
        return (x, y, w, h)

    def _shift_wins(self, wins, dx, dy, width, height):
        shifted = []
        for x0, y0, x1, y1 in wins:
            w = x1 - x0
            h = y1 - y0
            sx0 = max(0, min(x0 + dx, width - w))
            sy0 = max(0, min(y0 + dy, height - h))
            sx1 = sx0 + w
            sy1 = sy0 + h
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            shifted.append((sx0, sy0, sx1, sy1))
        return shifted

    def _reset_active_plan(self, reason=''):
        if reason:
            self.get_logger().info(reason)
        self.executing_trajectory = False
        self.allow_new_plan = True
        self.base_wins = []
        self.wins = []
        self.current_wp_idx = 0
        self.active_wp_start_time = None
        self.pending_tracker_init_bbox = None
        self.latest_tracked_bbox = None
        self.tracker_ref_center = None
        self.tracker_displacement = (0, 0)
        self.cv_tracker = None
        self._cancel_sweep_reset_timer()

    def largest_center_strip_bbox(self, mask: np.ndarray, win_w: int):

        h, w = mask.shape

        half = win_w // 2
        cx = w // 2
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        strip_mask = mask.copy()
        strip_mask[:, :x1] = 0
        strip_mask[:, x2:] = 0

        contours, _ = cv2.findContours(
            strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)

        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        color_img[mask > 0] = (255, 255, 255)
        color_img[strip_mask > 0] = (0, 0, 255)

        img_resized = cv2.resize(color_img, (240, 160))
        patched_msg = self.bridge.cv2_to_imgmsg(img_resized, encoding='bgr8')
        patched_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_publisher.publish(patched_msg)

        return (x, y, bw, bh)

    def get_ai_image_size_on_nav_image(self, ai_image_shape):
        # Estimate size of AI image in NAV image using depth and intrinsics
        f_ai_x, f_ai_y = self.ai_K[0, 0], self.ai_K[1, 1]
        f_nav_x, f_nav_y = self.nav_K[0, 0], self.nav_K[1, 1]

        ai_h, ai_w = ai_image_shape

        # Projected size of AI image in nav image
        scale_x = f_nav_x / f_ai_x
        scale_y = f_nav_y / f_ai_y

        # How large would ai_img appear in nav image (in pixels)
        proj_w = int(ai_w * scale_x)
        proj_h = int(ai_h * scale_y)        

        return (proj_w, proj_h)
    
    def sweep(self,
                    depth: Optional[np.ndarray],
          mask: np.ndarray,
          window_size: tuple[int,int],
          overlap: float = 0.1,
          threshold: float = 0.7,
                    visualize: bool = False,
                    compute_points: bool = True
         ) -> list[tuple[int,int,int,int]]:

        if compute_points and depth is None:
                raise ValueError('Depth image is required when compute_points=True in sweep().')
        
        win_w, win_h = window_size
        area = win_w * win_h

        # Find tight white‐pixel bbox
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return [], []
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min() - int(win_h/2), ys.max() + int(win_h/2)

        # Compute step sizes
        step_x = max(1, int(win_w * (1.0 - overlap)))
        step_y = max(1, int(win_h * (1.0 - overlap)))

        x0_list = []
        for x in range(x_min, x_max + 1, step_x):
            x0 = min(x, x_max - win_w)
            x0 = max(0, x0)
            x0_list.append(x0)
        x0_list = list(dict.fromkeys(x0_list))

        # Precompute all y0 positions (top to bottom), then reverse to bottom to top
        y0_list = []
        for y in range(y_min, y_max + 1, step_y):
            y0 = min(y, y_max - win_h)
            y0 = max(0, y0)
            y0_list.append(y0)
        y0_list = list(dict.fromkeys(y0_list))[::-1]

        points_gb = []
        boxes = []
        counter = 1

        
        # Sweep bottom-to-top with boustrophedon ordering:
        # even rows left->right, odd rows right->left.
        for row_idx, y0 in enumerate(y0_list):
            y1 = y0 + win_h
            row_x0_list = x0_list if (row_idx % 2 == 0) else list(reversed(x0_list))
            for x0 in row_x0_list:
                x1 = x0 + win_w
                # count white pixels in this window
                count_white = np.count_nonzero(mask[y0:y1, x0:x1])
                if count_white >= threshold * area:
                    boxes.append((x0, y0, x1, y1))
                    if compute_points:
                        # self.get_logger().info(f'depth: {depth.shape}, mask: {mask.shape}')
                        curr_mask = np.zeros(depth.shape, dtype=np.uint8)
                        curr_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1].copy()
                        p_gb = self.point_towards_box(depth, (x0, y0, win_w, win_h), mask=curr_mask)
                        points_gb.append(p_gb)
                    counter += 1


        return boxes, points_gb

    def _publish_sweep_visualization(self, wp_counter, tracked_bbox=None):
        if self.mask_latest is None:
            return

        vis = cv2.cvtColor(self.mask_latest, cv2.COLOR_GRAY2BGR)

        for idx, (x0, y0, x1, y1) in enumerate(self.wins):
            if idx != wp_counter:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)

        if 0 <= wp_counter < len(self.wins):
            x0, y0, x1, y1 = self.wins[wp_counter]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

        if tracked_bbox is not None:
            tx, ty, tw, th = tracked_bbox
            cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2)

        if self.downsample_visualization:
            vis_out = cv2.resize(vis, (self.vis_output_width, self.vis_output_height))
        else:
            vis_out = vis

        patched_msg = self.bridge.cv2_to_imgmsg(vis_out, encoding='bgr8')
        patched_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_publisher.publish(patched_msg)
    
    def publish_orientation(self, p_gb, scan_stamp, noise=0.0):

        if p_gb is None: 
            return
        
        x,y,z = p_gb
        yaw    = np.arctan2(y, x) + np.random.uniform(-noise, noise) * np.pi / 180
        pitch  = np.arctan2(-z, np.hypot(x,y)) + np.random.uniform(-noise, noise) * np.pi / 180
        roll   = 0.0

        # delta = self.get_clock().now() - Time.from_msg(scan_stamp)
        # self.get_logger().info(f"Image delay: {delta.nanoseconds * 1e-9:.3f} s")

        # publish the desired gimbal angle
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        q_msg = QuaternionStamped()
        q_msg.header.stamp = self.get_clock().now().to_msg()      
        q_msg.header.frame_id = 'gimbal_base'      
        q_msg.quaternion.x = qx
        q_msg.quaternion.y = qy
        q_msg.quaternion.z = qz
        q_msg.quaternion.w = qw
        self.gimbal_publisher.publish(q_msg)

    def point_towards_box(self, depth, box, mask=None):

        # center of the target box
        cx, cy = int(box[0]+box[2]/2), int(box[1]+box[3]/2)

        # depth around center
        h,w = depth.shape
        x0,y0 = max(0,cx-2), max(0,cy-2)
        x1,y1 = min(w,cx+3), min(h,cy+3)
        if mask is None:
            vals   = depth[y0:y1, x0:x1]
        else:
            vals = depth[mask > 0]
            # self.get_logger().info(f'min: {np.min(mask)}, max: {np.max(mask)}')

        vals   = vals[~np.isnan(vals)]
        if vals.size==0:
            # self.get_logger().info('No depth at object')
            return None
        depth_m = float(np.mean(vals)) 
        
        if not np.isfinite(depth_m) :
            # self.get_logger().info('No depth at object')
            return None

        # back-project to 3D in nav frame
        pts     = np.array([[[cx,cy]]], dtype=np.float32)
        # if self.img_type == "raw":
        #     und = cv2.undistortPoints(pts, self.nav_K, self.nav_D)
        # else:
        und = cv2.undistortPoints(pts, self.nav_K, np.zeros((5,), dtype=np.float64))

        x_n, y_n= und[0,0]
        vec_cam = np.array([x_n, y_n, 1.0])
        pos_nav = vec_cam * depth_m

        # transform to gimbal-base frame and calculate gimbal angles to that points gimbal towards target
        R_yaw = R.from_euler('zyx', [self.yaw_offset, self.pitch_offset, 0], degrees=True).as_matrix()
        p_gb = R_yaw @ self.T_nav_gb[:3,:3] @ pos_nav + self.T_nav_gb[:3,3]

        return p_gb

        
def main(args=None):
    rclpy.init(args=args)
    node = GimbalTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
