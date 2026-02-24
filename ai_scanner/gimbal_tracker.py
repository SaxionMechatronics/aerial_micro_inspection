#!/usr/bin/env python3
import sys
import cv2
import time
import numpy as np
import threading
import yaml
import os
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf_transformations import quaternion_from_euler
from sensor_msgs.msg import Image
from geometry_msgs.msg import QuaternionStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from ai_scanner_interfaces.msg import ObjectDetectionResult
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


from ai_scanner.utils import load_yaml, quaternion_to_rotmat

## TODO: 
##       Publish the tf for all frames
##       Fix launch file
##       Push code to bitbucket
##       Fix the wrong scale of position in the calibration
##       Increase the accuracy of calibration with iterative methods
##       Think of a scanning method
##       Literature review
##       Confluence documentation


class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        self.bridge = CvBridge()

        # ROS Parameters
        self.declare_parameter('mission_config_file', '')
        self.declare_parameter('configs_dir', '')
        param_file = self.get_parameter('mission_config_file').get_parameter_value().string_value
        with open(param_file, 'r') as f:
            self.config = yaml.safe_load(f)
        configs_dir  = self.get_parameter('configs_dir').value

        # Load configurations
        det_t  = self.config['output']['detection_topic']
        pln_img_t  = self.config['output']['gimbal_planning_img_topic']
        nav_img_t  = self.config['input']['nav_image_topic']
        ai_img_t   = self.config['input']['ai_image_topic']
        depth_t    = self.config['input']['depth_image_topic']
        states_t   = self.config['input']['drone_states_topic']
        gimbal_t   = self.config['input']['gimbal_orientation_topic']
        des_ori_t   = self.config['output']['gimbal_setpoint']
        calib_p = os.path.join(configs_dir, self.config['input']['pan_tilt_calibration'])
        self.planning_mode = self.config['planning']['mode']
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
        self.nav_h = np.array(calib['nav_camera']['image_height'])
        self.nav_w = np.array(calib['nav_camera']['image_width'])

        best_effort_qos = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # synchronized subscribers: detection results, nav depth, ai img, gimbal
        sub_det      = Subscriber(self, ObjectDetectionResult, det_t)
        sub_depth    = Subscriber(self, Image,                 depth_t)
        # sub_ai_image = Subscriber(self, Image,                 ai_img_t)
        sub_gim      = Subscriber(self, QuaternionStamped,     gimbal_t)
        # states_sub      = Subscriber(self, Odometry,     states_t, qos_profile=best_effort_qos)
        # subs_list = [sub_det, sub_depth, sub_ai_image, sub_gim]
        subs_list = [sub_det, sub_depth, sub_gim]

        callback_function = self.cb_synced

        # If realtime-correction is needed, also subscribe on nav image.
        if self.rt_correction:
            sub_nav_image = Subscriber(self, Image, nav_img_t)
            subs_list.append(sub_nav_image)
            callback_function = self.cb_synced_rt

        self.sync = ApproximateTimeSynchronizer(
            subs_list,
            queue_size=30,
            slop=0.1
        ) 
        self.sync.registerCallback(callback_function)

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/gimbal_adjustment',
            self.gimbal_adjusments_callback,
            10  
        )

        self.gimbal_publisher = self.create_publisher(QuaternionStamped, des_ori_t, 10)
        self.image_publisher = self.create_publisher(Image, pln_img_t, 10)

        self.executing_trajectory = False
        self.mask_gb = None
        self.ps_gb = []
        self.det_stamp = None
        self.detection_updated = False

        self.get_logger().info(f'ObjectTracker ready with {calib_p} config file.')


    def gimbal_adjusments_callback(self, msg):
        self.pitch_offset += msg.data[0]
        self.yaw_offset += msg.data[1]
        
    def cb_synced(self, det_msg, depth_msg, gimbal_stamped):
        self.process(det_msg, depth_msg, gimbal_stamped)

    def cb_synced_rt(self, det_msg, depth_msg, gimbal_stamped, nav_msg):
        self.process(det_msg, depth_msg, gimbal_stamped, nav_msg=nav_msg)

    def process(self, det_msg, depth_msg, gimbal_stamped, nav_msg=None):
        # decode messages
        # ai_img = self.bridge.imgmsg_to_cv2(ai_msg,   'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        if not (depth.shape[1] == self.nav_w and depth.shape[0] == self.nav_h and \
               det_msg.img_w == self.nav_w   and det_msg.img_h == self.nav_h):
            raise ValueError(f"The depth and are RGB image (source for object detection) size should \
                               match the calibrated camera image size.")
                
        # if self.vis_enabled:
        #     # The transformation from gimbal's end point to the gimbal's base frame.
        #     # The base frame is assumed to be attached to the robot's body and the gimbal orientation message
        #     # should contain gimbal's orientation w.r.t the base.
        #     gim_q = gimbal_stamped.quaternion
        #     Rg = quaternion_to_rotmat(gim_q)
        #     T_gimbal_to_gimbalbase = np.eye(4); T_gimbal_to_gimbalbase[:3,:3]=Rg

        #     # Transformation to take from camera coordinates to gimbal base
        #     T_ai_to_gimbal = np.array([[0, 0, 1, 0],
        #                             [1, 0, 0, 0],
        #                             [0, 1, 0, 0],
        #                             [0, 0, 0, 1]])
        #     T_ai_to_gimbalbase = T_gimbal_to_gimbalbase.dot(T_ai_to_gimbal)
        #     # project the detected object on gimbaled camera image
        #     gimbal_front = np.array([
        #         [x,y,z],
        #     ], dtype=np.float32)
        #     T_gimbalbase_to_ai = np.linalg.inv(T_ai_to_gimbalbase)
        #     pts4 = T_gimbalbase_to_ai[:3,:3] @ gimbal_front.T + T_gimbalbase_to_ai[:3,3:4] 
        #     uv, _ = cv2.projectPoints(
        #         pts4.T, np.zeros(3), np.zeros(3), self.ai_K, self.ai_D
        #     ) 
        #     uv = uv.reshape(-1,2).astype(int) 
        #     # cv2.polylines(ai_img, [uv.reshape(-1,1,2)], True, (0,255,0), 2)
        #     for i, p in enumerate(uv):
        #         cv2.circle(ai_img, p, (len(uv) - i)*5, (0,255,0), -1)
        #         # cv2.rectangle(ai_img, (int(p[0]-w_ai/2), int(p[1]-h_ai/2)), (int(p[0]+w_ai/2), int(p[1]+h_ai/2)), (0,0,255), 2)

        #     # annotate
        #     cv2.imshow('AI', ai_img)
        #     cv2.waitKey(1)

        if self.planning_mode == "sweeping":

            # self.last_odom = odom_msg
            
            win = self.get_ai_image_size_on_nav_image(self.ai_img_shape)
            mask = self.bridge.imgmsg_to_cv2(det_msg.mask, 'mono8')

            self.wins, self.ps_gb = self.sweep(depth, mask, win, overlap=0.5, threshold=0.3)
            # self.ps_gb = self.transform_to_map(self.ps_gb, odom_msg)
            self.mask_gb = mask.copy()

            self.det_stamp = det_msg.header.stamp

            if self.executing_trajectory:
                self.detection_updated = True
                return

            t = threading.Thread(target=self.execute_gimbal_trajectory)
            t.start()

            # cv2.imshow("Bottom→Top Boxes", vis)
            # cv2.waitKey(0)

        elif self.planning_mode == "pitch_yaw_lock":
            p_gb = self.point_towards_box(depth, (det_msg.x, det_msg.y, det_msg.w, det_msg.h))            
            if p_gb is None:
                return 
            self.publish_orientation(p_gb, det_msg.header.stamp)

        elif self.planning_mode == "pitch_lock":

            win_w, _ = self.get_ai_image_size_on_nav_image(self.ai_img_shape)
            win_w = 2*win_w
            mask = self.bridge.imgmsg_to_cv2(det_msg.mask, 'mono8')

            # Keep the heading aligned with body
            p_gb = self.point_towards_box(depth, self.largest_center_strip_bbox(mask, win_w))
            
            if p_gb is None:
                return 
            self.publish_orientation(p_gb, det_msg.header.stamp)

        # if nav_msg is not None:
        #     nav_img = self.bridge.imgmsg_to_cv2(nav_msg, 'bgr8')

        #     proj_w, proj_h = self.get_ai_image_size_on_nav_image(ai_img.shape[:2])

        #     search_w = int(proj_w * 2)
        #     search_h = int(proj_h * 2)

        #     # self.get_logger().info(f"Search zone size: {(search_w, search_h)}")

        #     # Use detection center (cx, cy) as center
        #     center_x, center_y = cx, cy
        #     x1 = max(0, center_x - search_w // 2)
        #     y1 = max(0, center_y - search_h // 2)
        #     x2 = min(nav_img.shape[1], center_x + search_w // 2)
        #     y2 = min(nav_img.shape[0], center_y + search_h // 2)

        #     # search_zone = nav_img[y1:y2, x1:x2]

        #     # # Resize ai_img to estimated projection size
        #     ai_img_resized = cv2.resize(ai_img, (proj_w, proj_h))

        #     # # --- Match resized AI image inside NAV search zone ---
        #     # if search_zone.shape[0] < ai_img_resized.shape[0] or search_zone.shape[1] < ai_img_resized.shape[1]:
        #     #     self.get_logger().warn("Search zone is smaller than AI image, skipping matching")
        #     #     return

        #     # # Convert to grayscale for matchTemplate
        #     # gray_ai   = cv2.cvtColor(ai_img_resized, cv2.COLOR_BGR2GRAY)
        #     # gray_nav  = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)

        #     # result = cv2.matchTemplate(gray_nav, gray_ai, cv2.TM_CCOEFF_NORMED)
        #     # _, max_val, _, max_loc = cv2.minMaxLoc(result)

        #     # match_top_left = (max_loc[0] + x1, max_loc[1] + y1)

        #     # --- Paste resized AI image directly into nav image ---
        #     blended = nav_img.copy()
        #     mh, mw = ai_img_resized.shape[:2]
        #     match_top_left = (center_x + x1, center_y + y1)
        #     x, y = match_top_left

        #     # Ensure bounds
        #     x_end = min(x + mw, blended.shape[1])
        #     y_end = min(y + mh, blended.shape[0])
        #     roi = blended[y:y_end, x:x_end]
        #     ai_crop = ai_img_resized[:y_end - y, :x_end - x]

        #     # Blend with transparency
        #     alpha = 0.8
        #     blended[y:y_end, x:x_end] = ai_crop
        #     cv2.rectangle(blended, (x, y), (x + mw, y + mh), color=(0, 255, 0), thickness=2)


        #     # --- Publish result ---
        #     patched_msg = self.bridge.cv2_to_imgmsg(blended, encoding='bgr8')
        #     patched_msg.header.stamp = self.get_clock().now().to_msg()
        #     patched_msg.header.frame_id = 'camera_link'
        #     self.image_publisher.publish(patched_msg)

        #     # Store transform (only translation in this case)
        #     self.last_transform = np.array([
        #         [1.0, 0.0, float(x)],
        #         [0.0, 1.0, float(y)]
        #     ], dtype=np.float32)

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
          depth: np.ndarray,
          mask: np.ndarray,
          window_size: tuple[int,int],
          overlap: float = 0.1,
          threshold: float = 0.7,
          visualize: bool = False
         ) -> list[tuple[int,int,int,int]]:
        """
        Slide a window over the white area of a binary mask, bottom→top, keeping
        only windows at least `threshold` full of white.

        Args:
        mask:        H×W mono8 image (0 or 255) where 255 is “white”.
        window_size: (width, height) of the sliding box in pixels.
        overlap:     Fractional overlap between adjacent windows [0..1).
        threshold:   Fraction (0..1) of pixels that must be white to keep the box.

        Returns:
        List of (x0,y0,x1,y1) windows, iterated bottom→top, left→right.
        """
        win_w, win_h = window_size
        area = win_w * win_h

        # 1) Find tight white‐pixel bbox
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return [], []
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min() - int(win_h/2), ys.max() + int(win_h/2)

        # 2) Compute step sizes
        step_x = max(1, int(win_w * (1.0 - overlap)))
        step_y = max(1, int(win_h * (1.0 - overlap)))

        # 3) Precompute all x0 positions (left→right)
        x0_list = []
        for x in range(x_min, x_max + 1, step_x):
            x0 = min(x, x_max - win_w)
            x0 = max(0, x0)
            x0_list.append(x0)
        # remove duplicates but keep order
        x0_list = list(dict.fromkeys(x0_list))

        # 4) Precompute all y0 positions (top→bottom), then reverse to bottom→top
        y0_list = []
        for y in range(y_min, y_max + 1, step_y):
            y0 = min(y, y_max - win_h)
            y0 = max(0, y0)
            y0_list.append(y0)
        y0_list = list(dict.fromkeys(y0_list))[::-1]

        points_gb = []
        boxes = []
        counter = 1
        # cv2.imwrite(f"/home/sarax/Desktop/debug/{0}.png", mask)
        # 5) Sweep bottom→top, left→right
        # self.get_logger().info(f'\n\n')
        for y0 in y0_list:
            y1 = y0 + win_h
            for x0 in x0_list:
                x1 = x0 + win_w
                # count white pixels in this window
                count_white = np.count_nonzero(mask[y0:y1, x0:x1])
                if count_white >= threshold * area:
                    boxes.append((x0, y0, x1, y1))
                    # self.get_logger().info(f'depth: {depth.shape}, mask: {mask.shape}')
                    curr_mask = np.zeros(depth.shape, dtype=np.uint8)
                    curr_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1].copy()
                    # cv2.imwrite(f"/home/sarax/Desktop/debug/{counter}.png", curr_mask)
                    p_gb = self.point_towards_box(depth, (x0, y0, win_w, win_h), mask=curr_mask)
                    
                    points_gb.append(p_gb)
                    counter += 1


        return boxes, points_gb
    
    def execute_gimbal_trajectory(self, wait_time=2.0):

        self.executing_trajectory = True
        wp_counter = 0
        end_counter = 0
        if len(self.ps_gb) == 0:
            return
        
        prev_box_time = time.time()
        while True:
            if wp_counter >= len(self.ps_gb):
                break
                # end_counter += 1
                # wp_counter = len(self.ps_gb) - 2

            p_gb = self.ps_gb[wp_counter]
            # p_gb = self.inverse_transform_point(p_gb, self.last_odom)

            # Visualize the segment breaking.
            if self.mask_gb is not None:
                vis = cv2.cvtColor(self.mask_gb, cv2.COLOR_GRAY2BGR)
                # Draw all non-active windows first, then draw active window last to keep it on top.
                for idx, (x0, y0, x1, y1) in enumerate(self.wins):
                    if idx != wp_counter:
                        cv2.rectangle(vis, (x0, y0), (x1, y1), (0,0,255), 2)

                if 0 <= wp_counter < len(self.wins):
                    x0, y0, x1, y1 = self.wins[wp_counter]
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0,255,0), 2)

                if self.downsample_visualization:
                    vis_out = cv2.resize(vis, (self.vis_output_width, self.vis_output_height))
                else:
                    vis_out = vis

                patched_msg = self.bridge.cv2_to_imgmsg(vis_out, encoding='bgr8')
                patched_msg.header.stamp = self.get_clock().now().to_msg()
                self.image_publisher.publish(patched_msg)

            interrupted = False
            while time.time() - prev_box_time <= wait_time:

                if self.detection_updated:
                    self.detection_updated = False
                    interrupted = True
                    break
                
                self.publish_orientation(p_gb, self.det_stamp)
                time.sleep(0.03)
            if not interrupted:
                prev_box_time = time.time()

                wp_counter += 1

            # if end_counter > 2:
            #     break

        self.executing_trajectory = False
        

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
    node = ObjectTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
