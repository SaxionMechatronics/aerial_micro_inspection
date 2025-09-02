#!/usr/bin/env python3
import os
import sys
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import QuaternionStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from scipy.spatial.transform import Rotation as R

def load_yaml(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def quaternion_to_rotmat(q):
    x,y,z,w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

class DualCameraApril(Node):
    def __init__(self):
        super().__init__('dual_camera_april')
        self.bridge = CvBridge()

        # params
        self.declare_parameter('ai_image_topic', '/camera/image_raw')
        self.declare_parameter('ai_camera_info_yaml', 'config/sony/ost.yaml')
        self.declare_parameter('ai_camera_info_topic', '')

        self.declare_parameter('nav_image_topic', '/zed/zed_node/right_raw/image_raw_color')
        self.declare_parameter('nav_camera_info_yaml', '')
        self.declare_parameter('nav_camera_info_topic', '/zed/zed_node/right_raw/camera_info')

        self.declare_parameter('aruco_yaml', 'config/aruco.yaml')
        self.declare_parameter('gimbal_topic', '/gimbal_orientation')
        self.declare_parameter('visualize', False)
        self.declare_parameter('sync_slop', 0.2)  # seconds tolerance
        

        # fetch
        ai_img_t   = self.get_parameter('ai_image_topic').value
        ai_yaml_p  = self.get_parameter('ai_camera_info_yaml').value
        ai_info_t  = self.get_parameter('ai_camera_info_topic').value

        nav_img_t  = self.get_parameter('nav_image_topic').value
        nav_yaml_p = self.get_parameter('nav_camera_info_yaml').value
        nav_info_t = self.get_parameter('nav_camera_info_topic').value

        aruco_p    = self.get_parameter('aruco_yaml').value
        gimbal_t   = self.get_parameter('gimbal_topic').value

        self.visualize = self.get_parameter('visualize').value
        self.slop      = self.get_parameter('sync_slop').value

        # load
        self.ai_yaml    = load_yaml(ai_yaml_p)
        self.nav_yaml   = load_yaml(nav_yaml_p)
        self.aruco_cfg = load_yaml(aruco_p)


        # validate
        if not self.ai_yaml and not ai_info_t:
            self.get_logger().error("ai_camera needs ai_camera_info_yaml or ai_camera_info_topic")
            rclpy.shutdown(); sys.exit(1)
        if not self.nav_yaml and not nav_info_t:
            self.get_logger().error("nav_camera needs nav_camera_info_yaml or nav_camera_info_topic")
            rclpy.shutdown(); sys.exit(1)
        if not self.aruco_cfg:
            self.get_logger().error(f"Could not load ArUco config: {aruco_p}")
            sys.exit(1)

        self.use_nav_info = bool(nav_info_t)

        dict_name       = self.aruco_cfg['dictionary']
        self.marker_id  = int(self.aruco_cfg['marker_id'])
        self.marker_size= float(self.aruco_cfg['marker_size'])

        # Initialize the opencv's aruco module based on the opencv version
        self.opencv_minor_version = int(cv2.__version__.split('.')[1])
        if self.opencv_minor_version >= 7:
            aruco_dict_id    = getattr(cv2.aruco, dict_name)
            self.aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.aruco_params
            )
        else:
            aruco_dict_id  = getattr(cv2.aruco, dict_name)
            self.aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        # synchronized subscribers
        sub_ai    = Subscriber(self, Image,            ai_img_t)
        sub_nav   = Subscriber(self, Image,            nav_img_t)
        sub_gimbal= Subscriber(self, QuaternionStamped, gimbal_t)

        self.sync = ApproximateTimeSynchronizer(
            [sub_ai, sub_nav, sub_gimbal],
            queue_size=50,
            slop=self.slop
        )
        self.sync.registerCallback(self.cb_synced)

        # nav camera info
        self.last_nav_info = None
        if self.use_nav_info:
            self.create_subscription(CameraInfo, nav_info_t, self._nav_info_cb, 10)

        # visualization
        if self.visualize:
            cv2.namedWindow('AI Camera',  cv2.WINDOW_NORMAL)
            cv2.namedWindow('Nav Camera', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"DualCameraApril ready (slop={self.slop}s, visualize={self.visualize})"
        )

        self.transforms = []

    # def destroy_node(self):
    #     # before shutdown, compute & save
    #     if self.transforms:
    #         avg_T = self._compute_average_transform(self.transforms)
    #         self._save_yaml(avg_T, filename='cameras_calib.yaml')
    #     super().destroy_node()

    def _compute_average_transform(self, Ts):
        # Ts: list of 4×4 numpy arrays
        # average translation
        translations = np.stack([T[:3,3] for T in Ts], axis=0)
        mean_t       = translations.mean(axis=0)

        # average rotation via SVD
        R_sum = np.zeros((3,3))
        for T in Ts:
            R_sum += T[:3,:3]
        U, _, Vt = np.linalg.svd(R_sum)
        R_avg = U @ Vt
        # ensure a proper rotation (det=+1)
        if np.linalg.det(R_avg) < 0:
            U[:,-1] *= -1
            R_avg = U @ Vt

        # build avg_T
        avg_T = np.eye(4)
        avg_T[:3,:3] = R_avg
        avg_T[:3,3]  = mean_t
        return avg_T
    
    def _save_yaml(self, avg_T, filename='camera_calib.yaml'):
        def reshape_and_cast(k_flat):
            """Turn length-9 iterable into 3×3 list of Python floats."""
            flat = [float(x) for x in k_flat]
            return [
                flat[0:3],
                flat[3:6],
                flat[6:9],
            ]
        def cast_list(lst):
            """Cast any iterable of numbers to a list of Python floats."""
            return [float(x) for x in lst]
        
        def extract_cam_params(yaml_cfg, last_info, topic_param):
            topic = self.get_parameter(topic_param).value
            if yaml_cfg:
                return {
                    'topic_name':  topic,
                    'image_width':  yaml_cfg['image_width'],
                    'image_height': yaml_cfg['image_height'],
                    'camera_matrix': yaml_cfg['camera_matrix']['data'],
                    'distortion':    yaml_cfg['distortion_coefficients']['data'],
                }
            elif last_info:
                # print("\n\n K Matrix: ", last_info.k, "\n", list(last_info.k), "\n\n")
                return {
                    'topic_name':  topic,
                    'image_width':  last_info.width,
                    'image_height': last_info.height,
                    'camera_matrix': reshape_and_cast(last_info.k),
                    'distortion':    list(last_info.d),
                }
            else:
                return {
                    'topic_name': topic,
                    'error': 'no calibration available'
                }

        out = {
            # 'transforms': [
            #     T.flatten().tolist() for T in self.transforms
            # ],
            'average_transform_gimbalbase_to_nav': avg_T.flatten().tolist(),
            'ai_camera': extract_cam_params(
                getattr(self, 'ai_yaml', None),
                getattr(self, 'last_ai_info', None),
                'ai_image_topic'
            ),
            'nav_camera': extract_cam_params(
                getattr(self, 'nav_yaml', None),
                getattr(self, 'last_nav_info', None),
                'nav_image_topic'
            )
        }

        with open(filename, 'w') as f:
            yaml.dump(out, f, sort_keys=False)

        # self.get_logger().info(
        #     f"Saved {len(self.transforms)} raw transforms "
        #     f"+ average to {filename}"
        # )

    def _nav_info_cb(self, info: CameraInfo):
        self.last_nav_info = info

    def cb_synced(self, ai_img_msg, nav_img_msg, gimbal_stamped):
        # convert ROS→CV
        img_ai  = self.bridge.imgmsg_to_cv2(ai_img_msg,  'bgr8')
        img_nav = self.bridge.imgmsg_to_cv2(nav_img_msg, 'bgr8')
        gim_q   = gimbal_stamped.quaternion

        # camera intrinsics
        K1 = np.array(self.ai_yaml['camera_matrix']['data']).reshape(3,3)
        D1 = np.array(self.ai_yaml['distortion_coefficients']['data'])
        if self.use_nav_info and self.last_nav_info:
            K2 = np.array(self.last_nav_info.k).reshape(3,3)
            D2 = np.array(self.last_nav_info.d)
        else:
            return
            K2 = np.array(self.nav_yaml['camera_matrix']['data']).reshape(3,3)
            D2 = np.array(self.nav_yaml['distortion_coefficients']['data'])


        # AI camera
        r1, t1, ok1, obj_c = self.detect_and_draw_aruco(img_ai, K1, D1)
        if ok1:
            cv2.drawFrameAxes(img_ai, K1, D1, r1, t1, self.aruco_cfg['marker_size'])

        # Nav camera
        r2, t2, ok2, _ = self.detect_and_draw_aruco(img_nav, K2, D2)
        if ok2:
            cv2.drawFrameAxes(img_nav, K2, D2, r2, t2, self.aruco_cfg['marker_size'])

        # show both
        # cv2.imshow('AI Camera',  img_ai)
        # cv2.imshow('Nav Camera', img_nav)
        # cv2.waitKey(1)

        # if either had no pose, skip transform
        if not ok1 or not ok2:
            return

        # compute inter-camera transform
        # build full 4×4 from rvec/tvec
        def to_T(rvec, tvec):
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.flatten()
            return T

        # The outputs of the SolvePnP alway show the target in camera frame.
        # In other words, they show how to move and rotate camera to get to target frame.
        # Thus, the transformations calculated from these values will transform a point from target frame to camera frame.
        T_target_to_ai = to_T(r1, t1)
        T_target_to_nav = to_T(r2, t2)

        # For localization purposes, the following transform is more useful where it shows how to get from target coordinates to camera pose.
        # i.e, it shows camera pose in target frame.
        T_ai_to_target = np.linalg.inv(T_target_to_ai)

        # Direct transformation between the two cameras at this moment. This transformation depends on the gimbal orientation.
        T_ai_to_nav = T_target_to_nav.dot(np.linalg.inv(T_target_to_ai))
        # Transformation from camera to the gimbal's endpoint frame.
        T_ai_to_gimbal = np.array([[0, 0, 1, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]])
        # T_ai_to_gimbal = np.linalg.inv(T_ai_to_gimbal)
        # T_ai_to_nav = np.linalg.inv(T_ai_to_nav)
        
        # The transformation from gimbal's end point to the gimbal's base frame.
        # The base frame is assumed to be attached to the robot's body and the gimbal orientation message
        # should contain gimbal's orientation w.r.t the base.
        Rg = quaternion_to_rotmat(gim_q)
        T_gimbal_to_gimbalbase = np.eye(4); T_gimbal_to_gimbalbase[:3,:3]=Rg

        # Transformation to take from camera coordinates to gimbal base
        T_ai_to_gimbalbase = T_gimbal_to_gimbalbase.dot(T_ai_to_gimbal)

        # Final transformation from gimbal base to nav camera
        final_T = T_ai_to_nav.dot(np.linalg.inv(T_ai_to_gimbalbase))

        eul_1 = R.from_matrix(T_ai_to_nav[:3,:3]).as_euler('zyx', degrees=True)
        eul_2 = R.from_matrix(final_T[:3,:3]).as_euler('zyx', degrees=True)
        eul_3 = R.from_matrix(Rg).as_euler('zyx', degrees=True)
        eul_4 = R.from_matrix(T_ai_to_gimbalbase[:3,:3]).as_euler('zyx', degrees=True)
        # eul_5 = R.from_matrix(T_ai_to_gimbal[:3,:3]).as_euler('zyx', degrees=True)
        self.get_logger().info(f"T(ai → nav):\n{eul_1}\n{eul_2}\n{eul_3}")


        self.transforms.append(final_T)
        self.get_logger().info("Recorded transform #%d" % len(self.transforms))

        avg_T = final_T
        if len(self.transforms) > 0:
            avg_T = self._compute_average_transform(self.transforms)
            self._save_yaml(avg_T, filename='cameras_calib.yaml')

        # project AI marker corners into Nav image
        pts1 = (T_target_to_ai[:3,:3] @ obj_c.T) + T_target_to_ai[:3,3:4]   # shape (3,4)
        T_est = avg_T.dot(T_ai_to_gimbalbase)
        pts2 = T_est[:3,:3] @ pts1 + T_est[:3,3:4]  # (3,4)
        uv, _ = cv2.projectPoints(
            pts2.T, np.zeros(3), np.zeros(3), K2, D2
        ) 
        uv = uv.reshape(-1,2).astype(int)  
        cv2.polylines(img_nav, [uv.reshape(-1,1,2)], True, (0,0,255), 2)

        # project gimbal base's front axis on the nav image to approximately the correctness of calibration.
        gimbal_front = np.array([
            [0.8, 0, 0],
            [1.2, 0, 0],
            [1.6, 0, 0],
            [2.0, 0, 0],
        ], dtype=np.float32)
        pts3 = avg_T[:3,:3] @ gimbal_front.T + avg_T[:3,3:4] 
        uv, _ = cv2.projectPoints(
            pts3.T, np.zeros(3), np.zeros(3), K2, D2
        ) 
        uv = uv.reshape(-1,2).astype(int) 
        cv2.polylines(img_nav, [uv.reshape(-1,1,2)], True, (0,255,0), 2)
        for i, p in enumerate(uv):
            cv2.circle(img_nav, p, (len(uv) - i)*5, (0,255,0), -1)

        # project gimbal base's front axis on the ai image to check corrected of the ai cam to gimbal transformation.
        ang = 0*3.1415/180
        gimbal_front = np.array([
            [0.2*np.cos(ang), 0.2*np.sin(ang), 0.0],
            [0.6*np.cos(ang), 0.6*np.sin(ang), -0.1],
            [1.0*np.cos(ang), 1.0*np.sin(ang), -0.2],
            [1.4*np.cos(ang), 1.4*np.sin(ang), -0.3],
        ], dtype=np.float32)
        T_gimbalbase_to_ai = np.linalg.inv(T_ai_to_gimbalbase)
        pts4 = T_gimbalbase_to_ai[:3,:3] @ gimbal_front.T + T_gimbalbase_to_ai[:3,3:4] 
        uv, _ = cv2.projectPoints(
            pts4.T, np.zeros(3), np.zeros(3), K1, D1
        ) 
        uv = uv.reshape(-1,2).astype(int) 
        cv2.polylines(img_ai, [uv.reshape(-1,1,2)], True, (0,255,0), 2)
        for i, p in enumerate(uv):
            cv2.circle(img_ai, p, (len(uv) - i)*5, (0,255,0), -1)

        # refresh window
        cv2.imshow('AI Camera',  img_ai)
        cv2.imshow('Nav Camera', img_nav)
        cv2.waitKey(1)


    def detect_and_draw_aruco(self, img, K, D):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.opencv_minor_version >= 7:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, 
                                                      self.aruco_dict, 
                                                      parameters=self.aruco_params
                                                     )
        if ids is None or self.marker_id not in ids.flatten():
            cv2.putText(img, "No detection", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            return None, None, False, None

        # draw all actual detections
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        idx = list(ids.flatten()).index(self.marker_id)
        img_c = corners[idx].reshape(4,2)  # (4,2) image corners
        # define 3D object corners (marker-frame), same order:
        s = self.marker_size
        obj_c = np.array([
            [-s/2, -s/2, 0],
            [ s/2, -s/2, 0],
            [ s/2,  s/2, 0],
            [-s/2,  s/2, 0]
        ], dtype=np.float32)

        # solvePnP to get pose of marker in camera frame
        ok, rvec, tvec = cv2.solvePnP(obj_c, img_c, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            cv2.putText(img, "Pose failed", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            return None, None, False, None

        # draw axes
        cv2.drawFrameAxes(img, K, D, rvec, tvec, s*0.5)
        return rvec, tvec, True, obj_c

def main(args=None):
    rclpy.init(args=args)
    node = DualCameraApril()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.visualize:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
