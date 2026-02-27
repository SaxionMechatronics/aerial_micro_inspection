#!/usr/bin/env python3
import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ai_scanner_interfaces.msg import ObjectDetectionResult
from cv_bridge import CvBridge
from rclpy.duration import Duration

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.bridge = CvBridge()

        self.declare_parameter('mission_config_file', 'mission_config_file')
        param_file = self.get_parameter('mission_config_file').value
        with open(param_file, 'r') as f:
            self.config = yaml.safe_load(f)

        color_cfg = self.config.get('color_segmentation', self.config.get('detection', {}))

        input_cfg = self.config.get('input', {})
        output_cfg = self.config.get('output', {})
        img_t  = input_cfg.get('nav_rgb_topic', input_cfg.get('nav_image_topic'))
        det_t  = output_cfg.get('surface_segmentation_topic', output_cfg.get('detection_topic'))
        self.vis_enabled  = bool(self.config['visualization'])
        self.min_size = color_cfg['min_size']
        self.img_type = self.config['input']['img_type']
        vis_t = output_cfg.get('surface_segmentation_vis_topic', output_cfg.get('detection_vis_topic'))
        if not self.img_type in ['raw', 'rect']:
            raise ValueError("Parameter img_type can either be \'raw\' or \'rect\'.")

        self.low_h = color_cfg['h_min'];  self.high_h = color_cfg['h_max']
        self.low_s = color_cfg['s_min'];  self.high_s = color_cfg['s_max']
        self.low_v = color_cfg['v_min'];  self.high_v = color_cfg['v_max']

        # create display windows + trackbars
        # cv2.namedWindow('Detection',   cv2.WINDOW_NORMAL)
        # cv2.createTrackbar('LowH','Detection', self.low_h, 180, lambda v: setattr(self, 'low_h', v))
        # cv2.createTrackbar('HighH','Detection',self.high_h,180, lambda v: setattr(self, 'high_h', v))
        # cv2.createTrackbar('LowS','Detection', self.low_s, 255, lambda v: setattr(self, 'low_s', v))
        # cv2.createTrackbar('HighS','Detection',self.high_s,255, lambda v: setattr(self, 'high_s', v))
        # cv2.createTrackbar('LowV','Detection', self.low_v, 255, lambda v: setattr(self, 'low_v', v))
        # cv2.createTrackbar('HighV','Detection',self.high_v,255, lambda v: setattr(self, 'high_v', v))
        
        if self.vis_enabled:
            self.vis_publisher = self.create_publisher(Image, vis_t, 10)

        self.subscription = self.create_subscription(
            Image,
            img_t,
            self.img_cb,
            10)
        self.det_publisher = self.create_publisher(ObjectDetectionResult, det_t, 10)

        self._last_time = self.get_clock().now() - Duration(seconds=1.0)

        self.get_logger().info('ColorDetector ready')


    def visualize(self, img, mask):
        vis_img = cv2.vconcat([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow('Detection', vis_img)
        cv2.waitKey(1)

    def publish_detection_image(self, img, mask, stamp):
        vis_img = cv2.vconcat([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        img_resized = cv2.resize(vis_img, (240, 160))
        img_msg = self.bridge.cv2_to_imgmsg(img_resized, encoding='bgr8')
        img_msg.header.stamp = stamp
        self.vis_publisher.publish(img_msg)
        

    def img_cb(self, img_msg):

        # No faster than 10 Hz
        now = self.get_clock().now()
        if now - self._last_time < Duration(seconds=0.1):
            return
        self._last_time = now
        
        img   = self.bridge.imgmsg_to_cv2(img_msg,   'bgr8')


        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = (self.low_h, self.low_s, self.low_v)
        upper = (self.high_h,self.high_s,self.high_v)
        mask  = cv2.inRange(hsv, lower, upper)

        # find contours
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            # self.get_logger().info('No colored object detected')
            # self.visualize(img, mask)
            if self.vis_enabled:
                self.publish_detection_image(img, mask, img_msg.header.stamp)
            return

        # center of largest contour
        cnt = max(cnts, key=cv2.contourArea)
        M   = cv2.moments(cnt)
        rect = cv2.boundingRect(cnt)
        if M['m00']==0 or rect[2] < self.min_size or rect[3] < self.min_size:
            # self.visualize(img, mask)
            if self.vis_enabled:
                self.publish_detection_image(img, mask, img_msg.header.stamp)
            return

        det_msg = ObjectDetectionResult()
        det_msg.header.stamp = img_msg.header.stamp
        det_msg.mask = self.bridge.cv2_to_imgmsg(mask, 'mono8')
        det_msg.x = rect[0]; det_msg.y = rect[1]; det_msg.w = rect[2]; det_msg.h = rect[3]
        det_msg.img_w = img.shape[1]; det_msg.img_h = img.shape[0]
        det_msg.src_type = self.img_type
        self.det_publisher.publish(det_msg)

        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        cv2.circle(img, (cx,cy), 2, (0,0,255), -1)
        cv2.rectangle(img, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[2]), (0,0,255), 2)
        
        # self.visualize(img, mask)
        if self.vis_enabled:
            self.publish_detection_image(img, mask, img_msg.header.stamp)
            

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
