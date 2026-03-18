#!/usr/bin/env python3
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class HsvTuner(Node):
    NAV_IMAGE_TOPIC = '/zed/zed_node/left/image_rect_color'

    def __init__(self):
        super().__init__('hsv_tuner')
        self.bridge = CvBridge()
        self.latest_img = None

        self.image_topic = self.NAV_IMAGE_TOPIC

        self.low_h = 0
        self.high_h = 180
        self.low_s = 0
        self.high_s = 255
        self.low_v = 0
        self.high_v = 255

        self.window_name = 'HSV Tuner'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Low H', self.window_name, self.low_h, 180, self._noop)
        cv2.createTrackbar('High H', self.window_name, self.high_h, 180, self._noop)
        cv2.createTrackbar('Low S', self.window_name, self.low_s, 255, self._noop)
        cv2.createTrackbar('High S', self.window_name, self.high_s, 255, self._noop)
        cv2.createTrackbar('Low V', self.window_name, self.low_v, 255, self._noop)
        cv2.createTrackbar('High V', self.window_name, self.high_v, 255, self._noop)

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.ui_timer = self.create_timer(0.03, self.update_ui)

        self.get_logger().info(
            f"HSV tuner ready. Subscribed to '{self.image_topic}'. "
            "Press 'p' to print YAML values, 'q' or ESC to quit."
        )

    @staticmethod
    def _noop(_):
        return

    def image_callback(self, msg: Image):
        self.latest_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _read_trackbar_values(self):
        self.low_h = cv2.getTrackbarPos('Low H', self.window_name)
        self.high_h = cv2.getTrackbarPos('High H', self.window_name)
        self.low_s = cv2.getTrackbarPos('Low S', self.window_name)
        self.high_s = cv2.getTrackbarPos('High S', self.window_name)
        self.low_v = cv2.getTrackbarPos('Low V', self.window_name)
        self.high_v = cv2.getTrackbarPos('High V', self.window_name)

    def _validate_ranges(self):
        if self.low_h > self.high_h:
            self.high_h = self.low_h
            cv2.setTrackbarPos('High H', self.window_name, self.high_h)
        if self.low_s > self.high_s:
            self.high_s = self.low_s
            cv2.setTrackbarPos('High S', self.window_name, self.high_s)
        if self.low_v > self.high_v:
            self.high_v = self.low_v
            cv2.setTrackbarPos('High V', self.window_name, self.high_v)

    def _print_yaml_hint(self):
        msg = (
            'color_segmentation:\n'
            f'  h_min: {self.low_h}\n'
            f'  h_max: {self.high_h}\n'
            f'  s_min: {self.low_s}\n'
            f'  s_max: {self.high_s}\n'
            f'  v_min: {self.low_v}\n'
            f'  v_max: {self.high_v}'
        )
        self.get_logger().info('\n' + msg)

    def update_ui(self):
        self._read_trackbar_values()
        self._validate_ranges()

        if self.latest_img is not None:
            hsv = cv2.cvtColor(self.latest_img, cv2.COLOR_BGR2HSV)
            lower = (self.low_h, self.low_s, self.low_v)
            upper = (self.high_h, self.high_s, self.high_v)
            mask = cv2.inRange(hsv, lower, upper)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            view = cv2.hconcat([self.latest_img, mask_bgr])
            cv2.imshow(self.window_name, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            self.get_logger().info('Closing HSV tuner.')
            rclpy.shutdown()
        elif key == ord('p'):
            self._print_yaml_hint()

    def destroy_node(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HsvTuner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
