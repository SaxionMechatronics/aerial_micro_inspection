import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import yaml
import time

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('mission_config_file', '')
        self.declare_parameter('fps', 5)

        param_file = self.get_parameter('mission_config_file').get_parameter_value().string_value
        with open(param_file, 'r') as f:
            self.config = yaml.safe_load(f)
        fps = self.get_parameter('fps').get_parameter_value().integer_value
        device = self.config['input']['video_device']

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video device: {device}")
            raise RuntimeError("Camera open failed")

        input_cfg = self.config.get('input', {})
        inspection_topic = input_cfg.get('inspection_topic', input_cfg.get('ai_image_topic'))
        self.publisher_ = self.create_publisher(CompressedImage, inspection_topic, 1)
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame capture failed")
            return

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not success:
            self.get_logger().warn("JPEG encoding failed")
            return

        msg = CompressedImage()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = encoded.tobytes()

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
