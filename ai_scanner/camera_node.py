import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml

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

        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video device: {device}")
            raise RuntimeError("Camera open failed")

        input_cfg = self.config.get('input', {})
        inspection_topic = input_cfg.get('inspection_topic', input_cfg.get('ai_image_topic'))
        self.publisher_ = self.create_publisher(Image, inspection_topic, 1)
        self.br = CvBridge()
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame capture failed")
            return
        msg = self.br.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
