#!/usr/bin/env python3
"""
ROS2 node that listens to the transform between 'nav_cam' and 'ai_camera'
and prints it at 10 Hz using tf2_ros.
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

class TfPrinter(Node):
    def __init__(self):
        super().__init__('tf_printer_node')
        # Initialize tf2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Timer to periodically lookup and print the transform
        self.timer = self.create_timer(0.1, self.print_transform)
        self.get_logger().info('TF Printer node started.')

    def print_transform(self):
        try:
            # Lookup the latest transform from 'nav_cam' to 'ai_camera'
            trans = self.tf_buffer.lookup_transform(
                'nav_cam',  # target frame
                'ai_camera',    # source frame
                rclpy.time.Time())  # latest available

            # Extract translation and rotation
            t = trans.transform
            self.get_logger().info(
                f"Transform nav_cam → ai_camera: "
                f"translation = ({t.translation.x:.3f}, {t.translation.y:.3f}, {t.translation.z:.3f}); "
                f"rotation = ({t.rotation.x:.3f}, {t.rotation.y:.3f}, {t.rotation.z:.3f}, {t.rotation.w:.3f})"
            )
        except Exception as e:
            self.get_logger().warning(f'Could not transform nav_cam → ai_camera: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TfPrinter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
