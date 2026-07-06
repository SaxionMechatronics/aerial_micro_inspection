#!/usr/bin/env python3

import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_srvs.srv import Trigger

from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

SAVE_FOLDER = "/home/sarax/Documents/scanner_ws/src/aerial_micro_inspection/images"

IMAGE_TOPIC = "/camera/image_compressed"

DATA_TIMEOUT_SEC = 10.0


# =========================================================
# NODE
# =========================================================

class ImageCaptureNode(Node):

    def __init__(self):
        super().__init__("image_capture_node")

        self.latest_image = None

        # ReentrantCallbackGroup allows the service callback (which
        # blocks in a wait loop) and the image subscriber to run
        # concurrently on the MultiThreadedExecutor.
        self._cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            CompressedImage,
            IMAGE_TOPIC,
            self.image_callback,
            10,
            callback_group=self._cb_group,
        )

        self._service = self.create_service(
            Trigger,
            "inspection/take_photo",
            self.take_photo_callback,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            "ImageCaptureNode ready — service 'inspection/take_photo' is live."
        )

    # =====================================================
    # TOPIC CALLBACK
    # =====================================================

    def image_callback(self, msg: CompressedImage):
        self.latest_image = msg

    # =====================================================
    # SERVICE CALLBACK
    # =====================================================

    def take_photo_callback(self, request, response):
        self.get_logger().info("take_photo called — waiting for image...")

        start = self.get_clock().now()

        while self.latest_image is None:
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            if elapsed > DATA_TIMEOUT_SEC:
                msg = (
                    f"Timed out after {DATA_TIMEOUT_SEC}s — "
                    f"no image received on '{IMAGE_TOPIC}'."
                )
                self.get_logger().error(msg)
                response.success = False
                response.message = msg
                return response

            time.sleep(0.05)

        self.get_logger().info("Image ready — saving...")
        image_path = self._save_capture()

        if image_path is None:
            response.success = False
            response.message = "Image save failed — check node logs for details."
        else:
            response.success = True
            response.message = image_path  # caller can log where the file landed

        return response

    # =====================================================
    # SAVE
    # =====================================================

    def _save_capture(self) -> str | None:
        os.makedirs(SAVE_FOLDER, exist_ok=True)

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"capture_{timestamp}.jpg"
        image_path = os.path.join(SAVE_FOLDER, image_name)

        try:

            np_arr   = np.frombuffer(self.latest_image.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                raise ValueError(
                    f"cv2.imdecode returned None — "
                    f"unrecognised format '{self.latest_image.format}'."
                )

            cv2.imwrite(image_path, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        except Exception as e:
            self.get_logger().error(f"Failed saving image: {e}")
            return None

        self.get_logger().info(f"Image saved: {image_path}")
        return image_path


# =========================================================
# MAIN
# =========================================================

def main(args=None):
    rclpy.init(args=args)

    node = ImageCaptureNode()

    # MultiThreadedExecutor is required so that the service callback
    # (blocking wait loop) and the image subscriber run on separate threads.
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
