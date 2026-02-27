#!/usr/bin/env python3
import os
import yaml

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

import torch
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

from ros2_yolo_interfaces.msg import BoundingBox, BoundingBoxArray
from std_msgs.msg import Header


class MicroDetectorNode(Node):
    def __init__(self):
        super().__init__('micro_detector_node')

        self.declare_parameter('mission_config_file', '')
        mission_config_file = self.get_parameter('mission_config_file').value
        with open(mission_config_file, 'r') as config_stream:
            self.config = yaml.safe_load(config_stream) or {}

        micro_cfg = self.config.get('micro_detection', {})
        input_cfg = self.config.get('input', {})
        output_cfg = self.config.get('output', {})
        runtime_cfg = self.config.get('runtime', {})

        model_path = micro_cfg.get('path', 'weights/yolov10n_eggs.pt')
        model_path = self._resolve_model_path(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)

        self.conf_thresh = float(micro_cfg.get('confidence_threshold', 0.05))
        self.iou_thresh = float(micro_cfg.get('iou_threshold', 0.1))
        self.camera_topic = input_cfg.get('inspection_topic', input_cfg.get('ai_image_topic', '/inspection_cam/rgb_image/image_raw'))
        self.bbox_topic = output_cfg.get('micro_detection_topic', '/micro_det/detections')
        self.detected_image_topic = output_cfg.get('micro_detection_vis_topic', '/micro_det/vis_image')
        self.downsample_output = bool(runtime_cfg.get('downsample_visualization', False))
        self.output_w = int(runtime_cfg.get('visualization_width', 240))
        self.output_h = int(runtime_cfg.get('visualization_height', 160))

        self.subscriber = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.bbox_publisher = self.create_publisher(BoundingBoxArray, self.bbox_topic, 10)
        self.image_publisher = self.create_publisher(Image, self.detected_image_topic, 10)
        self.bridge = CvBridge()

        self.get_logger().info(f'Micro detector model loaded: {model_path} on {self.device}')

    def _resolve_model_path(self, configured_path: str) -> str:
        if os.path.isabs(configured_path):
            return configured_path

        pkg_share = get_package_share_directory('aerial_micro_inspection')
        share_candidate = os.path.join(pkg_share, configured_path)
        if os.path.exists(share_candidate):
            return share_candidate

        ws_candidate = os.path.join('/ws/src/aerial_micro_inspection', configured_path)
        if os.path.exists(ws_candidate):
            return ws_candidate

        return configured_path

    def image_callback(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            source=rgb_img,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
            device=self.device
        )[0].cpu()

        annotated = results.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        self.publish_bboxes(results)
        self.publish_annotated_image(annotated, msg.header)

    def publish_annotated_image(self, image, header):
        if self.downsample_output:
            image_out = cv2.resize(image, (self.output_w, self.output_h))
        else:
            image_out = image
        msg = self.bridge.cv2_to_imgmsg(image_out, 'bgr8')
        msg.header = header
        self.image_publisher.publish(msg)

    def publish_bboxes(self, results: Results):
        boxes: Boxes = results.boxes
        org_imgsz = results.orig_shape
        conf_scores = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
        xywh = boxes.xywh.cpu().numpy() if boxes.xywh is not None else np.array([])
        class_indices = boxes.cls.cpu().numpy().astype(np.int32) if boxes.cls is not None else np.array([], dtype=np.int32)
        class_labels = [results.names[i] for i in class_indices] if results.names else []

        if len(xywh) > 0:
            x1y1 = xywh[:, :2] - (xywh[:, 2:] / 2)
            x2y2 = x1y1 + xywh[:, 2:]
        else:
            x1y1 = np.empty((0, 2))
            x2y2 = np.empty((0, 2))

        msg = BoundingBoxArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.original_image_shape = [int(org_imgsz[0]), int(org_imgsz[1])]

        for i in range(len(conf_scores)):
            bb = BoundingBox()
            bb.confidence = float(conf_scores[i])
            bb.class_id = int(class_indices[i])
            bb.class_label = class_labels[i]
            bb.center_x = float(xywh[i, 0])
            bb.center_y = float(xywh[i, 1])
            bb.width = float(xywh[i, 2])
            bb.height = float(xywh[i, 3])
            bb.x1 = float(x1y1[i, 0])
            bb.y1 = float(x1y1[i, 1])
            bb.x2 = float(x2y2[i, 0])
            bb.y2 = float(x2y2[i, 1])
            msg.boxes.append(bb)

        self.bbox_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MicroDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
