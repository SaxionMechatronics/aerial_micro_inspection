#!/usr/bin/env python3
import os
import yaml
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration

import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from ai_scanner_interfaces.msg import ObjectDetectionResult

from ultralytics import YOLO
from ultralytics.engine.results import Results


class SurfaceSegmentorNode(Node):
    def __init__(self):
        super().__init__('surface_segmentor_node')
        self.bridge = CvBridge()

        # --- load params ---
        self.declare_parameter('mission_config_file', '')
        param_file = self.get_parameter('mission_config_file').value
        with open(param_file, 'r') as f:
            self.config = yaml.safe_load(f)

        seg_cfg = self.config.get('surface_segmentation', self.config.get('detection', {}))

        # --- model setup ---
        model_path = seg_cfg['path']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_trt = bool(self.config.get('runtime', {}).get('use_trt', False))

        engine_path = model_path.replace('.pt', '.engine')
        if self.use_trt:
            if not os.path.exists(engine_path):

                self.get_logger().info(f"\n\n\n\n\n======= TensorRT Optimized Model Not Found =======\n\n\n\n\n")
                self.get_logger().info(f"Optimizing the model: {model_path}. This may take a while, but this operation will only happen in this run.\n\n")

                YOLO(model_path).export(format='trt', half=True, verbose=True)

                self.get_logger().info(f"\n\nOptimization Done.\n===================================================")

            if os.path.exists(engine_path):
                self.model = YOLO(engine_path, task='segment')
                self.get_logger().info(f"Loaded YOLO model: {engine_path}")
            else:
                self.get_logger().warning('TensorRT enabled but engine file is unavailable. Falling back to PyTorch model.')
                torch.backends.cudnn.benchmark = True
                self.model = YOLO(model_path).to(self.device)
                self.get_logger().info(f"Loaded YOLO model: {model_path}")
        else:
            torch.backends.cudnn.benchmark = True
            self.model = YOLO(model_path).to(self.device)
            self.get_logger().info(f"Loaded YOLO model: {model_path}")

        # determine if this checkpoint supports segmentation
        self.has_masks = hasattr(self.model.model, 'mosaic') and hasattr(self.model.model, 'segments') \
                         or hasattr(self.model.model, 'task') and self.model.model.task == 'segment'

        self.get_logger().info(f"segmentation={'yes' if self.has_masks else 'no'}")

        # thresholds & topics
        self.conf_thresh     = float(seg_cfg['confidence_threshold'])
        self.iou_thresh      = float(seg_cfg['iou_threshold'])
        self.target_class_id = int(seg_cfg.get('target_class_id', -1))
        input_cfg = self.config.get('input', {})
        output_cfg = self.config.get('output', {})
        self.camera_topic    = input_cfg.get('nav_rgb_topic', input_cfg.get('nav_image_topic'))
        self.img_type        = self.config['input']['img_type']
        det_topic            = output_cfg.get('surface_segmentation_topic', output_cfg.get('detection_topic'))
        vis_topic            = output_cfg.get('surface_segmentation_vis_topic', output_cfg.get('detection_vis_topic'))
        self.vis_enabled     = bool(self.config['visualization'])
        runtime_cfg          = self.config.get('runtime', {})
        self.downsample_visualization = bool(runtime_cfg.get('downsample_visualization', False))
        self.vis_output_width = int(runtime_cfg.get('visualization_width', 240))
        self.vis_output_height = int(runtime_cfg.get('visualization_height', 160))

        # --- ROS pubs/subs ---
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, qos)
        self.det_pub = self.create_publisher(
            ObjectDetectionResult, det_topic, 10)
        
        self.run_srv = self.create_service(
            Trigger,
            'detect_surface',
            self.handle_run_inference
        )

        # (optional) vis window
        if self.vis_enabled:
            self.vis_publisher = self.create_publisher(Image, vis_topic, 10)

        self.latest_img_msg = None
        self._last_time = self.get_clock().now() #- Duration(seconds=0.1)

    def process_image(self, repeat=False):

        if self.latest_img_msg is None:
            return False
        
        # convert input
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_img_msg, 'bgr8')
        rgb    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # inference
        res: Results = self.model.predict(
            source=rgb,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )[0].cpu()

        # annotated display
        ann = res.plot()
        ann_bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)

        # publish each instance
        boxes = res.boxes.xyxy.cpu().numpy()        # (N,4)
        confs = res.boxes.conf.cpu().numpy()        # (N,)
        class_ids = res.boxes.cls.cpu().numpy().astype(np.int32) if res.boxes.cls is not None else np.array([], dtype=np.int32)
        masks = getattr(res.masks, 'data', None)    # (N,H,W) or None
        
        H, W = cv_img.shape[:2]
        target_detected = False
        max_box_idx = -1
        max_box_score = -1
        candidate_indices = list(range(len(boxes)))
        if self.target_class_id >= 0 and len(class_ids) == len(boxes):
            candidate_indices = [i for i, cls_id in enumerate(class_ids) if int(cls_id) == self.target_class_id]

        for i in candidate_indices:
            box = boxes[i]

            target_detected = True

            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            area = w*h / (W*H)
            conf = confs[i]
            dist = np.linalg.norm([x1+w/2 - W/2, y1+h/2 - H/2]) / np.linalg.norm([W/2, H/2])
            score = area + conf + dist

            if score > max_box_score:
                max_box_score = w*h 
                max_box_idx = i

        if target_detected:

            box = boxes[max_box_idx]
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            det = ObjectDetectionResult()
            det.header.stamp = self.latest_img_msg.header.stamp

            # bounding box
            det.x = int(x1)
            det.y = int(y1)
            det.w = int(w)
            det.h = int(h)
            det.img_w = int(W)
            det.img_h = int(H)
            det.src_type = self.img_type

            # if segmentation masks exist, crop & publish the per-instance mask
            if masks is not None:

                # Output masks of yolo are not in the same size of input image.
                full_mask = (masks.data[max_box_idx].cpu().numpy() > 0.5).astype(np.uint8) * 255  
                mask_np = cv2.resize(
                    full_mask,
                    (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                # self.get_logger().info(f"detection mask size: {mask_np.shape}, img size: {rgb.shape}, ann size: {ann_bgr.shape}")
                # crop to bounding box for efficiency (optional)
                # x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                # crop = mask_np[y1i:y2i, x1i:x2i]
                det.mask = self.bridge.cv2_to_imgmsg(mask_np, 'mono8')
            else:
                # leave det.mask empty
                pass

            if repeat:
                for cnt in range(50):
                    self.det_pub.publish(det)
                    time.sleep(0.01)
            else:
                self.det_pub.publish(det)

        # show window if desired
        if self.vis_enabled:
            if self.downsample_visualization:
                vis_out = cv2.resize(ann_bgr, (self.vis_output_width, self.vis_output_height))
            else:
                vis_out = ann_bgr
            img_msg = self.bridge.cv2_to_imgmsg(vis_out, encoding='bgr8')
            img_msg.header.stamp = self.latest_img_msg.header.stamp
            self.vis_publisher.publish(img_msg)

        return target_detected

    def image_callback(self, msg: Image):

        delta = self.get_clock().now() - Time.from_msg(msg.header.stamp)
        delta = delta.nanoseconds * 1e-9
        if delta > 0.1:
            return 
        
        # We dont want to process images faster than 10 Hz
        now = self.get_clock().now()
        if now - self._last_time < Duration(seconds=0.1):
            return
        self._last_time = now

        self.latest_img_msg = msg
        self.process_image()
        

    def handle_run_inference(self, request, response):

        # succ = self.process_image(repeat=True)
        succ = False

        if succ:
            response.success = True
            response.message = 'Inference executed on the latest image.'
            return response
        else:
            response.success = False
            response.message = 'Nothing detected.'
            return response


    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SurfaceSegmentorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
