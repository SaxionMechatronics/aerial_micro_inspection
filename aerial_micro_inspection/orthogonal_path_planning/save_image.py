#!/usr/bin/env python3

import os
import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_srvs.srv import Trigger         

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from px4_msgs.msg import SensorGps

from cv_bridge import CvBridge
import cv2

from PIL import Image as PILImage
import piexif
import numpy as np
from scipy.spatial.transform import Rotation


# =========================================================
# CONFIG  (unchanged)
# =========================================================

SAVE_FOLDER = "/ws/src/aerial_micro_inspection/images"

IMAGE_TOPIC       = "/inspection_cam/rgb_image/image_raw"
GPS_TOPIC         = "/fmu/out/vehicle_gps_position"
POSE_TOPIC        = "/inspection/viewpoint"
CAMERA_INFO_TOPIC = "/inspection_cam/rgb_image/camera_info"

SENSOR_WIDTH_MM  = 6.17
SENSOR_HEIGHT_MM = 4.55

DATA_TIMEOUT_SEC = 10.0

GEO_FILENAME = "geo.txt"


# =========================================================
# HELPERS  (all unchanged)
# =========================================================

def decimal_to_dms(value: float):
    value = abs(value)
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60 * 1000)
    return (
        (degrees, 1),
        (minutes, 1),
        (seconds, 1000),
    )


def focal_px_to_mm(fx: float, image_width_px: int, sensor_width_mm: float) -> float:
    return fx * sensor_width_mm / image_width_px


def ned_quaternion_to_webodm_angles(pose: Pose):
    q = pose.orientation
    rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
    yaw_deg, pitch_deg, roll_deg = rotation.as_euler('ZYX', degrees=True)
    return yaw_deg, pitch_deg, roll_deg


# =========================================================
# NODE
# =========================================================

class ImageCaptureNode(Node):

    def __init__(self):
        super().__init__("image_capture_node")

        self.bridge = CvBridge()

        self.latest_image       = None
        self.latest_gps         = None
        self.latest_pose        = None
        self.latest_camera_info = None

        # ReentrantCallbackGroup lets the service callback and the topic
        # callbacks run concurrently inside the MultiThreadedExecutor.
        # Without this, the spin-wait loop inside the service handler
        # would starve all subscribers and the node would always time out.
        self._cb_group = ReentrantCallbackGroup()

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------------------------------
        # Subscriptions (unchanged topics,
        # only cb_group added to each)
        # ---------------------------------

        self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, 10,
            callback_group=self._cb_group
        )
        self.create_subscription(
            SensorGps, GPS_TOPIC, self.gps_callback, qos_profile_sub,
            callback_group=self._cb_group
        )
        self.create_subscription(
            Pose, POSE_TOPIC, self.pose_callback, 10,
            callback_group=self._cb_group
        )
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10,
            callback_group=self._cb_group
        )

        # ---------------------------------
        # Service server  ← NEW
        # Replaces the one-shot main() logic.
        # The planner calls /take_photo and
        # blocks until this returns.
        # ---------------------------------

        self._service = self.create_service(
            Trigger,
            'inspection/take_photo',
            self.take_photo_callback,
            callback_group=self._cb_group
        )

        self.get_logger().info(
            "ImageCaptureNode ready — service '/take_photo' is live."
        )

    # =====================================================
    # TOPIC CALLBACKS  (unchanged)
    # =====================================================

    def image_callback(self, msg):
        self.latest_image = msg

    def gps_callback(self, msg):
        self.latest_gps = msg

    def pose_callback(self, msg):
        self.latest_pose = msg

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    # =====================================================
    # SERVICE CALLBACK  ← NEW
    # Replaces the spin-wait loop that was in main().
    # Blocks until data is ready (or timeout), then saves.
    # =====================================================

    def take_photo_callback(self, request, response):
        self.get_logger().info("take_photo called — waiting for sensor data...")

        start = self.get_clock().now()

        while not self.data_ready():
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            if elapsed > DATA_TIMEOUT_SEC:
                msg = (
                    f"Timed out after {DATA_TIMEOUT_SEC}s — "
                    "one or more mandatory topics not received "
                    "(image / GPS / camera_info)."
                )
                self.get_logger().error(msg)
                response.success = False
                response.message = msg
                return response

            # Small sleep to avoid busy-spinning.
            # The MultiThreadedExecutor keeps subscriber callbacks firing
            # on other threads while this one sleeps.
            import time
            time.sleep(0.1)

        self.get_logger().info("All data ready — saving capture...")
        image_path = self._save_capture()

        if image_path is None:
            response.success = False
            response.message = "Image save failed — check node logs for details."
        else:
            response.success = True
            response.message = image_path   # planner can log where the file landed

        return response

    # =====================================================
    # READINESS CHECK  (unchanged)
    # =====================================================

    def data_ready(self) -> bool:
        return (
            self.latest_image       is not None and
            self.latest_gps         is not None and
            self.latest_camera_info is not None
        )

    # =====================================================
    # MAIN SAVE FUNCTION
    # Only change vs original: renamed to _save_capture
    # and returns the image path (str) or None on failure
    # instead of returning nothing.
    # =====================================================

    def _save_capture(self) -> str | None:

        os.makedirs(SAVE_FOLDER, exist_ok=True)

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"capture_{timestamp}.jpg"
        image_path = os.path.join(SAVE_FOLDER, image_name)

        # ROS Image -> PIL
        try:
            cv_image  = self.bridge.imgmsg_to_cv2(
                self.latest_image, desired_encoding="bgr8"
            )
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
        except Exception as e:
            self.get_logger().error(f"Failed converting ROS image: {e}")
            return None

        # Camera intrinsics
        K               = self.latest_camera_info.k
        fx              = K[0]
        fy              = K[4]
        image_width_px  = self.latest_camera_info.width
        image_height_px = self.latest_camera_info.height

        focal_length_mm = focal_px_to_mm(fx, image_width_px, SENSOR_WIDTH_MM)
        fplane_x = image_width_px  / SENSOR_WIDTH_MM
        fplane_y = image_height_px / SENSOR_HEIGHT_MM

        # GPS
        lat = self.latest_gps.latitude_deg
        lon = self.latest_gps.longitude_deg
        alt = self.latest_gps.altitude_msl_m

        lat_ref = b"N" if lat >= 0 else b"S"
        lon_ref = b"E" if lon >= 0 else b"W"

        # Orientation (optional)
        yaw_deg = pitch_deg = roll_deg = None

        if self.latest_pose is not None:
            yaw_deg, pitch_deg, roll_deg = ned_quaternion_to_webodm_angles(
                self.latest_pose
            )
            self.get_logger().info(
                f"  Orientation -> yaw={yaw_deg:.2f} "
                f"pitch={pitch_deg:.2f} roll={roll_deg:.2f}"
            )
        else:
            self.get_logger().warn(
                "No pose received — geo.txt will be written without orientation."
            )

        # EXIF
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make:        b"GazeboSim",
                piexif.ImageIFD.Model:       b"InspectionCamera",
                piexif.ImageIFD.ImageWidth:  image_width_px,
                piexif.ImageIFD.ImageLength: image_height_px,
            },
            "Exif": {
                piexif.ExifIFD.FocalLength: (int(focal_length_mm * 1000), 1000),
                piexif.ExifIFD.FocalPlaneXResolution: (int(fplane_x * 1000), 1000),
                piexif.ExifIFD.FocalPlaneYResolution: (int(fplane_y * 1000), 1000),
                piexif.ExifIFD.FocalPlaneResolutionUnit: 4,
                piexif.ExifIFD.DateTimeOriginal: (
                    datetime.now().strftime("%Y:%m:%d %H:%M:%S").encode()
                ),
            },
            "GPS": {
                piexif.GPSIFD.GPSVersionID: b"\x02\x03\x00\x00",
                piexif.GPSIFD.GPSLatitudeRef:  lat_ref,
                piexif.GPSIFD.GPSLatitude:     decimal_to_dms(lat),
                piexif.GPSIFD.GPSLongitudeRef: lon_ref,
                piexif.GPSIFD.GPSLongitude:    decimal_to_dms(lon),
                piexif.GPSIFD.GPSAltitude:     (int(alt * 1000), 1000),
                piexif.GPSIFD.GPSAltitudeRef:  0,
                **({
                    piexif.GPSIFD.GPSImgDirectionRef: b"T",
                    piexif.GPSIFD.GPSImgDirection: (int(yaw_deg % 360 * 100), 100),
                } if yaw_deg is not None else {}),
            }
        }

        try:
            exif_bytes = piexif.dump(exif_dict)

            if not exif_bytes.startswith(b"Exif\x00\x00"):
                exif_bytes = b"Exif\x00\x00" + exif_bytes

            pil_image.save(image_path, format="JPEG", quality=95, exif=exif_bytes)
        except Exception as e:
            self.get_logger().error(f"Failed writing image with EXIF: {e}")
            return None

        self.get_logger().info(
            f"Image saved: {image_path}\n"
            f"  GPS      -> lat={lat:.8f}, lon={lon:.8f}, alt={alt:.3f} m\n"
            f"  Focal    -> {focal_length_mm:.3f} mm  (fx={fx:.3f} px)\n"
            f"  Sensor   -> {SENSOR_WIDTH_MM} x {SENSOR_HEIGHT_MM} mm\n"
            f"  FPlane   -> {fplane_x:.2f} x {fplane_y:.2f} px/mm"
        )

        # self._append_geo_txt(image_name, lat, lon, alt, yaw_deg, pitch_deg, roll_deg)
        self._update_cameras_json(fx, fy, image_width_px, image_height_px)

        return image_path

    # =====================================================
    # GEO.TXT  (unchanged)
    # =====================================================

    def _append_geo_txt(self, image_name, lat, lon, alt,
                        yaw_deg=None, pitch_deg=None, roll_deg=None):
        geo_path    = os.path.join(SAVE_FOLDER, GEO_FILENAME)
        file_exists = os.path.isfile(geo_path)

        with open(geo_path, "a") as f:
            if not file_exists:
                f.write("EPSG:4326\n")
                self.get_logger().info(f"geo.txt created: {geo_path}")

            if yaw_deg is not None and pitch_deg is not None and roll_deg is not None:
                row = (
                    f"{image_name}\t{lat:.10f}\t{lon:.10f}\t{alt:.4f}\t"
                    f"{yaw_deg:.6f}\t{pitch_deg:.6f}\t{roll_deg:.6f}\n"
                )
            else:
                row = f"{image_name}\t{lat:.10f}\t{lon:.10f}\t{alt:.4f}\n"

            f.write(row)

        self.get_logger().info(
            f"geo.txt updated -> {image_name}  "
            f"lat={lat:.6f} lon={lon:.6f} alt={alt:.2f} m"
            + (f"  yaw={yaw_deg:.2f} pitch={pitch_deg:.2f} roll={roll_deg:.2f}"
               if yaw_deg is not None else "  (no orientation)")
        )

    # =====================================================
    # CAMERAS.JSON  (unchanged)
    # =====================================================

    def _update_cameras_json(self, fx, fy, width, height):
        K  = self.latest_camera_info.k
        cx = K[2]
        cy = K[5]
        D  = list(self.latest_camera_info.d)

        max_dim    = max(width, height)
        focal_norm = fx / max_dim
        c_x_norm   = (cx - width  / 2.0) / max_dim
        c_y_norm   = (cy - height / 2.0) / max_dim

        k1 = D[0] if len(D) > 0 else 0.0
        k2 = D[1] if len(D) > 1 else 0.0
        p1 = D[2] if len(D) > 2 else 0.0
        p2 = D[3] if len(D) > 3 else 0.0
        k3 = D[4] if len(D) > 4 else 0.0

        cameras_data = {
            "v2": {
                "GazeboSim InspectionCamera": {
                    "projection_type": "brown",
                    "width":  width,
                    "height": height,
                    "focal":  round(focal_norm, 10),
                    "c_x":    round(c_x_norm,   10),
                    "c_y":    round(c_y_norm,    10),
                    "k1": k1, "k2": k2,
                    "p1": p1, "p2": p2,
                    "k3": k3,
                }
            }
        }

        cameras_path = os.path.join(SAVE_FOLDER, "cameras.json")
        with open(cameras_path, "w") as f:
            json.dump(cameras_data, f, indent=2)

        self.get_logger().info(f"cameras.json written: {cameras_path}")


# =========================================================
# MAIN  ← only change: persistent spin with MultiThreadedExecutor
# instead of one-shot spin_once loop
# =========================================================

def main(args=None):
    rclpy.init(args=args)

    node = ImageCaptureNode()

    # MultiThreadedExecutor is required so that the service callback
    # (which blocks in a wait loop) and the topic subscription callbacks
    # can run concurrently on separate threads.
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