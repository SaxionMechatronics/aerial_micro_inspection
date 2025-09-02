#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('ai_scanner')

    # resolve your config files
    ai_yaml    = os.path.join(pkg_share, 'config', 'sony_55mm.yaml')
    aruco_yaml = os.path.join(pkg_share, 'config', 'aruco.yaml')

    return LaunchDescription([
        Node(
            package='ai_scanner',
            executable='calibrate',
            name='calibrate',
            output='screen',
            parameters=[
                # AI camera (only YAML, no CameraInfo topic)
                { 'ai_image_topic':         '/camera/image_raw' },
                { 'ai_camera_info_yaml':    ai_yaml },
                { 'ai_camera_info_topic':   '' },

                # Nav camera (only CameraInfo topic, no YAML)
                { 'nav_image_topic':        '/zed/zed_node/left_raw/image_raw_color' },
                { 'nav_camera_info_yaml':   '' },
                { 'nav_camera_info_topic':  '/zed/zed_node/left_raw/camera_info' },

                # AprilBoard spec
                { 'aruco_yaml':       aruco_yaml },

                # Gimbal orientation topic
                { 'gimbal_topic':           '/gimbal_orientation' },

                # Enable on-screen visualization
                { 'visualize':              True },
            ],
        ),
    ])
