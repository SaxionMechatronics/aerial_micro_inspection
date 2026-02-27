#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_share = get_package_share_directory('aerial_micro_inspection')

    # resolve your config files
    calib_yaml    = os.path.join(pkg_share, 'config', 'real_test', 'cameras_calib.yaml')

    # Launch arguments
    nav_img_arg = DeclareLaunchArgument(
        'nav_image_topic',
        default_value='/zed/zed_node/left_raw/image_raw_color',
        description='Topic for the Nav camera color image'
    )
    depth_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/zed/zed_node/depth/depth_registered',
        description='Topic for the Nav camera depth image'
    )
    gimbal_arg = DeclareLaunchArgument(
        'gimbal_topic',
        default_value='/gimbal_orientation',
        description='Topic for gimbal orientation (QuaternionStamped)'
    )
    calib_arg = DeclareLaunchArgument(
        'calib_yaml',
        default_value=calib_yaml,
        description='Path to the calibration YAML produced by calibration node'
    )

    # Node
    tracker_node = Node(
        package='aerial_micro_inspection',
        executable='gimbal_tracker',
        name='gimbal_tracker',
        output='screen',
        parameters=[{
            'nav_image_topic': LaunchConfiguration('nav_image_topic'),
            'depth_topic':     LaunchConfiguration('depth_topic'),
            'gimbal_topic':    LaunchConfiguration('gimbal_topic'),
            'calib_yaml':      LaunchConfiguration('calib_yaml'),
        }]
    )

    detector_node = Node(
        package='aerial_micro_inspection',
        executable='color_detector',
        name='color_detector',
        output='screen',
    )

    return LaunchDescription([
        nav_img_arg,
        depth_arg,
        gimbal_arg,
        calib_arg,
        tracker_node,
        detector_node
    ])
