#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir, PythonExpression, \
                                 PathJoinSubstitution, TextSubstitution, Command
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_share = get_package_share_directory('ai_scanner')
    configs_dir = PathJoinSubstitution([
        pkg_share,
        'config'
    ])

    # Launch arguments
    mission_config_arg = DeclareLaunchArgument(
        'mission_config_file',
    )
    det_mode_arg = DeclareLaunchArgument(
        'det_mode',
    )

    gimbal_node = Node(
        package='ai_scanner',
        executable='gimbal_node',
        name='gimbal_node',
        output='screen'
    )

    camera_node = Node(
        package='ai_scanner',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            'mission_config_file': LaunchConfiguration('mission_config_file'),
        }],
        output='screen'
    )

    epr_yolo_node = Node(
        package='usb_camera_yolo',
        executable='yolo_node',
        name='yolo_node',
        parameters=[{
            'param_file': 'params',
        }],
        output='screen'
    )

    tracker_node = Node(
        package='ai_scanner',
        executable='gimbal_tracker',
        name='gimbal_tracker',
        output='screen',
        parameters=[{
            'mission_config_file': LaunchConfiguration('mission_config_file'),
            'configs_dir':         configs_dir
        }]
    )

    #TODO: Make sure the given value is either 'ai' or 'color'.
    is_ai = PythonExpression(["'", LaunchConfiguration('det_mode'), "' == 'ai'"])
    ai_detector_node = Node(
        package='ai_scanner',
        executable='tree_trunk_detector',
        name='tree_trunk_detector',
        output='screen',
        parameters=[{
            'mission_config_file': LaunchConfiguration('mission_config_file')
        }],
        condition=IfCondition(is_ai)
    )
    color_detector_node = Node(
        package='ai_scanner',
        executable='color_detector',
        name='color_detector',
        output='screen',
        parameters=[{
            'mission_config_file': LaunchConfiguration('mission_config_file')
        }],
        condition=UnlessCondition(is_ai)
    )

    return LaunchDescription([
        mission_config_arg,
        det_mode_arg,
        camera_node,
        gimbal_node,
        tracker_node,
        color_detector_node,
        epr_yolo_node,
        ai_detector_node
    ])
