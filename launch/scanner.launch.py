#!/usr/bin/env python3
import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir, PythonExpression, \
                                 PathJoinSubstitution, TextSubstitution, Command
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory


def _build_nodes(context):

    pkg_share = get_package_share_directory('ai_scanner')
    configs_dir = PathJoinSubstitution([
        pkg_share,
        'config'
    ])

    mission_config_file = LaunchConfiguration('mission_config_file').perform(context)
    use_sim_time = False
    try:
        with open(mission_config_file, 'r') as config_stream:
            mission_config = yaml.safe_load(config_stream) or {}
        use_sim_time = bool(mission_config.get('runtime', {}).get('use_sim_time', False))
    except Exception:
        use_sim_time = False

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
        output='screen',
        parameters=[{
            'mission_config_file': LaunchConfiguration('mission_config_file')
        }]
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
        executable='surface_segmentor',
        name='surface_segmentor',
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

    # This is necessary to make sure the ROS time (self.get_clock().now()) is the same as the time inside Gazebo. If not in simulation mode, then use_sim_time will be false and this will have no effect.
    param = SetParameter(name='use_sim_time', value=use_sim_time)

    return [
        param,
        gimbal_node,
        tracker_node,
        color_detector_node,
        epr_yolo_node,
        ai_detector_node
    ]


def generate_launch_description():
    # Launch arguments
    mission_config_arg = DeclareLaunchArgument(
        'mission_config_file',
    )
    det_mode_arg = DeclareLaunchArgument(
        'det_mode',
    )

    return LaunchDescription([
        mission_config_arg,
        det_mode_arg,
        OpaqueFunction(function=_build_nodes),
    ])
