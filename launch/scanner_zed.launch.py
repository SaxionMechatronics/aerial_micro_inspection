from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    aerial_micro_inspection_share = get_package_share_directory('aerial_micro_inspection')

    # Mission configurations
    mission_configs_arg = DeclareLaunchArgument('mission_config_file')
    det_mode_arg = DeclareLaunchArgument(
        'det_mode',
    )

    mission_config = PathJoinSubstitution([
        aerial_micro_inspection_share,
        'config',
        LaunchConfiguration('mission_config_file')
    ])

    # Launch scanner nodes (gimbal node, usb camera, gimbal tracker, detection)
    scanner_system_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(aerial_micro_inspection_share, 'launch', 'scanner.launch.py')
        ),
        launch_arguments={
            'mission_config_file': mission_config,
            'det_mode': LaunchConfiguration('det_mode')
        }.items()
    )

    # Launch inspection USB camera publisher used by the detector/tracker pipeline
    usb_camera_node = Node(
        package='aerial_micro_inspection',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'mission_config_file': mission_config
        }]
    )

    # Launch ZED camera disabling their navigation and extra components
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed_camera.launch.py')
        ),
        launch_arguments={
            'camera_model': 'zed2',
            'ros_params_override_path': os.path.join(get_package_share_directory('aerial_micro_inspection'), 
                                                     'config', 'zed2_rgb_depth_config.yaml'),
            'publish_tf': 'false',
            'publish_map_tf': 'false',
        }.items()
    )

    return LaunchDescription([
        mission_configs_arg,
        det_mode_arg,
        zed_launch,
        usb_camera_node,
        scanner_system_launch
    ])
