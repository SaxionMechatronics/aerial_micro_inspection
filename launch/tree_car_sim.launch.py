#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    aerial_micro_inspection_share = get_package_share_directory('aerial_micro_inspection')
    ros_ign_gazebo_share = get_package_share_directory('ros_ign_gazebo')

    worlds_dir = os.path.join(aerial_micro_inspection_share, 'simulation', 'gazebo', 'worlds')
    world_path = os.path.join(worlds_dir, 'tree_car_world.sdf')

    current_ign_resource_path = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    current_gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')

    ign_resource_path = worlds_dir
    if current_ign_resource_path:
        ign_resource_path = worlds_dir + ':' + current_ign_resource_path

    gz_resource_path = worlds_dir
    if current_gz_resource_path:
        gz_resource_path = worlds_dir + ':' + current_gz_resource_path

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_ign_gazebo_share, 'launch', 'ign_gazebo.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -v 4 {world_path}',
        }.items()
    )

    return LaunchDescription([
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', ign_resource_path),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
        gazebo,
    ])
