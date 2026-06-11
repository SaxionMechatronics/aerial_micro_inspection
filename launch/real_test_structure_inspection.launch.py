#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    aerial_micro_inspection_share = get_package_share_directory('aerial_micro_inspection')

    worlds_dir = '/ws/src/aerial_micro_inspection/simulation/gazebo/worlds/'
    world_path = '/ws/src/aerial_micro_inspection/simulation/gazebo/worlds/tree_car_world.sdf'
    world_name = 'tree_car_world' 
    sim_script_path = '/ws/src/aerial_micro_inspection/simulation/gazebo/scripts/simulation-gazebo'

    # --- Launch args ---
    px4_dir_arg = DeclareLaunchArgument('px4_dir', default_value='/opt/PX4-Autopilot')
    px4_make_target_arg = DeclareLaunchArgument('px4_make_target', default_value='gz_x500_inspection')
    px4_gz_model_pose_arg = DeclareLaunchArgument('px4_gz_model_pose', default_value='30,-2,0.5,0,0,0')
    world_arg = DeclareLaunchArgument('world', default_value=world_path)
    extra_resource_path_arg = DeclareLaunchArgument('extra_resource_path', default_value=worlds_dir)
    xrce_udp_port_arg = DeclareLaunchArgument('xrce_udp_port', default_value='8888')
    run_mission_arg = DeclareLaunchArgument('run_mission', default_value='true')
    run_camera_bridge_arg = DeclareLaunchArgument('run_camera_bridge', default_value='true')
    mission_config_file_arg = DeclareLaunchArgument('mission_config_file',default_value='/home/sarax/Documents/scanner_ws/src/aerial_micro_inspection/config/real_test/structural_inspection_config.yaml')

    # --- Launch configs ---
    px4_dir = LaunchConfiguration('px4_dir')
    px4_make_target = LaunchConfiguration('px4_make_target')
    px4_gz_model_pose = LaunchConfiguration('px4_gz_model_pose')
    world = LaunchConfiguration('world')
    extra_resource_path = LaunchConfiguration('extra_resource_path')
    xrce_udp_port = LaunchConfiguration('xrce_udp_port')
    run_mission = LaunchConfiguration('run_mission')
    run_camera_bridge = LaunchConfiguration('run_camera_bridge')
    mission_config_file = LaunchConfiguration('mission_config_file')

    # start_px4 = ExecuteProcess(
    #     cmd=[
    #         'bash', '-lc',
    #         [
    #             'cd ', px4_dir,
    #             # Apply local airframe overrides right before PX4 start.
    #             ' && cp /ws/src/aerial_micro_inspection/simulation/gazebo/px4_airframe/2026_gz_x500_inspection '
    #             '/opt/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/',
    #             ' && python3 /ws/src/aerial_micro_inspection/simulation/gazebo/px4_airframe/modify_px4.py',
    #             ' && export PX4_GZ_MODEL_POSE=', px4_gz_model_pose,
    #             ' && export PX4_GZ_STANDALONE=1',
    #             ' && export PX4_GZ_WORLD=', world_name,
    #             ' && make px4_sitl ', px4_make_target
    #         ]
    #     ],
    #     output='screen'
    # )

    # run_simulation_gazebo = ExecuteProcess(
    #     cmd=[
    #         'bash', '-lc',
    #         [
    #             'pkill -f "[g]z sim" || true; '
    #             'pkill -f "[g]zserver" || true; '
    #             'pkill -f "[g]zclient" || true; '
    #             'python3 ', sim_script_path,
    #             ' --world ', world,
    #             ' --extra_resource_path ', extra_resource_path
    #         ]
    #     ],
    #     output='screen'
    # )

    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', xrce_udp_port],
        output='screen'
    )

    pkg_share = get_package_share_directory("aerial_micro_inspection") 

    # bridge = Node(
    #     package="ros_gz_bridge",
    #     executable="parameter_bridge",
    #     name="camera_bridge",
    #     parameters=[
    #         {
    #             "config_file": os.path.join(
    #                 pkg_share,
    #                 "config",
    #                 "ros_gz_bridge_config.yaml",
    #             ),
    #         },
    #         # Useful if you bridge tf_static later; safe to keep
    #         {"qos_overrides./tf_static.publisher.durability": "transient_local"},
    #     ],
    #     output="screen",
    # )


    # mission_node = Node(
    #     package='aerial_micro_inspection',
    #     executable='structural_inspection_mission',
    #     name='structural_inspection_mission',
    #     output='screen',
    #     parameters=[{
    #     }],
    #     condition=IfCondition(run_mission),
    # )

    # rviz_config = os.path.join(pkg_share, "config", "real_test_inspection_cam.rviz")

    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     arguments=['-d', rviz_config],
    #     output='screen'
    # )

    gimbal_node = Node(
        package='aerial_micro_inspection',
        executable='gimbal_node',
        name='gimbal_node',
        output='screen',
        parameters=[{
            'mission_config_file': mission_config_file
        }]
    )

    camera_node = Node(
        package='aerial_micro_inspection',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'mission_config_file': mission_config_file
        }]
    )

    # photo_node = Node(
    #     package='aerial_micro_inspection',
    #     executable='real_test_save_image',
    #     name='photo_node',
    #     output='screen'
    # )

    ned_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_ned_tf',
        arguments=[
            '0', '0', '0',                 # translation
            '0.7071068', '0.7071068', '0', '0',  # quaternion (ENU -> NED approx)
            'world',
            'ned'
        ],
        output='screen'
    )

    param = SetParameter(name='use_sim_time', value=False)

    return LaunchDescription([
        param,
        px4_dir_arg,
        px4_make_target_arg,
        px4_gz_model_pose_arg,
        world_arg,
        extra_resource_path_arg,
        xrce_udp_port_arg,
        run_mission_arg,
        run_camera_bridge_arg,
        mission_config_file_arg,    
        ned_tf_node,
        #TimerAction(period=1.0, actions=[start_px4]),
        #TimerAction(period=3.0, actions=[run_simulation_gazebo]),
        TimerAction(period=6.0, actions=[xrce_agent]),
        #TimerAction(period=8.0, actions=[bridge]),
        #TimerAction(period=9.0, actions=[mission_node]),
        TimerAction(period=9.5, actions=[gimbal_node]),
        TimerAction(period=10.0, actions=[camera_node]),
        #TimerAction(period=10.5, actions=[rviz_node]),
        #TimerAction(period=11.0, actions=[photo_node]),
    ])
