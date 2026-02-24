#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    ai_scanner_share = get_package_share_directory('ai_scanner')

    worlds_dir = '/ws/src/ai_scanner/simulation/gazebo/worlds/'
    world_path = '/ws/src/ai_scanner/simulation/gazebo/worlds/tree_car_world.sdf'
    world_name = 'tree_car_world' 
    sim_script_path = '/ws/src/ai_scanner/simulation/gazebo/scripts/simulation-gazebo'

    # --- Launch args ---
    px4_dir_arg = DeclareLaunchArgument('px4_dir', default_value='/opt/PX4-Autopilot')
    px4_make_target_arg = DeclareLaunchArgument('px4_make_target', default_value='gz_x500_inspection')
    px4_gz_model_pose_arg = DeclareLaunchArgument('px4_gz_model_pose', default_value='10,-2,0.5,0,0,0')
    world_arg = DeclareLaunchArgument('world', default_value=world_path)
    extra_resource_path_arg = DeclareLaunchArgument('extra_resource_path', default_value=worlds_dir)
    xrce_udp_port_arg = DeclareLaunchArgument('xrce_udp_port', default_value='8888')
    run_mission_arg = DeclareLaunchArgument('run_mission', default_value='true')
    run_camera_bridge_arg = DeclareLaunchArgument('run_camera_bridge', default_value='true')
    target_waypoint_x_arg = DeclareLaunchArgument('target_waypoint_x', default_value='3.0')
    target_waypoint_y_arg = DeclareLaunchArgument('target_waypoint_y', default_value='-6.5')
    target_waypoint_z_arg = DeclareLaunchArgument('target_waypoint_z', default_value='-2.0')
    target_yaw_deg_arg = DeclareLaunchArgument('target_yaw_deg', default_value='-90.0')

    # --- Launch configs ---
    px4_dir = LaunchConfiguration('px4_dir')
    px4_make_target = LaunchConfiguration('px4_make_target')
    px4_gz_model_pose = LaunchConfiguration('px4_gz_model_pose')
    world = LaunchConfiguration('world')
    extra_resource_path = LaunchConfiguration('extra_resource_path')
    xrce_udp_port = LaunchConfiguration('xrce_udp_port')
    run_mission = LaunchConfiguration('run_mission')
    run_camera_bridge = LaunchConfiguration('run_camera_bridge')
    target_waypoint_x = LaunchConfiguration('target_waypoint_x')
    target_waypoint_y = LaunchConfiguration('target_waypoint_y')
    target_waypoint_z = LaunchConfiguration('target_waypoint_z')
    target_yaw_deg = LaunchConfiguration('target_yaw_deg')

    start_px4 = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            [
                'cd ', px4_dir,
                # Apply local airframe overrides right before PX4 start.
                ' && cp /ws/src/ai_scanner/simulation/gazebo/px4_airframe/2026_gz_x500_inspection '
                '/opt/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/',
                ' && python3 /ws/src/ai_scanner/simulation/gazebo/px4_airframe/modify_px4.py',
                ' && export PX4_GZ_MODEL_POSE=', px4_gz_model_pose,
                ' && export PX4_GZ_STANDALONE=1',
                ' && export PX4_GZ_WORLD=', world_name,
                ' && make px4_sitl ', px4_make_target
            ]
        ],
        output='screen'
    )

    run_simulation_gazebo = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            [
                'pkill -f "[g]z sim" || true; '
                'pkill -f "[g]zserver" || true; '
                'pkill -f "[g]zclient" || true; '
                'python3 ', sim_script_path,
                ' --world ', world,
                ' --extra_resource_path ', extra_resource_path
            ]
        ],
        output='screen'
    )

    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', xrce_udp_port],
        output='screen'
    )

    pkg_share = get_package_share_directory("ai_scanner") 

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="camera_bridge",
        parameters=[
            {
                "config_file": os.path.join(
                    pkg_share,
                    "config",
                    "ros_gz_bridge_config.yaml",
                ),
            },
            # Useful if you bridge tf_static later; safe to keep
            {"qos_overrides./tf_static.publisher.durability": "transient_local"},
        ],
        output="screen",
    )


    mission_node = Node(
        package='ai_scanner',
        executable='px4_xrce_mission',
        name='px4_xrce_mission',
        output='screen',
        parameters=[{
            'target_waypoint_x': target_waypoint_x,
            'target_waypoint_y': target_waypoint_y,
            'target_waypoint_z': target_waypoint_z,
            'target_yaw_deg': target_yaw_deg,
        }],
        condition=IfCondition(run_mission),
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
        target_waypoint_x_arg,
        target_waypoint_y_arg,
        target_waypoint_z_arg,
        target_yaw_deg_arg,
        TimerAction(period=1.0, actions=[start_px4]),
        TimerAction(period=3.0, actions=[run_simulation_gazebo]),
        TimerAction(period=6.0, actions=[xrce_agent]),
        TimerAction(period=8.0, actions=[bridge]),
        TimerAction(period=9.0, actions=[mission_node]),
    ])
