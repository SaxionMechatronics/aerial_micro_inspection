#!/usr/bin/env python

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import yaml 
import numpy as np
import os

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleStatus


class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        viewpoints = self.load_viewpoints()

        self.declare_parameter('target_waypoint_x', viewpoints[0]["position"][0])
        self.declare_parameter('target_waypoint_y', viewpoints[0]["position"][1])
        self.declare_parameter('target_waypoint_z', viewpoints[0]["position"][2])
        self.declare_parameter('target_yaw_deg', -90.0)#-90
        self.declare_parameter('takeoff_altitude_m', 5.0)
        self.declare_parameter('takeoff_hold_s', 20.0)

                # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile_sub)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_vehicle_command = self.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', qos_profile_pub)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.offboard_setpoint_counter = 0
        self.command_interval_cycles = 50
        self.takeoff_phase_start_s = None
        self.local_x = 0.0
        self.local_y = 0.0
        self.has_local_position = False
        self.takeoff_x = 0.0
        self.takeoff_y = 0.0

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        vehicle_command = VehicleCommand()
        vehicle_command.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        vehicle_command.command = command
        vehicle_command.param1 = float(param1)
        vehicle_command.param2 = float(param2)
        vehicle_command.target_system = 1
        vehicle_command.target_component = 1
        vehicle_command.source_system = 1
        vehicle_command.source_component = 1
        vehicle_command.from_external = True
        self.publisher_vehicle_command.publish(vehicle_command)

    def vehicle_status_callback(self, msg):
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def vehicle_local_position_callback(self, msg):
        self.local_x = float(msg.x)
        self.local_y = float(msg.y)
        self.has_local_position = True

    def cmdloop_callback(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        now_s = self.get_clock().now().nanoseconds / 1e9

        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now_us
        offboard_msg.position=True
        offboard_msg.velocity=False
        offboard_msg.acceleration=False
        self.publisher_offboard_mode.publish(offboard_msg)

        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.timestamp = now_us

        if self.takeoff_phase_start_s is None:
            trajectory_msg.position[0] = 0.0
            trajectory_msg.position[1] = 0.0
            trajectory_msg.position[2] = -abs(float(self.get_parameter('takeoff_altitude_m').value))
        else:
            takeoff_elapsed_s = now_s - self.takeoff_phase_start_s
            if takeoff_elapsed_s < float(self.get_parameter('takeoff_hold_s').value):
                trajectory_msg.position[0] = 0.0
                trajectory_msg.position[1] = 0.0
                trajectory_msg.position[2] = -abs(float(self.get_parameter('takeoff_altitude_m').value))
            else:
                trajectory_msg.position[0] = self.get_parameter('target_waypoint_x').value
                trajectory_msg.position[1] = self.get_parameter('target_waypoint_y').value
                trajectory_msg.position[2] = self.get_parameter('target_waypoint_z').value

        trajectory_msg.yaw = math.radians(float(self.get_parameter('target_yaw_deg').value))
        self.publisher_trajectory.publish(trajectory_msg)

        if self.offboard_setpoint_counter < 10:
            self.offboard_setpoint_counter += 1
            return

        if (
            self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and self.offboard_setpoint_counter % self.command_interval_cycles == 0
        ):
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        if (
            self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and self.arming_state != VehicleStatus.ARMING_STATE_ARMED
            and self.offboard_setpoint_counter % self.command_interval_cycles == 0
        ):
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

        if (
            self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and self.arming_state == VehicleStatus.ARMING_STATE_ARMED
            and self.takeoff_phase_start_s is None
        ):
            self.takeoff_x = 0.0
            self.takeoff_y = 0.0
            self.takeoff_phase_start_s = now_s

        self.offboard_setpoint_counter += 1

    def load_viewpoints(self,relative_path="../config/viewpoints.yaml"):
        """
        Loads viewpoints from a YAML file.
        Parameters:
            relative_path (str): Path to the YAML file relative to the calling script
        Returns:
            list of dicts with 'position' and 'target' as numpy arrays
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, relative_path)

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        viewpoints = []
        for vp in data["viewpoints"]:
            viewpoints.append({
                "position": np.array(vp["position"]),
                "target":   np.array(vp["target"]),
                "normal":   np.array(vp["normal"]) if "normal" in vp else None
            })

        return viewpoints



def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()