#!/usr/bin/env python

import math
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path

import yaml 
import numpy as np
import os

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry


class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        # viewpoints = self.load_viewpoints()

        # self.declare_parameter('target_waypoint_x', viewpoints[100]["position"][0])
        # self.declare_parameter('target_waypoint_y', viewpoints[100]["position"][1])
        # self.declare_parameter('target_waypoint_z', viewpoints[100]["position"][2])
        # self.declare_parameter('target_yaw_deg', viewpoints[100]["yaw"])
        # self.declare_parameter('target_pitch_rad', viewpoints[100]["pitch"][0])

        self.inspection_viewpoint_x = 0.0
        self.inspection_viewpoint_y = 0.0
        self.inspection_viewpoint_z = 0.0
        self.inspection_viewpoint_yaw = 0.0
        self.inspection_viewpoint_recieved = False

        self.declare_parameter('takeoff_altitude_m', 5.0)
        self.declare_parameter('takeoff_hold_s', 20.0)

                # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
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
        self.position_sub = self.create_subscription(
            Pose,
            'inspection/viewpoint',
            self.inspection_viewpoint_callback,
            qos_profile_sub)
        self.odometry_sub = self.create_subscription(
            VehicleOdometry,
            'fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_profile_sub)
        
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_vehicle_command = self.create_publisher(VehicleCommand, 'fmu/in/vehicle_command', qos_profile_pub)
        self.path_publisher = self.create_publisher(Path, "inspection/robot_path", 10)

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

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"  


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

    def inspection_viewpoint_callback(self, msg:Pose):
        self.inspection_viewpoint_x = msg.position.x
        self.inspection_viewpoint_y = msg.position.y
        self.inspection_viewpoint_z = msg.position.z

        x=msg.orientation.x
        y=msg.orientation.y
        z=msg.orientation.z
        w=msg.orientation.w
        _,_,self.inspection_viewpoint_yaw=self.quaternion_to_euler(w,x,y,z)

        self.inspection_viewpoint_recieved = True

    def vehicle_odometry_callback(self, msg):
        pose_stamped = PoseStamped()

        # --- Header ---
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "odom"

        # --- Position: NED -> ENU ---
        pose_stamped.pose.position.x =  float(msg.position[1])  # ENU.x = NED.y
        pose_stamped.pose.position.y =  float(msg.position[0])  # ENU.y = NED.x
        pose_stamped.pose.position.z = -float(msg.position[2])  # ENU.z = -NED.z

        # --- Orientation: NED -> ENU ---
        # PX4 quaternion is [w, x, y, z], geometry_msgs expects [x, y, z, w]
        q_ned = R.from_quat([
            float(msg.q[1]),  # x
            float(msg.q[2]),  # y
            float(msg.q[3]),  # z
            float(msg.q[0])   # w
        ])

        # Fixed rotation to go from NED to ENU frame
        ned_to_enu = R.from_euler('x', 180, degrees=True)
        q_enu = ned_to_enu * q_ned
        q = q_enu.as_quat()  # returns [x, y, z, w]

        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]

        # --- Append to path and publish ---
        self.path_msg.poses.append(pose_stamped)
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_publisher.publish(self.path_msg)
            

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
            if takeoff_elapsed_s < float(self.get_parameter('takeoff_hold_s').value) or not self.inspection_viewpoint_recieved:
                trajectory_msg.position[0] = 0.0
                trajectory_msg.position[1] = 0.0
                trajectory_msg.position[2] = -abs(float(self.get_parameter('takeoff_altitude_m').value))
            else:
                trajectory_msg.position[0] = self.inspection_viewpoint_x
                trajectory_msg.position[1] = self.inspection_viewpoint_y
                trajectory_msg.position[2] = self.inspection_viewpoint_z

        trajectory_msg.yaw = self.inspection_viewpoint_yaw
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

    def load_viewpoints(self,relative_path="../../config/path.yaml"):
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
                "yaw": vp["yaw"],
                "pitch": vp["pitch"],
                "target":   np.array(vp["target"]),
                "normal":   np.array(vp["normal"]) if "normal" in vp else None
            })

        return viewpoints
    
    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float):
        """
        Convert Euler angles (in radians) to a quaternion.
        
        Uses the ZYX convention (yaw → pitch → roll), which is
        the standard used in ROS2 / aerospace applications.

        Args:
            roll:  Rotation around X-axis (radians)
            pitch: Rotation around Y-axis (radians)
            yaw:   Rotation around Z-axis (radians)

        Returns:
            (x, y, z, w) quaternion tuple
        """
        cy = math.cos(yaw   * 0.5)
        sy = math.sin(yaw   * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll  * 0.5)
        sr = math.sin(roll  * 0.5)

        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy

        return x, y, z, w


    def quaternion_to_euler(self, w, x, y, z, degrees=False):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            w, x, y, z: Quaternion components
            degrees: If True, return angles in degrees; otherwise radians
        
        Returns:
            (roll, pitch, yaw) tuple
        """

        # Roll (rotation around X-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (rotation around Y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Gimbal lock: clamp to ±90°
        else:
            pitch = math.asin(sinp)
        

        # Yaw (rotation around Z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if degrees:
            roll  = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw   = math.degrees(yaw)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()