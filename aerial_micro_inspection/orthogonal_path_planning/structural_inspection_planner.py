import rclpy
import math
import yaml 
import numpy as np
import os
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from transitions import Machine

from geometry_msgs.msg import Pose,QuaternionStamped
from px4_msgs.msg import VehicleOdometry

from std_srvs.srv import Trigger


class InspectionPlanner(Node):

    states = ['idle', 'navigate', 'inspect']

    def __init__(self):
        super().__init__('inspection_planner')

        self.viewpoints = self.load_viewpoints()
        self.navigation_index = 0

        self.current_pose = None
        self.target_pose = None
        self.gimbal_orientation = None
        self.gimbal_time = None
        self._photo_future = None
        self.arrival_threshold_position = 0.1 #0.05 # 5cm
        self.arrival_threshold_angle = 0.02 # ~1º
        self.viewpoints_sorted=False

        self.machine = Machine(
            model=self,
            states=InspectionPlanner.states,
            initial='navigate'
        )
        

        ############## MACHINE TRANSITIONS ####################################

        self.machine.add_transition(
            trigger='start_navigation',
            source='idle',
            dest='navigate'
        )

        self.machine.add_transition(
            trigger='arrived_position',
            source='navigate',
            dest='inspect'
        )

        self.machine.add_transition(
            trigger='next_position',
            source='inspect',
            dest='navigate'
        )

        self.machine.add_transition(
            trigger='finish_inspection',
            source='navigate',
            dest='idle'
        )

        ############## SUBRCIBERS AND PUBLISHERS ##############################

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.inspection_viewpoint_publisher = self.create_publisher(Pose, 'inspection/viewpoint', 10)

        self.odometry_sub = self.create_subscription(
            VehicleOdometry,
            'fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_profile_sub)
        
        self.gimbal_sub = self.create_subscription(
            QuaternionStamped,
            'gimbal_orientation',
            self.gimbal_orientation_callback,
            10
        )

        self._photo_client = self.create_client(Trigger, 'inspection/take_photo')
        if not self._photo_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('take_photo service not available after 5s!')


        ############ TIMERS #######################
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Inspection has started.')

    ################### ON ENTER #########################

    def on_enter_navigate(self):

        self.navigation_index += 1

        if self.navigation_index == len(self.viewpoints):
            self.finish_inspection()
        else:
            self.target_pose = self.viewpoints[self.navigation_index]
            self.gimbal_time = self.get_clock().now()


    def on_enter_inspect(self):

        self._photo_future = self._photo_client.call_async(Trigger.Request())

    def on_enter_idle(self):
        self.get_logger().info("Inspection finished")


    ################### CALLBACKS ########################

    def vehicle_odometry_callback(self,msg):
        self.current_pose=msg

        if not self.viewpoints_sorted:
            self.sort_viewpoints()
            self.target_pose = self.viewpoints[self.navigation_index]
            self.gimbal_time = self.get_clock().now()
            self.get_logger().info('Viewpoints sorted')
        
        if self.target_pose is not None and self.gimbal_orientation is not None and self.is_navigate():
            
            # Position check
            dx = self.current_pose.position[0] - self.target_pose['position'][0]
            dy = self.current_pose.position[1] - self.target_pose['position'][1]
            dz = self.current_pose.position[2] - self.target_pose['position'][2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Drone angle check
            w = self.current_pose.q[0]
            x = self.current_pose.q[1]
            y = self.current_pose.q[2]
            z = self.current_pose.q[3]
            _,_,yaw = self.quaternion_to_euler(w,x,y,z)
            dyaw = yaw - self.target_pose['yaw']

            # Gimbal angle check, maybe usable with real drone but without proper gimbal feedback, useless
            g_orientation = self.gimbal_orientation
            _,g_pitch,g_yaw = self.quaternion_to_euler(g_orientation.w,g_orientation.x,g_orientation.y,g_orientation.z)
            dpitch = g_pitch - self.target_pose['pitch'][0]
            dg_yaw = g_yaw - 0 #TODO implement gimbal yaw check with respect required angle for oblique inspection

            # if not dist < self.arrival_threshold_position:
            #     self.get_logger().info('Position not reached')
            # if not dyaw < self.arrival_threshold_angle:
            #     self.get_logger().info('Drone yaw not reached')
            # if not dpitch < self.arrival_threshold_angle:
            #     self.get_logger().info('Gimbal pitch not reached')
            # if not dg_yaw < self.arrival_threshold_angle:
            #     self.get_logger().info('Gimbal yaw not reached')

            elapsed = (self.get_clock().now() - self.gimbal_time).nanoseconds / 1e9
            # if dist < self.arrival_threshold_position and dyaw < self.arrival_threshold_angle and dpitch < self.arrival_threshold_angle and dg_yaw < self.arrival_threshold_angle:
            if dist < self.arrival_threshold_position and dyaw < self.arrival_threshold_angle and elapsed > 1.0:
            
                self.get_logger().info("Arrived to viewpoint, requesting photo...")
                self.arrived_position()

    def gimbal_orientation_callback(self,msg):
        self.gimbal_orientation=msg.quaternion

    def timer_callback(self):
        
        if self.target_pose is not None and self.is_navigate():
            msg = Pose()
            msg.position.x=self.target_pose['position'][0]
            msg.position.y=self.target_pose['position'][1]
            msg.position.z=self.target_pose['position'][2]

            pitch=self.target_pose['pitch'][0]
            yaw=self.target_pose['yaw']

            msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w=self.euler_to_quaternion(0.0,pitch,yaw)

            self.inspection_viewpoint_publisher.publish(msg)

        if self.is_inspect():
            if not self._photo_future.done():
                return
            
            result = self._photo_future.result()
            if result.success:
                self.get_logger().info("Photo Taken")
                self.next_position()
            else:
                self.get_logger().warn(f'Photo failed: {result.message}, retrying...')
                self._photo_future = self._photo_client.call_async(Trigger.Request())




    ################### AUXILIAR FUNCTIONS ##########################
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
    
    def sort_viewpoints(self):

        start_position = np.array([self.current_pose.position[0],self.current_pose.position[1],self.current_pose.position[2]])

        distances = [np.linalg.norm(vp["position"] - start_position) for vp in self.viewpoints]
        start_idx = int(np.argmin(distances))

        self.viewpoints = self.viewpoints[start_idx:] + self.viewpoints[:start_idx] + [self.viewpoints[start_idx]]

        self.viewpoints_sorted=True  

def main(args=None):
    rclpy.init(args=args)

    node = InspectionPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()