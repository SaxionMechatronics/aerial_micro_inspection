import rclpy
import math
import yaml 
import numpy as np
import os
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from transitions import Machine

from geometry_msgs.msg import QuaternionStamped
from geographic_msgs.msg import GeoPoseStamped


from std_srvs.srv import Trigger


class InspectionPlanner(Node):

    states = ['idle', 'navigate', 'inspect', 'photo']

    def __init__(self):
        super().__init__('inspection_planner')

        self.ref_gps = None
        self.viewpoints = None
        self.viewpoints_gps = self.load_viewpoints()
        
        self.navigation_index = 0
        self.gimbal_index = 0

        self.current_pose = None
        self.target_pose = None
        self.gimbal_orientation = None
        self.gimbal_time = None
        self._photo_future = None
        self.arrival_threshold_position = 0.1 #0.05 # 10cm
        self.arrival_threshold_angle = 0.02 # ~1º
        self.viewpoints_sorted=False
        
 
        self.start = self.get_clock().now()

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
            trigger='take_photo',
            source='inspect',
            dest='photo'
        )

        self.machine.add_transition(
            trigger='photo_taken',
            source='photo',
            dest='inspect'
        )

        self.machine.add_transition(
            trigger='finish_inspection',
            source='inspect',
            dest='idle'
        )

        ############## SUBRCIBERS AND PUBLISHERS ##############################

        self.inspection_viewpoint_publisher = self.create_publisher(GeoPoseStamped, 'inspection/viewpoint', 10)
        self.publisher_gimbal = self.create_publisher(QuaternionStamped,'gimbal/setpoint',10)

        self.odometry_sub = self.create_subscription(
            GeoPoseStamped,
            'inspection/gps_pose',
            self.vehicle_odometry_callback,
            10)
        
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
        self.gimbal_index = 0

        self.target_pose = self.viewpoints[self.navigation_index]

        self.get_logger().info("Going to the next viewpoint...")
            


    def on_enter_inspect(self):
        
        if self.navigation_index == len(self.viewpoints)-1:
            self.finish_inspection()

        if self.gimbal_index == len(self.target_pose['gimbal_pitch']):
            self.next_position()
        else:

            self.gimbal_time = self.get_clock().now()



    def on_enter_photo(self):

        self._photo_future = self._photo_client.call_async(Trigger.Request())

    def on_enter_idle(self):
        self.get_logger().info(f"Inspection finished in {(self.get_clock().now() - self.start).nanoseconds / 1e9} seconds")


    ################### CALLBACKS ########################

    def vehicle_odometry_callback(self,msg):
        
        if not self.ref_gps==None:
            lat=msg.pose.position.latitude
            long=msg.pose.position.longitude
            alt=msg.pose.position.altitude
            self.current_pose=self.gps_to_ned(lat,long,alt,self.ref_gps[0],self.ref_gps[1],self.ref_gps[2])
            

            if not self.viewpoints_sorted and not self.viewpoints==None:
                self.sort_viewpoints()
                self.target_pose = self.viewpoints[self.navigation_index]
                
                self.get_logger().info('Viewpoints sorted')
            
            if self.target_pose is not None and self.gimbal_orientation is not None and self.is_navigate():
                
                # Position check
                dx = self.current_pose[0] - self.target_pose['position'][0]
                dy = self.current_pose[1] - self.target_pose['position'][1]
                dz = self.current_pose[2] - self.target_pose['position'][2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                # Drone angle check
                w = msg.pose.orientation.w
                x = msg.pose.orientation.x
                y = msg.pose.orientation.y 
                z = msg.pose.orientation.z
                _,_,yaw = self.quaternion_to_euler(w,x,y,z)
                dyaw = yaw - self.target_pose['yaw']

                # Gimbal angle check, maybe usable with real drone but without proper gimbal feedback, useless
                # g_orientation = self.gimbal_orientation
                # _,g_pitch,g_yaw = self.quaternion_to_euler(g_orientation.w,g_orientation.x,g_orientation.y,g_orientation.z)
                # dpitch = g_pitch - self.target_pose['pitch'][0]
                # dg_yaw = g_yaw - 0 #TODO implement gimbal yaw check with respect required angle for oblique inspection

                #For debug only:
                self.get_logger().info(f"dx: {dx}")
                self.get_logger().info(f"dy: {dy}")
                self.get_logger().info(f"dz: {dz}")
                self.get_logger().info(f"dyaw: {dyaw}")


                
                # if dist < self.arrival_threshold_position and dyaw < self.arrival_threshold_angle and dpitch < self.arrival_threshold_angle and dg_yaw < self.arrival_threshold_angle:
                if dist < self.arrival_threshold_position and dyaw < self.arrival_threshold_angle :
                
                    self.get_logger().info("Arrived to viewpoint, requesting photo...")
                    self.arrived_position()

    def gimbal_orientation_callback(self,msg):
        self.gimbal_orientation=msg.quaternion

            

    def timer_callback(self):
        
        if self.target_pose is not None and self.is_navigate():
            gps_pose = self.viewpoints_gps[self.navigation_index]
            msg = GeoPoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.pose.position.latitude=gps_pose['position'][0]
            msg.pose.position.longitude=gps_pose['position'][1]
            msg.pose.position.altitude=gps_pose['position'][2]

            yaw=gps_pose['yaw']

            msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w =self.euler_to_quaternion(0.0,0.0,yaw)

            self.inspection_viewpoint_publisher.publish(msg)

        if self.is_inspect():
            elapsed = (self.get_clock().now() - self.gimbal_time).nanoseconds / 1e9

            x,y,z,w= self.euler_to_quaternion(0.0,self.target_pose['gimbal_pitch'][self.gimbal_index],self.target_pose['gimbal_yaw'][self.gimbal_index]-self.target_pose['yaw'])
            gimbal_msg = QuaternionStamped()
            gimbal_msg.header.stamp = self.get_clock().now().to_msg()
            gimbal_msg.header.frame_id = 'gimbal_base' 
            gimbal_msg.quaternion.x = x
            gimbal_msg.quaternion.y = y
            gimbal_msg.quaternion.z = z
            gimbal_msg.quaternion.w = w
            self.publisher_gimbal.publish(gimbal_msg)

            if elapsed > 1.5:
                self.take_photo()


        if self.is_photo():
            if not self._photo_future.done():
                return
            
            result = self._photo_future.result()
            if result.success:
                self.get_logger().info("Photo Taken")
                self.gimbal_index += 1
                self.photo_taken()
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
                "gimbal_yaw": np.array(vp["gimbal_yaw"]),
                "gimbal_pitch": np.array(vp["gimbal_pitch"]),
                "targets":   vp["targets"]
            })

        self.ref_gps = data["GPS_ref"]

        ned_viewpoints = []
        for vp in viewpoints:
            ned_vp = dict(vp)  # shallow copy, angles carry over as-is
            ned_vp["position"] = self.gps_to_ned(
                vp["position"][0], vp["position"][1], vp["position"][2],
                self.ref_gps[0], self.ref_gps[1], self.ref_gps[2]
            )
            ned_vp["targets"] = [
                self.gps_to_ned(t[0], t[1], t[2], self.ref_gps[0], self.ref_gps[1], self.ref_gps[2])
                for t in vp["targets"]
            ]
            # yaw, gimbal_yaw, gimbal_pitch → untouched, already in NED
            ned_viewpoints.append(ned_vp)

        self.viewpoints=ned_viewpoints
        self.get_logger().info("Viewpoints converted to local NED, ready to fly.")

        return viewpoints
    
    def sort_viewpoints(self):

        start_position = np.array([self.current_pose[0],self.current_pose[1],self.current_pose[2]])

        distances = [np.linalg.norm(vp["position"] - start_position) for vp in self.viewpoints]
        start_idx = int(np.argmin(distances))

        self.viewpoints = self.viewpoints[start_idx:] + self.viewpoints[:start_idx] + [self.viewpoints[start_idx]]
        self.viewpoints_gps = self.viewpoints_gps[start_idx:] + self.viewpoints_gps[:start_idx] + [self.viewpoints_gps[start_idx]]

        self.viewpoints_sorted=True  

    def gps_to_ned(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        ref_lat_deg: float,
        ref_lon_deg: float,
        ref_alt_m: float,
    ):
        """
        Converts absolute GPS (WGS84) to local NED [north, east, down] in metres,
        given the local frame origin (ref_lat/ref_lon/ref_alt ).
        """
        
        _WGS84_A  = 6_378_137.0
        _WGS84_E2 = 6.6943799901414e-3

        lat0 = math.radians(ref_lat_deg)
        lon0 = math.radians(ref_lon_deg)

        
        N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat0)**2)
        M = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * math.sin(lat0)**2)**1.5

        delta_lat = math.radians(lat_deg  - ref_lat_deg)
        delta_lon = math.radians(lon_deg  - ref_lon_deg)

        north =  delta_lat * M
        east  =  delta_lon * N * math.cos(lat0)
        down  = -(alt_m - ref_alt_m)          # up → down sign flip for NED

        return [north, east, down]

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