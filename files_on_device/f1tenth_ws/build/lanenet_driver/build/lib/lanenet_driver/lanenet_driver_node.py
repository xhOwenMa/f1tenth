#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from points_vector.msg import PointsVector
from ackermann_msgs.msg import AckermannDriveStamped

from geometry_msgs.msg import Point
from sensor_msgs.msg import Joy

from .stanley_controller import cubic_spline_planner
from .stanley_controller.stanley_controller import State, calc_target_index, pid_control, stanley_control


class LanenetDriver(Node):

    def __init__(self):
        super().__init__("lanenet_driver")
        self.get_logger().info("Started lanenet driver...")

        self.subscriber_ = self.create_subscription(PointsVector, '/lanenet_path', self.driver_callback, 1)
        self.joy_subscriber_ = self.create_subscription(Joy,'/joy', self.joy_callback, 1)
        
        # self.publisher_ = self.create_publisher(AckermannDriveStamped, "drive", 1)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 1)
        self.position_publisher_ = self.create_publisher(Point, '/position', 1)

        self.drive_exists = False

        self.ctl_loop = self.create_timer(0.05,self.main_control)
        
        self.state = State(x=0.0, y=0.0, yaw=np.radians(90.0), v=0.0)
        
        self.driving = False
        
    def driver_callback(self, data):
        self.drive_exists = True
        self.get_logger().info("New data received...")

        vector = data.points
        x_coeff = data.x_coeff
        self.ax = []
        self.ay = []

        for pt in vector:
            self.ax.append(pt.x)
            self.ay.append(pt.y)
        
        self.cx, self.cy, self.cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            self.ax, self.ay, ds=0.01)

        self.target_speed = 0.5 / 3.6  # [m/s]
        # self.target_speed = 0.3


        # Initial state
        # self.state = State(x=640*x_coeff, y=0.4, yaw=np.radians(90.0), v=0.5)
        self.state.x = 640 * x_coeff
        self.state.y = -0.584
        
        self.last_idx = len(self.cx) - 1
        self.target_idx, _ = calc_target_index(self.state, self.cx, self.cy)
        
    def joy_callback(self, data):
        new_status = (data.buttons[5] == 1)
        if self.driving and not new_status:
            self.state.v = 0.0
            self.state.yaw = np.radians(90.0)
        self.driving = new_status

    def main_control(self):
        if self.drive_exists and self.driving:
            self.publish_position()
            
            self.get_logger().info("Stanley controlling...")
            self.ai = pid_control(self.target_speed, self.state.v)
            self.di, self.target_idx = stanley_control(self.state, self.cx, self.cy, self.cyaw, self.target_idx)
            self.delta, self.v = self.state.update(self.ai, self.di)
            # self.drive_with_steer(self.delta, 0.05, self.v)
            self.drive_with_steer(self.delta, 0.05, self.v, self.ai)

    def drive_with_steer(self, steering_angle, steering_velocity, speed, acceleration):
        msg = AckermannDriveStamped()

        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "lanenet_drive"
        msg.drive.steering_angle = steering_angle
        msg.drive.steering_angle_velocity = steering_velocity
        msg.drive.speed = speed
        msg.drive.acceleration = acceleration
        msg.drive.jerk = 0.0

        self.get_logger().info("Publishing on drive...")
        self.publisher_.publish(msg)
        
    def publish_position(self):
        pt = Point()
        pt.x = self.state.x
        pt.y = self.state.y
        pt.z = self.state.yaw
        self.get_logger().info("Publishing position info...")
        self.position_publisher_.publish(pt)
        

def main(args=None):
    try:
        rclpy.init(args=args)

        lanenet_driver = LanenetDriver()

        rclpy.spin(lanenet_driver)
        lanenet_driver.destroy_node()
        rclpy.shutdown()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

