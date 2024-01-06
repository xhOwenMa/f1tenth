#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from points_vector.msg import PointsVector
from ackermann_msgs.msg import AckermannDriveStamped
from drive_msg.msg import Drivemsg

from geometry_msgs.msg import Point
from sensor_msgs.msg import Joy

from sensor_msgs.msg import Imu

from .stanley_controller import cubic_spline_planner
from .stanley_controller.stanley_controller import State, calc_target_index, pid_control, stanley_control


class LanenetDriver(Node):

    def __init__(self):
        super().__init__("lanenet_driver")
        self.get_logger().info("Started lanenet driver...")

        # self.subscriber_ = self.create_subscription(PointsVector, '/lanenet_path', self.driver_callback, 1)
        # self.joy_subscriber_ = self.create_subscription(Joy,'/joy', self.joy_callback, 1)
        self.drive_subscriber_ = self.create_subscription(Drivemsg, '/drive_ts', self.drive_callback, 1)
        # self.imu_subscriber_ = self.create_subscription(Imu, '/imu', self.imu_callback, 1)

        # self.publisher_ = self.create_publisher(AckermannDriveStamped, "drive", 1)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 1)
        self.position_publisher_ = self.create_publisher(Point, '/position', 1)

        self.drive_ts_exists = False
        self.throttle = 0.0
        self.steer = 0.0

        self.ctl_loop = self.create_timer(0.05, self.main_control)

        self.state = State(x=0.0, y=0.0, yaw=np.radians(90.0), v=0.0)

        # modified this from False to True
        self.driving = True

    def drive_callback(self, data):
        self.drive_ts_exists = True
        self.get_logger().info("New data received...")

        self.throttle = data.throttle
        self.steer = data.steer

    def joy_callback(self, data):
        new_status = (data.buttons[5] == 1)
        if self.driving and not new_status:
            self.state.v = 0.0
            self.state.yaw = np.radians(90.0)
        self.driving = new_status

    def imu_callback(self, data):
        try:
            self.get_logger().info("receive Imu data...")
            self.get_logger().info(f"{data}")
        except Exception as e:
            self.get_logger().info(f"Problem receiving Imu data: {e}")

    def main_control(self):
        if self.drive_ts_exists:
            self.publish_position()

            self.get_logger().info("Driving based on ts data")

            self.drive_with_steer(self.throttle, self.steer)

    def drive_with_steer(self, acceleration, steer, speed=0.5 / 2):
        msg = AckermannDriveStamped()

        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "lanenet_drive"
        msg.drive.steering_angle = steer
        msg.drive.steering_angle_velocity = 0.05
        msg.drive.speed = 1.0
        msg.drive.acceleration = 0.15
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

