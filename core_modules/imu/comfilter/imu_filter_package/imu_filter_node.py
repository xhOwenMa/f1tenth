import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import math
from rclpy.node import Node
import rclpy.logging
import numpy as np


class IMUFilterNode(Node):
    def __init__(self):
        super().__init__('imu_filter_node')
        self.subscription = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Complementary filter parameters
        self.alpha = 0.98  # Adjust this parameter based on your requirements
        self.last_time = self.get_clock().now()

        # Initialize orientation estimates
        self.pitch = 0.0
        self.roll = 0.0

        # Initialize speed estimates
        self.current_speed = 0.0
        self.last_acc = []  # last acceleration in x, y, z
        self.ave_acc = [0.0, 0.0, 0.0]
        # Speed and ZUPT parameters
        self.stationary_threshold = 0.05  # Threshold for considering the device stationary
        self.stationary_time = 0.0  # Time for which the device has been stationary
        self.stationary_time_threshold = 1.0  # Time threshold to trigger ZUPT

        # imu filter publiser
        self.speed_publisher = self.create_publisher(Float64, '/speed', 10)
        self.get_logger().info("imu filter set up done.")

    def imu_callback(self, msg):
        # Extract accelerometer and gyroscope data
        acc_x = msg.linear_acceleration.x + 0.14
        acc_y = msg.linear_acceleration.y - 0.5
        acc_z = msg.linear_acceleration.z

        if len(self.last_acc) < 10:
            # Append the new acceleration reading
            self.last_acc.append([acc_x, acc_y, acc_z])
        else:
            # Once we have 9 readings, calculate the average
            self.ave_acc = np.array(self.last_acc)
            self.ave_acc = np.mean(self.ave_acc, axis=0)

            # Update the list with the new reading, removing the oldest one
            self.last_acc.pop(0)
            self.last_acc.append([acc_x, acc_y, acc_z])

        gyro_x = msg.angular_velocity.x
        gyro_y = msg.angular_velocity.y
        gyro_z = msg.angular_velocity.z

        # # Compute accelerometer angles
        # acc_pitch = math.atan2(acc_y, acc_z)
        # acc_roll = math.atan2(-acc_x, math.sqrt(acc_y**2 + acc_z**2))

        # Time difference
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # # Complementary filter
        # self.pitch = self.alpha * (self.pitch + gyro_x * dt) + (1 - self.alpha) * acc_pitch
        # self.roll = self.alpha * (self.roll + gyro_y * dt) + (1 - self.alpha) * acc_roll

        # Calculate norm of the acceleration
        acc_norm = math.sqrt(self.ave_acc[0] ** 2)
        # Detect stationary periods for ZUPT
        if acc_norm < self.stationary_threshold:
            self.stationary_time += dt
            if self.stationary_time > self.stationary_time_threshold:
                self.current_speed = 0.0  # Reset speed on ZUPT
        else:
            self.stationary_time = 0.0  # Reset stationary time
        # Update speed if not stationary
        if self.stationary_time == 0.0:
            self.current_speed += acc_norm * dt

        speed_msg = Float64()
        speed_msg.data = self.current_speed
        self.speed_publisher.publish(speed_msg)

        # Do something with the filtered orientation data
        self.get_logger().info(f"\nx_acc{self.ave_acc[0]}, \ny_acc{self.ave_acc[1]}, \nz_acc{self.ave_acc[2]}")


def main(args=None):
    rclpy.init(args=args)
    imu_filter_node = IMUFilterNode()
    rclpy.spin(imu_filter_node)
    imu_filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

