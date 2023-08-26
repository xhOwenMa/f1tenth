#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from ackermann_msgs.msg import AckermannDriveStamped


class PlannedPath(Node):

    def __init__(self):
        super().__init__("planned_path")
        self.get_logger().info("Started planned path...")

        self._publisher = self.create_publisher(AckermannDriveStamped, "drive", 1)
        
        self.start_time = self.get_clock().now()
        self.seg_1 = Duration(seconds=6)
        self.seg_2 = Duration(seconds=7,nanoseconds=4*10**8) 
        self.seg_3 = Duration(seconds=8) 

        self.ctl_loop = self.create_timer(0.2,self.main_control)
        
    
    def main_control(self):
        self.cur_duration = self.get_clock().now()-self.start_time

        if self.cur_duration < self.seg_1:
            self.drive_fwd()
            self.get_logger().info('Publishing: driving forward...')
        elif self.cur_duration < self.seg_2:
            self.drive_turn_right()
            self.get_logger().info('Publishing: turning right...')
        elif self.cur_duration < self.seg_3:
            self.drive_fwd()
            self.get_logger().info('Publishing: driving forward...')


    def drive_fwd(self):
        msg = AckermannDriveStamped()

        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "planned_path"
        msg.drive.steering_angle = 0.0
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = 1.0
        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0

        self._publisher.publish(msg)

    def drive_turn_right(self):
        msg = AckermannDriveStamped()

        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = "planned_path"
        msg.drive.steering_angle = -0.7
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = 1.0
        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0

        self._publisher.publish(msg)

    def stop_timer(self):
        self.period.destroy()
        

def main(args=None):
    try:
        rclpy.init(args=args)
        path_publisher = PlannedPath()
        rclpy.spin(path_publisher)
        path_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

