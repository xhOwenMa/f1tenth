import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import rclpy.logging
import serial
import math
import sys
import time
from tf_transformations import quaternion_from_euler

degrees2rad = math.pi / 180.0
imu_yaw_calibration = 0.0


class RazorIMUNode(Node):
    def __init__(self):
        super().__init__('imu_ros2_node')

        # Declare parameters
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('topic', '/imu')
        self.declare_parameter('frame_id', 'base_imu_link')
        self.declare_parameter('accel_x_min', -250.0)
        self.declare_parameter('accel_x_max', 250.0)
        self.declare_parameter('accel_y_min', -250.0)
        self.declare_parameter('accel_y_max', 250.0)
        self.declare_parameter('accel_z_min', -250.0)
        self.declare_parameter('accel_z_max', 250.0)
        self.declare_parameter('magn_x_min', -600.0)
        self.declare_parameter('magn_x_max', 600.0)
        self.declare_parameter('magn_y_min', -600.0)
        self.declare_parameter('magn_y_max', 600.0)
        self.declare_parameter('magn_z_min', -600.0)
        self.declare_parameter('magn_z_max', 600.0)
        self.declare_parameter('calibration_magn_use_extended', False)
        self.declare_parameter('magn_ellipsoid_center', [0, 0, 0])
        # self.declare_parameter('magn_ellipsoid_transform', [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.magn_ellipsoid_transform = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.declare_parameter('imu_yaw_calibration', 0.0)
        self.declare_parameter('gyro_average_offset_x', 0.0)
        self.declare_parameter('gyro_average_offset_y', 0.0)
        self.declare_parameter('gyro_average_offset_z', 0.0)

        self.port = self.get_parameter('port').value
        self.topic = self.get_parameter('topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.accel_x_min = self.get_parameter('accel_x_min').value
        self.accel_x_max = self.get_parameter('accel_x_max').value
        self.accel_y_min = self.get_parameter('accel_y_min').value
        self.accel_y_max = self.get_parameter('accel_y_max').value
        self.accel_z_min = self.get_parameter('accel_z_min').value
        self.accel_z_max = self.get_parameter('accel_z_max').value
        self.magn_x_min = self.get_parameter('magn_x_min').value
        self.magn_x_max = self.get_parameter('magn_x_max').value
        self.magn_y_min = self.get_parameter('magn_y_min').value
        self.magn_y_max = self.get_parameter('magn_y_max').value
        self.magn_z_min = self.get_parameter('magn_z_min').value
        self.magn_z_max = self.get_parameter('magn_z_max').value
        self.calibration_magn_use_extended = self.get_parameter('calibration_magn_use_extended').value
        self.magn_ellipsoid_center = self.get_parameter('magn_ellipsoid_center').value
        # self.magn_ellipsoid_transform = self.get_parameter('magn_ellipsoid_transform').value
        self.imu_yaw_calibration = self.get_parameter('imu_yaw_calibration').value
        self.gyro_average_offset_x = self.get_parameter('gyro_average_offset_x').value
        self.gyro_average_offset_y = self.get_parameter('gyro_average_offset_y').value
        self.gyro_average_offset_z = self.get_parameter('gyro_average_offset_z').value

        self.publisher_ = self.create_publisher(Imu, self.topic, 10)

        # Add other initializations here...
        self.ser = None
        self.imuMsg = Imu()
        self.setup_imu_message()
        self.errcount = 0
        self.seq = 0
        self.accel_factor = 0

        # Add serial port setup and other logic...
        # self.ser = serial.Serial(port=self.port, baudrate=57600, timeout=1)

        # self.publish_imu_data()
        # Call publish_imu_data every second (1.0 second interval)
        # self.timer = self.create_timer(0.1, self.publish_imu_data)
        self.publish_imu_data()

    def setup_imu_message(self):
        # Orientation covariance estimation:
        # Observed orientation noise: 0.3 degrees in x, y, 0.6 degrees in z
        # Magnetometer linearity: 0.1% of full scale (+/- 2 gauss) => 4 milligauss
        # Earth's magnetic field strength is ~0.5 gauss, so magnetometer nonlinearity could
        # cause ~0.8% yaw error (4mgauss/0.5 gauss = 0.008) => 2.8 degrees, or 0.050 radians
        # i.e. variance in yaw: 0.0025
        # Accelerometer non-linearity: 0.2% of 4G => 0.008G. This could cause
        # static roll/pitch error of 0.8%, owing to gravity orientation sensing
        # error => 2.8 degrees, or 0.05 radians. i.e. variance in roll/pitch: 0.0025
        # so set all covariances the same.
        self.imuMsg.orientation_covariance = [
            0.0025, 0.0, 0.0,
            0.0, 0.0025, 0.0,
            0.0, 0.0, 0.0025
        ]

        # Angular velocity covariance estimation:
        # Observed gyro noise: 4 counts => 0.28 degrees/sec
        # nonlinearity spec: 0.2% of full scale => 8 degrees/sec = 0.14 rad/sec
        # Choosing the larger (0.14) as std dev, variance = 0.14^2 ~= 0.02
        self.imuMsg.angular_velocity_covariance = [
            0.02, 0.0, 0.0,
            0.0, 0.02, 0.0,
            0.0, 0.0, 0.02
        ]

        # linear acceleration covariance estimation:
        # observed acceleration noise: 5 counts => 20milli-G's ~= 0.2m/s^2
        # nonliniarity spec: 0.5% of full scale => 0.2m/s^2
        # Choosing 0.2 as std dev, variance = 0.2^2 = 0.04
        self.imuMsg.linear_acceleration_covariance = [
            0.04, 0.0, 0.0,
            0.0, 0.04, 0.0,
            0.0, 0.0, 0.04
        ]

        # Setup the IMU message, like setting covariance values
        try:
            self.ser = serial.Serial(port=self.port, baudrate=57600, timeout=1)
            # ser = serial.Serial(port=port, baudrate=57600, timeout=1, rtscts=True, dsrdtr=True) # For compatibility with some virtual serial ports (e.g. created by socat) in Python 2.7
        except serial.serialutil.SerialException:
            self.get_logger().info(
                "IMU not found at port " + self.port + ". Did you specify the correct port in the launch file?")
            # exit
            sys.exit(2)

        self.accel_factor = 9.806 / 256.0  # sensor reports accel as 256.0 = 1G (9.8m/s^2). Convert to m/s^2.
        self.get_logger().info("Giving the razor IMU board 5 seconds to boot...")
        time.sleep(5)  # Sleep for 5 seconds to wait for the board to boot

        ### configure board ###
        # stop datastream
        self.ser.write(('#o0').encode("utf-8"))

        # discard old input
        # automatic flush - NOT WORKING
        # ser.flushInput()  #discard old input, still in invalid format
        # flush manually, as above command is not working
        discard = self.ser.readlines()

        # set output mode
        self.ser.write(('#ox').encode("utf-8"))  # To start display angle and sensor reading in text

        self.get_logger().info("Writing calibration values to razor IMU board...")
        # set calibration values
        self.ser.write(('#caxm' + str(self.accel_x_min)).encode("utf-8"))
        self.ser.write(('#caxM' + str(self.accel_x_max)).encode("utf-8"))
        self.ser.write(('#caym' + str(self.accel_y_min)).encode("utf-8"))
        self.ser.write(('#cayM' + str(self.accel_y_max)).encode("utf-8"))
        self.ser.write(('#cazm' + str(self.accel_z_min)).encode("utf-8"))
        self.ser.write(('#cazM' + str(self.accel_z_max)).encode("utf-8"))

        if (not self.calibration_magn_use_extended):
            self.ser.write(('#cmxm' + str(self.magn_x_min)).encode("utf-8"))
            self.ser.write(('#cmxM' + str(self.magn_x_max)).encode("utf-8"))
            self.ser.write(('#cmym' + str(self.magn_y_min)).encode("utf-8"))
            self.ser.write(('#cmyM' + str(self.magn_y_max)).encode("utf-8"))
            self.ser.write(('#cmzm' + str(self.magn_z_min)).encode("utf-8"))
            self.ser.write(('#cmzM' + str(self.magn_z_max)).encode("utf-8"))
        else:
            self.ser.write(('#ccx' + str(self.magn_ellipsoid_center[0])).encode("utf-8"))
            self.ser.write(('#ccy' + str(self.magn_ellipsoid_center[1])).encode("utf-8"))
            self.ser.write(('#ccz' + str(self.magn_ellipsoid_center[2])).encode("utf-8"))
            self.ser.write(('#ctxX' + str(self.magn_ellipsoid_transform[0][0])).encode("utf-8"))
            self.ser.write(('#ctxY' + str(self.magn_ellipsoid_transform[0][1])).encode("utf-8"))
            self.ser.write(('#ctxZ' + str(self.magn_ellipsoid_transform[0][2])).encode("utf-8"))
            self.ser.write(('#ctyX' + str(self.magn_ellipsoid_transform[1][0])).encode("utf-8"))
            self.ser.write(('#ctyY' + str(self.magn_ellipsoid_transform[1][1])).encode("utf-8"))
            self.ser.write(('#ctyZ' + str(self.magn_ellipsoid_transform[1][2])).encode("utf-8"))
            self.ser.write(('#ctzX' + str(self.magn_ellipsoid_transform[2][0])).encode("utf-8"))
            self.ser.write(('#ctzY' + str(self.magn_ellipsoid_transform[2][1])).encode("utf-8"))
            self.ser.write(('#ctzZ' + str(self.magn_ellipsoid_transform[2][2])).encode("utf-8"))

        self.ser.write(('#cgx' + str(self.gyro_average_offset_x)).encode("utf-8"))
        self.ser.write(('#cgy' + str(self.gyro_average_offset_y)).encode("utf-8"))
        self.ser.write(('#cgz' + str(self.gyro_average_offset_z)).encode("utf-8"))

        # print calibration values for verification by user
        self.ser.flushInput()
        self.ser.write(('#p').encode("utf-8"))
        calib_data = self.ser.readlines()
        calib_data_print = "Printing set calibration values:\r\n"
        for row in calib_data:
            line = bytearray(row).decode("utf-8")
            calib_data_print += line
        self.get_logger().info(calib_data_print)

        # start datastream
        self.ser.write(('#o1').encode("utf-8"))

        # automatic flush - NOT WORKING
        # ser.flushInput()  #discard old input, still in invalid format
        # flush manually, as above command is not working - it breaks the serial connection
        self.get_logger().info("Flushing first 200 IMU entries...")
        for x in range(0, 200):
            line = bytearray(self.ser.readline()).decode("utf-8")
        self.get_logger().info("Publishing IMU data...")
        # f = open("raw_imu_data.log", 'w')

    def publish_imu_data(self):
        while self.errcount < 10:
            line = bytearray(self.ser.readline()).decode("utf-8")
            if ((line.find("#YPRAG=") == -1) or (line.find("\r\n") == -1)):
                self.get_logger().warn("Bad IMU data or bad sync")
                self.errcount = self.errcount + 1
                return
            line = line.replace("#YPRAG=", "")  # Delete "#YPRAG="
            # f.write(line)                     # Write to the output log file
            line = line.replace("\r\n", "")  # Delete "\r\n"
            words = line.split(",")  # Fields split
            if len(words) != 9:
                self.get_logger().warn("Bad IMU data or bad sync")
                self.errcount = self.errcount + 1
                return
            else:
                self.errcount = 0
                # in AHRS firmware z axis points down, in ROS z axis points up (see REP 103)
                yaw_deg = -float(words[0])
                yaw_deg = yaw_deg + imu_yaw_calibration
                if yaw_deg > 180.0:
                    yaw_deg = yaw_deg - 360.0
                if yaw_deg < -180.0:
                    yaw_deg = yaw_deg + 360.0
                self.yaw = yaw_deg * degrees2rad
                # in AHRS firmware y axis points right, in ROS y axis points left (see REP 103)
                self.pitch = -float(words[1]) * degrees2rad
                self.roll = float(words[2]) * degrees2rad

                # Publish message
                # AHRS firmware accelerations are negated
                # This means y and z are correct for ROS, but x needs reversing
                self.imuMsg.linear_acceleration.x = -float(words[3]) * self.accel_factor
                self.imuMsg.linear_acceleration.y = float(words[4]) * self.accel_factor
                self.imuMsg.linear_acceleration.z = float(words[5]) * self.accel_factor

                self.imuMsg.angular_velocity.x = float(words[6])
                # in AHRS firmware y axis points right, in ROS y axis points left (see REP 103)
                self.imuMsg.angular_velocity.y = -float(words[7])
                # in AHRS firmware z axis points down, in ROS z axis points up (see REP 103)
                self.imuMsg.angular_velocity.z = -float(words[8])

            q = quaternion_from_euler(self.roll, self.pitch, self.yaw)
            self.imuMsg.orientation.x = q[0]
            self.imuMsg.orientation.y = q[1]
            self.imuMsg.orientation.z = q[2]
            self.imuMsg.orientation.w = q[3]
            self.imuMsg.header.stamp = self.get_clock().now().to_msg()
            self.imuMsg.header.frame_id = self.frame_id
            # self.imuMsg.header.seq = self.seq
            # self.seq = self.seq + 1
            self.publisher_.publish(self.imuMsg)
        self.ser.close
        self.get_logger().error("imu_ros2 stopped because of too many warnings.")


def main(args=None):
    rclpy.init(args=args)
    imu_ros2_node = RazorIMUNode()

    try:
        rclpy.spin(imu_ros2_node)
    except KeyboardInterrupt:
        pass
    finally:
        imu_ros2_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
