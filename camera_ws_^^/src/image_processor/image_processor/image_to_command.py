#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped

import torch
import torch.nn as nn
import joblib
from pathlib import Path
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

from lane_detection.lane_detector import LaneDetector  # TODO: modify where this is


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Adjust the sizes
        self.fc2 = nn.Linear(50, 20)  # Adjust the sizes
        self.fc3 = nn.Linear(20, 2)  # Output size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImageToCommand(Node):
    """Handles feeding camera frame into lanenet, converting outputs into path to be followed, and publishes that
    path."""

    def __init__(self):
        super().__init__('image_to_command')
        self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.process_image, 1)
        self.position_subscriber_ = self.create_subscription(Point, '/position', self.position_callback, 1)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 1)
        self.bridge = CvBridge()
        self.image_width = 1280
        self.image_height = 720
        # self.WP_TO_M_Coeff = [0.0027011123406120653, 0.0022825322344103183]
        # self.WP_TO_M_Coeff = [0.003175, 0.00524]
        self.WP_TO_M_Coeff = [0.00199, 0.00297]
        self.max_lane_y = 520
        self.WARP_RADIUS = 112.2
        self.lane_detector = LaneDetector(model_path=Path(
            "/home/yvxaiver/lanenet-lane-detection/model_pytorch/loss=0.1277_miou=0.5893_epoch=5.pth".absolute()))
        self.left_lane_pts = []
        self.right_lane_pts = []
        self.following_path = []

        self.image_serial_n = 0

        src = np.float32(
            [[0, self.max_lane_y - 1], [self.image_width - 1, self.max_lane_y - 1], [0, 0], [self.image_width - 1, 0]])
        dst = np.float32([[self.image_width / 2 - self.WARP_RADIUS, 0], [self.image_width / 2 + self.WARP_RADIUS, 0],
                          [0, self.max_lane_y - 1], [self.image_width - 1, self.max_lane_y - 1]])
        self.M = np.array(cv2.getPerspectiveTransform(src, dst))

        self.x = self.image_width * self.WP_TO_M_Coeff[0] / 2
        self.y = -0.584
        self.yaw = np.radians(90.0)

    def process_image(self, data):
        try:
            self.get_logger().info("Starting Image Processing...")
            # Load trained policy and necessary data here
            policy = self.load_model('best_model.pth')
            scaler_X = joblib.load('scaler_X.pkl')
            scaler_y = joblib.load('scaler_y.pkl')

            trajectory, img, left_coeffs, right_coeffs, coff_check_left, coff_check_right = self.get_trajectory_from_lane_detector(
                self.lane_detector, data)
            target_speed = 0.5 / 3.6  # [m/s]
            curr_speed = 0  # need IMU data for this
            # Assuming you have defined left_coeffs, right_coeffs, move_speed, and speed
            input_features = np.concatenate((left_coeffs, right_coeffs, [target_speed, curr_speed]))

            # Reshape the input to 2D for the scaler (1 sample, many features)
            input_features_2d = input_features.reshape(1, -1)

            # Apply the standard scaler transformation
            scaled_input = scaler_X.transform(input_features_2d)

            # Convert scaled input to PyTorch tensor
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
            with torch.no_grad():
                # Load the saved scalers
                predictions = policy(input_tensor)
                predictions_numpy = predictions.numpy()
                predictions_original_scale = scaler_y.inverse_transform(predictions_numpy)
                throttle = predictions_original_scale[0][0]
                steer = predictions_original_scale[0][1]
            self.get_logger().info(f"Computed throttle and steer: throttle is {throttle}; steer is {steer}")
            msg = AckermannDriveStamped()

            msg.header.frame_id = "controller"
            msg.drive.steering_angle = steer
            msg.drive.steering_angle_velocity = 0.05
            msg.drive.speed = curr_speed
            msg.drive.acceleration = throttle
            msg.drive.jerk = 0

            self.get_logger().info("Publishing on drive...")
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().info(f"{e}")

    def load_model(self, model_path):
        model = NeuralNetwork()  # Make sure this is the same architecture as used during training
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model

    def img_to_array(self, image_raw):
        array = np.frombuffer(image_raw.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_raw.height, image_raw.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def get_trajectory_from_lane_detector(self, lane_detector, image):
        # get lane boundaries using the lane detector
        image_arr = self.img_to_array(image)

        poly_left, poly_right, img_left, img_right, left_coeffs, right_coeffs, coff_check_left, coff_check_right = lane_detector(
            image_arr)
        # https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
        img = img_left + img_right
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img.astype(np.uint8)
        img = cv2.resize(img, (600, 400))
        # cv2.imshow("image", img)

        # trajectory to follow is the mean of left and right lane boundary
        # note that we multiply with -0.5 instead of 0.5 in the formula for y below
        # according to our lane detector x is forward and y is left, but
        # according to Carla x is forward and y is right.
        x = np.arange(-2, 60, 1.0)
        y = -0.5 * (poly_left(x) + poly_right(x))
        # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
        # hence correct x coordinates
        x += 0.5
        trajectory = np.stack((x, y)).T
        cv2.circle(img, (int(trajectory[0]), int(trajectory[1])), 5, (0, 0, 255), -1)
        cv2.imshow("image", img)
        return trajectory, img, left_coeffs, right_coeffs, coff_check_left, coff_check_right

    def position_callback(self, data):
        self.x = data.x
        self.y = data.y
        self.yaw = data.z


def main(args=None):
    rclpy.init(args=args)

    image_to_command = ImageToCommand()
    rclpy.spin(image_to_command)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
