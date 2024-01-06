#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from points_vector.msg import PointsVector
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from drive_msg.msg import Drivemsg

import sklearn
import torch
import torch.nn as nn
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import joblib

from .bridge.lanenet_bridge import LaneNetImageProcessor


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


class ImageProcessorNode(Node):
    '''Handles feeding camera frame into lanenet, converting outputs into path to be followed, and publishes that path.'''

    def __init__(self):
        super().__init__('image_processor')
        self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.image_callback, 1)
        self.position_subscriber_ = self.create_subscription(Point, '/position', self.position_callback, 1)
        # self.publisher_ = self.create_publisher(PointsVector, '/lanenet_path', 1)
        self.drive_publisher_ = self.create_publisher(Drivemsg, '/drive_ts', 1)

        self.bridge = CvBridge()
        # self.weights_path = "/home/yvxaiver/lanenet-lane-detection/modelv2/tusimple/bisenetv2_lanenet/tusimple_val_miou=0.6843.ckpt-1328"
        self.weights_path = "/home/yvxaiver/lanenet-lane-detection/model_pytorch/loss=0.1277_miou=0.5893_epoch=5.pth"
        self.image_width = 1280
        self.image_height = 720
        # self.WP_TO_M_Coeff = [0.0027011123406120653, 0.0022825322344103183]
        # self.WP_TO_M_Coeff = [0.003175, 0.00524]
        self.WP_TO_M_Coeff = [0.00199, 0.00297]
        self.max_lane_y = 520
        self.WARP_RADIUS = 112.2
        self.processor = LaneNetImageProcessor(self.weights_path, self.image_width, self.image_height, self.max_lane_y,
                                               self.WARP_RADIUS, self.WP_TO_M_Coeff)
        self.lanenet_status = self.processor.init_lanenet()

        self.left_lane_pts = []
        self.right_lane_pts = []
        self.following_path = []
        self.image_serial_n = 0

        self.throttle = 0.0
        self.steer = 0.0

        src = np.float32(
            [[0, self.max_lane_y - 1], [self.image_width - 1, self.max_lane_y - 1], [0, 0], [self.image_width - 1, 0]])
        dst = np.float32([[self.image_width / 2 - self.WARP_RADIUS, 0], [self.image_width / 2 + self.WARP_RADIUS, 0],
                          [0, self.max_lane_y - 1], [self.image_width - 1, self.max_lane_y - 1]])
        self.M = np.array(cv2.getPerspectiveTransform(src, dst))

        self.x = self.image_width * self.WP_TO_M_Coeff[0] / 2
        self.y = -0.584
        self.yaw = np.radians(90.0)
        self.curr_speed = 0

    def image_callback(self, data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_frame = cv2.resize(cv_frame, (512, 256))
            if self.lanenet_status:
                tmp_x = self.x
                tmp_y = self.y
                tmp_yaw = self.yaw

                # policy for drive commands
                policy = self.load_model('/home/yvxaiver/LaneNet_to_Trajectory/best_model.pth')
                scaler_X = joblib.load('/home/yvxaiver/LaneNet_to_Trajectory/scaler_X.pkl')
                scaler_y = joblib.load('/home/yvxaiver/LaneNet_to_Trajectory/scaler_y.pkl')

                # changed, left_lane_pts, right_lane_pts, following_path, left_coeffs, right_coeffs = self.processor.image_to_trajectory(cv_frame,
                #                                                                                             tmp_x,
                #                                                                                             tmp_y,
                #                                                                                             tmp_yaw)

                left_coeffs, right_coeffs = self.processor.image_to_trajectory(cv_frame, tmp_x, tmp_y, tmp_yaw)
                print(f"left_coeffs are {left_coeffs}")
                # self.left_lane_pts = left_lane_pts
                # self.right_lane_pts = right_lane_pts
                # self.following_path = following_path

                target_speed = 25  # m/s
                self.curr_speed = 0.01  # need IMU data to update this
                # Assuming you have defined left_coeffs, right_coeffs, move_speed, and speed
                input_features = np.concatenate((left_coeffs, right_coeffs, [target_speed, self.curr_speed]))

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
                    self.throttle = float(predictions_original_scale[0][0])
                    self.steer = float(predictions_original_scale[0][1])

                print(f"published throttle: {self.throttle}; steer: {self.steer}")
                # self.throttle = np.clip(self.throttle, 0.0, 1.0)
                self.steer = np.clip(self.steer, -0.15, 0.15)
                self.throttle = np.clip(self.throttle, 0.0, 0.2)

                msg = Drivemsg()
                msg.throttle = self.throttle
                msg.steer = self.steer

                # print(f"published throttle: {self.throttle}; steer: {self.steer}")
                self.drive_publisher_.publish(msg)

                # if changed:
                #     self.left_lane_pts = left_lane_pts
                #     self.right_lane_pts = right_lane_pts
                #     self.following_path = following_path
                #     msg = self.processor.get_point_vector_path()
                #     if msg: self.publisher_.publish(msg)

                # else:
                #     if left_lane_pts:
                #         self.left_lane_pts = self.processor.shift(left_lane_pts, tmp_x, tmp_y, tmp_yaw, old_to_new=1)

                #     if right_lane_pts:
                #         self.right_lane_pts = self.processor.shift(right_lane_pts, tmp_x, tmp_y, tmp_yaw, old_to_new=1)

                #     if following_path:
                #         self.following_path = self.processor.shift(following_path, tmp_x, tmp_y, tmp_yaw, old_to_new=1,
                #                                                    pixels=False)

            # self.image_save(cv_frame)
            # self.image_display(cv_frame)

        except Exception as e:
            print(e)

    def load_model(self, model_path):
        model = NeuralNetwork()  # Make sure this is the same architecture as used during training
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model

    def position_callback(self, data):
        self.x = data.x
        self.y = data.y
        self.yaw = data.z

    def image_display(self, cv_frame):
        # displaying left lane, right lane, and estimated trajectory for visualization only

        # cv2.imwrite('/home/yvxaiver/output/orig/%d.png' % self.image_serial_n, cv_frame)
        # cv2.imwrite('/home/yvxaiver/output/orig/%d.png' % self.image_serial_n, cv_frame)

        warped = cv2.warpPerspective(cv_frame[self.image_height - self.max_lane_y:self.image_height], self.M,
                                     (self.image_width, self.max_lane_y))

        if self.left_lane_pts:
            for i in range(len(self.left_lane_pts[0])):
                pt_x = np.int_(self.left_lane_pts[0][i])
                pt_y = np.int_(self.left_lane_pts[1][i])
                if pt_x >= 0 and pt_x < self.image_width and pt_y >= 0 and pt_y < self.image_height:
                    cv2.circle(warped, (pt_x, pt_y), 5, (0, 255, 0), -1)

        if self.right_lane_pts:
            for i in range(len(self.right_lane_pts[0])):
                pt_x = np.int_(self.right_lane_pts[0][i])
                pt_y = np.int_(self.right_lane_pts[1][i])
                if pt_x >= 0 and pt_x < self.image_width and pt_y >= 0 and pt_y < self.image_height:
                    cv2.circle(warped, (pt_x, pt_y), 5, (0, 255, 0), -1)

        if self.following_path:
            for i in range(len(self.following_path[0])):
                pt_x = np.int_(self.following_path[0][i] / self.WP_TO_M_Coeff[0])
                pt_y = np.int_(self.following_path[1][i] / self.WP_TO_M_Coeff[1])
                if pt_x >= 0 and pt_x < self.image_width and pt_y >= 0 and pt_y < self.image_height:
                    cv2.circle(warped, (pt_x, pt_y), 5, (0, 0, 255), -1)

        # self.image_save(cv2.flip(warped, 0))

        cv2.imshow("left, right lane and trajectory", cv2.flip(warped, 0))
        cv2.waitKey(1)

    def image_save(self, cv_frame):
        status = cv2.imwrite('/home/yvxaiver/output/0/' + str(self.image_serial_n) + ".png", cv_frame)
        self.image_serial_n += 1
        # print(status)


def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessorNode()
    rclpy.spin(image_processor)

    rclpy.shutdown()


if __name__ == '__main__':
    main()