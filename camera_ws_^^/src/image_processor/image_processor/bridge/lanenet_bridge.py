#!/usr/bin/env python3
"""
ros2 Image to LaneNet Bridge
"""

import os.path as ops
import sys
import time

from points_vector.msg import PointsVector
from geometry_msgs.msg import Point

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from numpy.polynomial import Polynomial

sys.path.append('/home/yvxaiver/lanenet-lane-detection')
# from lanenet_model import lanenet
from model.lanenet.LaneNet import LaneNet

from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

sys.path.append('/home/yvxaiver/LaneNet_to_Trajectory')
# from LaneNetToTrajectory import LaneProcessing, DualLanesToTrajectory


class LaneNetImageProcessor():
    def __init__(self, weights_path, image_width, image_height, max_lane_y=420, WARP_RADIUS=20, WP_TO_M_Coeff=[1, 1],
                 lane_sep=50, new_lane_x=20, new_lane_y=100, max_y_gap=25, min_lane_pts=10):
        self.weights_path = weights_path
        self.image_width = image_width
        self.image_height = image_height
        self.max_lane_y = max_lane_y
        self.WARP_RADIUS = WARP_RADIUS
        self.WP_TO_M_Coeff = WP_TO_M_Coeff
        self.lane_sep = lane_sep
        self.new_lane_x = new_lane_x
        self.new_lane_y = new_lane_y
        self.max_y_gap = max_y_gap
        self.min_lane_pts = min_lane_pts
        # self.lane_processor = LaneProcessing(self.image_width,self.image_height,self.max_lane_y,self.WARP_RADIUS,self.WP_TO_M_Coeff)

        self.calibration = False
        self.full_lane_pts = []
        self.following_path = []

        # to maintain history of detected lanes
        self.BUFFER_SIZE = 3
        self.lane_buffer = []

        self.transform = A.Compose([
            A.Resize(256, 512),
            A.Normalize(),
            ToTensorV2()
        ])

        src = np.float32([[0, max_lane_y - 1], [image_width - 1, max_lane_y - 1], [0, 0], [image_width - 1, 0]])
        dst = np.float32([[image_width / 2 - WARP_RADIUS, 0], [image_width / 2 + WARP_RADIUS, 0], [0, max_lane_y - 1],
                          [image_width - 1, max_lane_y - 1]])
        self.M = np.array(cv2.getPerspectiveTransform(src, dst))

        self.x_pos = self.image_width * self.WP_TO_M_Coeff[0] / 2
        self.y_pos = -0.584  # distance from back axle to camera view [m]
        self.yaw = np.radians(90.0)

        self.left_lane = None
        self.right_lane = None
        self.left_lane_pts = None
        self.right_lane_pts = None
        self.left_lane_fullpts = []
        self.right_lane_fullpts = []

    def init_lanenet(self):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LaneNet()
        self.model.load_state_dict(torch.load(self.weights_path))
        self.model.eval()
        self.model.to(self.DEVICE)
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(self.image_width, self.image_height, 512, 256,
                                                                      self.max_lane_y, self.WARP_RADIUS)

        self.img_counter = 0

        return True

    def image_to_trajectory(self, cv_image, x_pos, y_pos, yaw, lane_fit=True):

        T_start = time.time()

        cv2.imshow("original frame", cv_image)

        image_vis = cv_image
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        image = image.to(self.DEVICE)
        image = torch.unsqueeze(image, dim=0)

        T_resize = time.time()
        LOG.info('Image Preprocessing cost time: {:.5f}s'.format(T_resize - T_start))

        """
        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_ret, self.instance_seg_ret],
            feed_dict={self.input_tensor: [image]}
        )

        """

        with torch.no_grad():
            outputs = self.model(image)

        binary_seg_image = torch.squeeze(outputs['binary_seg_pred'].detach().to('cpu')).numpy()

        instance_seg_image = torch.squeeze(outputs['instance_seg_logits']).to('cpu').numpy()
        instance_seg_image = np.swapaxes(instance_seg_image, 0, 2)
        instance_seg_image = np.swapaxes(instance_seg_image, 0, 1)

        T_seg_inference = time.time()

        LOG.info('TOTAL Segmentation Inference cost time: {:.5f}s'.format(T_seg_inference - T_resize))

        if not lane_fit:
            out_dict = self.postprocessor.postprocess(
                binary_seg_result=binary_seg_image,
                instance_seg_result=instance_seg_image,
                source_image=image_vis,
                with_lane_fit=lane_fit,
                data_source='tusimple'
            )
            return out_dict

        """
        full_lane_pts = self.postprocessor.postprocess_lanepts(
            binary_seg_result=binary_seg_image,
            instance_seg_result=instance_seg_image,
            source_image=image_vis,
            data_source='tusimple'
        )
        self.full_lane_pts = full_lane_pts
        """

        lane_pts = self.postprocessor.postprocess_lanepts(
            binary_seg_result=binary_seg_image,
            instance_seg_result=instance_seg_image,
            source_image=image_vis,
            data_source='tusimple'
        )

        if lane_pts is None:
            return None, None
            # return None, None, None

        lane_pts = lane_pts[lane_pts[:, 1].argsort()]
        lanes = []

        warped = cv2.warpPerspective(cv_image[self.image_height - self.max_lane_y: self.image_height], self.M,
                                     (self.image_width, self.max_lane_y))
        warped2 = warped.copy()
        boundary_slope = (self.image_height - self.WARP_RADIUS) / self.max_lane_y

        for pt in lane_pts:
            x = pt[0]
            y = pt[1]

            cv2.circle(warped, (np.int_(x), np.int_(y)), 5, (0, 255, 0), -1)

            lane_counter = 0

            while True:
                if lane_counter == len(lanes):
                    if y < self.new_lane_y or x > self.image_width / 2 + self.WARP_RADIUS + boundary_slope * y - self.new_lane_x:
                        lanes.append([[x, y]])
                    break

                current_lane_x = lanes[lane_counter][-1][0]

                if x < current_lane_x - self.lane_sep:
                    if y < self.new_lane_y or x < self.image_width / 2 - self.WARP_RADIUS - boundary_slope * y + self.new_lane_x:
                        lanes.insert(lane_counter, [[x, y]])
                    break

                elif x > current_lane_x + self.lane_sep or y > lanes[lane_counter][-1][-1] + self.max_y_gap:
                    lane_counter += 1

                else:
                    lanes[lane_counter].append([x, y])
                    break

        # cv2.imshow("Binary mask", binary_seg_image * 1.0)
        # cv2.waitKey(1)

        # cv2.imshow("Instance image", instance_seg_image)
        # cv2.waitKey(1)

        # cv2.imwrite('/home/yvxaiver/output/binary_masks/%d.png' % self.img_counter, binary_seg_image * 255)
        # cv2.imwrite('/home/yvxaiver/output/instance_masks/%d.png' % self.img_counter, instance_seg_image * 255)

        # cv2.imwrite('/home/yvxaiver/output/preprocessed/%d.png' % self.img_counter, cv2.flip(warped, 0))
        # self.img_counter += 1

        cv2.imshow("Birds eye view before processing", cv2.flip(warped, 0))
        cv2.waitKey(1)

        for i in range(len(lanes)):
            lanes[i] = np.transpose(np.array(lanes[i]))

        for i in range(len(lanes) - 1, -1, -1):
            if lanes[i].shape[1] < 10:
                del lanes[i]

        for i in range(len(lanes) - 2, 0, -1):
            if lanes[i][1][0] >= lanes[i - 1][1][-1] and lanes[i][1][0] >= lanes[i + 1][1][-1]:
                del lanes[i]

        # below is a moving average approach to smooth the lanes
        # for i in range(len(lanes)):
        #     window_size = 5  # this is hard-coded, experiment with it and choose the best performing one
        #     weights = np.ones(window_size) / window_size
        #     smoothed_lane_x = np.convolve(lanes[i][0], weights, mode='valid')
        #     lanes[i][0] = smoothed_lane_x

        full_lane_pts = []
        polynomials = []

        # give a polynomial fit to each lane
        for i in range(len(lanes)):
            p = Polynomial.fit(lanes[i][1], lanes[i][0], 3)

            # smooth the lanes again, leveraging the polynomial fit
            # smoothed_x = np.polyval(p, lanes[i][1])
            # residual_x = lanes[i][0] - smoothed_x
            # x_deviate_threshold = 2 * np.std(residual_x)
            # alpha = 0.5  # weighting factor for adjustment
            # adjusted_x = np.copy(lanes[i][0])
            # for idx, res in enumerate(residual_x):
            #     if abs(res) > x_deviate_threshold:
            #         adjusted_x[idx] = alpha * lanes[i][0][idx] + (1 - alpha) * smoothed_x[idx]
            # lanes[i][0] = adjusted_x

            dy, dx = p.linspace(20)
            # full_lane_pts.append(np.dstack((np.int_(dx), np.int_(dy))).reshape(2, 20))
            full_lane_pts.append(np.array([dx, dy]))
            polynomials.append(p)

            for j in range(20):
                cv2.circle(warped2, (np.int_(dx[j]), np.int_(dy[j])), 5, (0, 255, 0), -1)

        cv2.imshow("polynomial fit", cv2.flip(warped2, 0))
        # cv2.imwrite('/home/yvxaiver/output/lanes/%d.png' % self.img_counter, cv2.flip(warped2, 0))

        self.full_lane_pts = full_lane_pts

        # if self.calibration:
        #     print(self.lane_processor.WARP_RADIUS)
        #     print(self.lane_processor.get_wp_to_m_coeff())
        #     self.calibration = False
        #     raise SystemExit

        car_x_pos = self.image_width / 2.0
        right_lanes = []
        left_lanes = []
        for i, lane in enumerate(self.full_lane_pts):
            median_x = np.median(lane[0])
            self.get_logger().info(f"median of x coordinate for lane {i} is {median_x}")
            diff = abs(median_x - car_x_pos)
            if median_x < car_x_pos:
                left_lanes.append(np.array([i, diff]))
            else:
                right_lanes.append(np.array([i, diff]))

        if left_lanes:
            # Find the index of the lane with the smallest diff
            left_lane_index = int(min(left_lanes, key=lambda lane_diff: lane_diff[1])[0])
            shifted_left_lane_pts = self.shift(lanes[left_lane_index], x_pos, y_pos, yaw)
            self.left_lane = Polynomial.fit(shifted_left_lane_pts[1], shifted_left_lane_pts[0], 3)
        else:
            self.left_lane = None  # or some default value
        if right_lanes:
            # Find the index of the lane with the smallest diff
            right_lane_index = int(min(right_lanes, key=lambda lane_diff: lane_diff[1])[0])
            shifted_right_lane_pts = self.shift(lanes[right_lane_index], x_pos, y_pos, yaw)
            self.right_lane = Polynomial.fit(shifted_right_lane_pts[1], shifted_right_lane_pts[0], 3)
        else:
            self.right_lane = None  # or some default value

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.yaw = yaw

        dy, dx = self.left_lane.linspace(20)
        for x, y in zip(np.int_(dx), np.int_(dy)):
            cv2.circle(warped2, (x, y), 5, (0, 255, 0), -1)  # Green circle

        dy, dx = self.right_lane.linspace(20)
        for x, y in zip(np.int_(dx), np.int_(dy)):
            cv2.circle(warped2, (x, y), 5, (0, 255, 0), -1)  # Green circle

        cv2.imshow("left and right lane on original frame", cv_image)

        T_post_process = time.time()
        LOG.info('Image Post-Process cost time: {:.5f}s'.format(T_post_process - T_seg_inference))

        return self.left_lane, self.right_lane

        # changed = False
        #
        # if self.full_lane_pts and self.full_lane_pts[0][0][0] < self.image_width / 2 < self.full_lane_pts[-1][0][0]:
        #     for i in range(1, len(lanes)):
        #         if self.full_lane_pts[i][0][0] > self.image_width / 2:
        #             shifted_lane_pts = self.shift(lanes[i - 1], x_pos, y_pos, yaw)
        #             shifted_p = Polynomial.fit(shifted_lane_pts[1], shifted_lane_pts[0], polynomials[i - 1].degree())
        #
        #             if self.left_lane is None or self.compare_lanes(self.left_lane, shifted_p):
        #                 self.left_lane = shifted_p
        #                 self.left_lane_pts = shifted_lane_pts
        #                 changed = True
        #                 LOG.info('Found new left lane')
        #
        #             shifted_lane_pts = self.shift(lanes[i], x_pos, y_pos, yaw)
        #             shifted_p = Polynomial.fit(shifted_lane_pts[1], shifted_lane_pts[0], polynomials[i].degree())
        #
        #             if self.right_lane is None or self.compare_lanes(self.right_lane, shifted_p):
        #                 self.right_lane = shifted_p
        #                 self.right_lane_pts = shifted_lane_pts
        #                 changed = True
        #                 LOG.info('Found new right lane')
        #
        #             break
        #
        # if changed:
        #     if self.left_lane is not None:
        #         self.left_lane_pts = self.shift(self.left_lane_pts, x_pos, y_pos, yaw, old_to_new=1)
        #         self.left_lane = Polynomial.fit(self.left_lane_pts[1], self.left_lane_pts[0], self.left_lane.degree())
        #         dy, dx = self.left_lane.linspace(20)
        #         self.left_lane_fullpts = [dx, dy]
        #
        #     if self.right_lane is not None:
        #         self.right_lane_pts = self.shift(self.right_lane_pts, x_pos, y_pos, yaw, old_to_new=1)
        #         self.right_lane = Polynomial.fit(self.right_lane_pts[1], self.right_lane_pts[0],
        #                                          self.right_lane.degree())
        #         dy, dx = self.right_lane.linspace(20)
        #         self.right_lane_fullpts = [dx, dy]

        # if changed and self.left_lane is not None and self.right_lane is not None:
        #     dy = np.linspace(0, min(self.left_lane.domain[1], self.right_lane.domain[1]), 20)
        #     dx = (self.left_lane(dy) + self.right_lane(dy)) / 2
        #     self.following_path = [dx * self.WP_TO_M_Coeff[0], dy * self.WP_TO_M_Coeff[1]]

    def shift(self, pts, x, y, yaw, old_to_new=0, pixels=True):
        sign = np.power(-1, old_to_new)
        x_pixel = sign * (x - self.x_pos) / self.WP_TO_M_Coeff[0]
        y_pixel = sign * (y - self.y_pos) / self.WP_TO_M_Coeff[1]
        yaw_change = sign * (self.yaw - yaw)

        if not pixels:
            shifted_x = pts[0] * np.cos(yaw_change) - pts[1] * np.sin(yaw_change) + x_pixel * self.WP_TO_M_Coeff[0]
            shifted_y = pts[0] * np.sin(yaw_change) + pts[1] * np.cos(yaw_change) + y_pixel * self.WP_TO_M_Coeff[1]

        else:
            shifted_x = pts[0] * np.cos(yaw_change) - pts[1] * np.sin(yaw_change) + x_pixel
            shifted_y = pts[0] * np.sin(yaw_change) + pts[1] * np.cos(yaw_change) + y_pixel

        return [shifted_x, shifted_y]

    def compare_lanes(self, old, shifted_p, num_pts=20, max_avg_distance=20):
        dy = np.linspace(shifted_p.domain[0], shifted_p.domain[1], num_pts)
        return np.sum(np.abs(old(dy) - shifted_p(dy))) / num_pts <= max_avg_distance and shifted_p.domain[1] >= \
            old.domain[1]

    def get_point_vector_path(self):
        # print(self.following_path)
        if self.following_path:
            vector = []
            for i in range(len(self.following_path[0])):
                pt = Point()
                pt.x = self.following_path[0][i]
                pt.y = self.following_path[1][i]
                pt.z = 0.0
                vector.append(pt)
            ptv = PointsVector()
            ptv.points = vector
            # ptv.x_coeff = float(self.lane_processor.get_wp_to_m_coeff()[0])
            ptv.x_coeff = self.WP_TO_M_Coeff[0]
            return ptv
        return None
