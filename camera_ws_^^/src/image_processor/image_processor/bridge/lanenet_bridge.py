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

from .camera_geometry import CameraGeometry

sys.path.append('/home/yvxaiver/lanenet-lane-detection')
# from lanenet_model import lanenet
from model.lanenet.LaneNet import LaneNet

from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

sys.path.append('/home/yvxaiver/LaneNet_to_Trajectory')

from numpy.polynomial import Polynomial


class LaneNetImageProcessor():
    def __init__(self, weights_path, image_width, image_height, max_lane_y=420, WARP_RADIUS=20, WP_TO_M_Coeff=[1, 1],
                 lane_sep=50, new_lane_x=20, new_lane_y=100, max_y_gap=25, min_lane_pts=10,
                 cam_geom=CameraGeometry(image_width=512, image_height=256)):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()

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

        self.calibration = False
        self.full_lane_pts = []
        self.following_path = []
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
        self.left_lane_poly = None
        self.right_lane_poly = None
        self.left_lane_pts = None
        self.right_lane_pts = None
        self.left_p_coef = None
        self.right_p_coef = None
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

    def fit_poly(self, probs, probs_measure):
        points = np.array(np.argwhere(probs_measure > 0)).transpose()
        coff_check = True
        # coeffs = np.polyfit(points[1], points[0], deg=3)
        poly = Polynomial.fit(points[1], points[0], deg=3)

        # print(self.cut_v)
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        # coff_check = True
        if not np.any(points):
            # Handle the empty case (e.g., return a default polynomial or None)
            default_coeffs = [0]
            coff_check = False
            return np.poly1d(default_coeffs), default_coeffs, coff_check

        coeffs = np.polyfit(self.grid[:, 0][mask], self.grid[:, 1][mask], deg=3)
        # poly = Polynomial.fit(self.grid[:, 0][mask], (256-self.grid[:, 1][mask]), deg=3)
        return poly, coeffs, coff_check

    def measure_driving(self, cv_image, left_polynomial, right_polynomial):
        measurement = 0

        left_x, left_y = left_polynomial.linspace(20)
        right_x, right_y = right_polynomial.linspace(20)

        # cv2.circle(cv_image, (np.int_(10), np.int_(10)), 8, (255, 0, 0), -1)

        for i in range(20):
            cv2.circle(cv_image, (np.int_(left_x[i]), np.int_(left_y[i])), 5, (0, 255, 0), -1)

        for i in range(20):
            cv2.circle(cv_image, (np.int_(right_x[i]), np.int_(right_y[i])), 5, (0, 255, 0), -1)

        cv2.imshow("lanes based on polynomials from segmentation", cv_image)
        cv2.waitKey(1)

        return measurement

    def image_to_trajectory(self, cv_image, x_pos, y_pos, yaw, lane_fit=True):

        T_start = time.time()

        image_vis = cv_image
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        image = image.to(self.DEVICE)
        image = torch.unsqueeze(image, dim=0)

        T_resize = time.time()
        LOG.info('Image Preprocessing cost time: {:.5f}s'.format(T_resize - T_start))

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

        # lane_pts, pts_orig = self.postprocessor.postprocess_lanepts(
        #     binary_seg_result=binary_seg_image,
        #     instance_seg_result=instance_seg_image,
        #     source_image=image_vis,
        #     data_source='tusimple'
        # )

        left, right, left_measure, right_measure = self.postprocessor.postprocess_lanepts(
            binary_seg_result=binary_seg_image,
            instance_seg_result=instance_seg_image,
            source_image=image_vis,
            data_source='tusimple',
            y_cut=100
        )

        left_poly, left_coeffs, coff_check_left = self.fit_poly(left, left_measure)
        right_poly, right_coeffs, coff_check_right = self.fit_poly(right, right_measure)

        measurement = self.measure_driving(cv_image, left_poly, right_poly)

        return left_coeffs, right_coeffs

        if lane_pts is None:
            return None, None, None

        lane_pts = lane_pts[lane_pts[:, 1].argsort()]
        lanes = []
        lane_counter = -1

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

        orig_lane_pts = pts_orig[pts_orig[:, 1].argsort()]
        orig_lanes = []
        o_lane_counter = -1
        for pt in orig_lane_pts:
            x = pt[0]
            y = pt[1]

            o_lane_counter = 0

            while True:
                if o_lane_counter == len(orig_lanes):
                    if y < self.new_lane_y or x > self.image_width / 2 - self.new_lane_x:
                        orig_lanes.append([[x, y]])
                    break

                current_lane_x = orig_lanes[o_lane_counter][-1][0]

                if x < current_lane_x - self.lane_sep:
                    if y < self.new_lane_y or x < self.image_width / 2 + self.new_lane_x:
                        orig_lanes.insert(o_lane_counter, [[x, y]])
                    break

                elif x > current_lane_x + self.lane_sep or y > orig_lanes[o_lane_counter][-1][-1] + self.max_y_gap:
                    o_lane_counter += 1

                else:
                    orig_lanes[o_lane_counter].append([x, y])
                    break

        resized_image = cv2.resize(cv_image, (512, 256))
        y_shift = (self.image_height - self.max_lane_y) * 256 / self.image_height
        for l in orig_lanes:
            for pt in l:
                x = pt[0]
                pt[1] = pt[1] + y_shift
                y = pt[1]
                # cv2.circle(resized_image, (np.int_(x), np.int_(y)), 5, (0, 255, 0), -1)

        # cv2.imshow("original points on correct resized images", resized_image)
        self.img_counter += 1

        # cv2.imshow("Birds eye view before processing", cv2.flip(warped, 0))
        # cv2.waitKey(1)

        full_lane_pts = []
        polynomials = []
        # print(f"compare numbers of lanes: {lane_counter == o_lane_counter}")
        for i in range(len(lanes)):
            lanes[i] = np.transpose(np.array(lanes[i]))
            orig_lanes[i] = np.transpose(np.array(orig_lanes[i]))

        for i in range(len(lanes) - 1, -1, -1):
            if lanes[i].shape[1] < 10:
                del lanes[i]
                # del orig_lanes[i]

        for i in range(len(lanes) - 2, 0, -1):
            if lanes[i][1][0] >= lanes[i - 1][1][-1] and lanes[i][1][0] >= lanes[i + 1][1][-1]:
                del lanes[i]
                # del orig_lanes[i]

        for i in range(len(lanes)):
            warped_p = Polynomial.fit(lanes[i][1], lanes[i][0], 3)
            p = Polynomial.fit(orig_lanes[i][1], orig_lanes[i][0], 3)

            dy, dx = p.linspace(20)
            wdy, wdx = warped_p.linspace(20)
            full_lane_pts.append(np.array([wdx, wdy]))
            polynomials.append(warped_p)

            for i in range(20):
                cv2.circle(resized_image, (np.int_(dx[i]), np.int_(dy[i])), 5, (0, 255, 0), -1)

        # cv2.imshow("polynomial fitted lanes", cv2.flip(warped2, 0))
        cv2.imshow("polynomial fitted lanes", resized_image)
        cv2.waitKey(1)

        # cv2.imwrite('/home/yvxaiver/output/lanes/%d.png' % self.img_counter, cv2.flip(warped2, 0))

        self.full_lane_pts = full_lane_pts

        # if self.calibration:
        #     print(self.lane_processor.WARP_RADIUS)
        #     print(self.lane_processor.get_wp_to_m_coeff())
        #     self.calibration = False
        #     raise SystemExit

        changed = True
        l_index = -1
        r_index = -1

        if self.full_lane_pts and self.full_lane_pts[0][0][0] < self.image_width / 2 and self.full_lane_pts[-1][0][
            0] > self.image_width / 2:
            for i in range(1, len(lanes)):
                if self.full_lane_pts[i][0][0] > self.image_width / 2:
                    shifted_lane_pts = self.shift(lanes[i - 1], x_pos, y_pos, yaw)
                    shifted_p = Polynomial.fit(shifted_lane_pts[1], shifted_lane_pts[0], 3)
                    self.left_lane = shifted_p
                    self.left_lane_pts = shifted_lane_pts
                    # if self.left_lane is None or self.compare_lanes(self.left_lane, shifted_p):
                    #     self.left_lane = shifted_p
                    #     self.left_lane_pts = shifted_lane_pts
                    #     changed = True
                    #     LOG.info('Found new left lane')

                    shifted_lane_pts = self.shift(lanes[i], x_pos, y_pos, yaw)
                    shifted_p = Polynomial.fit(shifted_lane_pts[1], shifted_lane_pts[0], 3)
                    self.right_lane = shifted_p
                    self.right_lane_pts = shifted_lane_pts
                    # if self.right_lane is None or self.compare_lanes(self.right_lane, shifted_p):
                    #     self.right_lane = shifted_p
                    #     self.right_lane_pts = shifted_lane_pts
                    #     changed = True
                    #     LOG.info('Found new right lane')

                    break

        # find the left and right lanes for the original image
        for i in range(1, len(orig_lanes)):
            if orig_lanes[i][-1][0] > 512 / 2:
                # this is right lane
                orig_lanes[i] = np.array(orig_lanes[i]).transpose()
                right_p = Polynomial.fit(orig_lanes[i][0], orig_lanes[i][1], 3)
                self.right_p_coef = np.polyfit(orig_lanes[i][0], orig_lanes[i][1], 3)
                right_dy, right_dx = right_p.linspace(20)
                # for j in range(20):
                #     cv2.circle(resized_image, (np.int_(right_dx[j]), np.int_(right_dy[j])), 5, (0, 255, 0), -1)

                # left lane
                orig_lanes[i - 1] = np.array(orig_lanes[i - 1]).transpose()
                # left_p = Polynomial.fit(orig_lanes[i-1][1], orig_lanes[i-1][0], 3)
                self.left_p_coef = np.polyfit(orig_lanes[i - 1][0], orig_lanes[i - 1][1], 3)
                # left_dy, left_dx = left_p.linspace(20)

                # cv2.imshow("right lane", resized_image)

                break

        if changed:
            if self.left_lane is not None:
                self.left_lane_pts = self.shift(self.left_lane_pts, x_pos, y_pos, yaw, old_to_new=1)
                self.left_lane_poly = Polynomial.fit(self.left_lane_pts[1], self.left_lane_pts[0], 3)
                dy, dx = self.left_lane_poly.linspace(20)
                self.left_lane_fullpts = [dx, dy]
                self.left_lane = np.polyfit(self.left_lane_pts[1], self.left_lane_pts[0], 3)

            if self.right_lane is not None:
                self.right_lane_pts = self.shift(self.right_lane_pts, x_pos, y_pos, yaw, old_to_new=1)
                self.right_lane_poly = Polynomial.fit(self.right_lane_pts[1], self.right_lane_pts[0], 3)
                dy, dx = self.right_lane_poly.linspace(20)
                self.right_lane_fullpts = [dx, dy]
                self.right_lane = np.polyfit(self.right_lane_pts[1], self.right_lane_pts[0], 3)

            self.x_pos = x_pos
            self.y_pos = y_pos
            self.yaw = yaw

        if changed and self.left_lane is not None and self.right_lane is not None:
            dy = np.linspace(0, min(self.left_lane_poly.domain[1], self.right_lane_poly.domain[1]), 20)
            dx = (self.left_lane_poly(dy) + self.right_lane_poly(dy)) / 2
            self.following_path = [dx * self.WP_TO_M_Coeff[0], dy * self.WP_TO_M_Coeff[1]]

        T_post_process = time.time()
        LOG.info('Image Post-Process cost time: {:.5f}s'.format(T_post_process - T_seg_inference))

        return changed, self.left_lane_fullpts, self.right_lane_fullpts, self.following_path, self.left_p_coef, self.right_p_coef

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
