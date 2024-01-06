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
