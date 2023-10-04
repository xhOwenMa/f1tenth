#!/usr/bin/env python3
"""
ros2 Image to LaneNet Bridge
"""

import os.path as ops
import sys
import time

from points_vector.msg import PointsVector
from geometry_msgs.msg import Point

# import sklearn 
import cv2
import numpy as np
# import tensorflow as tf
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

from numpy.polynomial import Polynomial


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

        """
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

        self.net = lanenet.LaneNet(phase='test', cfg=CFG)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)

        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        # define saver
        self.saver = tf.train.Saver(variables_to_restore)

        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=self.weights_path)
        """

        return True

    def image_to_trajectory(self, cv_image, x_pos, y_pos, yaw, lane_fit=True):

        T_start = time.time()

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
            return None, None, None

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
        for i in range(len(lanes)):
            window_size = 5  # this is hard-coded, experiment with it and choose the best performing one
            weights = np.ones(window_size) / window_size
            smoothed_lane_x = np.convolve(lanes[i][0], weights, mode='valid')
            lanes[i][0] = smoothed_lane_x

        full_lane_pts = []
        polynomials = []

        for i in range(len(lanes)):
            p1, e1 = Polynomial.fit(lanes[i][1], lanes[i][0], 1, full=True)
            p3, e3 = Polynomial.fit(lanes[i][1], lanes[i][0], 3, full=True)

            if 8 * e3[0][0] > e1[0][0]:
                p = p1
            else:
                p = p3

            # smooth the lanes again, leveraging the polynomial fit
            smoothed_x = np.polyval(p, lanes[i][1])
            residual_x = lanes[i][0] - smoothed_x
            x_deviate_threshold = 2 * np.std(residual_x)
            alpha = 0.5  # weighting factor for adjustment
            adjusted_x = np.copy(lanes[i][0])
            for idx, res in enumerate(residual_x):
                if abs(res) > x_deviate_threshold:
                    adjusted_x[idx] = alpha * lanes[i][0][idx] + (1 - alpha) * smoothed_x[idx]
            lanes[i][0] = adjusted_x

            dy, dx = p.linspace(20)
            # full_lane_pts.append(np.dstack((np.int_(dx), np.int_(dy))).reshape(2, 20))
            full_lane_pts.append(np.array([dx, dy]))
            polynomials.append(p)

            for j in range(20):
                cv2.circle(warped2, (np.int_(dx[j]), np.int_(dy[j])), 5, (0, 255, 0), -1)

        # cv2.imwrite('/home/yvxaiver/output/lanes/%d.png' % self.img_counter, cv2.flip(warped2, 0))

        self.full_lane_pts = full_lane_pts

        # print('Number of full lane pts: {}'.format(len(self.full_lane_pts)))

        # self.lane_processor.process_next_lane(full_lane_pts)
        # full_lane_pts = self.lane_processor.get_full_lane_pts()
        # physical_fullpts = self.lane_processor.get_physical_fullpts()
        """
        physical_fullpts = []
        for lane in self.full_lane_pts:
            physical_fullpts.append(lane * self.WP_TO_M_Coeff)
        """
        # print('Number of physical full pts: {}'.format(len(physical_fullpts)))

        """
        for i in range(len(full_lane_pts)):
            for j in range(len(full_lane_pts[i])):
                print('Lane PIXELS: ', full_lane_pts[i][j][0], ' ', full_lane_pts[i][j][1])
                print('Lane METERS: ', physical_fullpts[i][j][0], ' ', physical_fullpts[i][j][1])
                print('----------------------------------------')
	"""

        if self.calibration:
            print(self.lane_processor.WARP_RADIUS)
            print(self.lane_processor.get_wp_to_m_coeff())
            self.calibration = False
            raise SystemExit

        """
        phy_centerpts = []
        phy_splines = []
        closest_lane_dist = float('inf')
        closest_lane_idx = 0
        if physical_fullpts:
            #print(physical_fullpts)
            for i in range(len(physical_fullpts)):
                if not i: continue
                traj = DualLanesToTrajectory(physical_fullpts[i-1],physical_fullpts[i],N_centerpts=20)
                phy_centerpts.append(traj.get_centerpoints())
                phy_splines.append(traj.get_spline())
            min_center_y_val = float('inf')
            #for lane in phy_centerpts:
                #if min(lane[1]) < min_center_y_val: min_center_y_val = min(lane[1])
            for i in range(len(phy_splines)):
                new_dist = abs(phy_splines[i](0.2)-(self.image_width*self.WP_TO_M_Coeff[0])/2)
                if new_dist < closest_lane_dist:
                    closest_lane_dist = new_dist
                    closest_lane_idx = i
            if phy_centerpts: self.following_path = phy_centerpts[closest_lane_idx]



        # For display output
        centerpts = []
        if full_lane_pts:
            for i in range(len(full_lane_pts)):
                if not i: continue
                traj = DualLanesToTrajectory(full_lane_pts[i-1],full_lane_pts[i],N_centerpts=20)
                centerpts.append(traj.get_centerpoints())
        """

        changed = False

        if self.full_lane_pts and self.full_lane_pts[0][0][0] < self.image_width / 2 < self.full_lane_pts[-1][0][0]:
            for i in range(1, len(lanes)):
                if self.full_lane_pts[i][0][0] > self.image_width / 2:
                    shifted_lane_pts = self.shift(lanes[i - 1], x_pos, y_pos, yaw)
                    shifted_p = Polynomial.fit(shifted_lane_pts[1], shifted_lane_pts[0], polynomials[i - 1].degree())

                    if self.left_lane is None or self.compare_lanes(self.left_lane, shifted_p):
                        self.left_lane = shifted_p
                        self.left_lane_pts = shifted_lane_pts
                        changed = True
                        LOG.info('Found new left lane')

                    shifted_lane_pts = self.shift(lanes[i], x_pos, y_pos, yaw)
                    shifted_p = Polynomial.fit(shifted_lane_pts[1], shifted_lane_pts[0], polynomials[i].degree())

                    if self.right_lane is None or self.compare_lanes(self.right_lane, shifted_p):
                        self.right_lane = shifted_p
                        self.right_lane_pts = shifted_lane_pts
                        changed = True
                        LOG.info('Found new right lane')

                    break

        if changed:
            if self.left_lane is not None:
                self.left_lane_pts = self.shift(self.left_lane_pts, x_pos, y_pos, yaw, old_to_new=1)
                self.left_lane = Polynomial.fit(self.left_lane_pts[1], self.left_lane_pts[0], self.left_lane.degree())
                dy, dx = self.left_lane.linspace(20)
                self.left_lane_fullpts = [dx, dy]

            if self.right_lane is not None:
                self.right_lane_pts = self.shift(self.right_lane_pts, x_pos, y_pos, yaw, old_to_new=1)
                self.right_lane = Polynomial.fit(self.right_lane_pts[1], self.right_lane_pts[0],
                                                 self.right_lane.degree())
                dy, dx = self.right_lane.linspace(20)
                self.right_lane_fullpts = [dx, dy]

            self.x_pos = x_pos
            self.y_pos = y_pos
            self.yaw = yaw

        """
        phy_centerpts = []
        for i in range(len(lanes)-1): 
            y_min = max(polynomials[i].domain[0], polynomials[i+1].domain[0])
            y_max = min(polynomials[i].domain[1], polynomials[i+1].domain[1])

            dy = np.linspace(y_min, y_max, 20)
            dx = (polynomials[i](dy) + polynomials[i+1](dy)) / 2
            phy_centerpts.append([dx * self.WP_TO_M_Coeff[0], dy * self.WP_TO_M_Coeff[1]])

        self.phy_centerpts = phy_centerpts
        """

        if changed and self.left_lane is not None and self.right_lane is not None:
            dy = np.linspace(0, min(self.left_lane.domain[1], self.right_lane.domain[1]), 20)
            dx = (self.left_lane(dy) + self.right_lane(dy)) / 2
            self.following_path = [dx * self.WP_TO_M_Coeff[0], dy * self.WP_TO_M_Coeff[1]]

        T_post_process = time.time()
        LOG.info('Image Post-Process cost time: {:.5f}s'.format(T_post_process - T_seg_inference))

        # return changed, self.full_lane_pts, self.phy_centerpts, self.following_path

        return changed, self.left_lane_fullpts, self.right_lane_fullpts, self.following_path

        # if centerpts: return full_lane_pts, centerpts, self.following_path
        # return None, None, None

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
