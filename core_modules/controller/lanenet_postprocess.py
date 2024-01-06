#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math
import time

import cv2
import numpy as np
import loguru
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from local_utils.log_util import init_logger

from numpy.polynomial import Polynomial  # 6/5/23

LOG = loguru.logger
TEST_LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """

    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        print(np.shape(embedding_image_feats))
        db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            LOG.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        import sys
        np.set_printoptions(threshold=sys.maxsize)

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        T_db_start = time.time()

        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        T_pre_db = time.time()
        LOG.info('*** *** Pre-DB treament cost time: {:.5f}s'.format(T_pre_db - T_db_start))

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        T_db = time.time()
        LOG.info('*** *** DBSCAN cost time: {:.5f}s'.format(T_db - T_pre_db))

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []
        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        T_db_post = time.time()
        LOG.info('*** *** Post-db treatment cost time: {:.5f}s'.format(T_db_post - T_db))

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """

    def __init__(self, image_width, image_height, resize_width, resize_height, max_lane_y, WARP_RADIUS):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        # assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        # self._cfg = cfg
        # self._cluster = _LaneNetCluster(cfg=cfg)
        # self._ipm_remap_file_path = ipm_remap_file_path

        # remap_file_load_ret = self._load_remap_matrix()
        # self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        # self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

        self.image_width = image_width
        self.image_height = image_height
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.max_lane_y = max_lane_y

        src = np.float32([[0, self.max_lane_y * self.resize_height / self.image_height - 1],
                          [self.resize_width - 1, self.max_lane_y * self.resize_height / self.image_height - 1], [0, 0],
                          [self.resize_width - 1, 0]])
        dst = np.float32(
            [[self.image_width / 2 - WARP_RADIUS, 0], [self.image_width / 2 + WARP_RADIUS, 0], [0, self.max_lane_y - 1],
             [self.image_width - 1, self.max_lane_y - 1]])
        self.M = np.array(cv2.getPerspectiveTransform(src, dst))

        self.img_counter = 0

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    with_lane_fit=True, data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param with_lane_fit:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        # if mask_image is None:
        #     return {
        #         'mask_image': None,
        #         'fit_params': None,
        #         'source_image': None,
        #     }
        # if not with_lane_fit:
        #     tmp_mask = cv2.resize(
        #         mask_image,
        #         dsize=(source_image.shape[1], source_image.shape[0]),
        #         interpolation=cv2.INTER_NEAREST
        #     )
        #     source_image = cv2.addWeighted(source_image, 0.6, tmp_mask, 0.4, 0.0, dst=source_image)
        #     return {
        #         'mask_image': mask_image,
        #         'fit_params': None,
        #         'source_image': source_image,
        #     }

        # source_image_width = source_image.shape[1]
        # source_image_height = source_image.shape[0]

        # fitted_lane_coords = []
        # fit_params = []
        # for lane in lane_coords:
        #     coord_x = np.int_(lane[:,0] * source_image_width / 512)
        #     coord_y = np.int_(lane[:,1] * source_image_height / 256)

        #     fit_param = np.polyfit(coord_y, coord_x,deg=8)
        #     fit_params.append(fit_param)

        #     spl = np.poly1d(fit_param)
        #     dy = np.linspace(min(coord_y), max(coord_y), 50)
        #     fitted_lane_coords.append(np.dstack((np.int_(spl(dy)), np.int_(dy))).reshape(len(dy),2))

        # full_lane_pts = []
        # for i in range(len(fitted_lane_coords)):
        #     final_single_lane_pts = []
        #     for pts in fitted_lane_coords[i]:
        #         if pts[0] > source_image_width or pts[0] < 10 or \
        #                 pts[1] > source_image_height or pts[1] < 0:
        #             continue
        #         lane_color = self._color_map[i].tolist()
        #         cv2.circle(source_image, (pts[0],
        #                                   pts[1]), 5, lane_color, -1)

        #         final_single_lane_pts.append([pts[0],pts[1]])

        #     full_lane_pts.append(np.array(final_single_lane_pts))

        # ret = {
        #     'mask_image': mask_image,
        #     'fit_params': fit_params,
        #     'source_image': source_image,
        # }

        ret = {
            'mask_image': mask_image,
            'fit_params': None,
            'source_image': source_image,
        }

        return ret

    def postprocess_lanepts(self, binary_seg_result, instance_seg_result=None,
                            min_area_threshold=200, source_image=None, data_source='tusimple', y_cut=0):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        T_postprocess_start = time.time()

        instance_mask = np.ones((self.resize_height, self.resize_width), dtype=np.int64)
        instance_mask[np.where((instance_seg_result[:, :, 0] > 5 / 255) & (instance_seg_result[:, :, 1] < 215 / 255) & (
                    instance_seg_result[:, :, 2] < 215 / 255))] = 0

        # cv2.imshow("Instance mask", instance_mask * 1.0)
        # cv2.waitKey(1)

        segmentation_map = binary_seg_result * 255
        segmentation_map = segmentation_map.astype(np.uint8)

        # Display the segmentation map
        cv2.imshow('Segmentation', segmentation_map)
        cv2.waitKey(1)

        left_lane_mask = np.zeros_like(segmentation_map)
        left_mask_for_measure = np.zeros_like(segmentation_map)
        right_lane_mask = np.zeros_like(segmentation_map)
        right_mask_for_measure = np.zeros_like(segmentation_map)
        lane_pixels = np.argwhere(segmentation_map > 0)  # Assuming non-zero pixels are lane pixels
        if len(lane_pixels) > 0:
            median_x_coordinate = 512 / 2  # assume the car is in the middle of the image

            for y, x in lane_pixels:
                if x < median_x_coordinate:
                    # print("Found left lane point")
                    left_lane_mask[y, x] = 255
                    if y > y_cut:
                        left_mask_for_measure[y, x] = 255
                else:
                    # print("Found right lane point")
                    right_lane_mask[y, x] = 255
                    if y > y_cut:
                        right_mask_for_measure[y, x] = 255

        T_postprocess_end = time.time()
        LOG.info('*** Post-processing cost time: {:.5f}s'.format(T_postprocess_end - T_postprocess_start))

        return left_lane_mask, right_lane_mask, left_mask_for_measure, right_mask_for_measure
