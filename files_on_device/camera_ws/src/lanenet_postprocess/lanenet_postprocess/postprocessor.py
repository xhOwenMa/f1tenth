#!/usr/bin/env python3
import sys
import time

import sklearn
import cv2
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from points_vector.msg import OrderedPointsVector
from geometry_msgs.msg import Point
from lanenet_out.msg import OrderedSegmentation
import ros2_numpy as rnp

sys.path.append('/home/yvxaiver/lanenet-lane-detection')
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg

sys.path.append('/home/yvxaiver/LaneNet_to_Trajectory')
from LaneNetToTrajectory import LaneProcessing, DualLanesToTrajectory

class SegmentationPostProcessor(Node):

    def __init__(self):
        super().__init__("seg_postprocess")
        self.declare_parameter('topic')
        if self.get_parameter('topic'):
            self.topic = str(self.get_parameter('topic').value)
        else:
            ValueError('No subscription topic specified.')
        self.subscriber_ = self.create_subscription(OrderedSegmentation, self.topic, self.postprocess_callback, 10)
        self.publisher_ = self.create_publisher(OrderedPointsVector, '/lanenet_parall_path', 1)

        # parameters
        # self.weights_path = "/home/yvxaiver/lanenet-lane-detection/modelv2/tusimple/bisenetv2_lanenet/tusimple_val_miou=0.6843.ckpt-1328"
        self.weights_path = "/home/yvxaiver/lanenet-lane-detection/model2023/tusimple_val_loss=2.353586_miou=0.6781.ckpt-139"
        self.image_width = 1280
        self.image_height = 720
        self.calibration = False
        self.WP_TO_M_Coeff = [0.005187456983582503, 0.0046280422281588405]
        self.lane_processor = LaneProcessing(self.image_width,self.image_height,520,70.8,self.WP_TO_M_Coeff)
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)


    def postprocess_callback(self, data):
        binary_seg_image = rnp.numpify(data.binary_seg)
        instance_seg_image = rnp.numpify(data.instance_seg)
        image_vis = rnp.numpify(data.source_img)

        self.full_lanepts, self.centerpts, self.following_path = self.get_lane_outputs(binary_seg_image, instance_seg_image, image_vis)
        msg = self.get_ordered_pv_msg(data.order)
        if msg: self.publisher_.publish(msg)

        self.image_display(image_vis)

    
    def get_lane_outputs(self, binary_seg_image, instance_seg_image, image_vis):
        T_start = time.time()

        full_lane_pts = self.postprocessor.postprocess_lanepts(
            binary_seg_result=binary_seg_image,
            instance_seg_result=instance_seg_image,
            source_image=image_vis,
            data_source='tusimple'
        )
        self.full_lane_pts = full_lane_pts
        
        self.lane_processor.process_next_lane(full_lane_pts)
        full_lane_pts = self.lane_processor.get_full_lane_pts()
        physical_fullpts = self.lane_processor.get_physical_fullpts()

        if self.calibration:
            print(self.lane_processor.WARP_RADIUS)
            print(self.lane_processor.get_wp_to_m_coeff())
            self.calibration = False
            raise SystemExit

        phy_centerpts = []
        phy_splines = []
        closest_lane_dist = float('inf')
        closest_lane_idx = 0
        if physical_fullpts:
            for i in range(len(physical_fullpts)):
                if not i: continue
                traj = DualLanesToTrajectory(physical_fullpts[i-1],physical_fullpts[i],N_centerpts=20)
                phy_centerpts.append(traj.get_centerpoints())
                phy_splines.append(traj.get_spline())
            for i in range(len(phy_splines)):
                new_dist = abs(phy_splines[i](0.2)-(self.image_width*self.WP_TO_M_Coeff[0])/2)
                if new_dist < closest_lane_dist:
                    closest_lane_dist = new_dist
                    closest_lane_idx = i
            if phy_centerpts: following_path = phy_centerpts[closest_lane_idx]

        # For display output
        centerpts = []
        if full_lane_pts:
            for i in range(len(full_lane_pts)):
                if not i: continue
                traj = DualLanesToTrajectory(full_lane_pts[i-1],full_lane_pts[i],N_centerpts=20)
                centerpts.append(traj.get_centerpoints())
        
        T_post_process = time.time()
        print('Image Post-Process cost time: {:.5f}s'.format(T_post_process-T_start))

        if centerpts: return full_lane_pts, centerpts, following_path
        return None, None, None


    def get_ordered_pv_msg(self, order):
        if self.following_path and self.full_lane_pts:
            vector = []
            for i in range(len(self.following_path[0])):
                pt = Point()
                pt.x = self.following_path[0][i]
                pt.y = self.following_path[1][i]
                pt.z = 0.0
                vector.append(pt)
            ptv = OrderedPointsVector()
            ptv.order = order
            ptv.points = vector
            ptv.x_coeff = float(self.lane_processor.get_wp_to_m_coeff()[0])
            return ptv
        return None

    def image_display(self, cv_frame):
        if self.full_lanepts:
                for lane in self.full_lanepts:
                    for pt in lane:
                        cv2.circle(cv_frame,tuple(([0,self.image_height] - pt)*[-1,1]), 5, (0, 255, 0), -1)
        if self.centerpts:
            for centerlane in self.centerpts:
                for i in range(len(centerlane[0])):
                    cv2.circle(cv_frame,(int(centerlane[0][i]),
                                        self.image_height-int(centerlane[1][i])), 5, (0, 0, 255), -1)
        if self.following_path:
            plt.close()
            plt.plot(self.following_path[0], self.following_path[1], ".r", label="path")
            plt.grid(True)
            #plt.show()
        
        cv2.imshow("camera", cv_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args) 

    lanenet_pp = SegmentationPostProcessor()

    rclpy.spin(lanenet_pp)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
