#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from points_vector.msg import PointsVector

from geometry_msgs.msg import Point

import sklearn 
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

from .bridge.lanenet_bridge import LaneNetImageProcessor



class ImageProcessorNode(Node):

    '''Handles feeding camera frame into lanenet, converting outputs into path to be followed, and publishes that path.'''

    def __init__(self):
        super().__init__('image_processor')
        self.subscriber_ = self.create_subscription(Image, '/raw_frame', self.image_callback, 1)
        self.position_subscriber_ = self.create_subscription(Point, '/position', self.position_callback, 1)
        self.publisher_ = self.create_publisher(PointsVector, '/lanenet_path', 1)
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
        self.processor = LaneNetImageProcessor(self.weights_path,self.image_width,self.image_height,self.max_lane_y,self.WARP_RADIUS,self.WP_TO_M_Coeff)
        self.lanenet_status = self.processor.init_lanenet()
        #self.lanenet_status = False
        
        self.left_lane_pts = []
        self.right_lane_pts = []
        self.following_path = []
        
        # self.phy_centerpts = []
        # self.full_lanepts = []
        
        self.image_serial_n = 0
        
        src = np.float32([[0, self.max_lane_y - 1], [self.image_width - 1, self.max_lane_y - 1], [0, 0], [self.image_width - 1, 0]])
        dst = np.float32([[self.image_width/2 - self.WARP_RADIUS, 0], [self.image_width/2 + self.WARP_RADIUS, 0], [0, self.max_lane_y - 1], [self.image_width - 1, self.max_lane_y - 1]])
        self.M = np.array(cv2.getPerspectiveTransform(src, dst))
        
        self.x = self.image_width * self.WP_TO_M_Coeff[0] / 2
        self.y = -0.584
        self.yaw = np.radians(90.0)
        
        
    def image_callback(self, data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            if self.lanenet_status:
                # cv2.imwrite('/home/yvxaiver/output/orig/'+str(self.image_serial_n)+".png",cv_frame)
                # changed, full_lanepts, phy_centerpts, following_path = self.processor.image_to_trajectory(cv_frame, self.x, self.y, self.yaw)
                
                tmp_x = self.x
                tmp_y = self.y
                tmp_yaw = self.yaw
                
                # print(tmp_y, '\n\n\n')
                changed, left_lane_pts, right_lane_pts, following_path = self.processor.image_to_trajectory(cv_frame, tmp_x, tmp_y, tmp_yaw)
                
                #print(changed)
                #print(full_lanepts)
                #print(self.full_lanepts)
                #print(self.phy_centerpts)

                
                if changed:
                    self.left_lane_pts = left_lane_pts
                    self.right_lane_pts = right_lane_pts
                    self.following_path = following_path
                    msg = self.processor.get_point_vector_path()
                    if msg: self.publisher_.publish(msg)
                
                else:
                    if left_lane_pts:
                        self.left_lane_pts = self.processor.shift(left_lane_pts, tmp_x, tmp_y, tmp_yaw, old_to_new=1)
                    
                    if right_lane_pts:
                        self.right_lane_pts = self.processor.shift(right_lane_pts, tmp_x, tmp_y, tmp_yaw, old_to_new=1)
                    
                    if following_path:
                        self.following_path = self.processor.shift(following_path, tmp_x, tmp_y, tmp_yaw, old_to_new=1, pixels=False)
                        
            # self.image_save(cv_frame) 
            self.image_display(cv_frame)

        except Exception as e:
            print(e)
            
    def position_callback(self, data):
        self.x = data.x
        self.y = data.y
        self.yaw = data.z

    
    def image_display(self, cv_frame):
        """
        if self.full_lanepts:
                for lane in self.full_lanepts:
                    for pt in lane:
                        cv2.circle(cv_frame,tuple(([0,self.image_height] - pt)*[-1,1]), 5, (0, 255, 0), -1)
        if self.centerpts:
            for centerlane in self.centerpts:
                for i in range(len(centerlane[0])):
                    cv2.circle(cv_frame,(int(centerlane[0][i]),
                                        self.image_height-int(centerlane[1][i])), 5, (0, 0, 255), -1)
                    
                    # print('Centerpoint PIXELS: ', centerlane[0][i], ' ', centerlane[1][i])
                    # print('Centerpoint METERS: ', self.following_path[0][i], ' ', self.following_path[1][i])
                    # print('----------------------------------------')
                    
        print('Closest centerpoint (m): ({},{})'.format(self.following_path[0][0], self.following_path[1][0]))
        
        if self.following_path:
            plt.close()
            plt.plot(self.following_path[0], self.following_path[1], ".r", label="path")
            plt.grid(True)
            plt.show(block=False)
            #for pt in range(len(self.following_path)):
                #cv2.circle(cv_frame, (int(self.following_path[0][pt]), int(self.following_path[1][pt])), 5, (138,43,226), -1)
        
        # self.image_save(cv_frame)
        cv2.imshow("camera", cv_frame)
        cv2.waitKey(1)
        """
        
        # cv2.imwrite('/home/yvxaiver/output/orig/%d.png' % self.image_serial_n, cv_frame)
        cv2.imwrite('/home/yvxaiver/output/orig/%d.png' % self.image_serial_n, cv_frame)
        
        warped = cv2.warpPerspective(cv_frame[self.image_height - self.max_lane_y:self.image_height], self.M, (self.image_width,self.max_lane_y))
        
        
        # cv2.imshow("Warped unannotated", cv2.flip(warped, 0))
        # cv2.waitKey(1)
        
        """
        for lane in self.full_lanepts:
            for i in range(len(lane[0])):
                cv2.circle(warped, (np.int_(lane[0][i]), np.int_(lane[1][i])), 5, (0, 255, 0), -1)
        
        if self.phy_centerpts:
            for centerlane in self.phy_centerpts:
                for i in range(len(centerlane[0])):
                    cv2.circle(warped, (np.int_(centerlane[0][i] / self.WP_TO_M_Coeff[0]), np.int_(centerlane[1][i] / self.WP_TO_M_Coeff[1])), 5, (0, 0, 255), -1)
                
        if self.following_path:
            for i in range(len(self.following_path[0])):
                cv2.circle(warped, (np.int_(self.following_path[0][i] / self.WP_TO_M_Coeff[0]), np.int_(self.following_path[1][i] / self.WP_TO_M_Coeff[1])), 5, (255, 0, 0), -1)
        """
        
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
                
        cv2.imshow("Birds eye view", cv2.flip(warped, 0))
        # cv2.waitKey(1)
        cv2.waitKey(5)

    
    def image_save(self, cv_frame):
        status = cv2.imwrite('/home/yvxaiver/output/0/'+str(self.image_serial_n)+".png",cv_frame)
        self.image_serial_n += 1
        # print(status)
        

def main(args=None):
    rclpy.init(args=args)

    image_processor = ImageProcessorNode()
    rclpy.spin(image_processor)
    
    """
    e1 = Executor()
    e1.add_node(image_processor)
    t1 = Thread(target=rclpy.spin, args=(e1, ), daemon=True)
    t1.start()
    # e1.spin()
    
    e2 = Executor()
    e2.add_node(image_processor.processor)
    t2 = Thread(target=rclpy.spin, args=(e2, ), daemon=True)
    t2.start()
    # e2.spin()
    # rclpy.spin(image_processor.processor, executor=e2)
    """

    rclpy.shutdown()

if __name__ == '__main__':
    main()
