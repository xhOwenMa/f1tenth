from concurrent.futures import process
import numpy as np
from scipy.interpolate import UnivariateSpline
import math
import cv2


class LaneProcessing():
    """Takes in LaneNet's raw output vectors and process then into left to right ordered lanes"""

    def __init__(self,image_width,image_height,max_lane_y=420,WARP_RADIUS=20,WP_TO_M_Coeff=[1,1]):
        self.image_width = image_width
        self.image_height = image_height
        self.max_lane_y = max_lane_y
        self.WARP_RADIUS = WARP_RADIUS
        self.M = np.empty(shape=(0,0))

        # Warped Pixel to Meter Conversion Coeff.
        self.WP_TO_M_Coeff = WP_TO_M_Coeff

    def process_next_lane(self, full_lane_pts):
        self.full_lane_pts = full_lane_pts
        self._full_lanes_transformation()
        self._ordering_lanes()

    def get_physical_fullpts(self):
        self._warp_full_lane_pts()
        self.physical_fullpts = []
        for lane in self.warped_fullpts:
            physical_lane = lane * self.WP_TO_M_Coeff
            self.physical_fullpts.append(physical_lane)
        return self.physical_fullpts

    def get_full_lane_pts(self):
        return self.full_lane_pts

    def _ordering_lanes(self):
        if not self.full_lane_pts:
            return

        max_y_pts = []
        min_y_pts = []
        min_y_VAL = []

        for lane in self.full_lane_pts:
            max_y_pts.append(lane[np.argmax(lane[:,1])])
            min_y_pts.append(lane[np.argmin(lane[:,1])])
            min_y_VAL.append(lane[np.argmin(lane[:,1]),1])

        max_y_pts = np.array(max_y_pts)
        min_y_pts = np.array(min_y_pts)
        maxmin_y_val = max(min_y_VAL)

        slopes = (max_y_pts[:,1]-min_y_pts[:,1])/(max_y_pts[:,0]-min_y_pts[:,0])
        intercepts = min_y_pts[:,1] - slopes*min_y_pts[:,0]
        order_x_values = (maxmin_y_val - intercepts)/slopes

        lane_ordering = []
        for i in range(len(order_x_values)):
            lane_ordering.append((order_x_values[i],self.full_lane_pts[i]))
        dtype = [("x_values",float),("lane_pts",list)]
        lane_ordering = np.array(lane_ordering,dtype=dtype)
        lane_ordering = np.sort(lane_ordering, order="x_values")

        ordered_lane_pts = []
        for _,lane_pts in lane_ordering:
            ordered_lane_pts.append(lane_pts)

        self.full_lane_pts = ordered_lane_pts

    def _full_lanes_transformation(self):
        for i in range(len(self.full_lane_pts) - 1, -1, -1):
            if len(self.full_lane_pts[i]) == 0:
                del self.full_lane_pts[i]
                
        for i in range(len(self.full_lane_pts)):
            self.full_lane_pts[i] = ([0,self.image_height] - self.full_lane_pts[i]) * [-1,1]
            idx = np.argsort(self.full_lane_pts[i], axis=0)
            self.full_lane_pts[i][:,0] = np.take_along_axis(self.full_lane_pts[i][:,0], idx[:,1], axis=0)
            self.full_lane_pts[i][:,1] = np.take_along_axis(self.full_lane_pts[i][:,1], idx[:,1], axis=0)

    def auto_warp_radius_calibration(self,FRAME_BOTTOM_PHYSICAL_WIDTH=None):
        """
        Auto calibrates the warp radius given lane lines of straight line lanes.
        FRAME_BOTTOM_PHYSICAL_WIDTH given in meters.
        """
        if len(self.full_lane_pts) >= 2:

            Y_MAX_CUTOFF = self.max_lane_y
            X_MAX_CUTOFF = self.image_width
            src = np.float32([[0, 0], [X_MAX_CUTOFF, 0], [0, Y_MAX_CUTOFF], [X_MAX_CUTOFF, Y_MAX_CUTOFF]])
            self.WARP_RADIUS = X_MAX_CUTOFF/20
            warp_r_temp = self.WARP_RADIUS

            error = float('inf')
            while abs(error) > 2:
                self.WARP_RADIUS = warp_r_temp
                dst = np.float32([[X_MAX_CUTOFF/2 - self.WARP_RADIUS, 0], [X_MAX_CUTOFF/2 + self.WARP_RADIUS, 0], [0, Y_MAX_CUTOFF], [X_MAX_CUTOFF, Y_MAX_CUTOFF]])
                self.M = np.array(cv2.getPerspectiveTransform(src, dst))

                self._warp_full_lane_pts()
                pt1 = self.warped_fullpts[0][0]
                pt2 = self.warped_fullpts[0][-1]
                pt3 = self.warped_fullpts[1][0]
                pt4 = self.warped_fullpts[1][-1]
                line1 = np.poly1d(np.polyfit([pt1[1],pt2[1]],[pt1[0],pt2[0]],deg=1))
                line2 = np.poly1d(np.polyfit([pt3[1],pt4[1]],[pt3[0],pt4[0]],deg=1))
                error = abs(line2(self.max_lane_y)-line1(self.max_lane_y)) - abs(line2(0)-line1(0))
                warp_r_temp += error/6.0

            if FRAME_BOTTOM_PHYSICAL_WIDTH:
                self.WP_TO_M_Coeff[0] = FRAME_BOTTOM_PHYSICAL_WIDTH/(self.WARP_RADIUS*2)

        else:
            raise

    def y_dist_calibration_tool(self, cv_frame):
        '''
        Mark points on the screen parallel to car that are 1 meter in distance in the physical world.
        Alert: Do auto_warp_radius_calibration before y_dist_calibration_tool to get the correct warp radius.
        '''

        double_point = []

        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(cv_frame, tuple([x,y]), 5, (0, 0, 255), -1)
                cv2.imshow('y_calibration_tool', cv_frame)
                process_pts([x,y])
        
        def process_pts(pts):
            if len(double_point) < 1:
                double_point.append(pts)
            else:
                double_point.append(pts)

                self.temp_full_lanes = self.full_lane_pts
                self.full_lane_pts = [np.array(double_point)]

                self._warp_full_lane_pts()
                self.WP_TO_M_Coeff[1] = 1.0 / abs(self.warped_fullpts[0][0,1] - self.warped_fullpts[0][1,1])

                self.full_lane_pts = self.temp_full_lanes
                self._warp_full_lane_pts()
        
        cv2.imshow('y_calibration_tool', cv_frame)
        cv2.setMouseCallback('y_calibration_tool', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _warp_full_lane_pts(self):
        appended_lane = []
        for lane in self.full_lane_pts:
            lane = np.append(lane,np.ones([len(lane),1]),1)
            appended_lane.append(lane)
        
        if self.M.size == 0:
            Y_MAX_CUTOFF = self.max_lane_y
            X_MAX_CUTOFF = self.image_width
            src = np.float32([[0, 0], [X_MAX_CUTOFF, 0], [0, Y_MAX_CUTOFF], [X_MAX_CUTOFF, Y_MAX_CUTOFF]])
            dst = np.float32([[X_MAX_CUTOFF/2 - self.WARP_RADIUS, 0], [X_MAX_CUTOFF/2 + self.WARP_RADIUS, 0], [0, Y_MAX_CUTOFF], [X_MAX_CUTOFF, Y_MAX_CUTOFF]])
            self.M = np.array(cv2.getPerspectiveTransform(src, dst))

        self.warped_fullpts = []
        for lane in appended_lane:
            warped_lane = []
            for pt in lane: 
                f = self.M.dot(pt.reshape(3,1)).reshape(3,)
                warped_lane.append([f[0]/f[2], f[1]/f[2]])
            self.warped_fullpts.append(np.array(warped_lane))
        
    def get_wp_to_m_coeff(self):
        return self.WP_TO_M_Coeff

    


class DualLanesToTrajectory():
    """Takes two x by 2 shaped lane line point vector arrays and outputs estimated centerlines."""

    def __init__(self,lane_left_pts,lane_right_pts,N_centerpts=10):
        self.lane_pts = [lane_left_pts,lane_right_pts]
        self.tot_dist = []
        self.seg_cum_dists = []
        self.seg_vectors = []
        self.matching_pts = []
        self.centerpoints = []

        self.N_centerpts = N_centerpts
        self.result_status = True

        self._update_centerpoints()


    def _pre_processing_lane_pts(self):
        for i in range(len(self.lane_pts)):
            lane_side_pts = self.lane_pts[i]
            if self.lane_pts[i].size <= 1:
                self.result_status = False
                return None
            else:
                self.result_status = True
            """
            if lane_side_pts[0,1] >= 0:
                extrapo_vector = lane_side_pts[1] - lane_side_pts[0]
                new_pt_x = lane_side_pts[0,0] + (lane_side_pts[0,1] * -1.0 * extrapo_vector[0] / extrapo_vector[1])
                self.lane_pts[i] = np.insert(lane_side_pts, 0, np.array([new_pt_x, 0]), axis=0)
            else:
                self.lane_pts[i] = lane_side_pts
            """

    def _cal_segment_param(self):
        for lane_side_pts in self.lane_pts:
            total_distance = 0
            segment_cumulative_distances = []
            segment_vectors = []
            for i in range(len(lane_side_pts)):
                if i==0: continue
                seg_dist = math.sqrt((lane_side_pts[i][0] - lane_side_pts[i-1][0])**2 + (lane_side_pts[i][1] - lane_side_pts[i-1][1])**2)
                segment_cumulative_distances.append(total_distance)
                segment_vectors.append(lane_side_pts[i]-lane_side_pts[i-1])
                total_distance += seg_dist
            self.tot_dist.append(total_distance)
            self.seg_cum_dists.append(np.array(segment_cumulative_distances))
            self.seg_vectors.append(np.array(segment_vectors))


    def _cal_centerpts_pairs(self):
        for i in range(len(self.lane_pts)):
            total_distance = self.tot_dist[i]
            segment_cumulative_distances = self.seg_cum_dists[i]
            segment_vectors = self.seg_vectors[i]
            lane_pts = self.lane_pts[i]
            wedge_dists = np.linspace(0, total_distance, self.N_centerpts, endpoint=True)
            starting_pts_index = np.searchsorted(segment_cumulative_distances,wedge_dists, side="left") - 1
            lacking_dists = wedge_dists-np.take(segment_cumulative_distances, starting_pts_index)
            matching_pts = []
            i = 0
            for idx in starting_pts_index:
                theta = math.atan(segment_vectors[idx][1]/segment_vectors[idx][0] if segment_vectors[idx][0] != 0 else 0)
                x,y = 0,0
                if theta > 0:
                    x = lacking_dists[i] * math.cos(theta) + lane_pts[idx][0]
                    y = lacking_dists[i] * math.sin(theta) + lane_pts[idx][1]
                else:
                    x = -1 * lacking_dists[i] * math.cos(theta) + lane_pts[idx][0]
                    y = -1 * lacking_dists[i] * math.sin(theta) + lane_pts[idx][1]
                matching_pts.append([x,y])
                i += 1

            self.matching_pts.append(np.array(matching_pts))


    def _cal_centerpts(self):
        left_pts = self.matching_pts[0]
        right_pts = self.matching_pts[1]
        self.centerpoints = (left_pts[:,0]+right_pts[:,0])/2 , (left_pts[:,1]+right_pts[:,1])/2

    def update_input(self,lane_left_pts,lane_right_pts):
        self.lane_pts = [np.array(lane_left_pts),np.array(lane_right_pts)]
        self._update_centerpoints()


    def _update_centerpoints(self):
        self._pre_processing_lane_pts()
        if self.result_status: 
            self._cal_segment_param()
            self._cal_centerpts_pairs()
            self._cal_centerpts()

    def get_centerpoints(self):
        return self.centerpoints

    def get_matching_points(self):
        return self.matching_pts

    def get_spline(self):
        x_center, y_center = self.centerpoints
        return np.poly1d(np.polyfit(y_center, x_center, deg=2))
