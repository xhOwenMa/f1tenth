import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from LaneNetToTrajectory import DualLanesToTrajectory


def main():
    lane_left_pts = np.array(
        [[10, 1], [11, 2], [12, 3], [13, 4], [14, 5], [15, 6], [15.5, 6.8], [16.4, 7.8], [16.9, 8.6], [17.7, 9.2],
        [18.3, 9.8], [19.3, 10.9], [20.2, 12], [21.6, 13.7], [22.4, 15], [23.7, 16.5], [24.7, 18], [25.7, 19.4],
        [26.7, 21], [27.6, 22.6], [28.4, 24.4], [29, 25.3], [29.3, 26.2], [30, 27.8], [30.5, 29.2], [30.8, 30.2],
        [31, 30.9], [31.3, 32.7], [31.4, 33.7], [31.4, 34.7], [31.3, 35.6], [31.1, 37], [30.7, 38.4]])
    lane_right_pts = np.array(
        [[60, 5], [59.6, 6], [59, 7], [58.6, 8], [58, 8.8], [56.7, 10.4], [55.3, 12.4], [54.6, 13.2], [54, 14.1],
        [53.5, 15.1], [52, 17.1], [51.5, 18.1], [50.8, 19.3], [50, 20.4], [49, 21.9], [48, 23.4], [47, 24.8],
        [46, 26.2], [45, 27.6], [44, 28.9], [43, 30.2], [42, 31.4], [41, 32.5], [40, 33.6], [39, 34.6], [38, 35.5],
        [37, 36.4], [35.5, 37.7], [34, 38.8]])
    # lane_right_pts = np.array(
    #     [[50.8, 19.3], [50, 20.4], [49, 21.9], [48, 23.4], [47, 24.8], [46, 26.2], [45, 27.6], [44, 28.9], [43, 30.2],
    #      [42, 31.4], [41, 32.5], [40, 33.6], [39, 34.6], [38, 35.5], [37, 36.4], [35.5, 37.7], [34, 38.8]])


    t1_start = perf_counter()

    vtt = DualLanesToTrajectory(lane_left_pts,lane_right_pts)
    matching_pts = vtt.get_matching_points()
    centerpts = vtt.get_centerpoints()
    spl = vtt.get_spline()
    ys = np.arange(1, 40, 0.1)

    t1_stop = perf_counter()


    print("Centerpoint generation finished in", t1_stop-t1_start, "seconds")


    left_match_pts = matching_pts[0]
    right_match_pts = matching_pts[1]
    x_center = centerpts[0]
    y_center = centerpts[1]


    figure(figsize=(15, 8), dpi=80)
    # figure(figsize=(15, 5), dpi=80)
    plt.scatter(lane_left_pts[:,0],lane_left_pts[:,1],label='LaneNet Left')
    plt.scatter(lane_right_pts[:,0],lane_right_pts[:,1],label='LaneNet Right')
    plt.scatter(left_match_pts[:,0],left_match_pts[:,1],label='Left Segment Pts')
    plt.scatter(right_match_pts[:,0],right_match_pts[:,1],label='Right Segment Pts')
    plt.scatter(x_center,y_center,label='Centerpoints')
    plt.plot(spl(ys), ys,label='Cubic Interpolation')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()