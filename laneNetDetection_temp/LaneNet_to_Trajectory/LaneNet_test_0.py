import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from LaneNetToTrajectory import LaneProcessing, DualLanesToTrajectory



full_lanenet_output = [np.array([[482, 248],
       [500, 261],
       [507, 271],
       [510, 278],
       [514, 288],
       [516, 301],
       [514, 311],
       [511, 323],
       [510, 329],
       [498, 346],
       [494, 356],
       [493, 357],
       [487, 371],
       [471, 387],
       [467, 391],
       [463, 396],
       [441, 419],
       [433, 426],
       [425, 435],
       [415, 446],
       [404, 457],
       [391, 470],
       [368, 486],
       [385, 476],
       [360, 493],
       [341, 512],
       [351, 502],
       [325, 524],
       [294, 549],
       [304, 539],
       [277, 566],
       [287, 556],
       [246, 587],
       [262, 577],
       [216, 613],
       [227, 603],
       [171, 644],
       [187, 634],
       [203, 624],
       [143, 670],
       [154, 660],
       [165, 650],
       [ 95, 703],
       [111, 693],
       [126, 683]]), np.array([[ 535,  249],
       [ 576,  262],
       [ 602,  270],
       [ 632,  282],
       [ 656,  291],
       [ 688,  304],
       [ 708,  313],
       [ 731,  324],
       [ 740,  330],
       [ 773,  346],
       [ 791,  355],
       [ 809,  366],
       [ 811,  367],
       [ 838,  381],
       [ 864,  399],
       [ 871,  403],
       [ 879,  409],
       [ 889,  415],
       [ 928,  441],
       [ 942,  451],
       [ 959,  462],
       [ 977,  475],
       [ 962,  464],
       [ 983,  478],
       [1006,  494],
       [1032,  512],
       [1017,  502],
       [1044,  522],
       [1071,  545],
       [1056,  535],
       [1093,  560],
       [1078,  550],
       [1119,  579],
       [1105,  568],
       [1145,  601],
       [1136,  591],
       [1179,  628],
       [1164,  618]]), np.array([[ 681,  249],
       [ 718,  258],
       [ 765,  271],
       [ 804,  283],
       [ 836,  292],
       [ 856,  299],
       [ 881,  308],
       [ 912,  319],
       [ 949,  333],
       [ 967,  340],
       [ 989,  349],
       [1014,  360],
       [1045,  373],
       [1081,  388],
       [1090,  392],
       [1100,  396],
       [1150,  417],
       [1163,  425],
       [1179,  433],
       [1200,  442],
       [1225,  453],
       [1248,  465],
       [1276,  479]]), np.array([[451, 250],
       [451, 260],
       [443, 269],
       [424, 283],
       [418, 288],
       [392, 302],
       [368, 313],
       [353, 318],
       [314, 334],
       [291, 343],
       [261, 354],
       [223, 367],
       [215, 369],
       [162, 387],
       [147, 391],
       [128, 397],
       [ 45, 422]]), np.array([[499, 251],
       [445, 260],
       [378, 272],
       [324, 283],
       [291, 290],
       [251, 299],
       [206, 310],
       [155, 323],
       [127, 330],
       [ 97, 339],
       [ 61, 349],
       [ 21, 361]])]



# Original Dataset
# for i in range(len(full_lanenet_output)):
#     lane_pts = full_lanenet_output[i]
#     plt.scatter(lane_pts[:,0],lane_pts[:,1])
# plt.show()

t1_start = perf_counter()

lp = LaneProcessing(full_lanenet_output,1280,720)
full_lane_pts = lp.get_full_lane_pts()

# After Processing
# for i in range(len(full_lane_pts)):
#     lane_pts = full_lane_pts[i]
#     plt.scatter(lane_pts[:,0],lane_pts[:,1])
# plt.show()


trajectories = []
centerpoints = []
splines = []
for i in range(len(full_lane_pts)):
    if not i: continue
    traj = DualLanesToTrajectory(full_lane_pts[i-1],full_lane_pts[i])
    trajectories.append(traj)
    centerpoints.append(traj.get_centerpoints())
    splines.append(traj.get_spline())

max_y = 0
for lane in centerpoints:
    new_max = max(lane[1])
    if new_max > max_y: max_y = new_max


t1_stop = perf_counter()
print("Centerlane generation finished in", t1_stop-t1_start, "seconds")


print(centerpoints)
ys = np.arange(1, max_y, 0.1)
for i in range(len(full_lane_pts)):
    lane_pts = full_lane_pts[i]
    plt.scatter(lane_pts[:,0],lane_pts[:,1])
    if i == len(full_lane_pts)-1: continue
    plt.plot(splines[i](ys), ys)
    print(splines[i])
    plt.scatter(centerpoints[i][0],centerpoints[i][1])
plt.show()  
