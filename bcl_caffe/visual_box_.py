import numpy as np
import open3d
from open3d import *
from tqdm import tqdm
from random import sample
import math as m


def generate_box(anchors, color = False, rotate = False):
    """
    input:
    Anchor (n, 7)
    color: if true then the generated box will display red, otherwise black
    """
    anchor_boxes = []
    for anchor in tqdm(anchors):
        anchor[3:-1] = np.array([anchor[4],anchor[3],anchor[5]])
        box = np.zeros(shape = (8,3))
        box[:4] = anchor[:3] - anchor[3:-1]/2
        box[4:] = anchor[:3] + anchor[3:-1]/2
        for i in range(1,4):
            box[i,i-1] += anchor[i-1+3]
            # for j in range(3):
            #     box[i,j] = anchor[j] + anchor[j+3]
        # print(box)
        for i in range(1,4):
            box[i+4,i-1] -= anchor[i-1+3]
            # for j in range(3):
            #     box[i,j] = anchor[j] + anchor[j+3]
        # print(box)
        # exit()
        test_line = np.array([[0,1],[0,2],[0,3],[4,5],[4,6],[4,7],[2,5],[2,7],[3,6],[3,5],[1,6],[1,7]])
        line_set = LineSet()
        line_set.points = Vector3dVector(box)
        line_set.lines = Vector2iVector(test_line)
        if color:
            colors = [[1, 0, 0] for i in range(len(test_line))]
            line_set.colors = Vector3dVector(colors)
        anchor_boxes.append(line_set)
    return anchor_boxes


"""
To display point cloud, read the point cloud file and reshape to (n, 3)
create a PointCloud variable and put points in via Vector3dVector.
install open3d to use
"""
voxels = np.load("before_sample_points.npy")#.reshape(-1,4)
pcd = PointCloud()
pcd.points = Vector3dVector(voxels[:,:3])

v = np.load("voxels.npy").reshape(-1,4)
pcc = PointCloud()
# v[:,2] += 20 #Shift point cloud in the Z axis
pcc.points = Vector3dVector(v[:,:3])

v = np.load("anchors_fg.npy")
anchor = PointCloud()
# v[:,2] += 20
anchor.points = Vector3dVector(v[:,:3])
box_list = generate_box(v)

v = np.load("gt_boxes.npy")
gt = PointCloud()
# v[:,2] += 20
gt.points = Vector3dVector(v[:,:3])
gt_list = generate_box(v, color = True)

"""
draw_geometries display point cloud and anchor boxes
To display multiple boxes and point cloud
Put them into a list to display all of them
Anchor box needed to have a star to function properly
"""
draw_geometries([pcc, *box_list, *gt_list])#[pcc, org])
