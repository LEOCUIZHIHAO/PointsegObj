import numpy as np
import open3d
from open3d import *
from tqdm import tqdm
from random import sample
import math as m
import os
import pickle

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles+1.57)
    rot_cos = np.cos(angles+1.57)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")
    return np.einsum('ij,jk->ik', points, rot_mat_T)

def generate_box(anchors, color = [], rotate = False):
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
        for i in range(1,4):
            box[i+4,i-1] -= anchor[i-1+3]
        angles = anchor[-1]
        # offset to origin
        box -= anchor[:3]
        # Rotate about Z axis
        box = rotation_3d_in_axis(box, angles, axis=2)
        box += anchor[:3]
        # Draw lines for each single points to make bounding box
        test_line = np.array([[0,1],[0,2],[0,3],[4,5],[4,6],[4,7],[2,5],[2,7],[3,6],[3,5],[1,6],[1,7]])
        line_set = LineSet()
        line_set.points = Vector3dVector(box)
        line_set.lines = Vector2iVector(test_line)
        if len(color) != 0:
            colors = [color for i in range(len(test_line))]
            line_set.colors = Vector3dVector(colors)
        anchor_boxes.append(line_set)
    return anchor_boxes


"""
To display point cloud, read the point cloud file and reshape to (n, 3)
create a PointCloud variable and put points in via Vector3dVector.
install open3d to use
"""

dir_path = "./3d_visual"

# Original points pickle path
points_path = os.path.join(dir_path, "points.pkl")
try: p = open(points_path, "rb")
except: p = []
# Ground truth box pickle path
gt_boxes_path = os.path.join(dir_path, "gt_boxes.pkl")
try: gt_b = open(gt_boxes_path, "rb")
except: gt_b = []
# Ground truth points pickle path
gt_points_path = os.path.join(dir_path, "gt_points.pkl")
try: gt_p = open(gt_points_path, "rb")
except: gt_p = []
# Predicted segmentation points pickle path
seg_points_path = os.path.join(dir_path, "seg_points.pkl")
try: seg_p = open(seg_points_path, "rb")
except: seg_p = []
# Foreground anchor box pickle path
fg_boxes_path = os.path.join(dir_path, "fg_boxes.pkl")
try: fb_b = open(fg_boxes_path, "rb")
except: fb_b = []
# predicted box pickle path
pd_boxes_path = os.path.join(dir_path, "pd_boxes.pkl")
try: pd_b = open(pd_boxes_path, "rb")
except: pd_b = []
# predicted box pickle path
img_name_path = os.path.join(dir_path, "img_name.pkl")
try: img_n = open(img_name_path, "rb")
except: img_n = []

while True:
    try:
        # Display segmented result points
        v = pickle.load(seg_p)#.reshape(-1,3)
        seg = PointCloud()
        seg.points = Vector3dVector(v[:,:3])
        seg.paint_uniform_color([0.7,0,0])
    except EOFError as e:
        print("!!! End of File " + seg_points_path)
        break
    except:
        seg = PointCloud()
        print("!!! Cannot find segmentation result file: " + seg_points_path)

    try:
        # Display ground truth car points
        v = pickle.load(gt_p)#.reshape(-1,3)
        gt = PointCloud()
        gt.points = Vector3dVector(v[:,:3])
        gt.paint_uniform_color([0,0.7,0])
    except EOFError as e:
        print("!!! End of File " + gt_points_path)
        break
    except:
        gt = PointCloud()
        print("!!! Cannot find ground truth points file: " + gt_points_path)

    try:
        # Display original points
        v = pickle.load(p)#.reshape(-1,3)
        pt = PointCloud()
        pt.points = Vector3dVector(v[:,:3])
        pt.paint_uniform_color([0.7,0.7,0.7])
    except EOFError as e:
        print("!!! End of File " + points_path)
        break
    except:
        pt = PointCloud()
        print("!!! Cannot find original points file: " + points_path)

    try:
        # Display ground truth boxes
        v = pickle.load(gt_b)#.reshape(-1,3)
        # Color Green
        gt_list = generate_box(v, color = [0,1,0])
    except EOFError as e:
        print("!!! End of File " + gt_boxes_path)
        break
    except:
        print("!!! Cannot find ground truth box file: " + gt_boxes_path)
        gt_list = [PointCloud()]

    try:
        # Display foreground boxes
        v = pickle.load(fb_b)#.reshape(-1,3)
        # Color Blue
        fg_list = generate_box(v, color = [0,0,1])
    except EOFError as e:
        print("!!! End of File " + fg_boxes_path)
        break
    except:
        print("!!! Cannot find foreground anchor file: " + fg_boxes_path)
        fg_list = [PointCloud()]

    try:
        # Display foreground boxes
        v = pickle.load(pd_b)#.reshape(-1,3)
        # Color Red
        pd_list = generate_box(v, color = [1,0,0])
    except EOFError as e:
        print("!!! End of File " + pd_boxes_path)
        break
    except:
        print("!!! Cannot find foreground anchor file: " + pd_boxes_path)
        pd_list = [PointCloud()]

    """
    draw_geometries display point cloud and anchor boxes
    To display multiple boxes and point cloud
    Put them into a list to display all of them
    Anchor box needed to have a star to function properly
    """
    try:
        # Display foreground boxes
        name = pickle.load(img_n)#.reshape(-1,3)
    except:
        name = 0
        print("!!! Cannot find foreground anchor file: " + pd_boxes_path)
    draw_geometries([seg, gt, pt, *pd_list, *gt_list, *fg_list], window_name = "this is image {}".format(name))#[pcc, org])
