from __future__ import absolute_import, division, print_function
import open3d as o3d
import sys
import os
import glob
import argparse
import numpy as np
import PIL.Image as pil

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

import fusion
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt

from pcdUtil import ExtractPCD, scaleIntrinsic



rgb = cv2.imread("test/image.png")
rgb = rgb[300:365, 550:690, :]
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
rgb_copy = rgb.copy()

depth = np.ones(shape=(rgb.shape[0], rgb.shape[1]))

for i in range(rgb.shape[0]):
    depth[rgb.shape[0]-i-1, :] += 0.1 * i

intrinsic = np.loadtxt("camera_intrinsic.txt")
intrinsic[0][2] -= 550
intrinsic[1][2] -= 300

pcd = ExtractPCD(depth, intrinsic)
print(pcd[0, :])
rgb_copy = np.reshape(rgb_copy, newshape=(-1, 3))
# cv2.imshow("rgb", rgb)
# cv2.imshow("depth", ((depth) / depth.max()))
# cv2.waitKey(0)


pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_copy/rgb_copy.max())
o3d.visualization.draw_geometries([pcd_o3d])




