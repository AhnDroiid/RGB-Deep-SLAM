import open3d as o3d
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from matplotlib import pyplot as plt
import numpy as np
from pcdUtil import ExtractPCD

left = cv2.imread("test/left.png")
right = cv2.imread("test/right.png")
rgb = left.copy()
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
rgb = np.reshape(rgb, newshape=(-1, 3))

img_left_bw = cv2.blur(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),(3,3))
img_right_bw = cv2.blur(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),(3,3))

stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)
disparity = stereo.compute(img_left_bw,img_right_bw)

intrinsic = np.loadtxt("camera_intrinsic.txt")
focal_length = intrinsic[0][0]

img = disparity.copy()
baseline = 0.54
depth = 0.54 * focal_length / disparity

pcd = ExtractPCD(depth, intrinsic)

pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
pcd_o3d.colors = o3d.utility.Vector3dVector(rgb/rgb.max())
o3d.visualization.draw_geometries([pcd_o3d])

# plt.imshow(img, 'CMRmap_r')
# plt.waitforbuttonpress()

# pcd = o3d.io.read_point_cloud("out.ply")
# o3d.visualization.draw_geometries([pcd])