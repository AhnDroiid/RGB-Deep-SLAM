import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

base = "/home/chan/dataset-color/sequences/00/image_2"
imagePath_1 = os.path.join(base, "000000.png")
imagePath_2 = os.path.join(base, "000001.png")

image_1 = cv2.imread(imagePath_1)
image_2 = cv2.imread(imagePath_2)

rgb_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
rgb_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create()

key_1, decriptor_1 = orb.detectAndCompute(gray_1, None)
key_2, decriptor_2 = orb.detectAndCompute(gray_2, None)

dst_1 = rgb_1.copy()
dst_2 = rgb_2.copy()

cv2.drawKeypoints(rgb_1, key_1, dst_1, color=(0, 255, 0))
cv2.drawKeypoints(rgb_2, key_2, dst_2, color=(0, 255, 0))


bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# perform matching
matches = bf.match(decriptor_1, decriptor_2)
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(rgb_1, key_1, rgb_2, key_2, matches1to2=matches, outImg=dst_2)


cv2.imshow("match", result)
cv2.waitKey(0)