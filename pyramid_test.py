import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os

base = "/home/chan/dataset-color/sequences/00/image_2"
img = cv2.imread(os.path.join(base, "000000.png"))
half_img = cv2.pyrDown(img)
half_half_img = cv2.pyrDown(half_img)

cv2.imshow("img", img)
cv2.imshow("half", half_img)
cv2.imshow("half half", half_half_img)
cv2.waitKey(0)