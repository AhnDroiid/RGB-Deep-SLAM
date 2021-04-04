import sys
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import sys
import os
import glob
import argparse
import PIL.Image as pil
import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

base = "/home/chan/dataset-color/sequences/00/image_2"
imagePath_1 = os.path.join(base, "000000.png")
image_1 = cv2.imread(imagePath_1)

def sky_detection(image, width, height):
    image_1 = cv2.resize(image, (width, height))

    canny = cv2.Canny(cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY), 0, 255)
    #kernel = np.ones((5, 5), np.uint8)
    #canny = cv2.dilate(canny, kernel, iterations=2)

    #sobel = cv2.Sobel(cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0, ksize=5)

    #sobel = cv2.Laplacian(cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY), cv2.CV_64F, ksize=15)


    #cv2.imshow("canny", canny)
    #cv2.waitKey(0)

    # for i in range(canny.shape[1]):
    #     rows = np.asarray(np.where(canny[:, i] == 255))
    #
    #     if rows.shape[1] != 0:
    #         row_min = np.min(rows)
    #         canny[row_min:, i] = 255

    return canny


# cv2.imshow("canny", sky_detection(image_1, 1024, 320))
# cv2.waitKey(0)