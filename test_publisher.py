import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:
    print("removed ros_path")
    sys.path.remove(ros_path)

import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("/usr/lib/python2.7/dist-packages")
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


import time
import os
rospy.init_node("ddd")


print("load kitti test images")
image_path = "/home/chan/dataset-color/sequences/00/image_2"
img_list = sorted(os.listdir(image_path))
# test_img = test_img[0:372, 0:1240]
pub = rospy.Publisher("image", Image, queue_size=10)
b = CvBridge()
idx = 0
while idx != len(img_list):

    print(os.path.join(image_path, img_list[idx]))
    test_img = cv2.imread(os.path.join(image_path, img_list[idx]))
    test_img = test_img[0:372, 0:1240]
    cv2.imshow("rgb", test_img)
    cv2.waitKey(1)
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    msg = b.cv2_to_imgmsg(test_img, encoding="passthrough")
    #print(msg.encoding)
    pub.publish(msg)
    idx += 1
    time.sleep(0.1)

