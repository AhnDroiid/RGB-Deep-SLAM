# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from pcdUtil import ExtractPCD, scaleIntrinsic, NormalizeRGBImage
from sky_detection import sky_detection
from optimize_test import gauss_newton
import time
from liegroups import SO3
from pcdUtil import TransformPCD, ProjectPCD, downsampleDepth

device = torch.device("cuda")

model_path = os.path.join("models", "mono+stereo_1024x320")
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()

def test_simple(image_path, num_level):
    """Function to predict for a single image or folder of images
    """
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # Load image and preprocess
        rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        input_width = rgb.shape[1]
        input_height = rgb.shape[0]
        rgb = cv2.resize(rgb, dsize=(feed_width, feed_height))
        rgb_original = rgb.copy()


        rgb = np.reshape(rgb, newshape=(-1, 3))
        intrinsic = np.loadtxt("camera_intrinsic.txt", dtype=np.float64)

        # Scale intrinsic parameter according to scaling factor.
        intrinsic = scaleIntrinsic(intrinsic=intrinsic, xscaleRatio=feed_width / input_width,
                                   yscaleRatio=feed_height / input_height)

        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

        depth *= 5.4
        depth = depth.squeeze().cpu().numpy()  # [height, width, 1]


        #depth_cut = np.array(np.where((np.reshape(depth.copy(), newshape=(1, -1)))[0, :] < 80))[0]
        #sky_cut = np.array(np.where((np.reshape(sky_.copy(), newshape=(1, -1)))[0, :] != 0))[0]
        #cut = np.intersect1d(depth_cut, sky_cut)

        pcd_pyramid = []
        rgb_pyramid = []
        intrinsic_pyramid=[]

        # original data
        pcd = ExtractPCD(depth, intrinsic)
        pcd_cut = pcd[:, :]
        rgb_cut = rgb[:, :]

        depth_pyr = depth.copy()
        intrinsic_pyr = intrinsic.copy()
        rgb_pyr = rgb_original.copy()

        # pyramid data
        for idx in range(0, num_level):
            # downsample depth and adjust intrinsic
            intrinsic_pyr = intrinsic_pyr / 2
            depth_pyr = downsampleDepth(depth_pyr)
            rgb_pyr = cv2.pyrDown(rgb_pyr)
            pcd_pyr = ExtractPCD(depth_pyr, intrinsic_pyr)

            pcd_pyr_cut = pcd_pyr[:, :]
            rgb_pyr_cut = rgb_pyr[:, :]

            pcd_pyramid.append(pcd_pyr_cut)
            rgb_pyramid.append(NormalizeRGBImage(rgb_pyr_cut))
            intrinsic_pyramid.append(intrinsic_pyr)


        #print("Point cloud position shape:", pcd.shape, "Point cloud color shape:", rgb.shape)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_cut)
        pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_cut / rgb_cut.max())

        return pcd_o3d, pcd, rgb, pcd_pyramid, rgb_pyramid, intrinsic_pyramid



if __name__ == '__main__':
    base = "/home/chan/dataset-color/sequences/00/image_2"
    image_1 = os.path.join(base, "000002.png")
    image_2 = os.path.join(base, "000000.png")
    pcd_integrate = o3d.geometry.PointCloud()
    gt_pose = np.loadtxt("/home/chan/dataset-color/poses/00.txt")[1:, :]
    current_transformation = np.eye(4)

    for root, dir, files in os.walk(base):
        length = len(files)
        for i, name in enumerate(sorted(files)):
            if i >= 190 and i <= 230:

                st = time.time()
                image_1 = os.path.join(base, sorted(files)[i+1])
                image_2 = os.path.join(base, sorted(files)[i])

                current_pcd_o3d, current_original_pcd, current_color, current_pcd_pyramid, current_rgb_pyramid, current_intrinsic_pyramid = test_simple(image_1, num_level=3)  # N x 3 pcd
                prev_pcd_o3d, prev_original_pcd, prev_color, prev_pcd_pyramid, prev_rgb_pyramid, prev_intrinsic_pyramid  = test_simple(image_2, num_level=3)  # N x 3 pcd

                print("current pcd shape : {}, prev pcd shape : {}".format(current_original_pcd.shape, prev_original_pcd.shape))

                # Do Optimization...

                #print("preparation time:", time.time() - st)

                st = time.time()
                optimizedPose = gauss_newton(prev_pcd_pyramid=prev_pcd_pyramid,
                                             current_image_pyramid=current_rgb_pyramid, prev_image_pyramid=prev_rgb_pyramid,
                                             intrinsic_pyramid=current_intrinsic_pyramid, num_level=3)
                print("Optimization time:", time.time() - st)

                relativePose = np.eye(4)
                relativePose[:3, :3] = SO3.exp(np.reshape(optimizedPose[3:], (3,))).mat #(current frame to prev frame)
                relativePose[:3, 3] = optimizedPose[:3, 0]

                current_transformation = relativePose @ current_transformation
                inv_current_transformation = np.linalg.inv(current_transformation)



                gt_transformation = np.zeros(shape=(3, 4))
                gt_transformation[0, :] = gt_pose[i, 0:4]
                gt_transformation[1, :] = gt_pose[i, 4:8]
                gt_transformation[2, :] = gt_pose[i, 8:12]

                # a = inv_current_transformation[:3, :4]
                # print(a)

                print("index :{}, Optimized Finished!".format(i))


                #print("w_x: {}, w_y: {}, w_z: {}, t_x: {}, t_y : {}, t_z : {}".format(pose[3], pose[4], pose[5], pose[0], pose[1], pose[2]))
                # Transformation
                # inv_current_transformation = np.reshape(np.array([9.999978e-01, 5.272628e-04 ,-2.066935e-03 ,-4.690294e-02 ,
                #                                                   -5.296506e-04, 9.999992e-01 ,-1.154865e-03 ,-2.839928e-02 ,
                #                                                   2.066324e-03 ,1.155958e-03, 9.999971e-01, 8.586941e-01]), newshape=(3, 4))


                current_original_pcd = TransformPCD(None, current_original_pcd.T,
                                                inv_current_transformation[:3, :3],
                                                np.reshape(inv_current_transformation[:3, 3], (3, 1)), mode="CPU")

                # original_point_1 = TransformPCD(original_point_1.T, gt_transformation[:3, :3],
                #                                 np.reshape(gt_transformation[:3, 3], (3,1)), mode="CPU")


                current_original_pcd = current_original_pcd.T

                print(current_original_pcd.shape, prev_original_pcd.shape)

                # After Optimization...
                if i == 190:
                    pcd_integrate_points = np.concatenate([current_original_pcd, prev_original_pcd], axis=0)
                    pcd_integrate_colors = np.concatenate([current_color, prev_color], axis=0)
                    pcd_integrate.points = o3d.utility.Vector3dVector(pcd_integrate_points)
                    pcd_integrate.colors = o3d.utility.Vector3dVector(pcd_integrate_colors / pcd_integrate_colors.max())
                    #vis.add_geometry(pcd_integrate)
                else:
                    pcd_integrate_points = np.concatenate([pcd_integrate_points, current_original_pcd], axis=0)
                    pcd_integrate_colors = np.concatenate([pcd_integrate_colors, current_color], axis=0)
                    pcd_integrate.points = o3d.utility.Vector3dVector(pcd_integrate_points)
                    pcd_integrate.colors = o3d.utility.Vector3dVector(pcd_integrate_colors/pcd_integrate_colors.max())

                    # vis.update_geometry()
                    # vis.poll_events()
                    # vis.update_renderer()

    o3d.visualization.draw_geometries([pcd_integrate])

        #exit(-1)

    # pcd_1, original_point_1, original_color_1, point_1, color_1, intrinsic_1, rgb_1 = test_simple(image_1)  # N x 3 pcd
    # pcd_2, original_point_2, original_color_2, point_2, color_2, intrinsic_2, rgb_2 = test_simple(image_2)  # N x 3 pcd
    #
    #
    # norm_grey_1 = NormalizeRGBImage(rgb_1.copy())
    # norm_grey_2 = NormalizeRGBImage(rgb_2.copy())
    #
    # # Do Optimization...
    #
    # optimizedPose = gauss_newton(pcd= point_1.copy(), image_prev=norm_grey_1, image_current=norm_grey_2, intrinsic=intrinsic_1)
    # #print(optimizedPose)
    #
    # # Transformation
    # original_point_2 = TransformPCD(original_point_2.T, rot=SO3.exp(np.reshape(-1 * optimizedPose[3:], (3,))).mat, trans=-1 * optimizedPose[:3, :])
    # original_point_2 = original_point_2.T
    #
    # # After Optimization...
    #
    #
    # pcd_integrate_points = np.concatenate([original_point_1, original_point_2], axis=0)
    # pcd_integrate_colors = np.concatenate([original_color_1, original_color_2], axis=0)
    #
    # pcd_integrate = o3d.geometry.PointCloud()
    # pcd_integrate.points = o3d.utility.Vector3dVector(pcd_integrate_points)
    # pcd_integrate.colors = o3d.utility.Vector3dVector(pcd_integrate_colors/pcd_integrate_colors.max())
    #
    # o3d.visualization.draw_geometries([pcd_integrate])