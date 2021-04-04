from __future__ import absolute_import, division, print_function
import open3d as o3d
import sys
import os
import argparse
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from pcdUtil import ExtractPCD, scaleIntrinsic, NormalizeRGBImage
from optimize_test import gauss_newton
import time
from liegroups import SO3
from pcdUtil import downsampleDepth
import networks
from layers import disp_to_depth
from pcdUtil import ExtractPCD, scaleIntrinsic
from multiprocessing import Process, Queue, Lock
from threading import Thread
import threading
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
import signal
from pcdUtil import TransformPCD, returnPoses, getSparsedPcd
import psutil

from GPUFunc import getComputeImgJacobianFunc, getComputeResidualFunc, getcomputePoseProjectionFunc, getGpuPoseMatMulFunc, getGpuProjectionFunc
from GPUFunc import getGpuLargeMediumMatMulFunc

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        default="mono+stereo_1024x320",
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

global_encoder = None
global_depth_decoder = None
depth_pub = rospy.Publisher("/Real_Scale_Depth", Image, queue_size=100)
br = CvBridge()
encoder_path = None
loaded_dict_enc = None

keyframe_queue = Queue()
mapping_queue = Queue()
Pose_queue = Queue()
main_pid = os.getpid()
map_pid = None


intrinsic = np.loadtxt("camera_intrinsic.txt", dtype=np.float64)
feed_width, feed_height = 1024, 320
input_width, input_height = 1241, 376

# Scale intrinsic parameter according to scaling factor.
intrinsic = scaleIntrinsic(intrinsic=intrinsic, xscaleRatio=feed_width / input_width,
                                       yscaleRatio=feed_height / input_height)
n_frame = 0

def DepthEstimation(msg, num_level):
    global global_encoder, global_depth_decoder, depth_pub, encoder_path, loaded_dict_enc, intrinsic
    global keyframe_queue, mapping_queue, n_frame
    # Get keyframe every 3 frame.
    #print("n_frame:", n_frame)

    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1) # [height, width, channel(3)]
    if image is None:
        return
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
         # LOADING PRETRAINED MODEL
        # Load image and preprocess
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb, dsize=(feed_width, feed_height))
        rgb_original = rgb.copy()

        rgb_flatten = np.reshape(rgb.copy(), newshape=(-1, 3))

        input_image = pil.fromarray(rgb).convert("RGB")
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

        pcd_pyramid = []
        rgb_pyramid = []
        intrinsic_pyramid = []

        # original data
        pcd = ExtractPCD(depth, intrinsic)
        depth_pyr = depth.copy()
        intrinsic_pyr = intrinsic.copy()
        rgb_pyr = rgb_original.copy()

        # pyramid data
        for idx in range(0, num_level+1):

            # downsample depth and adjust intrinsic
            intrinsic_pyr = intrinsic_pyr / 2
            depth_pyr = downsampleDepth(depth_pyr)
            rgb_pyr = cv2.pyrDown(rgb_pyr)
            pcd_pyr = ExtractPCD(depth_pyr, intrinsic_pyr)

            pcd_pyr_cut = pcd_pyr[:, :]
            rgb_pyr_cut = rgb_pyr[:, :]
            if idx is 0: continue

            pcd_pyramid.append(pcd_pyr_cut)
            rgb_pyramid.append(NormalizeRGBImage(rgb_pyr_cut))
            intrinsic_pyramid.append(intrinsic_pyr)

        keyframe_queue.put(np.array([pcd_pyramid, rgb_pyramid, intrinsic_pyramid]))

        if n_frame % 3 == 0:
            n_frame += 1
            mapping_queue.put(getSparsedPcd(pcd.astype(np.float64), rgb_flatten.astype(np.float64)))
            return
        n_frame += 1

def Tracking(keyframe_queue, Pose_queue,
             computeImgJacobianFunc, computeResidualFunc, computePoseProjectionJacobianFunc, GpuPoseMatMulFunc, GPUProjectionFunc):
    keyframe_dict = {}
    Posegraph = o3d.registration.PoseGraph()
    current_pose = np.eye(4)

    current_frame = 0
    convergenceCriteria = o3d.registration.GlobalOptimizationConvergenceCriteria()
    convergenceCriteria.max_iteration = 20
    convergenceCriteria.max_iteration_lm = 20
    optimOption = o3d.registration.GlobalOptimizationOption()
    optimOption.reference_node = 0
    PGO_LM = o3d.registration.GlobalOptimizationLevenbergMarquardt()


    while True:
        if not keyframe_queue.empty():
            # print("current working keyframe:", current_frame)
            tmp_keyframe = keyframe_queue.get()
            if current_frame == 0:
                keyframe_dict[0] = tmp_keyframe
                Posegraph.nodes.append(o3d.registration.PoseGraphNode(pose=current_pose))
                Pose_queue.put(returnPoses(Posegraph=Posegraph))
            else:

                keyframe_dict[current_frame] = tmp_keyframe
                prev_pcd_pyramid, prev_rgb_pyramid, prev_intrinsic_pyramid = keyframe_dict[current_frame - 1]
                current_pcd_pyramid, current_rgb_pyramid, current_intrinsic_pyramid = keyframe_dict[current_frame]

                # Eliminate pyramid data for memory save
                keyframe_dict[current_frame-1] = None

                st = time.time()
                optimizedPose = gauss_newton(prev_pcd_pyramid=prev_pcd_pyramid,
                                             current_image_pyramid=current_rgb_pyramid,
                                             prev_image_pyramid=prev_rgb_pyramid,
                                             intrinsic_pyramid=current_intrinsic_pyramid, num_level=3,
                                             computeImgJacobianFunc=computeImgJacobianFunc,
                                             computeResidualFunc=computeResidualFunc,
                                             computePoseProcjectionJacobianFunc=computePoseProjectionJacobianFunc,
                                             GpuPoseMatMulFunc=GpuPoseMatMulFunc,
                                             GPUProjectionFunc=GPUProjectionFunc)
                #print("Elapsed time:", time.time()-st)

                transformationMatrix = np.eye(4)
                transformationMatrix[:3, :3] = SO3.exp(np.reshape(optimizedPose[3:, 0], (3,))).mat
                transformationMatrix[:3, 3] = optimizedPose[:3, 0] # transformation matrix that transform current frame to previous frame.

                # Update current pose by regular frame
                current_pose = current_pose @ np.linalg.inv(transformationMatrix)

                # if current frame is keyframe, append current global pose into Pose graph
                if current_frame % 3 == 0:
                    #print("Pose graph update!!!")
                    Posegraph.nodes.append(o3d.registration.PoseGraphNode(pose=current_pose))
                    Posegraph.edges.append(o3d.registration.PoseGraphEdge(
                    source_node_id=int(current_frame/3)-1, target_node_id=int(current_frame/3),
                    transformation=np.linalg.inv(Posegraph.nodes[int(current_frame/3)-1].pose) @ Posegraph.nodes[int(current_frame/3)].pose))

                    #PGO_LM.OptimizePoseGraph(pose_graph=Posegraph, criteria=convergenceCriteria, option=optimOption)
                    Pose_queue.put(returnPoses(Posegraph=Posegraph))

            current_frame += 1
        else:
            continue

def Mapping(mapping_queue, Pose_queue):

    def exitProgram(vis):
        global main_pid, map_pid
        vis.destroy_window()
        os.kill(main_pid, signal.SIGTERM)
        os.kill(map_pid, signal.SIGTERM)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    view_point_set = False
    vis.register_key_callback(ord('K'), exitProgram)
    vis.create_window()
    pcd_o3d = o3d.geometry.PointCloud()

    mapping_pcd = {}
    mapped_idx = []
    map_c = 0
    Poses = None


    while True:
        if not Pose_queue.empty():
            Poses = Pose_queue.get()
        if not mapping_queue.empty():
            mapping_pcd[len(mapping_pcd.keys())] = mapping_queue.get()
            #print(mapping_pcd[len(mapping_pcd.keys())-1][0].shape)

        if Poses is not None:
            for idx in range(len(Poses)):
                if idx not in mapped_idx and idx in mapping_pcd.keys():
                    item = mapping_pcd[idx]
                    transformed_pcd = TransformPCD(mode="CPU", pcd=item[0].T, rot=Poses[idx][:3, :3], trans=np.reshape(Poses[idx][:3, 3], (3, 1))).T
                    mapped_idx.append(idx)

                    if map_c == 0:
                        pcd_o3d.points = o3d.utility.Vector3dVector(transformed_pcd)
                        pcd_o3d.colors = o3d.utility.Vector3dVector(item[1])
                        map_c += 1
                        vis.add_geometry(pcd_o3d)
                        if view_point_set:
                            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                            cam.extrinsic = Poses[idx]
                            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                    else:
                        pcd_o3d.points.extend(o3d.utility.Vector3dVector(transformed_pcd))
                        pcd_o3d.colors.extend(o3d.utility.Vector3dVector(item[1]))
                        map_c += 1
                        if view_point_set:
                            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                            cam.extrinsic = np.linalg.inv(Poses[idx])
                            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)


        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()


if __name__ == '__main__':
    # global global_encoder, global_depth_decoder, depth_pub, encoder_path, loaded_dict_enc, feed_width, feed_height,
    # global keyframe_queue, mapping_queue , Posegraph_queue
    args = parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        print("CUDA Available is checked!")
        print(device)
    else:
        device = torch.device("cpu")



    ######################################### MODEL INIT ######################################################################
    model_path = os.path.join("models", args.model_name)

    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    global_encoder = encoder
    global_depth_decoder = depth_decoder

    ###############################################################################################

    rospy.init_node(name="DepthPred")
    sub = rospy.Subscriber("image", Image, queue_size=100, callback=DepthEstimation, callback_args=3)

    # Start Mapping Process// should Use multiprocess for non blocking SLAM.
    MappingProc = Process(target=Mapping, args=(mapping_queue, Pose_queue))
    map_pid = MappingProc.pid
    MappingProc.start()

    computeImgJacobianFunc = getComputeImgJacobianFunc()
    computeResidualFunc = getComputeResidualFunc()
    computePoseProjectionJacobianFunc = getcomputePoseProjectionFunc()
    GpuPoseMatMulFunc = getGpuPoseMatMulFunc()
    GPUProjectionFunc = getGpuProjectionFunc()

    # Start Tracking Process
    Tracking(keyframe_queue,
             Pose_queue,
             computeImgJacobianFunc, computeResidualFunc, computePoseProjectionJacobianFunc, GpuPoseMatMulFunc, GPUProjectionFunc)



