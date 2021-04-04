import numpy as np
import torch
import struct
import sys
import open3d as o3d
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from GPUpcdUtil import GPUcomputeResidual, GPUcomputeImgJacobian, GPUcomputePoseProjectionJacobian, GPUmatmul, GPUProjection, transferToGPU
import time

def returnPoses(Posegraph):
    poses = []
    for i in range(len(Posegraph.nodes)):
        poses.append(Posegraph.nodes[i].pose)
    return poses
def getSparsedPcd(pcd, rgb):
    #st = time.time()
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb/rgb.max())
    #pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=0.1)
    pcd_o3d = pcd_o3d.uniform_down_sample(45)

    return np.array([np.array(pcd_o3d.points), np.array(pcd_o3d.colors)])


def scaleIntrinsic(intrinsic, xscaleRatio, yscaleRatio):
    scaled_intrinsic = intrinsic.copy()
    scaled_intrinsic[0][0] = intrinsic[0][0] * xscaleRatio
    scaled_intrinsic[0][2] = (intrinsic[0][2] + 0.5) * xscaleRatio - 0.5

    scaled_intrinsic[1][1] = intrinsic[1][1] * yscaleRatio
    scaled_intrinsic[1][2] = (intrinsic[1][2] + 0.5) * yscaleRatio - 0.5

    return scaled_intrinsic

def ProjectPCD(func=None, pcd=None, current_image_coord_gpu=None, intrinsic=None, pcd_gpu=None, validIdx_gpu=None, intrinsic_gpu=None, size_gpu=None, mode="GPU"):

    '''
    Arguments

    pcd: Pointcloud (shape : 3 x N )
    intrinsic : Camera intrinsic

    return : image coord (2 x N) u, v
    '''
    if mode is "CPU":
        pcd_copy = pcd.copy()
        fX, cX, fY, cY = float(intrinsic[0][0]), float(intrinsic[0][2]), float(intrinsic[1][1]), float(intrinsic[1][2])
        image_normal_coord = pcd_copy[:2, :]
        image_normal_coord[0, :] /= pcd_copy[2, :]
        image_normal_coord[1, :] /= pcd_copy[2, :]

        image_coord = image_normal_coord.copy()
        image_coord[0, :] = image_coord[0, :] * fX + cX
        image_coord[1, :] = image_coord[1, :] * fY + cY
        image_coord = image_coord.astype(np.float32)
        return image_coord
    elif mode is "GPU":
        GPUProjection(func=func, pcd_gpu=pcd_gpu, current_image_coord_gpu=current_image_coord_gpu, validIdx_gpu=validIdx_gpu, intrinsic_gpu=intrinsic_gpu, size_gpu=size_gpu)




def ExtractPCD(depth, intrinsic):

    '''
    Arguments

    depth: real metric depth image (h, w)
    intrinsic : camera intrinsic matrix 4x4
    return Nx3 shape pcd
    '''
    h, w = depth.shape[0], depth.shape[1]
    fX, cX, fY, cY = float(intrinsic[0][0]), float(intrinsic[0][2]), float(intrinsic[1][1]), float(intrinsic[1][2])

    flatten_depth = np.reshape(depth, newshape=(-1, depth.shape[0] * depth.shape[1]))

    x, y = np.meshgrid(range(w), range(h))

    x = np.reshape(x, newshape=(-1, x.shape[0] * x.shape[1])).astype(np.float32)
    y = np.reshape(y, newshape=(-1, y.shape[0] * y.shape[1])).astype(np.float32)

    # From pixel plane to normal pixel plane
    x = (x-cX)/fX
    y = (y-cY)/fY

    # Add z=1 coordinate to x-y normal plane
    pix_coord = np.concatenate([x, y], axis=0)
    ones = np.ones(shape=(1, x.shape[1]))
    pix_coord = np.concatenate([pix_coord, ones], axis=0)

    # Adjust Depth value ( Normal plane to real 3d point)
    camera_coord = pix_coord * flatten_depth

    #camera_coord = camera_coord[:, np.array(np.where(camera_coord[2, :] < 80))[0]]
    #print(camera_coord.shape)
    camera_coord = camera_coord.T

    return camera_coord

def TransformPCD(func=None, pcd=None, rot=None, trans=None, pcd_gpu=None, tr_pcd_gpu=None, row=None, medium=None, col=None, mode="GPU"):
    '''
    pcd : numpy pointcloud matrix (shape : 3 x N)
    rot : rotation matrix ( shape : 3 x 3 )
    trans : translation vector ( shape : 3 x 1 )

    V' = RV + t ( V' : transformed point cloud , V: current point cloud)

    return new pcd (shape : 3 x N )
    '''

    if mode is "CPU":
        if pcd is None:
            print("pcd Should be given")
            exit(-1)
        tmp = pcd.copy()
        tr_pcd = rot @ tmp + trans
        return tr_pcd

    elif mode is "GPU":
        GPUmatmul(func, rot.astype(np.float32), trans.astype(np.float32), pcd_gpu, tr_pcd_gpu, row, medium, col)

def computeImgGradientMap(image_gray):
    w, h = image_gray.shape[1], image_gray.shape[0]
    new_image_gray = np.zeros(shape=(h + 2, w + 2))
    new_image_gray[1:h+1, 1:w+1] = image_gray

    # copy border values of original image 4 sides.(left, right, top ,bottom)
    new_image_gray[0, 1:w+1] = image_gray[0, :]
    new_image_gray[h + 1, 1:w+1] = image_gray[h - 1, :]
    new_image_gray[1:h+1, 0] = image_gray[:, 0]
    new_image_gray[1:h+1, w + 1] = image_gray[:, w - 1]

    grad_x = new_image_gray[1:h+1, 2:] - new_image_gray[1:h+1, 0:w]
    grad_y = new_image_gray[2:, 1:w+1] - new_image_gray[0:h, 1:w+1]

    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def computeImgJacobian(func=None, grad_x=None, grad_y=None, current_image_coord=None, img_jacobian_gpu=None, grad_x_gpu=None, grad_y_gpu=None, current_image_coord_gpu=None, validIdx_gpu=None, size_gpu=None, mode="GPU"):
    '''
    Args
    grad_x : gradient map along x axis ( h x w )
    grad_y : gradient map along y axis ( h x w )
    image_coord : image coordinate (u, v) (2 x N)
    validIdx : valid point index(valid N, )

    return: ImgJacobian ( N x 2 )
    '''

    if mode is "CPU":
        validIdx =  getValidIdxCPU(current_image_coord, grad_x.shape[1], grad_x.shape[0])
        N = validIdx.shape[0]

        j_img = np.zeros(shape=(N, 2), dtype=np.float32)

        for i, idx in enumerate(validIdx):
            x, y = current_image_coord[0, int(idx)] , current_image_coord[1, int(idx)]
            j_img[i, 0] = compute_bilinear(grad_x, x, y)  # gradient along axis X
            j_img[i, 1] = compute_bilinear(grad_y, x, y)  # gradient along axis Y

        return j_img

    elif mode is "GPU":
        blockSize = (25, 35, 1)
        gridSize = (35, 35, 1)
        GPUcomputeImgJacobian(func, img_jacobian_gpu, grad_x_gpu, grad_y_gpu, current_image_coord_gpu, validIdx_gpu, size_gpu, blockSize, gridSize)


def computeProjectionJacobian(fX, fY, tr_X, tr_Y, tr_Z):
    """
    Args
    f - > focal length
    tr_X, tr_Y, tr_Z -> 3D point After transformation

    return Projection Jacobian (2 X 3)
    """
    return np.reshape(np.asarray([[fX / tr_Z, 0, -fX * tr_X / (tr_Z * tr_Z)], [0, fY / tr_Z, -fY * tr_Y / (tr_Z * tr_Z)]]), (2, 3))

def computePoseJacobian(X, Y, Z):
    """
    Args
    X, Y, Z -> 3D point before transformation

    Use Approximation of Jacobian of Rotation matrix  in Multi View Geometry

    R = I + [t]

    return Pose Jacobian (3 X 6)

    """
    poseJac = np.concatenate([np.eye(3), np.array([[0, Z, -Y], [-Z, 0, X] , [Y, -X, 0]])], axis=1) # Shape: 3 X 6

    return poseJac


def compute_bilinear(value_map, u, v):
    if u == int(u) and v == int(v):
        return value_map[int(v), int(u)]

    # Else -> u or v is not integer but float
    # Bilinear Interpolation

    width ,height = value_map.shape[1], value_map.shape[0]

    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = u0 + 1, v0 + 1

    u1_weight = u - u0
    v1_weight = v - v0
    u0_weight = 1 - u1_weight
    v0_weight = 1 - v1_weight

    if u0 < 0 or u0 >= width: u0_weight = 0
    if u1 < 0 or u1 >= width: u1_weight = 0
    if v0 < 0 or v0 >= height: v0_weight = 0
    if v1 < 0 or v1 >= height: v1_weight = 0


    w00 = u0_weight * v0_weight
    w10 = u1_weight * v0_weight
    w01 = u0_weight * v1_weight
    w11 = u1_weight * v1_weight

    sum_weights = w00 + w10 + w01 + w11
    total = 0

    if w00 > 0: total += value_map[v0, u0] * w00
    if w10 > 0: total += value_map[v0, u1] * w10
    if w01 > 0: total += value_map[v1, u0] * w01
    if w11 > 0: total += value_map[v1, u1] * w11

    return total / sum_weights

def getValidIdxCPU(image_coord, width, height):
    validBool = (image_coord[0, :] > 0) * (image_coord[0, :] < width) * (image_coord[1, :] > 0) * (image_coord[1, :] < height)
    validIdx = np.where(image_coord[validBool]).astype(np.float32)

    return validIdx

def computeResidual(func=None, width=None, height=None, image_prev_gray=None, image_current_gray=None, residual_gpu=None, prev_image_gpu=None, prev_image_coord=None, current_image_coord=None, current_image_gpu=None, prev_image_coord_gpu=None, current_image_coord_gpu=None, validIdx_gpu=None, size_gpu=None, mode="GPU"):


    # Already points out of boundary is filtered out.

    #print("coord:", current_image_coord.shape, "valid:", validIdx.shape)



    if mode is "CPU":
        validIdx = getValidIdxCPU(image_coord=current_image_coord, width=width, height=height)
        residual = np.zeros(shape=(validIdx.shape[0], 1),
                            dtype=np.float32)  # shape : N x 1 column vector , np.float32 date type.
        #------- CPU MODE --------#
        for i, idx in enumerate(validIdx):
            idx = int(idx)
            t_u, t_v, u, v = current_image_coord[0, idx], current_image_coord[1, idx], prev_image_coord[0, idx], prev_image_coord[1, idx]

            # residual - > I_1(u, v) - I_2(t_u, t_v)
            residual[i][0] = compute_bilinear(image_prev_gray, u=u, v=v) - compute_bilinear(image_current_gray, u=t_u, v=t_v)
        return residual

    elif mode is "GPU":
        #------- GPU MODE --------#

        blockSize = (25, 35, 1)
        gridSize = (25, 25, 1)

        GPUcomputeResidual(function=func,
                           residual_gpu=residual_gpu,
                           current_image_coord_gpu=current_image_coord_gpu, prev_image_coord_gpu=prev_image_coord_gpu,
                           current_image_gpu=current_image_gpu, prev_image_gpu=prev_image_gpu,
                           validIdx_gpu=validIdx_gpu, size_gpu=size_gpu,
                           block=blockSize, grid=gridSize)



def computePoseProjectionJacobian(func=None, img_jacobian=None, w=None, h=None, current_image_coord=None, transformed_pcd=None, pcd=None, intrinsic=None, jacobian_gpu=None, jacobian_t_gpu=None, img_jacobian_gpu=None, validIdx_gpu=None, transformed_pcd_gpu=None, pcd_gpu=None, intrinsic_gpu=None, size_gpu=None, mode="GPU"):


    if mode == "CPU":
        fX, cX, fY, cY = float(intrinsic[0][0]), float(intrinsic[0][2]), float(intrinsic[1][1]), float(intrinsic[1][2])
        validIdx = getValidIdxCPU(current_image_coord, width=w, height=h)
        Jacobian = np.zeros(shape=(validIdx.shape[0], 6))  # initialize total jacobian shape (N x 6)
        for i, idx in enumerate(validIdx):
            JacImg = np.reshape(img_jacobian[i, :], newshape=(1, 2))  # shape : 1 x 2
            idx = int(idx)
            JacProj = computeProjectionJacobian(fX=fX, fY=fY, tr_X=transformed_pcd[0, idx], tr_Y=transformed_pcd[1, idx],
                                                tr_Z=transformed_pcd[2, idx])  # 2x3
            JacPose = computePoseJacobian(X=pcd[0, idx], Y=pcd[1, idx], Z=pcd[2, idx])  # 3 x 6
            Jacobian[i, :] = -1 * np.reshape(JacImg @ JacProj @ JacPose, newshape=(6,))
        return Jacobian
    elif mode == "GPU":
        blockSize = (25, 35, 1)
        gridSize = (35, 35, 1)
        return GPUcomputePoseProjectionJacobian(function=func,
                                                jacobian_gpu=jacobian_gpu, jacobian_t_gpu=jacobian_t_gpu,
                                                img_jacobian_gpu=img_jacobian_gpu,
                                                validIdx_gpu=validIdx_gpu, intrinsic_gpu=intrinsic_gpu,
                                                transformed_pcd_gpu=transformed_pcd_gpu, pcd_gpu=pcd_gpu,
                                                size_gpu=size_gpu, block=blockSize, grid=gridSize)


def residualSum(residuals):

    return np.sum(np.abs(residuals), axis=0)

def checkInRange(width, height, u, v):
    if u >= 0 and u < width and v >= 0 and v < height: return True
    return False


def NormalizeRGBImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.normalize(image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)


def downsampleDepth(depth):
    return (depth[0::2, 0::2] + depth[1::2, 0::2] + depth[0::2, 1::2] + depth[1::2, 1::2]) / 4






def GPUarrays(pcd_pyramid, current_image_pyramid, prev_image_pyramid, intrinsic_pyramid, num_level):
    prev_image_coord_gpu_list, current_image_coord_gpu_list, residual_gpu_list, validIdx_gpu_list = [], [], [], []
    grad_x_gpu_list, grad_y_gpu_list = [], []
    img_jacobian_gpu_list = []
    jacobian_gpu_list, jacobian_t_gpu_list = [], []
    # square_jacobian_gpu_list, intermediate_square_jacobian_gpu_list = [], []
    # jacobian_residual_gpu_list, intermediate_jacobian_residual_gpu_list = [], []
    current_image_gpu_list, prev_image_gpu_list = [], []
    pcd_gpu_list, tr_pcd_gpu_list = [], []
    size_gpu_list = []
    # square_size_gpu_list, jac_size_gpu_list = [], []
    intrinsic_gpu_list = []
    for idx in range(0, num_level):
        #print("idx:", idx)
        pcd_pyramid[idx] = pcd_pyramid[idx].transpose((1, 0)).copy()
        image_current, image_prev = current_image_pyramid[idx], prev_image_pyramid[idx]
        intrinsic = intrinsic_pyramid[idx]

        prev_image_coord = ProjectPCD(pcd=pcd_pyramid[idx], intrinsic=intrinsic_pyramid[idx], mode="CPU")

        # Send tr, src image_coord from CPU to GPU
        prev_image_coord_gpu_list.append(transferToGPU(prev_image_coord))

        current_image_coord_gpu_list.append(transferToGPU(np.empty_like(prev_image_coord).astype(np.float32)))

        # Define Residual array in GPU
        residual_gpu_list.append(transferToGPU(np.zeros(shape=(pcd_pyramid[idx].shape[1], 1), dtype=np.float32)))

        # Define validIdx array in GPU
        validIdx_gpu_list.append(transferToGPU(np.zeros(shape=(1, pcd_pyramid[idx].shape[1]), dtype=np.float32)))

        # Send grad_x, grad_y from CPU to GPU
        grad_x, grad_y = computeImgGradientMap(image_current)
        grad_x_gpu_list.append(transferToGPU(grad_x))
        grad_y_gpu_list.append(transferToGPU(grad_y))

        # Define ImgJacobian in GPU
        img_jacobian_gpu_list.append(transferToGPU(np.zeros(shape=(pcd_pyramid[idx].shape[1], 2), dtype=np.float32)))

        # Define Final Jacobian and Jacobian transpose in GPU
        jacobian_gpu_list.append(transferToGPU(np.zeros(shape=(pcd_pyramid[idx].shape[1], 6), dtype=np.float32)))
        jacobian_t_gpu_list.append(transferToGPU(np.zeros(shape=(6, pcd_pyramid[idx].shape[1]), dtype=np.float32)))

        # Define J_T * J (6x6) , J_T * residual in GPU for faster update of pose parameter
        # square_jacobian_gpu_list.append(transferToGPU(np.zeros(shape=(6, 6), dtype=np.float32)))
        # intermediate_square_jacobian_gpu_list.append(transferToGPU(np.zeros(shape=(pcd_pyramid[idx].shape[1], 6, 6), dtype=np.float32)))

        # jacobian_residual_gpu_list.append(transferToGPU(np.zeros(shape=(6, 1), dtype=np.float32)))
        # intermediate_jacobian_residual_gpu_list.append(transferToGPU(np.zeros(shape=(pcd_pyramid[idx].shape[1], 6, 1), dtype=np.float32)))

        # Send current, prev image from CPU to GPU
        current_image_gpu_list.append(transferToGPU(image_current.astype(np.float32)))
        prev_image_gpu_list.append(transferToGPU(image_prev.astype(np.float32)))

        # Send tr, src Pointcloud value from CPU to GPu
        pcd_gpu_list.append(transferToGPU(pcd_pyramid[idx].astype(np.float32)))
        tr_pcd_gpu_list.append(transferToGPU(np.empty_like(pcd_pyramid[idx], dtype=np.float32)))

        # Send dimension parameters from CPU to GPU

        size_gpu_list.append(transferToGPU(np.array([[pcd_pyramid[idx].shape[1], image_current.shape[1], image_current.shape[0]]],dtype=np.float32)))
        # (Number of pcd points), width, height)

        # Send square ,jac size dimension parameters from CPU to GPU
        # [row , initial reduction, col, medium]
        # square_size_gpu_list.append(transferToGPU(np.array([[6, pcd_pyramid[idx].shape[1], 6, pcd_pyramid[idx].shape[1]]], dtype=np.float32)))
        #
        # # [row , initial reduction, col, medium]
        # jac_size_gpu_list.append(transferToGPU(np.array([[6, pcd_pyramid[idx].shape[1], 1, pcd_pyramid[idx].shape[1]]], dtype=np.float32)))

        # Send inrinsic parameters from CPU to GPU
        intrinsic_gpu_list.append(transferToGPU(np.array([intrinsic[0][0], intrinsic[0][2], intrinsic[1][1], intrinsic[1][2]],
                                               dtype=np.float32)))  # fx, cx, fy, cy

    return prev_image_coord_gpu_list, current_image_coord_gpu_list, residual_gpu_list, validIdx_gpu_list, \
           grad_x_gpu_list, grad_y_gpu_list, \
           img_jacobian_gpu_list, \
           jacobian_gpu_list, jacobian_t_gpu_list, \
           current_image_gpu_list, prev_image_gpu_list, \
           pcd_gpu_list, tr_pcd_gpu_list, \
           size_gpu_list, \
           intrinsic_gpu_list
