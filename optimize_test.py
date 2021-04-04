import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule
import time
from liegroups import SO3
from pcdUtil import TransformPCD, ProjectPCD, computeResidual, computeImgJacobian, computePoseProjectionJacobian
from pcdUtil import computeImgGradientMap, residualSum, GPUarrays
from GPUpcdUtil import transferToGPU, GPUposeUpdate
from GPUFunc import getComputeImgJacobianFunc, getComputeResidualFunc, getcomputePoseProjectionFunc, getGpuPoseMatMulFunc, getGpuProjectionFunc
from GPUFunc import getGpuLargeMediumMatMulFunc

import cv2


def gauss_newton(prev_pcd_pyramid, current_image_pyramid, prev_image_pyramid, intrinsic_pyramid, num_level,
                 computeImgJacobianFunc, computeResidualFunc, computePoseProcjectionJacobianFunc, GpuPoseMatMulFunc, GPUProjectionFunc
                 ):

    # TODO : 1) image pyramid optimization ( downsample origianl rgb and depth )
    # TODO : 2) Jacobian weighting function
    # TODO : 3) point to plane icp ( depth element optimization ) -> t-distribution
    '''
    pcd: Pointcloud of prev frame

    image_prev: Image of previous frame
    image_current: Image of next frame
    tolerance: convergence tolerance

    image_prev <-> pcd!
    image_current <-> transformed pcd!

    want to know ! R & t that satisfy ->   reference_frame = R * target_frame + t  = T * target_frame that maximize photo-consistency!
    '''

    st = time.time()
    prev_image_coord_gpu_list, current_image_coord_gpu_list, residual_gpu_list, validIdx_gpu_list, \
    grad_x_gpu_list, grad_y_gpu_list, \
    img_jacobian_gpu_list, \
    jacobian_gpu_list, jacobian_t_gpu_list, \
    current_image_gpu_list, prev_image_gpu_list, \
    pcd_gpu_list, tr_pcd_gpu_list, \
    size_gpu_list, \
    intrinsic_gpu_list = GPUarrays(pcd_pyramid=prev_pcd_pyramid, current_image_pyramid=current_image_pyramid, prev_image_pyramid=prev_image_pyramid,
                                   intrinsic_pyramid=intrinsic_pyramid, num_level=num_level)

    error_prev = 1e6
    elapsed_time = 0
    pose = np.zeros(shape=(6, 1), dtype=np.float32)  # 6 x 1 column vector. first 3 -> translation , last 3 -> rotation

    iteration = 20
    #print(time.time()-st)

    for level in range(num_level-1, -1, -1):
        #print("level:", level)
        Jacobian = np.zeros(shape=(prev_pcd_pyramid[level].shape[1], 6), dtype=np.float32)
        residual = np.zeros(shape=(prev_pcd_pyramid[level].shape[1], 1), dtype=np.float32)
        for i in range(iteration):

            TransformPCD(func=GpuPoseMatMulFunc,
                         rot=SO3.exp(np.reshape(pose[3:], (3,))).mat, trans=pose[:3, :],
                         pcd_gpu=pcd_gpu_list[level], tr_pcd_gpu=tr_pcd_gpu_list[level],
                         row=3, medium=3, col=prev_pcd_pyramid[level].shape[1],
                          mode="GPU")


            # PROJECT trasformed pcd into image coordinate(current_image_coord_gpu)
            ProjectPCD(func=GPUProjectionFunc, pcd_gpu=tr_pcd_gpu_list[level],
                       current_image_coord_gpu=current_image_coord_gpu_list[level], validIdx_gpu=validIdx_gpu_list[level],
                       intrinsic_gpu=intrinsic_gpu_list[level], size_gpu=size_gpu_list[level], mode="GPU")


            computeResidual(func=computeResidualFunc,
                            residual_gpu=residual_gpu_list[level],
                            prev_image_gpu=prev_image_gpu_list[level], current_image_gpu=current_image_gpu_list[level],
                            prev_image_coord_gpu=prev_image_coord_gpu_list[level], current_image_coord_gpu=current_image_coord_gpu_list[level],
                            validIdx_gpu=validIdx_gpu_list[level], size_gpu=size_gpu_list[level],
                            mode="GPU")


            # Calculate Image intensity gradient of current image frame.
            # Jacobian = (Jacobian of Image intensity) * (Jacobian of Projection image coordinate) * (Jacobian of 3d point to 6 dof pose)

            #st = time.time()
            computeImgJacobian(func=computeImgJacobianFunc,
                                          img_jacobian_gpu=img_jacobian_gpu_list[level],
                                          grad_x_gpu=grad_x_gpu_list[level], grad_y_gpu=grad_y_gpu_list[level],
                                          current_image_coord_gpu=current_image_coord_gpu_list[level],
                                          validIdx_gpu=validIdx_gpu_list[level], size_gpu=size_gpu_list[level],
                                          mode="GPU") # (N x 2 )

            #print(time.time() - st)
            # 0.005sec

            #st = time.time()
            computePoseProjectionJacobian(func=computePoseProcjectionJacobianFunc,
                                                     jacobian_gpu=jacobian_gpu_list[level], jacobian_t_gpu=jacobian_t_gpu_list[level],
                                                    img_jacobian_gpu=img_jacobian_gpu_list[level],
                                                     validIdx_gpu=validIdx_gpu_list[level],
                                                     transformed_pcd_gpu=tr_pcd_gpu_list[level], pcd_gpu=pcd_gpu_list[level],
                                                     intrinsic_gpu=intrinsic_gpu_list[level], size_gpu=size_gpu_list[level],
                                                     mode="GPU") # ( N x 6 ) Jacobian

            cuda.memcpy_dtoh(residual, residual_gpu_list[level])
            cuda.memcpy_dtoh(Jacobian, jacobian_gpu_list[level])

            pose = pose - np.linalg.inv(Jacobian.T @ Jacobian) @ Jacobian.T @ residual

            # Gpu version of Pose update.. but slower than CPU version now due to memory copy delay.. Need some optimization.
            # pose = GPUposeUpdate(function=GpuLargeMediumMatmulFunc,
            #                      pose=pose,
            #                      jacobian_gpu=jacobian_gpu, jacobian_t_gpu=jacobian_t_gpu,
            #                      residual_gpu=residual_gpu,
            #                      intermediate_jacobian_residual_gpu=intermediate_jacobian_residual_gpu, intermediate_square_jacobian_gpu=intermediate_square_jacobian_gpu,
            #                      square_jacobian_gpu=square_jacobian_gpu, jacobian_residual_gpu=jacobian_residual_gpu,
            #                      square_size_gpu=square_size_gpu, jac_size_gpu=jac_size_gpu,
            #                      pcd_size=pcd.shape[1])


            error = residualSum(residual)

            converge_diff = error - error_prev

            #print("error: {}, w_x: {}, w_y: {}, w_z: {}, t_x: {}, t_y : {}, t_z : {}, Converge diff: {}".format(error, pose[3], pose[4], pose[5], pose[0], pose[1], pose[2] , converge_diff))
            if abs(converge_diff) < 0.1:
                break
            error_prev = error

    #print("Elapsed time:", time.time()-st, "  ", iter_c)
    #print(time.time() - st)
    return pose

