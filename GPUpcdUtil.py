import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

def transferToGPU(array):
    gpu = cuda.mem_alloc(array.nbytes)
    cuda.memcpy_htod(gpu, array)
    return gpu

def GPUProjection(func, pcd_gpu, current_image_coord_gpu, validIdx_gpu, intrinsic_gpu, size_gpu):
    func(pcd_gpu, current_image_coord_gpu, validIdx_gpu, intrinsic_gpu, size_gpu, block=(25, 25, 1), grid=(5, 5, 15))

def reduction(function, array_size, intermediate_mat_gpu, save_mat_gpu, size_gpu, block, grid):
    reduction_func = function[0]
    half_size = function[1]
    while array_size != 1:
        array_size = np.ceil(array_size / 2)
        half_size(size_gpu, block=(1, 1, 1))
        reduction_func(intermediate_mat_gpu, save_mat_gpu, size_gpu, block=block, grid=grid)
    half_size(size_gpu, block=(1, 1, 1))

def GPUposeUpdate(function=None, pose=None, jacobian_gpu=None, jacobian_t_gpu=None, residual_gpu=None, intermediate_square_jacobian_gpu=None, square_jacobian_gpu=None, intermediate_jacobian_residual_gpu=None, jacobian_residual_gpu=None, square_size_gpu=None, jac_size_gpu=None, pcd_size=None):
    '''
    Args

    function : gpu function (GPUmatmul)
    pose : current pose to update
    jacobian gpu : final jacobian in gpu ( N x 6 )
    residual gpu : residual in gpu ( N x 1 )
    square_jacobian gpu : J_T * J in gpu ( 6 x 6 )
    jacobian_residual gpu : J_T * residual ( 6 x 1 )
    pcd_size : N (pointcloud size)
    '''
    block_size = (15, 15, 1)
    grid_size = (50, 50, 7)
    matmul_func, reduction_func = function[0], function[1:]

    # compute square jacobian

    # 0.006sec
    #st = time.time()
    matmul_func(jacobian_t_gpu, jacobian_gpu, intermediate_square_jacobian_gpu, square_size_gpu, block=block_size, grid=grid_size)
    reduction(reduction_func, pcd_size, intermediate_square_jacobian_gpu, square_jacobian_gpu, square_size_gpu, block_size, grid_size)
    #print(time.time()-st)

    # compute jacobian residual
    matmul_func(jacobian_t_gpu, residual_gpu, intermediate_jacobian_residual_gpu, jac_size_gpu, block=block_size, grid=grid_size)
    reduction(reduction_func, pcd_size, intermediate_jacobian_residual_gpu, jacobian_residual_gpu, jac_size_gpu, block_size, grid_size)
    #print(time.time()-st)

    #st = time.time()
    jacobian_residual = np.zeros(shape=(6, 1), dtype=np.float32)
    cuda.memcpy_dtoh(jacobian_residual, jacobian_residual_gpu)
    #print(time.time() - st)


    #st = time.time()
    square_jacobian = np.zeros(shape=(6, 6), dtype=np.float32)
    cuda.memcpy_dtoh(square_jacobian, square_jacobian_gpu)
    #print(time.time()-st)

    # transfer jacobian residual  from GPU to CPU


    # compute new pose
    #st = time.time()
    pose = pose - np.linalg.inv(square_jacobian) @ jacobian_residual

    # st =time.time()
    # pose_gpu = cuda.mem_alloc(pose.nbytes)
    # cuda.memcpy_htod(pose_gpu, pose)
    # # print(time.time()-st)
    #
    # st = time.time()
    # cuda.memcpy_dtoh(pose, pose_gpu)
    # print(time.time() - st)

    return pose


def GPUmatmul(func, rot_pose, trans_pose, pcd_gpu, tr_pcd_gpu, row, medium, col):

    #cuda.memcpy_htod(rot_pose_gpu, rot_pose)
    #cuda.memcpy_htod(trans_pose_gpu, trans_pose)

    size = np.array([[row, medium, col]], dtype=np.float32) # size = (row, medium, col) -> (row x medium) x (medium x col)
    func(cuda.In(rot_pose), cuda.In(trans_pose), pcd_gpu, tr_pcd_gpu, cuda.In(size), block=(25, 25, 1), grid=(25, 25, 5))


def GPUcomputeResidual(function=None, residual_gpu=None, prev_image_gpu=None, current_image_gpu=None, prev_image_coord_gpu=None, current_image_coord_gpu=None, validIdx_gpu=None, size_gpu=None, block=None, grid=None):

    # validIdx size , image height, image width

    '''
    set Datatype to np.float32 

    residual -> (N x 1)
    size -> (1 x 3) # (Number of points, width, height)
    validIdx -> (N, )
    current_image_coord -> (2 x N)
    prev_image_coord -> (2 x N)
    
    current_image_gray -> (height x width)
    prev_image_gray -> (height x width)
    
    '''
    function(residual_gpu, size_gpu, validIdx_gpu, current_image_coord_gpu, prev_image_coord_gpu, current_image_gpu, prev_image_gpu, block=block, grid=grid)


def GPUcomputeImgJacobian(function, img_jacobian_gpu, grad_x_gpu, grad_y_gpu, current_image_coord_gpu, validIdx_gpu, size_gpu, block, grid):
    # validIdx size , image height, image width

    '''
    set Datatype to np.float32 

    *** (valid N) <= N ***
    j_img -> (valid N x 1 )
    sizeParams -> (1 x 4) [(Number of point), (Number of valid point), height, width ]
    validIdx -> (valid N, )
    image_coord -> (2 x N)
    grad_x -> (h x w)
    grad_y -> (h x w)
    '''

    #st = time.time()
    function(img_jacobian_gpu, size_gpu, validIdx_gpu, current_image_coord_gpu, grad_x_gpu, grad_y_gpu,
                       block=block, grid=grid)


def GPUcomputePoseProjectionJacobian(function, jacobian_gpu, jacobian_t_gpu, img_jacobian_gpu, validIdx_gpu, intrinsic_gpu, transformed_pcd_gpu, pcd_gpu, size_gpu, block, grid):

    '''
    Imagejacobian (valid N x 2)
    validIdx( valid N, )
    intrinsic ( 3 x 3 )
    transformed_pcd ( 3 x N )
    pcd ( 3 x N )

    return full jacobian ( N x 6 ) -> ImageJacobian x ProjectionJacobian x Pose Jacobian -> (N x 6)
    '''

    function(jacobian_gpu, jacobian_t_gpu, img_jacobian_gpu, size_gpu, validIdx_gpu, transformed_pcd_gpu, pcd_gpu, intrinsic_gpu,
                       block=block, grid=grid)

