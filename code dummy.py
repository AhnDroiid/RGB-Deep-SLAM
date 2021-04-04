# pcd = pcd.transpose((1, 0)).copy() # (N x 3 -> 3 x N)  # Don't use .T !  it changes memory address to Fortran contiguous format.
#
#     height, width = image_prev.shape[0], image_prev.shape[1]
#
#
#     prev_image_coord = ProjectPCD(pcd=pcd, intrinsic=intrinsic, mode="CPU")
#
#     iter_c = 0
#
#     # Send tr, src image_coord from CPU to GPU
#     prev_image_coord_gpu = transferToGPU(prev_image_coord)
#     current_image_coord_gpu = transferToGPU(np.empty_like(prev_image_coord).astype(np.float32))
#
#
#     # Define Residual array in GPU
#     residual_gpu = transferToGPU(np.zeros(shape=(pcd.shape[1], 1), dtype=np.float32))
#
#     # Define validIdx array in GPU
#     validIdx_gpu = transferToGPU(np.zeros(shape=(1, pcd.shape[1]), dtype=np.float32))
#
#     # Send grad_x, grad_y from CPU to GPU
#     grad_x, grad_y = computeImgGradientMap(image_current)
#     grad_x_gpu, grad_y_gpu = transferToGPU(grad_x), transferToGPU(grad_y)
#
#     # Define ImgJacobian in GPU
#     img_jacobian_gpu = transferToGPU(np.zeros(shape=(pcd.shape[1], 2), dtype=np.float32))
#
#     # Define Final Jacobian and Jacobian transpose in GPU
#     jacobian_gpu = transferToGPU(np.zeros(shape=(pcd.shape[1], 6), dtype=np.float32))
#     jacobian_t_gpu = transferToGPU(np.zeros(shape=(6, pcd.shape[1]), dtype=np.float32))
#
#     # Define J_T * J (6x6) , J_T * residual in GPU for faster update of pose parameter
#     square_jacobian_gpu = transferToGPU(np.zeros(shape=(6, 6), dtype=np.float32))
#     intermediate_square_jacobian_gpu = transferToGPU(np.zeros(shape=(pcd.shape[1], 6, 6), dtype=np.float32))
#
#     jacobian_residual_gpu = transferToGPU(np.zeros(shape=(6, 1), dtype=np.float32))
#     intermediate_jacobian_residual_gpu = transferToGPU(np.zeros(shape=(pcd.shape[1], 6, 1), dtype=np.float32))
#
#     # Send current, prev image from CPU to GPU
#     current_image_gpu = transferToGPU(image_current.astype(np.float32))
#     prev_image_gpu = transferToGPU(image_prev.astype(np.float32))
#
#     # Send tr, src Pointcloud value from CPU to GPu
#     pcd_gpu = transferToGPU(pcd.astype(np.float32))
#     tr_pcd = np.empty_like(pcd, dtype=np.float32)
#     tr_pcd_gpu = transferToGPU(tr_pcd)
#
#     # Send dimension parameters from CPU to GPU
#     size = np.array([[pcd.shape[1], image_current.shape[1], image_current.shape[0]]], dtype=np.float32) # (Number of pcd points), width, height
#     size_gpu = transferToGPU(size)
#
#     # Send square ,jac size dimension parameters from CPU to GPU
#
#     square_size = np.array([[6, pcd.shape[1], 6, pcd.shape[1]]], dtype=np.float32) # [row , initial reduction, col, medium]
#     square_size_gpu = transferToGPU(square_size)
#
#     jac_size = np.array([[6, pcd.shape[1], 1, pcd.shape[1]]], dtype=np.float32) # [row , initial reduction, col, medium]
#     jac_size_gpu = transferToGPU(jac_size)
#
#     # Send inrinsic parameters from CPU to GPU
#     intrinsic_gpu = transferToGPU(np.array([intrinsic[0][0], intrinsic[0][2], intrinsic[1][1], intrinsic[1][2]], dtype=np.float32)) # fx, cx, fy, cy
