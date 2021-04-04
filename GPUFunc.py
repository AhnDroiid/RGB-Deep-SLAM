from pycuda.compiler import SourceModule



def getGpuProjectionFunc():
    return SourceModule("""
        __global__ void ProjectPCD(float * pcd, float * current_image_coord, float * validIdx, float * intrinsic, float * size){
            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                            blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
            float fx = intrinsic[0];
            float cx = intrinsic[1];
            float fy = intrinsic[2];
            float cy = intrinsic[3];
            
            int N = int(size[0]);
            float width = size[1];
            float height = size[2];
            
            
            
            while(tid < N){
                float X = pcd[0 * N + tid];
                float Y = pcd[1 * N + tid];
                float Z = pcd[2 * N + tid];              
                
                float u = (X / Z) * fx + cx;
                float v = (Y / Z) * fy + cy;
                //printf("fx: %f, cx: %f, fy: %f, cy: %f, N: %d, width: %f, height: %f, X:%f, Y:%f, Z:%f\\n", fx, cx, fy, cy, N, width, height, X, Y, Z);
                
                if(u >= 0 && u < width && v >=0 && v < height){
                    validIdx[tid] = 1;
                    current_image_coord[0 * N + tid] = u;
                    current_image_coord[1 * N + tid] = v;
                }
                
                tid += gridSize * blockSize;
            }
            
        }    
    """).get_function("ProjectPCD")

def getGpuLargeMediumMatMulFunc():
    return [
        SourceModule("""
        __global__ void LargeMediumMatMul(float * src_mat, float * tar_mat, float * intermediate_mat, float * size){
            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                            blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
            int row = int(size[0]);
            int medium = int(size[3]);
            int col = int(size[2]);
            
            int tar_row = int(int(tid/medium) / col);
            int tar_col = (int(tid/medium) % col);
            int res = (tid)%(medium);
            
            
            while(tar_row < row){ 
                //if(tar_row == 6)
                   // printf("res : %d, tar_row: %d, tar_col : %d, row : %d, col : %d\\n", res, tar_row, tar_col, row, col);
                intermediate_mat[res*row*col + tar_row * col + tar_col] = src_mat[tar_row * medium + res] * tar_mat[res * col + tar_col]; 
                
                tid += gridSize * blockSize;
                tar_row = int(int(tid/medium) / col);
                tar_col = (int(tid/medium) % col);
                res = (tid)%(medium);
            }
            
            /*if(tid < medium){
                //printf("tid : %d\\n", tid);
                for(int i = 0; i < row; i++)
                {
                    for(int j=0; j < col; j++){
                        intermediate_mat[tid*row*col + i * col + j] = src_mat[i * medium + tid] * tar_mat[tid * col + j]; 
                    }
                }
            
            }*/
        }
        """).get_function("LargeMediumMatMul"),

        SourceModule("""
        __global__ void half_reduction(float * intermediate_mat, float * save, float * size){
            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                            blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
            int row = int(size[0]);
            int reduction_size = int(size[1]);
            int col = int(size[2]);
            int tar_row = int(int(tid/reduction_size) / col);
            int tar_col = (int(tid/reduction_size) % col);
            int res = (tid)%(reduction_size);

            while(tar_row < row){
               intermediate_mat[(res) * row * col + tar_row * col + tar_col] += intermediate_mat[((res)+reduction_size) * row * col + tar_row * col + tar_col];
                if(reduction_size == 1){
                    save[tar_row * col + tar_col] = intermediate_mat[0 * row * col + tar_row * col + tar_col];
                }
                tid += gridSize * blockSize;
                tar_row = int(int(tid/reduction_size) / col);
                tar_col = (int(tid/reduction_size) % col);
                res = (tid)%(reduction_size);
            }
            
        }
        """).get_function("half_reduction")
        ,
        SourceModule("""
        __global__ void half_size(float * size){
            if(size[1] == 1){
                size[1] = size[3];
            }
            else{
                size[1] = ceilf(size[1] / 2);
            }
        }
        """).get_function("half_size")
    ]


def getGpuSmallMediumMatMulFunc():
    return SourceModule("""
        __global__ void matmul(float * src_mat, float * tar_mat, float * save, float * size){
            int row = int(size[0]);
            int medium = int(size[1]);
            int col = int(size[2]);

            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                            blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
            int tar_row = int(tid / col);
            int tar_col = (tid % col);
            
            while(tar_row <= (row-1))
            {
                float reduction = 0.0;

                for(int i =0; i < medium; i++){
                    reduction += src_mat[tar_row * medium + i] * tar_mat[i * col + tar_col];
                }

                save[tar_row * col + tar_col] = reduction;
                tid += gridSize * blockSize;
                tar_row = int(tid / col);
                tar_col = (tid % col);
            }
        }
    """).get_function(name="matmul")

def getGpuPoseMatMulFunc():
    return SourceModule("""
        __global__ void Posematmul(float * rot_pose, float* trans_pose, float * pcd, float * tr_pcd, float * size){
            int row = int(size[0]);
            int medium = int(size[1]);
            int col = int(size[2]);
            
            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                            blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
            int tar_row = int(tid / col);
            int tar_col = (tid % col);
            //printf("row: %d, medium : %d, col: %d\\n", row, medium, col);
            //printf("tid: %d\\n", tid);
            while(tar_row <= (row-1))
            {
                float reduction = 0.0;
                
                for(int i =0; i < medium; i++){
                    reduction += rot_pose[tar_row * medium + i] * pcd[i * col + tar_col];
                }
                
                tr_pcd[tar_row * col + tar_col] = reduction + trans_pose[tar_row];
                tid += gridSize * blockSize;
                tar_row = int(tid / col);
                tar_col = (tid % col);
            }

        }
    """).get_function(name="Posematmul")

def getComputeResidualFunc():
    return SourceModule(
        """
        __global__ void computeResidual(float * residual, float * size, float * validIdx, float * current_image_coord, float * prev_image_coord, float * current_image_gray, float * prev_image_gray){
            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                        blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
            
            int height = int(size[2]); 
            int width = int(size[1]);
            int N = int(size[0]);

            while(tid < N){

                if(validIdx[tid] == 0){
                 tid += blockDim.x * blockDim.y * blockDim.z * blockSize;
                 continue;
                }
                float t_u = current_image_coord[tid];
                float t_v = current_image_coord[N + tid];
                float u = prev_image_coord[tid];
                float v = prev_image_coord[N + tid];


                float prev_bilinear = 0.0;
                float current_bilinear = 0.0;

                if(u == int(u) && v == int(v)){
                    prev_bilinear = prev_image_gray[int(v) * width + int(u)];
                }
                else{
                    int u0 = int(u);
                    int v0 = int(v);
                    int u1 = u0 + 1;
                    int v1 = v0 + 1;

                    float u1_weight = u - u0;
                    float v1_weight = v - v0;
                    float u0_weight = 1 - u1_weight;
                    float v0_weight = 1 - v1_weight;

                    if(u0 < 0 || u0 >= width) u0_weight = 0;
                    if(u1 < 0 || u1 >= width) u1_weight = 0;
                    if(v0 < 0 || v0 >= height) v0_weight = 0;
                    if(v1 < 0 || v1 >= height) v1_weight = 0;



                    float w00 = u0_weight * v0_weight;
                    float w10 = u1_weight * v0_weight;
                    float w01 = u0_weight * v1_weight;
                    float w11 = u1_weight * v1_weight;

                    float sum_weights = w00 + w10 + w01 + w11;
                    float total = 0;

                    if(w00 > 0) total += prev_image_gray[v0 * width + u0] * w00;
                    if(w10 > 0) total += prev_image_gray[v0 * width + u1] * w10;
                    if(w01 > 0) total += prev_image_gray[v1 * width + u0] * w01;
                    if(w11 > 0) total += prev_image_gray[v1 * width + u1] * w11;

                    prev_bilinear = total / sum_weights;
                }       

                if(t_u == int(t_u) && t_v == int(t_v)){
                    current_bilinear = current_image_gray[int(t_v) * width + int(t_u)];
                }
                else{
                    int t_u0 = int(t_u);
                    int t_v0 = int(t_v);
                    int t_u1 = t_u0 + 1;
                    int t_v1 = t_v0 + 1;

                    float t_u1_weight = t_u - t_u0;
                    float t_v1_weight = t_v - t_v0;
                    float t_u0_weight = 1 - t_u1_weight;
                    float t_v0_weight = 1 - t_v1_weight;

                    if(t_u0 < 0 || t_u0 >= width) t_u0_weight = 0;
                    if(t_u1 < 0 || t_u1 >= width) t_u1_weight = 0;
                    if(t_v0 < 0 || t_v0 >= height) t_v0_weight = 0;
                    if(t_v1 < 0 || t_v1 >= height) t_v1_weight = 0;

                    float t_w00 = t_u0_weight * t_v0_weight;
                    float t_w10 = t_u1_weight * t_v0_weight;
                    float t_w01 = t_u0_weight * t_v1_weight;
                    float t_w11 = t_u1_weight * t_v1_weight;

                    float sum_weights = t_w00 + t_w10 + t_w01 + t_w11;
                    float total = 0;

                    if(t_w00 > 0) total += current_image_gray[t_v0 * width + t_u0] * t_w00;
                    if(t_w10 > 0) total += current_image_gray[t_v0 * width + t_u1] * t_w10;
                    if(t_w01 > 0) total += current_image_gray[t_v1 * width + t_u0] * t_w01;
                    if(t_w11 > 0) total += current_image_gray[t_v1 * width + t_u1] * t_w11;

                    current_bilinear = total / sum_weights;
                }

                residual[tid] = prev_bilinear - current_bilinear;

                tid += gridSize * blockSize;
            }


        }
        """
    ).get_function(name="computeResidual")

def getComputeImgJacobianFunc():

    return SourceModule(
        """
        __global__ void computeImgJacobian(float * j_img, float * size, float * validIdx, float * image_coord, float * grad_x, float * grad_y){
            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                        blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
    
            int height = int(size[2]); 
            int width = int(size[1]);
            int N = int(size[0]);

            while(tid < N){
                
                if(validIdx[tid]==0){
                 tid += blockDim.x * blockDim.y * blockDim.z * blockSize;
                 continue;
                }

                float u = image_coord[tid];
                float v = image_coord[N + tid];

         
                float j_0 = 0.0;
                float j_1 = 0.0;

                if(u == int(u) && v == int(v)){
                    j_0 = grad_x[int(v) * width + int(u)];
                    j_1 = grad_y[int(v) * width + int(u)];
                }
                else{
                    int u0 = int(u);
                    int v0 = int(v);
                    int u1 = u0 + 1;
                    int v1 = v0 + 1;

                    float u1_weight = u - u0;
                    float v1_weight = v - v0;
                    float u0_weight = 1 - u1_weight;
                    float v0_weight = 1 - v1_weight;

                    if(u0 < 0 || u0 >= width) u0_weight = 0;
                    if(u1 < 0 || u1 >= width) u1_weight = 0;
                    if(v0 < 0 || v0 >= height) v0_weight = 0;
                    if(v1 < 0 || v1 >= height) v1_weight = 0;



                    float w00 = u0_weight * v0_weight;
                    float w10 = u1_weight * v0_weight;
                    float w01 = u0_weight * v1_weight;
                    float w11 = u1_weight * v1_weight;

                    float sum_weights = w00 + w10 + w01 + w11;

                    float total_0 = 0.0;
                    float total_1 = 0.0;

                    if(w00 > 0) {
                        total_0 += grad_x[v0 * width + u0] * w00;
                        total_1 += grad_y[v0 * width + u0] * w00;
                    }

                    if(w10 > 0) {
                        total_0 += grad_x[v0 * width + u1] * w10;
                        total_1 += grad_y[v0 * width + u1] * w10;
                    }

                    if(w01 > 0) {
                        total_0 += grad_x[v1 * width + u0] * w01;
                        total_1 += grad_y[v1 * width + u0] * w01;
                    }

                    if(w11 > 0) {
                        total_0 += grad_x[v1 * width + u1] * w11;
                        total_1 += grad_y[v1 * width + u1] * w11;
                    }

                    j_0 = total_0 / sum_weights;
                    j_1 = total_1 / sum_weights;
                }       

                j_img[tid * 2] = j_0;
                j_img[tid * 2 + 1] = j_1;

                tid += gridSize * blockSize;
            }


        }
        """
    ).get_function(name="computeImgJacobian")


def getcomputePoseProjectionFunc():
    return SourceModule(
        """
        __global__ void computePoseProjectionJacobian(float * Jacobian, float * Jacobian_transpose, float * ImageJacobian, float * size, float * validIdx, float * tr_pcd, float * pcd, float * intrinsic){
            int blockSize = blockDim.x * blockDim.y * blockDim.z;
            int gridSize = gridDim.x * gridDim.y * gridDim.z;
            int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                        blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;

            int N = int(size[0]);
        
            float fX = intrinsic[0];
            float fY = intrinsic[2];


            while(tid < N){
                if(validIdx[tid] == 0){
                 tid += blockDim.x * blockDim.y * blockDim.z * blockSize;
                 continue;
                }
                
                float x = pcd[tid];
                float y = pcd[tid + N];
                float z = pcd[tid + N * 2]; 

                float tr_x = tr_pcd[tid];
                float tr_y = tr_pcd[tid + N];
                float tr_z = tr_pcd[tid + N * 2];

                float I_x = ImageJacobian[tid * 2];
                float I_y = ImageJacobian[tid * 2 + 1];

                float Proj_x = (fX * I_x) / (tr_z);
                float Proj_y = (fY * I_y) / (tr_z);
                float Proj_z = -1 * (fX * tr_x * I_x + fY * tr_y * I_y) / (tr_z * tr_z);

                float Pose_t_x = Proj_x;
                float Pose_t_y = Proj_y;
                float Pose_t_z = Proj_z;
                float Pose_w_x = -1 * z * Proj_y + y * Proj_z;
                float Pose_w_y = z * Proj_x - x * Proj_z;
                float Pose_w_z = - 1 * y * Proj_x + x * Proj_y;
                
                // save jacobian in Jacobian mat
                Jacobian[tid * 6 + 0] = -1 * Pose_t_x;
                Jacobian[tid * 6 + 1] = -1 * Pose_t_y; 
                Jacobian[tid * 6 + 2] = -1 * Pose_t_z;
                Jacobian[tid * 6 + 3] = -1 * Pose_w_x;
                Jacobian[tid * 6 + 4] = -1 * Pose_w_y;
                Jacobian[tid * 6 + 5] = -1 * Pose_w_z;
                
                
                // save jacobian in Jacobian transpose mat
                
                Jacobian_transpose[0 * N + tid] = -1 * Pose_t_x;
                Jacobian_transpose[1 * N + tid] = -1 * Pose_t_y; 
                Jacobian_transpose[2 * N + tid] = -1 * Pose_t_z;
                Jacobian_transpose[3 * N + tid] = -1 * Pose_w_x;
                Jacobian_transpose[4 * N + tid] = -1 * Pose_w_y;
                Jacobian_transpose[5 * N + tid] = -1 * Pose_w_z;
                
                //printf("tid : %d, jaco: %f, jaco_t:%f\\n", tid, Jacobian[tid * 6 + 0], Jacobian_transpose[0*6 + tid]);
                
                tid += gridSize * blockSize;
            }
        }
        """
    ).get_function(name="computePoseProjectionJacobian")
