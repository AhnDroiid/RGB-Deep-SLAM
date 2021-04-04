import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

a = np.ones((36, 633), dtype=np.float32)

st =time.time()
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
print(time.time()-st)


#st = time.time()
cuda.memcpy_dtoh(a, a_gpu)
#print(time.time()-st)



mod = SourceModule("""
    __global__ void matmul(float * a){
    
        int blockSize = blockDim.x * blockDim.y * blockDim.z;
        int tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z + 
                        blockSize * blockIdx.x + blockSize * gridDim.x * blockIdx.y + blockSize * gridDim.x * gridDim.y * blockIdx.z;
        int b = 1;
        int c = ceilf(1.1);
        a[0] += 1;
        //printf("tid: %d, c: %d, wait..\\n", tid, c);
    }
""")

func = mod.get_function(name="matmul")

#st = time.time()
func(a_gpu, block=(15, 5, 1), grid=(55, 55, 15))
#print(time.time() - st)


#st = time.time()
cuda.memcpy_dtoh(a, a_gpu)
#print(time.time()-st)
#cuda.memcpy_dtoh(c, c_gpu)

#print(np.allclose(c, np.zeros_like(c)))
#print(a_doubled)
#print(a)

