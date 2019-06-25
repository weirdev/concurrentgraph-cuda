// Compressed sparse row format
#include <math.h>
#include "cuda_runtime.h"
#include "npmmv_csr_vector_kernel.h"

__global__ void spmv_csr_vector_kernel(unsigned int computation_restriction_factor,
                                        const unsigned int* cum_row_indexes, const unsigned int* column_indexes, 
                                        const float* matrix_data , const float* in_vector, float* out_vector, 
                                        const unsigned int outerdim) { 
    __shared__ float vals[32];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    // global thread index 
    int warp_id = thread_id / 32; 
    // global warp index 
    int lane = thread_id & (32 - 1); 
    // thread index within the warp

    int row = warp_id / computation_restriction_factor;
    if (row < outerdim) {
        int row_start = cum_row_indexes[row]; 
        int row_end = cum_row_indexes[row+1];

        // compute running prod per thread 
        vals[threadIdx.x] = 1; 
        for (int i = row_start + lane; i < row_end; i += 32) {
            vals[threadIdx.x] *= 1 - (matrix_data[i] * in_vector[column_indexes[i]]);
        }

        // parallel reduction in shared memory 
        if (lane < 16) vals[threadIdx.x] *= vals[threadIdx.x + 16]; 
        if (lane < 8) vals[threadIdx.x] *= vals[threadIdx.x + 8]; 
        if (lane < 4) vals[threadIdx.x] *= vals[threadIdx.x + 4]; 
        if (lane < 2) vals[threadIdx.x] *= vals[threadIdx.x + 2]; 
        if (lane < 1) vals[threadIdx.x] *= vals[threadIdx.x + 1];

        // first thread writes the result 
        if (lane == 0) out_vector[row] = vals[threadIdx.x];
    }
    
}

// NOTE: out_vector must be initialized with identity element (1)
void internal_spmv_csr_veck_gpu(unsigned int computation_restriction_factor,
            unsigned int* cum_row_indexes, unsigned int* column_indexes, 
            float* matrix_data, float* in_vector, float* out_vector, unsigned int outerdim) {
    // computation_restriction_factor wastes warps to simulate computation efficiency on a GPU with
    // core count = actual_cores / computation_restriction_factor

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(outerdim*32*computation_restriction_factor);
    dim3 blocksPerGrid(1);
    if (outerdim*32 > 32) {
        threadsPerBlock.x = 32;
        blocksPerGrid.x = ceil(double(outerdim*32*computation_restriction_factor)/double(threadsPerBlock.x));
    }

    spmv_csr_vector_kernel<<<blocksPerGrid,threadsPerBlock>>>(computation_restriction_factor, 
        cum_row_indexes, column_indexes, matrix_data, 
        in_vector, out_vector, outerdim);
}
