#include <math.h>
#include "cuda_runtime.h"
#include "npmmv_kernel.h"

__global__ void negative_prob_multiply_matrix_vector_kernel(float* matrix, float* in_vector, float* out_vector, 
    unsigned int outerdim, unsigned int innerdim) {
    // We parallelize at the level of matrix rows, 
    unsigned int ROW = blockIdx.x*blockDim.x+threadIdx.x;

    float prob = 1.0;

    if (ROW < outerdim) {
        // each thread computes one element of the output vector
        for (int i = 0; i < innerdim; i++) {
            prob *= 1.0 - (matrix[ROW * innerdim + i] * in_vector[i]);
        }
        out_vector[ROW] = prob;
    }
}

void internal_negative_prob_multiply_matrix_vector_gpu(float* matrix, float* in_vector, float* out_vector, 
                                                unsigned int outerdim, unsigned int innerdim) {
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(outerdim);
    dim3 blocksPerGrid(1);
    if (outerdim > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(outerdim)/double(threadsPerBlock.x));
    }

    negative_prob_multiply_matrix_vector_kernel<<<blocksPerGrid,threadsPerBlock>>>(matrix, in_vector, out_vector, outerdim, innerdim);
}
