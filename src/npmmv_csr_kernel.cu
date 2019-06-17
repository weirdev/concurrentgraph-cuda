// Compressed sparse row format
#include <math.h>
#include "cuda_runtime.h"
#include "npmmv_csr_kernel.h"

__global__ void negative_prob_multiply_csr_matrix_vector_kernel(unsigned int* cum_row_indexes, unsigned int* column_indexes, 
                                                float* matrix_data, float* in_vector, float* out_vector, unsigned int outerdim) {

    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < outerdim) {
        float prob = 1.0;

        unsigned int row_start = cum_row_indexes[row]; 
        unsigned int row_end = cum_row_indexes[row+1];

        for (int i = row_start; i < row_end; i++) {
            prob *= 1.0 - (matrix_data[i] * in_vector[column_indexes[i]]);
        }
        out_vector[row] = prob;
    }
}

void internal_negative_prob_multiply_csr_matrix_vector_gpu(unsigned int* cum_row_indexes, unsigned int* column_indexes, 
                                            float* matrix_data, float* in_vector, float* out_vector, unsigned int outerdim) {
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(outerdim);
    dim3 blocksPerGrid(1);
    if (outerdim > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(outerdim)/double(threadsPerBlock.x));
    }

    negative_prob_multiply_csr_matrix_vector_kernel<<<blocksPerGrid,threadsPerBlock>>>(cum_row_indexes, column_indexes, matrix_data, in_vector, out_vector, outerdim);
}
