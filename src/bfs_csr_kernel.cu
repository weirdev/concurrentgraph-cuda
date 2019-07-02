// Compressed sparse row format
// Rows transmit to columns
#include <math.h>
#include "cuda_runtime.h"
#include "bfs_csr_kernel.h"

__global__ void breadth_first_search_csr_gpu(unsigned int* cum_row_indexes, unsigned int* column_indexes, 
                                        int* matrix_data, unsigned int* in_infections,
                                        unsigned int* out_infections, unsigned int rows) {
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows) {
        if (in_infections[row] == 1) {
            out_infections[row] = 1;

            unsigned int row_start = cum_row_indexes[row]; 
            unsigned int row_end = cum_row_indexes[row+1];

            for (int i = row_start; i < row_end; i++) {
                int timesteps_to_transmission = matrix_data[i];
                if (timesteps_to_transmission != 0) {
                    if (timesteps_to_transmission == 1) {
                        out_infections[column_indexes[i]] = 1;
                    }
                    matrix_data[i] -= 1;
                }
            }
        }
    }
}

void internal_breadth_first_search_csr_gpu(unsigned int* cum_row_indexes, unsigned int* column_indexes, 
                                        int* matrix_data, unsigned int* in_infections,
                                        unsigned int* out_infections, unsigned int rows) {
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(rows);
    dim3 blocksPerGrid(1);
    if (rows > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(rows)/double(threadsPerBlock.x));
    }

    breadth_first_search_csr_gpu<<<blocksPerGrid,threadsPerBlock>>>(cum_row_indexes, 
        column_indexes, matrix_data, in_infections, out_infections, rows);
}
