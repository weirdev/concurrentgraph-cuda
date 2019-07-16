// Compressed sparse row format
// Rows transmit to columns
#include <math.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "graph_determ_weights.h"


__global__ void graph_determ_weights(unsigned int* contact_mat_cum_row_indexes, unsigned int* contact_mat_column_indexes, 
        float* contact_mat_values, unsigned int rows, unsigned int values, float* immunities, float* shedding_curve, 
        unsigned int infection_length, float transmission_rate, int* infection_mat_values) {

    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    curandState state;
    curand_init(1234 + row, 0, 0, &state);
    if (row < rows) {
        for (int j=contact_mat_cum_row_indexes[row]; j<contact_mat_cum_row_indexes[row+1]; j++) {
            float pinf_noshed = contact_mat_values[j] * transmission_rate * (1.0 - immunities[contact_mat_column_indexes[j]]);
            int delay;
            for (delay=1; delay<infection_length+1; delay++) {
                if (curand_uniform(&state) < pinf_noshed * shedding_curve[delay - 1]) {
                    break;
                }
            }
            if (delay > infection_length) {
                delay = -1;
            }
            infection_mat_values[j] = delay;
        }
    }
}

void internal_graph_determ_weights(unsigned int* contact_mat_cum_row_indexes, unsigned int* contact_mat_column_indexes, 
        float* contact_mat_values, unsigned int rows, unsigned int values, float* immunities, float* shedding_curve, 
        unsigned int infection_length, float transmission_rate, int* infection_mat_values) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(rows);
    dim3 blocksPerGrid(1);
    if (rows > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(rows)/double(threadsPerBlock.x));
    }

    graph_determ_weights<<<blocksPerGrid,threadsPerBlock>>>(contact_mat_cum_row_indexes, contact_mat_column_indexes, 
        contact_mat_values, rows, values, immunities, shedding_curve, infection_length, transmission_rate, infection_mat_values);
}