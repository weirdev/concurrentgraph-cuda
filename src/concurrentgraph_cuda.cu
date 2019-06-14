#include <stdio.h>
#include <cuda.h>

#include "dev_array.h"
#include "concurrentgraph_cuda.h"
#include "npmmv_kernel.h"

void negative_prob_multiply_matrix_vector_cpu(int iters, float* matrix, float* in_vector, float* out_vector, uint outerdim, uint innerdim) {
    for (int t=0; t<iters; t++) {
        for (int i=0; i < outerdim; i++) {
            float prob = 1;
            for (int j=0; j < innerdim; j++) {
                prob *= 1 - (matrix[i*innerdim + j] * in_vector[j]);
            }
            out_vector[i] = prob;
        }
    }
}

void negative_prob_multiply_matrix_vector_gpu(int iters, float* matrix, float* in_vector, float* out_vector, uint outerdim, uint innerdim) {
    // Allocate memory on the device
    dev_array<float> d_matrix(outerdim*innerdim);
    dev_array<float> d_in_vector(innerdim);
    dev_array<float> d_out_vector(outerdim);

    d_matrix.set(matrix, outerdim*innerdim);
    d_in_vector.set(in_vector, innerdim);

    for (int t=0; t<iters; t++) {
        internal_negative_prob_multiply_matrix_vector_gpu(d_matrix.getData(), d_in_vector.getData(), d_out_vector.getData(), innerdim, outerdim);
        cudaDeviceSynchronize();
    }

    d_out_vector.get(out_vector, outerdim);
    cudaDeviceSynchronize();
}
