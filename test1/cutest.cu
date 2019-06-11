#include <stdio.h>
#include <cuda.h>
#include "cutest.h"

int hello(void) {
    printf("Hello world!\n");
    return 0;
}

void negative_prob_multiply_matrix_vector(float* matrix, float* in_vector, float* out_vector, uint outerdim, uint innerdim) {
    for (int i=0; i < outerdim; i++) {
        float prob = 1;
        for (int j=0; j < innerdim; j++) {
            prob *= 1 - (matrix[i*innerdim + j] * in_vector[j]);
        }
        out_vector[i] = prob;
    }
}