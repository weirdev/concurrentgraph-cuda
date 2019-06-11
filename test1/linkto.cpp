#include "linkto.h"

void negative_prob_multiply_matrix_vector(float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim) {
    for (int i=0; i < outerdim; i++) {
        float prob = 1;
        for (int j=0; j < innerdim; j++) {
            prob *= 1 - (matrix[i*innerdim + j] * in_vector[j]);
        }
        out_vector[i] = prob;
    }
}