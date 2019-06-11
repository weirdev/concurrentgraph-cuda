#include <stdio.h>
#include <cuda.h>

/*
#ifdef __cplusplus
extern "C" {
#endif

void __declspec(dllexport) negative_prob_multiply_matrix_vector(float* matrix, float* in_vector, float* out_vector, uint outerdim, uint innerdim);

#ifdef __cplusplus
}
#endif
*/

int main(void) {
    printf("Hello world!\n");
    return 0;
}

void multiply_matrix_vector(float* matrix, float* in_vector, float* out_vector, int* outerdim, int* innerdim) {
    for (int i=0; i < outerdim; i++) {
        float sum = 0;
        for (int j=0; j < innerdim; j++) {
            sum += matrix[i*innerdim + j] * in_vector[j];
        }
        out_vector[i] = sum;
    }
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