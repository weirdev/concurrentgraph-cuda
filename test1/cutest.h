#ifndef CUTEST_H_
#define CUTEST_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void negative_prob_multiply_matrix_vector(float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
    int hello(void);
#ifdef __cplusplus
}
#endif
#endif