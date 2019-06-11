#ifndef LINKTO_H_
#define LINKTO_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void negative_prob_multiply_matrix_vector(float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
#ifdef __cplusplus
}
#endif
#endif