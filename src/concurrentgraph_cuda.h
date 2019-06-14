#ifndef CONCURRENTGRAPH_CUDA_H_
#define CONCURRENTGRAPH_CUDA_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void negative_prob_multiply_matrix_vector_cpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
    void negative_prob_multiply_matrix_vector_gpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
#ifdef __cplusplus
}
#endif
#endif