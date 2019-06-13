#ifndef CONCURRENTGRAPH_CUDA_H_
#define CONCURRENTGRAPH_CUDA_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void negative_prob_multiply_matrix_vector_cpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
    void negative_prob_multiply_matrix_vector_gpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);

    struct GpuFloatArray {
        float* start;
        float* end;
    };

    struct NpmmvGpuAllocations {
        struct GpuFloatArray matrix;
        struct GpuFloatArray in_vector;
        struct GpuFloatArray out_vector;
    };

    struct NpmmvGpuAllocations npmmv_gpu_allocate(uint outerdim, uint innerdim);
    void npmmv_gpu_free(struct NpmmvGpuAllocations gpu_allocations);
    void npmmv_gpu_set_array(float* src, uint size, struct GpuFloatArray dst);
    void npmmv_gpu_set_matrix(float* matrix_cpu, struct GpuFloatArray matrix_gpu, uint outerdim, uint innerdim);
    void npmmv_gpu_get_array(struct GpuFloatArray src, float* dst, uint size);
    void npmmv_gpu_compute(struct NpmmvGpuAllocations gpu_allocations, uint outerdim, uint innerdim);

#ifdef __cplusplus
}
#endif
#endif