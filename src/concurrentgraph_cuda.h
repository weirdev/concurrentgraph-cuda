#ifndef CONCURRENTGRAPH_CUDA_H_
#define CONCURRENTGRAPH_CUDA_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void negative_prob_multiply_dense_matrix_vector_cpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
    void negative_prob_multiply_dense_matrix_vector_gpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);

    struct GpuFloatArray {
        float* start;
        float* end;
    };

    struct GpuUIntArray {
        unsigned int* start;
        unsigned int* end;
    };
    
    void npmmv_gpu_set_float_array(float* src, uint size, struct GpuFloatArray dst);
    void npmmv_gpu_get_float_array(struct GpuFloatArray src, float* dst, uint size);
    void npmmv_gpu_set_uint_array(uint* src, uint size, struct GpuUIntArray dst);
    void npmmv_gpu_get_uint_array(struct GpuUIntArray src, uint* dst, uint size);

    struct NpmmvDenseGpuAllocations {
        struct GpuFloatArray matrix;
        struct GpuFloatArray in_vector;
        struct GpuFloatArray out_vector;
    };

    struct NpmmvDenseGpuAllocations npmmv_dense_gpu_allocate(uint outerdim, uint innerdim);
    void npmmv_dense_gpu_free(struct NpmmvDenseGpuAllocations gpu_allocations);
    void npmmv_gpu_set_dense_matrix(float* matrix_cpu, struct GpuFloatArray matrix_gpu, uint outerdim, uint innerdim);
    void npmmv_dense_gpu_compute(struct NpmmvDenseGpuAllocations gpu_allocations, uint outerdim, uint innerdim);

    struct CsrMatrixPtrs {
        uint* cum_row_indexes;
        uint* column_indexes;
        float* values;
    };

    struct NpmmvCsrGpuAllocations {
        struct GpuUIntArray mat_cum_row_indexes;
        struct GpuUIntArray mat_column_indexes;
        struct GpuFloatArray mat_values;
        struct GpuFloatArray in_vector;
        struct GpuFloatArray out_vector;
    };

    struct NpmmvCsrGpuAllocations npmmv_csr_gpu_allocate(uint outerdim, uint innerdim, uint values);
    void npmmv_csr_gpu_free(struct NpmmvCsrGpuAllocations gpu_allocations);
    void npmmv_gpu_set_csr_matrix(struct CsrMatrixPtrs matrix_cpu, struct NpmmvCsrGpuAllocations gpu_allocations, uint outerdim, uint values);
    void npmmv_csr_gpu_compute(struct NpmmvCsrGpuAllocations gpu_allocations, uint outerdim);

#ifdef __cplusplus
}
#endif
#endif