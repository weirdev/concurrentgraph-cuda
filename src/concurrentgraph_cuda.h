#ifndef CONCURRENTGRAPH_CUDA_H_
#define CONCURRENTGRAPH_CUDA_H_

#ifdef __cplusplus
extern "C" {
#endif
    void negative_prob_multiply_dense_matrix_vector_cpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
    void negative_prob_multiply_dense_matrix_vector_gpu(int iters, float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);

    struct NpmmvDenseGpuAllocations {
        struct GpuFloatArray matrix;
        struct GpuFloatArray in_vector;
        struct GpuFloatArray out_vector;
    };

    struct NpmmvDenseGpuAllocations npmmv_dense_gpu_allocate(uint outerdim, uint innerdim);
    void npmmv_dense_gpu_free(struct NpmmvDenseGpuAllocations gpu_allocations);
    void npmmv_gpu_set_dense_matrix(float* matrix_cpu, struct GpuFloatArray matrix_gpu, uint outerdim, uint innerdim);
    void npmmv_dense_gpu_compute(struct NpmmvDenseGpuAllocations gpu_allocations, uint outerdim, uint innerdim);

    struct NpmmvCsrGpuAllocations {
        struct GpuUIntArray mat_cum_row_indexes;
        struct GpuUIntArray mat_column_indexes;
        struct GpuFloatArray mat_values;
        struct GpuFloatArray in_vector;
        struct GpuFloatArray out_vector;
    };

    struct NpmmvCsrGpuAllocations npmmv_csr_gpu_allocate(uint outerdim, uint innerdim, uint values);
    void npmmv_csr_gpu_free(struct NpmmvCsrGpuAllocations gpu_allocations);
    void npmmv_gpu_set_csr_matrix(struct CsrFloatMatrixPtrs matrix_cpu, struct NpmmvCsrGpuAllocations gpu_allocations, uint outerdim, uint values);
    void npmmv_csr_gpu_compute(struct NpmmvCsrGpuAllocations gpu_allocations, uint outerdim, uint computation_restriction_factor);

    struct BfsCsrGpuAllocations {
        struct GpuUIntArray mat_cum_row_indexes;
        struct GpuUIntArray mat_column_indexes;
        struct GpuIntArray mat_values;
        struct GpuUIntArray in_infections;
        struct GpuUIntArray out_infections;
    };

    struct BfsCsrGpuAllocations bfs_csr_gpu_allocate(uint rows, uint values);
    void bfs_csr_gpu_free(struct BfsCsrGpuAllocations gpu_allocations);
    void bfs_gpu_set_csr_matrix(struct CsrIntMatrixPtrs matrix_cpu, struct BfsCsrGpuAllocations gpu_allocations, uint rows, uint values);
    void bfs_csr_gpu_compute(struct BfsCsrGpuAllocations gpu_allocations, uint rows);

    int* graph_deterministic_weights(struct CsrFloatMatrixPtrs matrix_cpu, uint rows, uint values, float* immunities, float* shedding_curve, uint infection_length, float transmission_rate);

#ifdef __cplusplus
}
#endif
#endif