#include <stdio.h>
#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "assert.h"
#include "gpu_types.h"

#include "concurrentgraph_cuda.h"
#include "npmmv_dense_kernel.h"
#include "npmmv_csr_kernel.h"
#include "npmmv_csr_vector_kernel.h"
#include "bfs_csr_kernel.h"
#include "graph_determ_weights.h"


void negative_prob_multiply_dense_matrix_vector_cpu(int iters, float* matrix, float* in_vector, 
                                                    float* out_vector, uint outerdim, uint innerdim) {
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

void negative_prob_multiply_dense_matrix_vector_gpu(int iters, float* matrix, float* in_vector, 
                                                    float* out_vector, uint outerdim, uint innerdim) {
    // Allocate memory on the device
    dev_array<float> d_matrix(outerdim*innerdim);
    dev_array<float> d_in_vector(innerdim);
    dev_array<float> d_out_vector(outerdim);

    d_matrix.set(matrix, outerdim*innerdim);
    d_in_vector.set(in_vector, innerdim);

    for (int t=0; t<iters; t++) {
        internal_negative_prob_multiply_dense_matrix_vector_gpu(d_matrix.getData(), d_in_vector.getData(), 
            d_out_vector.getData(), innerdim, outerdim);
        cudaDeviceSynchronize();
    }

    d_out_vector.get(out_vector, outerdim);
    cudaDeviceSynchronize();
}


struct NpmmvDenseGpuAllocations npmmv_dense_gpu_allocate(uint outerdim, uint innerdim) {
    struct GpuFloatArray matrix_mem = allocate_gpu_float_array(outerdim*innerdim);
    struct GpuFloatArray in_vector_mem = allocate_gpu_float_array(innerdim);
    struct GpuFloatArray out_vector_mem = allocate_gpu_float_array(outerdim);

    struct NpmmvDenseGpuAllocations gpu_allocations = {matrix_mem, in_vector_mem, out_vector_mem};
    return gpu_allocations;
}

void npmmv_dense_gpu_free(struct NpmmvDenseGpuAllocations gpu_allocations) {
    free_gpu_float_array(gpu_allocations.matrix);
    free_gpu_float_array(gpu_allocations.in_vector);
    free_gpu_float_array(gpu_allocations.out_vector);
}

void npmmv_gpu_set_dense_matrix(float* matrix_cpu, struct GpuFloatArray matrix_gpu, uint outerdim, uint innerdim) {
    set_gpu_float_array(matrix_cpu, outerdim*innerdim, matrix_gpu);
}

void npmmv_dense_gpu_compute(struct NpmmvDenseGpuAllocations gpu_allocations, uint outerdim, uint innerdim) {
    if (gpu_allocations.matrix.end - gpu_allocations.matrix.start < outerdim * innerdim ||
            gpu_allocations.in_vector.end - gpu_allocations.in_vector.start < innerdim || 
            gpu_allocations.out_vector.end - gpu_allocations.out_vector.start < outerdim) {
        
        throw std::out_of_range("Attempted to compute over more data than allocated on the device");
    }
    internal_negative_prob_multiply_dense_matrix_vector_gpu(gpu_allocations.matrix.start, 
        gpu_allocations.in_vector.start, gpu_allocations.out_vector.start, innerdim, outerdim);
    cudaDeviceSynchronize();
}

struct NpmmvCsrGpuAllocations npmmv_csr_gpu_allocate(uint outerdim, uint innerdim, uint values) {
    std::cout << "test prnt\n";
    struct GpuUIntArray mat_cum_row_indexes = allocate_gpu_uint_array(outerdim + 1);
    struct GpuUIntArray mat_column_indexes = allocate_gpu_uint_array(values);
    struct GpuFloatArray mat_values = allocate_gpu_float_array(values);
    struct GpuFloatArray in_vector_mem = allocate_gpu_float_array(innerdim);
    struct GpuFloatArray out_vector_mem = allocate_gpu_float_array(outerdim);

    struct NpmmvCsrGpuAllocations gpu_allocations = {mat_cum_row_indexes, mat_column_indexes, mat_values, 
        in_vector_mem, out_vector_mem};
    return gpu_allocations;
}

void npmmv_csr_gpu_free(struct NpmmvCsrGpuAllocations gpu_allocations) {
    free_gpu_uint_array(gpu_allocations.mat_cum_row_indexes);
    free_gpu_uint_array(gpu_allocations.mat_column_indexes);
    free_gpu_float_array(gpu_allocations.mat_values);
    free_gpu_float_array(gpu_allocations.in_vector);
    free_gpu_float_array(gpu_allocations.out_vector);
}

void npmmv_gpu_set_csr_matrix(struct CsrFloatMatrixPtrs matrix_cpu, struct NpmmvCsrGpuAllocations gpu_allocations, 
                                uint outerdim, uint values) {
    set_gpu_uint_array(matrix_cpu.cum_row_indexes, outerdim+1, gpu_allocations.mat_cum_row_indexes);
    set_gpu_uint_array(matrix_cpu.column_indexes, values, gpu_allocations.mat_column_indexes);
    set_gpu_float_array(matrix_cpu.values, values, gpu_allocations.mat_values);
}

void npmmv_csr_gpu_compute(struct NpmmvCsrGpuAllocations gpu_allocations, uint outerdim, uint computation_restriction_factor) {
    //internal_negative_prob_multiply_csr_matrix_vector_gpu
    internal_spmv_csr_veck_gpu(computation_restriction_factor,
        gpu_allocations.mat_cum_row_indexes.start, 
        gpu_allocations.mat_column_indexes.start, gpu_allocations.mat_values.start, 
        gpu_allocations.in_vector.start, gpu_allocations.out_vector.start, outerdim);
    gpuErrchk(cudaDeviceSynchronize());
}

struct BfsCsrGpuAllocations bfs_csr_gpu_allocate(uint rows, uint values) {
    struct GpuUIntArray mat_cum_row_indexes = allocate_gpu_uint_array(rows + 1);
    struct GpuUIntArray mat_column_indexes = allocate_gpu_uint_array(values);
    struct GpuIntArray mat_values = allocate_gpu_int_array(values);
    struct GpuUIntArray in_infections = allocate_gpu_uint_array(rows);
    struct GpuUIntArray out_infections = allocate_gpu_uint_array(rows);

    struct BfsCsrGpuAllocations gpu_allocations = {mat_cum_row_indexes, mat_column_indexes, mat_values, 
        in_infections, out_infections};
    return gpu_allocations;
}

void bfs_csr_gpu_free(struct BfsCsrGpuAllocations gpu_allocations) {
    free_gpu_uint_array(gpu_allocations.mat_cum_row_indexes);
    free_gpu_uint_array(gpu_allocations.mat_column_indexes);
    free_gpu_int_array(gpu_allocations.mat_values);
    free_gpu_uint_array(gpu_allocations.in_infections);
    free_gpu_uint_array(gpu_allocations.out_infections);
}

void bfs_gpu_set_csr_matrix(struct CsrIntMatrixPtrs matrix_cpu, struct BfsCsrGpuAllocations gpu_allocations, 
                            uint rows, uint values) {
    set_gpu_uint_array(matrix_cpu.cum_row_indexes, rows+1, gpu_allocations.mat_cum_row_indexes);
    set_gpu_uint_array(matrix_cpu.column_indexes, values, gpu_allocations.mat_column_indexes);
    set_gpu_int_array(matrix_cpu.values, values, gpu_allocations.mat_values);
}

void bfs_csr_gpu_compute(struct BfsCsrGpuAllocations gpu_allocations, uint rows) {
    internal_breadth_first_search_csr_gpu(gpu_allocations.mat_cum_row_indexes.start, 
        gpu_allocations.mat_column_indexes.start, gpu_allocations.mat_values.start,
        gpu_allocations.in_infections.start, gpu_allocations.out_infections.start, rows);
    gpuErrchk(cudaDeviceSynchronize());
}

int* graph_deterministic_weights(struct CsrFloatMatrixPtrs contact_matrix_cpu, 
        uint rows, uint values, float* immunities, float* shedding_curve, uint infection_length, 
        float transmission_rate) {
    printf("start\n");
    struct GpuUIntArray contact_mat_cum_row_indexes = allocate_gpu_uint_array(rows + 1);
    printf("mat alloc1\n");
    set_gpu_uint_array(contact_matrix_cpu.cum_row_indexes, rows + 1, contact_mat_cum_row_indexes);
    printf("mat alloc2\n");
    struct GpuUIntArray contact_mat_column_indexes = allocate_gpu_uint_array(values);
    printf("mat alloc3\n");
    set_gpu_uint_array(contact_matrix_cpu.column_indexes, values, contact_mat_column_indexes);
    printf("mat alloc4\n");
    struct GpuFloatArray contact_mat_values = allocate_gpu_float_array(values);
    printf("mat alloc5\n");
    set_gpu_float_array(contact_matrix_cpu.values, values, contact_mat_values);
    printf("mat alloc all\n");

    struct GpuFloatArray immunities_gpu = allocate_gpu_float_array(rows);
    set_gpu_float_array(immunities, rows, immunities_gpu);
    printf("immun alloc\n");
    struct GpuFloatArray shedding_curve_gpu = allocate_gpu_float_array(infection_length);
    set_gpu_float_array(shedding_curve, infection_length, shedding_curve_gpu);
    printf("shed alloc\n");

    struct GpuIntArray infection_mat_values = allocate_gpu_int_array(values);
    printf("inf alloc\n");
    
    printf("calc start\n");
    internal_graph_determ_weights(contact_mat_cum_row_indexes.start, contact_mat_column_indexes.start, 
        contact_mat_values.start, rows, values, immunities_gpu.start, shedding_curve_gpu.start, 
        infection_length, transmission_rate, infection_mat_values.start);
    gpuErrchk(cudaDeviceSynchronize());
    printf("calc done\n");
    int* csr_determ_weight_values = (int*)malloc(values * sizeof(int));
    get_gpu_int_array(infection_mat_values, csr_determ_weight_values, values);
    return csr_determ_weight_values;
}