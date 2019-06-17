#include <stdio.h>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dev_array.h"
#include "concurrentgraph_cuda.h"
#include "npmmv_dense_kernel.h"
#include "npmmv_csr_kernel.h"

void negative_prob_multiply_dense_matrix_vector_cpu(int iters, float* matrix, float* in_vector, float* out_vector, uint outerdim, uint innerdim) {
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

void negative_prob_multiply_dense_matrix_vector_gpu(int iters, float* matrix, float* in_vector, float* out_vector, uint outerdim, uint innerdim) {
    // Allocate memory on the device
    dev_array<float> d_matrix(outerdim*innerdim);
    dev_array<float> d_in_vector(innerdim);
    dev_array<float> d_out_vector(outerdim);

    d_matrix.set(matrix, outerdim*innerdim);
    d_in_vector.set(in_vector, innerdim);

    for (int t=0; t<iters; t++) {
        internal_negative_prob_multiply_dense_matrix_vector_gpu(d_matrix.getData(), d_in_vector.getData(), d_out_vector.getData(), innerdim, outerdim);
        cudaDeviceSynchronize();
    }

    d_out_vector.get(out_vector, outerdim);
    cudaDeviceSynchronize();
}

struct GpuFloatArray allocate_gpu_float_array(uint array_size) {
    float* start;
    float* end;
    cudaError_t result = cudaMalloc((void**)&start, array_size * sizeof(float));
    if (result != cudaSuccess)
    {
        start = end = 0;
        throw std::runtime_error("failed to allocate device memory");
    }
    end = start + array_size;
    struct GpuFloatArray arraystruct = {start, end};
    return arraystruct;
}

void free_gpu_float_array(struct GpuFloatArray array) {
    if (array.start != 0)
    {
        cudaFree(array.start);
    }
}

struct GpuUIntArray allocate_gpu_uint_array(uint array_size) {
    uint* start;
    uint* end;
    cudaError_t result = cudaMalloc((void**)&start, array_size * sizeof(uint));
    if (result != cudaSuccess)
    {
        start = end = 0;
        throw std::runtime_error("failed to allocate device memory");
    }
    end = start + array_size;
    struct GpuUIntArray arraystruct = {start, end};
    return arraystruct;
}

void free_gpu_uint_array(struct GpuUIntArray array) {
    if (array.start != 0)
    {
        cudaFree(array.start);
    }
}

void npmmv_gpu_set_float_array(float* src, uint size, struct GpuFloatArray dst) {
    if (dst.end - dst.start < size) {
        throw std::out_of_range("Attempted to copy more memory than allocated on the device");
    }
    cudaError_t result = cudaMemcpy(dst.start, src, size * sizeof(float), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to device memory");
    }
}

void npmmv_gpu_get_float_array(struct GpuFloatArray src, float* dst, uint size) {
    if (src.end - src.start < size) {
        throw std::out_of_range("Attempted to copy more memory to host than allocated on the device");
    }
    cudaError_t result = cudaMemcpy(dst, src.start, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to host memory");
    }
    cudaDeviceSynchronize();
}

void npmmv_gpu_set_uint_array(uint* src, uint size, struct GpuUIntArray dst) {
    if (dst.end - dst.start < size) {
        throw std::out_of_range("Attempted to copy more memory than allocated on the device");
    }
    cudaError_t result = cudaMemcpy(dst.start, src, size * sizeof(uint), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to device memory");
    }
}

void npmmv_gpu_get_uint_array(struct GpuUIntArray src, uint* dst, uint size) {
    if (src.end - src.start < size) {
        throw std::out_of_range("Attempted to copy more memory to host than allocated on the device");
    }
    cudaError_t result = cudaMemcpy(dst, src.start, size * sizeof(uint), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to host memory");
    }
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
    npmmv_gpu_set_float_array(matrix_cpu, outerdim*innerdim, matrix_gpu);
}

void npmmv_dense_gpu_compute(struct NpmmvDenseGpuAllocations gpu_allocations, uint outerdim, uint innerdim) {
    if (gpu_allocations.matrix.end - gpu_allocations.matrix.start < outerdim * innerdim ||
            gpu_allocations.in_vector.end - gpu_allocations.in_vector.start < innerdim || 
            gpu_allocations.out_vector.end - gpu_allocations.out_vector.start < outerdim) {
        
        throw std::out_of_range("Attempted to compute over more data than allocated on the device");
    }
    internal_negative_prob_multiply_dense_matrix_vector_gpu(gpu_allocations.matrix.start, gpu_allocations.in_vector.start, gpu_allocations.out_vector.start, innerdim, outerdim);
    cudaDeviceSynchronize();
}

struct NpmmvCsrGpuAllocations npmmv_csr_gpu_allocate(uint outerdim, uint innerdim, uint values) {
    struct GpuUIntArray mat_cum_row_indexes = allocate_gpu_uint_array(outerdim + 1);
    struct GpuUIntArray mat_column_indexes = allocate_gpu_uint_array(values);
    struct GpuFloatArray mat_values = allocate_gpu_float_array(values);
    struct GpuFloatArray in_vector_mem = allocate_gpu_float_array(innerdim);
    struct GpuFloatArray out_vector_mem = allocate_gpu_float_array(outerdim);

    struct NpmmvCsrGpuAllocations gpu_allocations = {mat_cum_row_indexes, mat_column_indexes, mat_values, in_vector_mem, out_vector_mem};
    return gpu_allocations;
}

void npmmv_csr_gpu_free(struct NpmmvCsrGpuAllocations gpu_allocations) {
    free_gpu_uint_array(gpu_allocations.mat_cum_row_indexes);
    free_gpu_uint_array(gpu_allocations.mat_column_indexes);
    free_gpu_float_array(gpu_allocations.mat_values);
    free_gpu_float_array(gpu_allocations.in_vector);
    free_gpu_float_array(gpu_allocations.out_vector);
}

void npmmv_gpu_set_csr_matrix(struct CsrMatrixPtrs matrix_cpu, struct NpmmvCsrGpuAllocations gpu_allocations, uint outerdim, uint values) {
    npmmv_gpu_set_uint_array(matrix_cpu.cum_row_indexes, outerdim+1, gpu_allocations.mat_cum_row_indexes);
    npmmv_gpu_set_uint_array(matrix_cpu.column_indexes, values, gpu_allocations.mat_column_indexes);
    npmmv_gpu_set_float_array(matrix_cpu.values, values, gpu_allocations.mat_values);
}

void npmmv_csr_gpu_compute(struct NpmmvCsrGpuAllocations gpu_allocations, uint outerdim) {
    internal_negative_prob_multiply_csr_matrix_vector_gpu(gpu_allocations.mat_cum_row_indexes.start, gpu_allocations.mat_column_indexes.start, 
        gpu_allocations.mat_values.start, gpu_allocations.in_vector.start, gpu_allocations.out_vector.start, outerdim);
    cudaDeviceSynchronize();
}