#include <stdio.h>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

#include "assert.h"
#include "gpu_types.h"

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

void set_gpu_float_array(float* src, uint size, struct GpuFloatArray dst) {
    if (dst.end - dst.start < size) {
        throw std::out_of_range("Attempted to copy more memory than allocated on the device");
    }
    cudaError_t result = cudaMemcpy(dst.start, src, size * sizeof(float), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to device memory");
    }
}

void get_gpu_float_array(struct GpuFloatArray src, float* dst, uint size) {
    if (src.end - src.start < size) {
        throw std::out_of_range("Attempted to copy more memory to host than allocated on the device");
    }
    
    gpuErrchk(cudaMemcpy(dst, src.start, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void set_gpu_uint_array(uint* src, uint size, struct GpuUIntArray dst) {
    if (dst.end - dst.start < size) {
        throw std::out_of_range("Attempted to copy more memory than allocated on the device");
    }
    cudaError_t result = cudaMemcpy(dst.start, src, size * sizeof(uint), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to device memory");
    }
}

void get_gpu_uint_array(struct GpuUIntArray src, uint* dst, uint size) {
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

struct GpuIntArray allocate_gpu_int_array(uint array_size) {
    int* start;
    int* end;
    cudaError_t result = cudaMalloc((void**)&start, array_size * sizeof(int));
    if (result != cudaSuccess)
    {
        start = end = 0;
        throw std::runtime_error("failed to allocate device memory");
    }
    end = start + array_size;
    struct GpuIntArray arraystruct = {start, end};
    return arraystruct;
}

void free_gpu_int_array(struct GpuIntArray array) {
    if (array.start != 0)
    {
        cudaFree(array.start);
    }
}

void set_gpu_int_array(int* src, uint size, struct GpuIntArray dst) {
    if (dst.end - dst.start < size) {
        throw std::out_of_range("Attempted to copy more memory than allocated on the device");
    }
    cudaError_t result = cudaMemcpy(dst.start, src, size * sizeof(int), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to device memory");
    }
}