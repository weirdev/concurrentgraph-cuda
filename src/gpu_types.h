#ifndef GPU_TYPES_H_
#define GPU_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif
    struct GpuFloatArray {
        float* start;
        float* end;
    };

    struct GpuUIntArray {
        unsigned int* start;
        unsigned int* end;
    };

    struct GpuIntArray {
        int* start;
        int* end;
    };

    struct GpuFloatArray allocate_gpu_float_array(uint array_size);
    void free_gpu_float_array(struct GpuFloatArray array);
    void set_gpu_float_array(float* src, uint size, struct GpuFloatArray dst);
    void get_gpu_float_array(struct GpuFloatArray src, float* dst, uint size);

    struct GpuUIntArray allocate_gpu_uint_array(uint array_size);
    void free_gpu_uint_array(struct GpuUIntArray array);
    void set_gpu_uint_array(uint* src, uint size, struct GpuUIntArray dst);
    void get_gpu_uint_array(struct GpuUIntArray src, uint* dst, uint size);
    
    struct GpuIntArray allocate_gpu_int_array(uint array_size);
    void free_gpu_int_array(struct GpuIntArray array);
    void set_gpu_int_array(int* src, uint size, struct GpuIntArray dst);
    void get_gpu_int_array(struct GpuIntArray src, int* dst, uint size);

    struct CsrFloatMatrixPtrs {
        uint* cum_row_indexes;
        uint* column_indexes;
        float* values;
    };

    struct CsrIntMatrixPtrs {
        uint* cum_row_indexes;
        uint* column_indexes;
        int* values;
    };

#ifdef __cplusplus
}
#endif
#endif