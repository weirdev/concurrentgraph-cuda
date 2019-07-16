#ifndef SSSP_H_
#define SSSP_H_

#ifdef __cplusplus
extern "C" {
#endif

    void sssp(int* cum_col_indexes, int* row_indexes, float* values, size_t nodes, size_t edges, float* output);

#ifdef __cplusplus
}
#endif
#endif