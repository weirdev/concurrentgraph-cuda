#ifndef SSSP_H_
#define SSSP_H_

#ifdef __cplusplus
extern "C" {
#endif

    void sssp(int* cum_col_indexes, int* row_indexes, float* values, unsigned int nodes_i, unsigned int edges_i, float* output);

#ifdef __cplusplus
}
#endif
#endif