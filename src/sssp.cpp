/*
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nvgraph.h"
#include "sssp.h"

/* Single Source Shortest Path (SSSP)
 *  Calculate the shortest path distance from a single vertex in the graph
 *  to all other vertices.
 */

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

void sssp(int* cum_col_indexes, int* row_indexes, float* values, unsigned int nodes_i, unsigned int edges_i, float* output)
{
    size_t nodes = (size_t)nodes_i;
    size_t edges = (size_t)edges_i;
    // row_indexes = src indexes
    const size_t vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));

    vertex_dim[0]= (void*)output;
    vertex_dimT[0] = CUDA_R_32F;

    check_status(nvgraphCreate(&handle));

    check_status(nvgraphCreateGraphDescr (handle, &graph));

    CSC_input->nvertices = nodes;
    CSC_input->nedges = edges;
    CSC_input->destination_offsets = cum_col_indexes;
    CSC_input->source_indices = row_indexes;

    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)values, 0));

    // Solve
    
    printf("precomp!\n");
    int source_vert = 0;
    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    
    printf("comp done!\n");

    check_status(nvgraphGetVertexData(handle, graph, (void*)output, 0));

    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    //Clean 
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));
}

int main(int argc, char **argv) {
    int* destination_offsets_h = (int*) malloc((6+1)*sizeof(int));
    int* source_indices_h = (int*) malloc(10*sizeof(int));
    float* weights_h = (float*)malloc(10*sizeof(float));
    float* dists = (float*)malloc(6*sizeof(float));

    weights_h [0] = 0.333333;
    weights_h [1] = 0.500000;
    weights_h [2] = 0.333333;
    weights_h [3] = 0.500000;
    weights_h [4] = 0.500000;
    weights_h [5] = 1.000000;
    weights_h [6] = 0.333333;
    weights_h [7] = 0.500000;
    weights_h [8] = 0.500000;
    weights_h [9] = 0.500000;

    destination_offsets_h [0] = 0;
    destination_offsets_h [1] = 1;
    destination_offsets_h [2] = 3;
    destination_offsets_h [3] = 4;
    destination_offsets_h [4] = 6;
    destination_offsets_h [5] = 8;
    destination_offsets_h [6] = 10;

    source_indices_h [0] = 2;
    source_indices_h [1] = 0;
    source_indices_h [2] = 2;
    source_indices_h [3] = 0;
    source_indices_h [4] = 4;
    source_indices_h [5] = 5;
    source_indices_h [6] = 2;
    source_indices_h [7] = 3;
    source_indices_h [8] = 3;
    source_indices_h [9] = 4;

    sssp(destination_offsets_h, source_indices_h, weights_h, 6, 10, dists);

    free(weights_h);
    free(destination_offsets_h);
    free(source_indices_h);

    // expect sssp_1_h = (0.000000 0.500000 0.500000 1.333333 0.833333 1.333333)^T
    printf("dists\n");
    for (int i = 0; i<6; i++)  printf("%f\n",dists[i]);
    printf("Done!\n");

    free(dists);
}