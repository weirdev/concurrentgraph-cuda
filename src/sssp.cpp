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
//#include "helper_cuda.h"
#include "nvgraph.h"

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

int main(int argc, char **argv)
{
    const size_t  n = 3, nnz = 5, vertex_numsets = 1, edge_numsets = 1;
    int i, *destination_offsets_h, *source_indices_h;
    float *weights_h, *sssp_1_h;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    destination_offsets_h = (int*) malloc((n+1)*sizeof(int));
    source_indices_h = (int*) malloc(nnz*sizeof(int));
    weights_h = (float*)malloc(nnz*sizeof(float));
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));

    printf("init\n");

    vertex_dim[0]= (void*)sssp_1_h;
    vertex_dimT[0] = CUDA_R_32F;

    weights_h [0] = .3;
    weights_h [1] = .5;
    weights_h [2] = .7;
    weights_h [3] = .9;
    weights_h [4] = .2;

    destination_offsets_h [0] = 0;
    destination_offsets_h [1] = 1;
    destination_offsets_h [2] = 5;

    source_indices_h [0] = 2;
    source_indices_h [1] = 0;
    source_indices_h [2] = 2;
    source_indices_h [3] = 0;
    source_indices_h [4] = 4;

    
    printf("precreated\n");

    check_status(nvgraphCreate(&handle));

    printf("created\n");

    check_status(nvgraphCreateGraphDescr (handle, &graph));

    printf("created desc\n");

    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));

    
    printf("configured\n");

    // Solve
    int source_vert = 0;
    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    
    printf("solved\n");


    check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    // expect sssp_1_h = (0.000000 0.500000 0.500000 1.333333 0.833333 1.333333)^T
    printf("sssp_1_h\n");
    for (i = 0; i<n; i++)  printf("%f\n",sssp_1_h[i]); printf("\n");
    printf("\nDone!\n");

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(sssp_1_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    //Clean 
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));

    return 0;
}
