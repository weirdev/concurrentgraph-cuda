nvcc --shared --compiler-options -fPIC -lnvgraph concurrentgraph_cuda.cu npmmv_dense_kernel.cu npmmv_csr_kernel.cu npmmv_csr_vector_kernel.cu bfs_csr_kernel.cu gpu_types.cu sssp.cpp graph_determ_weights.cu -o libconcurrentgraph_cuda.so