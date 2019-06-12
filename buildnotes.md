Build:
    nvcc -c cutest.cu
    nvcc -c testcu.c
    nvcc -o concurrentgraph-cuda.o cutest.o testcu.o

To static lib:
    ar rcs concurrentgraph-cuda.a concurrentgraph-cuda.o