Compile .cu file to shared library:
    nvcc --compiler-options -fPIC -c cutest.cu -o cutest.so

Staticly link C file with cuda file (nvcc only needed to compile .cu files, gcc and g++ can be used for other lines):
    nvcc -c cutest.cu
    nvcc -c testcu.c
    nvcc -o concurrentgraph-cuda.o cutest.o testcu.o

To static lib:
    ar rcs concurrentgraph-cuda.a concurrentgraph-cuda.o
