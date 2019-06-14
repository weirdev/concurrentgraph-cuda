4Full compilation of shared library, use this:
    nvcc --shared --compiler-options -fPIC concurrentgraph_cuda.cu npmmv_kernel.cu -o libconcurrentgraph_cuda.so

Compile .cu file to shared library:

    $ nvcc --compiler-options -fPIC -c cutest.cu -o cutest.so
    
    <possibly need the below instead>
    $ nvcc --compiler-options -fPIC -c cutest.cu -o cutest.so --shared
    <Without --shared to just get object file>

Staticly link C file with cuda file (nvcc only needed to compile .cu files, gcc and g++ can be used for other lines):

    nvcc -c cutest.cu
    nvcc -c testcu.c
    nvcc -o concurrentgraph-cuda.o cutest.o testcu.o

To static lib:

    $ ar rcs concurrentgraph-cuda.a concurrentgraph-cuda.o
