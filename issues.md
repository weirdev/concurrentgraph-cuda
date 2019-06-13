GPU matrix multiplication reallocates the matrix to the device with every call. 
Create seperate calls to:

* Allocate space for matrix and input and output vectors
* Load matrix
* Load input vector
* Read output vector
* Deallocate space


We only parallelize only over the `o` rows of the matrix/output vector. Possibly breakup each row into `s` sections. Produce an output `o by s` matrix. Then calculate the product of each of the `o` rows of the output matrix to get the output vector.