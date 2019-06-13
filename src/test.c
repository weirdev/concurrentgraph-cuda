#include "concurrentgraph_cuda.h"
#include <stdio.h>

#define NULL (void *)0

int main(void) {
	negative_prob_multiply_matrix_vector_cpu(NULL, NULL, NULL, 4, 5);
	printf("Hello world\n");
}
