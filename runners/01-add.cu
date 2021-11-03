#include <assert.h>
#include <stdio.h>
#include "KernelAdd.cuh"

int main() {
  int numElements = 1 << 28;
	float *x = NULL;
  float *y = NULL;
  float *result = NULL;

	cudaMallocManaged(&x, numElements * sizeof(*x));
	cudaMallocManaged(&y, numElements * sizeof(*y));
	cudaMallocManaged(&result, numElements * sizeof(*result));


	for (int i = 0; i < numElements; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	int blockSize = 256;

	int numBlocks = (numElements + blockSize - 1) / blockSize;

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
	KernelAdd<<<numBlocks, blockSize>>>(numElements, x, y, result);
  cudaEventRecord(stop);

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  printf("Elpased: %f\n", millis);

	cudaDeviceSynchronize();

  for(int i = 0; i < numElements; ++i) {
    assert(result[i] == 3.0f);
  }

  cudaFree(x);
  cudaFree(y);
  cudaFree(result);

  return 0;
}
