#include <assert.h>
#include "KernelMatrixAdd.cuh"

void FillMatrix(float* mat, int width, int height, float value) {
  for(int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      mat[row * width + col] = value;
    }
  }
}

int main() {
  int width = 10000;
  int height = 10000;

  float *h_A = new float[width * height];
  float *h_B = new float[width * height];
  float *h_C = new float[width * height];

  FillMatrix(h_A, width, height, 1.0f);
  FillMatrix(h_B, width, height, 2.0f);

  float *A = NULL;
  float *B = NULL;
  float *C = NULL;

  size_t pA = 0;
  size_t pB = 0;
  size_t pC = 0;

  cudaMallocPitch(&A, &pA, width, height);
  cudaMallocPitch(&B, &pB, width, height);
  cudaMallocPitch(&C, &pC, width, height);

  for (int row = 0; row < height; ++row) {
    float* rowA = (float*)((char*)A + row * pA);
    float* rowB = (float*)((char*)B + row * pB);
    cudaMemcpy(rowA, h_A + row * width, width, cudaMemcpyHostToDevice);
    cudaMemcpy(rowB, h_B + row * width, width, cudaMemcpyHostToDevice);
  }

  dim3 blockSize(256, 256);
  dim3 numBlocks((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

  KernelMatrixAdd<<<numBlocks, blockSize>>>(height, width, pA, pB, pC, A, B, C);
	cudaDeviceSynchronize();

  for (int row = 0; row < height; ++row) {
    float* rowC = (float*)((char*)C + row * pC);
    cudaMemcpy(h_C + row * width, rowC, width, cudaMemcpyDeviceToHost);
  }

  for (int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      assert(h_C[row * width + col] = 3.0f);
    }
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

