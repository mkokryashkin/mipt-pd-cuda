#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include "KernelMatrixAdd.cuh"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

void FillMatrix(float* mat, int width, int height, float value) {
  for(int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      mat[row * width + col] = value;
    }
  }
}

void PrintMatrix(float* mat, int width, int height) {
  for(int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      printf("%f ", mat[row * width + col]);
    }
    printf("\n");
  }
  printf("\n");
  fflush(stdout);
}

int main() {
  int width = 4;
  int height = 4;

  float *h_A = new float[width * height];
  float *h_B = new float[width * height];
  float *h_C = new float[width * height];

  FillMatrix(h_A, width, height, 1.0f);
  FillMatrix(h_B, width, height, 2.0f);

  PrintMatrix(h_A, width, height);
  PrintMatrix(h_B, width, height);

  float *A = NULL;
  float *B = NULL;
  float *C = NULL;

  size_t pA = 0;
  size_t pB = 0;
  size_t pC = 0;

  checkCudaErrors(cudaMallocPitch(&A, &pA, width * sizeof(float), height));
  checkCudaErrors(cudaMallocPitch(&B, &pB, width * sizeof(float), height));
  checkCudaErrors(cudaMallocPitch(&C, &pC, width * sizeof(float), height));
  checkCudaErrors(cudaMemcpy2D(A, pA, h_A, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(B, pB, h_B, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));

  dim3 blockSize(256, 256);
  dim3 numBlocks((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

  KernelMatrixAdd<<<numBlocks, blockSize>>>(height, width, pA, pB, pC, A, B, C);
	cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy2D(h_C, width * sizeof(float), C, pC, width * sizeof(float), height, cudaMemcpyDeviceToHost));

  PrintMatrix(h_C, width, height);

  for (int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      assert(h_C[row * width + col] == 3.0f);
    }
  }

  checkCudaErrors(cudaFree(A));
  checkCudaErrors(cudaFree(B));
  checkCudaErrors(cudaFree(C));

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

